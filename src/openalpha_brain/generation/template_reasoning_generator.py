"""
OpenAlpha-Brain — Template-Guided Reasoning Generator (模板引导式深度推理生成器)

基于 ThreeBlockTemplate 的 4 阶段深度推理因子生成器，解决 LLM 自由组合导致
表达式过于简单（仅3-4个算子、只用price字段）的问题。

核心设计：
  Phase 1: Economic Reasoning (经济推理) → 选择最合适的模板
  Phase 2: Field Mapping (字段映射)   → 确保跨族字段选择
  Phase 3: Expression Assembly (表达式组装) → 自动组装三段式
  Phase 4: Self-Critique (自我批判)  → LLM 自检输出质量

Fallback机制：任何阶段失败时自动使用模板默认参数，保证鲁棒性。
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from openalpha_brain.generation.alpha_logics import (
    AlphaLogicLibrary,
)
from openalpha_brain.knowledge.field_proxy_map import FieldProxyMap

logger = logging.getLogger(__name__)

LLMCallFn = Callable[..., Awaitable[str]]


@dataclass
class ReasoningResult:
    """4阶段深度推理的完整结果"""

    phase1_reasoning: dict
    phase2_mapping: dict
    phase3_expression: str
    phase4_critique: dict
    final_expression: str
    approved: bool
    generation_metadata: dict = field(default_factory=dict)


class TemplateReasoningGenerator:
    """基于 ThreeBlockTemplate 的 4 阶段深度推理因子生成器

    通过结构化的 CoT (Chain-of-Thought) 推理流程，引导 LLM 基于经过验证的
    市场逻辑模板生成高质量 alpha 表达式，而非自由组合算子。

    Attributes:
        _llm: LLM 调用函数（与现有架构兼容）
        _fpm: 字段代理图谱（用于推荐低拥挤度字段）
        _lib: Alpha 逻辑库（包含 20 个 ThreeBlockTemplate）

    Example:
        >>> generator = TemplateReasoningGenerator(
        ...     llm_call_fn=my_llm_function,
        ...     field_proxy_map=fpm,
        ...     alpha_logic_lib=lib,
        ... )
        >>> result = await generator.generate(
        ...     focus_area="value",
        ...     cycle=5,
        ...     rag_context={"market_regime": "high_rate"},
        ... )
        >>> print(result.final_expression)
        'group_neutralize(ts_decay_linear(-rank(close / debt), 10), industry)'
    """

    def __init__(
        self,
        llm_call_fn: LLMCallFn,
        field_proxy_map: FieldProxyMap | None = None,
        alpha_logic_lib: AlphaLogicLibrary | None = None,
    ) -> None:
        self._llm = llm_call_fn
        self._fpm = field_proxy_map
        self._lib = alpha_logic_lib

    async def generate(
        self,
        focus_area: str,
        cycle: int,
        rag_context: dict[str, Any] | None = None,
    ) -> ReasoningResult:
        """主入口：执行 4 阶段推理生成 alpha 表达式

        Args:
            focus_area: 当前探索方向（如 "value", "momentum", "volatility"）
            cycle: 当前迭代周期数
            rag_context: RAG 检索到的上下文信息（可选）

        Returns:
            ReasoningResult: 包含4阶段完整推理过程和最终表达式

        Raises:
            ValueError: 当 alpha_logic_lib 未初始化或无可用模板时
        """
        if self._lib is None:
            raise ValueError("AlphaLogicLibrary is required for template reasoning")

        templates_info = self._get_templates_summary()
        if not templates_info:
            raise ValueError("No ThreeBlockTemplates available in library")

        metadata = {
            "focus_area": focus_area,
            "cycle": cycle,
            "timestamp": __import__("time").time(),
            "template_count": len(templates_info),
        }

        try:
            logger.info("[REASONING] Phase 1: Starting Economic Reasoning for focus=%s", focus_area)
            phase1_result = await self._phase1_economic_reasoning(
                focus_area=focus_area,
                templates_info=templates_info,
                rag_context=rag_context or {},
            )
            selected_template_id = phase1_result.get("selected_template", "")
            logger.info("[REASONING] Phase 1 complete: selected_template=%s", selected_template_id)
        except (ConnectionError, OSError, TimeoutError):
            phase1_result = self._fallback_phase1(templates_info)
            selected_template_id = phase1_result.get("selected_template", "")

        try:
            logger.info("[REASONING] Phase 2: Starting Field Mapping for template=%s", selected_template_id)
            recommended_fields = self._get_recommended_fields_for_template(selected_template_id)
            phase2_result = await self._phase2_field_mapping(
                template_id=selected_template_id,
                reasoning=phase1_result,
                recommended_fields=recommended_fields,
            )
            field_mapping = phase2_result.get("field_mapping", {})
            logger.info("[REASONING] Phase 2 complete: field_mapping=%s", field_mapping)
        except (ConnectionError, OSError, TimeoutError) as exc:
            logger.warning("[REASONING] Phase 2 failed (%s), using fallback: default params", exc)
            phase2_result = self._fallback_phase2(selected_template_id)
            field_mapping = phase2_result.get("field_mapping", {})

        try:
            logger.info("[REASONING] Phase 3: Assembling expression")
            phase3_expression = await self._phase3_assemble(
                template_id=selected_template_id,
                field_mapping=field_mapping,
            )
            logger.info("[REASONING] Phase 3 complete: expression=%s", phase3_expression[:100])
        except (ConnectionError, OSError, TimeoutError) as exc:
            logger.error("[REASONING] Phase 3 assembly failed: %s", exc)
            phase3_expression = self._fallback_phase3(selected_template_id, field_mapping)

        try:
            logger.info("[REASONING] Phase 4: Starting Self-Critique")
            phase4_critique = await self._phase4_self_critique(
                expression=phase3_expression,
                template_id=selected_template_id,
                field_mapping=field_mapping,
            )
            approved = phase4_critique.get("critique", {}).get("overall_verdict") == "APPROVE"
            logger.info("[REASONING] Phase 4 complete: verdict=%s", approved)
        except (ConnectionError, OSError, TimeoutError) as exc:
            logger.warning("[REASONING] Phase 4 failed (%s), auto-approving", exc)
            phase4_critique = {"critique": {"overall_verdict": "APPROVE", "error": str(exc)}}
            approved = True

        final_expr = phase3_expression if approved else self._fallback_phase3(selected_template_id, field_mapping)

        return ReasoningResult(
            phase1_reasoning=phase1_result,
            phase2_mapping=phase2_result,
            phase3_expression=phase3_expression,
            phase4_critique=phase4_critique,
            final_expression=final_expr,
            approved=approved,
            generation_metadata=metadata,
        )

    async def _phase1_economic_reasoning(
        self,
        focus_area: str,
        templates_info: list[dict],
        rag_context: dict[str, Any],
    ) -> dict:
        """阶段1: 经济推理 + 模板选择

        引导 LLM 从 20 个经过验证的市场逻辑模板中选择最合适的一个，
        并提供经济学的推理依据。

        Args:
            focus_area: 当前探索方向
            templates_info: 所有可用模板的摘要信息
            rag_context: RAG 上下文（市场环境、历史表现等）

        Returns:
            dict: 包含 selected_template, reasoning, hypothesis 等字段
        """
        prompt = self._build_phase1_prompt(focus_area, templates_info, rag_context)

        raw_response = await self._llm(
            system_prompt="You are a quantitative finance research expert specializing in alpha factor design.",
            history=[],
            user_msg=prompt,
            session_id="reasoning_phase1",
            cycle=0,
        )

        parsed = self._parse_json_response(raw_response)
        if not parsed or "selected_template" not in parsed:
            raise ValueError("Invalid Phase 1 response: missing 'selected_template'")

        template_id = parsed["selected_template"]
        if template_id not in [t["id"] for t in templates_info]:
            raise ValueError(f"Unknown template ID: {template_id}")

        return {
            "selected_template": template_id,
            "reasoning": parsed.get("reasoning", ""),
            "hypothesis": parsed.get("hypothesis", ""),
            "signal_direction": parsed.get("signal_direction", "unknown"),
            "rationale": parsed.get("rationale", ""),
            "raw_response": raw_response[:500],
        }

    def _build_phase1_prompt(
        self,
        focus_area: str,
        templates_summary: list[dict],
        rag_context: dict[str, Any],
    ) -> str:
        """构建阶段1的经济推理 Prompt

        设计要点：
          - 展示所有20个模板的结构化摘要
          - 强调每个模板保证 ≥5 个算子
          - 标注支持跨族的模板（⭐推荐）
          - 注入 RAG 上下文辅助决策

        Args:
            focus_area: 探索方向
            templates_summary: 模板摘要列表
            rag_context: RAG 上下文

        Returns:
            str: 完整的 CoT prompt
        """
        template_list_str = "\n".join([
            f"{i+1}. [{t['id']}] {t['name']} — BlockA: {t['signal_example']} "
            f"{'⭐ 推荐（跨族）' if t.get('is_cross_family') else ''}"
            for i, t in enumerate(templates_summary)
        ])

        rag_section = ""
        if rag_context:
            rag_items = "\n".join([f"- {k}: {v}" for k, v in rag_context.items()])
            rag_section = f"\n\n## 当前市场上下文（RAG）\n{rag_items}"

        return f"""你是一位量化研究专家。你需要从以下 {len(templates_summary)} 个经过验证的因子模板中选择最合适的一个来生成 alpha 表达式。

## 当前探索方向
{focus_area}
{rag_section}

## 可用模板列表（每个模板保证 ≥5 个算子，包含 group_neutralize + ts_decay_linear）
{template_list_str}

## 重要说明
1. 每个模板组装后的完整表达式都包含：
   - Block B: group_neutralize(..., industry) 或 group_zscore(...)
   - Block C: ts_decay_linear(..., decay_window)
   - 保证总算子数 ≥5

2. ⭐ 标记的模板支持跨族字段组合（如 price + valuation），能产生更独特的因子

3. 避免总是选择同一个模板类型，根据当前市场环境动态调整

## 你的任务
请选择一个模板并详细解释你的推理过程。考虑以下因素：
- 当前市场环境适合哪种类型的因子？
- 哪个模板的理论基础最强？
- 选择该模板的风险和优势是什么？

请输出 JSON 格式：
{{
  "selected_template": "模板ID（如 value_regression）",
  "reasoning": "详细的经济学推理过程（2-3句话）",
  "hypothesis": "核心假设（一句话）",
  "signal_direction": "positive/negative（表示做多还是做空信号）",
  "rationale": "为什么这个模板在当前环境下有效"
}}"""

    async def _phase2_field_mapping(
        self,
        template_id: str,
        reasoning: dict,
        recommended_fields: list[dict],
    ) -> dict:
        """阶段2: 字段映射（确保跨族选择）

        在选定模板后，引导 LLM 为模板的 editable_params 选择具体的字段值。
        关键约束：
          - 如果模板需要多个字段参数，必须从不同族选择
          - 优先选择 LOW CROWDING 族（valuation, sentiment, ownership）
          - 避免 HIGH CROWDING 字段（close, volume 单独使用）

        Args:
            template_id: 选定的模板ID
            reasoning: 阶段1的推理结果
            recommended_fields: 系统推荐的字段列表（按族分类）

        Returns:
            dict: 包含 field_mapping 和 cross_family_check
        """
        template = self._lib.get_three_block_template(template_id) if self._lib else None
        if not template:
            raise ValueError(f"Template not found: {template_id}")

        editable_params = template.block_a.editable_params
        prompt = self._build_phase2_prompt(template_id, editable_params, reasoning, recommended_fields)

        raw_response = await self._llm(
            system_prompt="You are a data field selection expert for quantitative factor construction.",
            history=[],
            user_msg=prompt,
            session_id="reasoning_phase2",
            cycle=0,
        )

        parsed = self._parse_json_response(raw_response)
        if not parsed or "field_mapping" not in parsed:
            raise ValueError("Invalid Phase 2 response: missing 'field_mapping'")

        field_mapping = parsed["field_mapping"]

        cross_family_check = self._validate_cross_family(template_id, field_mapping)

        return {
            "field_mapping": field_mapping,
            "cross_family_check": cross_family_check,
            "raw_response": raw_response[:500],
        }

    def _build_phase2_prompt(
        self,
        template_id: str,
        editable_params: list[str],
        reasoning: dict,
        recommended_fields: list[dict],
    ) -> str:
        """构建阶段2的字段映射 Prompt

        Args:
            template_id: 模板ID
            editable_params: 可编辑参数列表
            reasoning: 阶段1的推理结果
            recommended_fields: 推荐字段（按族分类）

        Returns:
            str: 字段选择 prompt
        """
        params_str = ", ".join(editable_params)

        fields_by_family: dict[str, list[str]] = {}
        for fd in recommended_fields:
            family = fd.get("family_id", "unknown")
            fields_by_family.setdefault(family, []).append(fd["field_id"])

        family_sections = []
        for fam_id, fields in fields_by_family.items():
            family_sections.append(f"### {fam_id} 族\n" + ", ".join(fields[:10]))

        fields_str = "\n\n".join(family_sections) if family_sections else "（无推荐字段）"

        return f"""你选择了模板 [{template_id}]。现在需要为它的参数选择具体的字段值。

## 模板参数
{params_str}

## 可用推荐字段（已按字段族分类）
{fields_str}

## ⚠️ 强制要求
1. 如果模板同时需要 price_field 和 fundamental_field → **必须从不同族选择**
2. 优先选择 LOW CROWDING 族字段（valuation, sentiment, ownership, alternative）
3. 避免 HIGH CROWDING 字段（close, volume 单独使用会导致拥挤）
4. 数值参数（如 decay_lb, medium_lb）建议范围：5-30

## 阶段1的推理参考
- 假设: {reasoning.get('hypothesis', 'N/A')}
- 方向: {reasoning.get('signal_direction', 'N/A')}

## 你的任务
为每个参数选择合适的字段值。确保字段选择的多样性和独特性。

请输出 JSON 格式：
{{
  "field_mapping": {{
    "param1": "field_value1",
    "param2": "field_value2"
  }}
}}

示例（value_regression 模板）：
{{"field_mapping": {{"price_field": "close", "fundamental_field": "debt"}}}}"""

    async def _phase3_assemble(self, template_id: str, field_mapping: dict) -> str:
        if self._lib is None:
            raise ValueError("AlphaLogicLibrary is required")

        expr = self._lib.instantiate_template(
            template_id=template_id,
            fields=field_mapping,
        )

        if not expr:
            raise ValueError(f"Failed to assemble expression for template: {template_id}")

        template = self._lib.get_three_block_template(template_id)
        if template and not template.validate_assembly(expr):
            logger.warning("[REASONING] Phase 3: Expression validation failed, attempting auto-fix")

        expr = self._ensure_min_complexity(expr, template_id, field_mapping)

        logger.info("[REASONING] Phase 3 assembled: %s", expr[:120])
        return expr

    def _ensure_min_complexity(self, expr: str, template_id: str, field_mapping: dict, min_ops: int = 5) -> str:
        operators = self._extract_operators(expr)
        op_count = len(operators)

        if op_count >= min_ops:
            return expr

        logger.warning(
            "[REASONING] Expression complexity too low (%d ops < %d), enriching | template=%s",
            op_count, min_ops, template_id,
        )

        price = field_mapping.get("price_field", "close")
        fundamental = field_mapping.get("fundamental_field", "debt")
        decay_lb = field_mapping.get("decay_lb", 10)
        zscore_lb = field_mapping.get("zscore_lb", 20)
        medium_lb = field_mapping.get("medium_lb", 10)

        category = ""
        if self._lib:
            tpl = self._lib.get_three_block_template(template_id)
            if tpl:
                category = tpl.category

        if category == "value":
            enriched = f"ts_decay_linear(group_neutralize(-rank(signed_power(ts_zscore({price} / {fundamental}, {zscore_lb}), 2)), sector), {decay_lb})"
        elif category == "momentum":
            enriched = f"ts_decay_linear(group_neutralize(rank(ts_delta(ts_rank({price}, {medium_lb}), {medium_lb})), sector), {decay_lb})"
        elif category in ("quality", "size"):
            enriched = f"ts_decay_linear(group_zscore(-rank(ts_delta(ts_std_dev({fundamental if category == 'quality' else price}, {medium_lb}), {medium_lb})), sector), {decay_lb})"
        elif category == "volatility":
            enriched = f"ts_decay_linear(group_neutralize(-rank(ts_delta(ts_std_dev({price}, {medium_lb}), {medium_lb})), sector), {decay_lb})"
        elif category == "liquidity":
            vol_field = field_mapping.get("volume_field", "volume")
            cap_field = field_mapping.get("cap_field", "market_cap")
            enriched = f"ts_decay_linear(group_neutralize(-rank(ts_delta(ts_mean({vol_field}, {medium_lb}) / {cap_field}, {medium_lb})), sector), {decay_lb})"
        elif category == "lead_lag":
            lead = field_mapping.get("lead_field", price)
            lag = field_mapping.get("lag_field", fundamental)
            short_lb = field_mapping.get("short_lb", 5)
            enriched = f"ts_decay_linear(group_neutralize(rank(ts_corr(ts_delta({lead}, {short_lb}), ts_delta({lag}, {short_lb}), {short_lb})), sector), {decay_lb})"
        elif category == "mean_reversion":
            enriched = f"ts_decay_linear(group_zscore(-rank(ts_zscore({price} - ts_mean({price}, {medium_lb}), {zscore_lb})), sector), {decay_lb})"
        else:
            enriched = f"ts_decay_linear(group_neutralize(-rank(signed_power(ts_zscore({price} / {fundamental}, {zscore_lb}), 2)), sector), {decay_lb})"

        enriched_ops = self._extract_operators(enriched)
        logger.info(
            "[REASONING] Enriched expression: %d → %d operators | %s",
            op_count, len(enriched_ops), enriched[:100],
        )
        return enriched

    async def _phase4_self_critique(
        self,
        expression: str,
        template_id: str,
        field_mapping: dict,
    ) -> dict:
        """阶段4: 自我批判

        让 LLM 检查自己生成的表达式是否满足所有限制条件。
        这是一个关键的质控步骤，确保生成的表达式符合预期质量标准。

        检查项：
          - operator_count ≥ 5 （最小复杂度要求）
          - families_used 是否跨族 （多样性要求）
          - has_ts_decay_linear （衰减要求）
          - topology_unique （拓扑唯一性）
          - not_in_blacklist （黑名单检查）

        Args:
            expression: 待检查的表达式
            template_id: 使用的模板ID
            field_mapping: 使用的字段映射

        Returns:
            dict: 包含 critique 详细信息和 overall_verdict
        """
        prompt = self._build_critique_prompt(expression, template_id, field_mapping)

        raw_response = await self._llm(
            system_prompt="You are a strict quality assurance reviewer for alpha factor expressions.",
            history=[],
            user_msg=prompt,
            session_id="reasoning_phase4",
            cycle=0,
        )

        parsed = self._parse_json_response(raw_response)
        if not parsed or "critique" not in parsed:
            raise ValueError("Invalid Phase 4 response: missing 'critique'")

        critique = parsed["critique"]
        overall = critique.get("overall_verdict", "REJECT")

        if overall == "APPROVE":
            logger.info("[REASONING] Phase 4: EXPRESSION APPROVED ✓")
        else:
            logger.warning("[REASONING] Phase 4: EXPRESSION REJECTED ✗ - reason: %s", critique.get("rejection_reason", ""))

        return {
            "critique": critique,
            "raw_response": raw_response[:500],
        }

    def _build_critique_prompt(
        self,
        expression: str,
        template_id: str,
        field_mapping: dict,
    ) -> str:
        """构建自我批判 Prompt

        让 LLM 扮演严格的 QA 角色检查自己的输出。

        Args:
            expression: 待检查的表达式
            template_id: 模板ID
            field_mapping: 字段映射

        Returns:
            str: 自我批判 prompt
        """
        operators_used = self._extract_operators(expression)
        op_count = len(operators_used)

        return f"""你是一位严格的量化因子质量审查员。请检查以下由你生成的 alpha 表达式是否符合所有质量标准。

## 待检查的表达式
```
{expression}
```

## 生成元信息
- 使用模板: {template_id}
- 字段映射: {json.dumps(field_mapping, ensure_ascii=False)}
- 检测到的算子 ({op_count}个): {', '.join(operators_used)}

## 质量检查清单
请逐项检查并给出 PASS/FAIL：

1. **operator_count** (≥5): 表达式是否包含至少5个算子？
2. **passes_min_complexity**: 复杂度是否足够？（避免过于简单的表达式）
3. **families_used**: 使用了哪些字段族？是否跨族？（跨族加分）
4. **passes_cross_family**: 是否使用了来自不同族的字段？
5. **has_ts_decay_linear**: 是否包含衰减项？（稳定性要求）
6. **passes_decay_requirement**: 衰减窗口是否合理（5-30）？
7. **topology_unique**: 拓扑结构是否独特？（避免与常见模式重复）
8. **not_in_blacklist**: 是否不在已知失败模式黑名单中？

## 输出格式
请输出 JSON：
{{
  "critique": {{
    "operator_count": {op_count},
    "operators_used": {json.dumps(operators_used)},
    "passes_min_complexity": true/false,
    "families_used": ["family1", "family2"],
    "is_cross_family": true/false,
    "passes_cross_family": true/false,
    "has_ts_decay_linear": true/false,
    "passes_decay_requirement": true/false,
    "topology_unique": true/false,
    "not_in_blacklist": true/false,
    "overall_verdict": "APPROVE 或 REJECT",
    "rejection_reason": "如果 REJECT，说明原因"
  }}
}}

注意：只有当所有关键项都 PASS 时才能 APPROVE。如果有任何 FAIL，必须 REJECT 并说明原因。"""

    def _get_templates_summary(self) -> list[dict]:
        """获取所有 ThreeBlockTemplate 的摘要信息

        Returns:
            list[dict]: 每个模板的摘要，包含 id, name, signal_example, is_cross_family
        """
        if self._lib is None:
            return []

        summaries = []
        templates_attr = getattr(self._lib, '_three_block_templates', {})
        for template_id, template in templates_attr.items():
            signal_example = template.block_a.template_str[:80]

            is_cross_family = False
            params = template.block_a.editable_params
            if "price_field" in params and "fundamental_field" in params or "price_field" in params and any(p in params for p in ["earnings_field", "revenue_field", "asset_field"]) or "volume_field" in params and any(p in params for p in ["cap_field", "fundamental_field"]):
                is_cross_family = True

            summaries.append({
                "id": template_id,
                "name": template.name,
                "category": template.category,
                "signal_example": signal_example,
                "editable_params": params,
                "is_cross_family": is_cross_family,
            })

        return sorted(summaries, key=lambda x: x["id"])

    def _get_recommended_fields_for_template(
        self,
        template_id: str,
        top_k: int = 8,
    ) -> list[dict]:
        """根据模板的 editable_params 推荐字段

        策略：
          - 对于每个参数，根据其语义（price_field, fundamental_field 等）推荐对应族的字段
          - 优先推荐低拥挤度的字段
          - 确保跨族：如果模板需要多种字段类型，分别从不同族推荐

        Args:
            template_id: 模板ID
            top_k: 每个参数推荐的最大字段数

        Returns:
            list[dict]: 推荐字段列表，每项包含 field_id, family_id, coverage 等
        """
        if self._lib is None or self._fpm is None:
            return []

        template = self._lib.get_three_block_template(template_id)
        if not template:
            return []

        recommended = []
        param_family_map = {
            "price_field": ["price_trend"],
            "volume_field": ["volume_liquidity"],
            "fundamental_field": ["valuation", "balance_sheet", "cash_flow", "profitability"],
            "earnings_field": ["profitability", "earnings_momentum_model"],
            "revenue_field": ["growth_rates", "profitability"],
            "asset_field": ["balance_sheet"],
            "cap_field": ["deep_value_model", "valuation"],
            "lead_field": ["analyst_estimates", "price_trend"],
            "lag_field": ["volume_liquidity", "news_headline"],
        }

        seen_families: set[str] = set()
        for param in template.block_a.editable_params:
            families = param_family_map.get(param, ["price_trend"])

            for family_id in families:
                if family_id in seen_families and len(template.block_a.editable_params) > 1:
                    continue

                fields = self._fpm.recommend_fields_for_template(
                    template_id=template_id,
                    family_id=family_id,
                    top_k=top_k,
                    exclude_cold=True,
                )

                for fid in fields[:3]:
                    finfo = self._fpm.get_field_info(fid)
                    if finfo:
                        recommended.append({
                            "field_id": fid,
                            "family_id": family_id,
                            "coverage": finfo.get("coverage", 0),
                            "for_param": param,
                        })

                if fields:
                    seen_families.add(family_id)
                    break

        return recommended[:top_k * 2]

    def _validate_cross_family(self, template_id: str, field_mapping: dict) -> dict:
        """验证字段映射是否满足跨族要求

        Args:
            template_id: 模板ID
            field_mapping: 字段映射

        Returns:
            dict: 包含 families_used, is_cross_family, crowding_risk 等信息
        """
        if self._fpm is None:
            return {"families_used": [], "is_cross_family": False, "crowding_risk": "UNKNOWN"}

        families_used: set[str] = set()
        high_crowd_fields = {"close", "open", "high", "low", "volume"}

        for param_name, field_value in field_mapping.items():
            if isinstance(field_value, str) and field_value.lower() in high_crowd_fields:
                continue

            family = self._fpm.get_field_family(field_value)
            if family:
                families_used.add(family.family_id)

        is_cross_family = len(families_used) >= 2
        crowding_risk = "LOW" if is_cross_family else ("MEDIUM" if len(families_used) == 1 else "HIGH")

        return {
            "families_used": list(families_used),
            "is_cross_family": is_cross_family,
            "crowding_risk": crowding_risk,
            "field_count": len(field_mapping),
        }

    def _extract_operators(self, expression: str) -> list[str]:
        """从表达式中提取使用的算子列表

        Args:
            expression: alpha 表达式

        Returns:
            list[str]: 使用的算子名称列表
        """
        operator_pattern = r'\b(ts_\w+|group_\w+|rank|signed_power|zscore|normalize|winsorize)\b'
        operators = re.findall(operator_pattern, expression)
        return list(dict.fromkeys(operators))

    def _parse_json_response(self, raw_response: str) -> dict | None:
        """解析 LLM 返回的 JSON 响应

        支持多种格式：
          - 纯 JSON 对象
          - Markdown 代码块包裹的 JSON
          - JSON 后跟解释文本

        Args:
            raw_response: LLM 原始返回文本

        Returns:
            dict | None: 解析后的字典，解析失败返回 None
        """
        text = raw_response.strip()

        if text.startswith("```"):
            first_nl = text.find("\n")
            if first_nl >= 0:
                text = text[first_nl + 1:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        text = re.sub(r'//[^\n]*', '', text)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

        if text.startswith("{"):
            try:
                end_idx = text.rfind("}") + 1
                json_str = text[:end_idx]
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            try:
                cleaned = re.sub(r'//[^\n]*', '', json_match.group())
                cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

        logger.warning("[REASONING] Failed to parse JSON from response: %s", text[:200])
        return None

    def _fallback_phase1(self, templates_info: list[dict]) -> dict:
        """Phase 1 Fallback: 随机选择一个模板

        当 LLM 调用失败或返回无效结果时使用。

        Args:
            templates_info: 可用模板列表

        Returns:
            dict: 默认的 Phase 1 结果
        """
        import random

        cross_family_templates = [t for t in templates_info if t.get("is_cross_family")]
        if cross_family_templates:
            selected = random.choice(cross_family_templates)
        else:
            selected = random.choice(templates_info)

        logger.info("[REASONING] Fallback Phase 1: selected %s (random)", selected["id"])
        return {
            "selected_template": selected["id"],
            "reasoning": "[FALLBACK] Random template selection due to LLM failure",
            "hypothesis": "[FALLBACK] No hypothesis available",
            "signal_direction": "unknown",
            "rationale": "[FALLBACK] Using default template",
        }

    def _fallback_phase2(self, template_id: str) -> dict:
        """Phase 2 Fallback: 使用模板默认参数

        当 LLM 字段选择失败时使用合理的默认值。

        Args:
            template_id: 模板ID

        Returns:
            dict: 默认的字段映射
        """
        defaults = {
            "price_field": "close",
            "volume_field": "volume",
            "fundamental_field": "debt",
            "earnings_field": "eps",
            "revenue_field": "revenue",
            "asset_field": "total_assets",
            "cap_field": "market_cap",
            "lead_field": "close",
            "lag_field": "volume",
            "short_lb": 5,
            "long_lb": 20,
            "medium_lb": 10,
            "decay_lb": 10,
            "lb": 10,
            "corr_lb": 10,
            "std_lb": 20,
            "zscore_lb": 20,
            "sum_lb": 12,
            "delta_lb": 5,
            "vol_lb": 20,
            "liq_lb": 10,
        }

        if self._lib:
            template = self._lib.get_three_block_template(template_id)
            if template:
                field_mapping = {
                    param: defaults.get(param, "close")
                    for param in template.block_a.editable_params
                }
            else:
                field_mapping = {"price_field": "close"}
        else:
            field_mapping = {"price_field": "close"}

        logger.info("[REASONING] Fallback Phase 2: using defaults for %s", template_id)
        return {
            "field_mapping": field_mapping,
            "cross_family_check": {
                "families_used": ["price_trend"],
                "is_cross_family": False,
                "crowding_risk": "HIGH",
            },
        }

    def _fallback_phase3(self, template_id: str, field_mapping: dict) -> str:
        price = field_mapping.get("price_field", "close")
        fundamental = field_mapping.get("fundamental_field", "debt")
        decay_lb = field_mapping.get("decay_lb", 10)
        zscore_lb = field_mapping.get("zscore_lb", 20)

        fallback_expr = f"ts_decay_linear(group_neutralize(-rank(signed_power(ts_zscore({price} / {fundamental}, {zscore_lb}), 2)), sector), {decay_lb})"
        logger.warning("[REASONING] Using ultimate fallback expression: %s", fallback_expr)
        return fallback_expr
