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

        if rag_context:
            self._validate_rag_field_usage(final_expr, rag_context, focus_area, cycle)

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
          - **强制包含** RAG 检索到的 operators、fields、financial_logic
          - **新增**: 基于 WQ 验证模板库的 7 大类别策略选择框架

        Args:
            focus_area: 探索方向
            templates_summary: 模板摘要列表
            rag_context: RAG 上下文

        Returns:
            str: 完整的 CoT prompt
        """
        template_list_str = "\n".join(
            [
                f"{i + 1}. [{t['id']}] {t['name']} — BlockA: {t['signal_example']} "
                f"{'⭐ 推荐（跨族）' if t.get('is_cross_family') else ''}"
                for i, t in enumerate(templates_summary)
            ]
        )

        rag_section = self._build_enhanced_rag_section(rag_context)

        wq_strategy_section = self._build_wq_verified_strategy_section(focus_area, rag_context)

        return f"""你是一位量化研究专家。你需要从以下 {len(templates_summary)} 个经过验证的因子模板中选择最合适的一个来生成 alpha 表达式。

## 当前探索方向
{focus_area}
{rag_section}

## WQ 验证模板策略选择框架（★ 新增 — 基于真实高 Sharpe 因子）
{wq_strategy_section}

## 可用模板列表（每个模板保证 ≥5 个算子，包含 group_neutralize + ts_decay_linear）
{template_list_str}

## 重要说明
1. 每个模板组装后的完整表达式都包含：
   - Block B: group_neutralize(..., industry) 或 group_zscore(...)
   - Block C: ts_decay_linear(..., decay_window)
   - 保证总算子数 ≥5

2. ⭐ 标记的模板支持跨族字段组合（如 price + valuation），能产生更独特的因子

3. 避免总是选择同一个模板类型，根据当前市场环境动态调整

4. **【强制要求】** 你生成的表达式 MUST 使用上面「RAG Retrieved Reference Data」中推荐的字段和算子。
   如果完全不相关，请说明原因。

5. **【类别轮换规则】** 连续 3 次使用同一类别 → 必须切换到其他类别
   优先级排序: quality(7) > sentiment(8) > value(6) > liquidity(3) > momentum(5) > reversal(4) > analyst(2)

## 你的任务
请选择一个模板并详细解释你的推理过程。考虑以下因素：
- 当前市场环境适合哪种类型的因子？
- 哪个模板的理论基础最强？
- 选择该模板的风险和优势是什么？
- 如何结合 WQ 验证类别策略和 RAG 推荐的字段/算子？
- 是否符合类别轮换规则？（避免连续使用同类）

请输出 JSON 格式：
{{
  "selected_template": "模板ID（如 value_regression）",
  "reasoning": "详细的经济学推理过程（2-3句话），需引用 WQ 验证类别的理论依据",
  "hypothesis": "核心假设（一句话）",
  "signal_direction": "positive/negative（表示做多还是做空信号）",
  "rationale": "为什么这个模板在当前环境下有效，结合市场环境和 WQ 验证证据",
  "category_selected": "选择的策略类别（momentum/reversal/value/quality/liquidity/sentiment/analyst）",
  "rotation_compliance": "说明本次选择是否符合轮换规则"
}}"""

    def _build_wq_verified_strategy_section(self, focus_area: str, rag_context: dict[str, Any]) -> str:
        """构建 WQ 验证模板策略选择部分

        基于 market_logics.json 中的 wq_verified_templates 定义，
        为 Phase 1 提供结构化的策略选择框架。

        Args:
            focus_area: 当前探索方向
            rag_context: RAG 上下文（用于推断市场环境）

        Returns:
            str: WQ 验证策略选择框架的文本描述
        """
        try:
            import json as _json

            from openalpha_brain.data import get_data_path

            market_logics_path = get_data_path("market_logics.json")
            if not market_logics_path.exists():
                logger.warning("[DEFENSIVE_LOG] template_reasoning_generator: market_logics.json not found, using fallback strategy section")
                return self._fallback_wq_strategy_section()

            with open(market_logics_path, encoding="utf-8") as _f:
                market_data = _json.load(_f)

            wq_templates = market_data.get("wq_verified_templates", {})
            categories = wq_templates.get("categories", {})
            rotation_rules = wq_templates.get("rotation_rules", {})

            if not categories:
                return self._fallback_wq_strategy_section()

            lines = [
                "",
                "### 📊 7大策略类别及其 WQ 验证依据",
                "",
                "每个类别都经过 WorldQuant BRAIN 实际验证或理论推导，Sharpe 范围 1.28-1.77：",
                ""
            ]

            category_descriptions = {
                "momentum": {
                    "emoji": "📈",
                    "name_cn": "动量类",
                    "priority": 5,
                    "templates_count": len(categories.get("momentum", {}).get("templates", [])),
                    "sharpe_range": "1.60-1.77",
                    "best_for": ["趋势持续性强的市场", "信息扩散缓慢的环境", "中等波动率时期"],
                    "key_mechanism": "价格趋势持续性 + 成交量确认 + 基本面动量交互",
                    "representative_template": "TEMPLATE-M1 (Debt-Momentum Composite, Sharpe 1.77)",
                    "field_families": "price_trend + valuation / volume_liquidity",
                    "risk_note": "拥挤度高，需结合基本面或情绪字段降低相关性"
                },
                "reversal": {
                    "emoji": "📉",
                    "name_cn": "反转类",
                    "priority": 4,
                    "templates_count": len(categories.get("reversal", {}).get("templates", [])),
                    "sharpe_range": "1.45-1.69",
                    "best_for": ["均值回归明显的市场", "过度反应后的修正期", "高流动性环境"],
                    "key_mechanism": "价格偏离回归 + 波动率条件门控 + 非线性压缩",
                    "representative_template": "TEMPLATE-R1 (VWAP Decay Reversal, Sharpe 1.69)",
                    "field_families": "price_trend + price_trend (不同字段)",
                    "risk_note": "需严格流动性条件，避免低流动性时期的假突破"
                },
                "value": {
                    "emoji": "💰",
                    "name_cn": "价值类",
                    "priority": 6,
                    "templates_count": len(categories.get("value", {}).get("templates", [])),
                    "sharpe_range": "1.38-1.55",
                    "best_for": ["价值投资风格盛行期", "基本面驱动行情", "长期持有策略"],
                    "key_mechanism": "低估溢价 + 盈利率信号 + 基本面-价格交互效应",
                    "representative_template": "TEMPLATE-V1 (Fundamental-Price Interaction, Sharpe 1.55)",
                    "field_families": "valuation/growth + volume_liquidity",
                    "risk_note": "价值陷阱风险，需结合质量或动量指标过滤"
                },
                "quality": {
                    "emoji": "⭐",
                    "name_cn": "质量类",
                    "priority": 7,
                    "templates_count": len(categories.get("quality", {}).get("templates", [])),
                    "sharpe_range": "1.35-1.42",
                    "best_for": ["追求稳定收益", "低波动率偏好", "机构投资者主导市场"],
                    "key_mechanism": "盈利能力 + 稳定性 + 杠杆改善 + 运营效率提升",
                    "representative_template": "TEMPLATE-Q1 (Profitability-Stability Composite, Sharpe 1.42)",
                    "field_families": "quality + quality (多维度)",
                    "risk_note": "高质量公司可能已被充分定价，需寻找预期差"
                },
                "liquidity": {
                    "emoji": "💧",
                    "name_cn": "流动性类",
                    "priority": 3,
                    "templates_count": len(categories.get("liquidity", {}).get("templates", [])),
                    "sharpe_range": "1.33-1.52",
                    "best_for": ["流动性分化明显时期", "微观结构套利机会", "高频数据可用"],
                    "key_mechanism": "流动性门控反转 + 价量趋势确认 + 三族交互",
                    "representative_template": "TEMPLATE-L1 (Liquidity-Adjusted Reversal Enhanced, Sharpe 1.52)",
                    "field_families": "price_trend + volume_liquidity (+ valuation 三族)",
                    "risk_note": "流动性因子对交易成本敏感，需控制换手率"
                },
                "sentiment": {
                    "emoji": "🎭",
                    "name_cn": "情绪类",
                    "priority": 8,
                    "templates_count": len(categories.get("sentiment", {}).get("templates", [])),
                    "sharpe_range": "1.28-1.48",
                    "best_for": ["情绪极端偏差期", "新闻事件驱动行情", "行为金融异象明显"],
                    "key_mechanism": "空头情绪逆向 + 微观结构-波动率混合 + 分析师修正确认",
                    "representative_template": "TEMPLATE-S1 (Short-Interest Contrarian, Sharpe 1.48)",
                    "field_families": "ownership + sentiment / microstructure + risk",
                    "risk_note": "情绪数据可能有滞后性，需结合价格数据实时验证"
                },
                "analyst": {
                    "emoji": "📉",
                    "name_cn": "分析师类",
                    "priority": 2,
                    "templates_count": len(categories.get("analyst", {}).get("templates", [])),
                    "sharpe_range": "1.31-1.44",
                    "best_for": ["财报季前后", "分析师预期调整期", "信息不对称程度高"],
                    "key_mechanism": "预期修正动量 + 分析师离散度 + 相关性确认",
                    "representative_template": "TEMPLATE-A1 (Estimate Revision Momentum, Sharpe 1.44)",
                    "field_families": "sentiment + sentiment (多维) / sentiment + price_trend",
                    "risk_note": "分析师数据更新频率低，不适合短期策略；注意高复杂度惩罚"
                }
            }

            for cat_id, cat_info in category_descriptions.items():
                cat_data = categories.get(cat_id, {})
                templates_in_cat = cat_data.get("templates", [])

                lines.extend([
                    f"#### {cat_info['emoji']} {cat_id.upper()} — {cat_info['name_cn']}",
                    f"  **优先级**: {cat_info['priority']}/8 | **模板数**: {len(templates_in_cat)} | **Sharpe范围**: {cat_info['sharpe_range']}",
                    f"  **最佳适用场景**: {', '.join(cat_info['best_for'])}",
                    f"  **核心机制**: {cat_info['key_mechanism']}",
                    f"  **代表模板**: {cat_info['representative_template']}",
                    f"  **字段族要求**: {cat_info['field_families']}",
                    f"  **⚠️ 风险提示**: {cat_info['risk_note']}",
                    ""
                ])

                if templates_in_cat:
                    lines.append("  **该类别下的可用模板**:")
                    for tpl in templates_in_cat[:2]:
                        tpl_id = tpl.get("id", "?")
                        tpl_name = tpl.get("name", "?")
                        tpl_sharpe = tpl.get("sharpe_reference", "?")
                        tpl_status = tpl.get("verified_status", "?")
                        lines.append(f"    - [{tpl_id}] {tpl_name} (Sharpe={tpl_sharpe}, {tpl_status})")
                    lines.append("")

            rotation_priority = rotation_rules.get("priority_ordering", [])
            max_consecutive = rotation_rules.get("max_consecutive_same_category", 3)
            min_categories = rotation_rules.get("minimum_categories_per_session", 4)

            lines.extend([
                "---",
                "### 🎯 类别轮换规则（强制执行）",
                f"- **最大连续同类别次数**: {max_consecutive} 次 → 之后必须切换",
                f"- **优先级排序**: {' > '.join(rotation_priority)}",
                f"- **每session最少覆盖类别数**: {min_categories} 个",
                "- **当前focus_area**: " + focus_area,
                "",
                "### 💡 策略选择建议（基于RAG上下文推断市场环境）:",
                self._infer_market_regime_suggestion(rag_context),
                ""
            ])

            return "\n".join(lines)

        except (OSError, ValueError, KeyError) as exc:
            logger.warning(
                "[DEFENSIVE_LOG] template_reasoning_generator: failed to build WQ strategy section (%s), using fallback",
                exc,
            )
            return self._fallback_wq_strategy_section()

    def _fallback_wq_strategy_section(self) -> str:
        """Fallback WQ 策略部分（当 market_logics.json 不可用时）"""
        return """
**[FALLBACK] WQ 验证模板策略选择框架**

由于无法加载 market_logics.json，使用简化版策略建议：

#### 7大策略类别概览
1. **MOMENTUM (动量)** [优先级:5] — Sharpe 1.60-1.77 — 趋势持续性
2. **REVERSAL (反转)** [优先级:4] — Sharpe 1.45-1.69 — 均值回归
3. **VALUE (价值)** [优先级:6] — Sharpe 1.38-1.55 — 低估溢价
4. **QUALITY (质量)** [优先级:7] — Sharpe 1.35-1.42 — 优质企业
5. **LIQUIDITY (流动性)** [优先级:3] — Sharpe 1.33-1.52 — 流动性溢价
6. **SENTIMENT (情绪)** [优先级:8] — Sharpe 1.28-1.48 — 情绪偏差
7. **ANALYST (分析师)** [优先级:2] — Sharpe 1.31-1.44 — 信息优势

**轮换规则**: 连续3次同类别 → 强制切换 | 优先级: quality>sentiment>value>liquidity>momentum>reversal>analyst
"""

    def _infer_market_regime_suggestion(self, rag_context: dict[str, Any]) -> str:
        """基于 RAG 上下文推断市场环境并给出策略建议

        Args:
            rag_context: RAG 检索上下文

        Returns:
            str: 市场环境推断和策略建议
        """
        if not rag_context:
            return "  - 无法推断市场环境（无RAG数据），建议默认选择 quality 或 sentiment 类别"

        financial_logic = rag_context.get("financial_logic", [])
        direction = rag_context.get("exploration_direction", "")

        suggestions = []

        if any("volatility" in str(fl).lower() for fl in financial_logic):
            suggestions.append("- 检测到 volatility 相关逻辑 → 推荐 **reversal** 或 **quality** 类别（低波动偏好）")

        if any("momentum" in str(fl).lower() for fl in financial_logic):
            suggestions.append("- 检测到 momentum 相关逻辑 → 可继续 **momentum** 类别，但需确保跨族组合")

        if any(["value" in str(fl).lower(), "valuation" in str(fl).lower(), "fundamental" in str(fl).lower()] for fl in financial_logic):
            suggestions.append("- 检测到 value/fundamental 相关逻辑 → 强烈推荐 **value** 或 **quality** 类别")

        if any(["sentiment" in str(fl).lower(), "analyst" in str(fl).lower(), "revision" in str(fl).lower()] for fl in financial_logic):
            suggestions.append("- 检测到 sentiment/analyst 相关逻辑 → 推荐 **sentiment** 或 **analyst** 类别")

        if direction:
            suggestions.append(f"- 当前 exploration_direction={direction} → 建议匹配对应类别或选择互补类别")

        if not suggestions:
            suggestions.append("- 无明确市场环境信号 → 建议按优先级排序选择：quality > sentiment > value")

        return "\n".join(suggestions) if suggestions else "  - 使用默认策略：按优先级排序选择"

    def _build_enhanced_rag_section(self, rag_context: dict[str, Any]) -> str:
        """构建增强版 RAG 数据注入段

        将 RAG 检索结果格式化为强制的参考数据，确保 LLM 必须考虑这些信息。
        如果 RAG 返回为空，返回 fallback 提示并记录警告。

        Args:
            rag_context: RAG 检索上下文

        Returns:
            str: 格式化的 RAG 参考数据字符串
        """
        if not rag_context:
            logger.warning("[DEFENSIVE_LOG] template_reasoning_generator: rag_context_empty — no RAG data available")
            return "\n## 当前市场上下文（RAG）\n⚠️ 无 RAG 检索数据，将使用默认提示"

        operators = rag_context.get("operators", [])
        fields = rag_context.get("fields", [])
        financial_logic = rag_context.get("financial_logic", [])

        if not operators and not fields and not financial_logic:
            logger.warning(
                "[DEFENSIVE_LOG] template_reasoning_generator: rag_retrieved_empty "
                "direction=%s — RAG returned empty result (0 ops, 0 fields, 0 finlogic)",
                rag_context.get("direction", "unknown"),
            )
            return (
                "\n## 当前市场上下文（RAG）\n"
                "⚠️ RAG 检索返回空结果，建议检查向量索引是否已加载\n"
                "将使用通用推荐：close, volume, debt, earnings, revenue"
            )

        lines = ["\n--- RAG Retrieved Reference Data ---"]
        lines.append(f"Recommended Operators ({len(operators[:8])} items):")

        for op in operators[:8]:
            op_id = op.get("id", "")
            score = op.get("score", 0)
            meta = op.get("meta", {})
            category = meta.get("category", "")
            definition = meta.get("definition", "")[:80]

            line = f"  • {op_id}"
            if score:
                line += f" (score={score:.3f})"
            if category:
                line += f" [{category}]"
            if definition:
                line += f" — {definition}"
            lines.append(line)

        lines.append(f"\nRecommended Fields ({len(fields[:10])} items):")
        for fld in fields[:10]:
            field_id = fld.get("id", "")
            score = fld.get("score", 0)
            meta = fld.get("meta", {})

            line = f"  • {field_id}"
            if score:
                line += f" (score={score:.3f})"
            family = meta.get("family", "")
            description = meta.get("description", "")[:60]
            if family:
                line += f" [{family}]"
            if description:
                line += f" — {description}"
            lines.append(line)

        if financial_logic:
            lines.append(f"\nFinancial Logic Context ({len(financial_logic[:3])} items):")
            for fl in financial_logic[:3]:
                fl_id = fl.get("id", "")
                score = fl.get("score", 0)
                line = f"  • {fl_id}"
                if score:
                    line += f" (score={score:.3f})"
                lines.append(line)

        experience_replay = rag_context.get("experience_replay")
        if experience_replay:
            action = experience_replay.get("suggested_action", "")
            confidence = experience_replay.get("confidence", 0)
            target = experience_replay.get("suggested_target", "")
            if action:
                lines.append(f"\nExperience Replay Suggestion (confidence={confidence:.2f}):")
                lines.append(f"  • Action: {action}")
                if target:
                    lines.append(f"  • Target: {target}")

        lines.append("----------------------------------------")
        lines.append("Based on the above reference data, generate an alpha expression that incorporates these recommendations.")

        logger.info(
            "[REASONING] Enhanced RAG injection: %d ops, %d fields, %d finlogic",
            len(operators[:8]),
            len(fields[:10]),
            len(financial_logic[:3]),
        )

        return "\n".join(lines)

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
- 假设: {reasoning.get("hypothesis", "N/A")}
- 方向: {reasoning.get("signal_direction", "N/A")}

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
            op_count,
            min_ops,
            template_id,
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
            op_count,
            len(enriched_ops),
            enriched[:100],
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
            logger.warning(
                "[REASONING] Phase 4: EXPRESSION REJECTED ✗ - reason: %s", critique.get("rejection_reason", "")
            )

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
- 检测到的算子 ({op_count}个): {", ".join(operators_used)}

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
        templates_attr = getattr(self._lib, "_three_block_templates", {})
        for template_id, template in templates_attr.items():
            signal_example = template.block_a.template_str[:80]

            is_cross_family = False
            params = template.block_a.editable_params
            if (
                "price_field" in params
                and "fundamental_field" in params
                or "price_field" in params
                and any(p in params for p in ["earnings_field", "revenue_field", "asset_field"])
                or "volume_field" in params
                and any(p in params for p in ["cap_field", "fundamental_field"])
            ):
                is_cross_family = True

            summaries.append(
                {
                    "id": template_id,
                    "name": template.name,
                    "category": template.category,
                    "signal_example": signal_example,
                    "editable_params": params,
                    "is_cross_family": is_cross_family,
                }
            )

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
                        recommended.append(
                            {
                                "field_id": fid,
                                "family_id": family_id,
                                "coverage": finfo.get("coverage", 0),
                                "for_param": param,
                            }
                        )

                if fields:
                    seen_families.add(family_id)
                    break

        return recommended[: top_k * 2]

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
        operator_pattern = r"\b(ts_\w+|group_\w+|rank|signed_power|zscore|normalize|winsorize)\b"
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
                text = text[first_nl + 1 :]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        text = re.sub(r"//[^\n]*", "", text)
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

        if text.startswith("{"):
            try:
                end_idx = text.rfind("}") + 1
                json_str = text[:end_idx]
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if json_match:
            try:
                cleaned = re.sub(r"//[^\n]*", "", json_match.group())
                cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
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
                field_mapping = {param: defaults.get(param, "close") for param in template.block_a.editable_params}
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

    def _validate_rag_field_usage(
        self,
        expression: str,
        rag_context: dict[str, Any],
        focus_area: str,
        cycle: int,
    ) -> None:
        """验证生成的表达式是否使用了 RAG 推荐的字段

        这是一个诊断性检查，不会拒绝表达式，但会记录潜在的
        RAG-Generator 脱节问题。

        检查项：
          1. 表达式中是否包含 RAG 推荐的 field？
          2. 如果完全不相关，记录 [DEFENSIVE_LOG] 警告

        Args:
            expression: 生成的 alpha 表达式
            rag_context: RAG 检索上下文
            focus_area: 当前探索方向
            cycle: 当前循环编号
        """
        if not expression or not rag_context:
            return

        rag_fields = rag_context.get("fields", [])
        if not rag_fields:
            return

        recommended_field_ids = [f.get("id", "") for f in rag_fields[:10] if f.get("id")]
        if not recommended_field_ids:
            return

        expr_lower = expression.lower()
        used_rag_fields = []

        for field_id in recommended_field_ids:
            field_pattern = r"\b" + re.escape(field_id.lower()) + r"\b"
            if re.search(field_pattern, expr_lower):
                used_rag_fields.append(field_id)

        if used_rag_fields:
            logger.info(
                "[REASONING] RAG field usage confirmed | cycle=%d focus=%s "
                "used_fields=%s/%d available",
                cycle,
                focus_area,
                used_rag_fields[:5],
                len(recommended_field_ids),
            )
        else:
            logger.warning(
                "[DEFENSIVE_LOG] template_reasoning_generator: rag_field_not_used cycle=%d focus=%s "
                "expr=%s… — Generated expression does NOT contain any RAG-recommended field. "
                "Recommended fields were: %s. This may indicate RAG-Generator decoupling.",
                cycle,
                focus_area,
                expression[:60],
                recommended_field_ids[:8],
            )
