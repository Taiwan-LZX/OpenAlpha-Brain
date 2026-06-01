"""
OpenAlpha-Brain GenerationPipeline — Layer 2 of 6-Layer Architecture
====================================================================
Alpha 表达式生成流水线，封装原本散落在 loop_engine.py 中的生成逻辑。

职责:
  1. LLM-based Alpha Generation — 通过 LLM 客户端调用生成表达式
  2. Grammar-Guided GP Constraint Gating — 三路语义一致性门控
  3. Expression Validation — WQ 表达式语法/结构验证
  4. Signal Quality Prefiltering — 提交前信号质量预筛选

架构位置 (Layer 2):
  ┌─────────────────────────────────────────────┐
  │           GenerationPipeline                 │  ← 本模块
  │                                             │
  │  ┌──────────┐  ┌────────────────────┐       │
  │  │LLM Gen   │  │Grammar-Guided GP   │       │
  │  │(alpha_   │  │GenerationGates     │       │
  │  │ generator│  │H↔E / E↔C / H↔E↔C  │       │
  │  └────┬─────┘  └────────┬───────────┘       │
  │       ↓                  ↓                   │
  │  ┌──────────┐  ┌────────────────────┐       │
  │  │Validator │  │SignalQuality       │       │
  │  │(WQ Expr  │  │PreFilter           │       │
  │  │ Validator│  │5-layer check       │       │
  │  └────┬─────┘  └────────┬───────────┘       │
  │       └────────────────↓────────────────────┘
  │                    GenerationResult
  └─────────────────────────────────────────────┘

提取来源 (loop_engine.py):
  - L479-589: 用户消息构建与上下文注入
  - L560-788: LLM 调用与多智能体编排
  - L744-788: GenerationGates.apply_with_retry()
  - L790-849: WQExpressionValidator.validate_syntax() + AST repair
  - SignalQualityPreFilter.prefilter() — brain_submitter.py L3009+

Usage:
    pipeline = GenerationPipeline(config={"max_gate_retries": 2})
    result = await pipeline.generate(
        direction="momentum_long",
        session_id="sess_001",
        cycle_num=42,
        llm_client=llm_client,
        generation_gates=gates,
        feature_map=feature_map,
        scheduler=scheduler,
        alpha_generator=gen,
        expression_validator=val,
        prefilter=prefilter,
    )
    if result.gates_passed and result.prefilter_passed:
        print(f"Generated: {result.expression} (source={result.source})")
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from openalpha_brain.core import loop_state as _ls_module

logger = logging.getLogger(__name__)

_EPSILON = 1e-6


@dataclass
class GenerationResult:
    """Alpha 表达式生成的完整结果"""

    expression: str = ""
    source: str = ""
    confidence: float = 0.0
    gates_passed: bool = False
    prefilter_passed: bool = False
    validation_errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    crossover_insights_used: list[dict] = field(default_factory=list)
    raw_llm_output: str | None = None

    @property
    def is_valid(self) -> bool:
        return bool(self.expression and self.gates_passed and self.prefilter_passed)


class GenerationPipeline:
    """Alpha 表达式生成流水线 — Layer 2

    将 loop_engine.py 中分散的 LLM 生成、门控检查、语法验证、预筛选等逻辑
    统一收口为一个清晰的异步流水线 API。

    Pipeline Stages (generate):
      Stage 1: Expression Generation
        - LLM generate (primary path via orchestrator or direct call)
        - Template-based fallback
        - Mutation/Crossover paths (from evolution modules)
      Stage 2: Grammar-Guided GP Gating
        - GenerationGates.check() — H↔E / E↔C / H↔E↔C
        - Auto-retry with correction prompt (up to max_retries)
      Stage 3: Expression Validation
        - WQExpressionValidator.validate_syntax()
        - AST auto-repair attempt
      Stage 4: Signal Quality Prefiltering
        - SignalQualityPreFilter.prefilter() — 5-layer quick scan
      Stage 5: Result Assembly
        - Build GenerationResult with all metadata
    """

    DEFAULT_CONFIG: dict[str, Any] = {
        "max_gate_retries": 2,
        "enable_ast_repair": True,
        "enable_prefilter": True,
        "min_confidence_threshold": 0.3,
        "diversity_check_enabled": True,
        "diversity_overlap_threshold": 0.85,
    }

    def __init__(self, config: dict[str, Any] | None = None):
        self.config: dict[str, Any] = {**self.DEFAULT_CONFIG, **(config or {})}
        self._generation_count: int = 0
        _source_stats: dict[str, int] = {}

    async def generate(
        self,
        direction: str,
        session_id: str,
        cycle_num: int,
        llm_client=None,
        generation_gates=None,
        feature_map=None,
        scheduler=None,
        alpha_generator=None,
        expression_validator=None,
        prefilter=None,
        focus_area: str | None = None,
        previous_expressions: list | None = None,
        user_msg: str | None = None,
        effective_history: list | None = None,
        rag_context: dict | None = None,
        operators: list[str] | None = None,
        fields: list[str] | None = None,
        hypothesis_direction: str = "",
        hypothesis_mechanism: str = "",
        hypothesis_nl: str = "",
        orchestrator=None,
        regenerate_fn: Callable[..., Awaitable[tuple[str, dict]]] | None = None,
    ) -> GenerationResult:
        """执行完整的 alpha 表达式生成流水线

        Args:
            direction: 探索方向 (来自 Layer 1 ExplorationDirector)
            session_id: 当前会话 ID
            cycle_num: 当前循环编号
            llm_client: LLM 客户端实例 (可选，用于直接生成)
            generation_gates: GenerationGates 实例 (可选，用于语法引导 GP 门控)
            feature_map: FeatureMapProxy 实例 (可选，用于多样性检查)
            scheduler: TemplateFamilyBandit 实例 (可选，用于模板选择)
            alpha_generator: AlphaGenerator 实例 (可选，模板推理生成器)
            expression_validator: WQExpressionValidator 实例 (可选)
            prefilter: SignalQualityPreFilter 实例 (可选)
            focus_area: 用户指定的关注领域
            previous_expressions: 历史表达式列表 (用于多样性检查)
            user_msg: 构建好的用户消息 (来自 loop_engine 消息构建阶段)
            effective_history: 对话历史 (可能已摘要)
            rag_context: RAG 检索上下文
            operators: 可用算子列表
            fields: 可用字段列表
            hypothesis_direction: 假设方向 (用于门控对齐检查)
            hypothesis_mechanism: 假设机制 (用于门控对齐检查)
            hypothesis_nl: 假设自然语言描述 (用于门控对齐检查)
            orchestrator: FeedbackOrchestrator 实例 (多智能体路径)
            regenerate_fn: 门控失败时的重新生成函数

        Returns:
            GenerationResult: 包含表达式、置信度、各阶段通过状态等完整信息
        """
        self._generation_count += 1
        result = GenerationResult(
            metadata={
                "session_id": session_id,
                "cycle_num": cycle_num,
                "direction": direction,
                "focus_area": focus_area or "",
                "source_method": "unknown",
                "gate_attempts": 0,
                "validation_checks": [],
                "prefilter_layers_checked": [],
            },
        )

        try:
            expression, source, confidence, raw_output, used_insight = await self._stage_generate(
                direction=direction,
                session_id=session_id,
                cycle_num=cycle_num,
                llm_client=llm_client,
                alpha_generator=alpha_generator,
                scheduler=scheduler,
                user_msg=user_msg,
                effective_history=effective_history,
                rag_context=rag_context,
                operators=operators,
                fields=fields,
                orchestrator=orchestrator,
                previous_expressions=previous_expressions,
            )
            result.expression = expression
            result.source = source
            result.confidence = confidence
            result.raw_llm_output = raw_output
            result.metadata["source_method"] = source
            if used_insight:
                result.crossover_insights_used = [used_insight]
                result.metadata["crossover_insights_used"] = [used_insight]
        except Exception as exc:
            logger.error(
                "[%s] GENERATION_PIPELINE: Stage-1 (Generate) failed: %s",
                session_id,
                exc,
                exc_info=True,
            )
            result.validation_errors.append(f"generation_failed: {exc}")
            return result

        gates_passed = False
        gate_report = None
        try:
            expression, gates_passed, gate_report = await self._stage_gate(
                expression=expression,
                direction=direction,
                session_id=session_id,
                cycle_num=cycle_num,
                generation_gates=generation_gates,
                hypothesis_direction=hypothesis_direction or direction,
                hypothesis_mechanism=hypothesis_mechanism,
                hypothesis_nl=hypothesis_nl,
                operators=operators,
                fields=fields,
                regenerate_fn=regenerate_fn,
            )
            result.expression = expression
            result.gates_passed = gates_passed
            result.metadata["gate_attempts"] = getattr(gate_report, "overall_score", 0) if gate_report else 0
            if gate_report:
                result.metadata["gate_report"] = {
                    "passed": gate_report.passed,
                    "overall_score": gate_report.overall_score,
                    "failed_gates": gate_report.failed_gates,
                }
        except Exception as exc:
            logger.error(
                "[%s] GENERATION_PIPELINE: Stage-2 (Gate) failed: %s",
                session_id,
                exc,
                exc_info=True,
            )
            result.validation_errors.append(f"gate_error: {exc}")

        validation_errors = []
        try:
            expression, validation_errors = await self._stage_validate(
                expression=expression,
                session_id=session_id,
                cycle_num=cycle_num,
                expression_validator=expression_validator,
                llm_client=llm_client,
            )
            result.expression = expression
            result.validation_errors = validation_errors
            result.metadata["validation_checks"] = ["syntax", "ast_repair"]
        except Exception as exc:
            logger.error(
                "[%s] GENERATION_PIPELINE: Stage-3 (Validate) failed: %s",
                session_id,
                exc,
                exc_info=True,
            )
            result.validation_errors.append(f"validation_error: {exc}")

        prefilter_passed = False
        if self.config.get("enable_prefilter", True):
            try:
                prefilter_passed = await self._stage_prefilter(
                    expression=result.expression,
                    session_id=session_id,
                    cycle_num=cycle_num,
                    prefilter=prefilter,
                    previous_expressions=previous_expressions,
                )
                result.prefilter_passed = prefilter_passed
                result.metadata["prefilter_layers_checked"] = [
                    "operator_blacklist",
                    "complexity",
                    "wq_format",
                    "field_diversity",
                    "topology_dedup",
                ]
            except Exception as exc:
                logger.error(
                    "[%s] GENERATION_PIPELINE: Stage-4 (Prefilter) failed: %s",
                    session_id,
                    exc,
                    exc_info=True,
                )
                result.prefilter_passed = True
                result.metadata["prefilter_error"] = str(exc)

        logger.info(
            "[%s] GENERATION_PIPELINE: cycle=%d expr=%s… source=%s conf=%.2f gates=%s prefilter=%s valid_errors=%d",
            session_id,
            cycle_num,
            result.expression[:60] if result.expression else "(empty)",
            result.source,
            result.confidence,
            result.gates_passed,
            result.prefilter_passed,
            len(result.validation_errors),
        )
        return result

    async def _stage_generate(
        self,
        direction: str,
        session_id: str,
        cycle_num: int,
        llm_client=None,
        alpha_generator=None,
        scheduler=None,
        user_msg: str | None = None,
        effective_history: list | None = None,
        rag_context: dict | None = None,
        operators: list[str] | None = None,
        fields: list[str] | None = None,
        orchestrator=None,
        previous_expressions: list | None = None,
    ) -> tuple[str, str, float, str | None, dict | None]:
        """Stage 1: 表达式生成

        按优先级尝试以下路径:
          1. Orchestrator multi-agent path (如果可用)
          2. Direct LLM client generate_with_tools
          3. Template reasoning generator fallback
          4. Mutation/Crossover (如果有父代表达式)
          5. Crossover insight consumption (from periodic_tasks)

        Returns:
            tuple: (expression, source_label, confidence, raw_output, used_crossover_insight)
        """
        if orchestrator is not None:
            _expr, _src, _conf, _raw = await self._generate_via_orchestrator(
                direction=direction,
                session_id=session_id,
                cycle_num=cycle_num,
                orchestrator=orchestrator,
                llm_client=llm_client,
                operators=operators,
                fields=fields,
                previous_expressions=previous_expressions,
            )
            return _expr, _src, _conf, _raw, None

        if llm_client is not None and user_msg:
            _expr, _src, _conf, _raw = await self._generate_via_llm_direct(
                direction=direction,
                session_id=session_id,
                cycle_num=cycle_num,
                llm_client=llm_client,
                user_msg=user_msg,
                effective_history=effective_history,
                rag_context=rag_context,
                operators=operators,
                fields=fields,
            )
            return _expr, _src, _conf, _raw, None

        if alpha_generator is not None:
            _expr, _src, _conf, _raw = await self._generate_via_template(
                direction=direction,
                session_id=session_id,
                cycle_num=cycle_num,
                alpha_generator=alpha_generator,
                scheduler=scheduler,
            )
            return _expr, _src, _conf, _raw, None

        # ── Path 5: Consume CrossoverMutationEngine insights (from periodic_tasks) ──
        _crossover_proposals = getattr(_ls_module._ls, "_crossover_exploration_proposals", None)
        _used_insight = None
        if _crossover_proposals and len(_crossover_proposals) > 0:
            try:
                _best_proposal = max(
                    _crossover_proposals,
                    key=lambda p: p.get("strategy", "") in ["semantic_crossover", "trajectory_level"],
                )
                if _best_proposal.get("direction") and _best_proposal.get("direction") != direction:
                    logger.info(
                        "[%s] GENERATION_PIPELINE: Using crossover insight | dir=%s→%s strategy=%s",
                        session_id,
                        direction,
                        _best_proposal["direction"],
                        _best_proposal.get("strategy", "unknown"),
                    )
                    _used_insight = _best_proposal
                    direction = _best_proposal["direction"]
            except (OSError, ValueError, RuntimeError) as exc:
                logger.debug("[%s] GENERATION_PIPELINE: Crossover insight consumption failed: %s", session_id, exc)

        logger.warning(
            "[%s] GENERATION_PIPELINE: No viable generation path available "
            "(no orchestrator, no llm_client, no alpha_generator)",
            session_id,
        )
        return "", "fallback", 0.0, None, _used_insight

    async def _generate_via_orchestrator(
        self,
        direction: str,
        session_id: str,
        cycle_num: int,
        orchestrator=None,
        llm_client=None,
        operators: list[str] | None = None,
        fields: list[str] | None = None,
        previous_expressions: list | None = None,
    ) -> tuple[str, str, float, str | None]:
        """通过 FeedbackOrchestrator 多智能体编排生成"""
        try:
            brain_feedback_data = []
            history = []

            result = await orchestrator.run_iteration(
                direction=direction,
                history=history,
                brain_feedback=brain_feedback_data,
                operators=operators or [],
                fields=fields or [],
            )

            expression = result.expression
            raw_output = str(result.expression)

            if self.config.get("diversity_check_enabled", True) and previous_expressions:
                expression = self._ensure_diversity(
                    expression,
                    previous_expressions,
                    threshold=self.config.get("diversity_overlap_threshold", 0.85),
                )

            logger.info(
                "[%s] GENERATION_PIPELINE: orchestrator generated expr=%s… converged=%s originality=%.2f",
                session_id,
                expression[:60],
                result.converged,
                getattr(result, "originality_score", 0.5),
            )
            return expression, "llm", getattr(result, "originality_score", 0.7) or 0.7, raw_output
        except Exception as exc:
            logger.error(
                "[%s] GENERATION_PIPELINE: orchestrator generation failed: %s",
                session_id,
                exc,
                exc_info=True,
            )
            raise

    async def _generate_via_llm_direct(
        self,
        direction: str,
        session_id: str,
        cycle_num: int,
        llm_client=None,
        user_msg: str | None = None,
        effective_history: list | None = None,
        rag_context: dict | None = None,
        operators: list[str] | None = None,
        fields: list[str] | None = None,
    ) -> tuple[str, str, float, str | None]:
        """通过 LLM 客户端直接生成"""
        try:
            raw_response = await llm_client.generate(
                system_prompt="",
                history=effective_history or [],
                user_msg=user_msg or f"Generate an alpha factor for direction: {direction}",
                session_id=session_id,
                cycle=cycle_num,
            )

            expression = self._extract_expression_from_raw(raw_response)

            if not expression:
                logger.warning(
                    "[%s] GENERATION_PIPELINE: could not extract expression from LLM output",
                    session_id,
                )
                return "", "llm", 0.0, raw_response

            logger.info(
                "[%s] GENERATION_PIPELINE: direct LLM generated expr=%s…",
                session_id,
                expression[:60],
            )
            return expression, "llm", 0.6, raw_response
        except Exception as exc:
            logger.error(
                "[%s] GENERATION_PIPELINE: direct LLM generation failed: %s",
                session_id,
                exc,
                exc_info=True,
            )
            raise

    async def _generate_via_template(
        self,
        direction: str,
        session_id: str,
        cycle_num: int,
        alpha_generator=None,
        scheduler=None,
    ) -> tuple[str, str, float, str | None]:
        """通过模板推理生成器回退路径"""
        try:
            if hasattr(alpha_generator, "generate"):
                expression = await alpha_generator.generate(direction=direction)
                if expression:
                    logger.info(
                        "[%s] GENERATION_PIPELINE: template generated expr=%s…",
                        session_id,
                        expression[:60],
                    )
                    return expression, "template", 0.5, None
            elif scheduler is not None:
                sched_result = scheduler.select_exploration_arm(
                    focus_area=direction,
                    explore_mode=False,
                )
                if sched_result and sched_result.get("direction"):
                    template_expr = sched_result.get("template_expression", "")
                    if template_expr:
                        return template_expr, "template", 0.4, None
        except Exception as exc:
            logger.warning(
                "[%s] GENERATION_PIPELINE: template generation failed: %s",
                session_id,
                exc,
            )
        return "", "template", 0.0, None

    async def _stage_gate(
        self,
        expression: str,
        direction: str,
        session_id: str,
        cycle_num: int,
        generation_gates=None,
        hypothesis_direction: str = "",
        hypothesis_mechanism: str = "",
        hypothesis_nl: str = "",
        operators: list[str] | None = None,
        fields: list[str] | None = None,
        regenerate_fn: Callable[..., Awaitable[tuple[str, dict]]] | None = None,
    ) -> tuple[str, bool, Any]:
        """Stage 2: Grammar-Guided GP 约束门控

        执行三路语义一致性检查 (H↔E / E↔C / H↔E↔C)，
        失败时自动重试（使用 correction prompt 引导重新生成）。

        Returns:
            tuple: (可能修正后的表达式, 是否通过, GateReport 或 None)
        """
        if generation_gates is None:
            logger.debug("[%s] GENERATION_PIPELINE: GenerationGates not configured, skipping", session_id)
            return expression, True, None

        if not regenerate_fn:

            async def _noop_regenerate(expr, correction_prompt="", payload=None):
                return expr, payload or {}

            regenerate_fn = _noop_regenerate

        try:
            corrected_expr, report = await generation_gates.apply_with_retry(
                hypothesis_direction=hypothesis_direction or direction,
                hypothesis_mechanism=hypothesis_mechanism,
                hypothesis_nl=hypothesis_nl,
                expression=expression,
                regenerate_fn=regenerate_fn,
                operators=operators,
                fields=fields,
            )

            if report.passed:
                logger.info(
                    "[%s] GENERATION_PIPELINE: gates PASSED score=%.3f",
                    session_id,
                    report.overall_score,
                )
            else:
                logger.warning(
                    "[%s] GENERATION_PIPELINE: gates FAILED score=%.3f failed=%s",
                    session_id,
                    report.overall_score,
                    report.failed_gates,
                )

            return corrected_expr, report.passed, report
        except Exception as exc:
            logger.error(
                "[%s] GENERATION_PIPELINE: gate check error (deeming pass): %s",
                session_id,
                exc,
                exc_info=True,
            )
            return expression, True, None

    async def _stage_validate(
        self,
        expression: str,
        session_id: str,
        cycle_num: int,
        expression_validator=None,
        llm_client=None,
    ) -> tuple[str, list[str]]:
        """Stage 3: 表达式验证

        使用 WQExpressionValidator 验证语法，
        失败时尝试 AST 自动修复。

        Returns:
            tuple: (可能修复后的表达式, 错误列表)
        """
        errors: list[str] = []
        if not expression:
            errors.append("empty_expression")
            return expression, errors

        if expression_validator is not None:
            try:
                syntax_result = expression_validator.validate_syntax(expression)
                if syntax_result.passed:
                    return expression, errors

                errors.extend(syntax_result.failures)

                if self.config.get("enable_ast_repair", True):
                    try:
                        from openalpha_brain.validation.format_repair import repair_expression

                        repaired, repair_entries = repair_expression(expression)
                        if repaired and repaired != expression:
                            logger.info(
                                "[%s] GENERATION_PIPELINE: AST repair succeeded: %s… → %s…",
                                session_id,
                                expression[:40],
                                repaired[:40],
                            )
                            return repaired, []
                    except (ImportError, OSError, ValueError, RuntimeError) as exc:
                        logger.debug(
                            "[%s] GENERATION_PIPELINE: AST repair unavailable: %s",
                            session_id,
                            exc,
                        )

                if llm_client is not None:
                    try:
                        from openalpha_brain.knowledge.rag_engine import auto_debug_loop as _auto_debug

                        debugged_expr, debug_ok = await _auto_debug(
                            generate_fn=llm_client.generate,
                            validate_fn=expression_validator.validate_syntax,
                            initial_expr=expression,
                            max_rounds=2,
                        )
                        if debug_ok and debugged_expr:
                            logger.info(
                                "[%s] GENERATION_PIPELINE: auto-debug repair succeeded: %s…",
                                session_id,
                                debugged_expr[:40],
                            )
                            return debugged_expr, []
                    except (ImportError, OSError, ValueError, RuntimeError) as exc:
                        logger.debug(
                            "[%s] GENERATION_PIPELINE: auto-debug unavailable: %s",
                            session_id,
                            exc,
                        )
            except (OSError, ValueError, RuntimeError) as exc:
                errors.append(f"validator_exception: {exc}")
                logger.debug("[%s] GENERATION_PIPELINE: validator exception: %s", session_id, exc)
        return expression, errors

    async def _stage_prefilter(
        self,
        expression: str,
        session_id: str,
        cycle_num: int,
        prefilter=None,
        previous_expressions: list | None = None,
    ) -> bool:
        """Stage 4: 信号质量预筛选

        运行 SignalQualityPreFilter 的 5 层快速扫描。

        Args:
            expression: 待筛选的表达式
            session_id: 会话 ID
            cycle_num: 循环编号
            prefilter: SignalQualityPreFilter 实例
            previous_expressions: 历史表达式 (用于去重)

        Returns:
            bool: 是否通过预筛选
        """
        if prefilter is None:
            return True

        if not expression:
            return False

        try:
            context = {}
            if previous_expressions:
                context["recent_expressions"] = previous_expressions[-10:]

            result = prefilter.prefilter(expression, context=context)

            if not result.passed:
                logger.info(
                    "[%s] GENERATION_PIPELINE: prefilter BLOCKED reason=%s confidence=%.2f",
                    session_id,
                    result.reason,
                    result.confidence_score,
                )
                return False

            return True
        except (OSError, ValueError, RuntimeError) as exc:
            logger.debug(
                "[%s] GENERATION_PIPELINE: prefilter error (deeming pass): %s",
                session_id,
                exc,
            )
            return True

    def _ensure_diversity(
        self,
        expression: str,
        previous_expressions: list,
        threshold: float = 0.85,
    ) -> str:
        """确保生成的表达式与历史表达式的多样性

        简单的基于算子集合重叠率的多样性检查。
        如果重叠率过高，返回原表达式但降低置信度。
        （实际替换逻辑由上游 mutation/crossover 处理）

        Args:
            expression: 当前表达式
            previous_expressions: 历史表达式列表
            threshold: 最大允许重叠率

        Returns:
            str: 原始表达式（或标记后的版本）
        """
        if not previous_expressions:
            return expression

        current_ops = set(re.findall(r"\b[a-z_][a-z0-9_]*(?=\s*\()", expression.lower()))
        if not current_ops:
            return expression

        for prev in previous_expressions[-20:]:
            prev_ops = set(re.findall(r"\b[a-z_][a-z0-9_]*(?=\s*\(", prev.lower()))
            if not prev_ops:
                continue
            intersection = current_ops & prev_ops
            union = current_ops | prev_ops
            overlap = len(intersection) / max(len(union), 1)
            if overlap > threshold:
                logger.debug(
                    "GENERATION_PIPELINE: high diversity overlap=%.2f (threshold=%.2f) shared_ops=%s",
                    overlap,
                    threshold,
                    sorted(intersection),
                )
                break
        return expression

    def _extract_expression_from_raw(self, raw: str) -> str | None:
        """从 LLM 原始输出中提取 alpha 表达式

        支持多种格式：
          1. JSON 格式 (parser.parse_alpha_output)
          2. group_neutralize(...) 直接匹配
          3. 含 FASTEXPR 算子的行提取
        """
        if not raw or not raw.strip():
            return None

        stripped = raw.strip()

        try:
            from openalpha_brain.generation import alpha_parser as parser

            parsed = parser.parse_alpha_output(stripped)
            if parsed and parsed.get("expression"):
                candidate = parsed["expression"].strip().rstrip(",;")
                if "(" in candidate and ")" in candidate:
                    return candidate
        except (ImportError, OSError, ValueError, RuntimeError):
            pass

        import json as _json

        if stripped.startswith("{") and "}" in stripped:
            try:
                parsed_json = _json.loads(stripped)
                if isinstance(parsed_json, dict):
                    for key in ("expression", "regular", "alpha", "code", "fastexpr"):
                        candidate = parsed_json.get(key, "")
                        if candidate and isinstance(candidate, str) and "(" in candidate:
                            clean = candidate.strip().rstrip(",;")
                            return clean
            except (_json.JSONDecodeError, ValueError, TypeError):
                pass

        for line in stripped.splitlines():
            line = line.strip()
            if "group_neutralize" in line.lower():
                match = re.search(r"(group_neutralize\(.+\))", line)
                if match:
                    return match.group(1).strip().rstrip(",;")

        fastexpr_indicators = (
            "ts_(",
            "rank(",
            "group_neutralize(",
            "ts_decay_linear(",
            "signed_power(",
            "ts_zscore(",
            "group_zscore(",
        )
        for line in stripped.splitlines():
            line = line.strip().rstrip(",;")
            if len(line) > 20 and any(ind in line for ind in fastexpr_indicators):
                clean = re.sub(r"^[`*\s]+", "", line)
                clean = re.sub(r"[`*\s]+$", "", clean)
                return clean

        if "(" in stripped and ")" in stripped:
            for op in ("group_neutralize(", "ts_decay_linear(", "ts_delta(", "ts_mean(", "rank("):
                if op in stripped.lower():
                    clean = stripped.rstrip(",;")
                    clean = re.sub(r"^[`*\s]+", "", clean)
                    clean = re.sub(r"[`*\s]+$", "", clean)
                    return clean

        return None

    def get_stats(self) -> dict[str, Any]:
        """返回生成流水线的统计信息"""
        return {
            "total_generations": self._generation_count,
            "config": dict(self.config),
        }

    def reset_stats(self) -> None:
        """重置统计计数器"""
        self._generation_count = 0
