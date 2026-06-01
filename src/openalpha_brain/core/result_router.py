"""
OpenAlpha-Brain ResultRouter
============================
Pipeline Stage 1: WQ 结果接收、解析与评估

职责:
  1. 接收原始 (slot_info, brain_result)
  2. 提取/标准化 WQ 指标 (sharpe, turnover, fitness, drawdown)
  3. 调用 StabilityGuard 评估稳定性
  4. 调用 AntiOverfitDetector 检测过拟合
  5. 调用 Scorer 计算官方评分 + ICIR/MFR
  6. 输出结构化 ParsedWQResult 对象

架构位置:
  ┌──────────────┐     ┌─────────────────┐     ┌──────────────────────┐
  │ WQ Platform  │ →   │  ResultRouter    │ →   │  DecisionEngine      │
  │ (callback)   │     │  (本模块)        │     │  (Stage 2)           │
  └──────────────┘     └─────────────────┘     └──────────────────────┘

Usage:
    router = ResultRouter(
        stability_guard=stability_guard,
        anti_overfit=anti_overfit,
        scorer=scorer,
        adaptive_neutralizer=neutralizer,
    )
    parsed_result = await router.route(slot_info, brain_result)
    # parsed_result.wq_metrics -> 标准化指标字典
    # parsed_result.stability_result -> 稳定性评估
    # parsed_result.score_report -> 官方评分报告
    # parsed_result.is_stable -> 快速判断
    # parsed_result.should_restrict -> 是否需要限制
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from openalpha_brain.monitoring.algorithm_telemetry import AlgorithmTelemetryCollector

logger = logging.getLogger(__name__)


@dataclass
class ParsedWQResult:
    """WQ 结果的完整解析结果 (标准化输出)

    Attributes:
        task_id: 任务标识符
        expression: 因子表达式
        raw_sharpe: 原始 Sharpe 比率
        passed: 是否通过 WQ 审核
        wq_metrics: 标准化的 WQ 指标字典
        stability_result: StabilityGuard 评估结果 (可能为 None)
        is_stable: 是否稳定 (如果 stability_guard 可用)
        should_restrict: 是否应该限制该因子
        anti_fit_score: 过拟合检测得分 (0-100)
        anti_fit_recommendation: 过拟合检测建议 (CAUTIOUS/AGGRESSIVE等)
        score_report: 官方评分报告 (可能为 None)
        official_score: 官方评分分数 (0-100)
        official_grade: 官方评级 (A/B/C/D/F)
        icir_info: ICIR 指标信息 (可能为 None)
        mfr_info: 多面奖励信息 (可能为 None)
        is_high_icir_low_fitness: 是否高ICIR但低Fitness
        is_efficient_alpha: 是否是高效因子
        processing_time_sec: 处理耗时 (秒)
        error: 解析过程中的错误信息 (如果有)
    """
    task_id: str = ""
    expression: str = ""
    raw_sharpe: float = 0.0
    passed: bool = False

    wq_metrics: dict = field(default_factory=dict)
    stability_result: Optional[dict] = None
    is_stable: bool = True
    should_restrict: bool = False

    anti_fit_score: float = 50.0
    anti_fit_recommendation: str = "CAUTIOUS"

    score_report: Any = None
    official_score: float = 0.0
    official_grade: str = "F"

    icir_info: Any = None
    mfr_info: Any = None
    is_high_icir_low_fitness: bool = False
    is_efficient_alpha: bool = False

    processing_time_sec: float = 0.0
    error: Optional[str] = None

    @property
    def fitness(self) -> Optional[float]:
        """快捷访问 Fitness 值"""
        return self.wq_metrics.get("fitness")

    @property
    def turnover(self) -> Optional[float]:
        """快捷访问 Turnover 值"""
        return self.wq_metrics.get("turnover")

    @property
    def drawdown(self) -> Optional[float]:
        """快捷访问 Drawdown 值"""
        return self.wq_metrics.get("drawdown")

    def to_log_dict(self) -> dict:
        """转换为可日志记录的字典 (自动截断长字符串)"""
        return {
            "task_id": self.task_id,
            "expr": self.expression[:80] + "..." if len(self.expression) > 80 else self.expression,
            "sharpe": self.raw_sharpe,
            "passed": self.passed,
            "fitness": self.fitness,
            "turnover": self.turnover,
            "is_stable": self.is_stable,
            "should_restrict": self.should_restrict,
            "anti_fit_score": self.anti_fit_score,
            "official_score": self.official_score,
            "official_grade": self.official_grade,
            "processing_time_sec": round(self.processing_time_sec, 3),
            "error": self.error,
        }


class ResultRouter:
    """WQ 结果路由器 — Pipeline 第一阶段

    将原始 WQ 回调数据转换为标准化的、富含上下文的 ParsedWQResult 对象，
    以便下游的 DecisionEngine 和 ImprovementOrchestrator 使用。

    Design Principles:
    - 单一职责: 只负责接收、解析、评估，不做决策
    - 容错性: 所有外部调用都有 try/except 保护
    - 可观测性: 集成 AlgorithmTelemetry 记录关键节点
    - 标准化: 输出 ParsedWQResult 统一格式，消除下游对 brain_result 内部结构的依赖
    """

    def __init__(
        self,
        stability_guard: Any = None,
        anti_overfit: Any = None,
        scorer: Any = None,
        adaptive_neutralizer: Any = None,
        telemetry: Optional[AlgorithmTelemetryCollector] = None,
        cycle_num: int = 0,
    ):
        self._stability_guard = stability_guard
        self._anti_overfit = anti_overfit
        self._scorer = scorer
        self._adaptive_neutralizer = adaptive_neutralizer
        self._tel = telemetry or AlgorithmTelemetryCollector.get_instance()
        self._cycle_num = cycle_num

    def update_cycle_num(self, cycle_num: int) -> None:
        """更新当前循环编号 (由 FeedbackLoopOrchestrator 调用)"""
        self._cycle_num = cycle_num

    async def route(
        self,
        slot_info: Any,
        brain_result: Any,
    ) -> ParsedWQResult:
        """路由并解析 WQ 完成回调

        Args:
            slot_info: SlotManager 的 SlotInfo 对象 (包含 expression, slot_id 等)
            brain_result: WQ 平台返回的原始结果对象

        Returns:
            ParsedWQResult: 标准化的解析结果
        """
        t0 = time.perf_counter()
        eid = None
        try:
            eid = await self._tel.record_enter(
                "ResultRouter",
                cycle_id=str(self._cycle_num),
                expr_id=str(hash(getattr(slot_info, 'expression', '')) % 10000),
            )
        except (OSError, ValueError, RuntimeError):
            pass

        result = ParsedWQResult()

        try:
            result.task_id = getattr(slot_info, 'task_name', '') or f"slot_{getattr(slot_info, 'slot_id', '?')}"
            result.expression = getattr(slot_info, 'expression', '')
            result.raw_sharpe = getattr(brain_result, 'sharpe', None) or 0.0
            result.passed = getattr(brain_result, 'passed', False)

            logger.info(
                "[ROUTER] ▶ WQ RESULT RECEIVED | task=%s expr=%.50s... sharpe=%.3f passed=%s",
                result.task_id,
                result.expression,
                result.raw_sharpe,
                result.passed,
            )

            result.wq_metrics = self._extract_wq_metrics(brain_result)

            result.stability_result = await self._evaluate_stability(
                result.expression, result.raw_sharpe
            )
            if result.stability_result:
                result.is_stable = result.stability_result.get("is_stable", True)
                result.should_restrict = result.stability_result.get("should_restrict", False)
                result.wq_metrics["stability_restricted"] = result.should_restrict
                result.wq_metrics["stability_score"] = result.stability_result.get("stability_score")
                result.wq_metrics["is_stable"] = result.is_stable
                if result.stability_result.get("constraints"):
                    result.wq_metrics["stability_constraints"] = result.stability_result["constraints"]

            result.anti_fit_score, result.anti_fit_recommendation = \
                self._detect_overfitting(result.wq_metrics)

            result.score_report, result.official_score, result.official_grade, \
                result.icir_info, result.mfr_info, \
                result.is_high_icir_low_fitness, result.is_efficient_alpha = \
                    await self._compute_official_score(result.wq_metrics)

            logger.info(
                "[ROUTER] ◆ PARSED | task=%s sharpe=%.3f fitness=%s turnover=%s "
                "stable=%s restrict=%s anti_fit=%.0f score=%.0f grade=%s",
                result.task_id,
                result.raw_sharpe,
                result.fitness,
                result.turnover,
                result.is_stable,
                result.should_restrict,
                result.anti_fit_score,
                result.official_score,
                result.official_grade,
            )

        except Exception as route_exc:
            logger.error("[ROUTER] ❌ ROUTING FAILED | task=%s error=%s", result.task_id, route_exc, exc_info=True)
            result.error = str(route_exc)

        finally:
            result.processing_time_sec = time.perf_counter() - t0
            try:
                if eid:
                    await self._tel.record_exit("ResultRouter", eid)
            except (OSError, ValueError, RuntimeError):
                pass

        return result

    def _extract_wq_metrics(self, brain_result: Any) -> dict:
        """从 brain_result 中提取标准化的 WQ 指标"""
        return {
            "sharpe": getattr(brain_result, 'sharpe', None) or 0.0,
            "turnover": getattr(brain_result, 'turnover', None),
            "fitness": getattr(brain_result, 'fitness', None),
            "drawdown": getattr(brain_result, 'drawdown', None),
            "checks": getattr(brain_result, 'brain_checks', []) or [],
        }

    async def _evaluate_stability(
        self,
        expression: str,
        current_sharpe: float,
    ) -> Optional[dict]:
        """调用 StabilityGuard 评估因子稳定性"""
        if self._stability_guard is None:
            return None

        try:
            stability_result = self._stability_guard.evaluate_and_guard(
                expression=expression,
                cycle=self._cycle_num,
                current_sharpe=current_sharpe,
            )
            logger.info(
                "[ROUTER] [STABILITY] score=%.2f is_stable=%s type=%s severity=%.2f should_restrict=%s",
                stability_result["stability_score"],
                stability_result["is_stable"],
                stability_result.get("instability_type", "N/A"),
                stability_result["severity"],
                stability_result["should_restrict"],
            )

            if stability_result["should_restrict"]:
                logger.warning(
                    "[ROUTER] ⚠ STABILITY RESTRICTED | expr=%.50s... reason=%s",
                    expression,
                    stability_result.get("diagnosis", "Unknown instability"),
                )

            return stability_result

        except (OSError, ValueError, RuntimeError) as stab_exc:
            logger.debug("[DEFENSIVE_LOG] StabilityGuard evaluate failed: %s", stab_exc)
            return None

    def _detect_overfitting(self, wq_metrics: dict) -> tuple[float, str]:
        """调用 AntiOverfitDetector 检测过拟合"""
        if self._anti_overfit is None:
            return 50.0, "CAUTIOUS"

        try:
            anti_fit_result = self._anti_overfit.evaluate(wq_metrics)
            logger.info(
                "[ROUTER] [ANTI-FIT] score=%.0f rec=%s tests_passed=%d/%d",
                anti_fit_result.score,
                anti_fit_result.recommendation,
                anti_fit_result.passed_count,
                anti_fit_result.total_count,
            )
            return anti_fit_result.score, anti_fit_result.recommendation

        except (OSError, ValueError, RuntimeError) as fit_exc:
            logger.debug("[ROUTER] [ANTI-FIT] Detection failed: %s", fit_exc)
            return 50.0, "CAUTIOUS"

    async def _compute_official_score(
        self,
        wq_metrics: dict,
    ) -> tuple[Any, float, str, Any, Any, bool, bool]:
        """调用 Scorer 计算官方评分 + ICIR/MFR 信息

        Returns:
            (score_report, official_score, official_grade,
             icir_info, mfr_info, is_high_icir_low_fitness, is_efficient_alpha)
        """
        if self._scorer is None:
            return None, 0.0, "F", None, None, False, False

        score_report = None
        official_score = 0.0
        official_grade = "F"
        icir_info = None
        mfr_info = None
        is_high_icir_low_fitness = False
        is_efficient_alpha = False

        try:
            score_report = self._scorer.compute_score(wq_metrics)
            official_score = score_report.overall_score
            official_grade = score_report.grade

            if score_report.icir_metrics:
                icir_info = score_report.icir_metrics
                logger.info(
                    "[ROUTER] [ICIR] ic=%.4f ir=%.3f icir=%.3f pred_fitness=%.2f conf=%.2f",
                    icir_info.ic, icir_info.ir, icir_info.icir,
                    icir_info.predicted_fitness, icir_info.confidence,
                )

            if score_report.multi_faceted_reward:
                mfr_info = score_report.multi_faceted_reward
                logger.info(
                    "[ROUTER] [MULTI-REWARD] signal=%.2f stability=%.2f efficiency=%.2f "
                    "unique=%.2f total=%.2f efficient=%s",
                    mfr_info.signal_quality, mfr_info.stability,
                    mfr_info.efficiency, mfr_info.uniqueness,
                    mfr_info.total_reward, mfr_info.is_efficient_alpha,
                )

            if score_report.multi_layer_result:
                ml_result = score_report.multi_layer_result
                is_high_icir_low_fitness = ml_result.get("is_high_icir_low_fitness", False)
                is_efficient_alpha = ml_result.get("is_efficient_alpha", False)

                if is_high_icir_low_fitness:
                    logger.warning(
                        "[ROUTER] ⚠ HIGH_ICIR_LOW_FITNESS DETECTED | will prioritize turnover optimization",
                    )
                if is_efficient_alpha:
                    logger.info(
                        "[ROUTER] ✨ EFFICIENT_ALPHA DETECTED | will prioritize experience injection",
                    )

            logger.info(
                "[ROUTER] [OFFICIAL-SCORE] score=%.0f grade=%s passed=%s profile=%s",
                official_score,
                official_grade,
                score_report.passed,
                score_report.factor_profile or "N/A",
            )

            if score_report.improvement_hints:
                logger.info(
                    "[ROUTER] [SCORE-HINTS] %s",
                    "; ".join(score_report.improvement_hints[:2]),
                )

        except (OSError, ValueError, RuntimeError) as score_exc:
            logger.debug("[ROUTER] [OFFICIAL-SCORE] Scoring failed: %s", score_exc)

        return (
            score_report, official_score, official_grade,
            icir_info, mfr_info,
            is_high_icir_low_fitness, is_efficient_alpha,
        )

    def record_neutralizer_outcome(
        self,
        parsed_result: ParsedWQResult,
        expression: str = "",
        category: str = "momentum",
        level: str = "standard",
    ) -> None:
        """记录 AdaptiveNeutralizer 结果 (可选，在路由后调用)

        Args:
            parsed_result: 已解析的 WQ 结果
            expression: 因子表达式 (默认使用 parsed_result.expression)
            category: 因子类别 (momentum/reversal/mean_reversion等)
            level: 中性化级别 (standard/aggressive/minimal)
        """
        if self._adaptive_neutralizer is None:
            return

        try:
            from openalpha_brain.evolution.adaptive_neutralizer import NeutralizationTrial
            if NeutralizationTrial is None:
                return

            expr = expression or parsed_result.expression
            outcome = (
                "success" if parsed_result.passed and parsed_result.raw_sharpe >= 1.0
                else ("partial" if parsed_result.raw_sharpe >= 0.5 else "failure")
            )

            self._adaptive_neutralizer.record_outcome(
                expression=expr,
                category=category,
                level=level,
                result_metrics={
                    "sharpe_after": parsed_result.raw_sharpe,
                    "fitness": parsed_result.fitness,
                    "turnover": parsed_result.turnover,
                    "passed": parsed_result.passed,
                    "outcome": outcome,
                },
            )
            logger.debug(
                "[ROUTER] [ADAPT-NEUT] Recorded | task=%s cat=%s outcome=%s",
                parsed_result.task_id, category, outcome,
            )

        except (OSError, ValueError, RuntimeError) as rec_exc:
            logger.debug("[ROUTER] [ADAPT-NEUT] Record failed (non-critical): %s", rec_exc)
