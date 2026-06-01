"""
OpenAlpha-Brain DecisionEngine
==============================
Pipeline Stage 2: 因子改进决策引擎

职责:
  1. 接收 ParsedWQResult (来自 ResultRouter)
  2. 应用5档决策矩阵 (Sharpe-based)
  3. Anti-Overfit 调节 (score<40 时降级)
  4. ICIR/MFR 策略增强 (HIGH_ICIR_LOW_FITNESS / EFFICIENT_ALPHA)
  5. 输出结构化 DecisionOutcome

架构位置:
  ┌─────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
  │  ResultRouter    │ →   │  DecisionEngine   │ →   │ImprovementOrchestrator│
  │  (Stage 1)       │     │  (本模块)         │     │(Stage 3 - 未来)      │
  └─────────────────┘     └──────────────────┘     └──────────────────────┘

决策矩阵 (2026-05-31 升级版):
  Sharpe ≥ 1.25     → SUCCESS_PASS (优秀因子，直接记录)
  Sharpe ∈ [0.8, 1.25) → IMPROVE_AND_RESUBMIT (强信号，深度改进)
  Sharpe ∈ [0, 0.8)   → WEAK_IMPROVE (弱信号，给1次机会)
  Sharpe ∈ [-0.5, 0)  → REPAIR_AND_RETRY (可能格式/方向错误)
  Sharpe < -0.5       → NOISE_PENALIZE (纯噪音，惩罚)

Anti-Overfit 调节:
  anti_fit_score < 40 → 降一级 (IMPROVE→WEAK, WEAK→REPAIR)

策略D增强 (ICIR + Multi-Faceted Reward):
  - HIGH_ICIR_LOW_FITNESS → 强制 IMPROVE (Turnover优化优先)
  - EFFICIENT_ALPHA → 提升 IMPROVE 优先级 (Experience注入)

Usage:
    engine = DecisionEngine(config={
        "sharpe_pass_threshold": 1.25,
        "sharpe_improve_threshold": 0.8,
        ...
    })
    outcome = engine.decide(
        sharpe=1.1,
        passed=False,
        anti_fit_score=65.0,
        is_high_icir_low_fitness=False,
        is_efficient_alpha=True,
    )
    print(outcome.action)        # DecisionAction.IMPROVE_AND_RESUBMIT
    print(outcome.reason)        # 详细原因字符串
    print(outcome.should_improve) # bool 快捷判断
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum

from openalpha_brain.core.result_router import ParsedWQResult
from openalpha_brain.monitoring.algorithm_telemetry import AlgorithmTelemetryCollector

logger = logging.getLogger(__name__)


class DecisionAction(Enum):
    """WQ 完成回调决策动作"""

    SUCCESS_PASS = "success_pass"
    IMPROVE_AND_RESUBMIT = "improve_and_resubmit"
    WEAK_IMPROVE = "weak_improve"
    REPAIR_AND_RETRY = "repair_and_retry"
    WEAK_SIGNAL_RECORD = "weak_signal_record"
    NOISE_PENALIZE = "noise_penalize"


@dataclass
class DecisionContext:
    """决策输入上下文 (标准化)

    Attributes:
        sharpe: Sharpe 比率
        passed: 是否通过 WQ 审核
        anti_fit_score: 过拟合检测得分 (0-100, None表示未检测)
        parsed_result: 完整的 ParsedWQResult (可选，用于提取ICIR/MFR信息)
        is_high_icir_low_fitness: 是否高ICIR但低Fitness (从parsed_result提取)
        is_efficient_alpha: 是否是高效因子 (从parsed_result提取)
    """

    sharpe: float = 0.0
    passed: bool = False
    anti_fit_score: float | None = None
    parsed_result: ParsedWQResult | None = None
    is_high_icir_low_fitness: bool = False
    is_efficient_alpha: bool = False


@dataclass
class DecisionOutcome:
    """决策输出结果

    Attributes:
        action: 决策动作 (DecisionAction 枚举)
        reason: 决策原因 (人类可读的详细说明)
        should_improve: 是否需要改进 (快捷判断)
        improvement_priority: 改进优先级 (1-5, 5最高)
        anti_fit_penalty_applied: 是否应用了过拟合惩罚
        strategy_d_boost: 是否被策略D (ICIR/MFR) 增强
        processing_time_ms: 决策耗时 (毫秒)
    """

    action: DecisionAction = DecisionAction.NOISE_PENALIZE
    reason: str = ""
    should_improve: bool = False
    improvement_priority: int = 0
    anti_fit_penalty_applied: bool = False
    strategy_d_boost: bool = False
    processing_time_ms: float = 0.0

    @property
    def is_success(self) -> bool:
        """是否成功通过"""
        return self.action == DecisionAction.SUCCESS_PASS

    @property
    def is_noise(self) -> bool:
        """是否判定为噪音"""
        return self.action == DecisionAction.NOISE_PENALIZE

    def to_log_dict(self) -> dict:
        """转换为可日志记录的字典"""
        return {
            "action": self.action.value,
            "reason": self.reason[:100] + "..." if len(self.reason) > 100 else self.reason,
            "should_improve": self.should_improve,
            "priority": self.improvement_priority,
            "anti_fit_penalty": self.anti_fit_penalty_applied,
            "strategy_d_boost": self.strategy_d_boost,
            "time_ms": round(self.processing_time_ms, 2),
        }


class DecisionEngine:
    """因子改进决策引擎 — Pipeline 第二阶段

    将复杂的 if-else 决策逻辑封装为独立的、可测试的、可配置的组件。

    Design Principles:
    - 纯函数式: 相同输入永远产生相同输出 (除了telemetry)
    - 可配置性: 所有阈值通过 config 字典传入
    - 可观测性: 集成 Telemetry 记录每个决策
    - 策略增强: 支持 ICIR/MFR 等高级策略覆盖基础决策

    Config 参数:
        sharpe_pass_threshold: 通过阈值 (默认 1.25)
        sharpe_improve_threshold: 强改进阈值 (默认 0.8)
        weak_improve_threshold: 弱改进阈值 (默认 0.0)
        repair_retry_threshold: 修复重试阈值 (默认 -0.5)
        anti_fit_penalty_threshold: 过拟合惩罚阈值 (默认 40)
    """

    DEFAULT_CONFIG = {
        "sharpe_pass_threshold": 1.25,
        "sharpe_improve_threshold": 0.8,
        "weak_improve_threshold": 0.0,
        "repair_retry_threshold": -0.5,
        "anti_fit_penalty_threshold": 40,
    }

    def __init__(
        self,
        config: dict | None = None,
        telemetry: AlgorithmTelemetryCollector | None = None,
    ):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self._tel = telemetry or AlgorithmTelemetryCollector.get_instance()

    def decide(
        self,
        context: DecisionContext,
    ) -> DecisionOutcome:
        """执行核心决策逻辑

        Args:
            context: 决策上下文 (包含sharpe, passed, anti_fit_score等)

        Returns:
            DecisionOutcome: 结构化决策结果
        """
        t0 = time.perf_counter()
        eid = None
        try:
            self._tel.record_enter_sync(
                "DecisionEngine",
                expr_id=hash(str(context.sharpe)) % 10000,
            )
        except (OSError, ValueError, RuntimeError):
            pass

        outcome = self._apply_decision_matrix(context)

        outcome.processing_time_ms = (time.perf_counter() - t0) * 1000

        try:
            if eid:
                self._tel.record_exit_sync(
                    eid,
                    success=True,
                    metrics={"action": outcome.action.value},
                    duration_ms=outcome.processing_time_ms,
                )
        except (OSError, ValueError, RuntimeError):
            pass

        logger.info(
            "[DECISION] ◆ OUTCOME | action=%s priority=%d reason=%.80s...",
            outcome.action.value,
            outcome.improvement_priority,
            outcome.reason,
        )

        return outcome

    def decide_from_parsed_result(
        self,
        parsed_result: ParsedWQResult,
        anti_fit_score: float | None = None,
    ) -> DecisionOutcome:
        """便捷方法：直接从 ParsedWQResult 创建决策

        Args:
            parsed_result: ResultRouter 的输出
            anti_fit_score: 过拟合得分 (会覆盖 parsed_result 中的值)

        Returns:
            DecisionOutcome: 决策结果
        """
        context = DecisionContext(
            sharpe=parsed_result.raw_sharpe,
            passed=parsed_result.passed,
            anti_fit_score=anti_fit_score or parsed_result.anti_fit_score,
            parsed_result=parsed_result,
            is_high_icir_low_fitness=parsed_result.is_high_icir_low_fitness,
            is_efficient_alpha=parsed_result.is_efficient_alpha,
        )
        return self.decide(context)

    def _apply_decision_matrix(self, ctx: DecisionContext) -> DecisionOutcome:
        """应用核心决策矩阵 (纯函数逻辑)

        决策优先级 (从高到低):
        1. SUCCESS_PASS: passed AND sharpe ≥ threshold
        2. HIGH_ICIR_OVERRIDE: 策略D强制改进
        3. EFFICIENT_ALPHA_BOOST: 策略D提升优先级
        4. IMPROVE_AND_RESUBMIT: sharpe ∈ [improve, pass)
        5. WEAK_IMPROVE: sharpe ∈ [weak, improve)
        6. REPAIR_AND_RETRY: sharpe ∈ [repair, weak)
        7. NOISE_PENALIZE: sharpe < repair
        """
        outcome = DecisionOutcome()
        pass_th = self.config["sharpe_pass_threshold"]
        improve_th = self.config["sharpe_improve_threshold"]
        weak_th = self.config["weak_improve_threshold"]
        repair_th = self.config["repair_retry_threshold"]
        anti_fit_th = self.config["anti_fit_penalty_threshold"]

        anti_fit_penalty = ctx.anti_fit_score is not None and ctx.anti_fit_score < anti_fit_th
        outcome.anti_fit_penalty_applied = anti_fit_penalty

        icir_reason = ""
        mfr_reason = ""

        if ctx.is_high_icir_low_fitness:
            if ctx.parsed_result is not None and ctx.parsed_result.icir_info is not None:
                icir_info = (
                    getattr(ctx.parsed_result.icir_info, "__dict__", {})
                    if hasattr(ctx.parsed_result.icir_info, "__dict__")
                    else {}
                )
                icir_reason = (
                    f"HIGH_ICIR_LOW_FITNESS: ICIR={icir_info.get('icir', 'N/A')} but Fitness < 1.0 → 强制改进路径"
                )
            else:
                icir_reason = "HIGH_ICIR_LOW_FITNESS → 强制改进路径"
            outcome.strategy_d_boost = True

        if ctx.is_efficient_alpha:
            if ctx.parsed_result is not None and ctx.parsed_result.mfr_info is not None:
                mfr_info = (
                    getattr(ctx.parsed_result.mfr_info, "__dict__", {})
                    if hasattr(ctx.parsed_result.mfr_info, "__dict__")
                    else {}
                )
                mfr_reason = (
                    f"EFFICIENT_ALPHA: efficiency={mfr_info.get('efficiency', 'N/A')} > 0.3 "
                    f"→ Experience注入+LLM改进路径"
                )
            else:
                mfr_reason = "EFFICIENT_ALPHA → Experience注入优先"
            outcome.strategy_d_boost = True

        if ctx.passed and ctx.sharpe >= pass_th:
            outcome.action = DecisionAction.SUCCESS_PASS
            outcome.should_improve = False
            outcome.improvement_priority = 0
            extra = f" | ANTI-FIT⚠ score<{anti_fit_th} 仍通过" if anti_fit_penalty else ""
            mfr_extra = f" | {mfr_reason}" if ctx.is_efficient_alpha else ""
            outcome.reason = f"Sharpe={ctx.sharpe:.3f} >= {pass_th} (PASS){extra}{mfr_extra}"
            return outcome

        if ctx.is_high_icir_low_fitness and ctx.sharpe >= improve_th * 0.8:
            outcome.action = DecisionAction.IMPROVE_AND_RESUBMIT
            outcome.should_improve = True
            outcome.improvement_priority = 5
            outcome.reason = (
                f"Sharpe={ctx.sharpe:.3f} + HIGH_ICIR_LOW_FITNESS → 强制改进 (Turnover优化优先) | {icir_reason}"
            )
            return outcome

        if ctx.sharpe >= improve_th:
            outcome.should_improve = True
            if ctx.is_efficient_alpha:
                outcome.action = DecisionAction.IMPROVE_AND_RESUBMIT
                outcome.improvement_priority = 5
                outcome.reason = (
                    f"Sharpe={ctx.sharpe:.3f} in [{improve_th}, {pass_th}) + "
                    f"EFFICIENT_ALPHA → Experience注入优先 | {mfr_reason}"
                )
                return outcome

            if anti_fit_penalty:
                outcome.action = DecisionAction.WEAK_IMPROVE
                outcome.improvement_priority = 2
                outcome.reason = (
                    f"Sharpe={ctx.sharpe:.3f} in [{improve_th}, {pass_th}) → "
                    f"降级为弱改进 (ANTI-FIT score={ctx.anti_fit_score:.0f}<{anti_fit_th})"
                )
                return outcome

            outcome.action = DecisionAction.IMPROVE_AND_RESUBMIT
            outcome.improvement_priority = 4
            outcome.reason = f"Sharpe={ctx.sharpe:.3f} in [{improve_th}, {pass_th}) → 强改进"
            return outcome

        if ctx.sharpe >= weak_th:
            outcome.should_improve = True
            if anti_fit_penalty:
                outcome.action = DecisionAction.REPAIR_AND_RETRY
                outcome.improvement_priority = 1
                outcome.reason = (
                    f"Sharpe={ctx.sharpe:.3f} in [{weak_th}, {improve_th}) → "
                    f"降级为修复/重试 (ANTI-FIT score={ctx.anti_fit_score:.0f}<{anti_fit_th})"
                )
                return outcome

            outcome.action = DecisionAction.WEAK_IMPROVE
            outcome.improvement_priority = 2
            outcome.reason = f"Sharpe={ctx.sharpe:.3f} in [{weak_th}, {improve_th}) → 弱改进尝试"
            return outcome

        if ctx.sharpe >= repair_th:
            outcome.action = DecisionAction.REPAIR_AND_RETRY
            outcome.should_improve = True
            outcome.improvement_priority = 1
            outcome.reason = f"Sharpe={ctx.sharpe:.3f} in [{repair_th}, {weak_th}) → 修复/重试"
            return outcome

        outcome.action = DecisionAction.NOISE_PENALIZE
        outcome.should_improve = False
        outcome.improvement_priority = 0
        outcome.reason = f"Noise factor: Sharpe={ctx.sharpe:.3f} < {repair_th}"
        return outcome
