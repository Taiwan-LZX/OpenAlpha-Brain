"""
OpenAlpha-Brain FeedbackLoopOrchestrator
========================================
自循环 Alpha 挖掘管线编排器 — 将 LLM 生成、预筛选、SlotManager 提交、
WQ 反馈接收、LLM 改进串联成真正的自主闭环。

架构:
  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐
  │ LLM      │ → │ PreFilter│ → │SlotMgr   │ → │ WQ     │
  │ Generate │   │ (质量筛)  │   │(3-slot)  │   │ Platform│
  └──────────┘   └──────────┘   └────┬─────┘   └───┬────┘
                                      │             │
                              提交因子          返回指标
                                      │             ↓
                               ┌──────┴──────────────┐
                               │   Completion Callback│
                               └──────┬──────────────┘
                                      ↓
                         ┌──────────────────────┐
                         │ Sharpe >= 1.25?      │
                         │  YES → 记录成功       │
                         │  NO  → LLM 改进       │
                         │      → 优先提升重投   │
                         └──────────────────────┘
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import aiohttp

from openalpha_brain.core.decision_engine import DecisionEngine
from openalpha_brain.core.result_router import ResultRouter
from openalpha_brain.evolution.mutation_engine import BrainAwareMutationEngine
from openalpha_brain.evolution.near_pass_improver import NearPassCategory, NearPassImprover
from openalpha_brain.generation.prompts import (
    SYSTEM_PROMPT,
    build_start_trigger,
)
from openalpha_brain.monitoring.algorithm_telemetry import AlgorithmTelemetryCollector
from openalpha_brain.optimization.turnover_optimizer import TurnoverOptimizer
from openalpha_brain.services import llm_client
from openalpha_brain.services.brain_submitter import (
    FailureDiagnosis,
    ProxyEvaluator,
    ReflexionEngine,
    SignalQualityPreFilter,
)
from openalpha_brain.services.slot_manager import (
    SlotInfo,
    SlotManager,
)
from openalpha_brain.utils.extract_json_from_llm import extract_json_from_llm
from openalpha_brain.validation.anti_overfit_detector import LightweightAntiOverfitDetector
from openalpha_brain.validation.validator import compute_hierarchical_reward
from openalpha_brain.validation.wq_format_repair import WQFormatRepair

try:
    from openalpha_brain.evolution.adaptive_neutralizer import (
        AdaptiveNeutralizer,
        AdaptiveRecommendation,
        NeutralizationTrial,
    )
except (ImportError, ModuleNotFoundError, AttributeError, OSError):
    AdaptiveNeutralizer = None  # type: ignore[assignment]
    NeutralizationTrial = None  # type: ignore[assignment]
    AdaptiveRecommendation = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class GeneratedAlpha:
    """LLM 生成的候选因子"""

    expression: str
    rationale: str
    strategy: str
    confidence: float
    raw_output: str
    metadata: dict = field(default_factory=dict)


@dataclass
class CycleResult:
    """单个 cycle 的结果"""

    cycle_num: int
    generated: int = 0
    prefiltered: int = 0
    submitted: int = 0
    improved_and_resubmitted: int = 0
    passed: int = 0
    best_sharpe: float | None = None
    duration_sec: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass
class OrchestratorStats:
    """全局统计"""

    total_cycles: int = 0
    total_generated: int = 0
    total_submitted: int = 0
    total_completed: int = 0
    total_passed: int = 0
    total_improved: int = 0
    total_prefiltered: int = 0
    best_sharpe_ever: float = 0.0
    best_expression: str = ""
    pass_rate: float = 0.0
    avg_sharpe: float = 0.0
    improvement_success_rate: float = 0.0
    total_improvement_attempts: int = 0
    successful_improvements: int = 0


class DecisionAction(Enum):
    """WQ 完成回调决策动作"""

    SUCCESS_PASS = "success_pass"
    IMPROVE_AND_RESUBMIT = "improve_and_resubmit"
    WEAK_IMPROVE = "weak_improve"
    REPAIR_AND_RETRY = "repair_and_retry"
    WEAK_SIGNAL_RECORD = "weak_signal_record"
    NOISE_PENALIZE = "noise_penalize"


@dataclass
class ImprovementChain:
    """改进链追踪"""

    original_task_id: str
    original_expression: str
    current_generation: int = 0
    improvements: list[dict] = field(default_factory=list)
    final_sharpe: float | None = None
    final_decision: str = ""


class FeedbackLoopOrchestrator:
    """
    自循环 Alpha 挖掘管线编排器

    职责:
    1. 调用 LLM 生成因子表达式 (IdeaAgent + FactorAgent)
    2. 通过 SignalQualityPreFilter 预筛选
    3. 通过 SlotManager 提交到 WQ (带正确优先级)
    4. 接收 WQ 反馈，判断是否需要改进
    5. 如果需要: 调用 ReflexionEngine (LLM 改进 Block A)
    6. 改进版以更高优先级重新提交
    7. 更新 MAB reward + ExperienceReplay

    Usage:
        orchestrator = FeedbackLoopOrchestrator(
            cookies=cookies,
            slot_manager=slot_manager,
        )
        await orchestrator.start()

        result = await orchestrator.run_one_cycle(focus_area="momentum")

        await orchestrator.stop()
    """

    def __init__(
        self,
        cookies: Any,
        slot_manager: SlotManager,
        config: dict | None = None,
    ):
        """
        初始化 FeedbackLoopOrchestrator

        Args:
            cookies: WQ 认证凭证 (httpx.Cookies)
            slot_manager: SlotManager 实例（必须已初始化）
            config: 配置字典，可覆盖默认值
        """
        self.cookies = cookies
        self.slot_manager = slot_manager
        self.prefilter = SignalQualityPreFilter()

        self._reflexion_engine: ReflexionEngine | None = None
        self._mab: Any = None

        try:
            from openalpha_brain.knowledge.field_proxy_map import get_field_proxy_map

            self._field_proxy_map = get_field_proxy_map()
            logger.info(
                "[FIELD-REC] ✓ FieldProxyMap initialized | fields=%d families=%d",
                self._field_proxy_map.field_count,
                self._field_proxy_map.family_count,
            )
        except (ImportError, AttributeError, RuntimeError, OSError) as exc:
            self._field_proxy_map = None
            logger.warning("[FIELD-REC] ⚠ FieldProxyMap init failed (graceful degradation): %s", exc)

        self.stats = OrchestratorStats()

        self._mutation_engine = BrainAwareMutationEngine()
        self._near_pass_improver = NearPassImprover()
        self._anti_overfit = LightweightAntiOverfitDetector()
        self._format_repairer = WQFormatRepair()
        self._tel = AlgorithmTelemetryCollector.get_instance()

        try:
            from openalpha_brain.knowledge.graph_experience_db import GraphBasedExperienceDB

            self._graph_db = GraphBasedExperienceDB()
            self._graph_db.load()
            logger.info("[FEEDBACK] ✅ GraphBasedExperienceDB loaded successfully")
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("[FEEDBACK] ⚠️ Failed to load GraphBasedExperienceDB: %s", e)
            self._graph_db = None

        try:
            from openalpha_brain.validation.wq_expression_validator import WQExpressionValidator

            self._expr_validator = WQExpressionValidator()
            logger.info("[ORCH] ✓ WQExpressionValidator initialized")
        except (ImportError, AttributeError, RuntimeError, OSError) as exc:
            logger.warning("[ORCH] ⚠ WQExpressionValidator init failed (graceful degradation): %s", exc)
            self._expr_validator = None

        try:
            from openalpha_brain.validation.official_scorer import (
                OfficialScoringAdapter,
            )

            self._scorer = OfficialScoringAdapter()
            logger.info("[ORCH] ✓ OfficialScoringAdapter initialized")
        except (ImportError, AttributeError, RuntimeError, OSError) as exc:
            logger.warning("[ORCH] ⚠ OfficialScoringAdapter init failed (graceful degradation): %s", exc)
            self._scorer = None

        try:
            from openalpha_brain.evolution.fitness_boost import FitnessBoostEngine

            self._fitness_boost = FitnessBoostEngine()
            logger.info("[ORCH] ✓ FitnessBoostEngine initialized")
        except (ImportError, AttributeError, RuntimeError, OSError) as exc:
            logger.warning("[ORCH] ⚠ FitnessBoostEngine init failed (graceful degradation): %s", exc)
            self._fitness_boost = None

        try:
            from openalpha_brain.learning.reflection_engine import ReflectionEngine

            self._reflection_engine = ReflectionEngine()
            logger.info("[ORCH] ✓ ReflectionEngine initialized")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[ORCH] ⚠ ReflectionEngine init failed: %s", exc)
            self._reflection_engine = None

        try:
            self._turnover_optimizer = TurnoverOptimizer()
            logger.info("[ORCH] ✓ TurnoverOptimizer initialized")
        except (ImportError, AttributeError, RuntimeError, OSError) as exc:
            logger.warning("[ORCH] ⚠ TurnoverOptimizer init failed (graceful degradation): %s", exc)
            self._turnover_optimizer = None

        try:
            from openalpha_brain.validation.stability_guard import StabilityGuard

            self._stability_guard = StabilityGuard()
            logger.info("[ORCH] ✓ StabilityGuard initialized")
        except (ImportError, AttributeError, RuntimeError, OSError) as exc:
            logger.warning("[ORCH] ⚠ StabilityGuard init failed (graceful degradation): %s", exc)
            self._stability_guard = None

        try:
            if AdaptiveNeutralizer is not None:
                from openalpha_brain.data import get_data_path

                exp_path = get_data_path("neutralization_experience.json")
                self._adaptive_neutralizer = AdaptiveNeutralizer(
                    experience_path=exp_path,
                    config={
                        "min_trials_for_decision": 3,
                        "ehsani_correlation_estimate": 0.7,
                        "success_rate_threshold": 0.5,
                        "mab_exploration_weight": 0.3,
                    },
                )
                logger.info("[ORCH] ✓ AdaptiveNeutralizer initialized | path=%s", exp_path)
            else:
                self._adaptive_neutralizer = None
                logger.warning("[ORCH] ⚠ AdaptiveNeutralizer module not available")
        except (ValueError, TypeError, OSError) as exc:
            logger.warning("[ORCH] ⚠ AdaptiveNeutralizer init failed (graceful degradation): %s", exc)
            self._adaptive_neutralizer = None

        self._running = False
        self._cycle_num = 0
        self._improvement_chains: dict[str, ImprovementChain] = {}
        self._pending_completions: dict[str, asyncio.Event] = {}
        self._adaptive_weights: dict[str, float] | None = None

        self.config = {
            "max_improvement_generations": 2,
            "sharpe_pass_threshold": 1.25,
            "sharpe_improve_threshold": 0.8,
            "weak_improve_threshold": 0.0,
            "repair_retry_threshold": -0.5,
            "max_weak_improve_attempts": 1,
            "max_repair_retry_attempts": 1,
            "generate_batch_size": 3,
            "llm_generate_fn": llm_client.generate,
            "llm_improve_fn": llm_client.generate,
            "cycle_interval_sec": 2.0,
            "enable_mab_update": True,
            "enable_experience_replay": True,
        }

        if config:
            self.config.update(config)

        self._result_router = ResultRouter(
            stability_guard=self._stability_guard,
            anti_overfit=self._anti_overfit,
            scorer=self._scorer,
            adaptive_neutralizer=self._adaptive_neutralizer,
            telemetry=self._tel,
            cycle_num=self._cycle_num,
        )
        logger.info("[ORCH] ✓ ResultRouter initialized (Pipeline Stage 1)")

        self._decision_engine = DecisionEngine(
            config={
                "sharpe_pass_threshold": self.config.get("sharpe_pass_threshold", 1.25),
                "sharpe_improve_threshold": self.config.get("sharpe_improve_threshold", 0.8),
                "weak_improve_threshold": self.config.get("weak_improve_threshold", 0.0),
                "repair_retry_threshold": self.config.get("repair_retry_threshold", -0.5),
                "anti_fit_penalty_threshold": 40,
            },
            telemetry=self._tel,
        )
        logger.info("[ORCH] ✓ DecisionEngine initialized (Pipeline Stage 2)")

        try:
            from openalpha_brain.generation.alpha_logics import AlphaLogicLibrary
            from openalpha_brain.generation.template_reasoning_generator import TemplateReasoningGenerator

            self._lib = AlphaLogicLibrary()
            self._reasoning_generator = TemplateReasoningGenerator(
                llm_call_fn=self.config.get("llm_generate_fn", llm_client.generate),
                field_proxy_map=self._field_proxy_map,
                alpha_logic_lib=self._lib,
            )
            logger.info(
                "[ORCH] ✓ TemplateReasoningGenerator initialized with %d templates",
                len(self._lib._three_block_templates),
            )
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.warning("[ORCH] TemplateReasoningGenerator init failed: %s (fallback to direct LLM)", e)
            self._reasoning_generator = None
            self._lib = None

    async def start(self) -> None:
        """
        启动管线

        - 初始化 ReflexionEngine
        - 注册 WQ 完成回调到 SlotManager
        - 初始化 MAB（延迟）
        """
        if self._running:
            logger.warning("[ORCH] Already running")
            return

        self._running = True
        self._cycle_num = 0

        async def _default_llm(prompt: str, messages: list, system: str, **kwargs):
            return await self.config["llm_improve_fn"](
                system_prompt=system,
                history=messages,
                user_msg=prompt,
                **kwargs,
            )

        self._reflexion_engine = ReflexionEngine(
            llm_generate_fn=_default_llm,
            proxy_evaluator=ProxyEvaluator(),
        )

        self.slot_manager.register_callback(self._on_wq_completion)

        logger.info(
            "[ORCH] ✓ FeedbackLoopOrchestrator started | config=%s",
            json.dumps(
                {k: v for k, v in self.config.items() if k not in ("llm_generate_fn", "llm_improve_fn")}, default=str
            ),
        )

    async def stop(self) -> None:
        """
        停止管线

        - 注销回调
        - 清理资源
        """
        if not self._running:
            return

        self._running = False

        with contextlib.suppress(ValueError):
            self.slot_manager.unregister_callback(self._on_wq_completion)

        pending_count = len(self._pending_completions)
        if pending_count > 0:
            logger.info("[ORCH] Waiting for %d pending completions...", pending_count)
            for event in self._pending_completions.values():
                event.set()

        logger.info(
            "[ORCH] ✓ Stopped | stats: cycles=%d generated=%d submitted=%d passed=%d best=%.3f",
            self.stats.total_cycles,
            self.stats.total_generated,
            self.stats.total_submitted,
            self.stats.total_passed,
            self.stats.best_sharpe_ever,
        )

    async def run_one_cycle(
        self,
        focus_area: str = "momentum",
    ) -> CycleResult:
        """
        执行一个完整的挖掘 cycle

        流程:
          1. generate_batch() → 生成 N 个候选因子
          2. prefilter_batch() → 预筛选
          3. submit_to_slots() → 提交到 SlotManager (TIER_3 原始)
          4. (异步等待完成回调)

        Args:
            focus_area: 策略方向 ("momentum" / "reversal" / "volatility" / ...)

        Returns:
            CycleResult 包含本周期统计
        """
        if not self._running:
            raise RuntimeError("Orchestrator not started. Call start() first.")

        self._cycle_num += 1
        cycle_start = time.monotonic()
        result = CycleResult(cycle_num=self._cycle_num)

        logger.info(
            "[ORCH] ═══ Cycle %d START | focus=%s ═══",
            self._cycle_num,
            focus_area,
        )

        try:
            batch_size = self.config["generate_batch_size"]

            generated = await self.generate_batch(
                focus_area=focus_area,
                count=batch_size,
            )
            result.generated = len(generated)
            self.stats.total_generated += len(generated)

            logger.info(
                "[ORCH] Cycle %d: Generated %d candidates",
                self._cycle_num,
                len(generated),
            )

            filtered_batch, prefiltered_count = self.prefilter_batch(generated)
            result.prefiltered = prefiltered_count
            self.stats.total_prefiltered += prefiltered_count

            logger.info(
                "[ORCH] Cycle %d: Prefilter passed %d / %d (%d rejected)",
                self._cycle_num,
                len(filtered_batch),
                len(generated),
                prefiltered_count,
            )

            submitted_count = await self.submit_to_slots(filtered_batch)
            result.submitted = submitted_count
            self.stats.total_submitted += submitted_count

            logger.info(
                "[ORCH] Cycle %d: Submitted %d to SlotManager",
                self._cycle_num,
                submitted_count,
            )

        except Exception as exc:
            error_msg = f"Cycle {self._cycle_num} error: {exc}"
            logger.error("[ORCH] %s", error_msg, exc_info=True)
            result.errors.append(error_msg)

        result.duration_sec = time.monotonic() - cycle_start
        self.stats.total_cycles += 1

        self._update_global_stats()

        logger.info(
            "[ORCH] ═══ Cycle %d END | gen=%d filt=%d sub=%d dur=%.1fs ═══",
            self._cycle_num,
            result.generated,
            result.prefiltered,
            result.submitted,
            result.duration_sec,
        )

        return result

    async def run_continuous(
        self,
        max_cycles: int = 0,
        focus_area: str = "momentum",
        interval_sec: float | None = None,
    ) -> list[CycleResult]:
        """
        持续运行多个 cycle（主循环模式）

        Args:
            max_cycles: 最大 cycle 数 (0=无限)
            focus_area: 策略方向
            interval_sec: cycle 间隔秒数 (None=使用配置默认值)

        Returns:
            所有 cycle 的结果列表
        """
        results = []
        actual_interval = interval_sec or self.config.get("cycle_interval_sec", 2.0)
        cycle_limit = max_cycles if max_cycles > 0 else float("inf")

        logger.info(
            "[ORCH] Starting continuous mode | max_cycles=%s interval=%.1fs focus=%s",
            "∞" if max_cycles == 0 else max_cycles,
            actual_interval,
            focus_area,
        )

        while self._running and len(results) < cycle_limit:
            try:
                cycle_result = await self.run_one_cycle(focus_area=focus_area)
                results.append(cycle_result)

                if cycle_result.best_sharpe is not None and cycle_result.best_sharpe > self.stats.best_sharpe_ever:
                        self.stats.best_sharpe_ever = cycle_result.best_sharpe

                if self._running and len(results) < cycle_limit:
                    await asyncio.sleep(actual_interval)

            except asyncio.CancelledError:
                logger.info("[ORCH] Continuous run cancelled")
                break
            except Exception as exc:
                logger.error("[ORCH] Continuous run error: %s", exc, exc_info=True)
                await asyncio.sleep(actual_interval * 2)

        logger.info(
            "[ORCH] Continuous run completed | %d cycles executed",
            len(results),
        )
        return results

    async def _on_wq_completion(
        self,
        slot_info: SlotInfo,
        brain_result: Any,
    ) -> None:
        eid = None
        t0 = time.perf_counter()
        with contextlib.suppress(OSError, ValueError, RuntimeError):
            eid = await self._tel.record_enter(
                "FeedbackOrchestrator", cycle_id=self._cycle_num, expr_id=hash(slot_info.expression) % 10000
            )

        self._result_router.update_cycle_num(self._cycle_num)

        parsed_result = await self._result_router.route(slot_info, brain_result)

        task_id = parsed_result.task_id
        expression = parsed_result.expression
        sharpe = parsed_result.raw_sharpe
        passed = parsed_result.passed

        self.stats.total_completed += 1

        logger.info(
            "[ORCH] ▶ WQ COMPLETION (ROUTED) | task=%s expr=%.50s... sharpe=%.3f passed=%s fitness=%s",
            task_id,
            expression,
            sharpe,
            passed,
            parsed_result.fitness,
        )

        decision_outcome = self._decision_engine.decide_from_parsed_result(
            parsed_result,
            anti_fit_score=parsed_result.anti_fit_score,
        )

        decision = decision_outcome.action
        action = decision_outcome.reason

        logger.info(
            "[ORCH] ◆ DECISION | task=%s action=%s sharpe=%.3f priority=%d reason=%.80s...",
            task_id,
            decision.value,
            sharpe,
            decision_outcome.improvement_priority,
            action,
        )

        if self._adaptive_neutralizer is not None and NeutralizationTrial is not None:
            try:
                category = self._extract_direction(expression) or "momentum"
                self._result_router.record_neutralizer_outcome(
                    parsed_result,
                    expression=expression,
                    category=category,
                    level=self._infer_neutralization_level(expression),
                )
            except (OSError, ValueError, RuntimeError) as rec_exc:
                logger.debug("[ORCH] [ADAPT-NEUT] Record outcome failed (non-critical): %s", rec_exc)

        if decision == DecisionAction.SUCCESS_PASS:
            await self._handle_success(slot_info, brain_result, expression)

        elif decision == DecisionAction.IMPROVE_AND_RESUBMIT:
            await self._handle_improvement(
                slot_info,
                brain_result,
                expression,
                task_id,
            )

        elif decision == DecisionAction.WEAK_IMPROVE:
            await self._handle_weak_improve(
                slot_info,
                brain_result,
                expression,
                task_id,
                sharpe,
            )

        elif decision == DecisionAction.REPAIR_AND_RETRY:
            await self._handle_repair_retry(
                slot_info,
                brain_result,
                expression,
                task_id,
                sharpe,
            )

        elif decision == DecisionAction.WEAK_SIGNAL_RECORD:
            await self._handle_weak_signal(
                slot_info,
                brain_result,
                expression,
                sharpe,
            )

        elif decision == DecisionAction.NOISE_PENALIZE:
            await self._handle_noise(
                slot_info,
                brain_result,
                expression,
                sharpe,
            )

        completion_event = self._pending_completions.get(task_id)
        if completion_event:
            completion_event.set()
            try:
                ms = (time.perf_counter() - t0) * 1000
                _fitness = getattr(parsed_result, "fitness", 0) or 0.0
                _turnover = getattr(brain_result, "turnover", None) or 0
                await self._tel.record_exit(
                    "FeedbackOrchestrator",
                    eid,
                    metrics={
                        "sharpe": round(sharpe, 3),
                        "fitness": round(_fitness, 3),
                        "turnover": round(float(_turnover), 2) if _turnover else 0,
                        "decision": decision.value,
                    },
                    duration_ms=ms,
                )
            except (OSError, ValueError, RuntimeError):
                pass

    def _make_decision(
        self,
        sharpe: float,
        passed: bool,
        brain_result: Any,
        anti_fit_score: float | None = None,
        score_report=None,  # 🆕 策略 D: 可选的评分报告
    ) -> tuple[DecisionAction, str]:
        """
        核心决策逻辑 — 根据 Sharpe 和通过状态决定下一步动作

        5档决策矩阵 (2026-05-31 升级):
          ≥ 1.25     → SUCCESS_PASS (优秀因子)
          0.8 ~ 1.25 → IMPROVE_AND_RESUBMIT (强信号，值得深度改进)
          0 ~ 0.8    → WEAK_IMPROVE (弱信号但方向正确，给1次改进机会)
          -0.5 ~ 0   → REPAIR_AND_RETRY (可能格式错误或方向反转)
          < -0.5     → NOISE_PENALIZE (纯噪音，直接惩罚)

        Anti-Overfit 调节:
          score < 40 (不推荐) → 降级决策优先级 (IMPROVE→WEAK, WEAK→REPAIR)

        🆕 策略 D 增强 (ICIR + Multi-Faceted Reward):
          - HIGH_ICIR_LOW_FITNESS → 强制走 IMPROVE_AND_RESUBMIT (Turnover优化优先)
          - EFFICIENT_ALPHA → 提升 IMPROVE 优先级 (Experience注入优先)

        Returns:
            (decision, reason) 元组
        """
        time.perf_counter()
        with contextlib.suppress(OSError, ValueError, RuntimeError):
            self._tel.record_enter_sync(
                "FeedbackOrchestrator", cycle_id=self._cycle_num, expr_id=hash(str(sharpe)) % 10000
            )
        pass_threshold = self.config["sharpe_pass_threshold"]
        improve_threshold = self.config["sharpe_improve_threshold"]
        weak_improve_threshold = self.config["weak_improve_threshold"]
        repair_retry_threshold = self.config["repair_retry_threshold"]

        anti_fit_penalty = anti_fit_score is not None and anti_fit_score < 40

        # 🆕 策略 D: 提取 ICIR 和 MFR 信息
        is_high_icir_low_fitness = False
        is_efficient_alpha = False
        icir_override_reason = ""
        mfr_boost_reason = ""

        if score_report is not None and hasattr(score_report, "multi_layer_result") and score_report.multi_layer_result:
                ml_result = score_report.multi_layer_result
                if ml_result.get("is_high_icir_low_fitness"):
                    is_high_icir_low_fitness = True
                    icir_info = ml_result.get("icir_metrics", {})
                    icir_override_reason = (
                        f"HIGH_ICIR_LOW_FITNESS: ICIR={icir_info.get('icir', 'N/A')} but Fitness < 1.0 → 强制改进路径"
                    )

                # 检查 EFFICIENT_ALPHA
                if ml_result.get("is_efficient_alpha"):
                    is_efficient_alpha = True
                    mfr_info = ml_result.get("multi_faceted_reward", {})
                    mfr_boost_reason = (
                        f"EFFICIENT_ALPHA: efficiency={mfr_info.get('efficiency', 'N/A')} > 0.3 "
                        f"→ Experience注入+LLM改进路径"
                    )

        # 基础决策逻辑
        if passed and sharpe >= pass_threshold:
            return (
                DecisionAction.SUCCESS_PASS,
                f"Sharpe={sharpe:.3f} >= {pass_threshold} (PASS)"
                + (" | ANTI-FIT⚠ score<40 仍通过" if anti_fit_penalty else "")
                + (f" | {mfr_boost_reason}" if is_efficient_alpha else ""),
            )

        # 🆕 策略 D: HIGH_ICIR_LOW_FITNESS 特殊处理
        if is_high_icir_low_fitness and sharpe >= improve_threshold * 0.8:
            return (
                DecisionAction.IMPROVE_AND_RESUBMIT,
                f"Sharpe={sharpe:.3f} + HIGH_ICIR_LOW_FITNESS → 强制改进 (Turnover优化优先) | {icir_override_reason}",
            )

        if sharpe >= improve_threshold:
            # 🆕 策略 D: EFFICIENT_ALPHA 提升优先级
            if is_efficient_alpha:
                return (
                    DecisionAction.IMPROVE_AND_RESUBMIT,
                    f"Sharpe={sharpe:.3f} in [{improve_threshold}, {pass_threshold}) + "
                    f"EFFICIENT_ALPHA → Experience注入优先 | {mfr_boost_reason}",
                )

            if anti_fit_penalty:
                return (
                    DecisionAction.WEAK_IMPROVE,
                    f"Sharpe={sharpe:.3f} in [{improve_threshold}, {pass_threshold}) → "
                    f"降级为弱改进 (ANTI-FIT score={anti_fit_score:.0f}<40)",
                )
            return (
                DecisionAction.IMPROVE_AND_RESUBMIT,
                f"Sharpe={sharpe:.3f} in [{improve_threshold}, {pass_threshold}) → 强改进",
            )

        if sharpe >= weak_improve_threshold:
            if anti_fit_penalty:
                return (
                    DecisionAction.REPAIR_AND_RETRY,
                    f"Sharpe={sharpe:.3f} in [{weak_improve_threshold}, {improve_threshold}) → "
                    f"降级为修复/重试 (ANTI-FIT score={anti_fit_score:.0f}<40)",
                )
            return (
                DecisionAction.WEAK_IMPROVE,
                f"Sharpe={sharpe:.3f} in [{weak_improve_threshold}, {improve_threshold}) → 弱改进尝试",
            )

        if sharpe >= repair_retry_threshold:
            return (
                DecisionAction.REPAIR_AND_RETRY,
                f"Sharpe={sharpe:.3f} in [{repair_retry_threshold}, {weak_improve_threshold}) → 修复/重试",
            )

        return (
            DecisionAction.NOISE_PENALIZE,
            f"Noise factor: Sharpe={sharpe:.3f} < {repair_retry_threshold}",
        )

    async def _handle_success(
        self,
        slot_info: SlotInfo,
        brain_result: Any,
        expression: str,
    ) -> None:
        """处理成功通过的因子"""
        sharpe = getattr(brain_result, "sharpe", 0) or 0.0
        alpha_id = getattr(brain_result, "alpha_id", "") or ""

        self.stats.total_passed += 1

        if sharpe > self.stats.best_sharpe_ever:
            self.stats.best_sharpe_ever = sharpe
            self.stats.best_expression = expression
            logger.info(
                "[ORCH] 🏆 NEW BEST SHARPE! %.3f | expr=%.60s...",
                sharpe,
                expression,
            )

        if self.config.get("enable_mab_update") and self._mab:
            try:
                reward = compute_hierarchical_reward(
                    expression,
                    {
                        "sharpe": sharpe,
                        "passed": True,
                    },
                )
                direction = self._extract_direction(expression)
                self._mab.update(arm=direction, reward=reward)
                logger.debug("[ORCH] MAB updated: direction=%s reward=%.3f", direction, reward)
            except (OSError, ValueError, RuntimeError) as exc:
                logger.debug("[ORCH] MAB update failed (non-critical): %s", exc)

        logger.info(
            "[ORCH] ✅ SUCCESS | alpha_id=%s sharpe=%.3f expr=%.60s...",
            alpha_id[:12] if alpha_id else "n/a",
            sharpe,
            expression,
        )

    async def _handle_improvement(
        self,
        slot_info: SlotInfo,
        brain_result: Any,
        expression: str,
        task_id: str,
    ) -> None:
        eid = None
        t0 = time.perf_counter()
        with contextlib.suppress(OSError, ValueError, RuntimeError):
            eid = await self._tel.record_enter(
                "FeedbackOrchestrator", cycle_id=self._cycle_num, expr_id=hash(expression) % 10000
            )
        sharpe = getattr(brain_result, "sharpe", 0) or 0.0
        fitness = getattr(brain_result, "fitness", 0) or 0.0
        turnover = getattr(brain_result, "turnover", None)

        wq_feedback = {
            "sharpe": sharpe,
            "turnover": turnover,
            "fitness": fitness,
            "checks": getattr(brain_result, "brain_checks", []) or [],
        }

        if self._adaptive_neutralizer is not None and AdaptiveRecommendation is not None:
            try:
                category = self._extract_direction(expression) or "momentum"
                adap_rec = self._adaptive_neutralizer.analyze_and_recommend(
                    expression=expression,
                    category=category,
                    wq_metrics=wq_feedback,
                )
                wq_feedback["adaptive_neutralization"] = {
                    "recommended_level": adap_rec.recommended_level,
                    "confidence": adap_rec.confidence,
                    "reasoning": adap_rec.reasoning,
                    "is_forced": adap_rec.is_forced,
                    "mab_adjusted": adap_rec.mab_adjusted,
                }
                logger.info(
                    "[ORCH] [ADAPT-NEUT] recommendation: level=%s conf=%.2f forced=%s mab=%s reason=%.80s",
                    adap_rec.recommended_level,
                    adap_rec.confidence,
                    adap_rec.is_forced,
                    adap_rec.mab_adjusted,
                    adap_rec.reasoning,
                )
                if adap_rec.is_forced and "DOWNGRADE" in adap_rec.reasoning:
                    wq_feedback["neutralization_override"] = "downgrade"
                    logger.warning(
                        "[ORCH] [ADAPT-NEUT] ⚠ FORCED DOWNGRADE | task=%s → reducing neutralization level",
                        task_id,
                    )

                try:
                    sampling_weights = self._adaptive_neutralizer.get_sampling_weights(category)
                    if sampling_weights:
                        self._adaptive_weights = sampling_weights
                        try:
                            from openalpha_brain.core import loop_state as _ls_module

                            _ls_module._adaptive_weights = sampling_weights
                        except (ImportError, AttributeError):
                            pass
                        logger.info(
                            "[ORCH] [ADAPT-NEUT-WEIGHT] Weights generated: category=%s weights=%s",
                            category,
                            {k: round(v, 3) for k, v in list(sampling_weights.items())[:10]},
                        )
                except (OSError, ValueError, RuntimeError) as weight_exc:
                    logger.debug("[ORCH] [ADAPT-NEUT-WEIGHT] Weight retrieval failed: %s", weight_exc)
            except (OSError, ValueError, RuntimeError) as adap_exc:
                logger.debug("[ORCH] [ADAPT-NEUT] Analysis failed (non-critical): %s", adap_exc)

        reflection_result = None
        boost_context: dict = {}
        to_context: dict = {}
        diagnosis_available = False

        if self._reflection_engine is not None:
            try:
                brain_result_dict = {
                    "sharpe": sharpe,
                    "fitness": fitness,
                    "turnover": turnover,
                    "checks": wq_feedback.get("checks", []),
                }
                reflection_result = await self._reflection_engine.reflect_on_failure(
                    expression=expression,
                    brain_result=brain_result_dict,
                )
                diagnosis_available = getattr(reflection_result, "llm_diagnosis_available", False)

                if diagnosis_available:
                    boost_context["llm_root_cause"] = reflection_result.root_cause
                    boost_context["llm_composite_factors"] = list(reflection_result.composite_factors)
                    boost_context["primary_failure_stage"] = reflection_result.failure_stage
                    to_context["llm_root_cause"] = reflection_result.root_cause
                    to_context["llm_composite_factors"] = list(reflection_result.composite_factors)
                    to_context["primary_failure_stage"] = reflection_result.failure_stage

                logger.info(
                    "[REFLECTION-DIAG-FLOW] Diagnosis available=%s | "
                    "root_cause=%.60s | composite_factors=%s | stage=%s | task=%s",
                    "Yes" if diagnosis_available else "No",
                    getattr(reflection_result, "root_cause", "")[:60] if reflection_result else "",
                    list(getattr(reflection_result, "composite_factors", [])) if reflection_result else [],
                    getattr(reflection_result, "failure_stage", "") if reflection_result else "",
                    task_id,
                )
            except Exception as refl_early_exc:  # noqa: BLE001
                logger.debug(
                    "[REFLECTION-DIAG-FLOW] Early reflection failed (non-critical): %s",
                    refl_early_exc,
                )

        if self._stability_guard is not None:
            try:
                last_stability = self._stability_guard.get_summary()
                is_stable = last_stability.get("current_stability_score", 1.0) >= 0.35
                wq_feedback["is_stable"] = is_stable

                if not is_stable:
                    logger.info(
                        "[ORCH] [STABILITY-IMPROVE] Unstable environment detected | "
                        "task=%s score=%.2f → prioritizing low-risk improvements",
                        task_id,
                        last_stability.get("current_stability_score", 0),
                    )
                    wq_feedback["stability_mode"] = "conservative"
                    constraints = (
                        self._stability_guard._last_result.get("constraints")
                        if self._stability_guard._last_result
                        else None
                    )
                    if constraints:
                        wq_feedback["stability_constraints"] = constraints
                        constraint_prompt = (
                            f"\n\n## ⚠️ STABILITY CONSTRAINTS (Weak Evaluation Guard)\n"
                            f"Current expression pool is UNSTABLE. You MUST respect these constraints:\n"
                            f"- Locked root operator: {constraints.get('locked_root_op', 'N/A')}\n"
                            f"- Preferred composition: {constraints.get('preferred_composition', 'N/A')}\n"
                            f"- Reason: {constraints.get('reason', 'N/A')}\n"
                            f"Avoid drastic structural changes. Focus on parameter tuning only."
                        )
                        wq_feedback["targeted_feedback"] = wq_feedback.get("targeted_feedback", "") + constraint_prompt
            except (OSError, ValueError, RuntimeError) as stab_improve_exc:
                logger.debug("[ORCH] [STABILITY] improvement check failed: %s", stab_improve_exc)

        if self._turnover_optimizer is not None and turnover is not None:
            try:
                to_value = float(turnover) if isinstance(turnover, (int, float, str)) else None
                to_threshold = 0.25

                if diagnosis_available and reflection_result:
                    cf = reflection_result.composite_factors
                    if any(f in ("turnover", "signal_weakness") for f in cf):
                        to_threshold = 0.15
                        logger.info(
                            "[REFLECTION-DIAG-FLOW] Priority reordered: TurnoverOptimizer "
                            "threshold lowered %.2f→%.2f (composite_factors=%s) | task=%s",
                            0.25,
                            to_threshold,
                            cf,
                            task_id,
                        )

                if to_value is not None and to_value > to_threshold:
                    logger.info(
                        "[ORCH] [TURNOVER-OPT] High TO detected (%.1f%%) → launching adaptive optimization | task=%s",
                        to_value * 100,
                        task_id,
                    )
                    logger.info(
                        "[REFLECTION-DIAG-FLOW] Root cause passed to TurnoverOptimizer: %s | task=%s",
                        "Yes" if bool(to_context) else "No",
                        task_id,
                    )

                    to_result = self._turnover_optimizer.optimize(
                        expr=expression,
                        sharpe=sharpe,
                        fitness=fitness,
                        turnover=to_value,
                        max_variants=8,
                        context=to_context if to_context else None,
                    )

                    if to_result.analysis.is_bottleneck and to_result.variants:
                        to_submitted = 0
                        _chain = self._improvement_chains.get(task_id)
                        _gen = (_chain.current_generation if _chain else 0) + 1
                        for variant in to_result.safe_variants()[:6]:
                            pf_to = self.prefilter.prefilter(variant.expression)
                            if pf_to.passed:
                                try:
                                    from openalpha_brain.services.slot_manager import PriorityTier

                                    await self.slot_manager.submit_improved(
                                        expression=variant.expression,
                                        source="turnover_optimizer",
                                        improvement_generation=_gen,
                                        priority_tier=PriorityTier.TIER_1_HIGH_IMPROVED,
                                    )
                                    to_submitted += 1
                                    logger.info(
                                        "[ORCH] [TURNOVER-OPT] #%d [%s] ΔTO≈%.0% conf=%.0% %s",
                                        to_submitted,
                                        variant.strategy,
                                        variant.expected_to_reduction,
                                        variant.confidence,
                                        variant.expression[:60],
                                    )
                                except (TimeoutError, aiohttp.ClientError, ConnectionError) as to_exc:
                                    logger.warning("[ORCH] [TURNOVER-OPT] Submit failed: %s", to_exc)

                        if to_submitted > 0:
                            logger.info(
                                "[ORCH] [TURNOVER-OPT] Submitted %d low-TO variants | "
                                "bottleneck=%s potential_gain=%.0% | task=%s",
                                to_submitted,
                                to_result.analysis.severity.value,
                                to_result.analysis.potential_gain,
                                task_id,
                            )
                            self.stats.total_improved += to_submitted
            except (ValueError, TypeError, OSError, RuntimeError) as to_err:
                logger.warning("[ORCH] [TURNOVER-OPT] Engine error: %s", to_err)

        chain = self._improvement_chains.get(task_id)
        if chain is None:
            chain = ImprovementChain(
                original_task_id=task_id,
                original_expression=expression,
            )
            self._improvement_chains[task_id] = chain

        if chain.current_generation >= self.config["max_improvement_generations"]:
            logger.info(
                "[ORCH] ⚠ Max generations reached (%d) for task=%s",
                self.config["max_improvement_generations"],
                task_id,
            )
            return

        near_pass_analysis = self._near_pass_improver.analyze(
            sharpe=sharpe,
            fitness=fitness,
            turnover=turnover,
            checks=wq_feedback.get("checks", []),
        )

        if near_pass_analysis.category in (
            NearPassCategory.SHARPE_GOOD_FITNESS_POOR,
            NearPassCategory.BOTH_NEAR,
        ):
            logger.info(
                "[ORCH] 🎯 NEAR-PASS DETECTED | task=%s sharpe=%.2f fitness=%.2f "
                "target=%s gap(sharpe=%.2f fitness=%.2f) → trying deterministic variants first",
                task_id,
                sharpe,
                fitness,
                near_pass_analysis.primary_fix_target,
                near_pass_analysis.sharpe_gap,
                near_pass_analysis.fitness_gap,
            )
            variants = self._near_pass_improver.generate_deterministic_variants(
                expression=expression,
                analysis=near_pass_analysis,
                max_variants=10,
            )
            submitted_count = 0
            for variant in variants:
                pf_result = self.prefilter.prefilter(variant.expression)
                if pf_result.passed:
                    try:
                        from openalpha_brain.services.slot_manager import PriorityTier

                        await self.slot_manager.submit_improved(
                            expression=variant.expression,
                            source="near_pass_deterministic",
                            improvement_generation=chain.current_generation + 1,
                            priority_tier=PriorityTier.TIER_1_HIGH_IMPROVED,
                        )
                        submitted_count += 1
                        logger.info(
                            "[ORCH] [NEAR-PASS] Submitted variant #%d [%s] %s",
                            submitted_count,
                            variant.mutation_type,
                            variant.expression[:60],
                        )
                    except (TimeoutError, aiohttp.ClientError, ConnectionError) as submit_exc:
                        logger.warning("[ORCH] [NEAR-PASS] Submit failed: %s", submit_exc)

            if submitted_count > 0:
                logger.info(
                    "[ORCH] [NEAR-PASS] Submitted %d deterministic variants for task=%s",
                    submitted_count,
                    task_id,
                )
                chain.current_generation += 1
                self.stats.total_improved += submitted_count

            if (
                self._fitness_boost is not None
                and near_pass_analysis.category == NearPassCategory.SHARPE_GOOD_FITNESS_POOR
            ):
                try:
                    logger.info(
                        "[REFLECTION-DIAG-FLOW] Root cause passed to FitnessBoost: %s | task=%s",
                        "Yes" if bool(boost_context) else "No",
                        task_id,
                    )
                    fb_result = self._fitness_boost.generate_boost_variants(
                        expression=expression,
                        sharpe=sharpe,
                        fitness=fitness,
                        turnover=turnover,
                        max_variants=8,
                        context=boost_context if boost_context else None,
                    )
                    fb_submitted = 0
                    for fv in fb_result.variants[:6]:
                        pf_fb = self.prefilter.prefilter(fv.expression)
                        if pf_fb.passed:
                            try:
                                from openalpha_brain.services.slot_manager import PriorityTier

                                await self.slot_manager.submit_improved(
                                    expression=fv.expression,
                                    source="fitness_boost",
                                    improvement_generation=chain.current_generation + 1,
                                    priority_tier=PriorityTier.TIER_1_HIGH_IMPROVED,
                                )
                                fb_submitted += 1
                                logger.info(
                                    "[ORCH] [FITNESS-BOOST] #%d [%s] Δ=%.3f %s",
                                    fb_submitted,
                                    fv.boost_tier,
                                    fv.expected_fitness_delta,
                                    fv.expression[:60],
                                )
                            except (TimeoutError, aiohttp.ClientError, ConnectionError) as fb_exc:
                                logger.warning("[ORCH] [FITNESS-BOOST] Submit failed: %s", fb_exc)

                    if fb_submitted > 0:
                        logger.info(
                            "[ORCH] [FITNESS-BOOST] Submitted %d fitness-boosted variants | "
                            "bottleneck=%s best_Δ=%.3f | task=%s",
                            fb_submitted,
                            fb_result.analysis_summary,
                            fb_result.best_variant().expected_fitness_delta if fb_result.best_variant() else 0,
                            task_id,
                        )
                        self.stats.total_improved += fb_submitted
                except (TimeoutError, aiohttp.ClientError, ConnectionError) as fb_err:
                    logger.warning("[ORCH] [FITNESS-BOOST] Engine error: %s", fb_err)

        if self._graph_db is not None:
            try:
                query_start = time.monotonic()
                similar_experiences = self._graph_db.query_similar_experiences(
                    current_expression=expression,
                    top_k=3,
                    min_similarity=0.3,
                )
                query_elapsed = time.monotonic() - query_start

                logger.info(
                    "[EXPERIENCE] 🔍 Queried similar experiences | task=%s count=%d elapsed=%.3fs",
                    task_id,
                    len(similar_experiences),
                    query_elapsed,
                )

                if similar_experiences:
                    similarity_scores = [exp.get("similarity", 0) for exp in similar_experiences]
                    logger.info(
                        "[EXPERIENCE] 📚 Found %d experiences | similarity range=[%.2f, %.2f] avg=%.2f",
                        len(similar_experiences),
                        min(similarity_scores),
                        max(similarity_scores),
                        sum(similarity_scores) / len(similarity_scores),
                    )

                    experience_context = self._build_experience_context(similar_experiences)

                    current_analysis = {
                        "sharpe": sharpe,
                        "fitness": fitness,
                        "category": near_pass_analysis.category.name,
                        "turnover": turnover,
                    }

                    improved_expr = await self._call_llm_with_context(
                        expression=expression,
                        wq_feedback=wq_feedback,
                        experience_context=experience_context,
                        current_analysis=current_analysis,
                        improvement_gen=chain.current_generation + 1,
                        original_task_id=task_id,
                    )

                    if improved_expr:
                        chain.current_generation += 1
                        chain.improvements.append(
                            {
                                "gen": chain.current_generation,
                                "from_sharpe": sharpe,
                                "expression": improved_expr,
                                "timestamp": datetime.now(UTC).isoformat(),
                                "strategy": "llm_enhanced_with_experience",
                            }
                        )
                        self.stats.total_improved += 1
                        self.stats.total_improvement_attempts += 1

                        try:
                            category = self._categorize_result(sharpe, fitness)
                            self._graph_db.add_factor_experience(
                                expression=expression,
                                wq_feedback={
                                    "sharpe": sharpe,
                                    "fitness": fitness,
                                    "turnover": turnover,
                                    "checks": wq_feedback.get("checks", []),
                                },
                                improvement_result={
                                    "strategy": "llm_enhanced",
                                    "new_expression": improved_expr,
                                    "success": True,
                                },
                                category=category,
                            )
                            logger.info(
                                "[EXPERIENCE] 💾 Recorded experience | expr=%.50s... category=%s",
                                expression,
                                category,
                            )
                        except (OSError, ValueError, RuntimeError) as record_exc:
                            logger.warning(
                                "[EXPERIENCE] ⚠️ Failed to record experience: %s",
                                record_exc,
                            )

                        return
                    try:
                        category = self._categorize_result(sharpe, fitness)
                        self._graph_db.add_factor_experience(
                            expression=expression,
                            wq_feedback={
                                "sharpe": sharpe,
                                "fitness": fitness,
                                "turnover": turnover,
                                "checks": wq_feedback.get("checks", []),
                            },
                            improvement_result={
                                "strategy": "llm_enhanced",
                                "new_expression": None,
                                "success": False,
                            },
                            category=category,
                        )
                        logger.info(
                            "[EXPERIENCE] 💾 Recorded failed experience | expr=%.50s... category=%s",
                            expression,
                            category,
                        )
                    except (OSError, ValueError, RuntimeError) as record_exc:
                        logger.warning(
                            "[EXPERIENCE] ⚠️ Failed to record failed experience: %s",
                            record_exc,
                        )

            except (ValueError, TypeError, OSError, RuntimeError) as exp_exc:
                logger.warning(
                    "[EXPERIENCE] ⚠️ Experience injection failed, fallback to standard LLM: %s",
                    exp_exc,
                )

        if self._reflection_engine is not None and reflection_result is None:
            try:
                brain_result_dict = {
                    "sharpe": sharpe,
                    "fitness": fitness,
                    "turnover": turnover,
                    "checks": wq_feedback.get("checks", []),
                }

                reflection_result = await self._reflection_engine.reflect_on_failure(
                    expression=expression,
                    brain_result=brain_result_dict,
                )

                logger.info(
                    "[REFLECTION] 🔍 Analysis complete (fallback) | task=%s stage=%s reason=%s conf=%.2f",
                    task_id,
                    reflection_result.failure_stage,
                    reflection_result.failure_reason[:80],
                    reflection_result.confidence,
                )

                recent_reflections = self._reflection_engine.get_recent_reflections(n=5)
                failure_patterns = self._reflection_engine.get_failure_patterns()

                logger.info(
                    "[REFLECTION] 📊 Historical patterns | recent=%d patterns=%s",
                    len(recent_reflections),
                    json.dumps(failure_patterns, ensure_ascii=False),
                )

                is_high_icir_low_fitness = (
                    sharpe >= 0.8 and fitness < 1.0 and turnover is not None and float(turnover) > 0.30
                )

                if is_high_icir_low_fitness:
                    logger.warning(
                        "[REFLECTION] ⚠ HIGH_ICIR_LOW_FITNESS DETECTED | task=%s "
                        "sharpe=%.2f fitness=%.2f turnover=%.2f%% → specialized analysis",
                        task_id,
                        sharpe,
                        fitness,
                        float(turnover) * 100,
                    )

                    targeted_hypothesis = (
                        f"Signal strength is good (Sharpe={sharpe:.2f}) but execution cost is too high "
                        f"(Fitness={fitness:.2f}, Turnover={float(turnover) * 100:.1f}%). "
                        f"Root cause: high-frequency trading signal without proper smoothing. "
                        f"Recommended fix: apply ts_decay_linear(window=10-20) or increase lookback periods."
                    )

                    critique_result = await self._reflection_engine.self_critique(
                        expression=expression,
                        hypothesis=targeted_hypothesis,
                    )

                    if critique_result.critique_available:
                        logger.info(
                            "[REFLECTION] 🎯 Specialized critique | consistency=%.2f issues=%d suggestions=%d",
                            critique_result.consistency_score or 0,
                            len(critique_result.issues),
                            len(critique_result.suggestions),
                        )

                        reflection_context = self._build_reflection_context(
                            reflection_result=reflection_result,
                            critique_result=critique_result,
                            recent_reflections=recent_reflections,
                            failure_patterns=failure_patterns,
                            is_specialized=True,
                        )

                        improved_expr = await self._call_llm_with_reflection_context(
                            expression=expression,
                            wq_feedback=wq_feedback,
                            reflection_context=reflection_context,
                            improvement_gen=chain.current_generation + 1,
                            original_task_id=task_id,
                        )

                        if improved_expr:
                            chain.current_generation += 1
                            chain.improvements.append(
                                {
                                    "gen": chain.current_generation,
                                    "from_sharpe": sharpe,
                                    "expression": improved_expr,
                                    "timestamp": datetime.now(UTC).isoformat(),
                                    "strategy": "reflection_enhanced_specialized",
                                }
                            )
                            self.stats.total_improved += 1
                            self.stats.total_improvement_attempts += 1

                            try:
                                self._record_reflection_outcome(
                                    expression=expression,
                                    success=True,
                                    new_expression=improved_expr,
                                    strategy="specialized_high_icir_low_fitness",
                                )
                            except (OSError, ValueError, RuntimeError) as record_exc:
                                logger.warning("[REFLECTION] ⚠ Failed to record success: %s", record_exc)

                            return
                        try:
                            self._record_reflection_outcome(
                                expression=expression,
                                success=False,
                                new_expression=None,
                                strategy="specialized_high_icir_low_fitness",
                            )
                        except (OSError, ValueError, RuntimeError) as record_exc:
                            logger.warning("[REFLECTION] ⚠ Failed to record failure: %s", record_exc)

                if reflection_result.confidence >= 0.6:
                    hypothesis = (
                        f"Factor failed at stage '{reflection_result.failure_stage}': "
                        f"{reflection_result.failure_reason}. "
                        f"Suggested fix: {reflection_result.suggested_fix}"
                    )

                    critique_result = await self._reflection_engine.self_critique(
                        expression=expression,
                        hypothesis=hypothesis,
                    )

                    reflection_context = self._build_reflection_context(
                        reflection_result=reflection_result,
                        critique_result=critique_result,
                        recent_reflections=recent_reflections,
                        failure_patterns=failure_patterns,
                        is_specialized=False,
                    )

                    improved_expr = await self._call_llm_with_reflection_context(
                        expression=expression,
                        wq_feedback=wq_feedback,
                        reflection_context=reflection_context,
                        improvement_gen=chain.current_generation + 1,
                        original_task_id=task_id,
                    )

                    if improved_expr:
                        chain.current_generation += 1
                        chain.improvements.append(
                            {
                                "gen": chain.current_generation,
                                "from_sharpe": sharpe,
                                "expression": improved_expr,
                                "timestamp": datetime.now(UTC).isoformat(),
                                "strategy": "reflection_enhanced",
                            }
                        )
                        self.stats.total_improved += 1
                        self.stats.total_improvement_attempts += 1

                        try:
                            self._record_reflection_outcome(
                                expression=expression,
                                success=True,
                                new_expression=improved_expr,
                                strategy="standard_reflection",
                            )
                        except (OSError, ValueError, RuntimeError) as record_exc:
                            logger.warning("[REFLECTION] ⚠ Failed to record success: %s", record_exc)

                        return
                    try:
                        self._record_reflection_outcome(
                            expression=expression,
                            success=False,
                            new_expression=None,
                            strategy="standard_reflection",
                        )
                    except (OSError, ValueError, RuntimeError) as record_exc:
                        logger.warning("[REFLECTION] ⚠ Failed to record failure: %s", record_exc)

            except Exception as refl_exc:  # noqa: BLE001
                logger.warning(
                    "[REFLECTION] ⚠ ReflectionEngine integration failed, falling back to standard path: %s",
                    refl_exc,
                )

        improved_expr = await self._improve_and_resubmit(
            expression=expression,
            wq_feedback=wq_feedback,
            improvement_gen=chain.current_generation + 1,
            original_task_id=task_id,
        )

        if improved_expr:
            chain.current_generation += 1
            chain.improvements.append(
                {
                    "gen": chain.current_generation,
                    "from_sharpe": sharpe,
                    "expression": improved_expr,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
            self.stats.total_improved += 1
            self.stats.total_improvement_attempts += 1
            try:
                ms = (time.perf_counter() - t0) * 1000
                await self._tel.record_exit(
                    "FeedbackOrchestrator", eid, metrics={"improvement_path_taken": "completed"}, duration_ms=ms
                )
            except (OSError, ValueError, RuntimeError):
                pass

    async def _handle_weak_signal(
        self,
        slot_info: SlotInfo,
        brain_result: Any,
        expression: str,
        sharpe: float,
    ) -> None:
        """处理弱信号因子"""
        logger.info(
            "[ORCH] 📉 WEAK SIGNAL | sharpe=%.3f (not improving to save tokens)",
            sharpe,
        )

        if self.config.get("enable_mab_update") and self._mab:
            try:
                low_reward = sharpe / self.config["sharpe_pass_threshold"]
                direction = self._extract_direction(expression)
                self._mab.update(arm=direction, reward=low_reward)
            except (OSError, ValueError, RuntimeError):
                pass

    async def _handle_noise(
        self,
        slot_info: SlotInfo,
        brain_result: Any,
        expression: str,
        sharpe: float,
    ) -> None:
        """处理噪音因子"""
        logger.warning(
            "[ORCH] ❌ NOISE FACTOR | sharpe=%.3f < -0.5 (penalizing)",
            sharpe,
        )

        if self.config.get("enable_mab_update") and self._mab:
            try:
                direction = self._extract_direction(expression)
                self._mab.penalize(arm=direction, value=0.5)
            except (OSError, ValueError, RuntimeError):
                pass

    async def _handle_weak_improve(
        self,
        slot_info: SlotInfo,
        brain_result: Any,
        expression: str,
        task_id: str,
        sharpe: float,
    ) -> None:
        """
        处理弱改进信号 (0 ≤ Sharpe < 0.8)

        策略:
          - 信号方向可能正确，但强度不足
          - 给 LLM 1 次改进机会（生成 targeted feedback）
          - 改进后提升到 TIER_2 优先级（中等优先）
          - 超过 max_weak_improve_attempts 后降级为 WEAK_SIGNAL_RECORD
        """
        logger.info(
            "[ORCH] 📊 WEAK_IMPROVE | task=%s sharpe=%.3f (信号方向正确但强度不足, 尝试1次改进)",
            task_id,
            sharpe,
        )

        chain = self._improvement_chains.get(task_id)
        if chain is None:
            chain = ImprovementChain(
                original_task_id=task_id,
                original_expression=expression,
            )
            self._improvement_chains[task_id] = chain

        weak_improve_count = sum(1 for imp in chain.improvements if imp.get("decision_type") == "WEAK_IMPROVE")

        max_attempts = self.config.get("max_weak_improve_attempts", 1)

        if weak_improve_count >= max_attempts:
            logger.info(
                "[ORCH] ⚠ Weak improve max attempts (%d) reached for task=%s, downgrading to WEAK_SIGNAL",
                max_attempts,
                task_id,
            )
            await self._handle_weak_signal(slot_info, brain_result, expression, sharpe)
            return

        wq_metrics = {
            "sharpe": sharpe,
            "turnover": getattr(brain_result, "turnover", None) or 0,
            "fitness": getattr(brain_result, "fitness", None) or 0,
            "ic_mean": getattr(brain_result, "ic_mean", 0) or 0,
            "ic_ir": getattr(brain_result, "ic_ir", 0) or 0,
        }
        checks = getattr(brain_result, "brain_checks", []) or []

        if self._mutation_engine is not None:
            diagnosis = self._mutation_engine.diagnose(expression, wq_metrics, checks)
            mutation_prompt = self._mutation_engine.generate_mutation_prompt(
                diagnosis,
                expression,
                inspiration_exprs=self._get_inspiration_exprs(),
            )
        else:
            from openalpha_brain.evolution.mutation_engine import Diagnosis, MutationStrategy

            diagnosis = Diagnosis(
                strategy=MutationStrategy.REGENERATE_FULL, reason="mutation_engine_unavailable", composite_score=30
            )
            mutation_prompt = (
                f"Generate a new alpha expression based on: {expression}. Ensure >=5 operators, cross-family fields."
            )

        logger.info(
            "[ORCH] 🧬 MUTATION-ENGINE DIAGNOSIS | task=%s strategy=%s reason=%s",
            task_id,
            diagnosis.strategy.value,
            diagnosis.reason,
        )

        wq_feedback = {
            "sharpe": sharpe,
            "turnover": wq_metrics["turnover"],
            "fitness": wq_metrics["fitness"],
            "checks": checks,
            "targeted_feedback": mutation_prompt,
            "decision_type": "WEAK_IMPROVE",
            "mutation_diagnosis": {
                "strategy": diagnosis.strategy.value,
                "reason": diagnosis.reason,
                "details": diagnosis.details,
            },
        }

        improved_expr = await self._improve_and_resubmit(
            expression=expression,
            wq_feedback=wq_feedback,
            improvement_gen=chain.current_generation + 1,
            original_task_id=task_id,
            priority_tier="TIER_2",
        )

        if improved_expr:
            chain.current_generation += 1
            chain.improvements.append(
                {
                    "gen": chain.current_generation,
                    "from_sharpe": sharpe,
                    "expression": improved_expr,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "decision_type": "WEAK_IMPROVE",
                }
            )
            self.stats.total_improved += 1
            self.stats.total_improvement_attempts += 1
            logger.info(
                "[ORCH] ✅ WEAK_IMPROVE SUCCESS | task=%s improved_expr=%.60s...",
                task_id,
                improved_expr,
            )

    async def _handle_repair_retry(
        self,
        slot_info: SlotInfo,
        brain_result: Any,
        expression: str,
        task_id: str,
        sharpe: float,
    ) -> None:
        """
        处理修复/重试 (-0.5 ≤ Sharpe < 0)

        策略 (参考 QuantGPT MutationEngine):
          1. 先检查 WQ 格式错误（如 "lookback required"）
             → 如果是格式错误 → 自动修复后重新提交（不经过 LLM）
          2. 如果不是格式错误 → 尝试 1 次 LLM 改进
             → 可能的改进策略：
               a. IC 为负 → 反转信号方向
               b. 评分极低 → 完全重写
               c. 无非线性变换 → 注入 tanh/power
          3. 改进后提升到 TIER_3 优先级（较低优先）
          4. 超过 max_repair_retry_attempts 后降级为 NOISE_PENALIZE
        """
        logger.info(
            "[ORCH] 🔧 REPAIR_AND_RETRY | task=%s sharpe=%.3f (检查格式错误或尝试方向反转)",
            task_id,
            sharpe,
        )

        chain = self._improvement_chains.get(task_id)
        if chain is None:
            chain = ImprovementChain(
                original_task_id=task_id,
                original_expression=expression,
            )
            self._improvement_chains[task_id] = chain

        repair_count = sum(1 for imp in chain.improvements if imp.get("decision_type") == "REPAIR_AND_RETRY")

        max_attempts = self.config.get("max_repair_retry_attempts", 1)

        if repair_count >= max_attempts:
            logger.info(
                "[ORCH] ⚠ Repair retry max attempts (%d) reached for task=%s, downgrading to NOISE",
                max_attempts,
                task_id,
            )
            await self._handle_noise(slot_info, brain_result, expression, sharpe)
            return

        checks = getattr(brain_result, "brain_checks", []) or []
        format_error = self._detect_format_error(checks, expression)

        wq_metrics = {
            "sharpe": sharpe,
            "turnover": getattr(brain_result, "turnover", None) or 0,
            "fitness": getattr(brain_result, "fitness", None) or 0,
            "ic_mean": getattr(brain_result, "ic_mean", 0) or 0,
            "ic_ir": getattr(brain_result, "ic_ir", 0) or 0,
        }

        if format_error:
            logger.info(
                "[ORCH] 🔍 FORMAT ERROR DETECTED | task=%s error=%s (尝试 WQFormatRepair...)",
                task_id,
                format_error,
            )

            error_msgs = []
            for c in checks:
                result_val = c.get("result")
                if result_val is False or str(result_val).lower() in ("false", "fail", "failed"):
                    error_msgs.append(str(c.get("name", "")) + ": " + str(c.get("value", "")))

            repaired_expr = None
            for err_msg in error_msgs[:3]:
                try:
                    diagnosis = self._format_repairer.diagnose(err_msg, expression)
                    if diagnosis.confidence >= 0.7:
                        repaired = self._format_repairer.repair(diagnosis)
                        is_valid, warnings = self._format_repairer.validate_repaired(repaired)
                        if is_valid:
                            logger.info(
                                "[ORCH] [FORMAT-REPAIR] %s → %s (conf=%.2f strategy=%s)",
                                expression[:50],
                                repaired[:50],
                                diagnosis.confidence,
                                diagnosis.error_type,
                            )
                            repaired_expr = repaired
                            break
                        logger.info(
                            "[ORCH] [FORMAT-REPAIR] repaired but validation failed: %s",
                            warnings,
                        )
                except (ValueError, TypeError, SyntaxError) as exc:
                    logger.debug("[ORCH] [FORMAT-REPAIR] diagnose/repair failed: %s", exc)

            if repaired_expr is None:
                repaired_expr = self._auto_repair_expression(expression, format_error)

            if repaired_expr and repaired_expr != expression:
                pf_result = self.prefilter.prefilter(repaired_expr)
                if pf_result.passed:
                    try:
                        new_task_id = await self.slot_manager.submit_improved(
                            expression=repaired_expr,
                            original_task_id=task_id or f"repair_{repair_count + 1}",
                            wq_feedback={"sharpe": sharpe, "repair_type": "auto_format_fix"},
                            improvement_gen=chain.current_generation + 1,
                            predicted_pass_prob=0.3,
                        )

                        chain.current_generation += 1
                        chain.improvements.append(
                            {
                                "gen": chain.current_generation,
                                "from_sharpe": sharpe,
                                "expression": repaired_expr,
                                "timestamp": datetime.now(UTC).isoformat(),
                                "decision_type": "REPAIR_AND_RETRY",
                                "repair_type": "auto_format_fix",
                            }
                        )

                        logger.info(
                            "[ORCH] ✅ AUTO-REPAIRED | task=%s new_task=%s expr=%.60s...",
                            task_id,
                            new_task_id,
                            repaired_expr,
                        )
                        return
                    except (TimeoutError, aiohttp.ClientError, ConnectionError) as exc:
                        logger.error("[ORCH] Auto-repair submit failed: %s", exc)

        try:
            mutation_diagnosis = self._mutation_engine.diagnose(expression, wq_metrics, checks)
            mutation_prompt = self._mutation_engine.generate_mutation_prompt(
                mutation_diagnosis,
                expression,
                inspiration_exprs=self._get_inspiration_exprs(),
            )
            logger.info(
                "[ORCH] 🧬 MUTATION-ENGINE REPAIR DIAGNOSIS | strategy=%s reason=%s",
                mutation_diagnosis.strategy.value,
                mutation_diagnosis.reason,
            )
        except (ValueError, TypeError, SyntaxError) as exc:
            logger.warning("[ORCH] MutationEngine diagnosis failed in repair_retry: %s", exc)
            mutation_diagnosis = None
            mutation_prompt = self._generate_mutation_based_repair_feedback(
                expression,
                wq_metrics,
                checks,
            )

        wq_feedback = {
            "sharpe": sharpe,
            "turnover": wq_metrics["turnover"],
            "fitness": wq_metrics["fitness"],
            "checks": checks,
            "targeted_feedback": mutation_prompt,
            "decision_type": "REPAIR_AND_RETRY",
            "mutation_diagnosis": {
                "strategy": mutation_diagnosis.strategy.value if mutation_diagnosis else "unknown",
                "reason": mutation_diagnosis.reason if mutation_diagnosis else "diagnosis_failed",
                "details": mutation_diagnosis.details if mutation_diagnosis else {},
            },
        }

        improved_expr = await self._improve_and_resubmit(
            expression=expression,
            wq_feedback=wq_feedback,
            improvement_gen=chain.current_generation + 1,
            original_task_id=task_id,
            priority_tier="TIER_3",
        )

        if improved_expr:
            chain.current_generation += 1
            chain.improvements.append(
                {
                    "gen": chain.current_generation,
                    "from_sharpe": sharpe,
                    "expression": improved_expr,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "decision_type": "REPAIR_AND_RETRY",
                    "repair_type": "llm_improve",
                }
            )
            self.stats.total_improved += 1
            self.stats.total_improvement_attempts += 1
            logger.info(
                "[ORCH] ✅ REPAIR_AND_RETRY SUCCESS | task=%s improved_expr=%.60s...",
                task_id,
                improved_expr,
            )

    async def _improve_and_resubmit(
        self,
        expression: str,
        wq_feedback: dict,
        improvement_gen: int = 1,
        original_task_id: str = "",
        priority_tier: str | None = None,
    ) -> str | None:
        """
        调用 LLM 改进因子并重新提交

        流程:
          1. 构建 Reflexion prompt (包含 wq_feedback 的具体数值)
          2. 调用 LLM (通过 llm_client.generate())
          3. 解析新表达式
          4. PreFilter 检查
          5. submit_improved() 到 SlotManager (根据 priority_tier 选择 Tier)

        Args:
            expression: 原始因子表达式
            wq_feedback: WQ 返回的指标 {sharpe, turnover, ..., targeted_feedback, decision_type}
            improvement_gen: 当前改进代数
            original_task_id: 原任务 ID
            priority_tier: 优先级 ("TIER_1", "TIER_2", "TIER_3", None=auto)

        Returns:
            改进后的表达式字符串，失败返回 None
        """
        decision_type = wq_feedback.get("decision_type", "IMPROVE_AND_RESUBMIT")

        logger.info(
            "[ORCH] 🔧 IMPROVING | gen=%d original_task=%s decision=%s sharpe_input=%.3f tier=%s",
            improvement_gen,
            original_task_id,
            decision_type,
            wq_feedback.get("sharpe", 0),
            priority_tier or "auto",
        )

        try:
            _ft_module = __import__(
                "openalpha_brain.services.brain_submitter",
                fromlist=["FailureType"],
            )
            _failure_type = _ft_module.FailureType
            _failure_value = getattr(_failure_type, "LOW_SHARPE", None)

            if decision_type == "WEAK_IMPROVE":
                failure_type_str = "WEAK_SIGNAL"
                suggested_action = "enhance_strength"
                reason = f"Sharpe={wq_feedback.get('sharpe', 0):.3f} in [0, 0.8) - 信号强度不足"
            elif decision_type == "REPAIR_AND_RETRY":
                failure_type_str = "NEGATIVE_SHARPE"
                suggested_action = "repair_or_reverse"
                reason = f"Sharpe={wq_feedback.get('sharpe', 0):.3f} in [-0.5, 0) - 需要修复或反转"
            else:
                failure_type_str = "LOW_SHARPE"
                suggested_action = "mutate_A"
                reason = f"Sharpe={wq_feedback.get('sharpe', 0):.3f} below threshold"

            if _failure_value is None:
                logger.warning("[ORCH] FailureType not found, using raw string")
                failure_diagnosis = FailureDiagnosis(
                    failure_type=failure_type_str,
                    confidence=0.8,
                    raw_metrics=wq_feedback,
                    suggested_action=suggested_action,
                    reason=reason,
                )
            else:
                failure_diagnosis = FailureDiagnosis(
                    failure_type=_failure_value,
                    confidence=0.8,
                    raw_metrics=wq_feedback,
                    suggested_action=suggested_action,
                    reason=reason,
                )

            if self._reflexion_engine is None:
                logger.warning("[ORCH] ReflexionEngine not initialized, using simple LLM call")
                return await self._simple_llm_improve(expression, wq_feedback, decision_type)

            reflexion_result = await self._reflexion_engine.reflect_and_improve(
                original_expr=expression,
                mutated_expr=expression,
                failure=failure_diagnosis,
                max_reflections=1,
            )

            if not reflexion_result.passed_proxy:
                logger.info(
                    "[ORCH] Reflexion proxy rejected (score=%.3f), skipping",
                    reflexion_result.final_verdict.overall_score if reflexion_result.final_verdict else 0,
                )
                return None

            improved_expr = reflexion_result.final_expr
            if not improved_expr or not isinstance(improved_expr, str):
                logger.warning("[ORCH] ReflexionEngine returned empty expression")
                return None

        except (TimeoutError, ConnectionError, OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("[ORCH] ReflexionEngine failed: %s, fallback to simple LLM", exc)
            improved_expr = await self._simple_llm_improve(expression, wq_feedback, decision_type)
            if not improved_expr:
                return None

        if not improved_expr or not isinstance(improved_expr, str):
            logger.warning("[ORCH] improved_expr is empty after improvement, skipping submission")
            return None

        pf_result = self.prefilter.prefilter(improved_expr)
        if not pf_result.passed:
            logger.info(
                "[ORCH] Improved expression failed prefilter: %s",
                pf_result.reason,
            )
            return None

        try:
            new_task_id = await self.slot_manager.submit_improved(
                expression=improved_expr,
                original_task_id=original_task_id or f"gen{improvement_gen}",
                wq_feedback=wq_feedback,
                improvement_gen=improvement_gen,
                predicted_pass_prob=self._get_predicted_prob(decision_type),
                priority_tier=priority_tier,
            )

            logger.info(
                "[ORCH] ✅ IMPROVED & RESUBMITTED | new_task=%s tier=%s expr=%.60s...",
                new_task_id,
                priority_tier or "auto",
                improved_expr,
            )

            self.stats.successful_improvements += 1
            return improved_expr

        except (TimeoutError, aiohttp.ClientError, ConnectionError) as exc:
            logger.error("[ORCH] Failed to submit improved expression: %s", exc)
            return None

    def _get_predicted_prob(self, decision_type: str) -> float:
        """
        根据决策类型返回预测通过概率

        Args:
            decision_type: 决策类型字符串

        Returns:
            预测通过概率 (0.0 - 1.0)
        """
        prob_map = {
            "IMPROVE_AND_RESUBMIT": 0.6,
            "WEAK_IMPROVE": 0.4,
            "REPAIR_AND_RETRY": 0.3,
        }
        return prob_map.get(decision_type, 0.5)

    async def _simple_llm_improve(
        self,
        expression: str,
        wq_feedback: dict,
        decision_type: str = "IMPROVE_AND_RESUBMIT",
    ) -> str | None:
        """
        简单 LLM 改进回退方案（当 ReflexionEngine 不可用时）

        Args:
            expression: 原始表达式
            wq_feedback: WQ 反馈数据
            decision_type: 决策类型 (影响 prompt 生成策略)
        """
        sharpe = wq_feedback.get("sharpe", 0)
        turnover = wq_feedback.get("turnover")
        checks = wq_feedback.get("checks", [])
        targeted_feedback = wq_feedback.get("targeted_feedback", "")

        check_str = (
            "; ".join([f"{c.get('name', '?')}={c.get('value', '?')}" for c in checks if c.get("result") == "FAIL"])
            if checks
            else "none"
        )

        if decision_type == "WEAK_IMPROVE":
            improve_prompt = f"""You are a quantitative alpha factor engineer.
The following alpha shows weak positive signal and needs enhancement.

ORIGINAL EXPRESSION:
{expression}

BRAIN FEEDBACK:
- Sharpe Ratio: {sharpe:.3f} (positive but weak, need >= 1.25)
- Turnover: {turnover}% (ideal: 10-30%)
- Failed Checks: {check_str}

TARGETED DIAGNOSIS:
{targeted_feedback}

WEAK SIGNAL ENHANCEMENT GUIDELINES:
1. Signal strength is insufficient → add volatility normalization (ts_zscore, ts_std_dev)
2. Introduce non-linear transformation: wrap rank() with tanh() or signed_power(expr, 0.5)
3. Tighten group_neutralize conditions or add sector-level neutralization
4. If using simple returns → switch to risk-adjusted metrics (Sharpe-style normalization)
5. Add ts_decay_linear(window=5-10) to reduce noise
6. Keep the SAME economic logic, just enhance signal-to-noise ratio

Return ONLY the improved FastExpr expression (single line, no JSON, no explanation).
Example: group_neutralize(tanh(rank(ts_zscore(close, 20))), industry)"""

        elif decision_type == "REPAIR_AND_RETRY":
            improve_prompt = f"""You are a quantitative alpha factor engineer specialized in factor repair.
The following alpha has NEGATIVE Sharpe and needs fundamental repair or reversal.

ORIGINAL EXPRESSION:
{expression}

BRAIN FEEDBACK:
- Sharpe Ratio: {sharpe:.3f} (NEGATIVE! Critical issue)
- Turnover: {turnover}% (ideal: 10-30%)
- Failed Checks: {check_str}

TARGETED DIAGNOSIS (QuantGPT MutationEngine Strategy):
{targeted_feedback}

REPAIR/REVERSAL GUIDELINES (choose ONE strategy):
1. ⚠ REVERSE SIGNAL: If IC is negative → add minus sign before expression or reverse rank logic
2. ⚠ COMPLETE REWRITE: If multiple checks failed → use different data fields/operators entirely
3. 💡 INJECT NONLINEARITY: Missing tanh/power → add signed_power(expr, 0.5) for fat-tail robustness
4. 💡 REDUCE TURNOVER: High turnover → wrap with ts_decay_linear(expr, 10-15)
5. 💡 CHANGE OPERATOR FAMILY: ts_delta→ts_regression, rank→ts_zscore, etc.

CRITICAL: The current signal is HARMFUL (negative Sharpe). You MUST change the fundamental approach.
Do NOT make trivial changes - this needs significant mutation.

Return ONLY the repaired FastExpr expression (single line, no JSON, no explanation).
Example: -group_neutralize(ts_decay_linear(rank(ts_regression(close, volume, 20)), 10), industry)"""

        else:
            improve_prompt = f"""You are a quantitative alpha factor engineer.
The following alpha was submitted to WorldQuant BRAIN but needs improvement.

ORIGINAL EXPRESSION:
{expression}

BRAIN FEEDBACK:
- Sharpe Ratio: {sharpe:.3f} (need >= 1.25)
- Turnover: {turnover}% (ideal: 10-30%)
- Failed Checks: {check_str}

IMPROVEMENT GUIDELINES:
1. If Sharpe too low: add volatility normalization (ts_zscore, ts_std_dev), tighten neutralization
2. If Turnover too high: wrap with ts_decay_linear(expr, window) where window in [5, 10, 15]
3. Keep group_neutralize(...) as outer wrapper
4. Change operator family if needed (rank→ts_zscore, ts_delta→ts_regression)
5. Use different lookback windows (avoid trivial changes)

Return ONLY the improved FastExpr expression (single line, no JSON, no explanation).
Example: group_neutralize(ts_decay_linear(rank(ts_zscore(close, 20)), 10), industry)"""

        try:
            raw_response = await self.config["llm_improve_fn"](
                system_prompt="You are a quantitative alpha factor improver. Output ONLY the improved expression.",
                history=[],
                user_msg=improve_prompt,
                session_id="orch_improve",
                cycle=0,
            )

            improved = raw_response.strip()

            if improved.startswith("```"):
                improved = improved.split("\n")[1]
            if improved.endswith("```"):
                improved = improved.rsplit("```", 1)[0].strip()

            if improved and improved != expression and len(improved) > 10:
                return improved

            return None

        except Exception as exc:  # noqa: BLE001
            logger.error("[ORCH] Simple LLM improve failed: %s", exc)
            return None

    def _build_field_recommendation(self, focus_area: str, cycle: int) -> dict:
        """
        构建本轮推荐的字段列表（带家族标注和拥挤度）

        根据 focus_area 映射到推荐的字段族，返回结构化的字段推荐信息，
        用于注入到 LLM prompt 中引导生成更多样化的因子。

        Args:
            focus_area: 策略方向 ("momentum" / "reversal" / ...)
            cycle: 当前 cycle 编号

        Returns:
            包含字段推荐信息的字典，包含 families、crowding 级别等
        """
        if self._field_proxy_map is None:
            logger.warning("[FIELD-REC] FieldProxyMap not available, returning empty recommendation")
            return {
                "focus_area": focus_area,
                "cycle": cycle,
                "families": {},
                "total_fields_available": 0,
                "recommendation": "FieldProxyMap unavailable. Use default fields.",
            }

        fpm = self._field_proxy_map

        focus_to_families = {
            "momentum": ["price_trend", "volume_liquidity", "analyst_estimates"],
            "reversal": ["price_trend", "volatility_metrics", "valuation"],
            "volatility": ["volatility_metrics", "option_implied", "credit_risk"],
            "value": ["valuation", "profitability", "quality_factor_model"],
            "quality": ["quality_factor_model", "profitability", "growth_rates"],
            "liquidity": ["volume_liquidity", "alternative_web_traffic", "short_interest"],
            "lead_lag": ["analyst_estimates", "earnings_surprise", "sentiment"],
        }

        recommended_families = focus_to_families.get(focus_area, ["price_trend", "valuation"])

        fields_by_family = {}
        for family_id in recommended_families:
            family = fpm.get_family(family_id)
            if family:
                fields = fpm.get_fields_in_family(family_id, exclude_cold=True)[:5]
                if fields:
                    if family.l1_category == "price":
                        crowding = "HIGH"
                    elif family.l1_category in ("derived_factor", "microstructure"):
                        crowding = "MEDIUM"
                    else:
                        crowding = "LOW"

                    fields_by_family[family_id] = {
                        "family_name": family.family_name,
                        "l1_category": family.l1_category,
                        "crowding": crowding,
                        "fields": fields,
                    }

        total_fields = sum(len(v["fields"]) for v in fields_by_family.values())

        logger.info(
            "[FIELD-REC] Built recommendation | focus=%s cycle=%d families=%d fields=%d",
            focus_area,
            cycle,
            len(fields_by_family),
            total_fields,
        )

        return {
            "focus_area": focus_area,
            "cycle": cycle,
            "families": fields_by_family,
            "total_fields_available": total_fields,
            "recommendation": "优先使用 LOW CROWDING 族字段。强制至少使用 1 个非 price 族字段。",
        }

    async def _generate_single_direct(
        self,
        focus_area: str,
    ) -> GeneratedAlpha:
        """
        Fallback: 直接调用 LLM 生成单个因子（不使用模板推理）

        Args:
            focus_area: 策略方向

        Returns:
            单个 GeneratedAlpha 对象
        """
        trigger_msg = build_start_trigger(
            cycle=self._cycle_num,
            focus_area=focus_area,
        )

        field_rec = self._build_field_recommendation(focus_area, self._cycle_num)

        low_crowding_families = [
            f"{fid}: {', '.join(data['fields'][:5])}"
            for fid, data in field_rec["families"].items()
            if data["crowding"] == "LOW"
        ]
        medium_crowding_families = [
            f"{fid}: {', '.join(data['fields'][:5])}"
            for fid, data in field_rec["families"].items()
            if data["crowding"] == "MEDIUM"
        ]
        high_crowding_families = [
            f"{fid}: {', '.join(data['fields'][:5])}"
            for fid, data in field_rec["families"].items()
            if data["crowding"] == "HIGH"
        ]

        field_rec_prompt = ""
        if field_rec["total_fields_available"] > 0:
            field_rec_prompt = f"""

▶ FIELD RECOMMENDATION (from FieldProxyMap):
  Focus: {focus_area} | Cycle: {self._cycle_num}

  🟢 LOW CROWDING (PREFERRED):
"""
            if low_crowding_families:
                field_rec_prompt += "\n".join(f"    {f}" for f in low_crowding_families) + "\n"
            else:
                field_rec_prompt += "    (none available)\n"

            field_rec_prompt += """
  🟡 MEDIUM CROWDING:
"""
            if medium_crowding_families:
                field_rec_prompt += "\n".join(f"    {f}" for f in medium_crowding_families) + "\n"
            else:
                field_rec_prompt += "    (none available)\n"

            field_rec_prompt += """
  🔴 HIGH CROWDING (MINIMIZE):
"""
            if high_crowding_families:
                field_rec_prompt += "\n".join(f"    {f}" for f in high_crowding_families) + "\n"
            else:
                field_rec_prompt += "    (none available)\n"

            field_rec_prompt += """
  RULE: Your expression MUST include at least 1 field from 🟢 or 🟡 category.
  AVOID overusing 🔴 fields (close, volume, etc.) - they are crowded and have low alpha potential.
"""

        batch_request = f"""{trigger_msg}
{field_rec_prompt}
Generate exactly 1 alpha expression for {focus_area} strategy.
Output a JSON object containing:
- "expression": valid FastExpr string
- "rationale": economic logic (1 sentence)
- "strategy": momentum|reversal|volatility|value|quality
- "confidence": 0.0-1.0

CRITICAL: All expressions MUST include group_neutralize(..., industry)."""

        raw_response = await self.config["llm_generate_fn"](
            system_prompt=SYSTEM_PROMPT,
            history=[],
            user_msg=batch_request,
            session_id=f"orch_gen_{self._cycle_num}_direct",
            cycle=self._cycle_num,
        )

        parsed = extract_json_from_llm(raw_response)

        if isinstance(parsed, str):
            parsed = json.loads(parsed)

        if isinstance(parsed, dict):
            expr = parsed.get("expression", "").strip()
            if expr and len(expr) > 5:
                logger.info("[ORCH] [DIRECT] Generated single alpha: %s", expr[:80])
                return GeneratedAlpha(
                    expression=expr,
                    rationale=parsed.get("rationale", ""),
                    strategy=parsed.get("strategy", focus_area),
                    confidence=float(parsed.get("confidence", 0.7)),
                    raw_output=raw_response,
                    metadata={**parsed, "source": "direct_llm"},
                )
        elif isinstance(parsed, list) and len(parsed) > 0:
            item = parsed[0]
            if isinstance(item, dict):
                expr = item.get("expression", "").strip()
                if expr and len(expr) > 5:
                    logger.info("[ORCH] [DIRECT] Generated single alpha from list: %s", expr[:80])
                    return GeneratedAlpha(
                        expression=expr,
                        rationale=item.get("rationale", ""),
                        strategy=item.get("strategy", focus_area),
                        confidence=float(item.get("confidence", 0.7)),
                        raw_output=raw_response,
                        metadata={**item, "source": "direct_llm"},
                    )

        raise ValueError("Failed to parse valid alpha from LLM response")

    async def generate_batch(
        self,
        focus_area: str,
        count: int = 3,
    ) -> list[GeneratedAlpha]:
        """
        生成一批因子表达式 — 优先使用模板引导式深度推理

        Args:
            focus_area: 策略方向 ("momentum" / "reversal" / "volatility")
            count: 生成数量

        Returns:
            GeneratedAlpha 列表
        """
        logger.info(
            "[ORCH] Generating batch | focus=%s count=%d reasoning=%s",
            focus_area,
            count,
            self._reasoning_generator is not None,
        )

        alphas = []

        use_reasoning = self._reasoning_generator is not None

        for i in range(count):
            if use_reasoning:
                try:
                    reasoning_result = await self._reasoning_generator.generate(
                        focus_area=focus_area,
                        cycle=self._cycle_num,
                        rag_context=getattr(self, "_last_rag_context", None),
                    )

                    expr = reasoning_result.final_expression
                    approved = reasoning_result.approved

                    if self._expr_validator is not None:
                        try:
                            validation = self._expr_validator.validate_full(expr)
                            if not validation.passed:
                                errors = [c.details for c in validation.checks.values() if not c.passed]
                                logger.warning("[ORCH] [EXPR-VAL] 表达式验证未通过: %s", "; ".join(errors[:3]))
                                approved = False
                            else:
                                logger.info(
                                    "[ORCH] [EXPR-VAL] 表达式验证通过 (complexity=%.1f, tags=%s)",
                                    validation.complexity_score,
                                    ", ".join(validation.semantic_tags[:3]),
                                )
                        except (ValueError, TypeError, SyntaxError) as val_exc:
                            logger.debug("[ORCH] [EXPR-VAL] 验证异常，跳过: %s", val_exc)

                    logger.info(
                        "[ORCH] [REASONING] gen=%d template=%s approved=%s expr=%s",
                        i + 1,
                        reasoning_result.phase1_reasoning.get("selected_template", "?"),
                        approved,
                        expr[:80],
                    )

                    alphas.append(
                        GeneratedAlpha(
                            expression=expr,
                            rationale=reasoning_result.phase1_reasoning.get("reasoning", ""),
                            strategy=focus_area,
                            confidence=0.9 if approved else 0.7,
                            raw_output=json.dumps(
                                {
                                    "phase1": reasoning_result.phase1_reasoning,
                                    "phase2": reasoning_result.phase2_mapping,
                                    "phase3": reasoning_result.phase3_expression,
                                    "phase4": reasoning_result.phase4_critique,
                                },
                                ensure_ascii=False,
                            ),
                            metadata={
                                "source": "template_reasoning",
                                "template_id": reasoning_result.phase1_reasoning.get("selected_template"),
                                "approved": approved,
                                "families_used": reasoning_result.phase2_mapping.get("cross_family_check", {}).get(
                                    "families_used", []
                                ),
                                "phase4_verdict": reasoning_result.phase4_critique.get("critique", {}).get(
                                    "overall_verdict", "UNKNOWN"
                                ),
                            },
                        )
                    )

                except (ValueError, TypeError, OSError, RuntimeError) as e:
                    logger.warning(
                        "[ORCH] Reasoning generator failed for gen=%d: %s, fallback to direct LLM",
                        i + 1,
                        e,
                    )
                    alpha = await self._generate_single_direct(focus_area)
                    alphas.append(alpha)
            else:
                alpha = await self._generate_single_direct(focus_area)
                alphas.append(alpha)

        logger.info(
            "[ORCH] Generated %d alphas | reasoning=%d direct=%d",
            len(alphas),
            sum(1 for a in alphas if a.metadata.get("source") == "template_reasoning"),
            sum(1 for a in alphas if a.metadata.get("source") == "direct_llm"),
        )

        return alphas

    def prefilter_batch(
        self,
        alphas: list[GeneratedAlpha],
    ) -> tuple[list[GeneratedAlpha], int]:
        """
        批量预筛选

        Args:
            alphas: 生成的因子列表

        Returns:
            (通过筛选的列表, 被过滤的数量)
        """
        passed = []
        filtered_count = 0

        for alpha in alphas:
            pf_result = self.prefilter.prefilter(alpha.expression)

            if pf_result.passed:
                passed.append(alpha)
                self.prefilter.record_submission(alpha.expression)
            else:
                filtered_count += 1
                logger.info(
                    "[ORCH] Prefilter rejected: %s (reason=%s conf=%.2f)",
                    alpha.expression[:60],
                    pf_result.reason,
                    pf_result.confidence_score,
                )

        return passed, filtered_count

    async def submit_to_slots(
        self,
        alphas: list[GeneratedAlpha],
    ) -> int:
        """
        批量提交到 SlotManager

        Args:
            alphas: 通过预筛选的因子列表

        Returns:
            成功提交的数量
        """
        submitted = 0

        for alpha in alphas:
            try:
                task_id = await self.slot_manager.submit_raw(
                    expression=alpha.expression,
                    name=f"orch_{self._cycle_num}_{submitted}",
                    strategy=alpha.strategy,
                )

                completion_event = asyncio.Event()
                self._pending_completions[task_id] = completion_event

                submitted += 1

                logger.debug(
                    "[ORCH] Submitted to SlotManager: task=%s expr=%.50s...",
                    task_id,
                    alpha.expression,
                )

            except asyncio.QueueFull:
                logger.warning("[ORCH] SlotManager queue full, skipping alpha")
            except (TimeoutError, aiohttp.ClientError, ConnectionError) as exc:
                logger.error("[ORCH] Submit failed: %s", exc)

        return submitted

    def set_mab(self, mab: Any) -> None:
        """
        设置 MAB 实例（用于更新奖励）

        Args:
            mab: HierarchicalMAB 或兼容的 bandit 实例
        """
        self._mab = mab
        logger.info("[ORCH] MAB instance set: %s", type(mab).__name__)

    def get_status(self) -> dict:
        """
        获取编排器当前状态

        Returns:
            包含运行状态和统计信息的字典
        """
        return {
            "running": self._running,
            "current_cycle": self._cycle_num,
            "stats": {
                "total_cycles": self.stats.total_cycles,
                "total_generated": self.stats.total_generated,
                "total_submitted": self.stats.total_submitted,
                "total_completed": self.stats.total_completed,
                "total_passed": self.stats.total_passed,
                "total_improved": self.stats.total_improved,
                "total_prefiltered": self.stats.total_prefiltered,
                "best_sharpe_ever": self.stats.best_sharpe_ever,
                "best_expression": self.stats.best_expression[:80] if self.stats.best_expression else "",
                "pass_rate": self.stats.pass_rate,
                "avg_sharpe": self.stats.avg_sharpe,
                "improvement_success_rate": self.stats.improvement_success_rate,
            },
            "pending_completions": len(self._pending_completions),
            "active_improvement_chains": len(self._improvement_chains),
            "config": {k: v for k, v in self.config.items() if k not in ("llm_generate_fn", "llm_improve_fn")},
        }

    def status_summary(self) -> str:
        """生成人类可读的状态摘要"""
        status = self.get_status()
        s = status["stats"]

        lines = [
            "=" * 60,
            "  🔄 FEEDBACK LOOP ORCHESTRATOR STATUS",
            "=" * 60,
            f"  Running     : {'✅ YES' if status['running'] else '❌ NO'}",
            f"  Current Cycle: {status['current_cycle']}",
            "",
            "─" * 40,
            "  📊 STATISTICS",
            "─" * 40,
            f"  Total Cycles       : {s['total_cycles']}",
            f"  Total Generated    : {s['total_generated']}",
            f"  Total Submitted    : {s['total_submitted']}",
            f"  Total Completed    : {s['total_completed']}",
            "",
            f"  ✅ Passed           : {s['total_passed']}",
            f"  🔧 Improved         : {s['total_improved']}",
            f"  🚫 Prefiltered     : {s['total_prefiltered']}",
            "",
            "─" * 40,
            "  📈 PERFORMANCE",
            "─" * 40,
            f"  Pass Rate          : {s['pass_rate']:.1%}",
            f"  Avg Sharpe         : {s['avg_sharpe']:.3f}",
            f"  Best Sharpe Ever   : {s['best_sharpe_ever']:.3f}",
            f"  Improvement Success: {s['improvement_success_rate']:.1%}",
            "",
            "─" * 40,
            "  ⏳ QUEUE STATUS",
            "─" * 40,
            f"  Pending Completions : {status['pending_completions']}",
            f"  Active Improvement Chains: {status['active_improvement_chains']}",
            "=" * 60,
        ]

        return "\n".join(lines)

    def _extract_direction(self, expression: str) -> str:
        """
        从表达式中推断策略方向（简化版）

        用于 MAB arm 选择
        """
        expr_lower = expression.lower()

        if any(op in expr_lower for op in ["ts_delta", "ts_regression"]):
            return "momentum"
        if any(op in expr_lower for op in ["ts_mean", "ts_decay_linear", "-"]):
            return "reversal"
        if any(op in expr_lower for op in ["ts_std_dev", "ts_av_diff", "ts_corr"]):
            return "volatility"
        if any(fld in expr_lower for fld in ["earnings", "sales", "revenue", "cap"]):
            return "value"
        return "unknown"

    def _infer_neutralization_level(self, expression: str) -> str:
        if not expression or not isinstance(expression, str):
            return "none"
        expr_lower = expression.lower()
        neutralize_count = expr_lower.count("group_neutralize")
        has_subindustry = "subindustry" in expr_lower
        if neutralize_count >= 3:
            return "triple"
        if neutralize_count == 2:
            return "double"
        if has_subindustry:
            return "subindustry"
        if neutralize_count == 1 or "sector" in expr_lower or "industry" in expr_lower:
            return "industry"
        return "none"

    def _detect_format_error(self, checks: list, expression: str) -> str | None:
        if not expression or not isinstance(expression, str):
            return "empty_expression"
        if not checks:
            return None

        failed_checks = [c for c in checks if c.get("result") == "FAIL"]

        for check in failed_checks:
            name = check.get("name", "").lower()
            value = str(check.get("value", "")).lower()

            if "lookback" in name and "required" in value:
                return "lookback_window_required"
            if "group" in name and ("neutralize" in name or "missing" in value):
                return "missing_group_neutralize"
            if "operator" in name and "invalid" in value:
                return "invalid_operator"
            if "field" in name and ("not found" in value or "invalid" in value):
                return "invalid_field"
            if "syntax" in name or "parse" in name:
                return "syntax_error"

        expr_lower = expression.lower()

        if "group_neutralize" not in expr_lower:
            return "missing_group_neutralize"

        import re

        if not re.search(r"ts_\w+\(\w+,\s*\d+\)", expression):
            has_any_window = re.search(r"\w+\([^)]*,\s*\d+\)", expression)
            if not has_any_window:
                return "missing_lookback_window"

        return None

    def _auto_repair_expression(self, expression: str, error_type: str) -> str | None:
        """
        自动修复表达式中的常见格式错误

        Args:
            expression: 原始表达式
            error_type: 错误类型 (由 _detect_format_error 返回)

        Returns:
            修复后的表达式，如果无法修复返回 None
        """
        import re

        if error_type == "lookback_window_required":
            match = re.search(r"(ts_\w+)\(([^)]+)\)", expression)
            if match and "," not in match.group(2):
                operator = match.group(1)
                inner = match.group(2)
                if operator in ["ts_zscore", "ts_std_dev", "ts_mean", "ts_delta"]:
                    repaired = expression.replace(
                        f"{operator}({inner})",
                        f"{operator}({inner}, 20)",
                    )
                    logger.debug("[ORCH] Auto-repair: added lookback window=20 to %s", operator)
                    return repaired

        elif error_type == "missing_group_neutralize":
            if "group_neutralize" not in expression.lower():
                repaired = f"group_neutralize({expression}, industry)"
                logger.debug("[ORCH] Auto-repair: wrapped with group_neutralize(..., industry)")
                return repaired

        elif error_type == "syntax_error":
            try:
                compile(expression, "<string>", "eval")
                return expression
            except SyntaxError:
                logger.debug("[ORCH] Auto-repair: cannot fix syntax error automatically")
                return None

        logger.debug("[ORCH] Auto-repair: no repair rule for error_type=%s", error_type)
        return None

    def _generate_repair_feedback(self, sharpe: float, checks: list, expression: str) -> str:
        """
        生成针对修复/重试场景的 targeted feedback (参考 QuantGPT MutationEngine)

        策略:
          - IC 为负 (Sharpe < 0) → 反转信号方向
          - 检查失败多 → 完全重写
          - 无非线性变换 → 注入 tanh/power

        Args:
            sharpe: 当前 Sharpe 值
            checks: WQ 检查列表
            expression: 原始表达式

        Returns:
            targeted feedback 字符串
        """
        strategies = []

        if sharpe < -0.2:
            strategies.append("⚠ IC 为负值，建议反转信号方向：在整个表达式前加负号 (-expr) 或反转 rank 逻辑")

        failed_checks = len([c for c in checks if c.get("result") == "FAIL"])
        if failed_checks >= 3:
            strategies.append("⚠ 多项检查失败，建议完全重写因子逻辑，换用不同的数据字段或算子族")

        expr_lower = expression.lower()
        nonlinear_operators = ["tanh", "power", "signed_power", "sigmoid"]
        has_nonlinear = any(op in expr_lower for op in nonlinear_operators)

        if not has_nonlinear:
            strategies.append(
                "💡 缺少非线性变换，建议在 rank() 外层包裹 tanh() 或 signed_power(expr, 0.5) 提升信号区分度"
            )

        if "ts_decay_linear" not in expr_lower:
            strategies.append("💡 缺少时间衰减，建议用 ts_decay_linear(expr, 5-10) 降低换手率")

        turnover = None
        for c in checks:
            if "turnover" in c.get("name", "").lower():
                turnover = c.get("value")
                break

        if turnover:
            try:
                turnover_val = float(turnover.replace("%", ""))
                if turnover_val > 50:
                    strategies.append(f"⚠ 换手率过高 ({turnover_val}%)，必须增加 ts_decay_linear 或增大窗口期")
            except (ValueError, AttributeError):
                pass

        base_feedback = (
            f"Sharpe={sharpe:.3f} 为负值，因子可能存在根本性问题。\n诊断策略 (参考 QuantGPT MutationEngine):\n"
        )

        for i, strategy in enumerate(strategies, 1):
            base_feedback += f"  {i}. {strategy}\n"

        base_feedback += "\n请基于以上诊断选择最合适的改进方案重写因子。"

        return base_feedback

    def _generate_mutation_based_repair_feedback(
        self,
        expression: str,
        wq_metrics: dict,
        checks: list,
    ) -> str:
        """Generate repair feedback using BrainAwareMutationEngine diagnosis.

        Args:
            expression: Original factor expression
            wq_metrics: WQ metrics dict
            checks: WQ checks list

        Returns:
            Targeted feedback string from mutation engine
        """
        try:
            diagnosis = self._mutation_engine.diagnose(expression, wq_metrics, checks)
            mutation_prompt = self._mutation_engine.generate_mutation_prompt(
                diagnosis,
                expression,
                inspiration_exprs=self._get_inspiration_exprs(),
            )

            logger.info(
                "[ORCH] 🧬 MUTATION-ENGINE REPAIR DIAGNOSIS | strategy=%s reason=%s",
                diagnosis.strategy.value,
                diagnosis.reason,
            )

            header = (
                f"Sharpe={wq_metrics.get('sharpe', 0):.3f} 为负值，因子可能存在根本性问题。\n"
                f"MutationEngine 诊断策略: {diagnosis.strategy.value}\n"
                f"诊断原因: {diagnosis.reason}\n\n"
            )

            return header + mutation_prompt

        except (ValueError, TypeError, SyntaxError) as exc:
            logger.warning(
                "[ORCH] MutationEngine diagnosis failed, fallback to legacy: %s",
                exc,
            )
            return self._generate_repair_feedback(
                wq_metrics.get("sharpe", 0),
                checks,
                expression,
            )

    def _get_inspiration_exprs(self) -> list[str]:
        """Get list of inspiration expressions for mutation prompt generation.

        Returns:
            List of example factor expressions (empty if none available)
        """
        try:
            if hasattr(self, "_lib") and self._lib is not None:
                templates = getattr(self._lib, "_three_block_templates", [])
                if templates:
                    import random

                    return random.sample(
                        templates,
                        min(3, len(templates)),
                    )
        except (OSError, ValueError, RuntimeError) as exc:
            logger.debug("[ORCH] Failed to get inspiration exprs: %s", exc)

        return []

    def _update_global_stats(self) -> None:
        """更新全局统计数据"""
        completed = self.stats.total_completed
        if completed > 0:
            self.stats.pass_rate = self.stats.total_passed / completed

        if self.stats.total_passed > 0:
            self.stats.avg_sharpe = self.stats.best_sharpe_ever / max(1, self.stats.total_passed)

        attempts = self.stats.total_improvement_attempts
        if attempts > 0:
            self.stats.improvement_success_rate = self.stats.successful_improvements / attempts

    def _build_experience_context(self, experiences: list[dict]) -> str:
        """将查询到的经验格式化为可读的文本上下文。

        Args:
            experiences: 从 GraphBasedExperienceDB 查询到的相似经验列表

        Returns:
            格式化的经验上下文字符串
        """
        context_parts = [f"## 📚 历史类似因子的改进经验 (Top-{len(experiences)})\n"]

        for idx, exp in enumerate(experiences, 1):
            similarity = exp.get("similarity", 0)
            expression = exp.get("expression", "N/A")
            wq_feedback = exp.get("wq_feedback", {})
            improvement_result = exp.get("improvement_result", {})

            original_sharpe = wq_feedback.get("sharpe", "N/A")
            original_fitness = wq_feedback.get("fitness", "N/A")
            original_turnover = wq_feedback.get("turnover", "N/A")

            strategy = improvement_result.get("strategy", "unknown")
            new_expression = improvement_result.get("new_expression", "N/A")
            success = improvement_result.get("success", False)
            status_icon = "✅" if success else "❌"

            experience_text = f"""### 经验 #{idx} (相似度: {similarity:.2f})
- **表达式**: {expression[:100]}{"..." if len(expression) > 100 else ""}
- **原始指标**: Sharpe={original_sharpe}, Fitness={original_fitness}, Turnover={original_turnover}
- **改进策略**: {strategy}
- **改进后表达式**: {str(new_expression)[:80]}{"..." if new_expression and len(str(new_expression)) > 80 else ""}
- **改进结果**: {status_icon} {"PASS" if success else "FAIL"}"""

            context_parts.append(experience_text)

        context_parts.append("\n**请参考以上历史经验,选择最可能成功的改进策略。**")
        return "\n\n".join(context_parts)

    def _enhance_improvement_prompt(
        self,
        original_prompt: str,
        experience_context: str,
        current_analysis: dict,
    ) -> str:
        """将经验上下文注入到原始 prompt 中。

        Args:
            original_prompt: 原始的 LLM 改进 prompt
            experience_context: 从 _build_experience_context 生成的经验文本
            current_analysis: 当前因子的分析数据 {sharpe, fitness, category, turnover}

        Returns:
            增强后的 prompt 字符串
        """
        enhanced = f"""{original_prompt}

---

## 🎯 历史经验参考 (来自知识图谱)

{experience_context}

### 当前因子分析
- **Sharpe Ratio**: {current_analysis.get("sharpe", "N/A")}
- **Fitness**: {current_analysis.get("fitness", "N/A")}
- **Category**: {current_analysis.get("category", "N/A")}
- **Turnover**: {current_analysis.get("turnover", "N/A")}

**改进建议**:
基于以上历史经验,如果存在相似的改进成功案例,优先采用相同或类似的策略。
避免重复历史上失败的改进方向。
"""

        return enhanced

    async def _call_llm_with_context(
        self,
        expression: str,
        wq_feedback: dict,
        experience_context: str,
        current_analysis: dict,
        improvement_gen: int = 1,
        original_task_id: str = "",
    ) -> str | None:
        """使用增强 prompt 调用 LLM 进行改进 (包含历史经验)。

        Args:
            expression: 原始因子表达式
            wq_feedback: WQ 反馈数据
            experience_context: 经验上下文文本
            current_analysis: 当前分析数据
            improvement_gen: 改进代数
            original_task_id: 原任务 ID

        Returns:
            改进后的表达式字符串，失败返回 None
        """
        sharpe = wq_feedback.get("sharpe", 0)
        turnover = wq_feedback.get("turnover")
        checks = wq_feedback.get("checks", [])

        check_str = (
            "; ".join([f"{c.get('name', '?')}={c.get('value', '?')}" for c in checks if c.get("result") == "FAIL"])
            if checks
            else "none"
        )

        base_prompt = f"""You are a quantitative alpha factor engineer with access to historical improvement experiences.  # noqa: E501
The following alpha was submitted to WorldQuant BRAIN but needs improvement.

ORIGINAL EXPRESSION:
{expression}

BRAIN FEEDBACK:
- Sharpe Ratio: {sharpe:.3f} (need >= 1.25)
- Turnover: {turnover}% (ideal: 10-30%)
- Failed Checks: {check_str}

IMPROVEMENT GUIDELINES:
1. If Sharpe too low: add volatility normalization (ts_zscore, ts_std_dev), tighten neutralization
2. If Turnover too high: wrap with ts_decay_linear(expr, window) where window in [5, 10, 15]
3. Keep group_neutralize(...) as outer wrapper
4. Change operator family if needed (rank→ts_zscore, ts_delta→ts_regression)
5. Use different lookback windows (avoid trivial changes)

Return ONLY the improved FastExpr expression (single line, no JSON, no explanation).
Example: group_neutralize(ts_decay_linear(rank(ts_zscore(close, 20)), 10), industry)"""

        enhanced_prompt = self._enhance_improvement_prompt(
            original_prompt=base_prompt,
            experience_context=experience_context,
            current_analysis=current_analysis,
        )

        logger.info(
            "[EXPERIENCE] 🚀 Calling LLM with enhanced prompt | task=%s gen=%d expr_len=%d",
            original_task_id,
            improvement_gen,
            len(enhanced_prompt),
        )

        try:
            raw_response = await self.config["llm_improve_fn"](
                system_prompt="You are a quantitative alpha factor improver with historical experience. Output ONLY the improved expression.",  # noqa: E501
                history=[],
                user_msg=enhanced_prompt,
                session_id=f"orch_improve_exp_{original_task_id}",
                cycle=improvement_gen,
            )

            improved = raw_response.strip()

            if improved.startswith("```"):
                improved = improved.split("\n")[1]
            if improved.endswith("```"):
                improved = improved.rsplit("```", 1)[0].strip()

            if improved and improved != expression and len(improved) > 10:
                logger.info(
                    "[EXPERIENCE] ✅ LLM with context returned valid expression | expr=%.60s...",
                    improved,
                )
                return improved

            logger.warning(
                "[EXPERIENCE] ⚠️ LLM with context returned invalid/empty expression",
            )
            return None

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[EXPERIENCE] ❌ LLM call with context failed: %s",
                exc,
            )
            return None

    def _categorize_result(self, sharpe: float, fitness: float) -> str:
        """根据结果分类用于图谱存储。

        Args:
            sharpe: Sharpe 比率
            fitness: Fitness 指标

        Returns:
            分类字符串: success / near_pass / weak / noise
        """
        if sharpe >= 1.25 and fitness >= 1.0:
            return "success"
        if sharpe >= 0.8:
            return "near_pass"
        if sharpe >= 0:
            return "weak"
        return "noise"

    def _build_reflection_context(
        self,
        reflection_result: Any,
        critique_result: Any,
        recent_reflections: list[dict],
        failure_patterns: dict[str, int],
        is_specialized: bool = False,
    ) -> str:
        """构建反思上下文用于 LLM 改进 prompt。

        Args:
            reflection_result: ReflectionEngine.reflect_on_failure() 的返回值
            critique_result: ReflectionEngine.self_critique() 的返回值
            recent_reflections: 最近的历史反思记录
            failure_patterns: 失败模式统计
            is_specialized: 是否为 HIGH_ICIR_LOW_FITNESS 专门分析

        Returns:
            格式化的反思上下文字符串
        """
        context_parts = ["## 🤔 ReflectionEngine 智能分析结果\n"]

        if is_specialized:
            context_parts.append("### ⚠️ HIGH_ICIR_LOW_FITNESS 专门诊断\n")
            context_parts.append("**因子特征**: 信号强度良好但执行成本过高\n")
        else:
            context_parts.append("### 📋 失败原因分析\n")

        context_parts.append(f"- **失败阶段**: {reflection_result.failure_stage}")
        context_parts.append(f"- **失败原因**: {reflection_result.failure_reason}")
        context_parts.append(f"- **建议修复**: {reflection_result.suggested_fix}")
        context_parts.append(f"- **置信度**: {reflection_result.confidence:.2f}")

        if reflection_result.avoid_patterns:
            context_parts.append(f"- **避免模式**: {', '.join(reflection_result.avoid_patterns)}")

        if critique_result and critique_result.critique_available:
            context_parts.append("\n### 🔍 自我批评结果\n")
            context_parts.append(f"- **一致性评分**: {critique_result.consistency_score or 'N/A'}")
            if critique_result.issues:
                context_parts.append(f"- **发现的问题** ({len(critique_result.issues)}):")
                for issue in critique_result.issues[:5]:
                    context_parts.append(f"  - {issue}")
            if critique_result.suggestions:
                context_parts.append(f"- **改进建议** ({len(critique_result.suggestions)}):")
                for suggestion in critique_result.suggestions[:5]:
                    context_parts.append(f"  - 💡 {suggestion}")

        if recent_reflections:
            context_parts.append(f"\n### 📚 历史类似案例 (最近 {len(recent_reflections)} 条)\n")
            for idx, refl in enumerate(recent_reflections[:3], 1):
                stage = refl.get("failure_stage", "unknown")
                reason = refl.get("failure_reason", "")[:60]
                fix = refl.get("suggested_fix", "")[:60]
                context_parts.append(
                    f"{idx}. [{stage}] {reason}... → 建议: {fix}...",
                )

        if failure_patterns:
            context_parts.append("\n### 📊 失败模式统计\n")
            sorted_patterns = sorted(failure_patterns.items(), key=lambda x: x[1], reverse=True)
            for pattern, count in sorted_patterns[:5]:
                context_parts.append(f"- {pattern}: {count} 次")

        context_parts.append("\n**请基于以上深度分析,生成更有针对性的改进表达式。**\n")

        return "\n".join(context_parts)

    async def _call_llm_with_reflection_context(
        self,
        expression: str,
        wq_feedback: dict,
        reflection_context: str,
        improvement_gen: int = 1,
        original_task_id: str = "",
    ) -> str | None:
        """使用反思上下文调用 LLM 进行改进。

        Args:
            expression: 原始因子表达式
            wq_feedback: WQ 反馈数据
            reflection_context: 从 _build_reflection_context 生成的反思文本
            improvement_gen: 改进代数
            original_task_id: 原任务 ID

        Returns:
            改进后的表达式字符串，失败返回 None
        """
        sharpe = wq_feedback.get("sharpe", 0)
        turnover = wq_feedback.get("turnover")
        checks = wq_feedback.get("checks", [])

        check_str = (
            "; ".join([f"{c.get('name', '?')}={c.get('value', '?')}" for c in checks if c.get("result") == "FAIL"])
            if checks
            else "none"
        )

        base_prompt = f"""You are a senior quantitative alpha factor engineer with deep self-reflection capabilities.
The following alpha was submitted to WorldQuant BRAIN and analyzed by an intelligent reflection engine.

ORIGINAL EXPRESSION:
{expression}

BRAIN FEEDBACK:
- Sharpe Ratio: {sharpe:.3f} (need >= 1.25)
- Turnover: {turnover}% (ideal: 10-30%)
- Failed Checks: {check_str}

{reflection_context}

IMPROVEMENT GUIDELINES (based on reflection):
1. Prioritize fixes suggested by the reflection engine (high confidence suggestions)
2. Avoid patterns that historically led to failures
3. Apply targeted fixes based on the identified failure stage
4. If turnover is the issue: use ts_decay_linear(window=10-20)
5. If signal direction is wrong: negate the expression or reverse logic
6. Keep group_neutralize(...) as outer wrapper

Return ONLY the improved FastExpr expression (single line, no JSON, no explanation).
Example: group_neutralize(ts_decay_linear(rank(ts_zscore(close, 20)), 10), industry)"""

        logger.info(
            "[REFLECTION] 🚀 Calling LLM with reflection-enhanced prompt | task=%s gen=%d prompt_len=%d",
            original_task_id,
            improvement_gen,
            len(base_prompt),
        )

        try:
            raw_response = await self.config["llm_improve_fn"](
                system_prompt="You are a senior quantitative alpha improver with self-reflection abilities. Output ONLY the improved expression.",  # noqa: E501
                history=[],
                user_msg=base_prompt,
                session_id=f"orch_improve_refl_{original_task_id}",
                cycle=improvement_gen,
            )

            improved = raw_response.strip()

            if improved.startswith("```"):
                improved = improved.split("\n")[1]
            if improved.endswith("```"):
                improved = improved.rsplit("```", 1)[0].strip()

            if improved and improved != expression and len(improved) > 10:
                logger.info(
                    "[REFLECTION] ✅ LLM with reflection returned valid expression | expr=%.60s...",
                    improved,
                )
                return improved

            logger.warning(
                "[REFLECTION] ⚠️ LLM with reflection returned invalid/empty expression",
            )
            return None

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[REFLECTION] ❌ LLM call with reflection failed: %s",
                exc,
            )
            return None

    def _record_reflection_outcome(
        self,
        expression: str,
        success: bool,
        new_expression: str | None,
        strategy: str,
    ) -> None:
        """记录反思改进的结果（成功或失败）。

        当改进成功时（新 fitness > 旧 fitness），调用此方法记录成功经验。
        当改进失败时，调用此方法记录失败模式。

        Args:
            expression: 原始因子表达式
            success: 改进是否成功
            new_expression: 改进后的表达式（可为 None）
            strategy: 使用的策略名称
        """
        if self._graph_db is not None:
            try:
                if success:
                    logger.info(
                        "[REFLECTION] ✅ Recording SUCCESS | expr=%.50s... strategy=%s",
                        expression,
                        strategy,
                    )
                else:
                    logger.warning(
                        "[REFLECTION] ❌ Recording FAILURE | expr=%.50s... strategy=%s",
                        expression,
                        strategy,
                    )

            except (OSError, ValueError, RuntimeError) as exc:
                logger.warning("[REFLECTION] ⚠ Failed to prepare reflection outcome data: %s", exc)


async def create_orchestrator(
    cookies: Any,
    slot_manager: SlotManager,
    config: dict | None = None,
    auto_start: bool = True,
) -> FeedbackLoopOrchestrator:
    """
    工厂函数：创建并可选启动 FeedbackLoopOrchestrator

    Args:
        cookies: WQ 认证凭证
        slot_manager: 已初始化的 SlotManager
        config: 配置选项
        auto_start: 是否自动调用 start()

    Returns:
        已初始化（并可选启动）的 FeedbackLoopOrchestrator 实例
    """
    orchestrator = FeedbackLoopOrchestrator(
        cookies=cookies,
        slot_manager=slot_manager,
        config=config,
    )

    if auto_start:
        await orchestrator.start()

    return orchestrator
