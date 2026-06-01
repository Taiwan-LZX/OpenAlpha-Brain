"""
OpenAlpha-Brain PersistenceLayer — Layer 6 of 6-Layer Architecture
====================================================================
最终层：持久化、遥测与跨周期状态管理。

职责:
  1. CrossoverEngine 集成 — 记录 alpha 结果、轨迹、谱系追踪
  2. AlphaChannel 管理 — 将高质量 alpha 提交到持久化通道
  3. 遥测/统计 — 算法性能指标、利用率追踪
  4. 事件发射 — 发布事件供监控/可观测性使用

架构位置 (Layer 6):
  ┌──────────────────────┐     ┌─────────────────────┐
  │ RobustnessValidator   │ →   │  PersistenceLayer   │ ← 本模块
  │ (Layer 5)            │     │  (Layer 6 - Final)  │
  └──────────────────────┘     └───────┬─────────────┘
                                      │
                              ┌───────┴───────────┐
                              ▼                   ▼
                       CrossoverEngine      AlphaChannel
                       (lineage tracking)   (high-quality pool)
                              ▼
                        AlphaEventBus (telemetry events)

提取来源 (loop_engine.py):
  - L1834-1880: CrossoverEngine.record_alpha_outcome() + record_trajectory()
  - L1952-1981: AlphaChannel.submit() + batch processing
  - L1820-1832: AlphaEventBus.emit(EVENT_BRAIN_RESULT)

Usage:
    layer = PersistenceLayer(config={"min_sharpe_for_channel": 1.0})
    result = await layer.persist(
        expression="ts_decay_linear(rank(close/volume), sector), 10)",
        brain_result=brain_result,
        exploration_result=exploration_result,
        session_id="sess_001",
        cycle_num=42,
        crossover_engine=crossover_engine,
        alpha_channel=alpha_channel,
        event_bus=event_bus,
    )
    print(result.submitted_to_channel, result.recorded_in_crossover)
    stats = layer.get_utilization_stats()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_EPSILON = 1e-6


@dataclass
class PersistenceResult:
    """单次 persist() 调用的完整结果。

    Attributes:
        alpha_id: BRAIN 平台返回的 alpha ID。
        location_id: AlphaChannel 路由目标 ID。
        submitted_to_channel: 是否成功提交到 AlphaChannel。
        recorded_in_crossover: 是否记录到 CrossoverEngine。
        events_emitted: 本次发射的事件类型列表。
        metrics_recorded: 记录的指标字典。
        error: 错误信息（空字符串表示成功）。
    """

    alpha_id: str | None = None
    location_id: str | None = None
    submitted_to_channel: bool = False
    recorded_in_crossover: bool = False
    events_emitted: list[str] = field(default_factory=list)
    metrics_recorded: dict = field(default_factory=dict)
    error: str = ""


class PersistenceLayer:
    """Layer 6: 持久化与遥测层。

    封装所有跨周期状态管理逻辑，是 6 层架构的最终汇聚点。
    所有上游层（L1-L5）的结果在此处被记录、归档和发布为事件。

    设计原则:
    - 每个外部调用独立 try/except，单个失败不影响其他操作
    - 遵循最小惊讶原则：返回值完整描述所有操作的成败状态
    - 统计计数器线程安全（单线程 asyncio 上下文）
    """

    def __init__(self, config: dict | None = None):
        self._config = config or {}
        self._stats: dict[str, Any] = {
            "l1_calls": 0,
            "l2_calls": 0,
            "l3_calls": 0,
            "l4_calls": 0,
            "l5_calls": 0,
            "total_submitted": 0,
            "total_accepted": 0,
            "channel_successes": 0,
            "channel_failures": 0,
            "crossover_records": 0,
            "crossover_trajectories": 0,
            "events_emitted_total": 0,
            "errors_total": 0,
            "persist_calls": 0,
        }
        self._min_sharpe_for_channel: float = self._config.get("min_sharpe_for_channel", 0.75)
        self._min_sharpe_for_crossover: float = self._config.get("min_sharpe_for_crossover", 0.0)
        self._enable_telemetry: bool = self._config.get("enable_telemetry", True)

    async def persist(
        self,
        expression: str,
        brain_result: Any,
        exploration_result: Any = None,
        generation_result: Any = None,
        improvement_result: Any = None,
        robustness_result: Any = None,
        session_id: str = "",
        cycle_num: int = 0,
        crossover_engine: Any = None,
        alpha_channel: Any = None,
        event_bus: Any = None,
        min_sharpe_for_channel: float = 0.75,
        min_sharpe_for_crossover: float = 0.0,
        enable_telemetry: bool = True,
    ) -> PersistenceResult:
        """执行完整的持久化流水线。

        按顺序执行四个独立的持久化子任务，每个子任务有独立的错误边界：
        1. CrossoverEngine 结果记录 + 轨迹记录
        2. AlphaChannel 高质量 alpha 提交
        3. 遥测指标收集
        4. 事件总线发射

        Args:
            expression: Alpha 表达式字符串。
            brain_result: Layer 3 EvaluationGateway 返回的评估结果。
            exploration_result: Layer 1 ExplorationDirector 的探索结果。
            generation_result: Layer 2 GenerationPipeline 的生成结果。
            improvement_result: Layer 4 ImprovementOrchestra 的改进结果。
            robustness_result: Layer 5 RobustnessValidator 的鲁棒性结果。
            session_id: 当前会话 ID。
            cycle_num: 当前周期编号。
            crossover_engine: CrossoverMutationEngine 实例（可选）。
            alpha_channel: AlphaChannel 实例（可选）。
            event_bus: AlphaEventBus 实例（可选）。
            min_sharpe_for_channel: 提交到 Channel 的最低 Sharpe 阈值。
            min_sharpe_for_crossover: 记录到 Crossover 的最低 Sharpe 阈值（默认 0 = 全部记录）。
            enable_telemetry: 是否启用遥测收集。

        Returns:
            PersistenceResult 包含所有子任务的执行状态。
        """
        t_start = time.monotonic()
        result = PersistenceResult()
        self._stats["persist_calls"] += 1

        effective_min_channel = (
            min_sharpe_for_channel if min_sharpe_for_channel != 0.75 else self._min_sharpe_for_channel
        )
        effective_min_crossover = (
            min_sharpe_for_crossover if min_sharpe_for_crossover != 0.0 else self._min_sharpe_for_crossover
        )
        telemetry_on = enable_telemetry if enable_telemetry is not True else self._enable_telemetry

        sharpe = self._extract_sharpe(brain_result)
        result.alpha_id = self._extract_alpha_id(brain_result)

        accepted = self._extract_accepted(brain_result)
        if accepted:
            self._stats["total_accepted"] += 1

        direction = self._extract_direction(exploration_result, brain_result)

        _record_crossover_errors: list[str] = []
        if crossover_engine is not None:
            try:
                if sharpe >= effective_min_crossover:
                    reject_reason = self._extract_reject_reason(brain_result)
                    crossover_engine.record_alpha_outcome(
                        direction=direction,
                        sharpe=sharpe,
                        turnover=self._extract_turnover(brain_result),
                        complexity=len(expression or ""),
                        accepted=accepted,
                        reject_reason=reject_reason,
                    )
                    result.recorded_in_crossover = True
                    self._stats["crossover_records"] += 1

                    if accepted:
                        trajectory = self._build_trajectory(
                            expression,
                            direction,
                            brain_result,
                            exploration_result,
                        )
                        if trajectory is not None:
                            crossover_engine.record_trajectory(
                                trajectory,
                                {"id": result.alpha_id or "", "expression": expression, "direction": direction},
                            )
                            self._stats["crossover_trajectories"] += 1
            except (OSError, ValueError, RuntimeError, TypeError) as exc:
                _record_crossover_errors.append(f"crossover_record:{exc}")
                logger.debug("[%s] Persist: crossover record failed: %s", session_id, exc)
            except Exception as exc:
                _record_crossover_errors.append(f"crossover_record:{exc}")
                logger.warning("[%s] Persist: unexpected crossover error: %s", session_id, exc)

        _channel_errors: list[str] = []
        if alpha_channel is not None and sharpe >= effective_min_channel:
            try:
                route = await alpha_channel.submit(
                    alpha_id=result.alpha_id or "",
                    sharpe=sharpe,
                    expression=expression or "",
                    direction=direction,
                )
                result.location_id = route
                result.submitted_to_channel = True
                self._stats["total_submitted"] += 1
                self._stats["channel_successes"] += 1
            except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
                _channel_errors.append(f"channel_submit:{exc}")
                self._stats["channel_failures"] += 1
                logger.debug("[%s] Persist: channel submit failed: %s", session_id, exc)
            except Exception as exc:
                _channel_errors.append(f"channel_submit:{exc}")
                self._stats["channel_failures"] += 1
                logger.warning("[%s] Persist: unexpected channel error: %s", session_id, exc)

        _event_errors: list[str] = []
        if event_bus is not None:
            try:
                event_data = {
                    "session_id": session_id,
                    "cycle": cycle_num,
                    "alpha_id": result.alpha_id,
                    "expression": (expression or "")[:120],
                    "sharpe": sharpe,
                    "accepted": accepted,
                    "submitted_to_channel": result.submitted_to_channel,
                    "recorded_in_crossover": result.recorded_in_crossover,
                    "direction": direction,
                }
                from openalpha_brain.core.events import EVENT_BRAIN_RESULT

                event_bus.emit(EVENT_BRAIN_RESULT, event_data)
                result.events_emitted.append(EVENT_BRAIN_RESULT)
                self._stats["events_emitted_total"] += 1

                if result.submitted_to_channel:
                    from openalpha_brain.core.events import EVENT_ALPHA_VALIDATED

                    event_bus.emit(EVENT_ALPHA_VALIDATED, {**event_data, "location_id": result.location_id})
                    result.events_emitted.append(EVENT_ALPHA_VALIDATED)
                    self._stats["events_emitted_total"] += 1
            except (OSError, ValueError, RuntimeError) as exc:
                _event_errors.append(f"event_emit:{exc}")
                logger.debug("[%s] Persist: event emit failed: %s", session_id, exc)
            except Exception as exc:
                _event_errors.append(f"event_emit:{exc}")
                logger.warning("[%s] Persist: unexpected event error: %s", session_id, exc)

        if telemetry_on:
            result.metrics_recorded = {
                "sharpe": sharpe,
                "accepted": accepted,
                "direction": direction,
                "expression_length": len(expression or ""),
                "cycle_num": cycle_num,
                "duration_ms": round((time.monotonic() - t_start) * 1000, 2),
                "has_exploration_result": exploration_result is not None,
                "has_generation_result": generation_result is not None,
                "has_improvement_result": improvement_result is not None,
                "has_robustness_result": robustness_result is not None,
            }

        all_errors = _record_crossover_errors + _channel_errors + _event_errors
        if all_errors:
            result.error = "; ".join(all_errors)
            self._stats["errors_total"] += len(all_errors)

        if exploration_result is not None:
            self._stats["l1_calls"] += 1
        if generation_result is not None:
            self._stats["l2_calls"] += 1
        if brain_result is not None:
            self._stats["l3_calls"] += 1
        if improvement_result is not None:
            self._stats["l4_calls"] += 1
        if robustness_result is not None:
            self._stats["l5_calls"] += 1

        logger.info(
            "[%s] Persist-L6: cycle=%d sharpe=%.3f accepted=%s channel=%s crossover=%s events=%d errs=%d",
            session_id,
            cycle_num,
            sharpe,
            accepted,
            result.submitted_to_channel,
            result.recorded_in_crossover,
            len(result.events_emitted),
            len(all_errors),
        )

        return result

    def get_utilization_stats(self) -> dict:
        """返回算法利用率统计信息供监控使用。

        Returns:
            包含各层调用次数、提交/接受统计、成功率等指标的字典。
        """
        total_submitted = self._stats.get("total_submitted", 0)
        total_accepted = self._stats.get("total_accepted", 0)
        channel_successes = self._stats.get("channel_successes", 0)
        channel_failures = self._stats.get("channel_failures", 0)
        channel_attempts = channel_successes + channel_failures

        return {
            "layer_1_calls": self._stats.get("l1_calls", 0),
            "layer_2_calls": self._stats.get("l2_calls", 0),
            "layer_3_calls": self._stats.get("l3_calls", 0),
            "layer_4_calls": self._stats.get("l4_calls", 0),
            "layer_5_calls": self._stats.get("l5_calls", 0),
            "total_alphas_submitted": total_submitted,
            "total_alphas_accepted": total_accepted,
            "channel_success_rate": round(channel_successes / max(channel_attempts, 1), 4),
            "channel_attempts": channel_attempts,
            "crossover_records": self._stats.get("crossover_records", 0),
            "crossover_trajectories": self._stats.get("crossover_trajectories", 0),
            "events_emitted_total": self._stats.get("events_emitted_total", 0),
            "errors_total": self._stats.get("errors_total", 0),
            "persist_calls": self._stats.get("persist_calls", 0),
            "acceptance_rate": round(total_accepted / max(self._stats.get("persist_calls", 1), 1), 4),
        }

    def reset_stats(self) -> None:
        """重置所有统计计数器。"""
        for key in list(self._stats.keys()):
            self._stats[key] = 0

    @staticmethod
    def _extract_sharpe(brain_result: Any) -> float:
        if brain_result is None:
            return 0.0
        if hasattr(brain_result, "real_sharpe") and brain_result.real_sharpe is not None:
            return float(brain_result.real_sharpe)
        if hasattr(brain_result, "sharpe") and brain_result.sharpe is not None:
            return float(brain_result.sharpe)
        if isinstance(brain_result, dict):
            return float(brain_result.get("real_sharpe") or brain_result.get("sharpe") or 0.0)
        return 0.0

    @staticmethod
    def _extract_alpha_id(brain_result: Any) -> str | None:
        if brain_result is None:
            return None
        if hasattr(brain_result, "alpha_id"):
            return brain_result.alpha_id
        if isinstance(brain_result, dict):
            return brain_result.get("alpha_id")
        return None

    @staticmethod
    def _extract_accepted(brain_result: Any) -> bool:
        if brain_result is None:
            return False
        if hasattr(brain_result, "status"):
            from openalpha_brain.core.models import BrainSimStatus

            return brain_result.status == BrainSimStatus.PASS
        if isinstance(brain_result, dict):
            status = brain_result.get("status")
            if hasattr(status, "value"):
                from openalpha_brain.core.models import BrainSimStatus

                return status == BrainSimStatus.PASS
            return str(status) == "PASS"
        return False

    @staticmethod
    def _extract_turnover(brain_result: Any) -> float | None:
        if brain_result is None:
            return None
        if hasattr(brain_result, "real_turnover") and brain_result.real_turnover is not None:
            return float(brain_result.real_turnover)
        if hasattr(brain_result, "turnover") and brain_result.turnover is not None:
            return float(brain_result.turnover)
        if isinstance(brain_result, dict):
            return float(brain_result.get("real_turnover") or brain_result.get("turnover") or 0.0)
        return None

    @staticmethod
    def _extract_reject_reason(brain_result: Any) -> str | None:
        if brain_result is None:
            return None
        if hasattr(brain_result, "gate_failures") and brain_result.gate_failures:
            return "; ".join(brain_result.gate_failures[:3])
        if hasattr(brain_result, "status"):
            from openalpha_brain.core.models import BrainSimStatus

            if brain_result.status == BrainSimStatus.ERROR:
                return "BRAIN_ERROR"
            if hasattr(brain_result, "status") and brain_result.status != BrainSimStatus.PASS:
                return "BRAIN_FAIL"
        return None

    @staticmethod
    def _extract_direction(exploration_result: Any, brain_result: Any) -> str:
        if exploration_result is not None and hasattr(exploration_result, "direction"):
            return exploration_result.direction
        if exploration_result is not None and isinstance(exploration_result, dict):
            return exploration_result.get("direction", "momentum")
        if brain_result is not None and hasattr(brain_result, "exploration_direction"):
            return brain_result.exploration_direction or "momentum"
        return "momentum"

    @staticmethod
    def _build_trajectory(
        expression: str,
        direction: str,
        brain_result: Any,
        exploration_result: Any,
    ) -> Any:
        try:
            from openalpha_brain.evolution.evolution_types import AlphaTrajectory

            traj = AlphaTrajectory(
                hypothesis_direction=direction,
                hypothesis_mechanism="",
                expression_versions=[expression] if expression else [],
                final_status="PASS",
                final_sharpe=PersistenceLayer._extract_sharpe(brain_result),
            )
            feedback_data: dict[str, Any] = {}
            if brain_result is not None:
                if hasattr(brain_result, "status"):
                    feedback_data["status"] = (
                        brain_result.status.value if hasattr(brain_result.status, "value") else str(brain_result.status)
                    )
                if hasattr(brain_result, "real_sharpe"):
                    feedback_data["sharpe"] = brain_result.real_sharpe
                if hasattr(brain_result, "real_fitness"):
                    feedback_data["fitness"] = brain_result.real_fitness
                if hasattr(brain_result, "real_turnover"):
                    feedback_data["turnover"] = brain_result.real_turnover
            if feedback_data:
                traj.add_brain_feedback(feedback_data)
            if exploration_result is not None and hasattr(exploration_result, "hypothesis_mechanism"):
                traj.hypothesis_mechanism = exploration_result.hypothesis_mechanism
            return traj
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            logger.debug("Persist: build_trajectory failed: %s", exc)
            return None
