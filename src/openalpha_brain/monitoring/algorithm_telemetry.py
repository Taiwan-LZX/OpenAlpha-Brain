"""
Algorithm Telemetry System — Structured observability for 30+ algorithm modules.

Every important module emits structured telemetry events that are:
  1. Logged to console via [TELEMETRY] prefix
  2. Written to JSONL files for post-analysis
  3. Exposed via API for real-time dashboard consumption

Usage:
    collector = AlgorithmTelemetryCollector.get_instance()
    event_id = collector.record_enter("FitnessBoost", cycle_id="abc123", expr_id="xyz")
    collector.record_exit("FitnessBoost", event_id, metrics={"tier_used": 3})
    collector.record_decision("AdaptiveNeutralizer", "upgrade_to_subindustry", confidence=0.78)
    collector.record_error("ReflectionEngine", "LLM timeout", error_type="timeout")

Decorator shortcut:
    @telemetry_tracked("FitnessBoost")
    async def generate_boost_variants(self, expression, ...):
        ...
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_FLUSH_THRESHOLD = 50
_STUCK_ACTION_THRESHOLD = 5


@dataclass
class ModuleTelemetryEvent:
    timestamp: float
    module_name: str
    event_type: str
    cycle_id: str
    expression_id: str | None
    metrics: dict[str, Any]
    duration_ms: float | None
    metadata: dict[str, Any]


@dataclass
class ModuleHealthSignal:
    module_name: str
    status: str
    calls_this_cycle: int
    avg_duration_ms: float
    success_rate: float
    last_action: str
    data_quality_score: float
    warning_flags: list[str] = field(default_factory=list)


class AlgorithmTelemetryCollector:
    _instance: AlgorithmTelemetryCollector | None = None

    def __init__(self, telemetry_path: Path | None = None):
        self._events: list[ModuleTelemetryEvent] = []
        self._health_signals: dict[str, ModuleHealthSignal] = {}
        self._telemetry_path = telemetry_path
        self._current_cycle: str = ""
        self._lock = asyncio.Lock()
        self._event_count_since_flush: int = 0
        self._pending_events: list[ModuleTelemetryEvent] = []

    @classmethod
    def get_instance(cls) -> AlgorithmTelemetryCollector:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    def _log(self, message: str, level: int = logging.DEBUG) -> None:
        try:
            logger.log(level, "[TELEMETRY] %s", message)
        except (OSError, ValueError, RuntimeError):
            pass

    def _make_event_id(self) -> str:
        return uuid.uuid4().hex[:12]

    async def record_enter(
        self,
        module_name: str,
        cycle_id: str,
        expr_id: str | None = None,
        context: dict | None = None,
    ) -> str:
        try:
            event_id = self._make_event_id()
            event = ModuleTelemetryEvent(
                timestamp=time.time(),
                module_name=module_name,
                event_type="enter",
                cycle_id=cycle_id,
                expression_id=expr_id,
                metrics={},
                duration_ms=None,
                metadata={"event_id": event_id, **(context or {})},
            )
            async with self._lock:
                self._pending_events.append(event)
                self._event_count_since_flush += 1
            self._log(f"ENTER {module_name} cycle={cycle_id} id={event_id}")
            await self._auto_flush()
            return event_id
        except (OSError, ValueError, RuntimeError) as e:
            self._log(f"record_enter FAILED [{module_name}]: {e}", logging.WARNING)
            return self._make_event_id()

    async def record_exit(
        self,
        module_name: str,
        event_id: str,
        metrics: dict | None = None,
        duration_ms: float | None = None,
    ) -> None:
        try:
            cycle_id = self._current_cycle or "unknown"
            event = ModuleTelemetryEvent(
                timestamp=time.time(),
                module_name=module_name,
                event_type="exit",
                cycle_id=cycle_id,
                expression_id=None,
                metrics=metrics or {},
                duration_ms=duration_ms,
                metadata={"event_id": event_id},
            )
            async with self._lock:
                self._pending_events.append(event)
                self._event_count_since_flush += 1
            dur_str = f"{duration_ms:.1f}ms" if duration_ms is not None else "?ms"
            self._log(f"EXIT {module_name} id={event_id} {dur_str}")
            await self._auto_flush()
        except (OSError, ValueError, RuntimeError) as e:
            self._log(f"record_exit FAILED [{module_name}]: {e}", logging.WARNING)

    async def record_decision(
        self,
        module_name: str,
        action: str,
        confidence: float = 0.0,
        reason: str = "",
        **kwargs: Any,
    ) -> None:
        try:
            cycle_id = self._current_cycle or "unknown"
            event = ModuleTelemetryEvent(
                timestamp=time.time(),
                module_name=module_name,
                event_type="decision",
                cycle_id=cycle_id,
                expression_id=None,
                metrics={
                    "action": action,
                    "confidence": confidence,
                    "reason": reason[:200],
                    **kwargs,
                },
                duration_ms=None,
                metadata={},
            )
            async with self._lock:
                self._pending_events.append(event)
                self._event_count_since_flush += 1
            conf_str = f"conf={confidence:.2f}" if confidence > 0 else ""
            self._log(f"DECISION {module_name}: {action} {conf_str}")
            await self._auto_flush()
        except (OSError, ValueError, RuntimeError) as e:
            self._log(f"record_decision FAILED [{module_name}]: {e}", logging.WARNING)

    async def record_error(
        self,
        module_name: str,
        message: str,
        error_type: str = "unknown",
        **kwargs: Any,
    ) -> None:
        try:
            cycle_id = self._current_cycle or "unknown"
            event = ModuleTelemetryEvent(
                timestamp=time.time(),
                module_name=module_name,
                event_type="error",
                cycle_id=cycle_id,
                expression_id=None,
                metrics={
                    "message": message[:300],
                    "error_type": error_type,
                    **kwargs,
                },
                duration_ms=None,
                metadata={},
            )
            async with self._lock:
                self._pending_events.append(event)
                self._event_count_since_flush += 1
            self._log(f"ERROR {module_name} [{error_type}]: {message[:100]}", logging.WARNING)
            await self._auto_flush()
        except (OSError, ValueError, RuntimeError) as e:
            self._log(f"record_error FAILED [{module_name}]: {e}", logging.WARNING)

    async def record_data(
        self,
        module_name: str,
        data_label: str,
        value: Any,
        **kwargs: Any,
    ) -> None:
        try:
            cycle_id = self._current_cycle or "unknown"
            serialized = self._serialize_value(value)
            event = ModuleTelemetryEvent(
                timestamp=time.time(),
                module_name=module_name,
                event_type="data",
                cycle_id=cycle_id,
                expression_id=None,
                metrics={
                    "label": data_label,
                    "value": serialized,
                    **kwargs,
                },
                duration_ms=None,
                metadata={},
            )
            async with self._lock:
                self._pending_events.append(event)
                self._event_count_since_flush += 1
            val_str = str(serialized)[:80] if not isinstance(serialized, str) else serialized[:80]
            self._log(f"DATA {module_name}: {data_label}={val_str}")
            await self._auto_flush()
        except (OSError, ValueError, RuntimeError) as e:
            self._log(f"record_data FAILED [{module_name}]: {e}", logging.WARNING)

    async def update_health(self, signal: ModuleHealthSignal) -> None:
        try:
            self._health_signals[signal.module_name] = signal
            status_emoji = {
                "active": "\u25b6",
                "idle": "\u23f8",
                "degraded": "\u26a0\ufe0f",
                "error": "\u274c",
                "dead": "\u2606",
            }.get(signal.status, "?")
            self._log(
                f"HEALTH {status_emoji} {signal.module_name}: "
                f"calls={signal.calls_this_cycle} "
                f"avg={signal.avg_duration_ms:.1f}ms "
                f"success={signal.success_rate:.0%} "
                f"dq={signal.data_quality_score:.2f}"
            )
        except (OSError, ValueError, RuntimeError) as e:
            self._log(f"update_health FAILED: {e}", logging.WARNING)

    def set_current_cycle(self, cycle_id: str) -> None:
        self._current_cycle = cycle_id

    async def flush_to_disk(self) -> int:
        try:
            async with self._lock:
                if not self._pending_events:
                    return 0
                to_write = list(self._pending_events)
                self._pending_events.clear()
                self._event_count_since_flush = 0
                self._events.extend(to_write)

            path = self._resolve_path()
            if path is None:
                self._log("flush_to_disk: no telemetry_path configured")
                return len(to_write)

            path.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with open(path, "a", encoding="utf-8") as f:
                for ev in to_write:
                    try:
                        line = json.dumps(asdict(ev), ensure_ascii=False, default=str)
                        f.write(line + "\n")
                        written += 1
                    except (TypeError, ValueError):
                        continue
            self._log(f"FLUSH {written} events -> {path.name}")
            return written
        except (OSError, ValueError, RuntimeError) as e:
            self._log(f"flush_to_disk FAILED: {e}", logging.ERROR)
            return 0

    async def _auto_flush(self) -> None:
        if self._event_count_since_flush >= _FLUSH_THRESHOLD:
            await self.flush_to_disk()

    def _all_events(self) -> list[ModuleTelemetryEvent]:
        return self._events + self._pending_events

    def get_cycle_summary(self, cycle_id: str) -> dict:
        try:
            all_ev = [e for e in self._all_events() if e.cycle_id == cycle_id]
            if not all_ev:
                return {"cycle_id": cycle_id, "total_events": 0, "modules": {}}

            modules: dict[str, list[ModuleTelemetryEvent]] = {}
            for ev in all_ev:
                modules.setdefault(ev.module_name, []).append(ev)

            summary: dict[str, Any] = {
                "cycle_id": cycle_id,
                "total_events": len(all_ev),
                "time_range": {
                    "start": min(e.timestamp for e in all_ev),
                    "end": max(e.timestamp for e in all_ev),
                },
                "modules": {},
            }
            for mod_name, evts in modules.items():
                enters = [e for e in evts if e.event_type == "enter"]
                exits = [e for e in evts if e.event_type == "exit"]
                errors = [e for e in evts if e.event_type == "error"]
                decisions = [e for e in evts if e.event_type == "decision"]
                durations = [e.duration_ms for e in exits if e.duration_ms is not None]
                summary["modules"][mod_name] = {
                    "calls": len(enters),
                    "errors": len(errors),
                    "decisions": len(decisions),
                    "avg_duration_ms": (statistics.mean(durations) if durations else None),
                    "max_duration_ms": (max(durations) if durations else None),
                    "last_action": (decisions[-1].metrics.get("action", "") if decisions else ""),
                }
            return summary
        except (OSError, ValueError, RuntimeError) as e:
            self._log(f"get_cycle_summary FAILED: {e}", logging.WARNING)
            return {"cycle_id": cycle_id, "error": str(e)}

    def get_module_timeline(self, module_name: str) -> list[dict]:
        try:
            evts = sorted(
                [e for e in self._all_events() if e.module_name == module_name],
                key=lambda e: e.timestamp,
            )
            return [asdict(e) for e in evts]
        except (OSError, ValueError, RuntimeError) as e:
            self._log(f"get_module_timeline FAILED: {e}", logging.WARNING)
            return []

    def detect_anomalies(self) -> list[dict]:
        anomalies: list[dict] = []
        try:
            modules_seen: set[str] = {e.module_name for e in self._all_events()}
            known_modules = {
                "FitnessBoostEngine",
                "AdaptiveNeutralizer",
                "HybridFactorJudge",
                "StrategyClassifier",
                "ReflectionEngine",
                "HypothesisAligner",
                "NearPassImprover",
                "TurnoverOptimizer",
                "FeedbackOrchestrator",
                "BrainSubmitter",
            }

            dead_candidates = known_modules - modules_seen
            for mod in sorted(dead_candidates):
                anomalies.append(
                    {
                        "severity": "warning",
                        "type": "dead_module",
                        "module": mod,
                        "message": f"Module '{mod}' never emitted any telemetry — potential dead code or uninstrumented path",
                    }
                )

            per_module: dict[str, list[ModuleTelemetryEvent]] = {}
            for e in self._all_events():
                per_module.setdefault(e.module_name, []).append(e)

            for mod_name, evts in per_module.items():
                total = len(evts)
                errors = [e for e in evts if e.event_type == "error"]
                error_rate = len(errors) / total if total > 0 else 0

                if error_rate > 0.5 and total >= 3:
                    anomalies.append(
                        {
                            "severity": "critical",
                            "type": "high_error_rate",
                            "module": mod_name,
                            "message": f"Error rate {error_rate:.0%} ({len(errors)}/{total}) exceeds 50% threshold",
                            "details": {"error_count": len(errors), "total_calls": total},
                        }
                    )

                durations = []
                for e in evts:
                    if e.event_type == "exit" and e.duration_ms is not None:
                        durations.append(e.duration_ms)

                if len(durations) >= 3:
                    mean_d = statistics.mean(durations)
                    if len(durations) >= 2:
                        try:
                            stdev_d = statistics.stdev(durations)
                        except statistics.StatisticsError:
                            stdev_d = 0.0
                    else:
                        stdev_d = 0.0

                    if stdev_d > 0:
                        sigma_threshold = mean_d + 3 * stdev_d
                        outliers = [d for d in durations if d > sigma_threshold]
                        if outliers:
                            anomalies.append(
                                {
                                    "severity": "warning",
                                    "type": "execution_time_outlier",
                                    "module": mod_name,
                                    "message": (
                                        f"{len(outliers)} execution(s) exceeded 3\u03c3 "
                                        f"(mean={mean_d:.1f}ms, \u03c3={stdev_d:.1f}ms)"
                                    ),
                                    "details": {
                                        "mean_ms": round(mean_d, 1),
                                        "stdev_ms": round(stdev_d, 1),
                                        "outlier_values": [round(d, 1) for d in outliers],
                                        "threshold_ms": round(sigma_threshold, 1),
                                    },
                                }
                            )

                data_events = [e for e in evts if e.event_type == "data"]
                for de in data_events:
                    val = de.metrics.get("value")
                    if val is None:
                        anomalies.append(
                            {
                                "severity": "info",
                                "type": "null_data_value",
                                "module": mod_name,
                                "message": f"Data point '{de.metrics.get('label', '?')}' recorded as None",
                                "details": {"timestamp": de.timestamp},
                            }
                        )
                    elif isinstance(val, (list, str)) and len(val) == 0:
                        anomalies.append(
                            {
                                "severity": "info",
                                "type": "empty_data_value",
                                "module": mod_name,
                                "message": f"Data point '{de.metrics.get('label', '?')}' recorded as empty",
                                "details": {"timestamp": de.timestamp},
                            }
                        )

                decision_events = [e for e in evts if e.event_type == "decision"]
                if len(decision_events) >= _STUCK_ACTION_THRESHOLD:
                    recent_actions = [
                        d.metrics.get("action", "") for d in decision_events[-_STUCK_ACTION_THRESHOLD * 2 :]
                    ]
                    if len(recent_actions) >= _STUCK_ACTION_THRESHOLD:
                        last_n = recent_actions[-_STUCK_ACTION_THRESHOLD:]
                        if len(set(last_n)) == 1:
                            stuck_action = last_n[0]
                            anomalies.append(
                                {
                                    "severity": "warning",
                                    "type": "stuck_decision_pattern",
                                    "module": mod_name,
                                    "message": (
                                        f"Same action '{stuck_action}' repeated "
                                        f"{_STUCK_ACTION_THRESHOLD}+ times consecutively"
                                    ),
                                    "details": {
                                        "stuck_action": stuck_action,
                                        "repeat_count": _STUCK_ACTION_THRESHOLD,
                                    },
                                }
                            )

            health_anomalies = self._detect_health_anomalies()
            anomalies.extend(health_anomalies)

        except (OSError, ValueError, RuntimeError) as e:
            self._log(f"detect_anomalies FAILED: {e}", logging.ERROR)
            anomalies.append(
                {
                    "severity": "critical",
                    "type": "anomaly_detector_error",
                    "module": "_system_",
                    "message": f"Anomaly detection itself failed: {e}",
                }
            )
        return anomalies

    def _detect_health_anomalies(self) -> list[dict]:
        health_anomalies: list[dict] = []
        for mod_name, signal in self._health_signals.items():
            if signal.status == "dead":
                health_anomalies.append(
                    {
                        "severity": "critical",
                        "type": "module_dead",
                        "module": mod_name,
                        "message": f"Health signal reports module '{mod_name}' as DEAD",
                    }
                )
            elif signal.status == "error":
                health_anomalies.append(
                    {
                        "severity": "critical",
                        "type": "module_error_state",
                        "module": mod_name,
                        "message": f"Module '{mod_name}' in ERROR state",
                    }
                )
            elif signal.status == "degraded":
                health_anomalies.append(
                    {
                        "severity": "warning",
                        "type": "module_degraded",
                        "module": mod_name,
                        "message": f"Module '{mod_name}' DEGRADED: {', '.join(signal.warning_flags)}",
                        "details": {"warnings": signal.warning_flags},
                    }
                )
            if signal.success_rate < 0.5 and signal.calls_this_cycle > 0:
                health_anomalies.append(
                    {
                        "severity": "warning",
                        "type": "low_success_rate_health",
                        "module": mod_name,
                        "message": (f"Module '{mod_name}' health reports low success rate {signal.success_rate:.0%}"),
                    }
                )
            if signal.data_quality_score < 0.3:
                health_anomalies.append(
                    {
                        "severity": "warning",
                        "type": "poor_data_quality",
                        "module": mod_name,
                        "message": (
                            f"Module '{mod_name}' data quality score {signal.data_quality_score:.2f} below threshold"
                        ),
                    }
                )
        return health_anomalies

    def get_all_events(self) -> list[dict]:
        try:
            return [asdict(e) for e in self._all_events()]
        except (OSError, ValueError, RuntimeError):
            return []

    def get_recent_events(self, n: int = 50) -> list[dict]:
        try:
            all_ev = self._all_events()
            recent = all_ev[-n:] if n > 0 else []
            return [asdict(e) for e in recent]
        except (OSError, ValueError, RuntimeError):
            return []

    def get_health_snapshot(self) -> dict[str, Any]:
        try:
            return {
                "cycle_id": self._current_cycle,
                "total_events_recorded": len(self._all_events()),
                "pending_events": len(self._pending_events),
                "monitored_modules": list(self._health_signals.keys()),
                "health_signals": {name: asdict(sig) for name, sig in self._health_signals.items()},
            }
        except (OSError, ValueError, RuntimeError) as e:
            return {"error": str(e)}

    def clear_events(self) -> None:
        self._events.clear()
        self._pending_events.clear()
        self._event_count_since_flush = 0

    def _resolve_path(self) -> Path | None:
        if self._telemetry_path is not None:
            base = Path(self._telemetry_path)
        else:
            base = Path.cwd() / "data" / "telemetry"
        cycle = self._current_cycle or "unknown"
        return base / f"cycle_{cycle}.jsonl"

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (int, float, bool, str)):
            return value
        if isinstance(value, (list, tuple)):
            if len(value) > 20:
                return f"<list[{len(value)} items]>"
            return [AlgorithmTelemetryCollector._serialize_value(v) for v in value]
        if isinstance(value, dict):
            truncated = {k: AlgorithmTelemetryCollector._serialize_value(v) for k, v in list(value.items())[:10]}
            if len(value) > 10:
                truncated["__more_keys__"] = len(value) - 10
            return truncated
        return str(value)[:200]

    def record_enter_sync(
        self,
        module_name: str,
        cycle_id: str = "",
        expr_id: int = 0,
        metrics: dict | None = None,
    ) -> None:
        """Synchronous version of record_enter for use in non-async functions."""
        try:
            event = ModuleTelemetryEvent(
                timestamp=time.perf_counter(),
                module_name=module_name,
                event_type="enter",
                cycle_id=str(cycle_id),
                expression_id=str(expr_id) if expr_id else None,
                metrics=metrics or {},
                duration_ms=None,
                metadata={},
            )
            self._pending_events.append(event)
            self._event_count_since_flush += 1
            if self._event_count_since_flush >= _FLUSH_THRESHOLD:
                self.flush_to_disk_sync()
        except (OSError, ValueError, RuntimeError):
            pass

    def record_exit_sync(
        self,
        event_id_or_module: str,
        success: bool = True,
        metrics: dict | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Synchronous version of record_exit for use in non-async functions."""
        try:
            event = ModuleTelemetryEvent(
                timestamp=time.perf_counter(),
                module_name=event_id_or_module
                if not isinstance(event_id_or_module, str) or len(event_id_or_module) < 36
                else "unknown",
                event_type="exit",
                cycle_id="",
                expression_id=None,
                metrics=metrics or {},
                duration_ms=duration_ms,
                metadata={"success": success},
            )
            self._pending_events.append(event)
            self._event_count_since_flush += 1
        except (OSError, ValueError, RuntimeError):
            pass

    def record_error_sync(
        self,
        module_name: str,
        message: str,
        error_type: str = "error",
        metrics: dict | None = None,
    ) -> None:
        """Synchronous version of record_error for use in non-async functions."""
        try:
            event = ModuleTelemetryEvent(
                timestamp=time.perf_counter(),
                module_name=module_name,
                event_type="error",
                cycle_id="",
                expression_id=None,
                metrics={"error_type": error_type, "message": str(message)[:200], **(metrics or {})},
                duration_ms=None,
                metadata={},
            )
            self._pending_events.append(event)
            self._event_count_since_flush += 1
        except (OSError, ValueError, RuntimeError):
            pass

    def record_decision_sync(
        self,
        module_name: str,
        action: str,
        confidence: float = 0.0,
        metrics: dict | None = None,
    ) -> None:
        """Synchronous version of record_decision for use in non-async functions."""
        try:
            event = ModuleTelemetryEvent(
                timestamp=time.perf_counter(),
                module_name=module_name,
                event_type="decision",
                cycle_id="",
                expression_id=None,
                metrics={"action": action, "confidence": confidence, **(metrics or {})},
                duration_ms=None,
                metadata={},
            )
            self._pending_events.append(event)
            self._event_count_since_flush += 1
        except (OSError, ValueError, RuntimeError):
            pass

    def flush_to_disk_sync(self) -> int:
        """Synchronous flush to disk."""
        try:
            if not self._pending_events:
                return 0
            if self._telemetry_path is None:
                return 0
            to_write = self._pending_events[:]
            self._pending_events.clear()
            path = self._telemetry_path / f"cycle_{self._current_cycle}.jsonl"
            written = 0
            with open(path, "a", encoding="utf-8") as f:
                for ev in to_write:
                    try:
                        line = json.dumps(asdict(ev), ensure_ascii=False, default=str)
                        f.write(line + "\n")
                        written += 1
                    except (TypeError, ValueError):
                        continue
            return written
        except (OSError, ValueError, RuntimeError):
            return 0


def telemetry_tracked(module_name: str):
    """Decorator that auto-wraps async functions with enter/exit telemetry.

    Usage:
        @telemetry_tracked("FitnessBoost")
        async def generate_boost_variants(self, expression, ...):
            ...
    """
    import functools
    import inspect

    def decorator(func):
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                collector = AlgorithmTelemetryCollector.get_instance()
                cycle_id = collector._current_cycle or "unknown"

                expr_id = None
                if args and hasattr(args[0], "__dict__"):
                    obj = args[0]
                    expr_val = getattr(obj, "expression", None)
                    if expr_val and isinstance(expr_val, str):
                        expr_id = expr_val[:64]

                t0 = time.perf_counter()
                event_id = await collector.record_enter(
                    module_name,
                    cycle_id=cycle_id,
                    expr_id=expr_id,
                )
                try:
                    result = await func(*args, **kwargs)
                    duration_ms = (time.perf_counter() - t0) * 1000
                    metrics = {}
                    if isinstance(result, dict):
                        for k in ("variants_count", "recommendation", "score", "classification"):
                            if k in result:
                                metrics[k] = result[k]
                    elif isinstance(result, list):
                        metrics["result_count"] = len(result)
                    await collector.record_exit(module_name, event_id, metrics=metrics, duration_ms=duration_ms)
                    return result
                except Exception as exc:
                    duration_ms = (time.perf_counter() - t0) * 1000
                    await collector.record_error(
                        module_name,
                        message=str(exc)[:300],
                        error_type=type(exc).__name__,
                        event_id=event_id,
                        duration_ms=duration_ms,
                    )
                    raise

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            collector = AlgorithmTelemetryCollector.get_instance()
            cycle_id = collector._current_cycle or "unknown"

            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            expr_id = None
            if args and hasattr(args[0], "__dict__"):
                obj = args[0]
                expr_val = getattr(obj, "expression", None)
                if expr_val and isinstance(expr_val, str):
                    expr_id = expr_val[:64]

            t0 = time.perf_counter()

            async def _do_record_enter():
                return await collector.record_enter(module_name, cycle_id=cycle_id, expr_id=expr_id)

            async def _do_record_exit(eid, met, dur):
                await collector.record_exit(module_name, eid, metrics=met, duration_ms=dur)

            async def _do_record_err(eid, dur, exc):
                await collector.record_error(
                    module_name,
                    message=str(exc)[:300],
                    error_type=type(exc).__name__,
                    event_id=eid,
                    duration_ms=dur,
                )

            if loop is not None:
                event_id = asyncio.ensure_future(_do_record_enter())
                if asyncio.iscoroutine(event_id):
                    pass
            else:
                event_id = collector._make_event_id()

            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - t0) * 1000
                if loop is not None:
                    metrics = {}
                    if isinstance(result, dict):
                        for k in ("variants_count", "recommendation", "score", "classification"):
                            if k in result:
                                metrics[k] = result[k]
                    asyncio.ensure_future(_do_record_exit(event_id, metrics, duration_ms))
                return result
            except Exception as exc:
                duration_ms = (time.perf_counter() - t0) * 1000
                if loop is not None:
                    asyncio.ensure_future(_do_record_err(event_id, duration_ms, exc))
                raise

        return sync_wrapper

    return decorator
