from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass
class AlgoEvent:
    """[Brief description of class purpose.]"""

    timestamp: float
    level: str
    module: str
    step: str
    result: str
    session_id: str = ""
    alpha_id: str = ""
    duration_ms: float = 0.0


class AlgoMonitor:
    """[Brief description of class purpose.]"""

    _instance = None

    def __init__(self, max_events: int = 1000):
        """[Brief description of function purpose.]

        Args:
            max_events (int): [Description]
        """
        self._events: list[AlgoEvent] = []
        self._max_events = max_events
        self._module_stats: dict[str, dict] = defaultdict(
            lambda: {
                "total": 0,
                "pass": 0,
                "fail": 0,
                "skip": 0,
                "step": 0,
                "last_timestamp": 0.0,
            }
        )

    @classmethod
    def get_instance(cls) -> AlgoMonitor:
        """[Brief description of function purpose.]

        Returns:
            'AlgoMonitor': [Description]
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def record(
        self,
        level: str,
        module: str,
        step: str,
        result: str,
        session_id: str = "",
        alpha_id: str = "",
        duration_ms: float = 0.0,
    ) -> None:
        """[Brief description of function purpose.]

        Args:
            level (str): [Description]
            module (str): [Description]
            step (str): [Description]
            result (str): [Description]
            session_id (str): [Description]
            alpha_id (str): [Description]
            duration_ms (float): [Description]

        Returns:
            None: [Description]
        """
        event = AlgoEvent(
            timestamp=time.time(),
            level=level,
            module=module,
            step=step,
            result=result,
            session_id=session_id,
            alpha_id=alpha_id,
            duration_ms=duration_ms,
        )
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events :]
        stats = self._module_stats[module]
        stats["total"] += 1
        stats[level.lower()] = stats.get(level.lower(), 0) + 1
        stats["last_timestamp"] = event.timestamp

    def get_status(self) -> dict:
        """[Brief description of function purpose.]

        Returns:
            dict: [Description]
        """
        return {
            "modules": dict(self._module_stats),
            "recent_events": [
                {
                    "timestamp": e.timestamp,
                    "level": e.level,
                    "module": e.module,
                    "step": e.step,
                    "result": e.result,
                    "session_id": e.session_id,
                    "alpha_id": e.alpha_id,
                    "duration_ms": e.duration_ms,
                }
                for e in self._events[-50:]
            ],
            "total_events": len(self._events),
        }

    @staticmethod
    def aggregate_health_checks(modules: dict[str, Any]) -> dict[str, dict]:
        """[Brief description of function purpose.]

        Args:
            modules (dict[str, Any]): [Description]

        Returns:
            dict[str, dict]: [Description]
        """
        results: dict[str, dict] = {}
        for name, instance in modules.items():
            if hasattr(instance, "health_check") and callable(instance.health_check):
                try:
                    results[name] = instance.health_check()
                except (ValueError, TypeError, OSError) as exc:
                    results[name] = {"module": name, "status": "error", "error": str(exc)}
            else:
                results[name] = {"module": name, "status": "no_health_check"}
        return results

    @staticmethod
    def detect_ghost_algorithms(modules: dict[str, Any]) -> list[str]:
        """[Brief description of function purpose.]

        Args:
            modules (dict[str, Any]): [Description]

        Returns:
            list[str]: [Description]
        """
        ghosts: list[str] = []
        for name, instance in modules.items():
            if not hasattr(instance, "health_check") or not callable(instance.health_check):
                continue
            try:
                report = instance.health_check()
            except (ValueError, TypeError, OSError):
                continue
            status = report.get("status", "")
            if status in ("inactive", "disabled"):
                ghosts.append(name)
                continue
            if "update_count" in report and report["update_count"] == 0 and status == "active":
                ghosts.append(name)
        return ghosts
