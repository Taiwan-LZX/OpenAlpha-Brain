from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Event types emitted by the pipeline
EVENT_CYCLE_START = "cycle_start"
EVENT_ALPHA_GENERATED = "alpha_generated"
EVENT_ALPHA_VALIDATED = "alpha_validated"
EVENT_ALPHA_REJECTED = "alpha_rejected"
EVENT_BRAIN_SUBMIT = "brain_submit"
EVENT_BRAIN_RESULT = "brain_result"
EVENT_MAB_FEEDBACK = "mab_feedback"
EVENT_CYCLE_COMPLETE = "cycle_complete"
EVENT_MINING_COMPLETE = "mining_complete"
EVENT_ERROR = "error"
EVENT_WARNING = "warning"

_ALL_EVENTS = [
    EVENT_CYCLE_START, EVENT_ALPHA_GENERATED, EVENT_ALPHA_VALIDATED,
    EVENT_ALPHA_REJECTED, EVENT_BRAIN_SUBMIT, EVENT_BRAIN_RESULT,
    EVENT_MAB_FEEDBACK, EVENT_CYCLE_COMPLETE, EVENT_MINING_COMPLETE,
    EVENT_ERROR, EVENT_WARNING,
]


class AlphaEvent:
    __slots__ = ("data", "event_type", "timestamp")

    def __init__(self, event_type: str, data: dict[str, Any]):
        self.event_type = event_type
        self.data = data
        self.timestamp = time.time()

    def __repr__(self) -> str:
        return f"AlphaEvent({self.event_type}, t={self.timestamp:.1f})"


class AlphaEventBus:
    _instance: AlphaEventBus | None = None
    _lock = threading.Lock()

    def __new__(cls) -> AlphaEventBus:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._listeners: list[Callable] = []
                    cls._instance._history: deque[AlphaEvent] = deque(maxlen=200)
                    cls._instance._enabled = True
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool) -> None:
        self._enabled = val

    def subscribe(self, listener: Callable[[AlphaEvent], Any]) -> None:
        self._listeners.append(listener)

    def unsubscribe(self, listener: Callable) -> None:
        try:
            self._listeners.remove(listener)
        except ValueError:
            pass

    def emit(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        if not self._enabled:
            return
        data = data or {}
        event = AlphaEvent(event_type, data)
        self._history.append(event)
        for listener in list(self._listeners):
            try:
                if asyncio.iscoroutinefunction(listener):
                    asyncio.create_task(listener(event))
                else:
                    listener(event)
            except (OSError, ValueError, RuntimeError) as exc:
                logger.debug("Event listener error on %s: %s", event_type, exc)

    async def emit_async(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        if not self._enabled:
            return
        data = data or {}
        event = AlphaEvent(event_type, data)
        self._history.append(event)
        for listener in list(self._listeners):
            try:
                result = listener(event)
                if asyncio.iscoroutinefunction(listener):
                    await listener(event)
                elif asyncio.iscoroutine(result):
                    await result
                else:
                    listener(event)
            except (OSError, ValueError, RuntimeError) as exc:
                logger.debug("Event listener error (async) on %s: %s", event_type, exc)

    def recent_events(self, event_type: str | None = None, limit: int = 20) -> list[AlphaEvent]:
        events = list(self._history)
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]

    def clear_history(self) -> None:
        self._history.clear()


def get_event_bus() -> AlphaEventBus:
    return AlphaEventBus()
