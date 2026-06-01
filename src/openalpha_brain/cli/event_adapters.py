"""
Event Adapters Layer — Semantic Bridge between Algorithm and UI Layers

This module provides a unified data access interface for events emitted by the algorithm layer.
It establishes clear semantic contracts and decouples the algorithm layer from UI implementations.

Key Benefits:
1. Unified event data access interface with type-safe property access
2. Data validation and default value handling
3. Clear semantic contracts for each event type
4. Flexible extension mechanism for new event types
"""

from __future__ import annotations

import time
from abc import ABC
from typing import Any

from openalpha_brain.core.events import AlphaEvent


class BaseEventAdapter(ABC):
    """Base adapter providing unified event data access interface"""

    def __init__(self, event: AlphaEvent):
        self._event = event
        self._data = event.data or {}

    @property
    def event_type(self) -> str:
        return self._event.event_type

    @property
    def timestamp(self) -> float:
        return self._event.timestamp

    def get(self, key: str, default: Any = None) -> Any:
        """Safely retrieve event data with default value"""
        return self._data.get(key, default)

    @property
    def raw_data(self) -> dict[str, Any]:
        """Access raw event data for advanced use cases"""
        return self._data


class CycleStartAdapter(BaseEventAdapter):
    """Cycle start event adapter - provides structured access to cycle information"""

    @property
    def cycle(self) -> int:
        """Current cycle number (1-based)"""
        return self._data.get("cycle", 0)

    @property
    def max_cycles(self) -> int | str:
        """Maximum number of cycles in this session"""
        return self._data.get("max_cycles", "?")

    @property
    def mode(self) -> str:
        """Execution mode: 'pipeline' or 'sequential'"""
        return self._data.get("mode", "seq")

    @property
    def is_pipeline(self) -> bool:
        """Check if running in pipeline mode"""
        return self.mode == "pipeline"


class AlphaGeneratedAdapter(BaseEventAdapter):
    """Alpha generated event adapter - provides structured access to alpha generation data"""

    @property
    def expression(self) -> str:
        """Alpha expression string"""
        return self._data.get("expression", "?")

    @property
    def direction(self) -> str:
        """Exploration direction: long/short/neutral"""
        return self._data.get("direction", "?")


class AlphaValidatedAdapter(BaseEventAdapter):
    """Alpha validated event adapter - provides structured access to validation results"""

    @property
    def alpha_id(self) -> str:
        """Unique identifier for the validated alpha"""
        return self._data.get("alpha_id", "?")

    @property
    def expression(self) -> str:
        """Validated alpha expression"""
        return self._data.get("expression", "?")

    @property
    def direction(self) -> str:
        """Exploration direction of the alpha"""
        return self._data.get("direction", "?")

    @property
    def family(self) -> str:
        """Strategy family classification"""
        return self._data.get("family", "?")


class AlphaRejectedAdapter(BaseEventAdapter):
    """Alpha rejected event adapter - provides structured access to rejection details"""

    @property
    def expression(self) -> str:
        """Rejected alpha expression"""
        return self._data.get("expression", "?")

    @property
    def reason(self) -> str:
        """Rejection reason description"""
        return self._data.get("reason", "?")

    @property
    def reject_category(self) -> str:
        """Rejection category: syntax/validation/business"""
        return self._data.get("reject_category", "unknown")


class BrainSubmitAdapter(BaseEventAdapter):
    """Brain submit event adapter - provides structured access to submission data"""

    @property
    def alpha_id(self) -> str:
        """Identifier of the submitted alpha"""
        return self._data.get("alpha_id", "?")

    @property
    def worker_id(self) -> str:
        """Worker node ID handling the submission"""
        return self._data.get("worker_id", "")

    @property
    def submit_time(self) -> float:
        """Submission timestamp"""
        return self._data.get("submit_time", time.time())

    @property
    def has_worker_info(self) -> bool:
        """Check if worker information is available"""
        return bool(self.worker_id)


class BrainResultAdapter(BaseEventAdapter):
    """Brain result event adapter - provides structured access to brain simulation results"""

    @property
    def alpha_id(self) -> str:
        """Identifier of the evaluated alpha"""
        return self._data.get("alpha_id", "?")

    @property
    def status(self) -> str:
        """Evaluation status: PASS or FAIL"""
        return self._data.get("status", "?")

    @property
    def sharpe(self) -> float | None:
        """Sharpe ratio from simulation"""
        return self._data.get("sharpe")

    @property
    def fitness(self) -> float | None:
        """Fitness score from simulation"""
        return self._data.get("fitness")

    @property
    def turnover(self) -> float | None:
        """Turnover percentage"""
        return self._data.get("turnover")

    @property
    def drawdown(self) -> float | None:
        """Maximum drawdown percentage"""
        return self._data.get("drawdown")

    @property
    def is_pass(self) -> bool:
        """Check if the alpha passed evaluation"""
        return self.status == "PASS"

    @property
    def expression(self) -> str:
        """Original alpha expression"""
        return self._data.get("expression", "")

    @property
    def direction(self) -> str:
        """Alpha direction"""
        return self._data.get("direction", "")


class MabFeedbackAdapter(BaseEventAdapter):
    """MAB feedback event adapter - provides structured access to multi-armed bandit feedback"""

    @property
    def direction(self) -> str:
        """Direction that received feedback"""
        return self._data.get("direction", "?")

    @property
    def reward(self) -> float:
        """Reward value for the direction"""
        return self._data.get("reward", 0)


class MiningCompleteAdapter(BaseEventAdapter):
    """Mining complete event adapter - provides structured access to session summary"""

    @property
    def mode(self) -> str:
        """Mining mode used"""
        return self._data.get("mode", "?")

    @property
    def cycles(self) -> int:
        """Total cycles completed"""
        return self._data.get("cycles", 0)

    @property
    def alphas_passed(self) -> int:
        """Number of alphas that passed validation"""
        return self._data.get("alphas_passed", 0)


class ErrorAdapter(BaseEventAdapter):
    """Error event adapter - provides structured access to error information"""

    @property
    def message(self) -> str:
        """Error message"""
        return self._data.get("error") or self._data.get("message", "?")


class EventAdapterFactory:
    """
    Factory class for creating appropriate event adapters based on event type.

    This factory implements the Factory Pattern to decouple adapter creation
    from usage, allowing easy registration of new event types.
    """

    _adapters: dict[str, type[BaseEventAdapter]] = {
        "cycle_start": CycleStartAdapter,
        "alpha_generated": AlphaGeneratedAdapter,
        "alpha_validated": AlphaValidatedAdapter,
        "alpha_rejected": AlphaRejectedAdapter,
        "brain_submit": BrainSubmitAdapter,
        "brain_result": BrainResultAdapter,
        "mab_feedback": MabFeedbackAdapter,
        "mining_complete": MiningCompleteAdapter,
        "error": ErrorAdapter,
    }

    @classmethod
    def create(cls, event: AlphaEvent) -> BaseEventAdapter:
        """
        Create appropriate adapter based on event type.

        Args:
            event: Raw AlphaEvent from the event bus

        Returns:
            Appropriate adapter instance for the event type
        """
        adapter_class = cls._adapters.get(event.event_type, BaseEventAdapter)
        return adapter_class(event)

    @classmethod
    def register_adapter(cls, event_type: str, adapter_class: type[BaseEventAdapter]) -> None:
        """
        Register a new adapter for a custom event type.

        This allows extending the system with new event types without modifying core code.

        Args:
            event_type: Event type string identifier
            adapter_class: Adapter class to handle this event type
        """
        cls._adapters[event_type] = adapter_class

    @classmethod
    def get_registered_types(cls) -> list[str]:
        """Get list of all registered event types"""
        return list(cls._adapters.keys())
