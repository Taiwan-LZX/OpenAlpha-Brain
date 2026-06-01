"""
WebSocket Renderer — JSON Output Layer for Web Frontends

This module provides WebSocket-specific rendering logic, transforming structured
event data into JSON format suitable for web frontends.

Key Benefits:
1. Consistent JSON structure for all event types
2. Type-safe data extraction from adapters
3. Easy integration with JavaScript/TypeScript frontends
4. Separated from broadcast logic for better testability
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

from openalpha_brain.cli.event_adapters import (
    AlphaGeneratedAdapter,
    AlphaRejectedAdapter,
    AlphaValidatedAdapter,
    BaseEventAdapter,
    BrainResultAdapter,
    BrainSubmitAdapter,
    CycleStartAdapter,
    ErrorAdapter,
    MabFeedbackAdapter,
    MiningCompleteAdapter,
)


class BaseRenderer(ABC):
    """Base renderer interface (redefined for consistency)"""

    @abstractmethod
    def render(self, adapter: BaseEventAdapter) -> dict[str, Any]:
        """
        Render an event adapter into output format.

        Args:
            adapter: Event adapter containing structured data

        Returns:
            Dictionary suitable for serialization
        """
        pass


class WebSocketRenderer(BaseRenderer):
    """
    WebSocket-specific renderer that produces JSON-formatted output.

    This renderer transforms event data into a consistent JSON structure
    that can be easily consumed by web frontends.
    """

    # Module mapping for frontend log categorization
    MODULE_MAP = {
        "cycle_start": "System",
        "cycle_complete": "System",
        "alpha_generated": "FactorAgent",
        "alpha_validated": "EvalAgent",
        "alpha_rejected": "EvalAgent",
        "brain_submit": "BRAIN",
        "brain_result": "BRAIN",
        "mab_feedback": "MAB",
        "mining_complete": "System",
        "error": "System",
        "warning": "System",
    }

    def render(self, adapter: BaseEventAdapter) -> dict[str, Any]:
        """
        Route to appropriate renderer based on adapter type.

        Args:
            adapter: Event adapter to render

        Returns:
            Dictionary with type, module, timestamp, and data fields
        """
        return {
            "type": adapter.event_type,
            "module": self.MODULE_MAP.get(adapter.event_type, "System"),
            "timestamp": datetime.fromtimestamp(adapter.timestamp, tz=UTC).isoformat(),
            "data": self._extract_data(adapter),
        }

    def _extract_data(self, adapter: BaseEventAdapter) -> dict[str, Any]:
        """
        Extract structured data from adapter based on its type.

        This method provides type-safe data extraction with clear field names
        for frontend consumption.

        Args:
            adapter: Event adapter to extract data from

        Returns:
            Dictionary containing relevant data fields
        """
        if isinstance(adapter, CycleStartAdapter):
            return self._extract_cycle_start_data(adapter)
        elif isinstance(adapter, AlphaGeneratedAdapter):
            return self._extract_alpha_generated_data(adapter)
        elif isinstance(adapter, AlphaValidatedAdapter):
            return self._extract_alpha_validated_data(adapter)
        elif isinstance(adapter, AlphaRejectedAdapter):
            return self._extract_alpha_rejected_data(adapter)
        elif isinstance(adapter, BrainSubmitAdapter):
            return self._extract_brain_submit_data(adapter)
        elif isinstance(adapter, BrainResultAdapter):
            return self._extract_brain_result_data(adapter)
        elif isinstance(adapter, MabFeedbackAdapter):
            return self._extract_mab_feedback_data(adapter)
        elif isinstance(adapter, MiningCompleteAdapter):
            return self._extract_mining_complete_data(adapter)
        elif isinstance(adapter, ErrorAdapter):
            return self._extract_error_data(adapter)
        else:
            return adapter.raw_data

    def _extract_cycle_start_data(self, adapter: CycleStartAdapter) -> dict[str, Any]:
        """Extract cycle start data"""
        return {
            "cycle": adapter.cycle,
            "max_cycles": adapter.max_cycles,
            "mode": adapter.mode,
            "is_pipeline": adapter.is_pipeline,
        }

    def _extract_alpha_generated_data(self, adapter: AlphaGeneratedAdapter) -> dict[str, Any]:
        """Extract alpha generated data"""
        return {
            "expression": adapter.expression,
            "direction": adapter.direction,
        }

    def _extract_alpha_validated_data(self, adapter: AlphaValidatedAdapter) -> dict[str, Any]:
        """Extract alpha validated data"""
        return {
            "alpha_id": adapter.alpha_id,
            "expression": adapter.expression,
            "direction": adapter.direction,
            "family": adapter.family,
        }

    def _extract_alpha_rejected_data(self, adapter: AlphaRejectedAdapter) -> dict[str, Any]:
        """Extract alpha rejected data"""
        return {
            "expression": adapter.expression,
            "reason": adapter.reason,
            "reject_category": adapter.reject_category,
        }

    def _extract_brain_submit_data(self, adapter: BrainSubmitAdapter) -> dict[str, Any]:
        """Extract brain submit data"""
        return {
            "alpha_id": adapter.alpha_id,
            "worker_id": adapter.worker_id,
            "has_worker_info": adapter.has_worker_info,
        }

    def _extract_brain_result_data(self, adapter: BrainResultAdapter) -> dict[str, Any]:
        """Extract brain result data"""
        return {
            "alpha_id": adapter.alpha_id,
            "status": adapter.status,
            "sharpe": adapter.sharpe,
            "fitness": adapter.fitness,
            "turnover": adapter.turnover,
            "drawdown": adapter.drawdown,
            "is_pass": adapter.is_pass,
            "expression": adapter.expression,
            "direction": adapter.direction,
        }

    def _extract_mab_feedback_data(self, adapter: MabFeedbackAdapter) -> dict[str, Any]:
        """Extract MAB feedback data"""
        return {
            "direction": adapter.direction,
            "reward": adapter.reward,
        }

    def _extract_mining_complete_data(self, adapter: MiningCompleteAdapter) -> dict[str, Any]:
        """Extract mining complete data"""
        return {
            "mode": adapter.mode,
            "cycles": adapter.cycles,
            "alphas_passed": adapter.alphas_passed,
        }

    def _extract_error_data(self, adapter: ErrorAdapter) -> dict[str, Any]:
        """Extract error data"""
        return {
            "message": adapter.message,
        }


def get_ws_renderer() -> WebSocketRenderer:
    """Factory function for creating WebSocket renderer instance"""
    return WebSocketRenderer()


def render_to_json(adapter: BaseEventAdapter) -> str:
    """
    Convenience function to render adapter directly to JSON string.

    Args:
        adapter: Event adapter to render

    Returns:
        JSON string representation of the rendered event
    """
    renderer = get_ws_renderer()
    result = renderer.render(adapter)
    return json.dumps(result, default=str, ensure_ascii=False)
