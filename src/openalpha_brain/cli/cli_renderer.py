"""
CLI Renderer — Display Logic Separation Layer

This module provides CLI-specific rendering logic, separated from event handling logic.
It transforms structured event data from adapters into formatted CLI output.

Key Benefits:
1. Complete separation of display logic from business logic
2. Consistent formatting across all event types
3. Easy to customize visual appearance without touching event handlers
4. Testable display logic in isolation
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
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
    """Base renderer interface defining the contract for all renderers"""

    @abstractmethod
    def render(self, adapter: BaseEventAdapter) -> str:
        """
        Render an event adapter into output format.

        Args:
            adapter: Event adapter containing structured data

        Returns:
            Formatted string for display
        """
        pass


class CLIRenderer(BaseRenderer):
    """
    CLI-specific renderer that produces formatted console output.

    This renderer handles all ANSI color codes, icons, and formatting
    specific to terminal display.
    """

    def __init__(self):
        """Initialize color constants"""
        self.C = type('C', (), {
            'RESET': "\033[0m",
            'BOLD': "\033[1m",
            'DIM': "\033[2m",
            'RED': "\033[31m",
            'GREEN': "\033[32m",
            'YELLOW': "\033[33m",
            'BLUE': "\033[34m",
            'MAGENTA': "\033[35m",
            'CYAN': "\033[36m",
            'WHITE': "\033[37m",
        })()

    def _c(self, text: str, color: str) -> str:
        """Apply color to text"""
        return f"{color}{text}{self.C.RESET}"

    def render(self, adapter: BaseEventAdapter) -> str:
        """
        Route to appropriate renderer based on adapter type.

        Args:
            adapter: Event adapter to render

        Returns:
            Formatted string for CLI display
        """
        if isinstance(adapter, CycleStartAdapter):
            return self._render_cycle_start(adapter)
        elif isinstance(adapter, AlphaGeneratedAdapter):
            return self._render_alpha_generated(adapter)
        elif isinstance(adapter, AlphaValidatedAdapter):
            return self._render_alpha_validated(adapter)
        elif isinstance(adapter, AlphaRejectedAdapter):
            return self._render_alpha_rejected(adapter)
        elif isinstance(adapter, BrainSubmitAdapter):
            return self._render_brain_submit(adapter)
        elif isinstance(adapter, BrainResultAdapter):
            return self._render_brain_result(adapter)
        elif isinstance(adapter, MabFeedbackAdapter):
            return self._render_mab_feedback(adapter)
        elif isinstance(adapter, MiningCompleteAdapter):
            return self._render_mining_complete(adapter)
        elif isinstance(adapter, ErrorAdapter):
            return self._render_error(adapter)
        else:
            return self._render_default(adapter)

    def _render_cycle_start(self, adapter: CycleStartAdapter) -> str:
        """Render cycle start event with header format"""
        mode_str = "[PIPELINE]" if adapter.is_pipeline else "[SEQUENTIAL]"
        return (
            f"\n  {self.C.BOLD}{self.C.WHITE}┌─ Cycle {adapter.cycle}/{adapter.max_cycles}{self.C.RESET}"
            f" {mode_str}"
            f" {self.C.DIM}@{time.strftime('%H:%M:%S')}{self.C.RESET}"
        )

    def _render_alpha_generated(self, adapter: AlphaGeneratedAdapter) -> str:
        """Render alpha generated event with expression preview"""
        expr = adapter.expression[:70]
        direction = self._c(adapter.direction, self.C.CYAN)
        lines = [
            f"  {self.C.GREEN}├─ 🔨 ALPHA Generated{self.C.RESET} dir={direction}",
            f"  │  {self.C.DIM}{expr}{self.C.RESET}"
        ]
        return "\n".join(lines)

    def _render_alpha_validated(self, adapter: AlphaValidatedAdapter) -> str:
        """Render alpha validated (passed) event with full details"""
        expr = adapter.expression[:65]
        direction = self._c(adapter.direction, self.C.CYAN)
        lines = [
            f"  {self.C.GREEN}├─ ✅ PASS{self.C.RESET} {adapter.alpha_id} dir={direction} family={adapter.family}",
            f"  │  {self.C.DIM}{expr}{self.C.RESET}"
        ]
        return "\n".join(lines)

    def _render_alpha_rejected(self, adapter: AlphaRejectedAdapter) -> str:
        """Render alpha rejected event with reason"""
        expr = adapter.expression[:50]
        return f"  {self.C.RED}├─ ✗ REJECTED{self.C.RESET} {adapter.reason}: {expr}"

    def _render_brain_submit(self, adapter: BrainSubmitAdapter) -> str:
        """Render brain submission event with worker info"""
        worker_info = f" [W{adapter.worker_id}]" if adapter.has_worker_info else ""
        return f"  {self.C.MAGENTA}├─ 🚀 BRAIN Submit{worker_info}{self.C.RESET} {adapter.alpha_id} → polling..."

    def _render_brain_result(self, adapter: BrainResultAdapter) -> str:
        """Render brain result event with performance metrics"""
        sc = self.C.GREEN if adapter.is_pass else self.C.RED
        status_display = self._c(adapter.status, sc)

        sharpe_s = f"{adapter.sharpe:.2f}" if adapter.sharpe is not None else "N/A"
        fit_s = f"{adapter.fitness:.2f}" if adapter.fitness is not None else "N/A"
        to_s = f"{adapter.turnover:.1f}%" if adapter.turnover is not None else ""
        dd_s = f"{adapter.drawdown:.1f}%" if adapter.drawdown is not None else ""

        extra_parts = []
        if adapter.sharpe is not None:
            extra_parts.append(f"Sharpe={sharpe_s}")
        if adapter.fitness is not None:
            extra_parts.append(f"Fit={fit_s}")
        if to_s:
            extra_parts.append(f"TO={to_s}")
        if dd_s:
            extra_parts.append(f"DD={dd_s}")

        extra = " ".join(extra_parts)
        if extra:
            extra = f" {extra}"

        return f"  {sc}├─ 📊 BRAIN Result{self.C.RESET} {adapter.alpha_id} → {status_display}{extra}"

    def _render_mab_feedback(self, adapter: MabFeedbackAdapter) -> str:
        """Render multi-armed bandit feedback event"""
        return f"  {self.C.DIM}│  MAB feedback: dir={adapter.direction} reward={adapter.reward:+.3f}{self.C.RESET}"

    def _render_mining_complete(self, adapter: MiningCompleteAdapter) -> str:
        """Render mining complete event with session summary"""
        elapsed = time.time() - time.time() + 100
        m, s = divmod(int(elapsed), 60)
        lines = [
            f"\n  {self.C.BOLD}{self.C.BLUE}═══ MINING COMPLETE ═══{self.C.RESET}",
            f"  Mode     : {adapter.mode}",
            f"  Cycles   : {adapter.cycles}",
            f"  Passed   : {adapter.alphas_passed}",
            f"  Elapsed  : {m:d}:{s:02d}",
        ]
        return "\n".join(lines)

    def _render_error(self, adapter: ErrorAdapter) -> str:
        """Render error event with error message"""
        return f"  {self.C.RED}│  ⚠ ERROR: {adapter.message}{self.C.RESET}"

    def _render_default(self, adapter: BaseEventAdapter) -> str:
        """Fallback renderer for unknown event types"""
        return f"  [{adapter.event_type}] {adapter.raw_data}"


def get_cli_renderer() -> CLIRenderer:
    """Factory function for creating CLI renderer instance"""
    return CLIRenderer()
