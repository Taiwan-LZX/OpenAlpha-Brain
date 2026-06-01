from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from alpha_agent.failure_taxonomy import (
    FailureMode,
    classify_failures,
    primary_failure_mode,
    is_correlation_blocked,
    REPAIR_CATALOGUE,
    repair_severity,
)
from alpha_agent.llm_client import LLMClient
from alpha_agent.operator_registry import get_operator_registry
from alpha_agent.sao import SignalAlphaObject


@dataclass
class RepairResult:
    expression: str
    strategy_name: str
    strategy_index: int
    phase: int
    converged: bool
    stop_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expression": self.expression,
            "strategy_name": self.strategy_name,
            "strategy_index": self.strategy_index,
            "phase": self.phase,
            "converged": self.converged,
            "stop_reason": self.stop_reason,
        }


PHASE_BOUNDARIES = (2, 4)


class ConvergenceGovernance:
    MAX_ITERATIONS = 5
    SHARPE_THRESHOLD = 0.5
    DELTA_THRESHOLD = 0.05

    def __init__(
        self,
        max_iterations: int = MAX_ITERATIONS,
        sharpe_threshold: float = SHARPE_THRESHOLD,
        delta_threshold: float = DELTA_THRESHOLD,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.max_iterations = max_iterations
        self.sharpe_threshold = sharpe_threshold
        self.delta_threshold = delta_threshold
        self._logger = logger or logging.getLogger(__name__)

    def check_convergence(
        self,
        iteration: int,
        sharpe_history: Sequence[Optional[float]],
    ) -> Tuple[bool, Optional[str]]:
        if iteration >= self.max_iterations:
            return True, f"Max iterations ({self.max_iterations}) reached"

        valid = [s for s in sharpe_history if s is not None]
        if not valid:
            return False, None

        current = valid[-1]
        if current >= self.sharpe_threshold:
            return True, f"Sharpe {current:.3f} >= threshold {self.sharpe_threshold}"

        if len(valid) >= 3:
            recent = valid[-3:]
            if max(recent) - min(recent) < self.delta_threshold:
                return True, (
                    f"Converged: sharpe range {min(recent):.3f}-{max(recent):.3f} "
                    f"< delta {self.delta_threshold}"
                )

        return False, None


def wrap_expression(expr: str, wrapper: str) -> str:
    wrapper = wrapper.strip()
    if wrapper.startswith("increase ") or wrapper.startswith("reduce "):
        return RepairEngine.apply_meta_instruction(expr, wrapper)
    if wrapper.startswith("invert "):
        return f"-1 * ({expr})"
    if wrapper.startswith("-1 *"):
        return f"-1 * ({expr})"
    if wrapper.startswith("condition on "):
        return f"if_else(ts_std_dev(returns, 60) < ts_mean(ts_std_dev(returns, 60), 252), {expr}, 0)"
    if wrapper.startswith("relax "):
        return expr
    if "(x)" in wrapper:
        return wrapper.replace("(x)", f"({expr})")
    if "(expr)" in wrapper:
        return wrapper.replace("(expr)", expr)
    m = re.match(r"^(\w+)\(([^)]*)\)$", wrapper)
    if m:
        func_name = m.group(1)
        args = [a.strip() for a in m.group(2).split(",")]
        if args and args[0] in ("x", "expr"):
            remaining = ", ".join(args[1:])
            if remaining:
                return f"{func_name}({expr}, {remaining})"
            return f"{func_name}({expr})"
    return wrapper


class RepairEngine:
    def __init__(
        self,
        governance: Optional[ConvergenceGovernance] = None,
        llm_client: Optional[LLMClient] = None,
        rewrite_temperature: float = 0.3,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.governance = governance or ConvergenceGovernance()
        self.llm_client = llm_client
        self.rewrite_temperature = rewrite_temperature
        self._logger = logger or logging.getLogger(__name__)

    def execute(
        self,
        sao: SignalAlphaObject,
        failed_checks: Sequence[str],
    ) -> RepairResult:
        fms = classify_failures(failed_checks)
        primary = primary_failure_mode(failed_checks)
        correlation_blocked = is_correlation_blocked(failed_checks)

        self._logger.info(
            "M4 repair: %d failure modes, primary=%s, correlation_blocked=%s",
            len(fms), primary.code if primary else "none", correlation_blocked,
        )

        expr = sao.expression
        sharpe_history: List[float] = list(sao.sharpe_history or [])
        tried: set[str] = set(sao.tried_strategies or [])
        sao.failed_checks = list(failed_checks)

        converged, stop_reason = self.governance.check_convergence(
            len(sharpe_history), sharpe_history,
        )
        if converged:
            self._logger.info("M4 converged: %s", stop_reason)
            return RepairResult(
                expression=expr,
                strategy_name="converged",
                strategy_index=len(tried),
                phase=self._phase_for_iteration(len(tried)),
                converged=True,
                stop_reason=stop_reason,
            )

        iteration = len(tried)

        if correlation_blocked and not any("neutralize" in s for s in tried):
            wrapper = "group_neutralize(x, sector)"
            expr = wrap_expression(expr, wrapper)
            self._logger.info("M4 phase 1: applied correlation repair")
            return RepairResult(
                expression=expr,
                strategy_name="neutralize_sector",
                strategy_index=iteration,
                phase=1,
                converged=False,
            )

        strategy = self._select_best_untried_strategy(primary, fms, tried, iteration)
        if strategy is None:
            self._logger.info("M4: no untried strategies remain")
            return RepairResult(
                expression=expr,
                strategy_name="none_available",
                strategy_index=iteration,
                phase=self._phase_for_iteration(iteration),
                converged=True,
                stop_reason="No untried repair strategies available",
            )

        strategy_name, strategy_expr, _ = strategy

        if self.llm_client:
            try:
                rewritten = self._llm_rewrite_expression(sao, strategy_name, list(failed_checks))
                if rewritten and rewritten != expr:
                    expr = rewritten
                    self._logger.info("M4 phase 3: LLM rewrote expression via '%s'", strategy_name)
                else:
                    expr = wrap_expression(expr, strategy_expr)
            except Exception as exc:
                self._logger.warning("M4 LLM rewrite failed, falling back to wrap: %s", exc)
                expr = wrap_expression(expr, strategy_expr)
        else:
            expr = wrap_expression(expr, strategy_expr)

        self._logger.info(
            "M4 iteration %d phase %d: applying '%s'",
            iteration, self._phase_for_iteration(iteration), strategy_name,
        )

        return RepairResult(
            expression=expr,
            strategy_name=strategy_name,
            strategy_index=iteration,
            phase=self._phase_for_iteration(iteration),
            converged=False,
        )

    @staticmethod
    def _strategy_to_categories(strategy_name: str) -> List[str]:
        lower = strategy_name.lower()
        if "neutralize" in lower or "group" in lower or "sector" in lower:
            return ["Group", "Cross Sectional"]
        if "winsorize" in lower or "outlier" in lower or "clip" in lower:
            return ["Transformational", "Cross Sectional"]
        if "zscore" in lower or "normalize" in lower or "scale" in lower:
            return ["Cross Sectional", "Transformational"]
        if "rank" in lower:
            return ["Cross Sectional", "Logical"]
        if "turnover" in lower or "decay" in lower or "smooth" in lower:
            return ["Time Series", "Transformational"]
        return ["Group", "Cross Sectional", "Time Series"]

    def _build_rewrite_prompt(
        self,
        sao: SignalAlphaObject,
        strategy_name: str,
        failed_checks: List[str],
    ) -> Tuple[str, str]:
        system_prompt = (
            "You are a quantitative researcher repairing a WorldQuant FASTEXPR expression. "
            "Apply the selected repair strategy to fix the expression. "
            "Return strict JSON with keys: expression, reasoning."
        )
        history_str = "; ".join(sao.expression_history[-5:]) if sao.expression_history else "none"
        sharpe_str = "; ".join(f"{s:.3f}" for s in (sao.sharpe_history or [])[-5:]) if sao.sharpe_history else "none"

        reg = get_operator_registry()
        categories = self._strategy_to_categories(strategy_name)
        relevant: List[str] = []
        seen: Set[str] = set()
        for cat in categories:
            for op in reg.get_operators_for_category(cat, top_k=5):
                name = op.get("name", "")
                if name not in seen:
                    seen.add(name)
                    relevant.append(name)
        operators_block = reg.format_operators_for_prompt(relevant)
        if operators_block:
            operators_block = f"Available repair operators:\n{operators_block}\n\nApply the repair strategy using ONLY the operators listed above."

        user_prompt = (
            f"Repair strategy: {strategy_name}\n"
            f"Current expression: {sao.expression}\n"
            f"Failed checks: {', '.join(failed_checks) if failed_checks else 'none'}\n"
            f"Repair history (recent): {history_str}\n"
            f"Sharpe history (recent): {sharpe_str}\n"
            f"Original hypothesis: {sao.statement}\n\n"
            f"{operators_block}"
        )
        return system_prompt, user_prompt

    def _llm_rewrite_expression(
        self,
        sao: SignalAlphaObject,
        strategy_name: str,
        failed_checks: List[str],
    ) -> Optional[str]:
        if not self.llm_client:
            return None
        system_prompt, user_prompt = self._build_rewrite_prompt(sao, strategy_name, failed_checks)
        payload = self.llm_client.request_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.rewrite_temperature,
        )
        expression = str(payload.get("expression", "")).strip()
        if not expression:
            return None
        return expression

    def _phase_for_iteration(self, iteration: int) -> int:
        if iteration < 2:
            return 1
        if iteration < 4:
            return 2
        return 3

    def _select_best_untried_strategy(
        self,
        primary: Optional[FailureMode],
        fms: List[FailureMode],
        tried: set[str],
        iteration: int,
    ) -> Optional[Tuple[str, str, str]]:
        candidates: List[Tuple[str, str, str]] = []

        if primary is not None and primary in REPAIR_CATALOGUE:
            candidates.extend(REPAIR_CATALOGUE[primary])

        for fm in fms:
            if fm != primary and fm in REPAIR_CATALOGUE:
                candidates.extend(REPAIR_CATALOGUE[fm])

        seen: set[str] = set()
        unique: List[Tuple[str, str, str]] = []
        for c in candidates:
            if c[0] not in seen:
                seen.add(c[0])
                unique.append(c)

        untried = [c for c in unique if c[0] not in tried]
        if not untried:
            return None

        untried.sort(key=lambda c: repair_severity(c[0]), reverse=True)

        if iteration < 2:
            return untried[0]

        if len(untried) >= 2:
            return untried[-1]
        return untried[0]

    @staticmethod
    def apply_meta_instruction(expr: str, instruction: str) -> str:
        instruction = instruction.strip().lower()
        if instruction == "extend_lookback_2x" or instruction.startswith("increase "):
            def _double_window(m: re.Match) -> str:
                num = int(m.group(2))
                if num == 0:
                    return m.group(0)
                return f"{m.group(1)}{num * 2}{m.group(3)}"
            return re.sub(r'(,\s*)(\d+)(\s*\))', _double_window, expr)
        if instruction == "reduce_lookback_half" or instruction.startswith("reduce "):
            def _half_window(m: re.Match) -> str:
                num = int(m.group(2))
                if num == 0:
                    return m.group(0)
                return f"{m.group(1)}{max(2, num // 2)}{m.group(3)}"
            return re.sub(r'(,\s*)(\d+)(\s*\))', _half_window, expr)
        return expr

    def compute_repair_effectiveness(
        self, sao: SignalAlphaObject
    ) -> Dict[str, Any]:
        history = sao.sharpe_history or []
        if len(history) < 2:
            return {"delta": 0.0, "direction": "flat"}
        delta = history[-1] - history[0]
        return {
            "initial_sharpe": history[0],
            "final_sharpe": history[-1],
            "delta": round(delta, 4),
            "direction": "improved" if delta > 0 else "degraded" if delta < 0 else "flat",
            "iterations": len(history),
        }
