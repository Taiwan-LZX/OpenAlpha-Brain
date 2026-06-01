"""Near-Pass Targeted Improvement Engine.

For factors that are CLOSE to passing WQ BRAIN review (e.g., Sharpe≥1.0 but Fitness<1.0),
this engine applies targeted mutations to push them over the threshold.

Strategy: Hybrid (Deterministic first, then LLM)
- Phase 1: Deterministic mutations (fast, 5-15 variants in <1s)
- Phase 2: LLM-driven improvement (slower, but smarter)

Integration:
- Auto-triggered by FeedbackOrchestrator when NEAR_PASS decision is made
- Submits improved expressions with HIGH/EMERGENCY priority to SlotManager
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, Awaitable, Any

from openalpha_brain.utils.algo_logger import algo_log
from openalpha_brain.monitoring.algorithm_telemetry import AlgorithmTelemetryCollector

logger = logging.getLogger(__name__)

GATE_SHARPE_MIN = 1.25
GATE_FITNESS_MIN = 1.0

NEAR_PASS_SHARPE_THRESHOLD = 0.8
NEAR_PASS_FITNESS_THRESHOLD = 0.5
MAX_DETERMINISTIC_VARIANTS = 12
MAX_LLM_IMPROVEMENT_ATTEMPTS = 3


class NearPassCategory(Enum):
    SHARPE_GOOD_FITNESS_POOR = "sharpe_good_fitness_poor"
    FITNESS_GOOD_SHARPE_POOR = "fitness_good_sharpe_poor"
    BOTH_NEAR = "both_near"
    NOT_NEAR = "not_near"


@dataclass
class NearPassAnalysis:
    category: NearPassCategory = NearPassCategory.NOT_NEAR
    sharpe_gap: float = 0.0
    fitness_gap: float = 0.0
    turnover_too_high: bool = False
    primary_fix_target: str = ""
    improvement_priority: list[str] = field(default_factory=list)


@dataclass
class ImprovedVariant:
    expression: str
    mutation_type: str
    mutation_description: str
    expected_effect: str
    priority: int = 5


class NearPassImprover:
    """Targeted improvement engine for near-passing alpha factors."""

    DECAY_WINDOWS = [10, 20, 30, 40, 60]
    NEUTRALIZE_GROUPS = ["sector", "industry", "subindustry", "market"]
    POWER_VALUES = [1.5, 2.0, 2.5, 3.0]
    ZSCORE_WINDOWS = [10, 20, 30, 60]

    OPERATOR_REPLACEMENTS = {
        "rank": ["zscore", "scale", "normalize", "winsorize"],
        "zscore": ["rank", "scale", "normalize"],
        "ts_mean": ["ts_decay_linear", "ts_sum", "ts_av_diff"],
        "ts_decay_linear": ["ts_mean", "ts_sum", "ts_product"],
        "signed_power": [],  # removal candidate
        "group_neutralize": ["group_zscore"],
        "group_zscore": ["group_neutralize"],
        "ts_std_dev": ["ts_mean", "ts_sum"],
        "ts_delta": ["ts_rank", "ts_regression"],
    }

    def __init__(self):
        self._variant_count = 0
        self._llm_generate_fn: Callable[..., Awaitable[Any]] | None = None
        self._tel = AlgorithmTelemetryCollector.get_instance()

    def set_llm_generate_fn(self, fn: Callable[..., Awaitable]) -> None:
        self._llm_generate_fn = fn

    def analyze(self, sharpe: float, fitness: float, turnover: Optional[float] = None,
                checks: Optional[list] = None) -> NearPassAnalysis:
        eid = None
        try:
            eid = self._tel.record_enter_sync("NearPassImprover", cycle_id="unknown", expr_id=hash(str(sharpe) + str(fitness)) % 10000)
            t0 = time.perf_counter()
            analysis = NearPassAnalysis()
            analysis.sharpe_gap = max(0, GATE_SHARPE_MIN - sharpe)
            analysis.fitness_gap = max(0, GATE_FITNESS_MIN - fitness)

            sharpe_pass = sharpe >= GATE_SHARPE_MIN
            fitness_pass = fitness >= GATE_FITNESS_MIN
            sharpe_near = sharpe >= NEAR_PASS_SHARPE_THRESHOLD
            fitness_near = fitness >= NEAR_PASS_FITNESS_THRESHOLD

            if sharpe_pass and not fitness_pass:
                analysis.category = NearPassCategory.SHARPE_GOOD_FITNESS_POOR
                analysis.primary_fix_target = "fitness"
                analysis.turnover_too_high = (turnover or 0) > 25
                analysis.improvement_priority = [
                    "increase_decay_window",
                    "add_double_decay",
                    "replace_rank_with_zscore",
                    "remove_nonlinear_transform",
                    "upgrade_neutralization",
                ]
            elif fitness_pass and not sharpe_pass:
                analysis.category = NearPassCategory.FITNESS_GOOD_SHARPE_POOR
                analysis.primary_fix_target = "sharpe"
                analysis.improvement_priority = [
                    "change_signal_direction",
                    "swap_field_family",
                    "adjust_lookback_window",
                    "add_momentum_term",
                ]
            elif sharpe_near and not fitness_near:
                analysis.category = NearPassCategory.SHARPE_GOOD_FITNESS_POOR
                analysis.primary_fix_target = "fitness"
                analysis.improvement_priority = [
                    "increase_decay_window",
                    "add_double_decay",
                    "replace_rank_with_zscore",
                    "remove_nonlinear_transform",
                    "upgrade_neutralization",
                ]
            elif fitness_near and not sharpe_near:
                analysis.category = NearPassCategory.FITNESS_GOOD_SHARPE_POOR
                analysis.primary_fix_target = "sharpe"
                analysis.improvement_priority = [
                    "change_signal_direction",
                    "swap_field_family",
                    "adjust_lookback_window",
                    "add_momentum_term",
                ]
            elif sharpe_near and fitness_near:
                analysis.category = NearPassCategory.BOTH_NEAR
                analysis.primary_fix_target = "both"
                analysis.improvement_priority = [
                    "increase_decay_window",
                    "tune_parameters",
                    "upgrade_neutralization",
                ]
            else:
                analysis.category = NearPassCategory.NOT_NEAR

            logger.info(
                "[NEAR-PASS] Analysis: category=%s sharpe=%.2f(gap=%.2f) fitness=%.2f(gap=%.2f) target=%s",
                analysis.category.value, sharpe, analysis.sharpe_gap,
                fitness, analysis.fitness_gap, analysis.primary_fix_target,
            )
            ms = (time.perf_counter() - t0) * 1000
            try:
                self._tel.record_exit_sync("NearPassImprover", eid, metrics={"nearpass_category": analysis.category.value, "gaps_found": int(analysis.sharpe_gap > 0) + int(analysis.fitness_gap > 0)}, duration_ms=ms)
            except (OSError, ValueError, RuntimeError):
                pass
            return analysis
        except Exception as e:
            if eid:
                try:
                    self._tel.record_error_sync("NearPassImprover", str(e), type(e).__name__)
                except (OSError, ValueError, RuntimeError):
                    pass
            raise

    @algo_log(label="[NEAR-PASS-LLM].llm_pre_analyze")
    async def llm_pre_analyze(self, expression: str, analysis: NearPassAnalysis) -> NearPassAnalysis:
        eid = None
        try:
            eid = await self._tel.record_enter("NearPassImprover", cycle_id="unknown", expr_id=hash(expression) % 10000)
            t0 = time.perf_counter()

            if self._llm_generate_fn is None:
                logger.info("[NEAR-PASS-LLM] No LLM available, returning original priority order")
                ms = (time.perf_counter() - t0) * 1000
                try:
                    await self._tel.record_exit("NearPassImprover", eid, metrics={"priority_reordered": False}, duration_ms=ms)
                except (OSError, ValueError, RuntimeError):
                    pass
                return analysis

            expr_truncated = expression[:300] + ("..." if len(expression) > 300 else "")
            current_priority = list(analysis.improvement_priority)

            has_decay = "ts_decay_linear" in expression or "ts_decay_exp_window" in expression
            has_signed_power = "signed_power" in expression
            has_rank = re.search(r'\brank\s*\(', expression) is not None
            has_zscore = re.search(r'\bzscore\s*\(', expression) is not None
            has_neutralize = "group_neutralize" in expression or "group_zscore" in expression

            prompt = (
                "You are an alpha factor optimization expert. Reorder the improvement strategies.\n\n"
                f"Expression (truncated): {expr_truncated}\n"
                f"Category: {analysis.category.value}\n"
                f"Primary fix target: {analysis.primary_fix_target}\n"
                f"Sharpe gap: {analysis.sharpe_gap:.3f}, Fitness gap: {analysis.fitness_gap:.3f}\n"
                f"Current priority order: {current_priority}\n\n"
                f"Expression features:\n"
                f"  - Has decay smoothing: {has_decay}\n"
                f"  - Has signed_power (nonlinear): {has_signed_power}\n"
                f"  - Uses rank(): {has_rank}\n"
                f"  - Uses zscore(): {has_zscore}\n"
                f"  - Has neutralization: {has_neutralize}\n\n"
                "Task: Reorder strategies by likelihood of success (most likely FIRST).\n"
                "Return JSON only: {\"reordered\": [\"strategy1\", \"strategy2\", ...], \"reasoning\": \"<brief>\"}\n"
                "Use ONLY strategies from the current_priority list above."
            )

            try:
                response = await asyncio.wait_for(
                    self._llm_generate_fn(prompt),
                    timeout=10.0,
                )
                import json as _json
                parsed = _json.loads(response.strip()) if isinstance(response, str) else response
                reordered = parsed.get("reordered", current_priority)
                reasoning = parsed.get("reasoning", "")

                valid_set = set(current_priority)
                reordered_valid = [s for s in reordered if s in valid_set]
                missing = [s for s in current_priority if s not in reordered_valid]
                final_order = reordered_valid + missing

                result = NearPassAnalysis(
                    category=analysis.category,
                    sharpe_gap=analysis.sharpe_gap,
                    fitness_gap=analysis.fitness_gap,
                    turnover_too_high=analysis.turnover_too_high,
                    primary_fix_target=analysis.primary_fix_target,
                    improvement_priority=final_order,
                )

                logger.info(
                    "[NEAR-PASS-LLM] LLM reordered priority: %s → %s (reason=%s)",
                    current_priority[:3], final_order[:3], reasoning,
                )
                ms = (time.perf_counter() - t0) * 1000
                try:
                    await self._tel.record_exit("NearPassImprover", eid, metrics={"priority_reordered": True}, duration_ms=ms)
                except (OSError, ValueError, RuntimeError):
                    pass
                return result
            except asyncio.TimeoutError:
                logger.warning("[NEAR-PASS-LLM] LLM timed out after 10s, using original order")
                ms = (time.perf_counter() - t0) * 1000
                try:
                    await self._tel.record_exit("NearPassImprover", eid, metrics={"priority_reordered": False}, duration_ms=ms)
                except (OSError, ValueError, RuntimeError):
                    pass
                return analysis
            except (aiohttp.ClientError, ValueError, json.JSONDecodeError) as e:
                logger.warning("[NEAR-PASS-LLM] LLM call failed: %s, using original order", e)
                ms = (time.perf_counter() - t0) * 1000
                try:
                    await self._tel.record_exit("NearPassImprover", eid, metrics={"priority_reordered": False}, duration_ms=ms)
                except (OSError, ValueError, RuntimeError):
                    pass
                return analysis
        except Exception as e:
            if eid:
                try:
                    await self._tel.record_error("NearPassImprover", str(e), type(e).__name__)
                except (OSError, ValueError, RuntimeError):
                    pass
            raise

    def generate_deterministic_variants(
        self,
        expression: str,
        analysis: NearPassAnalysis,
        max_variants: int = MAX_DETERMINISTIC_VARIANTS,
    ) -> list[ImprovedVariant]:
        eid = None
        try:
            eid = self._tel.record_enter_sync("NearPassImprover", cycle_id="unknown", expr_id=hash(expression) % 10000)
            t0 = time.perf_counter()
            variants = []
            self._variant_count = 0

            for strategy_name in analysis.improvement_priority:
                if len(variants) >= max_variants:
                    break
                strategy_method = getattr(self, f"_mutate_{strategy_name}", None)
                if strategy_method:
                    new_variants = strategy_method(expression, analysis)
                    for v in new_variants:
                        if len(variants) >= max_variants:
                            break
                        if v.expression != expression and not any(
                            existing.expression == v.expression for existing in variants
                        ):
                            v.priority = len(variants) + 1
                            variants.append(v)

            logger.info(
                "[NEAR-PASS] Generated %d deterministic variants for target=%s",
                len(variants), analysis.primary_fix_target,
            )
            ms = (time.perf_counter() - t0) * 1000
            try:
                self._tel.record_exit_sync("NearPassImprover", eid, metrics={"variants_count": len(variants)}, duration_ms=ms)
            except (OSError, ValueError, RuntimeError):
                pass
            return variants
        except Exception as e:
            if eid:
                try:
                    self._tel.record_error_sync("NearPassImprover", str(e), type(e).__name__)
                except (OSError, ValueError, RuntimeError):
                    pass
            raise

    def _mutate_increase_decay_window(self, expr: str, analysis: NearPassAnalysis) -> list[ImprovedVariant]:
        variants = []
        for window in self.DECAY_WINDOWS:
            if window <= 10:
                continue
            new_expr = re.sub(
                r"ts_decay_linear\(([^,]+),\s*\d+\)",
                f"ts_decay_linear(\\1, {window})",
                expr,
            )
            if new_expr != expr:
                variants.append(ImprovedVariant(
                    expression=new_expr,
                    mutation_type="decay_window",
                    mutation_description=f"decay window → {window}",
                    expected_effect=f"reduce turnover (~{100//window}%)",
                    priority=1,
                ))
        return variants

    def _mutate_add_double_decay(self, expr: str, analysis: NearPassAnalysis) -> list[ImprovedVariant]:
        variants = []
        outer_windows = [20, 30]
        existing_count = expr.count("ts_decay_linear")
        for w in outer_windows:
            if existing_count >= 2:
                continue
            new_expr = f"ts_decay_linear({expr}, {w})"
            variants.append(ImprovedVariant(
                expression=new_expr,
                mutation_type="double_decay",
                mutation_description=f"wrap with ts_decay_linear(..., {w})",
                expected_effect="double smoothing, significant TO reduction",
                priority=2,
            ))
        return variants

    def _mutate_replace_rank_with_zscore(self, expr: str, analysis: NearPassAnalysis) -> list[ImprovedVariant]:
        variants = []
        replacements = [("rank", "zscore"), ("rank", "scale")]
        for old_op, new_op in replacements:
            if old_op in expr and new_op not in expr:
                pattern = rf"\b{old_op}\b"
                new_expr = re.sub(pattern, new_op, expr, count=1)
                if new_expr != expr:
                    variants.append(ImprovedVariant(
                        expression=new_expr,
                        mutation_type="operator_replace",
                        mutation_description=f"{old_op} → {new_op}",
                        expected_effect="change normalization, may improve stability",
                        priority=3,
                    ))
        return variants

    def _mutate_remove_nonlinear_transform(self, expr: str, analysis: NearPassAnalysis) -> list[ImprovedVariant]:
        variants = []
        if "signed_power" in expr:
            new_expr = re.sub(r"signed_power\(([^,]+),\s*[\d.]+\)", r"\1", expr)
            if new_expr != expr:
                variants.append(ImprovedVariant(
                    expression=new_expr,
                    mutation_type="remove_nonlinear",
                    mutation_description="remove signed_power wrapper",
                    expected_effect="simplify signal, reduce extreme values → lower TO",
                    priority=2,
                ))
        return variants

    def _mutate_upgrade_neutralization(self, expr: str, analysis: NearPassAnalysis) -> list[ImprovedVariant]:
        variants = []
        current_group = None
        for g in self.NEUTRALIZE_GROUPS:
            if f", {g})" in expr or f", {g} )" in expr:
                current_group = g
                break

        if current_group:
            idx = self.NEUTRALIZE_GROUPS.index(current_group)
            for g in self.NEUTRALIZE_GROUPS[idx + 1:idx + 3]:
                new_expr = expr.replace(f", {current_group})", f", {g})")
                if new_expr != expr:
                    variants.append(ImprovedVariant(
                        expression=new_expr,
                        mutation_type="neutralization_upgrade",
                        mutation_description=f"{current_group} → {g}",
                        expected_effect="finer neutralization → potentially better Fitness",
                        priority=4,
                    ))
        return variants

    def _mutate_change_signal_direction(self, expr: str, analysis: NearPassAnalysis) -> list[ImprovedVariant]:
        variants = []
        if expr.startswith("-"):
            variants.append(ImprovedVariant(
                expression=expr[1:],
                mutation_type="signal_direction",
                mutation_description="remove leading negation",
                expected_effect="flip signal direction (may fix negative Sharpe)",
                priority=1,
            ))
        else:
            variants.append(ImprovedVariant(
                expression=f"-({expr})",
                mutation_type="signal_direction",
                mutation_description="add negation wrapper",
                expected_effect="flip signal direction",
                priority=1,
            ))
        return variants

    def _mutate_swap_field_family(self, expr: str, analysis: NearPassAnalysis) -> list[ImprovedVariant]:
        return []

    def _mutate_adjust_lookback_window(self, expr: str, analysis: NearPassAnalysis) -> list[ImprovedVariant]:
        variants = []
        for old_w in [10, 20]:
            for new_w in [5, 15, 30]:
                pattern = rf",\s*{old_w}\)"
                if re.search(pattern, expr):
                    new_expr = re.sub(pattern, f", {new_w})", expr)
                    if new_expr != expr:
                        variants.append(ImprovedVariant(
                            expression=new_expr,
                            mutation_type="lookback_tune",
                            mutation_description=f"lookback {old_w} → {new_w}",
                            expected_effect="adjust signal timing",
                            priority=5,
                        ))
        return variants

    def _mutate_add_momentum_term(self, expr: str, analysis: NearPassAnalysis) -> list[ImprovedVariant]:
        return []

    def _mutate_tune_parameters(self, expr: str, analysis: NearPassAnalysis) -> list[ImprovedVariant]:
        variants = []
        power_pattern = r"signed_power\(([^,]+),\s*([\d.]+)\)"
        match = re.search(power_pattern, expr)
        if match:
            base = match.group(1)
            old_power = float(match.group(2))
            for new_p in [1.5, 2.5, 3.0]:
                if abs(new_p - old_power) > 0.1:
                    new_expr = expr[:match.start()] + f"signed_power({base}, {new_p})" + expr[match.end():]
                    variants.append(ImprovedVariant(
                        expression=new_expr,
                        mutation_type="power_tune",
                        mutation_description=f"power {old_power} → {new_p}",
                        expected_effect="adjust nonlinearity strength",
                        priority=4,
                    ))

        zscore_pattern = r"ts_zscore\(([^,]+),\s*(\d+)\)"
        match = re.search(zscore_pattern, expr)
        if match:
            base = match.group(1)
            old_w = int(match.group(2))
            for new_w in [10, 30, 60]:
                if new_w != old_w:
                    new_expr = expr[:match.start()] + f"ts_zscore({base}, {new_w})" + expr[match.end():]
                    variants.append(ImprovedVariant(
                        expression=new_expr,
                        mutation_type="zscore_window_tune",
                        mutation_description=f"zscore window {old_w} → {new_w}",
                        expected_effect="adjust normalization period",
                        priority=4,
                    ))
        return variants


def get_near_pass_improver() -> NearPassImprover:
    return NearPassImprover()
