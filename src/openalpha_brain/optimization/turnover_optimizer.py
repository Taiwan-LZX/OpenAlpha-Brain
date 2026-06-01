"""Turnover Adaptive Optimization Engine — Strategy C.

Core Formula (WQ Fitness):
  Fitness = Sharpe × √(|Returns| / max(TO, 0.125))

Key Insights:
  1. TO reduction → √(R/TO) increases → Fitness improves (non-linear!)
  2. But too low TO may weaken signal → need optimal sweet spot
  3. Based on Smart Rebalancing paper (FAJ 2024, Arnott et al.)
     - Optimal TO range: 10%~25% (universe dependent)
     - Prioritize strongest position changes, suppress noise trading

Variant Generation Strategies:
  1. Increase decay window (linear search 5→60)
  2. Wrap with hump(x, size) — size ∈ [0.01, 0.03, 0.05, 0.08]
  3. Wrap with trade_when(x, entry, exit) — conditional trigger
  4. Replace ts_decay_linear with ts_mean (smoother)
  5. Nested ts_decay_linear (double smoothing)
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from openalpha_brain.monitoring.algorithm_telemetry import AlgorithmTelemetryCollector
from openalpha_brain.utils.algo_logger import algo_log

logger = logging.getLogger(__name__)

TO_FLOOR = 0.125
TO_OPTIMAL_MIN = 0.10
TO_OPTIMAL_MAX = 0.25
TO_SEARCH_RANGE = (0.05, 0.70)
TO_SEARCH_STEPS = 100

DECAY_WINDOWS = [5, 10, 15, 20, 25, 30, 40, 50, 60]
HUMP_SIZES = [0.01, 0.03, 0.05, 0.08]
TRADE_WHEN_THRESHOLDS = {
    "entry": [0.02, 0.05, 0.10],
    "exit": [-0.02, -0.05, -0.10],
}


class TurnoverSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Variant:
    """Low-turnover variant candidate."""
    expression: str
    strategy: str
    description: str
    expected_to_reduction: float
    confidence: float
    risk_level: str


@dataclass
class TurnoverAnalysisResult:
    """Result of turnover bottleneck analysis."""
    current_to: float
    current_penalty: float
    optimal_to: float
    potential_gain: float
    severity: TurnoverSeverity
    is_bottleneck: bool
    recommended_strategies: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


@dataclass
class TurnoverOptimizationResult:
    """Complete optimization result with variants."""
    original_expression: str
    original_to: float
    analysis: TurnoverAnalysisResult
    variants: list[Variant] = field(default_factory=list)
    summary: str = ""

    def best_variant(self) -> Variant | None:
        if not self.variants:
            return None
        return max(self.variants, key=lambda v: v.expected_to_reduction * v.confidence)

    def safe_variants(self) -> list[Variant]:
        return [v for v in self.variants if v.risk_level in ("low", "medium")]


class TurnoverOptimizer:
    """Turnover adaptive optimization engine based on WQ Fitness formula.

    Design Principles:
      1. Preserve signal direction (never flip sign)
      2. Target TO reduction of 30-50% for high-TO factors
      3. Maintain Sharpe stability (<15% degradation expected)
      4. Generate diverse variants across 5 strategies
      5. Respect WQ operator constraints (66 operators)
    """

    OPERATOR_TURNOVER_IMPACT = {
        "ts_delta": 1.0,
        "ts_regression": 0.9,
        "ts_corr": 0.8,
        "ts_av_diff": 0.7,
        "ts_rank": 0.6,
        "ts_std_dev": 0.5,
        "ts_zscore": 0.4,
        "ts_mean": 0.2,
        "ts_decay_linear": -0.3,
        "ts_decay_exp_window": -0.4,
        "rank": 0.1,
        "zscore": 0.1,
        "signed_power": 0.0,
        "tanh": 0.0,
        "scale": 0.0,
        "group_neutralize": -0.1,
        "group_zscore": -0.1,
        "hump": -0.5,
        "trade_when": -0.6,
    }

    STRATEGY_PRIORITY = {
        "decay_extension": 1,
        "hump_filtering": 2,
        "trade_when_trigger": 3,
        "mean_replacement": 4,
        "double_smoothing": 5,
    }

    def __init__(self, config: dict | None = None):
        self.config = {
            "max_variants": 10,
            "to_threshold": 0.25,
            "min_to_reduction": 0.10,
            "enable_hump": True,
            "enable_trade_when": True,
            "enable_double_smoothing": True,
        }
        if config:
            self.config.update(config)
        self._llm_generate_fn: Callable[..., Awaitable[Any]] | None = None
        self._tel = AlgorithmTelemetryCollector.get_instance()

    def set_llm_generate_fn(self, fn: Callable[..., Awaitable]) -> None:
        self._llm_generate_fn = fn

    def compute_fitness(self, sharpe: float, returns_abs: float, to: float) -> float:
        """Compute WQ Fitness score.

        Args:
            sharpe: Sharpe ratio
            returns_abs: Absolute value of returns (e.g., 0.028 for 2.8%)
            to: Turnover rate (e.g., 0.314 for 31.4%)

        Returns:
            Fitness score
        """
        if to <= 0 or returns_abs <= 0:
            return 0.0
        effective_to = max(to, TO_FLOOR)
        return sharpe * math.sqrt(returns_abs / effective_to)

    def analyze_turnover_bottleneck(
        self,
        expr: str,
        sharpe: float,
        fitness: float,
        turnover: float,
    ) -> dict:
        """Analyze current TO's drag on Fitness.

        Computes:
          1. Current penalty from high TO
          2. Optimal TO that maximizes Fitness in [0.05, 0.70]
          3. Potential Fitness gain if TO optimized

        Args:
            expr: Factor expression
            sharpe: Current Sharpe ratio
            fitness: Current Fitness score
            turnover: Current turnover (decimal, e.g., 0.314)

        Returns:
            Dict with keys: current_penalty, optimal_to, potential_gain, severity, etc.
        """
        if turnover is None or turnover <= 0:
            turnover = 0.01

        inferred_returns = self._infer_returns_from_fitness(sharpe, fitness, turnover)
        current_penalty = self._compute_to_penalty(turnover, inferred_returns)

        optimal_to, max_fitness = self._find_optimal_turnover(
            sharpe, inferred_returns, turnover,
        )

        potential_gain = 0.0
        if fitness > 0 and max_fitness > fitness:
            potential_gain = (max_fitness - fitness) / fitness

        severity = self._classify_severity(turnover, potential_gain)
        is_bottleneck = turnover > self.config["to_threshold"] and potential_gain > 0.10

        strategies = self._select_strategies(expr, turnover, severity)

        result = TurnoverAnalysisResult(
            current_to=turnover,
            current_penalty=current_penalty,
            optimal_to=optimal_to,
            potential_gain=potential_gain,
            severity=severity,
            is_bottleneck=is_bottleneck,
            recommended_strategies=strategies,
            details={
                "inferred_returns": inferred_returns,
                "current_fitness": fitness,
                "optimal_fitness": max_fitness,
                "fitness_gain_absolute": max_fitness - fitness,
                "expression_length": len(expr),
                "operator_count": len(re.findall(r'ts_\w+|rank|zscore|group_\w+', expr)),
            },
        )

        return {
            "current_penalty": current_penalty,
            "optimal_to": optimal_to,
            "potential_gain": potential_gain,
            "severity": severity.value,
            "is_bottleneck": is_bottleneck,
            "recommended_strategies": strategies,
            "inferred_returns": inferred_returns,
            "current_fitness": fitness,
            "optimal_fitness": max_fitness,
            "fitness_gain_absolute": max_fitness - fitness,
            "expression_length": len(expr),
            "operator_count": len(re.findall(r'ts_\w+|rank|zscore|group_\w+', expr)),
            "details": result.details,
        }

    def generate_low_turnover_variants(
        self,
        expr: str,
        turnover: float,
        max_variants: int = 10,
    ) -> list[Variant]:
        eid = None
        try:
            eid = self._tel.record_enter_sync("TurnoverOptimizer", cycle_id="unknown", expr_id=hash(expr) % 10000)
            t0 = time.perf_counter()
            variants = []
            seen_expressions = set()

            strategy_generators = [
                ("decay_extension", self._generate_decay_variants),
                ("hump_filtering", self._generate_hump_variants),
                ("trade_when_trigger", self._generate_trade_when_variants),
                ("mean_replacement", self._generate_mean_variants),
                ("double_smoothing", self._generate_double_smoothing_variants),
            ]

            for strategy_name, generator in strategy_generators:
                try:
                    new_variants = generator(expr, turnover)
                    for v in new_variants:
                        if v.expression not in seen_expressions and v.expression != expr:
                            seen_expressions.add(v.expression)
                            variants.append(v)
                except (OSError, ValueError, RuntimeError) as e:
                    pass

            variants.sort(key=lambda v: v.expected_to_reduction * v.confidence, reverse=True)
            result = variants[:max_variants]
            ms = (time.perf_counter() - t0) * 1000
            try:
                self._tel.record_exit_sync("TurnoverOptimizer", eid, metrics={"strategy_used": "multi_strategy"}, duration_ms=ms)
            except (OSError, ValueError, RuntimeError):
                pass
            return result
        except Exception as e:
            if eid:
                try:
                    self._tel.record_error_sync("TurnoverOptimizer", str(e), type(e).__name__)
                except (OSError, ValueError, RuntimeError):
                    pass
            raise

    @algo_log(label="[TO-OPT-LLM].llm_review_variants")
    async def llm_review_variants(
        self,
        variants: list[Variant],
        original_expression: str,
    ) -> list[Variant]:
        eid = None
        try:
            eid = await self._tel.record_enter("TurnoverOptimizer", cycle_id="unknown", expr_id=hash(original_expression) % 10000)
            t0 = time.perf_counter()
            llm_called = False

            if self._llm_generate_fn is None or not variants:
                logger.info("[TO-OPT-LLM] No LLM available or empty variants, returning original order")
                ms = (time.perf_counter() - t0) * 1000
                try:
                    await self._tel.record_exit("TurnoverOptimizer", eid, metrics={"llm_called": llm_called, "reranked_count": 0}, duration_ms=ms)
                except (OSError, ValueError, RuntimeError):
                    pass
                return variants

            expr_truncated = original_expression[:300] + ("..." if len(original_expression) > 300 else "")
            variant_summaries = []
            for i, v in enumerate(variants):
                variant_summaries.append(
                    f"  [{i}] {v.expression[:200]} (strategy={v.strategy}, ΔTO≈{v.expected_to_reduction:.0%}, risk={v.risk_level})",
                )

            prompt = (
                "You are a quantitative factor optimization expert reviewing turnover-reduction variants.\n\n"
                f"Original expression: {expr_truncated}\n\n"
                f"Generated {len(variants)} low-turnover variants:\n" +
                "\n".join(variant_summaries) + "\n\n"
                "Task: Judge which variants best preserve the original signal logic while reducing turnover.\n"
                "Return JSON only:\n"
                '  {"ranked_indices": [3, 1, 0, 2, ...], "confidence_scores": [0.9, 0.8, 0.6, 0.5, ...], "reasoning": "<brief>"}\n'
                "- ranked_indices: indices of variants sorted by quality (best first)\n"
                "- confidence_scores: per-variant confidence that it preserves signal (0.0-1.0)\n"
                "- Include ALL indices exactly once."
            )

            response = await asyncio.wait_for(
                self._llm_generate_fn(prompt),
                timeout=15.0,
            )
            import json as _json
            parsed = _json.loads(response.strip()) if isinstance(response, str) else response
            ranked_indices = parsed.get("ranked_indices", list(range(len(variants))))
            confidence_scores = parsed.get("confidence_scores", [v.confidence for v in variants])
            reasoning = parsed.get("reasoning", "")
            llm_called = True

            seen = set()
            ordered_variants = []
            for idx in ranked_indices:
                if 0 <= idx < len(variants) and idx not in seen:
                    seen.add(idx)
                    conf = confidence_scores[idx] if idx < len(confidence_scores) else variants[idx].confidence
                    v = variants[idx]
                    ordered_variants.append(Variant(
                        expression=v.expression,
                        strategy=v.strategy,
                        description=v.description,
                        expected_to_reduction=v.expected_to_reduction,
                        confidence=conf,
                        risk_level=v.risk_level,
                    ))

            for i, v in enumerate(variants):
                if i not in seen:
                    ordered_variants.append(v)

            logger.info(
                "[TO-OPT-LLM] LLM reranked %d variants (top=%s, reason=%s)",
                len(ordered_variants),
                ordered_variants[0].strategy if ordered_variants else "N/A",
                reasoning,
            )
            ms = (time.perf_counter() - t0) * 1000
            try:
                await self._tel.record_exit("TurnoverOptimizer", eid, metrics={"llm_called": llm_called, "reranked_count": len(ordered_variants)}, duration_ms=ms)
            except (OSError, ValueError, RuntimeError):
                pass
            return ordered_variants
        except TimeoutError:
            logger.warning("[TO-OPT-LLM] LLM timed out after 15s, returning original order")
            ms = (time.perf_counter() - t0) * 1000 if 't0' in dir() else 0
            try:
                await self._tel.record_exit("TurnoverOptimizer", eid, metrics={"llm_called": False, "reranked_count": 0}, duration_ms=ms)
            except (OSError, ValueError, RuntimeError):
                pass
            return variants
        except (aiohttp.ClientError, ValueError, json.JSONDecodeError) as e:
            if eid:
                try:
                    await self._tel.record_error("TurnoverOptimizer", str(e), type(e).__name__)
                except (OSError, ValueError, RuntimeError):
                    pass
            raise

    def estimate_fitness_gain(
        self,
        current_to: float,
        target_to: float,
        sharpe: float,
        returns_estimate: float,
    ) -> float:
        """Estimate Fitness improvement from TO reduction.

        Uses the exact WQ Fitness formula:
          Fitness = Sharpe × √(|Returns| / max(TO, 0.125))

        Args:
            current_to: Current turnover
            target_to: Target turnover after optimization
            sharpe: Sharpe ratio (assumed stable)
            returns_estimate: Estimated absolute returns

        Returns:
            Expected Fitness gain ratio (e.g., 0.25 means +25% improvement)
        """
        if current_to <= 0 or target_to <= 0:
            return 0.0

        current_fitness = self.compute_fitness(sharpe, returns_estimate, current_to)
        target_fitness = self.compute_fitness(sharpe, returns_estimate, target_to)

        if current_fitness <= 0:
            return 0.0

        gain = (target_fitness - current_fitness) / current_fitness
        return max(gain, 0.0)

    def _infer_returns_from_fitness(
        self, sharpe: float, fitness: float, turnover: float,
    ) -> float:
        """Infer absolute returns from known Fitness formula components."""
        if sharpe == 0 or fitness == 0:
            return 0.028
        effective_to = max(turnover, TO_FLOOR)
        ratio = (fitness / sharpe) ** 2
        return ratio * effective_to

    def _compute_to_penalty(self, to: float, returns_abs: float) -> float:
        """Compute penalty factor from current TO vs optimal."""
        if to <= TO_OPTIMAL_MAX:
            return 0.0

        optimal_fitness = self.compute_fitness(1.0, returns_abs, TO_OPTIMAL_MAX)
        current_fitness = self.compute_fitness(1.0, returns_abs, to)

        if optimal_fitness <= 0:
            return 0.0

        return 1.0 - (current_fitness / optimal_fitness)

    def _find_optimal_turnover(
        self, sharpe: float, returns_abs: float, current_to: float,
    ) -> tuple[float, float]:
        """Find TO that maximizes Fitness in search range.

        Returns:
            (optimal_to, max_fitness)
        """
        best_to = current_to
        best_fitness = self.compute_fitness(sharpe, returns_abs, current_to)

        step = (TO_SEARCH_RANGE[1] - TO_SEARCH_RANGE[0]) / TO_SEARCH_STEPS

        for i in range(TO_SEARCH_STEPS + 1):
            test_to = TO_SEARCH_RANGE[0] + i * step
            test_fitness = self.compute_fitness(sharpe, returns_abs, test_to)

            if test_fitness > best_fitness:
                best_fitness = test_fitness
                best_to = test_to

        best_to = max(best_to, TO_OPTIMAL_MIN)
        return best_to, best_fitness

    def _classify_severity(self, to: float, potential_gain: float) -> TurnoverSeverity:
        """Classify turnover problem severity."""
        if to > 0.50 or potential_gain > 0.50:
            return TurnoverSeverity.CRITICAL
        if to > 0.35 or potential_gain > 0.30:
            return TurnoverSeverity.HIGH
        if to > 0.25 or potential_gain > 0.15:
            return TurnoverSeverity.MEDIUM
        return TurnoverSeverity.LOW

    def _select_strategies(
        self, expr: str, to: float, severity: TurnoverSeverity,
    ) -> list[str]:
        """Select appropriate optimization strategies based on analysis."""
        strategies = []

        has_decay = "ts_decay_linear" in expr or "ts_decay_exp_window" in expr
        has_high_freq_ops = any(
            op in expr for op in ["ts_delta", "ts_regression", "ts_corr"]
        )

        if has_decay:
            strategies.append("decay_extension")

        if self.config.get("enable_hump", True) and severity in (
            TurnoverSeverity.HIGH,
            TurnoverSeverity.CRITICAL,
        ):
            strategies.append("hump_filtering")

        if self.config.get("enable_trade_when", True) and has_high_freq_ops:
            strategies.append("trade_when_trigger")

        if not has_decay or severity in (TurnoverSeverity.HIGH, TurnoverSeverity.CRITICAL):
            strategies.append("mean_replacement")

        if self.config.get("enable_double_smoothing", True) and to > 0.35:
            strategies.append("double_smoothing")

        return strategies

    def _generate_decay_variants(self, expr: str, turnover: float) -> list[Variant]:
        """Strategy 1: Extend/increase decay window."""
        variants = []

        decay_match = re.search(r'ts_decay_linear\(([^,]+),\s*(\d+)\)', expr)
        if not decay_match:
            inner_expr = decay_match.group(1) if decay_match else expr
            current_window = int(decay_match.group(2)) if decay_match else 0

            for new_window in DECAY_WINDOWS:
                if new_window > current_window and current_window > 0:
                    new_expr = f"ts_decay_linear({inner_expr}, {new_window})"
                    reduction = min(0.4, (new_window - current_window) / current_window * 0.3)
                    variants.append(Variant(
                        expression=new_expr,
                        strategy="decay_extension",
                        description=f"Increase decay window {current_window}→{new_window}",
                        expected_to_reduction=reduction,
                        confidence=0.85 if new_window <= 30 else 0.7,
                        risk_level="low",
                    ))
        else:
            inner = decay_match.group(1)
            current_win = int(decay_match.group(2))

            for new_win in DECAY_WINDOWS:
                if new_win != current_win:
                    new_expr = re.sub(
                        r'ts_decay_linear\([^,]+,\s*\d+\)',
                        f'ts_decay_linear({inner}, {new_win})',
                        expr,
                        count=1,
                    )
                    if new_win > current_win:
                        reduction = min(0.4, (new_win - current_win) / max(current_win, 1) * 0.3)
                    else:
                        reduction = -0.1

                    variants.append(Variant(
                        expression=new_expr,
                        strategy="decay_extension",
                        description=f"Adjust decay window {current_win}→{new_win}",
                        expected_to_reduction=max(0, reduction),
                        confidence=0.85 if 10 <= new_win <= 30 else 0.65,
                        risk_level="low",
                    ))

        return variants

    def _generate_hump_variants(self, expr: str, turnover: float) -> list[Variant]:
        """Strategy 2: Wrap with hump() filter."""
        if not self.config.get("enable_hump", True):
            return []

        variants = []
        for size in HUMP_SIZES:
            new_expr = f"hump({expr}, {size})"
            reduction = 0.25 + (size * 2)
            variants.append(Variant(
                expression=new_expr,
                strategy="hump_filtering",
                description=f"Apply hump filter (size={size})",
                expected_to_reduction=min(reduction, 0.45),
                confidence=0.75,
                risk_level="medium",
            ))

        return variants

    def _generate_trade_when_variants(self, expr: str, turnover: float) -> list[Variant]:
        """Strategy 3: Wrap with trade_when() conditional trigger."""
        if not self.config.get("enable_trade_when", True):
            return []

        variants = []
        for entry in TRADE_WHEN_THRESHOLDS["entry"]:
            for exit_th in TRADE_WHEN_THRESHOLDS["exit"]:
                new_expr = f"trade_when({expr}, {entry}, {exit_th})"
                reduction = 0.35 + abs(entry) + abs(exit_th)
                variants.append(Variant(
                    expression=new_expr,
                    strategy="trade_when_trigger",
                    description=f"Conditional rebalance (entry={entry}, exit={exit_th})",
                    expected_to_reduction=min(reduction, 0.55),
                    confidence=0.65,
                    risk_level="medium",
                ))

        return variants[:6]

    def _generate_mean_variants(self, expr: str, turnover: float) -> list[Variant]:
        """Strategy 4: Replace ts_decay_linear with ts_mean (smoother)."""
        variants = []

        decay_matches = list(re.finditer(r'ts_decay_linear\(([^,]+),\s*(\d+)\)', expr))
        if not decay_matches:
            for window in [10, 20, 30]:
                new_expr = f"ts_mean({expr}, {window})"
                variants.append(Variant(
                    expression=new_expr,
                    strategy="mean_replacement",
                    description=f"Wrap with ts_mean(window={window})",
                    expected_to_reduction=0.20,
                    confidence=0.70,
                    risk_level="low",
                ))
        else:
            for match in decay_matches[:2]:
                inner = match.group(1)
                old_window = int(match.group(2))

                windows_to_try = list(set([max(old_window, 10), max(old_window, 20), old_window + 10]))
                for new_window in windows_to_try[:3]:
                    new_expr = expr[:match.start()] + f"ts_mean({inner}, {new_window})" + expr[match.end():]
                    reduction = 0.15 + (new_window - old_window) / max(old_window, 1) * 0.1
                    variants.append(Variant(
                        expression=new_expr,
                        strategy="mean_replacement",
                        description=f"Replace decay_linear with ts_mean({new_window})",
                        expected_to_reduction=min(reduction, 0.35),
                        confidence=0.72,
                        risk_level="low",
                    ))

        return variants

    def _generate_double_smoothing_variants(self, expr: str, turnover: float) -> list[Variant]:
        """Strategy 5: Nested ts_decay_linear (double smoothing)."""
        if not self.config.get("enable_double_smoothing", True):
            return []

        variants = []
        has_existing_decay = "ts_decay_linear" in expr

        if has_existing_decay:
            outer_windows = [10, 15, 20]
        else:
            outer_windows = [5, 10, 15]

        for outer_w in outer_windows:
            new_expr = f"ts_decay_linear({expr}, {outer_w})"
            reduction = 0.30 if has_existing_decay else 0.25
            variants.append(Variant(
                expression=new_expr,
                strategy="double_smoothing",
                description=f"Add outer decay layer (window={outer_w})",
                expected_to_reduction=reduction,
                confidence=0.68,
                risk_level="medium",
            ))

        return variants

    def optimize(
        self,
        expr: str,
        sharpe: float,
        fitness: float,
        turnover: float,
        max_variants: int = 10,
        context: dict | None = None,
    ) -> TurnoverOptimizationResult:
        eid = None
        try:
            eid = self._tel.record_enter_sync("TurnoverOptimizer", cycle_id="unknown", expr_id=hash(expr) % 10000)
            t0 = time.perf_counter()
            _ctx = context or {}
            composite_factors = _ctx.get("llm_composite_factors", [])
            if composite_factors:
                logger.info(
                    "[TURNOVER-OPT] Context received: root_cause=%.60s composite_factors=%s",
                    _ctx.get("llm_root_cause", "")[:60],
                    composite_factors,
                )
            analysis_dict = self.analyze_turnover_bottleneck(
                expr, sharpe, fitness, turnover,
            )

            analysis = TurnoverAnalysisResult(
                current_to=analysis_dict.get("current_to", turnover),
                current_penalty=analysis_dict.get("current_penalty", 0),
                optimal_to=analysis_dict.get("optimal_to", turnover),
                potential_gain=analysis_dict.get("potential_gain", 0),
                severity=TurnoverSeverity(analysis_dict.get("severity", "low")),
                is_bottleneck=analysis_dict.get("is_bottleneck", False),
                recommended_strategies=analysis_dict.get("recommended_strategies", []),
                details=analysis_dict.get("details", {}),
            )

            variants = []
            if analysis.is_bottleneck or turnover > self.config["to_threshold"]:
                variants = self.generate_low_turnover_variants(
                    expr, turnover, max_variants,
                )

            safe = [v for v in variants if v.risk_level in ("low", "medium")]
            summary = self._build_summary(analysis, variants)

            result = TurnoverOptimizationResult(
                original_expression=expr,
                original_to=turnover,
                analysis=analysis,
            variants=variants,
            summary=summary,
        )
            ms = (time.perf_counter() - t0) * 1000
            try:
                self._tel.record_exit_sync("TurnoverOptimizer", eid, metrics={"original_to": round(turnover, 4), "optimized_to": round(analysis.optimal_to, 4), "variants_generated": len(variants)}, duration_ms=ms)
            except (OSError, ValueError, RuntimeError):
                pass
            return result
        except Exception as e:
            if eid:
                try:
                    self._tel.record_error_sync("TurnoverOptimizer", str(e), type(e).__name__)
                except (OSError, ValueError, RuntimeError):
                    pass
            raise

    def _build_summary(
        self, analysis: TurnoverAnalysisResult, variants: list[Variant],
    ) -> str:
        """Build human-readable optimization summary."""
        parts = [
            f"TO Analysis: current={analysis.current_to:.1%}, "
            f"optimal={analysis.optimal_to:.1%}, "
            f"severity={analysis.severity.value}",
            f"Potential Fitness Gain: {analysis.potential_gain:+.1%}",
            f"Variants Generated: {len(variants)} ({len([v for v in variants if v.risk_level=='low'])} low-risk)",
        ]

        if variants:
            best = max(variants, key=lambda v: v.expected_to_reduction * v.confidence)
            parts.append(f"Best Variant: [{best.strategy}] ΔTO≈{best.expected_to_reduction:.0%} (conf={best.confidence:.0%})")

        return " | ".join(parts)


def get_turnover_optimizer(config: dict | None = None) -> TurnoverOptimizer:
    """Factory function for TurnoverOptimizer."""
    return TurnoverOptimizer(config=config)
