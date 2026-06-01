"""Fitness Boost Engine — Targeted optimization for WQ BRAIN Fitness score.

Root Cause Analysis (from E2E data):
  - Best factor: Sharpe=1.300, Fitness=0.690, TO=31.4%
  - Inferred Returns ~2.8% (too low for the Sharpe level)
  - Formula: Fitness ≈ Sharpe × √|Returns| / max(TO, 0.125)

Core Insight: High Sharpe + Low Fitness = "efficient but weak signal"
  → Need to amplify signal magnitude WITHOUT destabilizing

Optimization Strategies (5 tiers):
  Tier 1 — Decay Calibration: optimal window search
  Tier 2 — Signal Amplification: power/scale tuning
  Tier 3 — Neutralization Balance: group granularity tuning
  Tier 4 — Structure Streamlining: remove fitness-killing patterns
  Tier 5 — Composite Mutations: multi-parameter coordinated changes

Integration:
  - Called by FeedbackOrchestrator when SHARPE_GOOD_FITNESS_POOR detected
  - Works alongside NearPassImprover (complementary, not replacement)
  - Produces variants with TIER_1_HIGH_IMPROVED priority
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from openalpha_brain.monitoring.algorithm_telemetry import AlgorithmTelemetryCollector
from openalpha_brain.utils.algo_logger import algo_log

logger = logging.getLogger(__name__)

GATE_FITNESS_MIN = 1.0
GATE_SHARPE_MIN = 1.25

FITNESS_DECAY_WINDOWS = [5, 10, 15, 20, 30, 40, 60]
FITNESS_POWER_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0]
FITNESS_ZSCORE_WINDOWS = [5, 10, 20, 30, 60]
FITNESS_SCALE_RANGE = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]

LOW_TURNOVER_HUMP_SIZES = [0.01, 0.03, 0.05, 0.08]
LOW_TURNOVER_ENTRY_THRESHOLDS = [0.5, 1.0, 1.5]
LOW_TURNOVER_EXIT_THRESHOLDS = [-0.5, -0.3, -0.1]


class FitnessBoostTier(Enum):
    DECAY_CALIBRATION = "decay_calibration"
    SIGNAL_AMPLIFICATION = "signal_amplification"
    NEUTRALIZATION_BALANCE = "neutralization_balance"
    STRUCTURE_STREAMLINING = "structure_streamlining"
    COMPOSITE_MUTATION = "composite_mutation"
    LOW_TURNOVER_WRAPPING = "low_turnover_wrapping"


@dataclass
class FitnessVariant:
    expression: str
    boost_tier: str
    mutation_description: str
    expected_fitness_delta: float
    risk_level: str  # "low" | "medium" | "high"
    priority: int = 5


@dataclass
class FitnessBoostResult:
    original_expression: str
    original_sharpe: float
    original_fitness: float
    variants: list[FitnessVariant] = field(default_factory=list)
    analysis_summary: str = ""

    def best_variant(self) -> FitnessVariant | None:
        if not self.variants:
            return None
        return max(self.variants, key=lambda v: v.expected_fitness_delta)


class FitnessBoostEngine:
    """Targeted engine for boosting WQ Fitness score while preserving Sharpe.

    Design Principles:
      1. Preserve signal direction (never flip sign)
      2. Minimize Sharpe degradation (target <10% drop)
      3. Maximize Fitness gain (target >30% improvement)
      4. Stay within WQ operator constraints (66 operators)
      5. Respect ThreeBlockTemplate structure (A🟢/B🔒/C🔒)
    """

    OPERATOR_FITNESS_IMPACT = {
        "signed_power": -0.15,
        "ts_decay_linear": 0.10,
        "ts_decay_exp_window": 0.08,
        "group_neutralize": 0.05,
        "group_zscore": 0.03,
        "rank": 0.02,
        "zscore": 0.05,
        "scale": 0.03,
        "normalize": 0.02,
        "ts_std_dev": -0.05,
        "ts_av_diff": -0.03,
        "ts_regression": -0.08,
        "ts_corr": -0.10,
        "ts_mean": 0.00,
        "ts_sum": -0.02,
        "ts_product": -0.12,
        "winsorize": 0.04,
        "truncate": 0.04,
        "hump": 0.12,
        "trade_when": 0.15,
    }

    def __init__(self):
        self._variant_count = 0
        self._llm_generate_fn: Callable[..., Awaitable[Any]] | None = None
        self._tel = AlgorithmTelemetryCollector.get_instance()

    def set_llm_generate_fn(self, fn: Callable[..., Awaitable]) -> None:
        self._llm_generate_fn = fn

    def analyze_fitness_bottleneck(
        self,
        expression: str,
        sharpe: float,
        fitness: float,
        turnover: float | None = None,
    ) -> dict:
        eid = None
        try:
            eid = self._tel.record_enter_sync("FitnessBoost", cycle_id="unknown", expr_id=hash(expression) % 10000)
            t0 = time.perf_counter()
            features = self._extract_expression_features(expression)
            fitness_gap = GATE_FITNESS_MIN - fitness
            sharpe_ratio = sharpe / GATE_SHARPE_MIN if GATE_SHARPE_MIN > 0 else 0

            bottleneck = "complex"
            severity = min(1.0, fitness_gap / GATE_FITNESS_MIN)
            recommended_tiers = []

            has_strong_decay = features.get("decay_count", 0) >= 2 or features.get("max_decay_window", 0) >= 40
            has_signed_power = "signed_power" in features.get("operators", [])
            has_complex_ops = features.get("complex_op_count", 0) >= 3
            high_turnover = (turnover or 0) > 40
            fine_neutralize = features.get("neutralize_group", "") in ("subindustry", "sw")

            if sharpe >= 1.2 and fitness < 0.8:
                bottleneck = "weak_signal"
                if has_strong_decay:
                    bottleneck = "over_smoothed"
                elif fine_neutralize and not has_signed_power:
                    bottleneck = "over_neutralized"

            if has_signed_power and fitness < 0.75:
                recommended_tiers.extend(
                    [
                        FitnessBoostTier.SIGNAL_AMPLIFICATION,
                        FitnessBoostTier.STRUCTURE_STREAMLINING,
                    ]
                )
            if has_strong_decay:
                recommended_tiers.append(FitnessBoostTier.DECAY_CALIBRATION)
            if fine_neutralize:
                recommended_tiers.append(FitnessBoostTier.NEUTRALIZATION_BALANCE)
            if has_complex_ops:
                recommended_tiers.append(FitnessBoostTier.STRUCTURE_STREAMLINING)
            if high_turnover:
                recommended_tiers.append(FitnessBoostTier.LOW_TURNOVER_WRAPPING)
            if not recommended_tiers:
                recommended_tiers = [
                    FitnessBoostTier.DECAY_CALIBRATION,
                    FitnessBoostTier.SIGNAL_AMPLIFICATION,
                    FitnessBoostTier.COMPOSITE_MUTATION,
                ]

            result = {
                "bottleneck_type": bottleneck,
                "severity": severity,
                "fitness_gap": fitness_gap,
                "sharpe_quality": "good" if sharpe_ratio >= 1.0 else "acceptable" if sharpe_ratio >= 0.9 else "weak",
                "recommended_tiers": recommended_tiers,
                "expression_features": features,
            }

            logger.info(
                "[FITNESS-BOOST] Analysis: bottleneck=%s severity=%.2f gap=%.2f tiers=%s",
                bottleneck,
                severity,
                fitness_gap,
                [t.value for t in recommended_tiers],
            )
            ms = (time.perf_counter() - t0) * 1000
            with contextlib.suppress(OSError, ValueError, RuntimeError):
                self._tel.record_exit_sync("FitnessBoost", eid, metrics={"bottleneck_type": bottleneck}, duration_ms=ms)
            return result
        except Exception as e:
            if eid:
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    self._tel.record_error_sync("FitnessBoost", str(e), type(e).__name__)
            raise

    @algo_log(label="[FIT-BOOST-LLM].analyze_fitness_bottleneck_llm")
    async def analyze_fitness_bottleneck_llm(
        self,
        expression: str,
        wq_metrics: dict,
        rule_analysis: dict | None = None,
    ) -> dict:
        eid = None
        try:
            eid = await self._tel.record_enter("FitnessBoost", cycle_id="unknown", expr_id=hash(expression) % 10000)
            t0 = time.perf_counter()
            llm_called = False

            if self._llm_generate_fn is None:
                logger.info("[FIT-BOOST-LLM] No LLM available, falling back to rule-based")
                result = self.analyze_fitness_bottleneck(
                    expression,
                    wq_metrics.get("sharpe", 0),
                    wq_metrics.get("fitness", 0),
                    wq_metrics.get("turnover"),
                )
                ms = (time.perf_counter() - t0) * 1000
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    await self._tel.record_exit("FitnessBoost", eid, metrics={"llm_called": llm_called}, duration_ms=ms)
                return result

            sharpe = wq_metrics.get("sharpe", 0) or 0
            fitness = wq_metrics.get("fitness", 0) or 0
            turnover = wq_metrics.get("turnover") or 0
            expr_truncated = expression[:300] + ("..." if len(expression) > 300 else "")

            if rule_analysis is None:
                rule_analysis = self.analyze_fitness_bottleneck(expression, sharpe, fitness, turnover)

            bottleneck_types = []
            if rule_analysis.get("bottleneck_type"):
                bottleneck_types.append(rule_analysis["bottleneck_type"])
            if len(rule_analysis.get("recommended_tiers", [])) >= 2:
                for t in rule_analysis["recommended_tiers"]:
                    if t.value not in bottleneck_types:
                        bottleneck_types.append(t.value)

            if len(bottleneck_types) < 2:
                logger.info(
                    "[FIT-BOOST-LLM] Single bottleneck detected (%s), skipping LLM",
                    bottleneck_types[0] if bottleneck_types else "none",
                )
                result = dict(rule_analysis)
                result["llm_prioritized"] = False
                ms = (time.perf_counter() - t0) * 1000
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    await self._tel.record_exit("FitnessBoost", eid, metrics={"llm_called": llm_called}, duration_ms=ms)
                return result

            prompt = (
                "You are a quantitative factor analysis expert. Analyze this alpha factor's fitness bottleneck.\n\n"
                f"Expression (truncated): {expr_truncated}\n"
                f"Sharpe Ratio: {sharpe:.3f}\n"
                f"Fitness Score: {fitness:.3f}\n"
                f"Turnover: {turnover:.1%}\n"
                f"Rule-based bottlenecks detected: {bottleneck_types}\n\n"
                "Task: Given MULTIPLE simultaneous bottlenecks, decide which ONE to fix FIRST.\n"
                'Respond with JSON only: {"primary_bottleneck": "<type>", "reasoning": "<1 sentence>"}\n'
                "Valid types: weak_signal, over_smoothed, over_neutralized, complex"
            )

            try:
                response = await asyncio.wait_for(
                    self._llm_generate_fn(prompt),
                    timeout=12.0,
                )
                import json as _json

                parsed = _json.loads(response.strip()) if isinstance(response, str) else response
                primary = parsed.get("primary_bottleneck", rule_analysis["bottleneck_type"])
                reasoning = parsed.get("reasoning", "")
                llm_called = True

                result = dict(rule_analysis)
                result["bottleneck_type"] = primary
                result["llm_prioritized"] = True
                result["llm_reasoning"] = reasoning

                logger.info(
                    "[FIT-BOOST-LLM] LLM prioritized bottleneck=%s reason=%s (was %s)",
                    primary,
                    reasoning,
                    rule_analysis["bottleneck_type"],
                )
                ms = (time.perf_counter() - t0) * 1000
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    await self._tel.record_exit("FitnessBoost", eid, metrics={"llm_called": llm_called}, duration_ms=ms)
                return result
            except TimeoutError:
                logger.warning("[FIT-BOOST-LLM] LLM timed out after 12s, using rule-based result")
                result = dict(rule_analysis)
                result["llm_prioritized"] = False
                ms = (time.perf_counter() - t0) * 1000
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    await self._tel.record_exit("FitnessBoost", eid, metrics={"llm_called": llm_called}, duration_ms=ms)
                return result
            except (ValueError, TypeError, OSError, RuntimeError) as e:
                logger.warning("[FIT-BOOST-LLM] LLM call failed: %s, using rule-based result", e)
                result = dict(rule_analysis)
                result["llm_prioritized"] = False
                ms = (time.perf_counter() - t0) * 1000
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    await self._tel.record_exit("FitnessBoost", eid, metrics={"llm_called": llm_called}, duration_ms=ms)
                return result
        except Exception as e:
            if eid:
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    await self._tel.record_error("FitnessBoost", str(e), type(e).__name__)
            raise

    @algo_log(label="[FIT-BOOST-VARIANT].generate_fitness_variants")
    def generate_boost_variants(
        self,
        expression: str,
        sharpe: float,
        fitness: float,
        turnover: float | None = None,
        max_variants: int = 15,
        context: dict | None = None,
    ) -> FitnessBoostResult:
        eid = None
        try:
            eid = self._tel.record_enter_sync("FitnessBoost", cycle_id="unknown", expr_id=hash(expression) % 10000)
            t0 = time.perf_counter()
            _ctx = context or {}
            analysis = self.analyze_fitness_bottleneck(expression, sharpe, fitness, turnover)
            result = FitnessBoostResult(
                original_expression=expression,
                original_sharpe=sharpe,
                original_fitness=fitness,
                analysis_summary=f"{analysis['bottleneck_type']} (severity={analysis['severity']:.2f})",
            )

            if not expression or len(expression.strip()) < 3:
                ms = (time.perf_counter() - t0) * 1000
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    self._tel.record_exit_sync(
                        "FitnessBoost", eid, metrics={"variants_count": 0, "tier_used": "none"}, duration_ms=ms
                    )
                return result

            composite_factors = _ctx.get("llm_composite_factors", [])
            recommended_tiers = list(analysis["recommended_tiers"])
            if composite_factors:
                skip_tiers = set()
                if any(f in ("over_neutralization", "signal_killed") for f in composite_factors):
                    skip_tiers.add(FitnessBoostTier.NEUTRALIZATION_BALANCE)
                if skip_tiers:
                    recommended_tiers = [t for t in recommended_tiers if t not in skip_tiers]
                    logger.info(
                        "[FITNESS-BOOST] Context-aware tier filter: skipping %s due to composite_factors=%s",
                        [t.name for t in skip_tiers],
                        composite_factors,
                    )

            tier_methods = {
                FitnessBoostTier.DECAY_CALIBRATION: self._tier_decay_calibration,
                FitnessBoostTier.SIGNAL_AMPLIFICATION: self._tier_signal_amplification,
                FitnessBoostTier.NEUTRALIZATION_BALANCE: self._tier_neutralization_balance,
                FitnessBoostTier.STRUCTURE_STREAMLINING: self._tier_structure_streamlining,
                FitnessBoostTier.COMPOSITE_MUTATION: self._tier_composite_mutation,
                FitnessBoostTier.LOW_TURNOVER_WRAPPING: self._tier_low_turnover_wrapping,
            }

            top_tier = recommended_tiers[0].value if recommended_tiers else "none"

            for tier in recommended_tiers:
                if len(result.variants) >= max_variants:
                    break
                method = tier_methods.get(tier)
                if method:
                    new_variants = method(expression, analysis)
                    for v in new_variants:
                        if len(result.variants) >= max_variants:
                            break
                        if v.expression != expression and not any(
                            existing.expression == v.expression for existing in result.variants
                        ):
                            v.priority = len(result.variants) + 1
                            result.variants.append(v)

            result.variants.sort(key=lambda v: (-v.expected_fitness_delta, v.priority))

            logger.info(
                "[FITNESS-BOOST] Generated %d variants for fitness %.3f→? (gap=%.2f)",
                len(result.variants),
                fitness,
                analysis["fitness_gap"],
            )
            ms = (time.perf_counter() - t0) * 1000
            with contextlib.suppress(OSError, ValueError, RuntimeError):
                self._tel.record_exit_sync(
                    "FitnessBoost",
                    eid,
                    metrics={"variants_count": len(result.variants), "tier_used": top_tier},
                    duration_ms=ms,
                )
            return result
        except Exception as e:
            if eid:
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    self._tel.record_error_sync("FitnessBoost", str(e), type(e).__name__)
            raise

    def _extract_expression_features(self, expr: str) -> dict:
        """Extract structural features from expression for bottleneck analysis."""
        operators = re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", expr)
        decay_windows = self._extract_decay_windows(expr)
        power_value = self._extract_signed_power_value(expr)
        neutralize_match = re.search(r"group_neutralize\([^,]+,\s*([^)]+)\)", expr)

        complex_ops = ["ts_regression", "ts_corr", "ts_covariance", "ts_product"]
        complex_count = sum(1 for op in operators if op in complex_ops)

        return {
            "operators": operators,
            "operator_count": len(operators),
            "decay_count": len(decay_windows),
            "decay_windows": decay_windows,
            "max_decay_window": max(decay_windows) if decay_windows else 0,
            "has_signed_power": power_value is not None,
            "power_value": power_value,
            "neutralize_group": neutralize_match.group(1).strip() if neutralize_match else None,
            "complex_op_count": complex_count,
            "expression_length": len(expr),
            "nesting_depth": expr.count("("),
        }

    @staticmethod
    def _extract_signed_power_value(expr: str) -> float | None:
        """Extract signed_power parameter value, handling arbitrary nesting depth.

        Uses bracket-matching to find the correct closing paren for signed_power(...).
        """
        idx = expr.rfind("signed_power(")
        if idx == -1:
            return None
        start = idx + len("signed_power(")
        depth = 0
        for i in range(start, len(expr)):
            if expr[i] == "(":
                depth += 1
            elif expr[i] == ")":
                if depth == 0:
                    inner = expr[start:i]
                    comma_pos = inner.rfind(",")
                    if comma_pos >= 0:
                        try:
                            return float(inner[comma_pos + 1 :].strip())
                        except (ValueError, IndexError):
                            return None
                    return None
                depth -= 1
        return None

    @staticmethod
    def _extract_decay_windows(expr: str) -> list[int]:
        """Extract all ts_decay_linear window values, handling nested parentheses."""
        windows = []
        search_start = 0
        while True:
            idx = expr.find("ts_decay_linear(", search_start)
            if idx == -1:
                break
            start = idx + len("ts_decay_linear(")
            depth = 0
            for i in range(start, len(expr)):
                if expr[i] == "(":
                    depth += 1
                elif expr[i] == ")":
                    if depth == 0:
                        inner = expr[start:i]
                        comma_pos = inner.rfind(",")
                        if comma_pos >= 0:
                            try:
                                w = int(inner[comma_pos + 1 :].strip())
                                windows.append(w)
                            except (ValueError, IndexError):
                                pass
                        break
                    depth -= 1
            search_start = start
        return windows

    def _tier_decay_calibration(self, expr: str, analysis: dict) -> list[FitnessVariant]:
        """Tier 1: Find optimal decay window — balance smoothness vs signal retention.

        Hypothesis: Over-smoothed signals lose amplitude → lower absolute returns → lower fitness.
        Strategy: Try shorter decay windows to retain more signal energy.
        """
        variants = []
        features = analysis["expression_features"]
        current_windows = features.get("decay_windows", [])

        if not current_windows:
            for w in [10, 20, 30]:
                new_expr = f"ts_decay_linear({expr}, {w})"
                variants.append(
                    FitnessVariant(
                        expression=new_expr,
                        boost_tier="decay_calibration",
                        mutation_description=f"add ts_decay_linear(window={w})",
                        expected_fitness_delta=0.12,
                        risk_level="low",
                    )
                )
            return variants

        for current_w in current_windows:
            candidate_windows = [w for w in FITNESS_DECAY_WINDOWS if w < current_w and w >= 5]
            for new_w in candidate_windows[:3]:
                new_expr = re.sub(
                    rf"ts_decay_linear\(([^,]+),\s*{current_w}\)",
                    f"ts_decay_linear(\\1, {new_w})",
                    expr,
                )
                if new_expr != expr:
                    reduction_pct = (current_w - new_w) / current_w * 100
                    variants.append(
                        FitnessVariant(
                            expression=new_expr,
                            boost_tier="decay_calibration",
                            mutation_description=f"decay {current_w}→{new_w} (-{reduction_pct:.0f}% smooth)",
                            expected_fitness_delta=min(0.20, reduction_pct * 0.003),
                            risk_level="low",
                        )
                    )

            for new_w in [w for w in FITNESS_DECAY_WINDOWS if w > current_w]:
                new_expr = re.sub(
                    rf"ts_decay_linear\(([^,]+),\s*{current_w}\)",
                    f"ts_decay_linear(\\1, {new_w})",
                    expr,
                )
                if new_expr != expr and not any(v.expression == new_expr for v in variants):
                    variants.append(
                        FitnessVariant(
                            expression=new_expr,
                            boost_tier="decay_calibration",
                            mutation_description=f"decay {current_w}→{new_w} (+smooth)",
                            expected_fitness_delta=0.05,
                            risk_level="low",
                        )
                    )

        return variants

    def _tier_signal_amplification(self, expr: str, analysis: dict) -> list[FitnessVariant]:
        """Tier 2: Amplify signal magnitude through parameter tuning.

        Hypothesis: Weak signal amplitude → low absolute returns → low fitness.
        Strategy: Adjust power parameters, add scale factors.
        """
        variants = []
        features = analysis["expression_features"]

        power_value = features.get("power_value")
        if power_value is not None:
            for new_p in FITNESS_POWER_VALUES:
                if abs(new_p - power_value) > 0.1:
                    new_expr = re.sub(
                        r"signed_power\(([^,]+),\s*[\d.]+\)",
                        f"signed_power(\\1, {new_p})",
                        expr,
                    )
                    if new_expr != expr:
                        delta = (new_p - power_value) * 0.08
                        variants.append(
                            FitnessVariant(
                                expression=new_expr,
                                boost_tier="signal_amplification",
                                mutation_description=f"power {power_value}→{new_p}",
                                expected_fitness_delta=max(-0.05, min(0.18, delta)),
                                risk_level="medium" if abs(new_p) > 2.5 else "low",
                            )
                        )

        if "signed_power" not in expr and "scale" not in expr:
            for s in [1.2, 1.5, 2.0]:
                new_expr = f"scale({expr}, {s})"
                variants.append(
                    FitnessVariant(
                        expression=new_expr,
                        boost_tier="signal_amplification",
                        mutation_description=f"add scale(x, {s}) amplifier",
                        expected_fitness_delta=0.08 * s,
                        risk_level="medium",
                    )
                )

        zscore_match = re.search(r"ts_zscore\(([^,]+),\s*(\d+)\)", expr)
        if zscore_match:
            zscore_match.group(1)
            old_w = int(zscore_match.group(2))
            for new_w in FITNESS_ZSCORE_WINDOWS:
                if new_w != old_w:
                    new_expr = re.sub(
                        r"ts_zscore\(([^,]+),\s*\d+\)",
                        f"ts_zscore(\\1, {new_w})",
                        expr,
                    )
                    if new_expr != expr:
                        variants.append(
                            FitnessVariant(
                                expression=new_expr,
                                boost_tier="signal_amplification",
                                mutation_description=f"zscore window {old_w}→{new_w}",
                                expected_fitness_delta=0.04,
                                risk_level="low",
                            )
                        )

        return variants

    def _tier_neutralization_balance(self, expr: str, analysis: dict) -> list[FitnessVariant]:
        """Tier 3: Calibrate neutralization granularity + multi-layer upgrade.

        Hypothesis:
          - Over-neutralization (too fine-grained) kills signal → lower returns
          - Single-layer neutralize leaves industry/style exposure → lower pure alpha
        Strategy:
          - Coarsen neutralization group to preserve more cross-sectional signal
          - Upgrade to multi-layer neutralization for higher pure alpha
        """
        variants = []
        features = analysis["expression_features"]
        current_group = features.get("neutralize_group")

        if not current_group:
            return variants

        group_hierarchy = ["sw", "sector", "industry", "subindustry", "market"]
        coarser_groups = ["market", "sector", "industry"]

        try:
            current_idx = group_hierarchy.index(current_group)
        except ValueError:
            current_idx = len(group_hierarchy) // 2

        for target_group in coarser_groups:
            try:
                target_idx = group_hierarchy.index(target_group)
            except ValueError:
                continue
            if target_idx < current_idx:
                new_expr = re.sub(
                    rf"group_neutralize\(([^,]+),\s*{re.escape(current_group)}\)",
                    f"group_neutralize(\\1, {target_group})",
                    expr,
                )
                if new_expr != expr:
                    granularity_gain = (current_idx - target_idx) * 0.06
                    variants.append(
                        FitnessVariant(
                            expression=new_expr,
                            boost_tier="neutralization_balance",
                            mutation_description=f"neutralize {current_group}→{target_group} (coarser)",
                            expected_fitness_delta=granularity_gain,
                            risk_level="medium",
                        )
                    )

        if "group_zscore" not in expr and current_group in ("subindustry", "industry"):
            for g in ["market", "sector"]:
                new_expr = re.sub(
                    rf"group_neutralize\(([^,]+),\s*{re.escape(current_group)}\)",
                    f"group_zscore(\\1, {g})",
                    expr,
                )
                if new_expr != expr:
                    variants.append(
                        FitnessVariant(
                            expression=new_expr,
                            boost_tier="neutralization_balance",
                            mutation_description=f"group_neutralize→group_zscore({g})",
                            expected_fitness_delta=0.08,
                            risk_level="medium",
                        )
                    )

        single_layer_match = re.search(r"group_neutralize\(([^,]+),\s*([^)]+)\)", expr)
        if single_layer_match and expr.count("group_neutralize") == 1:
            inner_signal = single_layer_match.group(1)
            original_group = single_layer_match.group(2).strip()

            multi_layer_upgrades = [
                (
                    "B1",
                    f"group_neutralize(group_neutralize({inner_signal}, sector), industry)",
                    0.12,
                    "双层 sector→industry",
                ),
                (
                    "B2",
                    f"group_neutralize(group_neutralize(group_neutralize({inner_signal}, market), sector), industry)",
                    0.15,
                    "三层 market→sector→industry",
                ),
                (
                    "B3",
                    f"group_neutralize(group_zscore({inner_signal}, market), industry)",
                    0.13,
                    "混合 zscore(market)→neutralize(industry)",
                ),
            ]

            for strategy_id, new_neutralize_expr, delta, desc in multi_layer_upgrades:
                full_new_expr = re.sub(
                    r"group_neutralize\([^)]+\)[^,]*",
                    new_neutralize_expr,
                    expr,
                    count=1,
                )
                if full_new_expr != expr and not any(v.expression == full_new_expr for v in variants):
                    variants.append(
                        FitnessVariant(
                            expression=full_new_expr,
                            boost_tier="neutralization_balance",
                            mutation_description=f"升级多层中性化 {strategy_id}: {desc}",
                            expected_fitness_delta=delta,
                            risk_level="medium",
                        )
                    )

            if original_group in ("sector", "industry"):
                subindustry_upgrade = re.sub(
                    rf"group_neutralize\(([^,]+),\s*{re.escape(original_group)}\)",
                    r"group_neutralize(\1, subindustry)",
                    expr,
                )
                if subindustry_upgrade != expr and not any(v.expression == subindustry_upgrade for v in variants):
                    variants.append(
                        FitnessVariant(
                            expression=subindustry_upgrade,
                            boost_tier="neutralization_balance",
                            mutation_description="升级 subindustry 精细中性化 (B4)",
                            expected_fitness_delta=0.10,
                            risk_level="medium",
                        )
                    )

        return variants

    def _tier_structure_streamlining(self, expr: str, analysis: dict) -> list[FitnessVariant]:
        """Tier 4: Remove fitness-killing structural patterns.

        Patterns that hurt fitness:
          - Nested signed_power inside decay (double nonlinearity)
          - Multiple regression/correlation ops (overfitting-prone)
          - Deeply nested expressions (>6 levels)
        """
        variants = []
        features = analysis["expression_features"]

        if features.get("has_signed_power"):
            power_val = features.get("power_value")
            if power_val and power_val > 2.0:
                for lower_p in [1.5, 2.0]:
                    new_expr = re.sub(
                        r"signed_power\(([^,]+),\s*[\d.]+\)",
                        f"signed_power(\\1, {lower_p})",
                        expr,
                    )
                    if new_expr != expr:
                        variants.append(
                            FitnessVariant(
                                expression=new_expr,
                                boost_tier="structure_streamlining",
                                mutation_description=f"reduce power {power_val}→{lower_p}",
                                expected_fitness_delta=0.07,
                                risk_level="low",
                            )
                        )

            new_expr = re.sub(r"signed_power\(([^,]+),\s*[\d.]+\)", r"\1", expr)
            if new_expr != expr and len(new_expr) >= 5:
                variants.append(
                    FitnessVariant(
                        expression=new_expr,
                        boost_tier="structure_streamlining",
                        mutation_description="remove signed_power entirely",
                        expected_fitness_delta=0.12,
                        risk_level="medium",
                    )
                )

        for bad_op in ["ts_regression", "ts_corr", "ts_covariance", "ts_product"]:
            if bad_op in expr:
                replacements = {
                    "ts_regression": "ts_rank",
                    "ts_corr": "ts_rank",
                    "ts_covariance": "ts_rank",
                    "ts_product": "ts_sum",
                }
                if bad_op in replacements:
                    new_expr = re.sub(rf"\b{bad_op}\b", replacements[bad_op], expr, count=1)
                    if new_expr != expr:
                        variants.append(
                            FitnessVariant(
                                expression=new_expr,
                                boost_tier="structure_streamlining",
                                mutation_description=f"{bad_op}→{replacements[bad_op]} (simplify)",
                                expected_fitness_delta=self.OPERATOR_FITNESS_IMPACT.get(bad_op, 0) * -1,
                                risk_level="low",
                            )
                        )

        if features.get("nesting_depth", 0) > 6:
            inner_match = re.search(r"signed_power\((.+)\)", expr)
            if inner_match:
                simplified = inner_match.group(1)
                new_expr = re.sub(r"signed_power\(.+\)", simplified, expr)
                if new_expr != expr and len(new_expr) >= 5:
                    variants.append(
                        FitnessVariant(
                            expression=new_expr,
                            boost_tier="structure_streamlining",
                            mutation_description="reduce nesting depth (remove outer wrapper)",
                            expected_fitness_delta=0.06,
                            risk_level="medium",
                        )
                    )

        return variants

    def _tier_composite_mutation(self, expr: str, analysis: dict) -> list[FitnessVariant]:
        """Tier 5: Coordinated multi-parameter mutations.

        These are higher-risk but potentially higher-reward changes
        that modify multiple aspects simultaneously.
        """
        variants = []
        features = analysis["expression_features"]

        decay_windows = features.get("decay_windows", [])
        if decay_windows and features.get("has_signed_power"):
            primary_decay = decay_windows[0]
            for shorter_w in [max(5, primary_decay // 2), max(5, primary_decay - 10)]:
                if shorter_w < primary_decay and shorter_w >= 5:
                    new_expr = expr
                    new_expr = re.sub(
                        rf"ts_decay_linear\(([^,]+),\s*{primary_decay}\)",
                        f"ts_decay_linear(\\1, {shorter_w})",
                        new_expr,
                    )
                    power_val = features.get("power_value")
                    if power_val and power_val > 1.5:
                        new_expr = re.sub(
                            r"signed_power\(([^,]+),\s*[\d.]+\)",
                            f"signed_power(\\1, {max(1.0, power_val - 0.5)})",
                            new_expr,
                        )
                    if new_expr != expr:
                        variants.append(
                            FitnessVariant(
                                expression=new_expr,
                                boost_tier="composite_mutation",
                                mutation_description=f"composite: decay-{primary_decay}→{shorter_w} + reduce power",
                                expected_fitness_delta=0.18,
                                risk_level="high",
                            )
                        )

        current_group = features.get("neutralize_group")
        if current_group and decay_windows:
            primary_decay = decay_windows[0]
            for coarser in ["market", "sector"]:
                if coarser != current_group:
                    new_expr = expr
                    new_expr = re.sub(
                        rf"group_neutralize\(([^,]+),\s*{re.escape(current_group)}\)",
                        f"group_neutralize(\\1, {coarser})",
                        new_expr,
                    )
                    new_expr = re.sub(
                        rf"ts_decay_linear\(([^,]+),\s*{primary_decay}\)",
                        f"ts_decay_linear(\\1, {max(5, primary_decay - 5)})",
                        new_expr,
                    )
                    if new_expr != expr:
                        variants.append(
                            FitnessVariant(
                                expression=new_expr,
                                boost_tier="composite_mutation",
                                mutation_description=f"composite: neutralize→{coarser} + decay-{primary_decay}",
                                expected_fitness_delta=0.15,
                                risk_level="high",
                            )
                        )

        if features.get("complex_op_count", 0) >= 2:
            new_expr = expr
            for bad_op in ["ts_regression", "ts_corr", "ts_product"]:
                if bad_op in new_expr:
                    new_expr = re.sub(rf"\b{bad_op}\b", "ts_rank", new_expr, count=1)
            if "signed_power" in new_expr:
                new_expr = re.sub(r"signed_power\(([^,]+),\s*[\d.]+\)", r"\1", new_expr)
            if new_expr != expr and len(new_expr) >= 5:
                variants.append(
                    FitnessVariant(
                        expression=new_expr,
                        boost_tier="composite_mutation",
                        mutation_description="composite: simplify ops + remove nonlinear",
                        expected_fitness_delta=0.22,
                        risk_level="high",
                    )
                )

        return variants

    def _tier_low_turnover_wrapping(self, expr: str, analysis: dict) -> list[FitnessVariant]:
        """Tier 6: Wrap expression with hump() or trade_when() to reduce turnover.

        Hypothesis: High turnover → high transaction costs → lower fitness.
        Strategy:
          - hump(expr, size): limits signal change amplitude per period
          - trade_when(expr, entry, exit): only rebalance when signal crosses thresholds
        """
        variants = []
        analysis["expression_features"]

        for hump_size in LOW_TURNOVER_HUMP_SIZES:
            new_expr = f"hump({expr}, {hump_size})"
            variants.append(
                FitnessVariant(
                    expression=new_expr,
                    boost_tier="low_turnover_wrapping",
                    mutation_description=f"wrap with hump(size={hump_size})",
                    expected_fitness_delta=0.10 + hump_size * 0.5,
                    risk_level="low",
                )
            )

            if "ts_decay_linear" in expr:
                decay_wrapped = re.sub(
                    r"ts_decay_linear\(([^,]+),\s*(\d+)\)",
                    f"ts_decay_linear(hump(\\1, {hump_size}), \\2)",
                    expr,
                    count=1,
                )
                if decay_wrapped != expr:
                    variants.append(
                        FitnessVariant(
                            expression=decay_wrapped,
                            boost_tier="low_turnover_wrapping",
                            mutation_description=f"decay inner hump(size={hump_size})",
                            expected_fitness_delta=0.14 + hump_size * 0.3,
                            risk_level="low",
                        )
                    )

        for entry_thresh in LOW_TURNOVER_ENTRY_THRESHOLDS:
            for exit_thresh in LOW_TURNOVER_EXIT_THRESHOLDS:
                new_expr = f"trade_when({expr}, {entry_thresh}, {exit_thresh})"
                variants.append(
                    FitnessVariant(
                        expression=new_expr,
                        boost_tier="low_turnover_wrapping",
                        mutation_description=f"wrap with trade_when(entry={entry_thresh}, exit={exit_thresh})",
                        expected_fitness_delta=0.15 + abs(entry_thresh) * 0.05 + abs(exit_thresh) * 0.03,
                        risk_level="medium",
                    )
                )

                if "ts_decay_linear" in expr:
                    decay_trade = re.sub(
                        r"ts_decay_linear\(([^,]+),\s*(\d+)\)",
                        f"trade_when(ts_decay_linear(\\1, \\2), {entry_thresh}, {exit_thresh})",
                        expr,
                        count=1,
                    )
                    if decay_trade != expr and not any(v.expression == decay_trade for v in variants):
                        variants.append(
                            FitnessVariant(
                                expression=decay_trade,
                                boost_tier="low_turnover_wrapping",
                                mutation_description=f"decay + trade_when(entry={entry_thresh}, exit={exit_thresh})",
                                expected_fitness_delta=0.18,
                                risk_level="medium",
                            )
                        )

        for hump_size in [0.03, 0.05]:
            for entry_thresh in [1.0]:
                for exit_thresh in [-0.3]:
                    new_expr = f"trade_when(hump({expr}, {hump_size}), {entry_thresh}, {exit_thresh})"
                    variants.append(
                        FitnessVariant(
                            expression=new_expr,
                            boost_tier="low_turnover_wrapping",
                            mutation_description=f"dual: hump({hump_size}) + trade_when({entry_thresh},{exit_thresh})",
                            expected_fitness_delta=0.22,
                            risk_level="medium",
                        )
                    )

        return variants


def get_fitness_boost_engine() -> FitnessBoostEngine:
    return FitnessBoostEngine()
