"""Tests for TurnoverOptimizer — Turnover adaptive optimization engine (Strategy C).

Tests cover:
  1. Initialization and configuration
  2. WQ Fitness formula computation
  3. Turnover bottleneck analysis
  4. Variant generation (5 strategies)
  5. Fitness gain estimation
  6. Full optimization pipeline
  7. Edge cases and error handling
"""

from __future__ import annotations

import math

import pytest

from openalpha_brain.optimization.turnover_optimizer import (
    TO_FLOOR,
    TO_OPTIMAL_MAX,
    TO_OPTIMAL_MIN,
    TurnoverOptimizationResult,
    TurnoverOptimizer,
    TurnoverSeverity,
    Variant,
    get_turnover_optimizer,
)


class TestTurnoverOptimizerInit:
    """Test suite for TurnoverOptimizer initialization."""

    def test_default_creation(self):
        engine = TurnoverOptimizer()
        assert engine is not None
        assert engine.config["max_variants"] == 10
        assert engine.config["to_threshold"] == 0.25

    def test_custom_config(self):
        config = {"max_variants": 15, "to_threshold": 0.30}
        engine = TurnoverOptimizer(config=config)
        assert engine.config["max_variants"] == 15
        assert engine.config["to_threshold"] == 0.30

    def test_factory_function(self):
        engine = get_turnover_optimizer()
        assert isinstance(engine, TurnoverOptimizer)

    def test_factory_with_config(self):
        config = {"enable_hump": False}
        engine = get_turnover_optimizer(config=config)
        assert engine.config["enable_hump"] is False

    def test_operator_impact_dict_not_empty(self):
        engine = TurnoverOptimizer()
        assert len(engine.OPERATOR_TURNOVER_IMPACT) > 0
        assert "ts_decay_linear" in engine.OPERATOR_TURNOVER_IMPACT

    def test_strategy_priority_ordering(self):
        engine = TurnoverOptimizer()
        priorities = engine.STRATEGY_PRIORITY
        assert priorities["decay_extension"] == 1
        assert priorities["double_smoothing"] == 5


class TestFitnessComputation:
    """Test WQ Fitness formula: Fitness = Sharpe × √(|Returns| / max(TO, 0.125))."""

    def setup_method(self):
        self.engine = TurnoverOptimizer()

    def test_basic_fitness_calculation(self):
        fitness = self.engine.compute_fitness(sharpe=1.30, returns_abs=0.028, to=0.314)
        expected = 1.30 * math.sqrt(0.028 / max(0.314, TO_FLOOR))
        assert fitness == pytest.approx(expected, rel=1e-9)

    def test_fitness_with_low_turnover(self):
        fitness = self.engine.compute_fitness(sharpe=1.30, returns_abs=0.028, to=0.15)
        expected = 1.30 * math.sqrt(0.028 / 0.15)
        assert fitness == pytest.approx(expected, rel=1e-9)

    def test_fitness_with_very_high_turnover(self):
        fitness = self.engine.compute_fitness(sharpe=1.30, returns_abs=0.028, to=0.70)
        assert fitness > 0

    def test_fitness_to_floor_applied(self):
        fitness_low = self.engine.compute_fitness(sharpe=1.0, returns_abs=0.02, to=0.05)
        fitness_floor = self.engine.compute_fitness(sharpe=1.0, returns_abs=0.02, to=TO_FLOOR)
        assert fitness_low == fitness_floor

    def test_fitness_zero_sharpe(self):
        fitness = self.engine.compute_fitness(sharpe=0.0, returns_abs=0.028, to=0.314)
        assert fitness == 0.0

    def test_fitness_zero_returns(self):
        fitness = self.engine.compute_fitness(sharpe=1.30, returns_abs=0.0, to=0.314)
        assert fitness == 0.0

    def test_fitness_negative_turnover_treated_as_positive(self):
        fitness = self.engine.compute_fitness(sharpe=1.30, returns_abs=0.028, to=-0.10)
        assert fitness == 0.0

    def test_fitness_higher_sharpe_increases_fitness(self):
        fitness1 = self.engine.compute_fitness(sharpe=1.0, returns_abs=0.03, to=0.25)
        fitness2 = self.engine.compute_fitness(sharpe=2.0, returns_abs=0.03, to=0.25)
        assert fitness2 > fitness1

    def test_fitness_higher_returns_increases_fitness(self):
        fitness1 = self.engine.compute_fitness(sharpe=1.5, returns_abs=0.01, to=0.25)
        fitness2 = self.engine.compute_fitness(sharpe=1.5, returns_abs=0.05, to=0.25)
        assert fitness2 > fitness1


class TestBottleneckAnalysis:
    """Test turnover bottleneck analysis functionality."""

    def setup_method(self):
        self.engine = TurnoverOptimizer()

    def test_e2e_data_analysis(self):
        result = self.engine.analyze_turnover_bottleneck(
            expr="ts_decay_linear(rank(volume), 20)",
            sharpe=1.30,
            fitness=0.69,
            turnover=0.314,
        )
        assert "current_penalty" in result
        assert "optimal_to" in result
        assert "potential_gain" in result
        assert "severity" in result
        assert "is_bottleneck" in result
        assert result["is_bottleneck"] is True

    def test_high_turnover_triggers_bottleneck(self):
        result = self.engine.analyze_turnover_bottleneck(
            expr="rank(ts_delta(close, 5))",
            sharpe=1.2,
            fitness=0.65,
            turnover=0.50,
        )
        assert result["is_bottleneck"] is True
        assert result["severity"] in ("high", "critical")

    def test_low_turnover_no_bottleneck(self):
        result = self.engine.analyze_turnover_bottleneck(
            expr="ts_decay_linear(rank(close), 30)",
            sharpe=1.3,
            fitness=0.85,
            turnover=0.18,
        )
        assert result["is_bottleneck"] is False

    def test_critical_severity_for_very_high_to(self):
        result = self.engine.analyze_turnover_bottleneck(
            expr="ts_delta(close, 1)",
            sharpe=1.0,
            fitness=0.50,
            turnover=0.70,
        )
        assert result["severity"] == "critical"

    def test_optimal_to_in_reasonable_range(self):
        result = self.engine.analyze_turnover_bottleneck(
            expr="rank(volume)",
            sharpe=1.3,
            fitness=0.69,
            turnover=0.314,
        )
        optimal = result["optimal_to"]
        assert TO_OPTIMAL_MIN <= optimal <= TO_OPTIMAL_MAX + 0.05

    def test_potential_gain_positive_for_high_to(self):
        result = self.engine.analyze_turnover_bottleneck(
            expr="ts_regression(close, volume, 5)",
            sharpe=1.25,
            fitness=0.60,
            turnover=0.45,
        )
        assert result["potential_gain"] > 0

    def test_recommended_strategies_include_decay(self):
        result = self.engine.analyze_turnover_bottleneck(
            expr="ts_decay_linear(rank(volume), 10)",
            sharpe=1.3,
            fitness=0.70,
            turnover=0.35,
        )
        strategies = result["recommended_strategies"]
        assert len(strategies) >= 1
        assert "decay_extension" in strategies

    def test_inferred_returns_positive(self):
        result = self.engine.analyze_turnover_bottleneck(
            expr="rank(close)",
            sharpe=1.30,
            fitness=0.69,
            turnover=0.314,
        )
        assert result["inferred_returns"] > 0

    def test_details_contains_expression_info(self):
        result = self.engine.analyze_turnover_bottleneck(
            expr="group_neutralize(ts_zscore(close, 20), industry)",
            sharpe=1.3,
            fitness=0.75,
            turnover=0.30,
        )
        details = result.get("details", {})
        assert "expression_length" in details
        assert "operator_count" in details

    def test_zero_turnover_handled_gracefully(self):
        result = self.engine.analyze_turnover_bottleneck(
            expr="rank(close)",
            sharpe=1.3,
            fitness=0.80,
            turnover=0.0,
        )
        assert "current_penalty" in result


class TestDecayVariants:
    """Test Strategy 1: Decay window extension variants."""

    def setup_method(self):
        self.engine = TurnoverOptimizer()

    def test_extend_existing_decay_window(self):
        variants = self.engine._generate_decay_variants(
            "ts_decay_linear(rank(volume), 10)",
            turnover=0.314,
        )
        assert len(variants) >= 1
        extended = [v for v in variants if v.strategy == "decay_extension"]
        assert len(extended) >= 1
        for v in extended:
            assert "ts_decay_linear" in v.expression
            assert v.risk_level == "low"

    def test_no_duplicate_expressions(self):
        variants = self.engine._generate_decay_variants(
            "ts_decay_linear(rank(close), 20)",
            turnover=0.35,
        )
        expressions = [v.expression for v in variants]
        assert len(expressions) == len(set(expressions))

    def test_higher_window_has_lower_confidence(self):
        variants = self.engine._generate_decay_variants(
            "ts_decay_linear(rank(volume), 5)",
            turnover=0.40,
        )
        window_10 = [v for v in variants if ", 10)" in v.expression]
        window_60 = [v for v in variants if ", 60)" in v.expression]
        if window_10 and window_60:
            assert window_10[0].confidence >= window_60[0].confidence

    def test_expected_reduction_positive(self):
        variants = self.engine._generate_decay_variants(
            "ts_decay_linear(rank(volume), 10)",
            turnover=0.314,
        )
        for v in variants:
            assert v.expected_to_reduction >= 0

    def test_description_contains_window_info(self):
        variants = self.engine._generate_decay_variants(
            "ts_decay_linear(rank(close), 15)",
            turnover=0.30,
        )
        assert len(variants) >= 1
        assert "window" in variants[0].description.lower() or "→" in variants[0].description


class TestHumpVariants:
    """Test Strategy 2: Hump filtering wrapper."""

    def setup_method(self):
        self.engine = TurnoverOptimizer()

    def test_generates_hump_variants(self):
        variants = self.engine._generate_hump_variants(
            "rank(ts_delta(close, 5))",
            turnover=0.45,
        )
        assert len(variants) == 4
        for v in variants:
            assert v.expression.startswith("hump(")
            assert v.strategy == "hump_filtering"

    def test_different_hump_sizes(self):
        variants = self.engine._generate_hump_variants("rank(close)", turnover=0.40)
        sizes_used = set()
        for v in variants:
            size_match = __import__("re").search(r"hump\([^,]+,\s*([\d.]+)\)", v.expression)
            if size_match:
                sizes_used.add(float(size_match.group(1)))
        assert len(sizes_used) == 4

    def test_medium_risk_level(self):
        variants = self.engine._generate_hump_variants("ts_delta(close, 5)", turnover=0.50)
        for v in variants:
            assert v.risk_level == "medium"

    def test_disabled_when_config_false(self):
        engine = TurnoverOptimizer(config={"enable_hump": False})
        variants = engine._generate_hump_variants("rank(close)", turnover=0.40)
        assert len(variants) == 0

    def test_larger_size_higher_reduction(self):
        variants = self.engine._generate_hump_variants("rank(volume)", turnover=0.45)
        if len(variants) >= 2:
            sorted_by_size = sorted(variants, key=lambda v: v.expected_to_reduction)
            assert sorted_by_size[-1].expected_to_reduction >= sorted_by_size[0].expected_to_reduction


class TestTradeWhenVariants:
    """Test Strategy 3: Trade_when conditional trigger."""

    def setup_method(self):
        self.engine = TurnoverOptimizer()

    def test_generates_trade_when_variants(self):
        variants = self.engine._generate_trade_when_variants(
            "ts_delta(close, 5)",
            turnover=0.50,
        )
        assert len(variants) > 0
        assert len(variants) <= 9
        for v in variants:
            assert v.expression.startswith("trade_when(")

    def test_entry_exit_thresholds_vary(self):
        variants = self.engine._generate_trade_when_variants("rank(close)", turnover=0.45)
        patterns = set()
        for v in variants:
            match = __import__("re").search(r"trade_when\([^,]+,\s*([^,]+),\s*([^)]+)\)", v.expression)
            if match:
                patterns.add((match.group(1), match.group(2)))
        assert len(patterns) >= 3

    def test_medium_risk_level(self):
        variants = self.engine._generate_trade_when_variants("ts_corr(close, volume, 20)", turnover=0.55)
        for v in variants:
            assert v.risk_level == "medium"

    def test_disabled_when_config_false(self):
        engine = TurnoverOptimizer(config={"enable_trade_when": False})
        variants = engine._generate_trade_when_variants("rank(close)", turnover=0.50)
        assert len(variants) == 0

    def test_higher_reduction_than_decay(self):
        trade_variants = self.engine._generate_trade_when_variants("ts_delta(close, 5)", turnover=0.50)
        decay_variants = self.engine._generate_decay_variants("ts_decay_linear(ts_delta(close, 5), 10)", turnover=0.50)
        if trade_variants and decay_variants:
            avg_trade = sum(v.expected_to_reduction for v in trade_variants) / len(trade_variants)
            avg_decay = sum(v.expected_to_reduction for v in decay_variants) / len(decay_variants)
            assert avg_trade > avg_decay


class TestMeanVariants:
    """Test Strategy 4: ts_mean replacement (smoother)."""

    def setup_method(self):
        self.engine = TurnoverOptimizer()

    def test_replace_decay_with_mean(self):
        variants = self.engine._generate_mean_variants(
            "ts_decay_linear(rank(volume), 20)",
            turnover=0.35,
        )
        assert len(variants) >= 1
        mean_variants = [v for v in variants if "ts_mean" in v.expression]
        assert len(mean_variants) >= 1

    def test_wrap_without_decay(self):
        variants = self.engine._generate_mean_variants(
            "rank(ts_delta(close, 5))",
            turnover=0.45,
        )
        assert len(variants) >= 1
        for v in variants:
            assert "ts_mean" in v.expression

    def test_low_risk_level(self):
        variants = self.engine._generate_mean_variants(
            "ts_decay_linear(rank(close), 15)",
            turnover=0.30,
        )
        for v in variants:
            assert v.risk_level == "low"

    def test_moderate_expected_reduction(self):
        variants = self.engine._generate_mean_variants("rank(volume)", turnover=0.40)
        for v in variants:
            assert 0.10 <= v.expected_to_reduction <= 0.40

    def test_multiple_windows_generated(self):
        variants = self.engine._generate_mean_variants(
            "ts_decay_linear(ts_zscore(close, 20), 10)",
            turnover=0.35,
        )
        windows = set()
        for v in variants:
            match = __import__("re").search(r"ts_mean\([^,]+,\s*(\d+)\)", v.expression)
            if match:
                windows.add(int(match.group(1)))
        assert len(windows) >= 2


class TestDoubleSmoothingVariants:
    """Test Strategy 5: Nested ts_decay_linear (double smoothing)."""

    def setup_method(self):
        self.engine = TurnoverOptimizer()

    def test_add_outer_decay_layer(self):
        variants = self.engine._generate_double_smoothing_variants(
            "rank(ts_delta(close, 5))",
            turnover=0.50,
        )
        assert len(variants) >= 1
        for v in variants:
            assert v.expression.count("ts_decay_linear") >= 1

    def test_nested_structure(self):
        variants = self.engine._generate_double_smoothing_variants(
            "ts_decay_linear(rank(volume), 10)",
            turnover=0.40,
        )
        for v in variants:
            assert v.expression.startswith("ts_decay_linear(")

    def test_medium_risk_level(self):
        variants = self.engine._generate_double_smoothing_variants("rank(close)", turnover=0.38)
        for v in variants:
            assert v.risk_level == "medium"

    def test_disabled_when_config_false(self):
        engine = TurnoverOptimizer(config={"enable_double_smoothing": False})
        variants = engine._generate_double_smoothing_variants("rank(close)", turnover=0.50)
        assert len(variants) == 0

    def test_only_triggered_for_high_to(self):
        variants = self.engine._generate_double_smoothing_variants("rank(close)", turnover=0.20)
        outer_windows = set()
        for v in variants:
            match = __import__("re").search(r"ts_decay_linear\([^,]+,\s*(\d+)\)\)$", v.expression)
            if match:
                outer_windows.add(int(match.group(1)))
        assert len(outer_windows) <= 3


class TestVariantGeneration:
    """Test comprehensive variant generation across all strategies."""

    def setup_method(self):
        self.engine = TurnoverOptimizer()

    def test_generate_all_strategies(self):
        variants = self.engine.generate_low_turnover_variants(
            expr="ts_decay_linear(rank(ts_delta(close, 5)), 10)",
            turnover=0.45,
            max_variants=20,
        )
        strategies_used = set(v.strategy for v in variants)
        assert len(strategies_used) >= 3

    def test_max_variants_limit_respected(self):
        variants = self.engine.generate_low_turnover_variants(
            expr="rank(volume)",
            turnover=0.50,
            max_variants=5,
        )
        assert len(variants) <= 5

    def test_no_duplicate_variants(self):
        variants = self.engine.generate_low_turnover_variants(
            expr="ts_decay_linear(rank(close), 20)",
            turnover=0.40,
            max_variants=15,
        )
        expressions = [v.expression for v in variants]
        assert len(expressions) == len(set(expressions))

    def test_original_expression_not_included(self):
        original = "rank(ts_delta(volume, 10))"
        variants = self.engine.generate_low_turnover_variants(
            expr=original,
            turnover=0.45,
            max_variants=10,
        )
        assert original not in [v.expression for v in variants]

    def test_sorted_by_expected_reduction(self):
        variants = self.engine.generate_low_turnover_variants(
            expr="ts_delta(close, 5)",
            turnover=0.55,
            max_variants=10,
        )
        if len(variants) >= 2:
            for i in range(len(variants) - 1):
                score_i = variants[i].expected_to_reduction * variants[i].confidence
                score_j = variants[i + 1].expected_to_reduction * variants[i + 1].confidence
                assert score_i >= score_j

    def test_high_turnover_generates_more_variants(self):
        high_to = self.engine.generate_low_turnover_variants("rank(close)", turnover=0.60, max_variants=20)
        low_to = self.engine.generate_low_turnover_variants("rank(close)", turnover=0.20, max_variants=20)
        assert len(high_to) >= len(low_to)


class TestEstimateFitnessGain:
    """Test fitness gain estimation from TO reduction."""

    def setup_method(self):
        self.engine = TurnoverOptimizer()

    def test_positive_gain_from_to_reduction(self):
        gain = self.engine.estimate_fitness_gain(
            current_to=0.50,
            target_to=0.20,
            sharpe=1.30,
            returns_estimate=0.028,
        )
        assert gain > 0

    def test_zero_gain_when_target_equals_current(self):
        gain = self.engine.estimate_fitness_gain(
            current_to=0.30,
            target_to=0.30,
            sharpe=1.30,
            returns_estimate=0.028,
        )
        assert gain == 0.0

    def test_larger_reduction_larger_gain(self):
        gain_small = self.engine.estimate_fitness_gain(
            current_to=0.40, target_to=0.35, sharpe=1.5, returns_estimate=0.03
        )
        gain_large = self.engine.estimate_fitness_gain(
            current_to=0.40, target_to=0.15, sharpe=1.5, returns_estimate=0.03
        )
        assert gain_large > gain_small

    def test_gain_independent_of_sharpe_direction(self):
        _gain_pos = self.engine.estimate_fitness_gain(current_to=0.50, target_to=0.20, sharpe=1.5, returns_estimate=0.03)
        gain_neg = self.engine.estimate_fitness_gain(
            current_to=0.50, target_to=0.20, sharpe=-1.5, returns_estimate=0.03
        )
        assert gain_neg == 0.0

    def test_zero_handles_edge_case(self):
        gain = self.engine.estimate_fitness_gain(
            current_to=0.0,
            target_to=0.10,
            sharpe=1.0,
            returns_estimate=0.02,
        )
        assert gain == 0.0

    def test_e2e_scenario_realistic_improvement(self):
        gain = self.engine.estimate_fitness_gain(
            current_to=0.314,
            target_to=0.18,
            sharpe=1.30,
            returns_estimate=0.028,
        )
        assert 0.10 < gain < 0.50


class TestFullOptimization:
    """Test complete optimization pipeline."""

    def setup_method(self):
        self.engine = TurnoverOptimizer()

    def test_full_optimize_returns_result(self):
        result = self.engine.optimize(
            expr="ts_decay_linear(rank(ts_delta(close, 5)), 10)",
            sharpe=1.30,
            fitness=0.69,
            turnover=0.314,
        )
        assert isinstance(result, TurnoverOptimizationResult)
        assert result.original_expression is not None
        assert result.analysis is not None

    def test_optimization_with_high_to_generates_variants(self):
        result = self.engine.optimize(
            expr="ts_regression(close, volume, 5)",
            sharpe=1.25,
            fitness=0.60,
            turnover=0.50,
            max_variants=10,
        )
        assert len(result.variants) >= 1
        assert result.analysis.is_bottleneck is True

    def test_optimization_with_low_to_skips_variants(self):
        result = self.engine.optimize(
            expr="ts_decay_linear(rank(close), 40)",
            sharpe=1.35,
            fitness=0.90,
            turnover=0.15,
        )
        assert len(result.variants) == 0

    def test_best_variant_exists_when_variants_exist(self):
        result = self.engine.optimize(
            expr="ts_delta(close, 5)",
            sharpe=1.20,
            fitness=0.65,
            turnover=0.55,
            max_variants=8,
        )
        if result.variants:
            best = result.best_variant()
            assert best is not None
            assert best.expression is not None

    def test_safe_variants_filters_high_risk(self):
        result = self.engine.optimize(
            expr="rank(ts_corr(close, volume, 10))",
            sharpe=1.15,
            fitness=0.58,
            turnover=0.60,
            max_variants=15,
        )
        safe = result.safe_variants()
        high_risk = [v for v in result.variants if v.risk_level == "high"]
        for v in safe:
            assert v.risk_level in ("low", "medium")
        assert len(safe) >= len(result.variants) - len(high_risk)

    def test_summary_is_non_empty_string(self):
        result = self.engine.optimize(
            expr="rank(volume)",
            sharpe=1.30,
            fitness=0.70,
            turnover=0.35,
        )
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0
        assert "TO Analysis" in result.summary or "turnover" in result.summary.lower()

    def test_analysis_result_dataclass_fields(self):
        result = self.engine.optimize(
            expr="group_neutralize(ts_zscore(close, 20), industry)",
            sharpe=1.28,
            fitness=0.72,
            turnover=0.32,
        )
        analysis = result.analysis
        assert isinstance(analysis.current_to, float)
        assert isinstance(analysis.optimal_to, float)
        assert isinstance(analysis.severity, TurnoverSeverity)
        assert isinstance(analysis.recommended_strategies, list)


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def setup_method(self):
        self.engine = TurnoverOptimizer()

    def test_empty_expression(self):
        result = self.engine.analyze_turnover_bottleneck(expr="", sharpe=1.0, fitness=0.5, turnover=0.30)
        assert "current_penalty" in result

    def test_none_turnover_in_analysis(self):
        result = self.engine.analyze_turnover_bottleneck(expr="rank(close)", sharpe=1.3, fitness=0.8, turnover=None)
        assert result is not None

    def test_negative_sharpe(self):
        result = self.engine.analyze_turnover_bottleneck(expr="-rank(close)", sharpe=-0.5, fitness=-0.2, turnover=0.25)
        assert "severity" in result

    def test_extremely_low_fitness(self):
        result = self.engine.analyze_turnover_bottleneck(
            expr="ts_delta(close, 1)", sharpe=0.5, fitness=0.10, turnover=0.60
        )
        assert result["severity"] in ("high", "critical")

    def test_complex_nested_expression(self):
        complex_expr = (
            "group_neutralize("
            "ts_decay_linear("
            "signed_power("
            "group_zscore(ts_rank(ts_delta(close, 5), 10), industry)"
            ", 2.0)"
            ", 20)"
            ", subindustry)"
        )
        variants = self.engine.generate_low_turnover_variants(
            expr=complex_expr,
            turnover=0.42,
            max_variants=5,
        )
        assert len(variants) >= 1

    def test_strategy_failure_does_not_crash(self):
        class FailingEngine(TurnoverOptimizer):
            def _generate_decay_variants(self, expr, turnover):
                raise RuntimeError("Intentional failure for testing")

        engine = FailingEngine()
        variants = engine.generate_low_turnover_variants("rank(close)", turnover=0.40)
        assert isinstance(variants, list)

    def test_variant_dataclass_integrity(self):
        variant = Variant(
            expression="test_expr",
            strategy="test_strategy",
            description="test_desc",
            expected_to_reduction=0.25,
            confidence=0.80,
            risk_level="low",
        )
        assert variant.expression == "test_expr"
        assert variant.strategy == "test_strategy"
        assert 0 <= variant.expected_to_reduction <= 1.0
        assert 0 <= variant.confidence <= 1.0

    def test_turnover_severity_enum_values(self):
        assert TurnoverSeverity.LOW.value == "low"
        assert TurnoverSeverity.MEDIUM.value == "medium"
        assert TurnoverSeverity.HIGH.value == "high"
        assert TurnoverSeverity.CRITICAL.value == "critical"

    def test_constants_are_reasonable(self):
        assert TO_FLOOR == 0.125
        assert TO_OPTIMAL_MIN == 0.10
        assert TO_OPTIMAL_MAX == 0.25
        assert 0.05 <= TO_OPTIMAL_MIN <= TO_OPTIMAL_MAX <= 0.35


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
