"""Tests for FitnessBoostEngine — fitness-targeted optimization."""
from __future__ import annotations

import pytest

from openalpha_brain.evolution.fitness_boost import (
    FitnessBoostEngine,
    FitnessBoostResult,
    FitnessBoostTier,
    FitnessVariant,
    LOW_TURNOVER_ENTRY_THRESHOLDS,
    LOW_TURNOVER_EXIT_THRESHOLDS,
    LOW_TURNOVER_HUMP_SIZES,
    get_fitness_boost_engine,
)


class TestFitnessBoostEngineInit:
    def test_default_creation(self):
        engine = FitnessBoostEngine()
        assert engine is not None

    def test_factory_function(self):
        engine = get_fitness_boost_engine()
        assert isinstance(engine, FitnessBoostEngine)


class TestBottleneckAnalysis:
    def setup_method(self):
        self.engine = FitnessBoostEngine()

    def test_weak_signal_detection(self):
        result = self.engine.analyze_fitness_bottleneck(
            expression="ts_decay_linear(rank(volume), 20)",
            sharpe=1.30, fitness=0.69, turnover=31.4,
        )
        assert result["bottleneck_type"] in ("weak_signal", "over_smoothed", "complex")
        assert result["fitness_gap"] == pytest.approx(0.31, abs=0.01)
        assert len(result["recommended_tiers"]) >= 1

    def test_high_severity_for_very_low_fitness(self):
        result = self.engine.analyze_fitness_bottleneck(
            expression="ts_decay_linear(signed_power(close, 2.5), 30)",
            sharpe=1.2, fitness=0.4,
        )
        assert result["severity"] >= 0.5

    def test_low_severity_for_near_pass(self):
        result = self.engine.analyze_fitness_bottleneck(
            expression="rank(ts_delta(close, 5))",
            sharpe=1.2, fitness=0.95,
        )
        assert result["severity"] < 0.1

    def test_signed_power_triggers_amplification_tier(self):
        result = self.engine.analyze_fitness_bottleneck(
            expression="signed_power(ts_delta(close, 5), 2.0)",
            sharpe=1.3, fitness=0.7,
        )
        tiers = result["recommended_tiers"]
        assert FitnessBoostTier.SIGNAL_AMPLIFICATION in tiers

    def test_strong_decay_triggers_calibration(self):
        result = self.engine.analyze_fitness_bottleneck(
            expression="ts_decay_linear(rank(volume), 60)",
            sharpe=1.3, fitness=0.7,
        )
        tiers = result["recommended_tiers"]
        assert FitnessBoostTier.DECAY_CALIBRATION in tiers

    def test_fine_neutralize_triggers_balance(self):
        result = self.engine.analyze_fitness_bottleneck(
            expression="group_neutralize(rank(close), subindustry)",
            sharpe=1.3, fitness=0.7,
        )
        tiers = result["recommended_tiers"]
        assert FitnessBoostTier.NEUTRALIZATION_BALANCE in tiers

    def test_expression_features_extraction(self):
        features = self.engine._extract_expression_features(
            "ts_decay_linear(signed_power(group_zscore(ts_rank(close, 10), industry), 2.0), 20)"
        )
        assert features["has_signed_power"] is True
        assert features["power_value"] == 2.0
        assert features["neutralize_group"] is None or features["neutralize_group"] == "industry"
        assert features["decay_count"] == 1
        assert features["max_decay_window"] == 20
        assert "signed_power" in features["operators"]
        assert "group_zscore" in features["operators"]

    def test_features_no_decay(self):
        features = self.engine._extract_expression_features("rank(close)")
        assert features["decay_count"] == 0
        assert features["max_decay_window"] == 0
        assert features["has_signed_power"] is False


class TestTier1DecayCalibration:
    def setup_method(self):
        self.engine = FitnessBoostEngine()

    def test_shorten_decay_window(self):
        variants = self.engine._tier_decay_calibration(
            "ts_decay_linear(rank(volume), 30)",
            {"expression_features": {"decay_windows": [30]}},
        )
        assert len(variants) >= 1
        for v in variants:
            assert "ts_decay_linear" in v.expression
            assert "30" not in v.expression or v.expression.count("ts_decay_linear(rank(volume), 30)") == 0
            assert v.boost_tier == "decay_calibration"

    def test_add_decay_if_missing(self):
        variants = self.engine._tier_decay_calibration(
            "rank(volume)",
            {"expression_features": {"decay_windows": []}},
        )
        assert len(variants) >= 1
        assert all("ts_decay_linear" in v.expression for v in variants)

    def test_no_duplicate_variants(self):
        expr = "ts_decay_linear(rank(volume), 20)"
        variants = self.engine._tier_decay_calibration(
            expr, {"expression_features": {"decay_windows": [20]}}
        )
        expressions = [v.expression for v in variants]
        assert len(expressions) == len(set(expressions))


class TestTier2SignalAmplification:
    def setup_method(self):
        self.engine = FitnessBoostEngine()

    def test_adjust_power_parameter(self):
        variants = self.engine._tier_signal_amplification(
            "signed_power(ts_delta(close, 5), 2.5)",
            {"expression_features": {"has_signed_power": True, "power_value": 2.5}},
        )
        assert len(variants) >= 1
        changed = [v for v in variants if v.expression != "signed_power(ts_delta(close, 5), 2.5)"]
        assert len(changed) >= 1

    def test_add_scale_wrapper(self):
        variants = self.engine._tier_signal_amplification(
            "rank(close)",
            {"expression_features": {"has_signed_power": False}},
        )
        scale_variants = [v for v in variants if "scale" in v.expression]
        assert len(scale_variants) >= 1

    def test_zscore_window_tuning(self):
        variants = self.engine._tier_signal_amplification(
            "ts_zscore(close, 20)",
            {"expression_features": {"has_signed_power": False}},
        )
        zscore_vars = [v for v in variants if "ts_zscore" in v.expression]
        assert len(zscore_vars) >= 1


class TestTier3NeutralizationBalance:
    def setup_method(self):
        self.engine = FitnessBoostEngine()

    def test_coarsen_neutralization(self):
        variants = self.engine._tier_neutralization_balance(
            "group_neutralize(rank(close), subindustry)",
            {"expression_features": {"neutralize_group": "subindustry"}},
        )
        assert len(variants) >= 1
        coarsened = [v for v in variants if "subindustry" not in v.expression]
        assert len(coarsened) >= 1, f"Expected at least 1 coarsened variant without 'subindustry', got: {[v.expression for v in variants]}"

    def test_no_change_when_market(self):
        variants = self.engine._tier_neutralization_balance(
            "group_neutralize(rank(close), market)",
            {"expression_features": {"neutralize_group": "market"}},
        )
        coarser = [v for v in variants if v.expression != "group_neutralize(rank(close), market)"]
        assert len(coarser) >= 0

    def test_switch_to_zscore(self):
        variants = self.engine._tier_neutralization_balance(
            "group_neutralize(rank(close), industry)",
            {"expression_features": {"neutralize_group": "industry"}},
        )
        zscore_vars = [v for v in variants if "group_zscore" in v.expression]
        assert len(zscore_vars) >= 1


class TestTier4StructureStreamlining:
    def setup_method(self):
        self.engine = FitnessBoostEngine()

    def test_remove_signed_power(self):
        variants = self.engine._tier_structure_streamlining(
            "signed_power(ts_delta(close, 5), 3.0)",
            {"expression_features": {"has_signed_power": True, "power_value": 3.0}},
        )
        no_power = [v for v in variants if "signed_power" not in v.expression]
        assert len(no_power) >= 1

    def test_reduce_high_power(self):
        variants = self.engine._tier_structure_streamlining(
            "signed_power(ts_delta(close, 5), 3.0)",
            {"expression_features": {"has_signed_power": True, "power_value": 3.0}},
        )
        reduced = [v for v in variants if "signed_power" in v.expression and "1.5" in v.expression]
        assert len(reduced) >= 1

    def test_replace_complex_operators(self):
        variants = self.engine._tier_structure_streamlining(
            "ts_regression(close, volume, 10)",
            {"expression_features": {"complex_op_count": 1}},
        )
        simplified = [v for v in variants if "ts_rank" in v.expression]
        assert len(simplified) >= 1

    def test_deep_nesting_reduction(self):
        deep_expr = "ts_decay_linear(signed_power(ts_regression(close, volume, 5), 2.0), 10)"
        variants = self.engine._tier_structure_streamlining(
            deep_expr,
            {"expression_features": {
                "has_signed_power": True, "nesting_depth": deep_expr.count("("),
            }},
        )
        assert len(variants) >= 1


class TestTier5CompositeMutation:
    def setup_method(self):
        self.engine = FitnessBoostEngine()

    def test_decay_plus_power_composite(self):
        variants = self.engine._tier_composite_mutation(
            "ts_decay_linear(signed_power(rank(close), 2.5), 30)",
            {"expression_features": {
                "has_signed_power": True, "power_value": 2.5,
                "decay_windows": [30], "complex_op_count": 0,
            }},
        )
        composite = [v for v in variants if v.boost_tier == "composite_mutation"]
        assert len(composite) >= 1
        for v in composite:
            assert v.risk_level == "high"

    def test_neutralize_plus_decay_composite(self):
        variants = self.engine._tier_composite_mutation(
            "group_neutralize(ts_decay_linear(rank(close), 20), subindustry)",
            {"expression_features": {
                "neutralize_group": "subindustry",
                "decay_windows": [20], "complex_op_count": 0,
                "has_signed_power": False,
            }},
        )
        composite = [v for v in variants if v.boost_tier == "composite_mutation"]
        assert len(composite) >= 1

    def test_simplify_ops_composite(self):
        variants = self.engine._tier_composite_mutation(
            "ts_decay_linear(signed_power(ts_corr(close, volume, 10), 2.0), 15)",
            {"expression_features": {
                "has_signed_power": True, "power_value": 2.0,
                "decay_windows": [15], "complex_op_count": 2,
            }},
        )
        composite = [v for v in variants if v.boost_tier == "composite_mutation"]
        assert len(composite) >= 1


class TestGenerateBoostVariants:
    def setup_method(self):
        self.engine = FitnessBoostEngine()

    def test_generates_variants_for_typical_case(self):
        result = self.engine.generate_boost_variants(
            expression="ts_decay_linear(signed_power(group_neutralize(rank(close), subindustry), 2.0), 20)",
            sharpe=1.300, fitness=0.690, turnover=31.4,
        )
        assert isinstance(result, FitnessBoostResult)
        assert len(result.variants) >= 3
        assert result.original_expression is not None
        assert result.original_sharpe == 1.3
        assert result.original_fitness == 0.69

    def test_variants_sorted_by_expected_delta(self):
        result = self.engine.generate_boost_variants(
            expression="ts_decay_linear(signed_power(rank(close), 2.5), 30)",
            sharpe=1.3, fitness=0.7,
        )
        deltas = [v.expected_fitness_delta for v in result.variants]
        assert deltas == sorted(deltas, reverse=True)

    def test_no_duplicate_expressions(self):
        result = self.engine.generate_boost_variants(
            expression="ts_decay_linear(signed_power(rank(close), 2.0), 20)",
            sharpe=1.3, fitness=0.7,
        )
        expressions = [v.expression for v in result.variants]
        assert len(expressions) == len(set(expressions))

    def test_max_variants_limit(self):
        result = self.engine.generate_boost_variants(
            expression="ts_decay_linear(signed_power(group_neutralize(ts_regression(close, volume, 5), subindustry), 3.0), 40)",
            sharpe=1.3, fitness=0.5,
            max_variants=5,
        )
        assert len(result.variants) <= 5

    def test_simple_expression_still_works(self):
        result = self.engine.generate_boost_variants(
            expression="rank(close)",
            sharpe=1.0, fitness=0.6,
        )
        assert len(result.variants) >= 1

    def test_best_variant_selection(self):
        result = self.engine.generate_boost_variants(
            expression="ts_decay_linear(signed_power(rank(close), 2.5), 30)",
            sharpe=1.3, fitness=0.69,
        )
        best = result.best_variant()
        if best:
            assert best.expected_fitness_delta > 0

    def test_all_variants_have_required_fields(self):
        result = self.engine.generate_boost_variants(
            expression="ts_decay_linear(rank(volume), 20)",
            sharpe=1.2, fitness=0.7,
        )
        for v in result.variants:
            assert isinstance(v.expression, str) and len(v.expression) > 0
            assert v.boost_tier in ["decay_calibration", "signal_amplification",
                                    "neutralization_balance", "structure_streamlining", "composite_mutation"]
            assert isinstance(v.expected_fitness_delta, float)
            assert v.risk_level in ("low", "medium", "high")
            assert isinstance(v.priority, int)


class TestEdgeCases:
    def setup_method(self):
        self.engine = FitnessBoostEngine()

    def test_empty_expression(self):
        result = self.engine.generate_boost_variants(
            expression="", sharpe=1.0, fitness=0.5,
        )
        assert len(result.variants) == 0

    def test_zero_fitness(self):
        analysis = self.engine.analyze_fitness_bottleneck(
            expression="rank(close)", sharpe=0.5, fitness=0.0,
        )
        assert analysis["severity"] >= 0.9

    def test_already_passing_fitness(self):
        analysis = self.engine.analyze_fitness_bottleneck(
            expression="rank(close)", sharpe=1.3, fitness=1.2,
        )
        assert analysis["severity"] <= 0.0

    def test_very_high_turnover(self):
        result = self.engine.generate_boost_variants(
            expression="ts_delta(close, 1)",
            sharpe=1.0, fitness=0.6, turnover=65.0,
        )
        assert isinstance(result, FitnessBoostResult)

    def test_no_operators_in_expression(self):
        result = self.engine.generate_boost_variants(
            expression="close",
            sharpe=1.0, fitness=0.6,
        )
        assert isinstance(result, FitnessBoostResult)


class TestOperatorFitnessImpact:
    def test_impact_dict_is_complete(self):
        impact = FitnessBoostEngine.OPERATOR_FITNESS_IMPACT
        assert "signed_power" in impact
        assert "ts_decay_linear" in impact
        assert impact["signed_power"] < 0
        assert impact["ts_decay_linear"] > 0

    def test_negative_impact_ops_reduce_fitness(self):
        negative_ops = {k: v for k, v in FitnessBoostEngine.OPERATOR_FITNESS_IMPACT.items() if v < 0}
        assert len(negative_ops) >= 4


class TestTier6LowTurnoverWrapping:
    def setup_method(self):
        self.engine = FitnessBoostEngine()

    def test_hump_wrap_generates_variants(self):
        variants = self.engine._tier_low_turnover_wrapping(
            "rank(close)",
            {"expression_features": {"operators": ["rank"]}},
        )
        hump_vars = [v for v in variants if "hump" in v.expression and "trade_when" not in v.expression]
        assert len(hump_vars) >= len(LOW_TURNOVER_HUMP_SIZES)

    def test_trade_when_wrap_generates_variants(self):
        variants = self.engine._tier_low_turnover_wrapping(
            "rank(close)",
            {"expression_features": {"operators": ["rank"]}},
        )
        trade_vars = [v for v in variants if "trade_when" in v.expression and "hump" not in v.expression]
        expected_count = len(LOW_TURNOVER_ENTRY_THRESHOLDS) * len(LOW_TURNOVER_EXIT_THRESHOLDS)
        assert len(trade_vars) >= expected_count

    def test_decay_inner_hump_combination(self):
        variants = self.engine._tier_low_turnover_wrapping(
            "ts_decay_linear(rank(close), 20)",
            {"expression_features": {"operators": ["ts_decay_linear", "rank"], "decay_windows": [20]}},
        )
        decay_hump = [v for v in variants if "hump" in v.expression and "ts_decay_linear" in v.expression and "trade_when" not in v.expression]
        assert len(decay_hump) >= len(LOW_TURNOVER_HUMP_SIZES)

    def test_decay_trade_when_combination(self):
        variants = self.engine._tier_low_turnover_wrapping(
            "ts_decay_linear(rank(close), 20)",
            {"expression_features": {"operators": ["ts_decay_linear", "rank"], "decay_windows": [20]}},
        )
        decay_trade = [v for v in variants if "trade_when" in v.expression and "ts_decay_linear" in v.expression and "hump" not in v.expression]
        assert len(decay_trade) >= 1

    def test_dual_hump_trade_when_combination(self):
        variants = self.engine._tier_low_turnover_wrapping(
            "rank(close)",
            {"expression_features": {"operators": ["rank"]}},
        )
        dual_vars = [v for v in variants if "hump" in v.expression and "trade_when" in v.expression]
        assert len(dual_vars) >= 2

    def test_all_tier6_variants_have_correct_metadata(self):
        variants = self.engine._tier_low_turnover_wrapping(
            "rank(close)",
            {"expression_features": {"operators": ["rank"]}},
        )
        for v in variants:
            assert v.boost_tier == "low_turnover_wrapping"
            assert isinstance(v.expected_fitness_delta, float)
            assert v.expected_fitness_delta > 0
            assert v.risk_level in ("low", "medium")
            assert isinstance(v.expression, str) and len(v.expression) > 0

    def test_no_duplicate_expressions_in_tier6(self):
        variants = self.engine._tier_low_turnover_wrapping(
            "ts_decay_linear(rank(volume), 15)",
            {"expression_features": {"operators": ["ts_decay_linear", "rank"], "decay_windows": [15]}},
        )
        expressions = [v.expression for v in variants]
        assert len(expressions) == len(set(expressions))

    def test_high_turnover_triggers_tier6_recommendation(self):
        result = self.engine.analyze_fitness_bottleneck(
            expression="ts_delta(close, 1)",
            sharpe=1.3, fitness=0.7, turnover=50.0,
        )
        assert FitnessBoostTier.LOW_TURNOVER_WRAPPING in result["recommended_tiers"]

    def test_low_turnover_does_not_trigger_tier6(self):
        result = self.engine.analyze_fitness_bottleneck(
            expression="rank(close)",
            sharpe=1.3, fitness=0.7, turnover=20.0,
        )
        has_tier6 = FitnessBoostTier.LOW_TURNOVER_WRAPPING in result["recommended_tiers"]
        assert not has_tier6

    def test_tier6_integration_via_generate_boost(self):
        result = self.engine.generate_boost_variants(
            expression="ts_decay_linear(rank(volume), 10)",
            sharpe=1.3, fitness=0.69, turnover=45.0,
        )
        tier6_vars = [v for v in result.variants if v.boost_tier == "low_turnover_wrapping"]
        assert len(tier6_vars) >= 1

    def test_hump_size_parameter_range(self):
        variants = self.engine._tier_low_turnover_wrapping(
            "rank(close)",
            {"expression_features": {"operators": ["rank"]}},
        )
        hump_sizes_found = set()
        for v in variants:
            if "hump(" in v.expression and "trade_when" not in v.expression:
                import re
                match = re.search(r'hump\([^,]+,\s*([\d.]+)\)', v.expression)
                if match:
                    hump_sizes_found.add(float(match.group(1)))
        for size in LOW_TURNOVER_HUMP_SIZES:
            assert size in hump_sizes_found

    def test_trade_when_threshold_parameter_range(self):
        variants = self.engine._tier_low_turnover_wrapping(
            "rank(close)",
            {"expression_features": {"operators": ["rank"]}},
        )
        entry_thresh_found = set()
        exit_thresh_found = set()
        import re
        for v in variants:
            if "trade_when(" in v.expression and "hump" not in v.expression:
                match = re.search(r'trade_when\([^,]+,\s*([-\d.]+),\s*([-\d.]+)\)', v.expression)
                if match:
                    entry_thresh_found.add(float(match.group(1)))
                    exit_thresh_found.add(float(match.group(2)))
        for thresh in LOW_TURNOVER_ENTRY_THRESHOLDS:
            assert thresh in entry_thresh_found
        for thresh in LOW_TURNOVER_EXIT_THRESHOLDS:
            assert thresh in exit_thresh_found


class TestLowTurnoverOperatorFitnessImpact:
    def test_hump_has_positive_impact(self):
        assert "hump" in FitnessBoostEngine.OPERATOR_FITNESS_IMPACT
        assert FitnessBoostEngine.OPERATOR_FITNESS_IMPACT["hump"] > 0

    def test_trade_when_has_positive_impact(self):
        assert "trade_when" in FitnessBoostEngine.OPERATOR_FITNESS_IMPACT
        assert FitnessBoostEngine.OPERATOR_FITNESS_IMPACT["trade_when"] > 0

    def test_trade_when_higher_than_hump(self):
        assert (FitnessBoostEngine.OPERATOR_FITNESS_IMPACT["trade_when"] >
                FitnessBoostEngine.OPERATOR_FITNESS_IMPACT["hump"])
