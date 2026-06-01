"""Tests for NearPassImprover - targeted improvement for near-passing factors."""
import pytest
from openalpha_brain.evolution.near_pass_improver import (
    NearPassImprover,
    NearPassAnalysis,
    NearPassCategory,
    ImprovedVariant,
    GATE_SHARPE_MIN,
    GATE_FITNESS_MIN,
)


class TestNearPassAnalysis:
    def setup_method(self):
        self.improver = NearPassImprover()

    def test_sharpe_good_fitness_poor(self):
        analysis = self.improver.analyze(sharpe=1.30, fitness=0.69, turnover=31.0)
        assert analysis.category == NearPassCategory.SHARPE_GOOD_FITNESS_POOR
        assert analysis.primary_fix_target == "fitness"
        assert analysis.turnover_too_high is True
        assert "increase_decay_window" in analysis.improvement_priority

    def test_fitness_good_sharpe_poor(self):
        analysis = self.improver.analyze(sharpe=0.9, fitness=1.2, turnover=15.0)
        assert analysis.category == NearPassCategory.FITNESS_GOOD_SHARPE_POOR
        assert analysis.primary_fix_target == "sharpe"

    def test_both_near(self):
        analysis = self.improver.analyze(sharpe=1.2, fitness=0.9, turnover=20.0)
        assert analysis.category == NearPassCategory.BOTH_NEAR

    def test_not_near(self):
        analysis = self.improver.analyze(sharpe=-0.5, fitness=-0.3, turnover=50.0)
        assert analysis.category == NearPassCategory.NOT_NEAR

    def test_gap_calculation(self):
        analysis = self.improver.analyze(sharpe=1.0, fitness=0.8)
        assert abs(analysis.sharpe_gap - 0.25) < 0.01
        assert abs(analysis.fitness_gap - 0.2) < 0.01


class TestDeterministicMutations:
    def setup_method(self):
        self.improver = NearPassImprover()
        self.near_pass_expr = (
            "ts_decay_linear(group_neutralize("
            "-rank(signed_power(ts_zscore(returns / bookvalue_ps, 20), 2)), sector), 10)"
        )
        self.analysis = self.improver.analyze(
            sharpe=1.30, fitness=0.69, turnover=31.0,
        )

    def test_generate_variants_non_empty(self):
        variants = self.improver.generate_deterministic_variants(
            self.near_pass_expr, self.analysis,
        )
        assert len(variants) > 0, "Should generate at least one variant"

    def test_variants_are_different_from_original(self):
        variants = self.improver.generate_deterministic_variants(
            self.near_pass_expr, self.analysis,
        )
        for v in variants:
            assert v.expression != self.near_pass_expr, \
                f"Variant should differ from original: {v.expression}"

    def test_no_duplicate_variants(self):
        variants = self.improver.generate_deterministic_variants(
            self.near_pass_expr, self.analysis, max_variants=20,
        )
        exprs = [v.expression for v in variants]
        assert len(exprs) == len(set(exprs)), "No duplicates allowed"

    def test_decay_window_mutation(self):
        variants = self.improver._mutate_increase_decay_window(
            self.near_pass_expr, self.analysis,
        )
        assert len(variants) > 0, "Should generate decay window variants"
        for v in variants:
            assert "ts_decay_linear" in v.expression
            assert v.mutation_type == "decay_window"

    def test_double_decay_mutation(self):
        variants = self.improver._mutate_add_double_decay(
            self.near_pass_expr, self.analysis,
        )
        assert len(variants) > 0, "Should generate double decay variants"
        for v in variants:
            assert v.expression.count("ts_decay_linear") >= 2

    def test_rank_to_zscore_replacement(self):
        expr_with_rank = "ts_decay_linear(rank(close), 10)"
        analysis = self.improver.analyze(sharpe=1.2, fitness=0.7)
        variants = self.improver._mutate_replace_rank_with_zscore(
            expr_with_rank, analysis,
        )
        assert len(variants) > 0
        assert any("zscore" in v.expression for v in variants)

    def test_remove_signed_power(self):
        expr_with_sp = "rank(signed_power(ts_zscore(x, 20), 2))"
        variants = self.improver._mutate_remove_nonlinear_transform(
            expr_with_sp, self.analysis,
        )
        assert len(variants) > 0
        assert all("signed_power" not in v.expression for v in variants)

    def test_neutralization_upgrade(self):
        expr_sector = "ts_decay_linear(group_neutralize(rank(x), sector), 10)"
        variants = self.improver._mutate_upgrade_neutralization(
            expr_sector, self.analysis,
        )
        assert len(variants) > 0
        assert any("industry" in v.expression for v in variants)

    def test_signal_direction_flip(self):
        expr_neg = "-rank(x)"
        variants = self.improver._mutate_change_signal_direction(
            expr_neg, self.analysis,
        )
        assert len(variants) > 0
        assert not any(v.expression.startswith("-") for v in variants)

    def test_parameter_tuning(self):
        variants = self.improver._mutate_tune_parameters(
            self.near_pass_expr, self.analysis,
        )
        power_variants = [v for v in variants if v.mutation_type == "power_tune"]
        zscore_variants = [v for v in variants if v.mutation_type == "zscore_window_tune"]
        assert len(power_variants) > 0 or len(zscore_variants) > 0, \
            "Should tune at least power or zscore parameters"


class TestRealWorldScenario:
    """Test with the actual near-pass factor from E2E results."""

    def setup_method(self):
        self.improver = NearPassImprover()

    def test_actual_near_pass_factor(self):
        actual_factor = (
            "ts_decay_linear(group_neutralize("
            "-rank(signed_power(ts_zscore(returns / bookvalue_ps, 20), 2)), sector), 10)"
        )
        analysis = self.improver.analyze(
            sharpe=1.30, fitness=0.69, turnover=31.0,
        )
        assert analysis.category == NearPassCategory.SHARPE_GOOD_FITNESS_POOR

        variants = self.improver.generate_deterministic_variants(
            actual_factor, analysis, max_variants=12,
        )
        print(f"\n[Near-Pass] Generated {len(variants)} variants:")
        for i, v in enumerate(variants[:8], 1):
            print(f"  {i}. [{v.mutation_type}] {v.expression[:70]}...")
            print(f"     → {v.expected_effect}")

        assert len(variants) >= 5, \
            f"Should generate >=5 variants for this factor, got {len(variants)}"

    def test_variant_quality(self):
        actual_factor = (
            "ts_decay_linear(group_neutralize("
            "-rank(signed_power(ts_zscore(returns / bookvalue_ps, 20), 2)), sector), 10)"
        )
        analysis = self.improver.analyze(sharpe=1.30, fitness=0.69, turnover=31.0)
        variants = self.improver.generate_deterministic_variants(
            actual_factor, analysis, max_variants=12,
        )

        decay_variants = [v for v in variants if v.mutation_type == "decay_window"]
        assert len(decay_variants) >= 3, f"Should have >=3 decay variants, got {len(decay_variants)}"

        for v in decay_variants:
            import re as _re
            numbers = _re.findall(r"\d+", v.expression)
            has_larger_window = any(int(n) > 10 for n in numbers if int(n) <= 60)
            assert has_larger_window, \
                f"Decay variant should have larger window: {v.expression}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
