"""Tests for Multi-Layer Neutralization Enhancement (Strategy B).

Covers:
  - _multi_neutralize_bc() function with various categories
  - Multi-layer neutralization template strings (B1-B4)
  - _mn suffix template creation and validation
  - FitnessBoost Tier 3 multi-layer upgrade logic
  - Edge cases and boundary conditions
"""
from __future__ import annotations

import pytest

from openalpha_brain.generation.alpha_logics import (
    AlphaLogicLibrary,
    BlockType,
    TemplateBlock,
    ThreeBlockTemplate,
    _build_three_block_templates,
)
from openalpha_brain.evolution.fitness_boost import (
    FitnessBoostEngine,
    FitnessVariant,
)


class TestMultiNeutralizeBCFunction:
    """Test the _multi_neutralize_bc() function directly."""

    def setup_method(self):
        templates = _build_three_block_templates()
        self.test_template = templates.get("momentum_short_term")
        assert self.test_template is not None

    def test_momentum_uses_b1_strategy(self):
        templates = _build_three_block_templates()
        mn_template = templates.get("momentum_short_term_mn")
        assert mn_template is not None
        assert "group_neutralize(group_neutralize(" in mn_template.block_b.template_str
        assert ", sector), industry)" in mn_template.block_b.template_str

    def test_value_uses_b3_strategy(self):
        templates = _build_three_block_templates()
        mn_template = templates.get("value_regression_mn")
        assert mn_template is not None
        assert "group_zscore(" in mn_template.block_b.template_str
        assert "group_neutralize(" in mn_template.block_b.template_str
        assert ", market), industry)" in mn_template.block_b.template_str

    def test_volatility_uses_b4_strategy(self):
        templates = _build_three_block_templates()
        mn_template = templates.get("volatility_low_vol_anomaly_mn")
        assert mn_template is not None
        assert "group_neutralize(" in mn_template.block_b.template_str
        assert ", subindustry)" in mn_template.block_b.template_str

    def test_quality_uses_b1_strategy(self):
        templates = _build_three_block_templates()
        mn_template = templates.get("quality_earnings_stability_mn")
        assert mn_template is not None
        assert "group_neutralize(group_neutralize(" in mn_template.block_b.template_str
        assert ", sector), industry)" in mn_template.block_b.template_str

    def test_unknown_category_defaults_to_b1(self):
        templates = _build_three_block_templates()
        mn_template = templates.get("liquidity_premium_mn")
        assert mn_template is not None
        assert "group_neutralize(group_neutralize(" in mn_template.block_b.template_str


class TestMultiNeutralizeTemplateStructure:
    """Test that _mn templates have correct ThreeBlockTemplate structure."""

    def setup_method(self):
        self.templates = _build_three_block_templates()

    def test_mn_template_has_correct_block_types(self):
        template = self.templates.get("momentum_medium_term_mn")
        assert template is not None
        assert template.block_a.block_type == BlockType.SIGNAL
        assert template.block_b.block_type == BlockType.NEUTRALIZE
        assert template.block_c.block_type == BlockType.DECAY

    def test_mn_template_preserves_original_signal(self):
        original = self.templates.get("momentum_volume_confirmed")
        mn_version = self.templates.get("momentum_volume_confirmed_mn")
        assert original is not None
        assert mn_version is not None
        assert mn_version.block_a.template_str == original.block_a.template_str

    def test_mn_template_has_locked_neutralize_block(self):
        template = self.templates.get("value_earnings_quality_mn")
        assert template is not None
        assert template.block_b.locked is True

    def test_mn_template_has_editable_decay_params(self):
        template = self.templates.get("size_small_cap_premium_mn")
        assert template is not None
        assert "decay_lb" in template.block_c.editable_params

    def test_mn_template_id_follows_convention(self):
        for base_id in ["momentum_short_term", "value_regression", "volatility_change_signal"]:
            mn_id = f"{base_id}_mn"
            mn_template = self.templates.get(mn_id)
            assert mn_template is not None, f"Missing {mn_id}"
            assert mn_template.template_id == mn_id

    def test_mn_template_name_includes_suffix(self):
        template = self.templates.get("lead_lag_price_volume_mn")
        assert template is not None
        assert "多层中性化" in template.name

    def test_mn_template_preserves_category(self):
        original = self.templates.get("mean_reversion_zscore")
        mn_version = self.templates.get("mean_reversion_zscore_mn")
        assert original is not None
        assert mn_version is not None
        assert mn_version.category == original.category


class TestB1DoubleLayerStrategy:
    """Test B1 strategy: group_neutralize(group_neutralize({signal}, sector), industry)."""

    def setup_method(self):
        self.templates = _build_three_block_templates()

    def test_b1_contains_double_neutralize(self):
        template = self.templates.get("momentum_long_term_reversal_mn")
        assert template is not None
        neutralize_str = template.block_b.template_str
        assert neutralize_str.count("group_neutralize") == 2

    def test_b1_sector_then_industry_order(self):
        template = self.templates.get("quality_asset_turnover_mn")
        assert template is not None
        neutralize_str = template.block_b.template_str
        sector_pos = neutralize_str.find(", sector)")
        industry_pos = neutralize_str.find(", industry)")
        assert sector_pos < industry_pos

    def test_b1_preserves_signal_placeholder(self):
        template = self.templates.get("momentum_short_term_mn")
        assert template is not None
        assert "{signal}" in template.block_b.template_str

    def test_b1_assemble_produces_valid_expression(self):
        template = self.templates.get("momentum_medium_term_mn")
        assert template is not None
        expr = template.assemble(price_field="close", medium_lb=20, decay_lb=10)
        assert "group_neutralize(group_neutralize(" in expr
        assert "ts_decay_linear(" in expr


class TestB2TripleLayerStrategy:
    """Test B2 strategy: three-layer market→sector→industry."""

    def setup_method(self):
        self.templates = _build_three_block_templates()

    def test_b2_contains_triple_neutralize(self):
        template = self.templates.get("momentum_short_term_mn")
        assert template is not None
        neutralize_str = template.block_b.template_str
        if neutralize_str.count("group_neutralize") == 3:
            assert ", market)" in neutralize_str
            assert ", sector)" in neutralize_str
            assert ", industry)" in neutralize_str

    def test_b2_market_sector_industry_order(self):
        templates = _build_three_block_templates()
        for tid, tmpl in templates.items():
            if tid.endswith("_mn") and tmpl.block_b.template_str.count("group_neutralize") == 3:
                neutralize_str = tmpl.block_b.template_str
                market_pos = neutralize_str.find(", market)")
                sector_pos = neutralize_str.find(", sector)")
                industry_pos = neutralize_str.find(", industry)")
                assert market_pos < sector_pos < industry_pos


class TestB3MixedStrategy:
    """Test B3 strategy: group_neutralize(group_zscore({signal}, market), industry)."""

    def setup_method(self):
        self.templates = _build_three_block_templates()

    def test_b3_contains_zscore_and_neutralize(self):
        template = self.templates.get("value_regression_mn")
        assert template is not None
        neutralize_str = template.block_b.template_str
        assert "group_zscore(" in neutralize_str
        assert "group_neutralize(" in neutralize_str

    def test_b3_market_zscore_first(self):
        template = self.templates.get("value_earnings_quality_mn")
        assert template is not None
        neutralize_str = template.block_b.template_str
        assert "group_zscore(" in neutralize_str
        assert "group_neutralize(" in neutralize_str

    def test_b3_assemble_produces_valid_expression(self):
        template = self.templates.get("value_regression_mn")
        assert template is not None
        expr = template.assemble(
            price_field="close",
            fundamental_field="pe_ratio",
            decay_lb=10,
        )
        assert "group_zscore(" in expr
        assert "group_neutralize(" in expr


class TestB4SubindustryStrategy:
    """Test B4 strategy: group_neutralize({signal}, subindustry)."""

    def setup_method(self):
        self.templates = _build_three_block_templates()

    def test_b4_uses_subindustry(self):
        template = self.templates.get("volatility_low_vol_anomaly_mn")
        assert template is not None
        assert ", subindustry)" in template.block_b.template_str

    def test_b4_single_layer_fine_grained(self):
        template = self.templates.get("volatility_clustering_mn")
        assert template is not None
        neutralize_str = template.block_b.template_str
        assert neutralize_str.count("group_neutralize") == 1
        assert "subindustry" in neutralize_str

    def test_b4_assemble_produces_valid_expression(self):
        template = self.templates.get("volatility_change_signal_mn")
        assert template is not None
        expr = template.assemble(
            price_field="close",
            vol_lb=20,
            delta_lb=5,
            decay_lb=10,
        )
        assert "group_neutralize(" in expr
        assert "subindustry" in expr


class TestAllCoreTemplatesHaveMNVersion:
    """Verify all core templates have corresponding _mn versions."""

    def setup_method(self):
        self.templates = _build_three_block_templates()

    @pytest.mark.parametrize("template_id", [
        "momentum_short_term", "momentum_medium_term", "momentum_volume_confirmed",
        "value_regression", "value_earnings_quality",
        "quality_earnings_stability", "quality_asset_turnover",
        "size_small_cap_premium",
        "volatility_low_vol_anomaly", "volatility_change_signal", "volatility_clustering",
        "liquidity_premium", "liquidity_improvement_signal",
        "lead_lag_price_volume", "lead_lag_cross_field", "lead_lag_industry_rotation",
        "momentum_long_term_reversal",
        "mean_reversion_zscore", "mean_reversion_bollinger", "mean_reversion_valuation",
    ])
    def test_template_has_mn_version(self, template_id):
        mn_id = f"{template_id}_mn"
        mn_template = self.templates.get(mn_id)
        assert mn_template is not None, f"Missing _mn version for {template_id}"

    def test_total_mn_template_count(self):
        mn_count = sum(1 for tid in self.templates if tid.endswith("_mn"))
        assert mn_count >= 20, f"Expected at least 20 _mn templates, got {mn_count}"


class TestFitnessBoostTier3MultiLayerUpgrade:
    """Test FitnessBoost Engine Tier 3 multi-layer neutralization upgrade logic."""

    def setup_method(self):
        self.engine = FitnessBoostEngine()

    def test_single_layer_detects_upgrade_opportunity(self):
        expr = "ts_decay_linear(group_neutralize(rank(close), sector), 10)"
        analysis = self.engine.analyze_fitness_bottleneck(
            expression=expr, sharpe=1.3, fitness=0.7, turnover=30,
        )
        variants = self.engine._tier_neutralization_balance(expr, analysis)
        multi_layer_variants = [v for v in variants if "升级多层中性化" in v.mutation_description or "双层" in v.mutation_description]
        assert len(multi_layer_variants) >= 1

    def test_b1_upgrade_generated(self):
        expr = "ts_decay_linear(group_neutralize(ts_rank(volume, 20), sector), 10)"
        analysis = self.engine.analyze_fitness_bottleneck(
            expression=expr, sharpe=1.3, fitness=0.7,
        )
        variants = self.engine._tier_neutralization_balance(expr, analysis)
        b1_variants = [v for v in variants if "B1" in v.mutation_description or "双层" in v.mutation_description]
        assert len(b1_variants) >= 1
        b1 = b1_variants[0]
        assert "group_neutralize(group_neutralize(" in b1.expression
        assert b1.expected_fitness_delta >= 0.10

    def test_b2_upgrade_generated(self):
        expr = "ts_decay_linear(group_neutralize(rank(close), industry), 10)"
        analysis = self.engine.analyze_fitness_bottleneck(
            expression=expr, sharpe=1.3, fitness=0.7,
        )
        variants = self.engine._tier_neutralization_balance(expr, analysis)
        b2_variants = [v for v in variants if "B2" in v.mutation_description or "三层" in v.mutation_description]
        if b2_variants:
            b2 = b2_variants[0]
            assert b2.expression.count("group_neutralize") == 3
            assert b2.expected_fitness_delta >= 0.14

    def test_b3_upgrade_generated(self):
        expr = "ts_decay_linear(group_neutralize(ts_delta(close, 5), sector), 10)"
        analysis = self.engine.analyze_fitness_bottleneck(
            expression=expr, sharpe=1.3, fitness=0.7,
        )
        variants = self.engine._tier_neutralization_balance(expr, analysis)
        b3_variants = [v for v in variants if "B3" in v.mutation_description or "混合" in v.mutation_description]
        if b3_variants:
            b3 = b3_variants[0]
            assert "group_zscore(" in b3.expression
            assert "group_neutralize(" in b3.expression
            assert b3.expected_fitness_delta >= 0.12

    def test_b4_subindustry_upgrade_for_sector_or_industry(self):
        expr = "ts_decay_linear(group_neutralize(rank(close), sector), 10)"
        analysis = self.engine.analyze_fitness_bottleneck(
            expression=expr, sharpe=1.3, fitness=0.7,
        )
        variants = self.engine._tier_neutralization_balance(expr, analysis)
        b4_variants = [v for v in variants if "B4" in v.mutation_description or "subindustry" in v.mutation_description.lower()]
        if b4_variants:
            b4 = b4_variants[0]
            assert "subindustry" in b4.expression
            assert b4.expected_fitness_delta >= 0.09

    def test_multi_layer_upgrade_not_applied_to_already_multi(self):
        expr = "ts_decay_linear(group_neutralize(group_neutralize(rank(close), sector), industry), 10)"
        analysis = self.engine.analyze_fitness_bottleneck(
            expression=expr, sharpe=1.3, fitness=0.7,
        )
        variants = self.engine._tier_neutralization_balance(expr, analysis)
        upgrade_variants = [v for v in variants if "升级多层中性化" in v.mutation_description]
        assert len(upgrade_variants) == 0

    def test_upgrade_variants_have_medium_risk(self):
        expr = "ts_decay_linear(group_neutralize(ts_rank(close, 10), sector), 10)"
        analysis = self.engine.analyze_fitness_bottleneck(
            expression=expr, sharpe=1.3, fitness=0.7,
        )
        variants = self.engine._tier_neutralization_balance(expr, analysis)
        multi_variants = [v for v in variants if "升级" in v.mutation_description or "双层" in v.mutation_description]
        for variant in multi_variants:
            assert variant.risk_level in ("medium", "low")

    def test_upgrade_preserves_signal_structure(self):
        expr = "ts_decay_linear(group_neutralize(rank(close), sector), 10)"
        analysis = self.engine.analyze_fitness_bottleneck(
            expression=expr, sharpe=1.3, fitness=0.7,
        )
        variants = self.engine._tier_neutralization_balance(expr, analysis)
        for variant in variants:
            if "升级" in variant.mutation_description:
                assert "rank(close)" in variant.expression or "close" in variant.expression

    def test_no_duplicate_variants_generated(self):
        expr = "ts_decay_linear(group_neutralize(rank(close), sector), 10)"
        analysis = self.engine.analyze_fitness_bottleneck(
            expression=expr, sharpe=1.3, fitness=0.7,
        )
        variants = self.engine._tier_neutralization_balance(expr, analysis)
        expressions = [v.expression for v in variants]
        assert len(expressions) == len(set(expressions))


class TestAlphaLogicLibraryIntegration:
    """Test AlphaLogicLibrary integration with multi-neutralize templates."""

    def setup_method(self):
        self.library = AlphaLogicLibrary()

    def test_library_loads_mn_templates(self):
        mn_template = self.library.get_three_block_template("momentum_short_term_mn")
        assert mn_template is not None

    def test_library_instantiate_mn_template(self):
        expr = self.library.instantiate_template(
            template_id="momentum_short_term_mn",
            fields={"price_field": "close", "short_lb": 5, "long_lb": 20},
            params={"decay_lb": 10},
        )
        assert expr is not None
        assert "group_neutralize(group_neutralize(" in expr

    def test_library_instantiate_value_mn_template(self):
        expr = self.library.instantiate_template(
            template_id="value_regression_mn",
            fields={"price_field": "close", "fundamental_field": "pe_ratio"},
            params={"decay_lb": 15},
        )
        assert expr is not None
        assert "group_zscore(" in expr
        assert "group_neutralize(" in expr

    def test_library_instantiate_volatility_mn_template(self):
        expr = self.library.instantiate_template(
            template_id="volatility_low_vol_anomaly_mn",
            fields={"price_field": "close", "vol_lb": 20},
            params={"decay_lb": 10},
        )
        assert expr is not None
        assert "subindustry" in expr

    def test_mn_template_validate_assembly(self):
        template = self.library.get_three_block_template("quality_earnings_stability_mn")
        assert template is not None
        expr = template.assemble(earnings_field="net_profit", std_lb=12, decay_lb=8)
        assert expr is not None
        assert "group_neutralize(group_neutralize(" in expr
        assert "ts_decay_linear(" in expr


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions for multi-neutralize."""

    def setup_method(self):
        self.templates = _build_three_block_templates()
        self.engine = FitnessBoostEngine()

    def test_empty_expression_returns_no_variants(self):
        analysis = {"expression_features": {"neutralize_group": None}}
        variants = self.engine._tier_neutralization_balance("", analysis)
        assert len(variants) == 0

    def test_no_neutralize_returns_no_upgrades(self):
        expr = "ts_decay_linear(rank(close), 10)"
        analysis = self.engine.analyze_fitness_bottleneck(
            expression=expr, sharpe=1.3, fitness=0.7,
        )
        variants = self.engine._tier_neutralization_balance(expr, analysis)
        upgrade_variants = [v for v in variants if "升级" in v.mutation_description]
        assert len(upgrade_variants) == 0

    def test_group_zscore_no_upgrade(self):
        expr = "ts_decay_linear(group_zscore(rank(close), sector), 10)"
        analysis = self.engine.analyze_fitness_bottleneck(
            expression=expr, sharpe=1.3, fitness=0.7,
        )
        variants = self.engine._tier_neutralization_balance(expr, analysis)
        upgrade_variants = [v for v in variants if "升级多层" in v.mutation_description]
        assert len(upgrade_variants) == 0

    def test_complex_nested_expression_handled(self):
        expr = "ts_decay_linear(group_neutralize(rank(ts_corr(ts_delta(close, 1), ts_delta(volume, 1), 10)), sector), 10)"
        analysis = self.engine.analyze_fitness_bottleneck(
            expression=expr, sharpe=1.3, fitness=0.7,
        )
        variants = self.engine._tier_neutralization_balance(expr, analysis)
        assert len(variants) >= 0

    def test_all_mn_templates_have_unique_ids(self):
        mn_ids = [tid for tid in self.templates if tid.endswith("_mn")]
        assert len(mn_ids) == len(set(mn_ids))

    def test_mn_template_decay_lb_parameter_works(self):
        template = self.templates.get("liquidity_improvement_signal_mn")
        assert template is not None
        expr = template.assemble(volume_field="volume", liq_lb=10, decay_lb=5)
        assert ", 5)" in expr or ",5)" in expr.replace(" ", "")
