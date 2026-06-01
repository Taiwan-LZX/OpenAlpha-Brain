import pytest
from openalpha_brain.knowledge.operator_registry import (
    OperatorCategory,
    OperatorDef,
    OperatorRegistry,
    get_operator_registry,
)


class TestOperatorDef:
    def test_operator_def_creation(self):
        op = OperatorDef(
            name="ts_mean",
            category=OperatorCategory.TEMPORAL,
            min_args=2,
            max_args=2,
            args_type=["field", "window"],
            requires_lookback=True,
            default_lookback=20,
            description="时间序列均值",
            risk_level="medium",
        )
        assert op.name == "ts_mean"
        assert op.category == OperatorCategory.TEMPORAL
        assert op.requires_lookback is True
        assert op.default_lookback == 20
        assert op.use_count == 0
        assert op.success_count == 0
        assert op.avg_sharpe == 0.0

    def test_operator_def_defaults(self):
        op = OperatorDef(
            name="test_op",
            category=OperatorCategory.MATH,
            min_args=1,
            max_args=1,
        )
        assert op.args_type == []
        assert op.requires_lookback is False
        assert op.default_lookback is None
        assert op.description == ""
        assert op.risk_level == "medium"
        assert op.alternatives == []
        assert op.forbidden_combos == []


class TestOperatorRegistry:
    @pytest.fixture
    def registry(self):
        return OperatorRegistry()

    def test_register_and_get(self, registry):
        op_def = OperatorDef(
            name="custom_op",
            category=OperatorCategory.MATH,
            min_args=1,
            max_args=1,
            description="自定义算子",
        )
        registry.register(op_def)

        retrieved = registry.get("custom_op")
        assert retrieved is not None
        assert retrieved.name == "custom_op"
        assert retrieved.description == "自定义算子"

    def test_get_nonexistent_operator(self, registry):
        result = registry.get("nonexistent_operator")
        assert result is None

    def test_get_by_category(self, registry):
        temporal_ops = registry.get_by_category(OperatorCategory.TEMPORAL)
        assert len(temporal_ops) > 0
        for op in temporal_ops:
            assert op.category == OperatorCategory.TEMPORAL

        math_ops = registry.get_by_category(OperatorCategory.MATH)
        assert len(math_ops) > 0

    def test_get_temporal_operators(self, registry):
        temporal = registry.get_temporal_operators()
        assert len(temporal) > 0
        for op in temporal:
            assert op.category == OperatorCategory.TEMPORAL
        lookback_ops = [op for op in temporal if op.requires_lookback]
        assert len(lookback_ops) >= 20, "大部分时间序列算子应该需要 lookback"

    def test_get_low_risk_operators(self, registry):
        low_risk = registry.get_low_risk_operators()
        assert len(low_risk) > 0
        for op in low_risk:
            assert op.risk_level == "low"

        key_low_risk_ops = ["ts_decay_linear", "rank", "zscore", "ts_av_diff", "ts_corr"]
        low_risk_names = {op.name for op in low_risk}
        for key_op in key_low_risk_ops:
            assert key_op in low_risk_names, f"{key_op} 应该是低风险算子"

    def test_get_alternatives(self, registry):
        alts = registry.get_alternatives("ts_delta")
        assert len(alts) > 0
        assert "ts_decay_linear" in alts

        empty_alts = registry.get_alternatives("nonexistent")
        assert empty_alts == []

    def test_validate_expression_operators_valid(self, registry):
        expr = "rank(ts_decay_linear(ts_zscore(close, 20), 10))"
        is_valid, errors = registry.validate_expression_operators(expr)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_expression_operators_invalid(self, registry):
        expr = "unknown_operator(close, 20)"
        is_valid, errors = registry.validate_expression_operators(expr)
        assert is_valid is False
        assert len(errors) > 0
        assert any("未知算子" in e for e in errors)

    def test_count_operators(self, registry):
        expr = "rank(ts_decay_linear(ts_zscore(ts_delta(close, 5), 20), 10))"
        count = registry.count_operators(expr)
        assert count == 4

        simple_expr = "close"
        count_simple = registry.count_operators(simple_expr)
        assert count_simple == 0

    def test_get_forbidden_patterns(self, registry):
        patterns = registry.get_forbidden_patterns()
        assert len(patterns) >= 3

        pattern_strings = [p[0] for p in patterns]
        assert any("ts_delta" in p and "close" in p for p in pattern_strings)
        assert any("ts_delta" in p and "volume" in p for p in pattern_strings)

    def test_record_usage(self, registry):
        registry.record_usage("ts_mean", 1.5)
        registry.record_usage("ts_mean", -0.5)
        registry.record_usage("ts_mean", 2.0)

        op = registry.get("ts_mean")
        assert op is not None
        assert op.use_count == 3
        assert op.success_count == 2
        assert op.avg_sharpe == pytest.approx(1.75, rel=1e-2)

    def test_record_usage_nonexistent(self, registry):
        should_not_raise = lambda: registry.record_usage("nonexistent", 1.0)
        should_not_raise()

    def test_get_stats(self, registry):
        registry.record_usage("ts_mean", 1.5)
        registry.record_usage("ts_rank", 2.0)
        registry.record_usage("ts_mean", -0.5)

        stats = registry.get_stats()
        assert "total_operators" in stats
        assert "most_used" in stats
        assert "highest_success_rate" in stats
        assert stats["total_operators"] == len(registry.get_all_operator_names())
        assert len(stats["most_used"]) <= 5

    def test_get_all_operator_names(self, registry):
        names = registry.get_all_operator_names()
        assert len(names) >= 66
        assert "ts_mean" in names
        assert "rank" in names
        assert "group_neutralize" in names
        assert "trade_when" in names

    def test_get_operators_requiring_lookback(self, registry):
        lookback_ops = registry.get_operators_requiring_lookback()
        assert len(lookback_ops) > 0
        for op in lookback_ops:
            assert op.requires_lookback is True
            assert op.default_lookback is not None

        assert "ts_mean" in {op.name for op in lookback_ops}
        assert "rank" not in {op.name for op in lookback_ops}

    def test_suggest_alternative_for_high_risk(self, registry):
        alt = registry.suggest_alternative_for_high_risk("ts_delta")
        assert alt is not None
        assert alt in ["ts_decay_linear", "ts_av_diff"]

        no_alt = registry.suggest_alternative_for_high_risk("ts_mean")
        assert no_alt is None

        nonexistent = registry.suggest_alternative_for_high_risk("nonexistent")
        assert nonexistent is None


class TestGlobalSingleton:
    def test_singleton_pattern(self):
        reg1 = get_operator_registry()
        reg2 = get_operator_registry()
        assert reg1 is reg2

    def test_singleton_has_all_operators(self):
        reg = get_operator_registry()
        assert reg.get("ts_mean") is not None
        assert reg.get("rank") is not None
        assert len(reg.get_all_operator_names()) >= 66


class TestKeyOperators:
    @pytest.fixture
    def registry(self):
        return OperatorRegistry()

    def test_ts_decay_linear_is_low_risk(self, registry):
        op = registry.get("ts_decay_linear")
        assert op is not None
        assert op.risk_level == "low"
        assert op.default_lookback == 10
        assert "过审因子核心算子" in op.description

    def test_ts_delta_is_high_risk(self, registry):
        op = registry.get("ts_delta")
        assert op is not None
        assert op.risk_level == "high"
        assert len(op.forbidden_combos) > 0

    def test_rank_is_cross_sectional(self, registry):
        op = registry.get("rank")
        assert op is not None
        assert op.category == OperatorCategory.CROSS_SECTIONAL
        assert op.risk_level == "low"
        assert op.requires_lookback is False

    def test_group_neutralize_is_group_operator(self, registry):
        op = registry.get("group_neutralize")
        assert op is not None
        assert op.category == OperatorCategory.GROUP
        assert op.risk_level == "low"

    def test_trade_when_is_conditional(self, registry):
        op = registry.get("trade_when")
        assert op is not None
        assert op.category == OperatorCategory.CONDITIONAL

    def test_math_operators_no_lookback(self, registry):
        for op_name in ["add", "subtract", "multiply", "divide", "log", "abs"]:
            op = registry.get(op_name)
            assert op is not None
            assert op.category == OperatorCategory.MATH
            assert op.requires_lookback is False


class TestCategoryCoverage:
    @pytest.fixture
    def registry(self):
        return OperatorRegistry()

    def test_all_categories_populated(self, registry):
        categories = list(OperatorCategory)
        for cat in categories:
            ops = registry.get_by_category(cat)
            if cat in [OperatorCategory.TEMPORAL, OperatorCategory.MATH,
                       OperatorCategory.CROSS_SECTIONAL, OperatorCategory.LOGICAL]:
                assert len(ops) > 0, f"{cat.value} 分类应该有算子"

    def test_temporal_category_count(self, registry):
        temporal = registry.get_by_category(OperatorCategory.TEMPORAL)
        assert len(temporal) >= 24

    def test_math_category_count(self, registry):
        math_ops = registry.get_by_category(OperatorCategory.MATH)
        assert len(math_ops) >= 15

    def test_logical_operators_present(self, registry):
        logical = registry.get_by_category(OperatorCategory.LOGICAL)
        logical_names = {op.name for op in logical}
        assert "and" in logical_names
        assert "or" in logical_names
        assert "if_else" in logical_names


class TestEdgeCases:
    @pytest.fixture
    def registry(self):
        return OperatorRegistry()

    def test_empty_expression_validation(self, registry):
        is_valid, errors = registry.validate_expression_operators("")
        assert is_valid is True
        assert len(errors) == 0

    def test_complex_nested_expression(self, registry):
        expr = ("group_neutralize("
                "trade_when("
                "rank(ts_decay_linear(ts_zscore(ts_corr(close, volume, 20), 20), 10)), "
                "greater(volume, ts_mean(volume, 20)), "
                "less(volume, ts_mean(volume, 5))"
                "), industry)")
        count = registry.count_operators(expr)
        assert count >= 8

    def test_case_insensitive_validation(self, registry):
        expr = "RANK(TS_MEAN(CLOSE, 20))"
        is_valid, _ = registry.validate_expression_operators(expr)
        assert is_valid is True
