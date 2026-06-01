import pytest
from openalpha_brain.agents.multi_agent import (
    Hypothesis,
    AgentResult,
    SemanticAnchor,
    _check_semantic_alignment,
    _collect_operators_from_ast,
    _collect_fields_from_ast,
    _extract_max_lookback,
    _DIRECTION_OPERATOR_MAP,
    _DIRECTION_FIELD_MAP,
    _load_direction_operator_map,
    _load_direction_field_map,
    _MECHANISM_OPERATOR_MAP,
)


class TestCollectOperatorsFromAst:
    def test_simple_expression(self):
        ops = _collect_operators_from_ast("ts_delta(close, 5)")
        assert "ts_delta" in ops

    def test_nested_expression(self):
        ops = _collect_operators_from_ast("group_neutralize(rank(ts_zscore(close, 20)), INDUSTRY)")
        assert "group_neutralize" in ops
        assert "rank" in ops
        assert "ts_zscore" in ops

    def test_no_operators(self):
        ops = _collect_operators_from_ast("close")
        assert len(ops) == 0


class TestCollectFieldsFromAst:
    def test_simple_field(self):
        fields = _collect_fields_from_ast("ts_delta(close, 5)")
        assert "close" in fields

    def test_multiple_fields(self):
        fields = _collect_fields_from_ast("rank(close / vwap)")
        assert "close" in fields
        assert "vwap" in fields

    def test_no_fields(self):
        fields = _collect_fields_from_ast("rank(1)")
        assert len(fields) == 0


class TestExtractMaxLookback:
    def test_single_lookback(self):
        assert _extract_max_lookback("ts_delta(close, 5)") == 5

    def test_multiple_lookbacks(self):
        assert _extract_max_lookback("ts_mean(close, 20) + ts_zscore(volume, 60)") == 60

    def test_no_lookback(self):
        assert _extract_max_lookback("rank(close)") == 0


class TestCheckSemanticAlignment:
    def test_momentum_with_ts_delta(self):
        h = Hypothesis(direction="momentum", asset_class="equity", time_horizon="short", mechanism="test", natural_language="test")
        score = _check_semantic_alignment(h, "ts_delta(close, 5)")
        assert score > 0.0

    def test_momentum_with_rank_only(self):
        h = Hypothesis(direction="momentum", asset_class="equity", time_horizon="short", mechanism="test", natural_language="test")
        score = _check_semantic_alignment(h, "rank(cap)")
        assert score < 0.7

    def test_unknown_direction(self):
        h = Hypothesis(direction="unknown_direction", asset_class="equity", time_horizon="medium-term", mechanism="test", natural_language="test")
        score = _check_semantic_alignment(h, "rank(close)")
        assert score < 1.0
        assert score >= 0.0

    def test_short_time_horizon_long_lookback(self):
        h = Hypothesis(direction="momentum", asset_class="equity", time_horizon="short", mechanism="test", natural_language="test")
        score = _check_semantic_alignment(h, "ts_delta(close, 200)")
        assert score < 1.0

    def test_mechanism_momentum_penalty(self):
        h = Hypothesis(direction="momentum", asset_class="equity", time_horizon="medium-term", mechanism="momentum signal", natural_language="test")
        score_with = _check_semantic_alignment(h, "ts_delta(close, 10)")
        h2 = Hypothesis(direction="momentum", asset_class="equity", time_horizon="medium-term", mechanism="momentum signal", natural_language="test")
        score_without = _check_semantic_alignment(h2, "rank(close)")
        assert score_with >= score_without

    def test_mechanism_volatility_penalty(self):
        h = Hypothesis(direction="volatility", asset_class="equity", time_horizon="medium-term", mechanism="volatility breakout", natural_language="test")
        score = _check_semantic_alignment(h, "ts_std_dev(close, 20)")
        assert score > 0.0

    def test_mechanism_mean_reversion_no_penalty(self):
        h = Hypothesis(direction="mean_reversion", asset_class="equity", time_horizon="medium-term", mechanism="mean_reversion", natural_language="test")
        score = _check_semantic_alignment(h, "ts_zscore(close, 20)")
        assert score > 0.0

    def test_exception_returns_default(self):
        score = _check_semantic_alignment(None, "rank(close)")
        assert score >= 0.0


class TestDirectionMaps:
    def test_operator_map_reload(self):
        m = _load_direction_operator_map()
        assert isinstance(m, dict)

    def test_field_map_reload(self):
        m = _load_direction_field_map()
        assert isinstance(m, dict)


class TestSemanticAnchor:
    def test_create_anchor(self):
        anchor = SemanticAnchor(
            hypothesis_text="momentum signal test",
            direction="momentum",
            core_operators=["ts_delta", "ts_zscore"],
        )
        assert anchor.direction == "momentum"
        assert anchor.drift_threshold == 0.7
        assert anchor._embedding is None


class TestAgentResult:
    def test_result_has_new_fields(self):
        r = AgentResult(
            hypothesis=Hypothesis(direction="momentum", asset_class="equity", time_horizon="short", mechanism="test", natural_language="test"),
            expression="ts_delta(close, 5)",
            simulation_payload={},
            originality_score=0.8,
            complexity_metrics={},
            brain_sharpe=None,
            iterations=1,
            converged=False,
            semantic_alignment_score=0.7,
            variants=[],
        )
        assert r.semantic_alignment_score == 0.7
        assert r.variants == []
