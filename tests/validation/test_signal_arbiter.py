import asyncio

import pytest

from openalpha_brain.learning.mab import AssociationMatrix, HierarchicalMAB
from openalpha_brain.utils.market_state import MarketStateInferencer
from openalpha_brain.utils.whitelist import WhitelistManager
from openalpha_brain.validation.signal_arbiter import (
    ADJUSTMENT_INTERVAL,
    DEFAULT_WEIGHTS,
    WEIGHT_BOUNDS,
    AssociationSignalAdapter,
    MABSignalAdapter,
    MarketSignalAdapter,
    RAGSignalAdapter,
    SignalArbiter,
    SignalSource,
    WhitelistSignalAdapter,
)


class TestSignalArbiter:
    def test_default_weights_sum_to_one(self):
        assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)

    def test_arbitrate_single_signal(self):
        arbiter = SignalArbiter()
        signals = [SignalSource(source_name="rag", score=0.8, weight=1.0)]
        result = arbiter.arbitrate("item1", signals)
        assert result.final_score == pytest.approx(0.8)
        assert result.confidence == pytest.approx(0.2)

    def test_arbitrate_multiple_signals(self):
        arbiter = SignalArbiter()
        signals = [
            SignalSource(source_name="rag", score=0.8, weight=1.0),
            SignalSource(source_name="mab", score=0.6, weight=1.0),
        ]
        result = arbiter.arbitrate("item1", signals)
        expected = (0.2 * 1.0 * 0.8 + 0.3 * 1.0 * 0.6) / (0.2 + 0.3)
        assert result.final_score == pytest.approx(expected)
        assert "rag" in result.signal_breakdown
        assert "mab" in result.signal_breakdown

    def test_arbitrate_signal_with_non_unit_weight(self):
        arbiter = SignalArbiter()
        signals = [
            SignalSource(source_name="rag", score=0.8, weight=0.5),
            SignalSource(source_name="mab", score=0.6, weight=2.0),
        ]
        result = arbiter.arbitrate("item1", signals)
        expected = (0.2 * 0.5 * 0.8 + 0.3 * 2.0 * 0.6) / (0.2 + 0.3)
        assert result.final_score == pytest.approx(expected)
        assert result.signal_breakdown["rag"] == pytest.approx(0.2 * 0.5 * 0.8)
        assert result.signal_breakdown["mab"] == pytest.approx(0.3 * 2.0 * 0.6)

    def test_arbitrate_empty_signals(self):
        arbiter = SignalArbiter()
        result = arbiter.arbitrate("item1", [])
        assert result.final_score == 0.0
        assert result.confidence == 0.0
        assert result.signal_breakdown == {}

    def test_rank_fields_ordering(self):
        arbiter = SignalArbiter()
        field_signals = {
            "field_a": [SignalSource(source_name="rag", score=0.9, weight=1.0)],
            "field_b": [SignalSource(source_name="rag", score=0.3, weight=1.0)],
            "field_c": [SignalSource(source_name="rag", score=0.6, weight=1.0)],
        }
        results = arbiter.rank_fields(field_signals)
        assert [r.item_id for r in results] == ["field_a", "field_c", "field_b"]

    def test_rank_operators_ordering(self):
        arbiter = SignalArbiter()
        op_signals = {
            "op_x": [SignalSource(source_name="mab", score=0.5, weight=1.0)],
            "op_y": [SignalSource(source_name="mab", score=0.9, weight=1.0)],
        }
        results = arbiter.rank_operators(op_signals)
        assert [r.item_id for r in results] == ["op_y", "op_x"]

    def test_adjust_weights_low_success_rate(self):
        arbiter = SignalArbiter()
        initial_mab = arbiter.weights["mab"]
        for _ in range(ADJUSTMENT_INTERVAL):
            arbiter.adjust_weights(0.2)
        assert arbiter.weights["mab"] > initial_mab

    def test_adjust_weights_high_success_rate(self):
        arbiter = SignalArbiter()
        initial_market = arbiter.weights["market"]
        for _ in range(ADJUSTMENT_INTERVAL):
            arbiter.adjust_weights(0.8)
        assert arbiter.weights["market"] > initial_market

    def test_adjust_weights_normalization(self):
        arbiter = SignalArbiter()
        for _ in range(ADJUSTMENT_INTERVAL):
            arbiter.adjust_weights(0.2)
        assert sum(arbiter.weights.values()) == pytest.approx(1.0)

    def test_weight_bounds(self):
        arbiter = SignalArbiter()
        arbiter.set_weight("rag", 0.0)
        assert arbiter.weights["rag"] == WEIGHT_BOUNDS[0]
        arbiter.set_weight("rag", 1.0)
        assert arbiter.weights["rag"] == WEIGHT_BOUNDS[1]

    def test_to_dict_from_dict_roundtrip(self):
        arbiter = SignalArbiter()
        arbiter.set_weight("rag", 0.4)
        for _ in range(3):
            arbiter.adjust_weights(0.5)
        data = arbiter.to_dict()
        restored = SignalArbiter.from_dict(data)
        assert restored.weights == arbiter.weights
        assert restored._adjustment_counter == arbiter._adjustment_counter


class TestRAGSignalAdapter:
    def test_rag_adapter_fields(self):
        rag_result = {
            "fields": [
                {"id": "close", "score": 0.9},
                {"id": "volume", "score": 0.45},
            ],
            "operators": [],
        }
        adapter = RAGSignalAdapter(rag_result)
        result = adapter.adapt_fields()
        assert "close" in result
        assert "volume" in result
        assert len(result["close"]) == 1
        assert result["close"][0].source_name == "rag"
        assert result["close"][0].score == pytest.approx(1.0)
        assert result["volume"][0].score == pytest.approx(0.5)

    def test_rag_adapter_operators(self):
        rag_result = {
            "fields": [],
            "operators": [
                {"id": "ts_rank", "score": 0.8},
                {"id": "ts_decay_linear", "score": 0.4},
            ],
        }
        adapter = RAGSignalAdapter(rag_result)
        result = adapter.adapt_operators()
        assert "ts_rank" in result
        assert "ts_decay_linear" in result
        assert result["ts_rank"][0].score == pytest.approx(1.0)
        assert result["ts_decay_linear"][0].score == pytest.approx(0.5)

    def test_rag_adapter_empty(self):
        adapter = RAGSignalAdapter({})
        assert adapter.adapt_fields() == {}
        assert adapter.adapt_operators() == {}

    def test_rag_adapter_zero_max_score(self):
        rag_result = {"fields": [{"id": "close", "score": 0.0}], "operators": []}
        adapter = RAGSignalAdapter(rag_result)
        result = adapter.adapt_fields()
        assert result["close"][0].score == 0.0


class TestMABSignalAdapter:
    def test_mab_adapter_with_none(self):
        adapter = MABSignalAdapter(None)
        assert adapter.adapt_fields() == {}
        assert adapter.adapt_operators() == {}

    def test_mab_adapter_with_mab(self):
        mab = HierarchicalMAB()
        mab.update("momentum", ["ts_rank", "ts_decay_linear"], ["close", "volume"], reward=1.0)
        adapter = MABSignalAdapter(mab)
        fields = adapter.adapt_fields()
        ops = adapter.adapt_operators()
        assert "close" in fields
        assert "volume" in fields
        assert "ts_rank" in ops
        assert "ts_decay_linear" in ops
        assert fields["close"][0].source_name == "mab"
        assert ops["ts_rank"][0].source_name == "mab"


class TestAssociationSignalAdapter:
    def test_association_adapter_with_none(self):
        adapter = AssociationSignalAdapter(None, "ts_rank", "close")
        assert adapter.adapt_fields() == {}
        assert adapter.adapt_operators() == {}

    def test_association_adapter_with_matrix(self):
        assoc = AssociationMatrix()
        assoc.update("ts_rank", "close", reward=1.0)
        assoc.update("ts_rank", "volume", reward=0.5)
        assoc.update("ts_decay_linear", "close", reward=0.3)
        adapter = AssociationSignalAdapter(assoc, "ts_rank", "close")
        fields = adapter.adapt_fields()
        ops = adapter.adapt_operators()
        assert len(fields) > 0
        assert len(ops) > 0
        for fid, signals in fields.items():
            assert signals[0].source_name == "association"
        for oid, signals in ops.items():
            assert signals[0].source_name == "association"


class TestWhitelistSignalAdapter:
    def test_whitelist_adapter_with_none(self):
        adapter = WhitelistSignalAdapter(None)
        assert adapter.adapt_fields() == {}
        assert adapter.adapt_operators() == {}

    def test_whitelist_adapter_with_manager(self):
        wm = WhitelistManager(core_fields=set())
        wm.solidify_field("close")
        wm.solidify_field("volume")
        adapter = WhitelistSignalAdapter(wm)
        fields = adapter.adapt_fields()
        assert "close" in fields
        assert "volume" in fields
        assert fields["close"][0].source_name == "whitelist"
        assert adapter.adapt_operators() == {}


class TestMarketSignalAdapter:
    def test_market_adapter_with_none(self):
        adapter = MarketSignalAdapter(None, "momentum")
        assert adapter.adapt_fields() == {}
        assert adapter.adapt_operators() == {}

    def test_market_adapter_with_state(self, tmp_path):
        ms = MarketStateInferencer(path=str(tmp_path / "state.json"))
        adapter = MarketSignalAdapter(ms, "momentum", field_ids=["close", "volume"], op_ids=["ts_delta", "rank"])
        fields = adapter.adapt_fields()
        ops = adapter.adapt_operators()
        assert "close" in fields
        assert "volume" in fields
        assert fields["close"][0].source_name == "market"
        assert fields["close"][0].score > 0.5
        assert "ts_delta" in ops
        assert "rank" in ops
        assert ops["ts_delta"][0].source_name == "market"

    def test_market_adapter_no_sharpes(self, tmp_path):
        ms = MarketStateInferencer(path=str(tmp_path / "state.json"))
        ms._direction_sharpes = {}
        ms._yearly_states = {}
        adapter = MarketSignalAdapter(ms, "momentum")
        assert adapter.adapt_fields() == {}
        assert adapter.adapt_operators() == {}

    def test_market_adapter_neutral_score_not_dropped(self, tmp_path):
        ms = MarketStateInferencer(path=str(tmp_path / "state.json"))
        ms._direction_sharpes = {}
        ms._yearly_states = {}
        adapter = MarketSignalAdapter(ms, "momentum", field_ids=["close"], op_ids=["ts_delta"])
        fields = adapter.adapt_fields()
        ops = adapter.adapt_operators()
        assert "close" in fields
        assert "ts_delta" in ops
        assert fields["close"][0].score == pytest.approx(0.5)
        assert ops["ts_delta"][0].score == pytest.approx(0.5)


class TestRankWithAdapters:
    def test_rank_with_adapters(self):
        arbiter = SignalArbiter()
        rag_result = {
            "fields": [
                {"id": "close", "score": 0.9},
                {"id": "volume", "score": 0.3},
            ],
            "operators": [
                {"id": "ts_rank", "score": 0.8},
                {"id": "ts_decay_linear", "score": 0.4},
            ],
        }
        rag_adapter = RAGSignalAdapter(rag_result)
        field_results, op_results = asyncio.run(
            arbiter.rank_with_adapters(
                field_adapters=[rag_adapter],
                op_adapters=[rag_adapter],
            )
        )
        assert len(field_results) == 2
        assert field_results[0].item_id == "close"
        assert len(op_results) == 2
        assert op_results[0].item_id == "ts_rank"

    def test_rank_with_adapters_top_k(self):
        arbiter = SignalArbiter()
        rag_result = {
            "fields": [
                {"id": "close", "score": 0.9},
                {"id": "volume", "score": 0.3},
                {"id": "vwap", "score": 0.6},
            ],
            "operators": [
                {"id": "ts_rank", "score": 0.8},
            ],
        }
        rag_adapter = RAGSignalAdapter(rag_result)
        field_results, op_results = asyncio.run(
            arbiter.rank_with_adapters(
                field_adapters=[rag_adapter],
                op_adapters=[rag_adapter],
                top_k_fields=2,
                top_k_ops=1,
            )
        )
        assert len(field_results) == 2
        assert len(op_results) == 1

    def test_rank_with_adapters_multiple_adapters(self):
        arbiter = SignalArbiter()
        rag_result = {
            "fields": [{"id": "close", "score": 0.9}],
            "operators": [{"id": "ts_rank", "score": 0.8}],
        }
        rag_adapter = RAGSignalAdapter(rag_result)
        mab = HierarchicalMAB()
        mab.update("momentum", ["ts_rank"], ["close"], reward=1.0)
        mab_adapter = MABSignalAdapter(mab)
        field_results, op_results = asyncio.run(
            arbiter.rank_with_adapters(
                field_adapters=[rag_adapter, mab_adapter],
                op_adapters=[rag_adapter, mab_adapter],
            )
        )
        close_result = next(r for r in field_results if r.item_id == "close")
        assert "rag" in close_result.signal_breakdown
        assert "mab" in close_result.signal_breakdown
