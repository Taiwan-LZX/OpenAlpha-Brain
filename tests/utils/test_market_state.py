import json

import pytest
from openalpha_brain.utils.market_state import MarketState, MarketStateInferencer
from openalpha_brain.learning.mab import HierarchicalMAB


class TestInferFromBrainResults:
    def test_extracts_dominant_strategy(self, tmp_path):
        inferencer = MarketStateInferencer(path=str(tmp_path / "state.json"))
        inferencer._direction_sharpes = {}
        inferencer._yearly_states = {}
        results = [
            {"direction": "momentum", "sharpe": 1.5},
            {"direction": "momentum", "sharpe": 1.2},
            {"direction": "mean_reversion", "sharpe": 0.8},
        ]

        summary = inferencer.infer_from_brain_results(results)

        assert summary["current_dominant"] == "momentum"
        assert "momentum" in summary["avg_sharpes_by_direction"]
        assert abs(summary["avg_sharpes_by_direction"]["momentum"] - 1.35) < 1e-6

    def test_processes_yearly_breakdown(self, tmp_path):
        inferencer = MarketStateInferencer(path=str(tmp_path / "state.json"))
        inferencer._direction_sharpes = {}
        inferencer._yearly_states = {}
        results = [
            {
                "direction": "momentum",
                "sharpe": 1.0,
                "yearly_breakdown": [
                    {"year": 2023, "strategy": "momentum", "sharpe": 1.8},
                    {"year": 2023, "strategy": "mean_reversion", "sharpe": 0.5},
                ],
            },
        ]

        summary = inferencer.infer_from_brain_results(results)

        assert 2023 in summary["momentum_friendly_years"]
        state_2023 = inferencer.yearly_states.get(2023)
        assert state_2023 is not None
        assert state_2023.dominant_strategy == "momentum"

    def test_empty_results(self, tmp_path):
        inferencer = MarketStateInferencer(path=str(tmp_path / "state.json"))
        inferencer._direction_sharpes = {}
        inferencer._yearly_states = {}
        summary = inferencer.infer_from_brain_results([])

        assert summary["current_dominant"] == ""
        assert summary["avg_sharpes_by_direction"] == {}


class TestAdjustMabBias:
    def test_sets_bias_from_accumulated_data(self, tmp_path):
        inferencer = MarketStateInferencer(path=str(tmp_path / "state.json"))
        inferencer._direction_sharpes = {
            "momentum": [1.5, 1.2],
            "mean_reversion": [0.5, 0.3],
        }

        mab = HierarchicalMAB()
        inferencer.adjust_mab_bias(mab)

        momentum_arm = mab._outer._arms.get("momentum")
        reversion_arm = mab._outer._arms.get("mean_reversion")
        assert momentum_arm is not None
        assert momentum_arm.alpha > 1.0
        assert reversion_arm is not None
        assert momentum_arm.alpha > reversion_arm.alpha

    def test_noop_with_none_mab(self, tmp_path):
        inferencer = MarketStateInferencer(path=str(tmp_path / "state.json"))
        inferencer._direction_sharpes = {"momentum": [1.0]}
        inferencer.adjust_mab_bias(None)

    def test_noop_with_empty_sharpes(self, tmp_path):
        inferencer = MarketStateInferencer(path=str(tmp_path / "state.json"))
        mab = HierarchicalMAB()
        initial_arms = dict(mab._outer._arms)
        inferencer.adjust_mab_bias(mab)
        for direction, arm in mab._outer._arms.items():
            if direction in initial_arms:
                assert arm.alpha == initial_arms[direction].alpha


class TestGetMarketStateSummary:
    def test_returns_correct_format(self, tmp_path):
        inferencer = MarketStateInferencer(path=str(tmp_path / "state.json"))
        inferencer._direction_sharpes = {
            "momentum": [1.0, 2.0],
            "volatility": [0.5],
        }
        inferencer._yearly_states = {
            2023: MarketState(year=2023, momentum_sharpe=1.5, mean_reversion_sharpe=0.8, volatility_sharpe=0.3, dominant_strategy="momentum"),
            2024: MarketState(year=2024, momentum_sharpe=0.5, mean_reversion_sharpe=1.2, volatility_sharpe=0.4, dominant_strategy="mean_reversion"),
        }

        summary = inferencer.get_market_state_summary()

        assert "current_dominant" in summary
        assert "avg_sharpes_by_direction" in summary
        assert "momentum_friendly_years" in summary
        assert "mean_reversion_friendly_years" in summary


class TestMarketStatePersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "state.json")
        inferencer = MarketStateInferencer(path=path)
        inferencer._yearly_states = {
            2023: MarketState(year=2023, momentum_sharpe=1.5, mean_reversion_sharpe=0.8, volatility_sharpe=0.3, dominant_strategy="momentum"),
        }
        inferencer._direction_sharpes = {"momentum": [1.5, 1.2], "mean_reversion": [0.8]}
        inferencer._save()

        inferencer2 = MarketStateInferencer(path=path)
        assert 2023 in inferencer2._yearly_states
        loaded = inferencer2._yearly_states[2023]
        assert loaded.momentum_sharpe == 1.5
        assert loaded.dominant_strategy == "momentum"
        assert inferencer2._direction_sharpes["momentum"] == [1.5, 1.2]


class TestMarketStateDataclass:
    def test_to_dict(self):
        ms = MarketState(year=2023, momentum_sharpe=1.5, mean_reversion_sharpe=0.8, volatility_sharpe=0.3, dominant_strategy="momentum")
        d = ms.to_dict()

        assert d["year"] == 2023
        assert d["momentum_sharpe"] == 1.5
        assert d["dominant_strategy"] == "momentum"

    def test_from_dict(self):
        d = {"year": 2024, "momentum_sharpe": 2.0, "mean_reversion_sharpe": 1.0, "volatility_sharpe": 0.5, "dominant_strategy": "momentum"}
        ms = MarketState.from_dict(d)

        assert ms.year == 2024
        assert ms.momentum_sharpe == 2.0
        assert ms.dominant_strategy == "momentum"

    def test_from_dict_with_missing_fields(self):
        ms = MarketState.from_dict({"year": 2023})

        assert ms.year == 2023
        assert ms.momentum_sharpe == 0.0
        assert ms.dominant_strategy == ""

    def test_roundtrip_to_dict_from_dict(self):
        original = MarketState(year=2023, momentum_sharpe=1.5, mean_reversion_sharpe=0.8, volatility_sharpe=0.3, dominant_strategy="momentum")
        restored = MarketState.from_dict(original.to_dict())

        assert restored.year == original.year
        assert restored.momentum_sharpe == original.momentum_sharpe
        assert restored.dominant_strategy == original.dominant_strategy
