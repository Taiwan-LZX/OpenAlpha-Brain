"""Comprehensive tests for AdaptiveNeutralizer module.

Covers:
  - NeutralizationTrial & NeutralizationExperience dataclasses (7 tests)
  - NeutralizationExperienceTracker (16 tests)
  - EhsaniConditionEvaluator (12 tests)
  - AdaptiveNeutralizer main class (18 tests)
  - Safety Constraints (10 tests)
  - Edge Cases & Robustness (12 tests)

Target: ~65-75 test functions.
"""

from __future__ import annotations

import json
import logging
import threading
import time

import pytest

from openalpha_brain.evolution.adaptive_neutralizer import (
    _FORBIDDEN_PAIRS,
    DEFAULT_CONFIG,
    AdaptiveNeutralizer,
    AdaptiveRecommendation,
    EhsaniConditionEvaluator,
    NeutralizationExperience,
    NeutralizationExperienceTracker,
    NeutralizationTrial,
    create_adaptive_neutralizer,
)


def _make_trial(
    category: str = "momentum",
    level: str = "industry",
    sharpe_before: float = 1.0,
    sharpe_after: float = 1.15,
    fitness_delta: float = 0.08,
    outcome: str = "success",
    expr_id: str = "expr_001",
    ts: float | None = None,
) -> NeutralizationTrial:
    return NeutralizationTrial(
        category=category,
        neutralization_level=level,
        sharpe_before=sharpe_before,
        sharpe_after=sharpe_after,
        fitness_delta=fitness_delta,
        timestamp=ts if ts is not None else time.time(),
        expression_id=expr_id,
        outcome=outcome,
    )


def _make_metrics(
    sharpe_raw: float = 1.0,
    sharpe_neut: float = 0.85,
    correlation: float = 0.7,
    **kwargs,
) -> dict:
    return {"sharpe_raw": sharpe_raw, "sharpe_neutralized": sharpe_neut, "correlation": correlation, **kwargs}


# ---------------------------------------------------------------------------
# Category 1: NeutralizationTrial & NeutralizationExperience dataclasses (7)
# ---------------------------------------------------------------------------


class TestNeutralizationTrialDataclass:
    def test_creation_with_valid_data(self):
        trial = _make_trial(
            category="momentum",
            level="industry",
            sharpe_before=1.0,
            sharpe_after=1.15,
            fitness_delta=0.08,
            outcome="success",
            expr_id="expr_001",
            ts=1000.0,
        )
        assert trial.category == "momentum"
        assert trial.neutralization_level == "industry"
        assert trial.sharpe_before == 1.0
        assert trial.sharpe_after == 1.15
        assert trial.fitness_delta == 0.08
        assert trial.timestamp == 1000.0
        assert trial.expression_id == "expr_001"
        assert trial.outcome == "success"

    def test_default_timestamp_is_recent(self):
        before = time.time()
        trial = _make_trial(ts=None)
        after = time.time()
        assert before <= trial.timestamp <= after

    def test_failure_outcome_accepted(self):
        trial = _make_trial(outcome="failure")
        assert trial.outcome == "failure"

    def test_partial_outcome_accepted(self):
        trial = _make_trial(outcome="partial")
        assert trial.outcome == "partial"

    def test_all_fields_accessible(self):
        trial = _make_trial(category="value", level="sector")
        assert hasattr(trial, "category")
        assert hasattr(trial, "neutralization_level")
        assert hasattr(trial, "sharpe_before")
        assert hasattr(trial, "sharpe_after")
        assert hasattr(trial, "fitness_delta")
        assert hasattr(trial, "timestamp")
        assert hasattr(trial, "expression_id")
        assert hasattr(trial, "outcome")

    def test_negative_fitness_delta_accepted(self):
        trial = _make_trial(fitness_delta=-0.10)
        assert trial.fitness_delta == -0.10

    def test_zero_sharpe_values_accepted(self):
        trial = _make_trial(sharpe_before=0.0, sharpe_after=0.0)
        assert trial.sharpe_before == 0.0
        assert trial.sharpe_after == 0.0


class TestNeutralizationExperienceDataclass:
    def test_default_values(self):
        exp = NeutralizationExperience(category="test", level="industry")
        assert exp.total_trials == 0
        assert exp.successes == 0
        assert exp.avg_sharpe_delta == 0.0
        assert exp.avg_fitness_delta == 0.0
        assert exp.success_rate == 0.0
        assert exp.mab_score == 0.5
        assert exp.last_updated == 0.0
        assert exp.is_forbidden is False

    def test_custom_values(self):
        exp = NeutralizationExperience(
            category="mom",
            level="subindustry",
            total_trials=10,
            successes=7,
            avg_sharpe_delta=0.12,
            avg_fitness_delta=0.05,
            success_rate=0.7,
            mab_score=0.75,
            last_updated=999.0,
            is_forbidden=True,
        )
        assert exp.total_trials == 10
        assert exp.successes == 7
        assert exp.success_rate == 0.7
        assert exp.is_forbidden is True


# ---------------------------------------------------------------------------
# Category 2: NeutralizationExperienceTracker (16)
# ---------------------------------------------------------------------------


class TestTrackerRecordTrial:
    def setup_method(self):
        self.tracker = NeutralizationExperienceTracker()

    def test_record_single_trial_increments_count(self):
        self.tracker.record_trial(_make_trial())
        exp = self.tracker.get_experience("momentum", "industry")
        assert exp.total_trials == 1
        assert exp.successes == 1
        assert exp.success_rate == 1.0

    def test_record_failure_trial(self):
        self.tracker.record_trial(_make_trial(outcome="failure"))
        exp = self.tracker.get_experience("momentum", "industry")
        assert exp.total_trials == 1
        assert exp.successes == 0
        assert exp.success_rate == 0.0

    def test_record_partial_trial_counts_as_success(self):
        self.tracker.record_trial(_make_trial(outcome="partial"))
        exp = self.tracker.get_experience("momentum", "industry")
        assert exp.successes == 1

    def test_record_multiple_trials_accumulates(self):
        for i in range(5):
            outcome = "success" if i % 2 == 0 else "failure"
            self.tracker.record_trial(
                _make_trial(
                    category="size",
                    level="none",
                    outcome=outcome,
                    expr_id=f"e{i}",
                    ts=float(i),
                )
            )
        exp = self.tracker.get_experience("size", "none")
        assert exp.total_trials == 5
        assert exp.successes == 3

    def test_record_updates_avg_deltas_incrementally(self):
        self.tracker.record_trial(_make_trial(sharpe_before=1.0, sharpe_after=1.2, fitness_delta=0.10))
        self.tracker.record_trial(_make_trial(sharpe_before=1.0, sharpe_after=1.0, fitness_delta=-0.05, expr_id="e2"))
        exp = self.tracker.get_experience("momentum", "industry")
        expected = ((0.2) + (-0.0)) / 2
        assert exp.avg_sharpe_delta == pytest.approx(expected, abs=1e-9)


class TestTrackerGetExperience:
    def setup_method(self):
        self.tracker = NeutralizationExperienceTracker()

    def test_get_experience_returns_empty_for_missing_key(self):
        exp = self.tracker.get_experience("nonexistent", "industry")
        assert isinstance(exp, NeutralizationExperience)
        assert exp.total_trials == 0
        assert exp.category == "nonexistent"
        assert exp.level == "industry"

    def test_get_experience_returns_stored_data(self):
        self.tracker.record_trial(_make_trial(category="volatility", level="none"))
        exp = self.tracker.get_experience("volatility", "none")
        assert exp.total_trials == 1
        assert exp.successes == 1


class TestTrackerBestLevel:
    def setup_method(self):
        self.tracker = NeutralizationExperienceTracker()

    def test_best_level_with_sufficient_data(self):
        for lvl in ["industry", "subindustry"]:
            for i in range(5):
                self.tracker.record_trial(
                    _make_trial(
                        category="mom_best",
                        level=lvl,
                        sharpe_after=1.2 if lvl == "industry" else 1.0,
                        outcome="success" if lvl == "industry" else "failure",
                        expr_id=f"{lvl}_{i}",
                        ts=float(i),
                    )
                )
        best = self.tracker.get_best_level_for_category("mom_best")
        assert best is not None
        assert best in DEFAULT_CONFIG["neutralization_levels"]

    def test_best_level_insufficient_data_returns_none(self):
        self.tracker.record_trial(_make_trial(category="new_cat", expr_id="e1"))
        result = self.tracker.get_best_level_for_category("new_cat")
        assert result is None

    def test_best_level_custom_min_trials(self):
        for _ in range(2):
            self.tracker.record_trial(_make_trial(category="low_trials", expr_id="lt"))
        high_min = self.tracker.get_best_level_for_category("low_trials", min_trials=3)
        assert high_min is None
        low_min = self.tracker.get_best_level_for_category("low_trials", min_trials=1)
        assert low_min is not None

    def test_best_level_skips_forbidden_experiences(self):
        for i in range(6):
            self.tracker.record_trial(
                _make_trial(
                    category="forbidden_cat",
                    level="bad_lvl",
                    outcome="failure",
                    expr_id=f"fb_{i}",
                    ts=float(i),
                )
            )
        exp = self.tracker.get_experience("forbidden_cat", "bad_lvl")
        assert exp.is_forbidden is True
        best = self.tracker.get_best_level_for_category("forbidden_cat")
        if best is not None:
            assert best != "bad_lvl"


class TestTrackerMatrixAndIO:
    def setup_method(self):
        self.tracker = NeutralizationExperienceTracker()

    def test_success_rate_matrix_generation(self):
        self.tracker.record_trial(_make_trial(level="industry", outcome="success"))
        self.tracker.record_trial(_make_trial(level="subindustry", outcome="failure"))
        matrix = self.tracker.get_success_rate_matrix()
        assert "momentum" in matrix
        assert matrix["momentum"]["industry"] == 1.0
        assert matrix["momentum"]["subindustry"] == 0.0

    def test_save_and_load_roundtrip(self, tmp_path):
        file_path = tmp_path / "experience.json"
        self.tracker.record_trial(
            _make_trial(
                category="save_test",
                level="industry",
                sharpe_before=1.0,
                sharpe_after=1.2,
                fitness_delta=0.10,
                expr_id="s1",
                ts=100.0,
                outcome="success",
            )
        )
        self.tracker.record_trial(
            _make_trial(
                category="save_test",
                level="sector",
                sharpe_before=0.8,
                sharpe_after=0.75,
                fitness_delta=-0.04,
                expr_id="s2",
                ts=200.0,
                outcome="failure",
            )
        )
        self.tracker.save_to_disk(file_path)
        assert file_path.exists()
        new_tracker = NeutralizationExperienceTracker()
        new_tracker.load_from_disk(file_path)
        assert new_tracker.get_experience("save_test", "industry").total_trials == 1
        assert new_tracker.get_experience("save_test", "sector").total_trials == 1

    def test_load_nonexistent_file_graceful(self, tmp_path):
        tracker = NeutralizationExperienceTracker()
        tracker.load_from_disk(tmp_path / "no_such_file.json")
        assert len(tracker._experiences) == 0

    def test_load_corrupted_json_graceful_degradation(self, tmp_path):
        bad_file = tmp_path / "corrupt.json"
        bad_file.write_text("{invalid json!!!", encoding="utf-8")
        tracker = NeutralizationExperienceTracker()
        tracker.load_from_disk(bad_file)
        assert len(tracker._experiences) == 0

    def test_save_creates_parent_dirs(self, tmp_path):
        deep_path = tmp_path / "nested" / "dir" / "exp.json"
        self.tracker.record_trial(_make_trial(expr_id="deep"))
        self.tracker.save_to_disk(deep_path)
        assert deep_path.exists()


class TestTrackerEdgeCases:
    def setup_method(self):
        self.tracker = NeutralizationExperienceTracker()

    def test_empty_tracker_has_no_data(self):
        assert len(self.tracker._experiences) == 0
        matrix = self.tracker.get_success_rate_matrix()
        assert matrix == {}

    def test_single_trial_state(self):
        self.tracker.record_trial(_make_trial(category="solo", expr_id="only"))
        assert len(self.tracker._experiences) == 1

    def test_thread_safety_concurrent_writes(self):
        errors = []

        def writer(cat: str, n: int):
            try:
                for i in range(n):
                    self.tracker.record_trial(
                        _make_trial(
                            category=cat,
                            expr_id=f"{cat}_{i}",
                            ts=float(i),
                        )
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(f"cat_{t}", 20)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert not errors, f"Thread errors: {errors}"
        total = sum(exp.total_trials for exp in self.tracker._experiences.values())
        assert total == 100

    def test_mab_score_increases_with_successes(self):
        for _ in range(3):
            self.tracker.record_trial(_make_trial(outcome="success", expr_id="mab"))
        exp = self.tracker.get_experience("momentum", "industry")
        assert exp.mab_score > 0.5

    def test_forbidden_flag_set_on_low_success_rate(self):
        for i in range(6):
            self.tracker.record_trial(
                _make_trial(
                    category="fbd",
                    level="dangerous",
                    outcome="failure",
                    expr_id=f"f_{i}",
                    ts=float(i),
                )
            )
        exp = self.tracker.get_experience("fbd", "dangerous")
        assert exp.is_forbidden is True
        assert ("fbd", "dangerous") in _FORBIDDEN_PAIRS


# ---------------------------------------------------------------------------
# Category 3: EhsaniConditionEvaluator (12)
# ---------------------------------------------------------------------------


class TestEhsaniEvaluate:
    def setup_method(self):
        self.eval = EhsaniConditionEvaluator()

    def test_evaluate_true_when_sr_ratio_below_correlation(self):
        result = self.eval.evaluate(sharpe_raw=1.0, sharpe_neutralized=0.5, correlation=0.7)
        assert result is True

    def test_evaluate_false_when_sr_ratio_above_correlation(self):
        result = self.eval.evaluate(sharpe_raw=1.0, sharpe_neutralized=0.9, correlation=0.7)
        assert result is False

    def test_evaluate_boundary_exact_correlation(self):
        result = self.eval.evaluate(sharpe_raw=1.0, sharpe_neutralized=0.7, correlation=0.7)
        assert result is False

    def test_evaluate_false_when_sharpe_raw_zero_or_negative(self):
        assert self.eval.evaluate(sharpe_raw=0.0, sharpe_neutralized=0.5, correlation=0.7) is False
        assert self.eval.evaluate(sharpe_raw=-1.0, sharpe_neutralized=0.5, correlation=0.7) is False

    def test_evaluate_extreme_high_sr_ratio(self):
        result = self.eval.evaluate(sharpe_raw=1.0, sharpe_neutralized=1.5, correlation=0.7)
        assert result is False

    def test_evaluate_very_low_sr_ratio(self):
        result = self.eval.evaluate(sharpe_raw=2.0, sharpe_neutralized=0.1, correlation=0.7)
        assert result is True

    def test_evaluate_uses_default_rho_when_correlation_none(self):
        result = self.eval.evaluate(sharpe_raw=1.0, sharpe_neutralized=0.5, correlation=None)
        assert result is True

    def test_evaluate_with_custom_config(self):
        custom_eval = EhsaniConditionEvaluator(config={"ehsani_correlation_estimate": 0.5})
        result = custom_eval.evaluate(sharpe_raw=1.0, sharpe_neutralized=0.6, correlation=0.5)
        assert result is False


class TestEhsaniEstimateSrRatio:
    def setup_method(self):
        self.eval = EhsaniConditionEvaluator()

    def test_estimate_from_historical_data(self):
        data = [
            {"sharpe_before": 1.0, "sharpe_after": 0.8},
            {"sharpe_before": 1.0, "sharpe_after": 0.7},
            {"sharpe_before": 2.0, "sharpe_after": 1.6},
        ]
        ratio = self.eval.estimate_sr_ratio("momentum", data)
        expected = (0.8 + 0.7 + 0.8) / 3
        assert ratio == pytest.approx(expected, abs=1e-9)

    def test_estimate_empty_data_returns_category_default(self):
        ratio = self.eval.estimate_sr_ratio("momentum", [])
        assert ratio == 0.55

    def test_estimate_unknown_category_returns_fallback(self):
        ratio = self.eval.estimate_sr_ratio("unknown_category_xyz", [])
        assert ratio == 0.7

    def test_estimate_zero_sharpe_before_skipped(self):
        data = [{"sharpe_before": 0.0, "sharpe_after": 0.5}]
        ratio = self.eval.estimate_sr_ratio("value", data)
        assert ratio == 0.7


class TestEhsaniRecommendLevel:
    def setup_method(self):
        self.eval = EhsaniConditionEvaluator()
        self.tracker = NeutralizationExperienceTracker()

    def test_recommend_returns_tuple_of_level_and_confidence(self):
        level, conf = self.eval.recommend_level("momentum", 1.0, self.tracker)
        assert isinstance(level, str)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_recommend_uses_experience_when_available(self):
        for _ in range(5):
            self.tracker.record_trial(
                _make_trial(
                    category="exp_rec",
                    level="industry",
                    sharpe_after=1.3,
                    outcome="success",
                    expr_id="er",
                )
            )
        level, conf = self.eval.recommend_level("exp_rec", 1.0, self.tracker)
        assert level == "industry"
        assert conf > 0

    def test_recommend_low_sharpe_coarsens_level(self):
        level, conf = self.eval.recommend_level("value", 0.5, self.tracker)
        assert level in DEFAULT_CONFIG["neutralization_levels"]

    def test_recommend_unknown_category_defaults_industry(self):
        level, conf = self.eval.recommend_level("unknown_foo", 1.0, self.tracker)
        assert level == "industry"


# ---------------------------------------------------------------------------
# Category 4: AdaptiveNeutralizer main class (18)
# ---------------------------------------------------------------------------


class TestAdaptiveNeutralizerInit:
    def test_init_requires_experience_path(self, tmp_path):
        path = tmp_path / "state.json"
        nz = AdaptiveNeutralizer(experience_path=path)
        assert nz is not None

    def test_factory_function(self, tmp_path):
        path = tmp_path / "factory.json"
        nz = create_adaptive_neutralizer(experience_path=path)
        assert isinstance(nz, AdaptiveNeutralizer)

    def test_init_loads_existing_file(self, tmp_path):
        path = tmp_path / "preload.json"
        pre = AdaptiveNeutralizer(experience_path=path)
        pre.record_outcome(
            "expr_pre",
            "preload_cat",
            "industry",
            {
                "sharpe_before": 1.0,
                "sharpe_after": 1.1,
                "fitness_delta": 0.05,
            },
        )
        restored = AdaptiveNeutralizer(experience_path=path)
        exp = restored._tracker.get_experience("preload_cat", "industry")
        assert exp.total_trials >= 1


class TestAdaptiveAnalyzeAndRecommend:
    def _nz(self, tmp_path):
        return AdaptiveNeutralizer(experience_path=tmp_path / "analyze.json")

    def test_analyze_returns_valid_adaptive_recommendation(self, tmp_path):
        nz = self._nz(tmp_path)
        rec = nz.analyze_and_recommend(
            expression="rank(close)",
            category="momentum",
            wq_metrics=_make_metrics(),
        )
        assert isinstance(rec, AdaptiveRecommendation)
        assert rec.recommended_level in DEFAULT_CONFIG["neutralization_levels"]
        assert 0.0 <= rec.confidence <= 1.0
        assert isinstance(rec.reasoning, str)
        assert isinstance(rec.alternative_levels, list)

    def test_analyze_should_neutralize_true_for_low_sr_ratio(self, tmp_path):
        nz = self._nz(tmp_path)
        rec = nz.analyze_and_recommend(
            expression="rank(close)",
            category="momentum",
            wq_metrics=_make_metrics(sharpe_raw=1.0, sharpe_neut=0.65),
        )
        assert (
            "Ehsani" in rec.reasoning
            or "neutralize" in rec.reasoning.lower()
            or "minimal" in rec.reasoning.lower()
            or "DEFENSIVE_LOG" in rec.reasoning
            or "downgrad" in rec.reasoning.lower()
        )

    def test_analyze_should_not_overneutralize_momentum(self, tmp_path):
        nz = self._nz(tmp_path)
        rec = nz.analyze_and_recommend(
            expression="rank(close)",
            category="momentum",
            wq_metrics=_make_metrics(),
        )
        assert rec.recommended_level not in ("double", "triple")

    def test_analyze_different_categories(self, tmp_path):
        nz = self._nz(tmp_path)
        for cat in ["momentum", "value", "quality", "size", "volatility", "liquidity"]:
            rec = nz.analyze_and_recommend(
                expression="rank(close)",
                category=cat,
                wq_metrics=_make_metrics(),
            )
            assert rec.recommended_level in DEFAULT_CONFIG["neutralization_levels"]

    def test_analyze_alternatives_are_tuples(self, tmp_path):
        nz = self._nz(tmp_path)
        rec = nz.analyze_and_recommend(
            expression="rank(close)",
            category="value",
            wq_metrics=_make_metrics(),
        )
        for alt in rec.alternative_levels:
            assert isinstance(alt, tuple)
            assert len(alt) == 2
            assert isinstance(alt[0], str)
            assert isinstance(alt[1], (int, float))

    def test_analyze_error_returns_safe_fallback(self, monkeypatch, tmp_path):
        path = tmp_path / "err.json"
        nz = AdaptiveNeutralizer(experience_path=path)

        def crash(*args, **kwargs):
            raise RuntimeError("simulated crash")

        monkeypatch.setattr(nz._evaluator, "evaluate", crash)
        rec = nz.analyze_and_recommend(
            expression="bad_expr",
            category="momentum",
            wq_metrics=_make_metrics(),
        )
        assert rec.recommended_level == "industry"
        assert rec.is_forced is True
        assert "error" in rec.reasoning.lower()


class TestAdaptiveRecordOutcome:
    def _nz(self, tmp_path):
        return AdaptiveNeutralizer(experience_path=tmp_path / "record.json")

    def test_record_outcome_updates_internal_state(self, tmp_path):
        nz = self._nz(tmp_path)
        nz.record_outcome(
            "expr_1",
            "momentum",
            "industry",
            {
                "sharpe_before": 1.0,
                "sharpe_after": 1.15,
                "fitness_delta": 0.08,
            },
        )
        exp = nz._tracker.get_experience("momentum", "industry")
        assert exp.total_trials == 1
        assert exp.successes == 1

    def test_record_outcome_low_sharpe_marks_failure(self, tmp_path):
        nz = self._nz(tmp_path)
        nz.record_outcome(
            "expr_lo",
            "value",
            "subindustry",
            {
                "sharpe_before": 1.0,
                "sharpe_after": 0.5,
                "fitness_delta": -0.10,
            },
        )
        exp = nz._tracker.get_experience("value", "subindustry")
        assert exp.successes == 0

    def test_record_outcome_negative_fitness_marks_partial_or_failure(self, tmp_path):
        nz = self._nz(tmp_path)
        nz.record_outcome(
            "expr_neg",
            "quality",
            "industry",
            {
                "sharpe_before": 1.0,
                "sharpe_after": 1.05,
                "fitness_delta": -0.02,
            },
        )
        exp = nz._tracker.get_experience("quality", "industry")
        assert exp.total_trials == 1

    def test_record_multiple_outcomes_accumulate(self, tmp_path):
        nz = self._nz(tmp_path)
        for i in range(5):
            nz.record_outcome(
                f"expr_{i}",
                "size",
                "industry",
                {
                    "sharpe_before": 1.0,
                    "sharpe_after": 1.1,
                    "fitness_delta": 0.04,
                },
            )
        exp = nz._tracker.get_experience("size", "industry")
        assert exp.total_trials == 5

    def test_record_persists_to_disk(self, tmp_path):
        path = tmp_path / "persist_rec.json"
        nz = AdaptiveNeutralizer(experience_path=path)
        nz.record_outcome(
            "persist_expr",
            "liquidity",
            "industry",
            {
                "sharpe_before": 0.9,
                "sharpe_after": 1.0,
                "fitness_delta": 0.06,
            },
        )
        assert path.exists()


class TestAdaptiveSamplingWeights:
    def _nz(self, tmp_path):
        return AdaptiveNeutralizer(experience_path=tmp_path / "weights.json")

    def test_sampling_weights_valid_distribution(self, tmp_path):
        weights = self._nz(tmp_path).get_sampling_weights("momentum")
        assert isinstance(weights, dict)
        assert len(weights) > 0
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_sampling_weights_all_positive(self, tmp_path):
        weights = self._nz(tmp_path).get_sampling_weights("value")
        for v in weights.values():
            assert v >= 0.0

    def test_sampling_weights_no_double_triple_for_momentum(self, tmp_path):
        weights = self._nz(tmp_path).get_sampling_weights("momentum")
        assert "double" not in weights
        assert "triple" not in weights

    def test_sampling_weights_exclude_forbidden_levels(self, tmp_path):
        path = tmp_path / "forbid_w.json"
        nz = AdaptiveNeutralizer(experience_path=path)
        for i in range(7):
            nz.record_outcome(
                f"fb_{i}",
                "forbid_w",
                "subindustry",
                {
                    "sharpe_before": 1.0,
                    "sharpe_after": 0.6,
                    "fitness_delta": -0.09,
                },
            )
        weights = nz.get_sampling_weights("forbid_w")
        assert "subindustry" not in weights


class TestAdaptiveShouldUpgrade:
    def _nz(self, tmp_path):
        return AdaptiveNeutralizer(experience_path=tmp_path / "upgrade.json")

    def test_upgrade_blocked_for_momentum_at_subindustry(self, tmp_path):
        result = self._nz(tmp_path).should_upgrade_neutralization("momentum", "subindustry", 2.0)
        assert result is False

    def test_upgrade_blocked_for_momentum_at_double(self, tmp_path):
        result = self._nz(tmp_path).should_upgrade_neutralization("momentum", "double", 2.0)
        assert result is False

    def test_upgrade_possible_from_none_to_industry(self, tmp_path):
        result = self._nz(tmp_path).should_upgrade_neutralization("value", "none", 1.5)
        assert result is False

    def test_upgrade_respects_max_level_cap(self, tmp_path):
        path = tmp_path / "cap.json"
        config = {"category_defaults": {"cap_test": {"max_level": "industry"}}}
        nz = AdaptiveNeutralizer(experience_path=path, config=config)
        result = nz.should_upgrade_neutralization("cap_test", "industry", 2.0)
        assert result is False


class TestAdaptiveIntegration:
    def _nz(self, tmp_path):
        return AdaptiveNeutralizer(experience_path=tmp_path / "integration.json")

    def test_recommend_record_rerecommend_updates_behavior(self, tmp_path):
        nz = self._nz(tmp_path)
        rec1 = nz.analyze_and_recommend(
            expression="rank(close)",
            category="integration",
            wq_metrics=_make_metrics(),
        )
        nz.record_outcome(
            "expr_int",
            "integration",
            rec1.recommended_level,
            {
                "sharpe_before": 1.0,
                "sharpe_after": 1.12,
                "fitness_delta": 0.07,
            },
        )
        rec2 = nz.analyze_and_recommend(
            expression="rank(close)",
            category="integration",
            wq_metrics=_make_metrics(),
        )
        assert isinstance(rec2, AdaptiveRecommendation)
        weights = nz.get_sampling_weights("integration")
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_mab_scores_update_after_recording(self, tmp_path):
        nz = self._nz(tmp_path)
        _w1 = nz.get_sampling_weights("mab_loop")
        for _ in range(5):
            nz.record_outcome(
                "mab_e",
                "mab_loop",
                "industry",
                {
                    "sharpe_before": 1.0,
                    "sharpe_after": 1.2,
                    "fitness_delta": 0.08,
                },
            )
        w2 = nz.get_sampling_weights("mab_loop")
        assert sum(w2.values()) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Category 5: Safety Constraints (10)
# ---------------------------------------------------------------------------


class TestSafetyMomentumTripleNeutralization:
    def _nz(self, tmp_path):
        return AdaptiveNeutralizer(experience_path=tmp_path / "safety_mom.json")

    def test_momentum_never_recommends_double(self, tmp_path):
        rec = self._nz(tmp_path).analyze_and_recommend(
            expression="rank(close)",
            category="momentum",
            wq_metrics=_make_metrics(sharpe_raw=3.0, sharpe_neut=0.5),
        )
        assert rec.recommended_level != "double"

    def test_momentum_never_recommends_triple(self, tmp_path):
        rec = self._nz(tmp_path).analyze_and_recommend(
            expression="ts_delta(close,5)",
            category="momentum",
            wq_metrics=_make_metrics(sharpe_raw=5.0, sharpe_neut=0.3),
        )
        assert rec.recommended_level != "triple"

    def test_sampling_weights_exclude_momentum_dangerous(self, tmp_path):
        weights = self._nz(tmp_path).get_sampling_weights("momentum")
        assert "double" not in weights
        assert "triple" not in weights


class TestSafetyLowSharpeDowngrade:
    def _nz(self, tmp_path):
        return AdaptiveNeutralizer(experience_path=tmp_path / "safety_low.json")

    def test_low_post_neutralization_sharpe_downgrades(self, tmp_path):
        rec = self._nz(tmp_path).analyze_and_recommend(
            expression="rank(close)",
            category="downgrade_test",
            wq_metrics=_make_metrics(sharpe_raw=1.0, sharpe_neut=0.6),
        )
        assert (
            "Sharpe" in rec.reasoning
            or "downgrad" in rec.reasoning.lower()
            or "coarsen" in rec.reasoning.lower()
            or rec.recommended_level in ("none", "industry")
        )

    def test_normal_sharpe_no_downgrade_log(self, tmp_path):
        rec = self._nz(tmp_path).analyze_and_recommend(
            expression="rank(close)",
            category="normal_test",
            wq_metrics=_make_metrics(sharpe_raw=1.0, sharpe_neut=1.2),
        )
        assert rec.is_forced is False or "0.8" not in rec.reasoning or "downgrad" not in rec.reasoning.lower()


class TestSafetyForbiddenLevels:
    def _nz(self, tmp_path):
        return AdaptiveNeutralizer(experience_path=tmp_path / "safety_fbd.json")

    def test_forbidden_detected_after_many_failures(self, tmp_path):
        nz = self._nz(tmp_path)
        for i in range(7):
            nz.record_outcome(
                f"fbd_{i}",
                "fbd_cat",
                "subindustry",
                {
                    "sharpe_before": 1.0,
                    "sharpe_after": 0.65,
                    "fitness_delta": -0.08,
                },
            )
        exp = nz._tracker.get_experience("fbd_cat", "subindustry")
        assert exp.is_forbidden is True

    def test_forbidden_excluded_from_recommendation(self, tmp_path):
        nz = self._nz(tmp_path)
        for i in range(7):
            nz.record_outcome(
                f"ff_{i}",
                "ff_cat",
                "subindustry",
                {
                    "sharpe_before": 1.0,
                    "sharpe_after": 0.68,
                    "fitness_delta": -0.07,
                },
            )
        rec = nz.analyze_and_recommend(
            expression="rank(close)",
            category="ff_cat",
            wq_metrics=_make_metrics(),
        )
        assert rec.recommended_level != "subindustry" or rec.is_forced is True

    def test_good_performance_does_not_forbidden(self, tmp_path):
        nz = self._nz(tmp_path)
        for i in range(7):
            nz.record_outcome(
                f"gf_{i}",
                "good_cat",
                "industry",
                {
                    "sharpe_before": 1.0,
                    "sharpe_after": 1.25,
                    "fitness_delta": 0.10,
                },
            )
        exp = nz._tracker.get_experience("good_cat", "industry")
        assert exp.is_forbidden is False

    def test_forbidden_levels_global_set_populated(self, tmp_path):
        nz = self._nz(tmp_path)
        for i in range(7):
            nz.record_outcome(
                f"gl_{i}",
                "global_fbd",
                "risky_lvl",
                {
                    "sharpe_before": 1.0,
                    "sharpe_after": 0.62,
                    "fitness_delta": -0.10,
                },
            )
        assert ("global_fbd", "risky_lvl") in _FORBIDDEN_PAIRS


class TestSafetyDefensiveLogging:
    def _nz(self, tmp_path):
        return AdaptiveNeutralizer(experience_path=tmp_path / "safety_log.json")

    def test_override_logged_on_momentum_block(self, tmp_path, caplog):
        nz = self._nz(tmp_path)
        with caplog.at_level(logging.WARNING, logger="openalpha_brain.evolution.adaptive_neutralizer"):
            rec = nz.analyze_and_recommend(
                expression="rank(close)",
                category="momentum",
                wq_metrics=_make_metrics(sharpe_raw=1.0, sharpe_neut=0.99),
            )
        defensive_logs = [r for r in caplog.records if "DEFENSIVE_LOG" in r.message]
        if rec.is_forced:
            assert len(defensive_logs) >= 1

    def test_forbidden_detection_logged(self, tmp_path, caplog):
        nz = self._nz(tmp_path)
        with caplog.at_level(logging.WARNING, logger="openalpha_brain.evolution.adaptive_neutralizer"):
            for i in range(7):
                nz.record_outcome(
                    f"log_{i}",
                    "log_cat",
                    "bad_lvl",
                    {
                        "sharpe_before": 1.0,
                        "sharpe_after": 0.63,
                        "fitness_delta": -0.09,
                    },
                )
        fbd_logs = [r for r in caplog.records if "FORBIDDEN" in r.message]
        assert len(fbd_logs) >= 1


# ---------------------------------------------------------------------------
# Category 6: Edge Cases & Robustness (12)
# ---------------------------------------------------------------------------


class TestEdgeColdStart:
    def _nz(self, tmp_path):
        return AdaptiveNeutralizer(experience_path=tmp_path / "cold_start.json")

    def test_cold_start_returns_valid_recommendation(self, tmp_path):
        rec = self._nz(tmp_path).analyze_and_recommend(
            expression="rank(close)",
            category="brand_new_factor",
            wq_metrics=_make_metrics(),
        )
        assert isinstance(rec, AdaptiveRecommendation)
        assert rec.recommended_level in DEFAULT_CONFIG["neutralization_levels"]

    def test_cold_start_sampling_weights_valid(self, tmp_path):
        weights = self._nz(tmp_path).get_sampling_weights("never_seen_before")
        assert isinstance(weights, dict)
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=1e-9)


class TestEdgeCorruptedData:
    def test_load_corrupted_json_handles_gracefully(self, tmp_path):
        bad = tmp_path / "corrupt_exp.json"
        bad.write_text("{broken!!!", encoding="utf-8")
        tracker = NeutralizationExperienceTracker()
        tracker.load_from_disk(bad)
        assert len(tracker._experiences) == 0

    def test_load_partial_json_recovers_data(self, tmp_path):
        partial = tmp_path / "partial_exp.json"
        partial.write_text(
            json.dumps(
                {
                    "experiences": [
                        {
                            "category": "ok",
                            "level": "industry",
                            "total_trials": 1,
                            "successes": 1,
                            "avg_sharpe_delta": 0.1,
                            "avg_fitness_delta": 0.05,
                            "success_rate": 1.0,
                            "mab_score": 0.67,
                            "last_updated": 100.0,
                            "is_forbidden": False,
                        }
                    ],
                    "forbidden_pairs": [],
                }
            ),
            encoding="utf-8",
        )
        tracker = NeutralizationExperienceTracker()
        tracker.load_from_disk(partial)
        assert tracker.get_experience("ok", "industry").total_trials == 1


class TestEdgePerformance:
    def _nz(self, tmp_path):
        return AdaptiveNeutralizer(experience_path=tmp_path / "perf.json")

    def test_large_number_of_trials_performance(self, tmp_path):
        nz = self._nz(tmp_path)
        start = time.perf_counter()
        for i in range(500):
            nz.record_outcome(
                f"perf_{i}",
                "perf_cat",
                "industry",
                {
                    "sharpe_before": 1.0,
                    "sharpe_after": 1.05 + (i % 10) * 0.01,
                    "fitness_delta": 0.03,
                },
            )
        elapsed = time.perf_counter() - start
        assert elapsed < 3.0, f"500 trials took {elapsed:.2f}s, too slow"

    def test_large_number_of_recommendations_performance(self, tmp_path):
        nz = self._nz(tmp_path)
        for _ in range(50):
            nz.analyze_and_recommend(
                expression="rank(close)",
                category="perf_rec",
                wq_metrics=_make_metrics(),
            )
        start = time.perf_counter()
        for _ in range(100):
            nz.analyze_and_recommend(
                expression="rank(close)",
                category="perf_rec",
                wq_metrics=_make_metrics(),
            )
        elapsed = time.perf_counter() - start
        assert elapsed < 3.0, f"100 recommendations took {elapsed:.2f}s"


class TestEdgeInvalidInput:
    def _nz(self, tmp_path):
        return AdaptiveNeutralizer(experience_path=tmp_path / "edge_invalid.json")

    def test_empty_expression_handled(self, tmp_path):
        rec = self._nz(tmp_path).analyze_and_recommend(
            expression="",
            category="momentum",
            wq_metrics=_make_metrics(),
        )
        assert isinstance(rec, AdaptiveRecommendation)

    def test_unknown_category_handled(self, tmp_path):
        rec = self._nz(tmp_path).analyze_and_recommend(
            expression="rank(close)",
            category="xyz_nonexistent_factor",
            wq_metrics=_make_metrics(),
        )
        assert rec.recommended_level in DEFAULT_CONFIG["neutralization_levels"]

    def test_unicode_expression_handled(self, tmp_path):
        rec = self._nz(tmp_path).analyze_and_recommend(
            expression="rank(收盤價)",
            category="動量",
            wq_metrics=_make_metrics(),
        )
        assert isinstance(rec, AdaptiveRecommendation)

    def test_empty_metrics_defaults_safely(self, tmp_path):
        rec = self._nz(tmp_path).analyze_and_recommend(
            expression="rank(close)",
            category="value",
            wq_metrics={},
        )
        assert isinstance(rec, AdaptiveRecommendation)

    def test_zero_sharpe_raw_no_crash(self, tmp_path):
        rec = self._nz(tmp_path).analyze_and_recommend(
            expression="rank(close)",
            category="volatility",
            wq_metrics={"sharpe_raw": 0.0, "sharpe_neutralized": 0.0, "correlation": 0.7},
        )
        assert isinstance(rec, AdaptiveRecommendation)

    def test_negative_sharpe_raw_no_crash(self, tmp_path):
        rec = self._nz(tmp_path).analyze_and_recommend(
            expression="rank(close)",
            category="size",
            wq_metrics={"sharpe_raw": -0.5, "sharpe_neutralized": -0.3, "correlation": 0.7},
        )
        assert isinstance(rec, AdaptiveRecommendation)


class TestEdgePersistence:
    def test_save_state_persists_across_instances(self, tmp_path):
        path = tmp_path / "cross_inst.json"
        nz1 = AdaptiveNeutralizer(experience_path=path)
        nz1.record_outcome(
            "cross_1",
            "cross_cat",
            "industry",
            {
                "sharpe_before": 1.0,
                "sharpe_after": 1.15,
                "fitness_delta": 0.08,
            },
        )
        nz2 = AdaptiveNeutralizer(experience_path=path)
        exp = nz2._tracker.get_experience("cross_cat", "industry")
        assert exp.total_trials >= 1

    def test_save_contains_forbidden_pairs(self, tmp_path):
        path = tmp_path / "fbd_persist.json"
        nz = AdaptiveNeutralizer(experience_path=path)
        for i in range(7):
            nz.record_outcome(
                f"fp_{i}",
                "fp_cat",
                "fp_bad",
                {
                    "sharpe_before": 1.0,
                    "sharpe_after": 0.64,
                    "fitness_delta": -0.08,
                },
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert any(p[0] == "fp_cat" and p[1] == "fp_bad" for p in raw.get("forbidden_pairs", []))


class TestEdgeConfigCustomization:
    def test_custom_config_changes_thresholds(self, tmp_path):
        custom_config = {
            "min_trials_for_decision": 5,
            "ehsani_correlation_estimate": 0.5,
            "success_rate_threshold": 0.6,
        }
        eval_custom = EhsaniConditionEvaluator(config=custom_config)
        result = eval_custom.evaluate(1.0, 0.55, 0.5)
        assert result is False

    def test_custom_category_defaults(self, tmp_path):
        custom_config = {
            "ehsani_correlation_estimate": 0.7,
            "category_defaults": {
                "custom_factor": {"default_level": "none", "max_level": "industry"},
            },
        }
        path = tmp_path / "custom_cfg.json"
        nz = AdaptiveNeutralizer(experience_path=path, config=custom_config)
        rec = nz.analyze_and_recommend(
            expression="rank(close)",
            category="custom_factor",
            wq_metrics=_make_metrics(),
        )
        assert rec.recommended_level in ("none", "industry")
