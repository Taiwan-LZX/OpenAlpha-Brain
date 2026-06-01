import pytest

from openalpha_brain.validation.anti_overfit_detector import (
    AntiOverfitResult,
    FullAntiOverfitDetector,
    LightweightAntiOverfitDetector,
    TestResult,
)


class TestLightweightAntiOverfitDetector:
    """Tests for LightweightAntiOverfitDetector — pure numerical, no pandas required."""

    def _good_metrics(self) -> dict:
        return {
            "sharpe": 1.5,
            "fitness": 2.0,
            "turnover": 0.30,
            "returns": 0.15,
            "drawdown": 0.10,
            "checks": [
                {"name": "LOW_SHARPE", "value": 1.5, "limit": 0.5, "result": True},
                {"name": "HIGH_TURNOVER", "value": 0.30, "limit": 0.70, "result": True},
            ],
        }

    def test_evaluate_returns_result_object(self):
        detector = LightweightAntiOverfitDetector()
        result = detector.evaluate(self._good_metrics())
        assert isinstance(result, AntiOverfitResult)
        assert isinstance(result.score, float)
        assert result.recommendation in ("推荐", "谨慎", "需改进", "不推荐")
        assert result.total_count == 5

    def test_good_metrics_passes_all_tests(self):
        detector = LightweightAntiOverfitDetector()
        result = detector.evaluate(self._good_metrics())
        assert result.passed_count >= 4
        assert result.score >= 80
        assert result.recommendation == "推荐"

    def test_to_dict_serializable(self):
        detector = LightweightAntiOverfitDetector()
        result = detector.evaluate(self._good_metrics())
        d = result.to_dict()
        assert "score" in d
        assert "recommendation" in d
        assert "tests" in d
        assert len(d["tests"]) == 5

    # ---- Test 1: Sharpe Consistency ----

    def test_sharpe_consistency_no_history_passes(self):
        detector = LightweightAntiOverfitDetector()
        result = detector.evaluate({"sharpe": 2.0, "turnover": 0.3, "fitness": 1.5, "returns": 0.1})
        sharpe_test = [t for t in result.tests if t.name == "Sharpe一致性"][0]
        assert sharpe_test.passed
        assert "历史数据不足" in sharpe_test.details.get("note", "")

    def test_sharpe_consistency_within_2sigma_passes(self):
        history = [1.0, 1.2, 0.9, 1.1, 1.3]
        detector = LightweightAntiOverfitDetector(historical_sharpes=history)
        result = detector.evaluate({"sharpe": 1.15, "turnover": 0.3, "fitness": 1.5, "returns": 0.1})
        sharpe_test = [t for t in result.tests if t.name == "Sharpe一致性"][0]
        assert sharpe_test.passed

    def test_sharpe_consistency_outlier_fails(self):
        history = [0.5, 0.6, 0.55, 0.58, 0.52]
        detector = LightweightAntiOverfitDetector(historical_sharpes=history)
        result = detector.evaluate({"sharpe": 5.0, "turnover": 0.3, "fitness": 10.0, "returns": 0.5})
        sharpe_test = [t for t in result.tests if t.name == "Sharpe一致性"][0]
        assert not sharpe_test.passed
        assert sharpe_test.details.get("deviation_sigma", 0) > 2.0

    def test_sharpe_missing_data(self):
        detector = LightweightAntiOverfitDetector()
        result = detector.evaluate({"turnover": 0.3, "fitness": 1.5, "returns": 0.1})
        sharpe_test = [t for t in result.tests if t.name == "Sharpe一致性"][0]
        assert not sharpe_test.passed
        assert "缺少" in sharpe_test.details.get("error", "")

    # ---- Test 2: Turnover Sanity ----

    def test_turnover_ideal_range_passes(self):
        for to in [0.15, 0.25, 0.40, 0.50]:
            detector = LightweightAntiOverfitDetector()
            result = detector.evaluate({"sharpe": 1.0, "turnover": to, "fitness": 1.5, "returns": 0.1})
            to_test = [t for t in result.tests if t.name == "换手率合理性"][0]
            assert to_test.passed, f"TO={to} should pass"

    def test_turnover_too_low_fails(self):
        detector = LightweightAntiOverfitDetector()
        result = detector.evaluate({"sharpe": 1.0, "turnover": 0.005, "fitness": 1.5, "returns": 0.1})
        to_test = [t for t in result.tests if t.name == "换手率合理性"][0]
        assert not to_test.passed
        assert "过低" in to_test.details.get("reason", "")

    def test_turnover_too_high_fails(self):
        detector = LightweightAntiOverfitDetector()
        result = detector.evaluate({"sharpe": 1.0, "turnover": 0.85, "fitness": 1.5, "returns": 0.1})
        to_test = [t for t in result.tests if t.name == "换手率合理性"][0]
        assert not to_test.passed
        assert "过高" in to_test.details.get("reason", "")

    def test_turnover_percentage_format(self):
        detector = LightweightAntiOverfitDetector()
        result = detector.evaluate({"sharpe": 1.0, "turnover": 35.0, "fitness": 1.5, "returns": 0.1})
        to_test = [t for t in result.tests if t.name == "换手率合理性"][0]
        assert to_test.passed
        assert to_test.details.get("turnover_percent") == 35.0

    def test_turnover_zero_fails(self):
        detector = LightweightAntiOverfitDetector()
        result = detector.evaluate({"sharpe": 1.0, "turnover": 0.0, "fitness": 1.5, "returns": 0.1})
        to_test = [t for t in result.tests if t.name == "换手率合理性"][0]
        assert not to_test.passed

    def test_turnover_negative_fails(self):
        detector = LightweightAntiOverfitDetector()
        result = detector.evaluate({"sharpe": 1.0, "turnover": -0.1, "fitness": 1.5, "returns": 0.1})
        to_test = [t for t in result.tests if t.name == "换手率合理性"][0]
        assert not to_test.passed

    def test_turnover_missing(self):
        detector = LightweightAntiOverfitDetector()
        result = detector.evaluate({"sharpe": 1.0, "fitness": 1.5, "returns": 0.1})
        to_test = [t for t in result.tests if t.name == "换手率合理性"][0]
        assert not to_test.passed

    # ---- Test 3: Fitness Efficiency ----

    def test_fitness_normal_passes(self):
        detector = LightweightAntiOverfitDetector()
        result = detector.evaluate({"sharpe": 1.5, "fitness": 2.0, "turnover": 0.3, "returns": 0.1})
        fit_test = [t for t in result.tests if t.name == "Fitness效率"][0]
        assert fit_test.passed

    def test_fitness_suspicious_high_with_low_sharpe_fails(self):
        detector = LightweightAntiOverfitDetector()
        result = detector.evaluate(
            {
                "sharpe": 0.3,
                "fitness": 50.0,
                "turnover": 0.05,
                "returns": 0.01,
            }
        )
        fit_test = [t for t in result.tests if t.name == "Fitness效率"][0]
        assert not fit_test.passed
        assert fit_test.details.get("suspicious") is True

    def test_fitness_missing(self):
        detector = LightweightAntiOverfitDetector()
        result = detector.evaluate({"sharpe": 1.0, "turnover": 0.3, "returns": 0.1})
        fit_test = [t for t in result.tests if t.name == "Fitness效率"][0]
        assert not fit_test.passed

    # ---- Test 4: Drawdown Stability ----

    def test_drawdown_normal_passes(self):
        detector = LightweightAntiOverfitDetector()
        result = detector.evaluate(
            {
                "sharpe": 1.0,
                "turnover": 0.3,
                "fitness": 1.5,
                "returns": 0.1,
                "drawdown": 0.12,
            }
        )
        dd_test = [t for t in result.tests if t.name == "回撤稳定性"][0]
        assert dd_test.passed

    def test_drawdown_too_high_fails(self):
        detector = LightweightAntiOverfitDetector()
        result = detector.evaluate(
            {
                "sharpe": 1.0,
                "turnover": 0.3,
                "fitness": 1.5,
                "returns": 0.1,
                "drawdown": 0.30,
            }
        )
        dd_test = [t for t in result.tests if t.name == "回撤稳定性"][0]
        assert not dd_test.passed
        assert len(dd_test.details.get("warnings", [])) > 0

    def test_low_drawdown_high_sharpe_suspicious(self):
        detector = LightweightAntiOverfitDetector()
        result = detector.evaluate(
            {
                "sharpe": 3.0,
                "turnover": 0.3,
                "fitness": 5.0,
                "returns": 0.3,
                "drawdown": 0.02,
            }
        )
        dd_test = [t for t in result.tests if t.name == "回撤稳定性"][0]
        assert not dd_test.passed

    def test_drawdown_none_passes_by_default(self):
        detector = LightweightAntiOverfitDetector()
        result = detector.evaluate(
            {
                "sharpe": 1.0,
                "turnover": 0.3,
                "fitness": 1.5,
                "returns": 0.1,
            }
        )
        dd_test = [t for t in result.tests if t.name == "回撤稳定性"][0]
        assert dd_test.passed

    def test_drawdown_abs_value_used(self):
        detector = LightweightAntiOverfitDetector()
        result = detector.evaluate(
            {
                "sharpe": 1.0,
                "turnover": 0.3,
                "fitness": 1.5,
                "returns": 0.1,
                "drawdown": -0.20,
            }
        )
        dd_test = [t for t in result.tests if t.name == "回撤稳定性"][0]
        assert dd_test.details.get("drawdown") == 0.2

    # ---- Test 5: Check Pattern Analysis ----

    def test_check_pattern_no_failures_passes(self):
        detector = LightweightAntiOverfitDetector()
        metrics = self._good_metrics()
        metrics["checks"] = [
            {"name": "OK_1", "result": True},
            {"name": "OK_2", "result": True},
        ]
        result = detector.evaluate(metrics)
        cp_test = [t for t in result.tests if t.name == "检查模式分析"][0]
        assert cp_test.passed

    def test_check_pattern_low_sharpe_plus_low_fitness_fails(self):
        detector = LightweightAntiOverfitDetector()
        metrics = self._good_metrics()
        metrics["checks"] = [
            {"name": "LOW_SHARPE", "result": False},
            {"name": "LOW_FITNESS", "result": False},
        ]
        result = detector.evaluate(metrics)
        cp_test = [t for t in result.tests if t.name == "检查模式分析"][0]
        assert not cp_test.passed
        assert cp_test.details.get("severity") == "high"
        assert any("基本面" in p for p in cp_test.details.get("patterns", []))

    def test_check_pattern_high_turnover_low_sharpe_medium_severity(self):
        detector = LightweightAntiOverfitDetector()
        metrics = self._good_metrics()
        metrics["checks"] = [
            {"name": "HIGH_TURNOVER", "result": False},
            {"name": "LOW_SHARPE", "result": False},
        ]
        result = detector.evaluate(metrics)
        cp_test = [t for t in result.tests if t.name == "检查模式分析"][0]
        assert not cp_test.passed
        assert cp_test.details.get("severity") == "medium"

    def test_check_pattern_many_failures_high_severity(self):
        detector = LightweightAntiOverfitDetector()
        metrics = self._good_metrics()
        metrics["checks"] = [
            {"name": "FAIL_A", "result": False},
            {"name": "FAIL_B", "result": False},
            {"name": "FAIL_C", "result": False},
            {"name": "FAIL_D", "result": False},
        ]
        result = detector.evaluate(metrics)
        cp_test = [t for t in result.tests if t.name == "检查模式分析"][0]
        assert not cp_test.passed
        assert cp_test.details.get("severity") == "high"

    def test_check_pattern_empty_list_passes(self):
        detector = LightweightAntiOverfitDetector()
        metrics = self._good_metrics()
        metrics["checks"] = []
        result = detector.evaluate(metrics)
        cp_test = [t for t in result.tests if t.name == "检查模式分析"][0]
        assert cp_test.passed

    def test_check_pattern_string_results_handled(self):
        detector = LightweightAntiOverfitDetector()
        metrics = self._good_metrics()
        metrics["checks"] = [
            {"name": "TEST_CHECK", "result": "fail"},
        ]
        result = detector.evaluate(metrics)
        cp_test = [t for t in result.tests if t.name == "检查模式分析"][0]
        assert "TEST_CHECK" in cp_test.details.get("failed_checks", [])

    # ---- Composite Score Tests ----

    def test_recommendation_thresholds(self):
        all_pass = self._good_metrics()
        detector = LightweightAntiOverfitDetector()

        r5 = detector.evaluate(all_pass)
        assert r5.score == 100.0
        assert r5.recommendation == "推荐"

        metrics_3pass = dict(all_pass)
        metrics_3pass["turnover"] = 0.005
        r3 = detector.evaluate(metrics_3pass)
        assert 40 <= r3.score < 100

        metrics_fail = {
            "sharpe": -1.0,
            "fitness": 100.0,
            "turnover": 0.005,
            "returns": -0.1,
            "drawdown": 0.40,
            "checks": [
                {"name": "LOW_SHARPE", "result": False},
                {"name": "LOW_FITNESS", "result": False},
                {"name": "HIGH_TURNOVER", "result": False},
            ],
        }
        rf = detector.evaluate(metrics_fail)
        assert rf.score < 80


class TestFullAntiOverfitDetector:
    """Tests for FullAntiOverfitDetector — graceful fallback without pandas."""

    def test_init_without_dataframe_not_available(self):
        detector = FullAntiOverfitDetector()
        assert not detector.available
        assert detector.import_error is None

    def test_run_all_raises_without_init(self):
        detector = FullAntiOverfitDetector()
        with pytest.raises(RuntimeError, match="not available"):
            detector.run_all()

    def test_init_from_nonexistent_dataframe_graceful(self):
        detector = FullAntiOverfitDetector()
        detector.init_from_dataframe("not_a_real_dataframe")
        assert not detector.available
        assert detector.import_error is not None


class TestDataClasses:
    """Test dataclass structure and serialization."""

    def test_test_result_fields(self):
        tr = TestResult(name="test", passed=True, details={"key": "value"})
        assert tr.name == "test"
        assert tr.passed is True
        assert tr.details == {"key": "value"}

    def test_anti_overfit_result_defaults(self):
        result = AntiOverfitResult(score=75.0, recommendation="谨慎")
        assert result.tests == []
        assert result.passed_count == 0
        assert result.total_count == 0

    def test_anti_overfit_result_to_dict(self):
        result = AntiOverfitResult(
            score=60.0,
            recommendation="需改进",
            tests=[TestResult("T1", True, {}), TestResult("T2", False, {"err": "x"})],
            passed_count=1,
            total_count=2,
        )
        d = result.to_dict()
        assert d["score"] == 60.0
        assert d["recommendation"] == "需改进"
        assert len(d["tests"]) == 2
        assert d["tests"][0]["passed"] is True
        assert d["tests"][1]["passed"] is False
