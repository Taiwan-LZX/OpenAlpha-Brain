import pytest
from openalpha_brain.generation.prompts import (
    build_brain_failure_feedback,
    build_idea_diagnostic_feedback,
    build_fine_tune_guidance,
    _BRAIN_CHECK_DIAGNOSTICS,
)


class TestBrainCheckDiagnostics:
    def test_diagnostics_count(self):
        assert len(_BRAIN_CHECK_DIAGNOSTICS) >= 15

    def test_key_diagnostics_present(self):
        expected_keys = ["LOW_SHARPE", "HIGH_TURNOVER", "LOW_TURNOVER", "SELF_CORRELATION", "LOW_FITNESS", "CONCENTRATED_WEIGHT", "LOW_SUB_UNIVERSE_SHARPE"]
        for key in expected_keys:
            assert key in _BRAIN_CHECK_DIAGNOSTICS, f"Missing diagnostic: {key}"


class TestBuildBrainFailureFeedback:
    def test_build_feedback(self):
        checks = [{"name": "LOW_SHARPE", "value": 0.8, "limit": 1.25, "result": "FAIL"}]
        feedback = build_brain_failure_feedback(
            checks, "rank(close)", cycle=1,
            real_sharpe=0.8, real_fitness=0.9,
            real_turnover=0.5, real_returns=0.02,
            brain_alpha_id="abc123",
        )
        assert isinstance(feedback, str)
        assert len(feedback) > 0


class TestBuildIdeaDiagnosticFeedback:
    def test_build_diagnostic(self):
        checks = [{"name": "LOW_SHARPE", "value": 0.8, "limit": 1.25, "result": "FAIL"}]
        feedback = build_idea_diagnostic_feedback(
            checks, sharpe=0.8, fitness=0.9,
            turnover=0.5, direction="momentum",
        )
        assert isinstance(feedback, str)


class TestBuildFineTuneGuidance:
    def test_momentum_guidance(self):
        guidance = build_fine_tune_guidance("momentum")
        assert isinstance(guidance, str)

    def test_with_brain_checks(self):
        guidance = build_fine_tune_guidance("momentum", [{"name": "LOW_SHARPE"}])
        assert "LOW_SHARPE" in guidance or "normalization" in guidance.lower()
