from openalpha_brain.generation.alpha_generator import (
    _extract_expression_from_llm,
    _extract_hallucinations_from_failures,
    _extract_brain_hallucinations,
    _record_hallucination,
    _summarise_rejected,
)
from openalpha_brain.core.loop_engine import _family_locked
from openalpha_brain.core.models import SessionState
from openalpha_brain.validation.validator import validate_syntax


class TestExtractExpressionFromLLM:
    def test_standard_format(self):
        raw = "Expression: group_neutralize(rank(close), industry)"
        result = _extract_expression_from_llm(raw)
        assert result is not None
        assert "group_neutralize" in result

    def test_code_block_format(self):
        raw = "```\ngroup_neutralize(ts_decay_linear(close, 10), industry)\n```"
        result = _extract_expression_from_llm(raw)
        assert result is not None

    def test_no_expression_returns_none(self):
        raw = "This is just a random text with no alpha expression"
        result = _extract_expression_from_llm(raw)
        assert result is None


class TestHallucinationRecording:
    def test_record_hallucination(self):
        state = SessionState(id="test", focus_area="test")
        _record_hallucination(state, "short_ratio", "INVALID_VAR", "test message", "validator")
        assert len(state.hallucination_log) == 1
        assert state.hallucination_log[0]["variable"] == "short_ratio"

    def test_extract_from_syntax_failures(self):
        state = SessionState(id="test", focus_area="test")
        result = validate_syntax("group_neutralize(rank(short_ratio), industry)")
        _extract_hallucinations_from_failures(state, result)
        assert len(state.hallucination_log) > 0

    def test_extract_brain_hallucinations(self):
        state = SessionState(id="test", focus_area="test")
        _extract_brain_hallucinations(state, [
            "Attempted to use unknown variable 'short_ratio'"
        ])
        assert len(state.hallucination_log) == 1
        assert state.hallucination_log[0]["error_type"] == "BRAIN_UNKNOWN_VAR"


class TestFamilyLocked:
    def test_not_locked(self):
        state = SessionState(id="test", focus_area="test")
        state.family_run_tracker = ["Momentum", "Value", "Quality"]
        assert not _family_locked(state)

    def test_locked(self):
        state = SessionState(id="test", focus_area="test")
        state.family_run_tracker = ["Momentum", "Momentum", "Momentum"]
        assert _family_locked(state)
