"""Tests for ErrorPatternDB — BRAIN rejection error pattern learning database."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openalpha_brain.learning.error_pattern_db import ErrorPatternDB


@pytest.fixture
def temp_db(tmp_path: Path) -> ErrorPatternDB:
    db_path = tmp_path / "test_error_patterns.json"
    return ErrorPatternDB(db_path=db_path)


class TestErrorPatternExtraction:
    def test_extract_unknown_field(self, temp_db: ErrorPatternDB) -> None:
        pattern = temp_db.extract_error_pattern(
            expression="rank(ts_delta(return_lag_10, 5))",
            error_msg="Error: unknown field 'return_lag_10' in expression",
        )
        assert pattern["type"] == "unknown_field"
        assert pattern["content"] == "return_lag_10"
        assert "SAFE_FIELDS" in pattern["fix"]
        assert pattern["count"] == 1

    def test_extract_invalid_operator(self, temp_db: ErrorPatternDB) -> None:
        pattern = temp_db.extract_error_pattern(
            expression="lag(close, 5)",
            error_msg="Operator 'lag' is not recognized in BRAIN FastExpr",
        )
        assert pattern["type"] == "invalid_operator"
        assert pattern["content"] == "lag"
        assert "ts_delay" in pattern["fix"]

    def test_extract_syntax_error(self, temp_db: ErrorPatternDB) -> None:
        pattern = temp_db.extract_error_pattern(
            expression="rank(ts_delta(close, 5)",
            error_msg="Syntax error: unexpected token at position 25",
        )
        assert pattern["type"] == "syntax_error"
        assert "parentheses" in pattern["fix"].lower() or "syntax" in pattern["fix"].lower()

    def test_extract_timeout(self, temp_db: ErrorPatternDB) -> None:
        pattern = temp_db.extract_error_pattern(
            expression="ts_mean(ts_corr(close, volume, ts_sum(returns, 500), 500), 500)",
            error_msg="Simulation timeout exceeded after 120 seconds",
        )
        assert pattern["type"] == "timeout"

    def test_extract_correlation_too_high(self, temp_db: ErrorPatternDB) -> None:
        pattern = temp_db.extract_error_pattern(
            expression="rank(ts_delta(close, 5))",
            error_msg="Alpha rejected: correlation too high with existing submission",
        )
        assert pattern["type"] == "correlation_too_high"

    def test_extract_low_sharpe(self, temp_db: ErrorPatternDB) -> None:
        pattern = temp_db.extract_error_pattern(
            expression="rank(volume)",
            error_msg="Sharpe ratio too low: below threshold of 1.25",
        )
        assert pattern["type"] == "low_sharpe"

    def test_extract_high_turnover(self, temp_db: ErrorPatternDB) -> None:
        pattern = temp_db.extract_error_pattern(
            expression="rank(ts_delta(close, 1))",
            error_msg="Turnover too high: exceeds maximum allowed 70%",
        )
        assert pattern["type"] == "high_turnover"

    def test_extract_other_fallback(self, temp_db: ErrorPatternDB) -> None:
        pattern = temp_db.extract_error_pattern(
            expression="group_neutralize(rank(close), industry)",
            error_msg="Some completely unknown internal error happened during processing",
        )
        assert pattern["type"] == "other"
        assert len(pattern["content"]) > 0

    def test_extract_with_brain_result(self, temp_db: ErrorPatternDB) -> None:
        brain_result = {
            "status": "FAIL",
            "gate_failures": ["LOW_SHARPE", "HIGH_TURNOVER"],
            "sharpe": 0.8,
        }
        pattern = temp_db.extract_error_pattern(
            expression="rank(close)",
            error_msg="Sharpe ratio too low: below threshold of 1.25",
            brain_result=brain_result,
        )
        assert pattern.get("brain_status") == "FAIL"
        assert pattern.get("gate_failures") == ["LOW_SHARPE", "HIGH_TURNOVER"]
        assert pattern.get("sharpe") == 0.8


class TestStoreAndRetrieve:
    def test_store_and_retrieve_single(self, temp_db: ErrorPatternDB) -> None:
        pattern = temp_db.extract_error_pattern(
            expression="rank(ts_delta(return_lag_10, 5))",
            error_msg="Error: unknown field 'return_lag_10' in expression",
        )
        success = temp_db.store(pattern)
        assert success is True

        top_patterns = temp_db.get_top_patterns(n=10, min_count=1)
        assert len(top_patterns) >= 1
        assert top_patterns[0]["type"] == "unknown_field"
        assert top_patterns[0]["content"] == "return_lag_10"
        assert top_patterns[0]["count"] == 1

    def test_store_increments_count(self, temp_db: ErrorPatternDB) -> None:
        pattern1 = temp_db.extract_error_pattern(
            expression="rank(ts_delta(return_lag_10, 5))",
            error_msg="Error: unknown field 'return_lag_10' in expression",
        )
        temp_db.store(pattern1)

        pattern2 = temp_db.extract_error_pattern(
            expression="-1 * rank(ts_delta(return_lag_10, 10))",
            error_msg="Error: unknown field 'return_lag_10' in expression",
        )
        temp_db.store(pattern2)

        top_patterns = temp_db.get_top_patterns(n=10, min_count=1)
        assert top_patterns[0]["count"] == 2

    def test_store_multiple_types(self, temp_db: ErrorPatternDB) -> None:
        patterns_data = [
            ("rank(ts_delta(x, 5))", "unknown field 'x'", "unknown_field"),
            ("lag(close, 5)", "operator lag not recognized", "invalid_operator"),
            ("rank(ts_delta(close, 5))", "Syntax error: unexpected", "syntax_error"),
        ]
        for expr, msg, expected_type in patterns_data:
            pattern = temp_db.extract_error_pattern(expression=expr, error_msg=msg)
            temp_db.store(pattern)
            assert pattern["type"] == expected_type

        stats = temp_db.get_stats()
        assert stats["total_patterns"] == 3
        assert stats["unique_types"] == 3

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        db_path = tmp_path / "persist_test.json"
        db1 = ErrorPatternDB(db_path=db_path)
        pattern = db1.extract_error_pattern(
            expression="rank(ts_delta(bad_field, 5))",
            error_msg="unknown field 'bad_field'",
        )
        db1.store(pattern)

        db2 = ErrorPatternDB(db_path=db_path)
        top = db2.get_top_patterns(n=10, min_count=1)
        assert len(top) == 1
        assert top[0]["content"] == "bad_field"


class TestBuildNegativeConstraints:
    def test_build_negative_constraints_empty(self, temp_db: ErrorPatternDB) -> None:
        constraints = temp_db.build_negative_constraints(top_n=5)
        assert constraints == ""

    def test_build_negative_constraints_with_patterns(self, temp_db: ErrorPatternDB) -> None:
        for i in range(3):
            pattern = temp_db.extract_error_pattern(
                expression=f"rank(ts_delta(bad_field_{i}, 5))",
                error_msg=f"unknown field 'bad_field_{i}'",
            )
            temp_db.store(pattern)

        constraints = temp_db.build_negative_constraints(top_n=5)
        assert "LEARNED ERROR PATTERNS" in constraints
        assert "AVOID these" in constraints
        assert "bad_field" in constraints
        assert "rejected 3x" in constraints or "rejected" in constraints

    def test_build_negative_limits_to_top_n(self, temp_db: ErrorPatternDB) -> None:
        for i in range(10):
            pattern = temp_db.extract_error_pattern(
                expression=f"rank(field_{i})",
                error_msg=f"error type {i % 7} occurred",
            )
            temp_db.store(pattern)

        constraints = temp_db.build_negative_constraints(top_n=3)
        lines = [l for l in constraints.split("\n") if l.strip().startswith("  ")]
        numbered_lines = [l for l in lines if l.strip() and l.strip()[0].isdigit()]
        assert len(numbered_lines) <= 3


class TestGetStats:
    def test_get_stats_empty(self, temp_db: ErrorPatternDB) -> None:
        stats = temp_db.get_stats()
        assert stats["total_patterns"] == 0
        assert stats["total_rejections"] == 0
        assert stats["unique_types"] == 0

    def test_get_stats_after_storing(self, temp_db: ErrorPatternDB) -> None:
        pattern = temp_db.extract_error_pattern(
            expression="rank(x)",
            error_msg="unknown field 'x'",
        )
        temp_db.store(pattern)

        stats = temp_db.get_stats()
        assert stats["total_patterns"] == 1
        assert stats["total_rejections"] == 1
        assert stats["unique_types"] == 1

    def test_get_stats_multiple_same_type(self, temp_db: ErrorPatternDB) -> None:
        for _ in range(5):
            pattern = temp_db.extract_error_pattern(
                expression="rank(field_a)",
                error_msg="unknown field 'field_a'",
            )
            temp_db.store(pattern)

        stats = temp_db.get_stats()
        assert stats["total_patterns"] == 1
        assert stats["total_rejections"] == 5
        assert stats["unique_types"] == 1


class TestIncrementPattern:
    def test_increment_existing(self, temp_db: ErrorPatternDB) -> None:
        pattern = temp_db.extract_error_pattern(
            expression="rank(x)",
            error_msg="unknown field 'x'",
        )
        temp_db.store(pattern)
        temp_db.increment_pattern("unknown_field:x")

        top = temp_db.get_top_patterns(n=1, min_count=1)
        assert top[0]["count"] == 2

    def test_increment_nonexistent(self, temp_db: ErrorPatternDB) -> None:
        should_not_raise = lambda: temp_db.increment_pattern("nonexistent:key")
        should_not_raise()


class TestGetPatternsByType:
    def test_filter_by_type(self, temp_db: ErrorPatternDB) -> None:
        temp_db.store(temp_db.extract_error_pattern("x", "unknown field 'a'"))
        temp_db.store(temp_db.extract_error_pattern("y", "unknown field 'b'"))
        temp_db.store(temp_db.extract_error_pattern("z", "operator op not recognized"))

        unknown_patterns = temp_db.get_patterns_by_type("unknown_field")
        assert len(unknown_patterns) == 2

        invalid_patterns = temp_db.get_patterns_by_type("invalid_operator")
        assert len(invalid_patterns) == 1


class TestClear:
    def test_clear_all(self, temp_db: ErrorPatternDB) -> None:
        temp_db.store(temp_db.extract_error_pattern("x", "error"))
        assert temp_db.get_stats()["total_patterns"] == 1

        success = temp_db.clear()
        assert success is True
        assert temp_db.get_stats()["total_patterns"] == 0


class TestDatabaseFileFormat:
    def test_json_structure(self, tmp_path: Path) -> None:
        db_path = tmp_path / "format_test.json"
        db = ErrorPatternDB(db_path=db_path)
        db.store(db.extract_error_pattern("expr", "unknown field 'test_field'"))

        with open(db_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "patterns" in data
        assert "stats" in data
        assert "unknown_field:test_field" in data["patterns"]
        entry = data["patterns"]["unknown_field:test_field"]
        assert entry["type"] == "unknown_field"
        assert entry["content"] == "test_field"
        assert "count" in entry
        assert "last_seen" in entry
        assert "fix" in entry

    def test_handle_corrupted_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "corrupted.json"
        db_path.write_text("{invalid json", encoding="utf-8")

        db = ErrorPatternDB(db_path=db_path)
        stats = db.get_stats()
        assert stats["total_patterns"] == 0
