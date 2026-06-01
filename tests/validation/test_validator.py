from openalpha_brain.validation.validator import (
    PERMITTED_OPERATORS,
    VALID_BRAIN_VARS,
    validate_metrics,
    validate_syntax,
)


class TestValidateSyntax:
    def test_valid_expression_passes(self):
        expr = "group_neutralize(rank(close), industry)"
        result = validate_syntax(expr)
        assert result.passed, f"Should pass but got failures: {result.failures}"

    def test_missing_group_neutralize_fails(self):
        expr = "rank(close)"
        result = validate_syntax(expr)
        assert not result.passed
        assert any("group_neutralize" in f for f in result.failures)

    def test_unknown_operator_fails(self):
        expr = "group_neutralize(unknown_op(close), industry)"
        result = validate_syntax(expr)
        assert not result.passed
        assert any("forbidden" in f.lower() or "unknown" in f.lower() for f in result.failures)

    def test_invalid_variable_detected(self):
        expr = "group_neutralize(rank(short_ratio), industry)"
        result = validate_syntax(expr)
        assert not result.passed

    def test_balanced_parens_check(self):
        expr = "group_neutralize(rank(close, industry)"
        result = validate_syntax(expr)
        assert not result.passed

    def test_operators_loaded_from_json(self):
        assert len(PERMITTED_OPERATORS) >= 60, f"Only {len(PERMITTED_OPERATORS)} operators loaded"

    def test_strict_whitelist_variables(self):
        assert len(VALID_BRAIN_VARS) >= 18, f"Expected at least 18 whitelist vars, got {len(VALID_BRAIN_VARS)}"
        assert "close" in VALID_BRAIN_VARS
        assert "industry" in VALID_BRAIN_VARS
        assert "sales" in VALID_BRAIN_VARS

    def test_missing_lookback_detected(self):
        expr = "group_neutralize(ts_mean(close), industry)"
        result = validate_syntax(expr)
        assert not result.passed
        assert any("lookback" in f.lower() for f in result.failures)

    def test_lookback_provided_passes(self):
        expr = "group_neutralize(ts_mean(close, 20), industry)"
        result = validate_syntax(expr)
        assert result.passed, f"Should pass but got failures: {result.failures}"

    def test_string_literal_detected(self):
        expr = 'group_neutralize(ts_regression(close, vwap, 20, 3, "eps"), industry)'
        result = validate_syntax(expr)
        assert not result.passed
        assert any("string literal" in f.lower() for f in result.failures)

    def test_invalid_var_strict_whitelist(self):
        expr = "group_neutralize(rank(nonexistent_var_xyz), industry)"
        result = validate_syntax(expr)
        assert not result.passed
        assert any("invalid" in f.lower() or "variable" in f.lower() for f in result.failures)


class TestValidateMetrics:
    def test_good_metrics_pass(self):
        parsed = {
            "metrics": {
                "sharpe_min": 1.5,
                "sharpe_max": 2.0,
                "fitness_min": 1.0,
                "fitness_max": 1.5,
                "turnover_min": 10.0,
                "turnover_max": 30.0,
                "returns_pct": 15.0,
            }
        }
        result = validate_metrics(parsed)
        assert result.passed, f"Should pass but got failures: {result.failures}"

    def test_low_sharpe_fails(self):
        parsed = {
            "metrics": {
                "sharpe_min": 0.3,
                "sharpe_max": 0.5,
                "fitness_min": 0.5,
                "turnover_min": 10.0,
                "turnover_max": 30.0,
                "returns_pct": 5.0,
            }
        }
        result = validate_metrics(parsed)
        assert not result.passed

    def test_null_metrics_warning_only(self):
        parsed = {"metrics": {}}
        result = validate_metrics(parsed)
        assert len(result.warnings) > 0
