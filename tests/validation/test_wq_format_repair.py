"""Unit Tests for WQ Format Repair Mechanism

Tests for WQFormatRepair class and auto-repair functionality.
Covers all error types: lookback, unknown variable, unexpected character,
parse error, string literal, decay issues, and neutralization problems.
"""

import pytest
from openalpha_brain.validation.wq_format_repair import (
    WQFormatRepair,
    RepairDiagnosis,
    create_wq_format_repairer,
    auto_repair_wq_expression,
)


class TestWQFormatRepairInit:
    """Test initialization and configuration"""
    
    def test_default_init(self):
        repairer = WQFormatRepair()
        assert repairer.default_lookback == 20
    
    def test_custom_lookback(self):
        repairer = WQFormatRepair(default_lookback=10)
        assert repairer.default_lookback == 10
    
    def test_factory_function(self):
        repairer = create_wq_format_repairer(default_lookback=15)
        assert isinstance(repairer, WQFormatRepair)
        assert repairer.default_lookback == 15


class TestDiagnoseLookbackErrors:
    """Test diagnosis of missing lookback parameter errors"""
    
    def test_diagnose_missing_lookback(self):
        repairer = WQFormatRepair()
        error_msg = "Required attribute 'lookback' must have a value"
        expr = "ts_mean(close)"
        
        diagnosis = repairer.diagnose(error_msg, expr)
        
        assert diagnosis.error_type == 'lookback'
        assert diagnosis.confidence > 0.9
        assert 'ts_mean' in diagnosis.affected_operators
        assert diagnosis.original_expression == expr
    
    def test_diagnose_window_error(self):
        repairer = WQFormatRepair()
        error_msg = "Error: window parameter required for ts_rank"
        expr = "ts_zscore(volume)"
        
        diagnosis = repairer.diagnose(error_msg, expr)
        
        assert diagnosis.error_type == 'lookback'
        assert len(diagnosis.affected_operators) > 0
    
    def test_diagnose_multiple_ts_operators(self):
        repairer = WQFormatRepair()
        error_msg = "lookback must have a value"
        expr = "rank(ts_delta(ts_mean(close)))"
        
        diagnosis = repairer.diagnose(error_msg, expr)
        
        assert diagnosis.error_type == 'lookback'
        assert len(diagnosis.affected_operators) >= 2  # ts_delta and ts_mean


class TestDiagnoseUnknownVariable:
    """Test diagnosis of unknown variable/field errors"""
    
    def test_diagnose_unknown_field(self):
        repairer = WQFormatRepair()
        error_msg = "Error: unknown variable 'field_3921'"
        expr = "rank(field_3921)"
        
        diagnosis = repairer.diagnose(error_msg, expr)
        
        assert diagnosis.error_type == 'unknown variable'
        assert diagnosis.confidence > 0.8
    
    def test_diagnose_undefined_variable(self):
        repairer = WQFormatRepair()
        error_msg = "Undefined variable: 'price'"
        expr = "ts_mean(price)"
        
        diagnosis = repairer.diagnose(error_msg, expr)
        
        assert diagnosis.error_type == 'unknown variable'


class TestDiagnoseOtherErrors:
    """Test diagnosis of other error types"""
    
    def test_unexpected_character(self):
        repairer = WQFormatRepair()
        error_msg = "Unexpected character '@' in expression"
        expr = "ts_mean(close@invalid)"
        
        diagnosis = repairer.diagnose(error_msg, expr)
        
        assert diagnosis.error_type == 'unexpected character'
        assert diagnosis.confidence > 0.7
    
    def test_parse_error(self):
        repairer = WQFormatRepair()
        error_msg = "Parse error: unbalanced parentheses"
        expr = "rank((ts_mean(close)"
        
        diagnosis = repairer.diagnose(error_msg, expr)
        
        assert diagnosis.error_type == 'parse error'
    
    def test_string_literal_error(self):
        repairer = WQFormatRepair()
        error_msg = "String literal not expected, got 'industry'"
        expr = "group_neutralize(rank(close))"
        
        diagnosis = repairer.diagnose(error_msg, expr)
        
        assert diagnosis.error_type in ('string literal', 'neutralization')
    
    def test_decay_error(self):
        repairer = WQFormatRepair()
        error_msg = "Invalid decay parameter: value out of range"
        expr = "ts_decay_linear(rank(close), 100)"
        
        diagnosis = repairer.diagnose(error_msg, expr)
        
        assert diagnosis.error_type == 'decay'
    
    def test_neutralization_error(self):
        repairer = WQFormatRepair()
        error_msg = "group_neutralize requires group argument"
        expr = "rank(ts_mean(close, 20))"
        
        diagnosis = repairer.diagnose(error_msg, expr)
        
        assert diagnosis.error_type == 'neutralization'
    
    def test_unknown_error_pattern(self):
        repairer = WQFormatRepair()
        error_msg = "Some completely unrecognized error message"
        expr = "rank(close)"
        
        diagnosis = repairer.diagnose(error_msg, expr)
        
        assert diagnosis.error_type == 'unknown'
        assert diagnosis.confidence == 0.0
    
    def test_empty_inputs(self):
        repairer = WQFormatRepair()
        
        diagnosis = repairer.diagnose("", "")
        assert diagnosis.error_type == 'empty_input'
        
        diagnosis = repairer.diagnose(None, None)
        assert diagnosis.error_type == 'empty_input'


class TestRepairMissingLookback:
    """Test repair of missing lookback parameters"""
    
    def test_repair_single_missing_lookback(self):
        repairer = WQFormatRepair(default_lookback=20)
        expr = "ts_mean(close)"
        error_msg = "lookback required"
        
        diagnosis = repairer.diagnose(error_msg, expr)
        repaired = repairer.repair(diagnosis)
        
        assert "ts_mean(close, 20)" in repaired or repaired == "ts_mean(close, 20)"
    
    def test_repair_multiple_missing_lookbacks(self):
        repairer = WQFormatRepair(default_lookback=15)
        expr = "rank(ts_delta(ts_std_dev(close)))"
        error_msg = "window parameter missing"
        
        diagnosis = repairer.diagnose(error_msg, expr)
        repaired = repairer.repair(diagnosis)
        
        assert ", 15)" in repaired
        assert repaired != expr
    
    def test_no_repair_when_lookback_present(self):
        repairer = WQFormatRepair()
        expr = "ts_mean(close, 20)"
        error_msg = "some other error"
        
        diagnosis = repairer.diagnose(error_msg, expr)
        repaired = repairer.repair(diagnosis)
        
        assert repaired == expr
    
    def test_deterministic_output(self):
        repairer = WQFormatRepair(default_lookback=25)
        expr = "ts_rank(volume)"
        error_msg = "missing window"
        
        diagnosis = repairer.diagnose(error_msg, expr)
        result1 = repairer.repair(diagnosis)
        result2 = repairer.repair(diagnosis)
        
        assert result1 == result2, "Repair should be deterministic"


class TestRepairUnknownVariable:
    """Test repair of unknown variable references"""
    
    def test_repair_known_bad_field(self):
        repairer = WQFormatRepair()
        expr = "rank(field_3921)"
        error_msg = "unknown variable 'field_3921'"
        
        diagnosis = repairer.diagnose(error_msg, expr)
        repaired = repairer.repair(diagnosis)
        
        assert "close" in repaired
        assert "field_3921" not in repaired
    
    def test_repair_price_alias(self):
        repairer = WQFormatRepair()
        expr = "ts_mean(price)"
        error_msg = "undefined variable 'price'"
        
        diagnosis = repairer.diagnose(error_msg, expr)
        repaired = repairer.repair(diagnosis)
        
        assert "close" in repaired.lower()
    
    def test_repair_generic_field_pattern(self):
        repairer = WQFormatRepair()
        expr = "rank(field_1234)"
        error_msg = "unknown variable 'field_1234'"
        
        diagnosis = repairer.diagnose(error_msg, expr)
        repaired = repairer.repair(diagnosis)
        
        assert "close" in repaired


class TestRepairUnexpectedCharacter:
    """Test removal of illegal characters"""
    
    def test_remove_special_chars(self):
        repairer = WQFormatRepair()
        expr = "ts_mean(close@invalid#test$here)"
        error_msg = "Unexpected character"
        
        diagnosis = repairer.diagnose(error_msg, expr)
        repaired = repairer.repair(diagnosis)
        
        assert '@' not in repaired
        assert '#' not in repaired
        assert '$' not in repaired
        assert 'ts_mean' in repaired
    
    def test_preserve_valid_chars(self):
        repairer = WQFormatRepair()
        expr = "rank(ts_mean(close - open)) / volume"
        error_msg = "illegal character found"
        
        diagnosis = repairer.diagnose(error_msg, expr)
        repaired = repairer.repair(diagnosis)
        
        assert '(' in repaired
        assert ')' in repaired
        assert '-' in repaired
        assert '/' in repaired


class TestRepairParseError:
    """Test fixing of parse errors (unbalanced parentheses)"""
    
    def test_fix_missing_close_paren(self):
        repairer = WQFormatRepair()
        expr = "rank(ts_mean(close)"
        error_msg = "parse error: unbalanced parentheses"
        
        diagnosis = repairer.diagnose(error_msg, expr)
        repaired = repairer.repair(diagnosis)
        
        open_count = repaired.count('(')
        close_count = repaired.count(')')
        assert open_count == close_count
    
    def test_fix_missing_open_paren(self):
        repairer = WQFormatRepair()
        expr = "ts_mean(close))"
        error_msg = "syntax error"
        
        diagnosis = repairer.diagnose(error_msg, expr)
        repaired = repairer.repair(diagnosis)
        
        open_count = repaired.count('(')
        close_count = repaired.count(')')
        assert open_count == close_count
    
    def test_fix_trailing_comma(self):
        repairer = WQFormatRepair()
        expr = "rank(ts_mean(close, 20),)"
        error_msg = "parse error"
        
        diagnosis = repairer.diagnose(error_msg, expr)
        repaired = repairer.repair(diagnosis)
        
        assert not repaired.rstrip().endswith(',')


class TestRepairDecayRelated:
    """Test fixing decay parameter issues"""
    
    def test_fix_extreme_decay_window(self):
        repairer = WQFormatRepair()
        expr = "ts_decay_linear(rank(close), 100)"
        error_msg = "decay parameter invalid"
        
        diagnosis = repairer.diagnose(error_msg, expr)
        repaired = repairer.repair(diagnosis)
        
        assert "5)" in repaired  # Default decay window
        assert "100)" not in repaired
    
    def test_add_decay_wrapper(self):
        repairer = WQFormatRepair()
        expr = "group_neutralize(rank(ts_delta(close, 5)), industry)"
        error_msg = "decay error"
        
        diagnosis = repairer.diagnose(error_msg, expr)
        repaired = repairer.repair(diagnosis)
        
        assert "ts_decay_linear" in repaired


class TestRepairNeutralizationMissing:
    """Test adding missing neutralization"""
    
    def test_add_group_neutralize_wrapper(self):
        repairer = WQFormatRepair()
        expr = "rank(ts_mean(close, 20))"
        error_msg = "neutralization required"
        
        diagnosis = repairer.diagnose(error_msg, expr)
        repaired = repairer.repair(diagnosis)
        
        assert "group_neutralize" in repaired
        assert "industry" in repaired
    
    def test_fix_incomplete_neutralize(self):
        repairer = WQFormatRepair()
        expr = "group_neutralize(rank(close))"
        error_msg = "missing group argument"
        
        diagnosis = repairer.diagnose(error_msg, expr)
        repaired = repairer.repair(diagnosis)
        
        assert "industry" in repaired
        assert repaired.count(',') >= 1  # Should have comma before industry


class TestValidateRepaired:
    """Test validation of repaired expressions"""
    
    def test_valid_expression_passes(self):
        repairer = WQFormatRepair()
        expr = "group_neutralize(ts_decay_linear(rank(ts_mean(close, 20)), 10), industry)"
        
        is_valid, warnings = repairer.validate_repaired(expr)
        
        assert is_valid is True
        assert len(warnings) == 0
    
    def test_unbalanced_parens_fails(self):
        repairer = WQFormatRepair()
        expr = "rank(ts_mean(close)"
        
        is_valid, warnings = repairer.validate_repaired(expr)
        
        assert is_valid is False
        assert any("parentheses" in w.lower() for w in warnings)
    
    def test_unknown_operator_warning(self):
        repairer = WQFormatRepair()
        expr = "unknown_func(close)"
        
        is_valid, warnings = repairer.validate_repaired(expr)
        
        assert is_valid is False
        assert any("operator" in w.lower() for w in warnings)
    
    def test_negative_lookback_warning(self):
        repairer = WQFormatRepair()
        expr = "ts_mean(close, -5)"
        
        is_valid, warnings = repairer.validate_repaired(expr)
        
        assert is_valid is False
        assert any("lookback" in w.lower() or "parameter" in w.lower() for w in warnings)
    
    def test_empty_expression_fails(self):
        repairer = WQFormatRepair()
        
        is_valid, warnings = repairer.validate_repaired("")
        
        assert is_valid is False
        assert any("empty" in w.lower() for w in warnings)
    
    def test_none_expression_fails(self):
        repairer = WQFormatRepair()
        
        is_valid, warnings = repairer.validate_repaired(None)
        
        assert is_valid is False


class TestAutoRepairWorkflow:
    """Test the complete auto-repair workflow"""
    
    def test_successful_auto_repair(self):
        error_msg = "Required attribute 'lookback' must have a value"
        expr = "ts_mean(close)"
        
        repaired, diagnosis, was_repaired = auto_repair_wq_expression(error_msg, expr)
        
        assert was_repaired is True
        assert diagnosis.confidence > 0.7
        assert ", 20)" in repaired
        assert repaired != expr
    
    def test_low_confidence_no_repair(self):
        error_msg = "Some strange unrecognized error"
        expr = "rank(close)"
        
        repaired, diagnosis, was_repaired = auto_repair_wq_expression(error_msg, expr)
        
        assert was_repaired is False
        assert repaired == expr  # Original returned unchanged
    
    def test_custom_default_lookback(self):
        error_msg = "window parameter required"
        expr = "ts_rank(volume)"
        
        repaired, diagnosis, was_repaired = auto_repair_wq_expression(
            error_msg, expr, default_lookback=30
        )
        
        assert was_repaired is True
        assert ", 30)" in repaired
    
    def test_unknown_var_auto_repair(self):
        error_msg = "Error: unknown variable 'vol'"
        expr = "rank(vol)"
        
        repaired, diagnosis, was_repaired = auto_repair_wq_expression(error_msg, expr)
        
        assert was_repaired is True
        assert "volume" in repaired.lower()


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_very_long_expression(self):
        repairer = WQFormatRepair()
        long_expr = "rank(" * 50 + "close" + ")" * 50
        
        is_valid, warnings = repairer.validate_repaired(long_expr)
        
        assert any("depth" in w.lower() for w in warnings)
    
    def test_expression_with_comments(self):
        repairer = WQFormatRepair()
        expr_with_comment = "ts_mean(close, 20) # this is a comment"
        error_msg = "unexpected character"
        
        diagnosis = repairer.diagnose(error_msg, expr_with_comment)
        repaired = repairer.repair(diagnosis)
        
        # Comments should be stripped as invalid chars
        assert '#' not in repaired
    
    def test_nested_operators_deeply(self):
        repairer = WQFormatRepair()
        nested = "group_neutralize(ts_decay_linear(rank(ts_zscore(ts_delta(ts_mean(close, 10), 5), 20)), 10), industry)"
        
        is_valid, warnings = repairer.validate_repaired(nested)
        
        # Should be valid despite deep nesting (< 8 levels typically)
        # If it fails, should be depth warning
        if not is_valid:
            assert any("depth" in w.lower() for w in warnings)
    
    def test_unicode_characters(self):
        repairer = WQFormatRepair()
        unicode_expr = "ts_mean(收盘价, 20)"
        error_msg = "unexpected character"
        
        diagnosis = repairer.diagnose(error_msg, unicode_expr)
        repaired = repairer.repair(diagnosis)
        
        # Non-ASCII should be removed
        assert all(ord(c) < 128 for c in repaired)
    
    def test_pure_function_property(self):
        """Verify that repair operations are deterministic pure functions"""
        repairer = WQFormatRepair()
        expr = "ts_mean(close)"
        error_msg = "lookback required"
        
        diagnosis = repairer.diagnose(error_msg, expr)
        
        results = [repairer.repair(diagnosis) for _ in range(10)]
        
        assert len(set(results)) == 1, "All calls should return identical result"
    
    def test_original_unchanged(self):
        """Verify original expression is never modified"""
        repairer = WQFormatRepair()
        original = "ts_mean(close)"
        error_msg = "lookback missing"
        
        diagnosis = repairer.diagnose(error_msg, original)
        repaired = repairer.repair(diagnosis)
        
        assert original == "ts_mean(close)", "Original should be unchanged"
        assert repaired != original, "Repaired should be different"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
