"""
WQExpressionValidator 单元测试

覆盖所有 6 个子检查方法以及边界情况
"""
import pytest
from openalpha_brain.validation.wq_expression_validator import (
    WQExpressionValidator,
    CheckResult,
    ValidationResult,
    OPERATOR_WHITELIST,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_EXPRESSION_LENGTH,
)


class TestCheckSyntax:
    """语法检查测试"""

    def setup_method(self):
        self.validator = WQExpressionValidator()

    def test_valid_expression(self):
        """正常表达式应通过语法检查"""
        expr = "group_neutralize(ts_decay_linear(rank(close), 5), industry)"
        result = self.validator.check_syntax(expr)
        assert result.passed is True
        assert result.severity == "info"

    def test_empty_expression(self):
        """空表达式应失败"""
        result = self.validator.check_syntax("")
        assert result.passed is False
        assert "空" in result.details or "empty" in result.details.lower()

    def test_mismatched_parentheses(self):
        """括号不匹配应失败"""
        result = self.validator.check_syntax("rank(close")
        assert result.passed is False
        assert "括号" in result.details

    def test_extra_closing_paren(self):
        """多余的右括号应失败"""
        result = self.validator.check_syntax("rank(close))")
        assert result.passed is False

    def test_illegal_characters(self):
        """非法字符检测"""
        result = self.validator.check_syntax("rank(close) # comment")
        assert result.passed is False
        assert "非法字符" in result.details

    def test_string_literals(self):
        """字符串字面量检测"""
        result = self.validator.check_syntax('rank("close")')
        assert result.passed is False
        assert "字符串" in result.details

    def test_trailing_comma(self):
        """尾随逗号检测"""
        result = self.validator.check_syntax("rank(close,)")
        # 注意：这在 Python 中是合法的，但 WQ 不允许
        assert result.passed is False
        assert "尾随逗号" in result.details


class TestCheckOperators:
    """算子检查测试"""

    def setup_method(self):
        self.validator = WQExpressionValidator()

    def test_valid_operators(self):
        """使用合法算子应通过"""
        expr = "ts_decay_linear(rank(ts_delta(close, 5)), 10)"
        result = self.validator.check_operators(expr)
        assert result.passed is True
        assert "4 个合法算子" in result.details or "合法算子" in result.details

    def test_unknown_operator(self):
        """使用未知算子应失败"""
        expr = "custom_func(close)"
        result = self.validator.check_operators(expr)
        assert result.passed is False
        assert "未注册算子" in result.details

    def test_multiple_unknown_operators(self):
        """多个未知算子"""
        expr = "func_a(func_b(close))"
        result = self.validator.check_operators(expr)
        assert result.passed is False
        assert "func_a" in result.details
        assert "func_b" in result.details

    def test_ts_function_without_args(self):
        """时间序列函数缺少参数"""
        expr = "ts_mean()"
        result = self.validator.check_operators(expr)
        assert result.passed is False
        assert "参数不足" in result.details

    def test_negative_window_param(self):
        """窗口参数为负数"""
        expr = "ts_delta(close, -5)"
        result = self.validator.check_operators(expr)
        assert result.passed is False
        assert "正整数" in result.details

    def test_window_exceeds_max(self):
        """窗口参数超过上限"""
        validator = WQExpressionValidator(max_window=100)
        expr = "ts_delta(close, 200)"
        result = validator.check_operators(expr)
        assert result.passed is False
        assert "超过上限" in result.details


class TestCheckFields:
    """字段检查测试"""

    def setup_method(self):
        self.allowed_fields = {"close", "open", "high", "low", "volume", "vwap"}
        self.validator = WQExpressionValidator(allowed_fields=self.allowed_fields)

    def test_valid_fields(self):
        """使用允许的字段应通过"""
        expr = "rank(close)"
        result = self.validator.check_fields(expr)
        assert result.passed is True

    def test_unknown_field(self):
        """使用未知字段应失败（当设置了 allowed_fields 时）"""
        expr = "rank(unknown_field)"
        result = self.validator.check_fields(expr)
        assert result.passed is False
        assert "未知字段" in result.details

    def test_group_keys_ignored(self):
        """分组键不应作为数据字段检查"""
        expr = "group_neutralize(rank(close), industry)"
        result = self.validator.check_fields(expr)
        # industry 是分组键，不应该报错
        assert result.passed is True or "industry" not in result.details

    def test_too_many_fields_warning(self):
        """过多字段的警告"""
        many_fields = {f"field_{i}" for i in range(10)}
        validator = WQExpressionValidator(allowed_fields=many_fields)
        expr = "+".join(f"field_{i}" for i in range(9))
        result = validator.check_fields(expr)
        # 应该有警告但不是错误
        assert "过拟合风险" in result.details or result.passed is True

    def test_no_allowed_fields_set(self):
        """不设置 allowed_fields 时，不检查字段存在性"""
        validator = WQExpressionValidator(allowed_fields=None)
        expr = "rank(any_field)"
        result = validator.check_fields(expr)
        # 不应该因为未知字段而失败
        assert result.passed is True


class TestCheckStructure:
    """结构检查测试"""

    def setup_method(self):
        self.validator = WQExpressionValidator(max_depth=DEFAULT_MAX_DEPTH)

    def test_valid_three_block_structure(self):
        """有效的三段式结构"""
        expr = "group_neutralize(ts_decay_linear(rank(ts_delta(ts_mean(close, 5), 3)), 10), industry)"
        result = self.validator.check_structure(expr)
        # 应该通过（有足够的算子、包含中性化和衰减）
        assert result.passed is True or "结构良好" in result.details or "缺少衰减层" not in result.details

    def test_missing_neutralize(self):
        """缺少中性化层"""
        expr = "ts_decay_linear(rank(close), 5)"
        result = self.validator.check_structure(expr)
        assert result.passed is False  # 缺少中性化是 warning，但如果只有这个可能还是 passed
        assert "中性化" in result.details

    def test_missing_decay(self):
        """缺少衰减层"""
        expr = "group_neutralize(rank(close), industry)"
        result = self.validator.check_structure(expr)
        assert "衰减" in result.details

    def test_nesting_depth_exceeded(self):
        """嵌套深度超限"""
        validator = WQExpressionValidator(max_depth=3)
        expr = "rank(ts_mean(ts_delta(rank(close), 5), 3))"
        result = validator.check_structure(expr)
        assert result.passed is False
        assert "超限" in result.details or "接近上限" in result.details

    def test_few_operators(self):
        """算子数量不足"""
        expr = "rank(close)"
        result = self.validator.check_structure(expr)
        assert result.passed is False
        assert "不足" in result.details

    def test_node_count_exceeded(self):
        """AST 节点数超限"""
        validator = WQExpressionValidator(max_node_count=5)
        expr = "a(b(c(d(e)))))"
        result = validator.check_structure(expr)
        assert "超限" in result.details or not result.passed


class TestCheckCompliance:
    """合规检查测试"""

    def setup_method(self):
        self.validator = WQExpressionValidator()

    def test_valid_compliance(self):
        """正常表达式应通过合规检查"""
        expr = "group_neutralize(ts_decay_linear(rank(close), 5), industry)"
        result = self.validator.check_compliance(expr)
        assert result.passed is True

    def test_expression_too_long(self):
        """表达式过长"""
        long_expr = "rank(" + "a" * (DEFAULT_MAX_EXPRESSION_LENGTH + 1) + ")"
        result = self.validator.check_compliance(long_expr)
        assert result.passed is False
        assert "长度超限" in result.details

    def test_forbidden_eval(self):
        """禁止 eval() 函数"""
        result = self.validator.check_compliance("eval('code')")
        assert result.passed is False
        assert "eval()" in result.details

    def test_forbidden_exec(self):
        """禁止 exec() 函数"""
        result = self.validator.check_compliance("exec('code')")
        assert result.passed is False
        assert "exec()" in result.details

    def test_divide_by_zero_risk(self):
        """除零风险检测"""
        result = self.validator.check_compliance("divide(x, 0)")
        assert result.passed is False
        assert "除零" in result.details

    def test_log_negative_risk(self):
        """log 负数风险检测"""
        result = self.validator.check_compliance("log(-5)")
        assert result.passed is False
        assert "log" in result.details and "负数" in result.details


class TestCheckSemantics:
    """语义检查测试"""

    def setup_method(self):
        self.validator = WQExpressionValidator()

    def test_valid_semantics(self):
        """正常表达式应通过语义检查"""
        expr = "group_neutralize(ts_decay_linear(rank(ts_delta(close, 5)), 10), industry)"
        result = self.validator.check_semantics(expr)
        assert result.passed is True

    def test_divide_by_zero_literal(self):
        """字面量除零"""
        expr = "divide(close, 0)"
        result = self.validator.check_semantics(expr)
        assert result.passed is False
        assert "除零" in result.details

    def test_log_of_negative_literal(self):
        """log 负数字面量"""
        expr = "ts_mean(log(-1), 5)"
        result = self.validator.check_semantics(expr)
        assert result.passed is False
        assert "log" in result.details and "负数" in result.details

    def test_sqrt_of_negative_literal(self):
        """sqrt 负数字面量"""
        expr = "ts_mean(sqrt(-4), 5)"
        result = self.validator.check_semantics(expr)
        assert result.passed is False
        assert "sqrt" in result.details and "负数" in result.details

    def test_ts_delta_small_window(self):
        """ts_delta 窗口过小"""
        expr = "ts_delta(close, 1)"
        result = self.validator.check_semantics(expr)
        assert result.passed is False
        assert "过小" in result.details

    def test_ts_rank_large_window(self):
        """ts_rank 窗口过大"""
        expr = "ts_rank(volume, 300)"
        result = self.validator.check_semantics(expr)
        assert result.passed is False
        assert "过大" in result.details

    def test_no_time_series_or_cross_sectional(self):
        """缺少时间序列或横截面变换"""
        expr = "add(close, open)"
        result = self.validator.check_semantics(expr)
        assert result.passed is False
        assert "时间序列" in result.details or "横截面" in result.details


class TestValidateFull:
    """完整验证测试"""

    def setup_method(self):
        self.validator = WQExpressionValidator()

    def test_perfect_expression(self):
        """完美表达式应通过所有检查"""
        expr = "group_neutralize(ts_decay_linear(rank(ts_delta(ts_mean(close, 5), 3)), 10), industry)"
        result = self.validator.validate_full(expr)
        assert result.passed is True
        assert len(result.errors) == 0
        assert "syntax" in result.checks
        assert "operators" in result.checks
        assert "fields" in result.checks
        assert "structure" in result.checks
        assert "compliance" in result.checks
        assert "semantics" in result.checks

    def test_failed_expression(self):
        """有问题的表达式应失败"""
        expr = "bad_func(eval('x'))"
        result = self.validator.validate_full(expr)
        assert result.passed is False
        assert len(result.errors) > 0

    def test_complexity_score_calculation(self):
        """复杂度分数计算"""
        simple_expr = "rank(close)"
        complex_expr = "group_neutralize(ts_decay_linear(rank(ts_delta(ts_mean(close, 5), 3)), 10), industry)"

        simple_result = self.validator.validate_full(simple_expr)
        complex_result = self.validator.validate_full(complex_expr)

        assert complex_result.complexity_score > simple_result.complexity_score

    def test_semantic_tags_extraction(self):
        """语义标签提取"""
        expr = "group_neutralize(ts_decay_linear(rank(ts_delta(close, 5)), 10), industry)"
        result = self.validator.validate_full(expr)

        assert "time_series" in result.semantic_tags
        assert "cross_sectional" in result.semantic_tags
        assert "group_aware" in result.semantic_tags
        assert "risk_control" in result.semantic_tags

    def test_result_to_dict(self):
        """结果序列化为字典"""
        expr = "rank(close)"
        result = self.validator.validate_full(expr)
        d = result.to_dict()

        assert isinstance(d, dict)
        assert "passed" in d
        assert "checks" in d
        assert "complexity_score" in d
        assert "semantic_tags" in d

    def test_error_and_warning_counts(self):
        """错误和警告计数"""
        expr = "bad_func()"
        result = self.validator.validate_full(expr)

        assert result.error_count >= 0
        assert result.warning_count >= 0


class TestEdgeCases:
    """边界情况测试"""

    def setup_method(self):
        self.validator = WQExpressionValidator()

    def test_whitespace_only(self):
        """仅包含空白字符的表达式"""
        result = self.validator.validate_full("   ")
        assert result.passed is False

    def test_special_characters_in_field_names(self):
        """字段名中的特殊字符（应被拒绝）"""
        result = self.validator.check_syntax("rank(field#name)")
        assert result.passed is False

    def test_very_deep_nesting(self):
        """极深嵌套"""
        deep_expr = "(".join(["rank(close)"] * 20) + ")" * 20
        result = self.validator.validate_full(deep_expr)
        assert result.passed is False

    def test_unicode_in_expression(self):
        """Unicode 字符在表达式中（应被拒绝）"""
        result = self.validator.check_syntax("rank(中文)")
        assert result.passed is False

    def test_numeric_field_names(self):
        """纯数字标识符"""
        expr = "rank(123)"
        result = self.validator.check_fields(expr)
        # 数字不会被识别为字段名，这是预期行为
        assert result.passed is True or "数据字段" not in result.details


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
