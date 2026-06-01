"""
WQ BRAIN 表达式完整验证器 — 合并三个仓库的最佳实现

核心能力：
  1. 语法检查 — 括号匹配、非法字符、字符串字面量
  2. 算子检查 — 白名单验证、参数数量、lookback 正整数
  3. 字段检查 — 字段存在性、允许集合验证
  4. 结构检查 — 嵌套深度、算子总数、三段式结构
  5. 合规检查 — Redline Verifier 规则（禁用函数、长度限制等）
  6. 语义检查 — 除零风险、log/sqrt 负数风险、窗口参数合理性

来源：
  - expression_engine.py: ExpressionEngine 核心验证逻辑
  - alpha_checks.py: AlphaCheckRegistry 检查项
  - redline_verifier.py: 六大技术红线规则
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

# ── 数据结构 ──────────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    """单个检查结果"""

    passed: bool
    name: str
    details: str
    severity: str = "info"  # "error" | "warning" | "info"

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "name": self.name,
            "details": self.details,
            "severity": self.severity,
        }


@dataclass
class ValidationResult:
    """完整验证结果"""

    passed: bool
    checks: dict[str, CheckResult] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    expression: str = ""
    complexity_score: float = 0.0
    semantic_tags: list[str] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for c in self.checks.values() if c.severity == "error" and not c.passed)

    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.checks.values() if c.severity == "warning" and not c.passed)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "expression": self.expression,
            "complexity_score": self.complexity_score,
            "semantic_tags": self.semantic_tags,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "errors": self.errors,
            "warnings": self.warnings,
            "checks": {k: v.to_dict() for k, v in self.checks.items()},
        }


# ── 默认配置常量（来自 expression_engine.py）───────────────────────────────

DEFAULT_MAX_DEPTH = 8
DEFAULT_MAX_NODE_COUNT = 80
DEFAULT_MAX_WINDOW = 512
DEFAULT_MAX_EXPRESSION_LENGTH = 500
MIN_OPERATOR_COUNT = 5

# WQ 平台合法算子白名单（来自 ast_validator.py + brain_operators.json）
OPERATOR_WHITELIST: frozenset[str] = frozenset(
    {
        "add",
        "sqrt",
        "log",
        "subtract",
        "signed_power",
        "sign",
        "reverse",
        "power",
        "multiply",
        "min",
        "max",
        "inverse",
        "densify",
        "abs",
        "divide",
        "and",
        "equal",
        "or",
        "not_equal",
        "not",
        "greater",
        "greater_equal",
        "less_equal",
        "is_nan",
        "if_else",
        "less",
        "ts_sum",
        "ts_zscore",
        "ts_std_dev",
        "ts_mean",
        "ts_scale",
        "ts_rank",
        "ts_quantile",
        "ts_arg_min",
        "ts_regression",
        "kth_element",
        "ts_corr",
        "ts_count_nans",
        "ts_covariance",
        "ts_decay_linear",
        "ts_product",
        "ts_delay",
        "ts_backfill",
        "ts_av_diff",
        "hump",
        "ts_arg_max",
        "last_diff_value",
        "ts_step",
        "ts_delta",
        "days_from_last_change",
        "winsorize",
        "normalize",
        "quantile",
        "rank",
        "scale",
        "zscore",
        "vec_sum",
        "vec_avg",
        "bucket",
        "trade_when",
        "group_scale",
        "group_neutralize",
        "group_zscore",
        "group_backfill",
        "group_mean",
        "group_rank",
    }
)

# 被禁用的函数/模式（来自 redline_verifier.py 红线规则）
FORBIDDEN_PATTERNS: list[tuple[str, str]] = [
    (r"\beval\s*\(", "eval() 函数被禁止"),
    (r"\bexec\s*\(", "exec() 函数被禁止"),
    (r"\b__import__\s*\(", "__import__() 函数被禁止"),
    (r"\bos\.system\b", "os.system 被禁止"),
    (r"\bsubprocess\b", "subprocess 模块被禁止"),
]

# 中性化算子集合
NEUTRALIZE_OPS = frozenset({"group_neutralize", "group_zscore"})

# 衰减算子集合
DECAY_OPS = frozenset({"ts_decay_linear"})

# 分组键字段（不作为数据字段检查）
GROUP_KEYS = frozenset({"market", "sector", "industry", "subindustry", "country", "exchange", "market_cap"})


class WQExpressionValidator:
    """WQ BRAIN 表达式完整验证器

    Usage::
        validator = WQExpressionValidator()
        result = validator.validate_full("group_neutralize(ts_decay_linear(rank(close), 5), industry)")
        if result.passed:
            print("表达式通过所有检查")
        else:
            for check_name, check_result in result.checks.items():
                if not check_result.passed:
                    print(f"[{check_result.severity}] {check_name}: {check_result.details}")
    """

    def __init__(
        self,
        operator_registry: set[str] | None = None,
        allowed_fields: set[str] | None = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_node_count: int = DEFAULT_MAX_NODE_COUNT,
        max_window: int = DEFAULT_MAX_WINDOW,
        max_expression_length: int = DEFAULT_MAX_EXPRESSION_LENGTH,
    ) -> None:
        self._op_reg = operator_registry or set(OPERATOR_WHITELIST)
        self._allowed_fields = allowed_fields
        self._max_depth = max_depth
        self._max_node_count = max_node_count
        self._max_window = max_window
        self._max_expression_length = max_expression_length

    def validate_full(self, expression: str) -> ValidationResult:
        """完整验证（调用所有子检查）"""
        results = {
            "syntax": self.check_syntax(expression),
            "operators": self.check_operators(expression),
            "fields": self.check_fields(expression),
            "structure": self.check_structure(expression),
            "compliance": self.check_compliance(expression),
            "semantics": self.check_semantics(expression),
        }

        all_passed = all(r.passed for r in results.values())
        errors = [r.details for r in results.values() if not r.passed and r.severity == "error"]
        warnings = [r.details for r in results.values() if not r.passed and r.severity == "warning"]

        # 计算复杂度分数
        complexity = self._compute_complexity(expression)

        # 提取语义标签
        tags = self._extract_semantic_tags(expression)

        return ValidationResult(
            passed=all_passed,
            checks=results,
            errors=errors,
            warnings=warnings,
            expression=expression,
            complexity_score=complexity,
            semantic_tags=tags,
        )

    # ══════════════════════════════════════════════════════════════════════
    # 子检查方法
    # ══════════════════════════════════════════════════════════════════════

    def check_syntax(self, expr: str) -> CheckResult:
        """语法检查：括号匹配、非法字符、字符串字面量

        规则来源：expression_engine.py ExpressionEngine.validate()
        """
        issues = []

        if not expr or not expr.strip():
            return CheckResult(
                passed=False,
                name="syntax",
                details="表达式为空",
                severity="error",
            )

        # 1. 括号平衡
        depth = 0
        max_depth = 0
        for char in expr:
            if char == "(":
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == ")":
                depth -= 1
                if depth < 0:
                    issues.append("括号不匹配：右括号多于左括号")

        if depth > 0:
            issues.append(f"括号不匹配：缺少 {depth} 个右括号")

        # 2. 非法字符检查（只允许字母数字_(),./-*+<>及空格，以及连字符用于字段名）
        illegal_chars = re.findall(r"[^a-zA-Z0-9_\(\)\.\,\-\*\/\+\<\>\s]", expr)
        if illegal_chars:
            unique_illegal = sorted(set(illegal_chars))
            issues.append(f"包含非法字符: {unique_illegal}")

        # 3. 字符串字面量检测
        try:
            tree = ast.parse(expr, mode="eval")
            string_literals = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    string_literals.append(node.value[:30])
            if string_literals:
                issues.append(f"包含 {len(string_literals)} 个字符串字面量")
        except SyntaxError:
            pass  # 语法错误会在其他地方报告

        # 4. 尾随逗号检测（在最后一个参数位置）
        expr.rstrip()
        # 检查 ), 或 ,) 模式（WQ 不允许尾随逗号）
        if re.search(r"\,\s*[\)]", expr):
            issues.append("表达式包含尾随逗号")

        passed = len(issues) == 0
        return CheckResult(
            passed=passed,
            name="syntax",
            details="; ".join(issues) if issues else "语法检查通过",
            severity="error" if not passed else "info",
        )

    def check_operators(self, expr: str) -> CheckResult:
        """算子检查：是否在白名单、参数数量、lookback 是否为正整数

        规则来源：expression_engine.py + ast_validator.py
        """
        issues = []

        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            return CheckResult(
                passed=False,
                name="operators",
                details=f"无法解析表达式: {e.msg}",
                severity="error",
            )

        operators_used = []
        unknown_operators = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                op_name = node.func.id
                operators_used.append(op_name)

                # 检查是否在白名单
                if op_name not in self._op_reg:
                    unknown_operators.append(op_name)

                # 检查参数数量（基本合理性）
                arg_count = len(node.args)
                if op_name.startswith("ts_") and arg_count < 1:
                    issues.append(f"{op_name}() 参数不足（时间序列函数至少需要 1 个参数）")

                # 检查 lookback 参数是否为正整数
                if (
                    op_name in {"ts_delta", "ts_delay", "ts_mean", "ts_decay_linear", "ts_sum", "ts_std_dev", "ts_rank"}
                    and len(node.args) >= 2
                ):
                    window_arg = node.args[1]
                    # 处理字面量和一元运算符（如 -5）
                    actual_value = None
                    if isinstance(window_arg, ast.Constant):
                        actual_value = window_arg.value
                    elif isinstance(window_arg, ast.UnaryOp) and isinstance(window_arg.op, ast.USub):
                        if isinstance(window_arg.operand, ast.Constant):
                            actual_value = -window_arg.operand.value

                    if actual_value is not None and isinstance(actual_value, (int, float)):
                        if actual_value <= 0 or not isinstance(actual_value, int):
                            issues.append(f"{op_name}() 的窗口参数应为正整数，实际值: {actual_value}")
                        elif actual_value > self._max_window:
                            issues.append(f"{op_name}() 的窗口参数 {actual_value} 超过上限 {self._max_window}")

        if unknown_operators:
            unique_unknown = sorted(set(unknown_operators))
            issues.append(f"使用未注册算子: {unique_unknown}")

        passed = len(issues) == 0
        return CheckResult(
            passed=passed,
            name="operators",
            details="; ".join(issues) if issues else f"使用了 {len(set(operators_used))} 个合法算子",
            severity="error" if not passed else "info",
        )

    def check_fields(self, expr: str) -> CheckResult:
        """字段检查：字段是否存在、是否来自允许的集合

        规则来源：expression_engine.py ExpressionEngine.validate()
        """
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            return CheckResult(
                passed=False,
                name="fields",
                details=f"无法解析表达式: {e.msg}",
                severity="error",
            )

        # 收集所有标识符（排除算子和数字）
        operators_in_expr = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                operators_in_expr.add(node.func.id)

        all_identifiers = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id not in operators_in_expr:
                # 排除 Python 关键字和内置函数
                if node.id not in {"True", "False", "None", "nan", "inf", "std"}:
                    all_identifiers.add(node.id)

        # 排除分组键
        data_fields = all_identifiers - GROUP_KEYS

        issues = []
        if self._allowed_fields is not None:
            unknown_fields = sorted(data_fields - self._allowed_fields)
            if unknown_fields:
                issues.append(f"使用未知字段: {unknown_fields[:10]}{'...' if len(unknown_fields) > 10 else ''}")

        # 字段数量警告（来自 expression_engine.py _wq_semantic_issues）
        if len(data_fields) > 8:
            issues.append(f"使用了较多字段 ({len(data_fields)} 个)，可能存在过拟合风险")

        passed = len(issues) == 0
        return CheckResult(
            passed=passed,
            name="fields",
            details="; ".join(issues) if issues else f"使用了 {len(data_fields)} 个数据字段",
            severity="error" if any("未知字段" in i for i in issues) else ("warning" if issues else "info"),
        )

    def check_structure(self, expr: str) -> CheckResult:
        """结构检查：嵌套深度、算子总数、三段式结构

        规则来源：expression_engine.py + ast_validator.py 三段式结构识别
        """
        issues = []

        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            return CheckResult(
                passed=False,
                name="structure",
                details=f"无法解析表达式: {e.msg}",
                severity="error",
            )

        # 使用闭包方式收集结构信息
        class StructureCollector:
            def __init__(self):
                self.max_depth = 0
                self.operator_count = 0
                self.root_calls = []
                self._current_depth = 0

            def visit(self, node: ast.AST) -> None:
                if isinstance(node, ast.Call):
                    self._visit_call(node)
                else:
                    self._generic_visit(node)

            def _visit_call(self, node: ast.Call) -> None:
                if isinstance(node.func, ast.Name):
                    self.operator_count += 1
                self._current_depth += 1
                self.max_depth = max(self.max_depth, self._current_depth)
                if self._current_depth == 1 and isinstance(node.func, ast.Name):
                    self.root_calls.append(node.func.id)
                # 遍历子节点
                for child in ast.iter_child_nodes(node):
                    self.visit(child)
                self._current_depth -= 1

            def _generic_visit(self, node: ast.AST) -> None:
                for child in ast.iter_child_nodes(node):
                    self.visit(child)

        collector = StructureCollector()
        collector.visit(tree)

        max_depth = collector.max_depth
        operator_count = collector.operator_count
        root_calls = collector.root_calls

        # 检查嵌套深度
        if max_depth > self._max_depth:
            issues.append(f"嵌套深度超限: {max_depth} > {self._max_depth}")
        elif max_depth >= self._max_depth - 1:
            issues.append(f"嵌套深度接近上限: {max_depth}/{self._max_depth}")

        # 检查算子数量
        if operator_count < MIN_OPERATOR_COUNT:
            issues.append(f"算子数量不足: {operator_count} < {MIN_OPERATOR_COUNT}（建议至少使用 5 个算子）")

        # 检查节点总数
        node_count = sum(1 for _ in ast.walk(tree))
        if node_count > self._max_node_count:
            issues.append(f"AST 节点数超限: {node_count} > {self._max_node_count}")

        # 检查三段式结构（中性化 + 衰减）
        has_neutralize = any(op in NEUTRALIZE_OPS for op in root_calls)
        has_decay = any(op in DECAY_OPS for op in root_calls)

        if not has_neutralize:
            issues.append("缺少中性化层 (group_neutralize/group_zscore)")
        if not has_decay:
            issues.append("缺少衰减层 (ts_decay_linear/ts_av_diff)")

        passed = len([i for i in issues if "超限" in i or "不足" in i]) == 0
        return CheckResult(
            passed=passed,
            name="structure",
            details="; ".join(issues) if issues else f"结构良好 (深度={max_depth}, 算子={operator_count})",
            severity="error" if not passed else ("warning" if issues else "info"),
        )

    def check_compliance(self, expr: str) -> CheckResult:
        """合规检查（redline verifier）：WQ 平台规则

        规则来源：redline_verifier.py 六大技术红线
        """
        issues = []

        # 1. 表达式长度检查
        if len(expr) > self._max_expression_length:
            issues.append(f"表达式长度超限: {len(expr)} > {self._max_expression_length}")

        # 2. 禁用模式检查
        for pattern, desc in FORBIDDEN_PATTERNS:
            if re.search(pattern, expr, re.IGNORECASE):
                issues.append(desc)

        # 3. 危险操作符组合检查
        dangerous_combos = [
            (r"divide\s*\([^)]*0\s*\)", "除零风险：直接除以 0"),
            (r"log\s*\([^)]*-[0-9]", "log 负数风险：参数可能为负"),
            (r"sqrt\s*\([^)]*-", "sqrt 负数风险：参数可能为负"),
        ]
        for pattern, desc in dangerous_combos:
            if re.search(pattern, expr, re.IGNORECASE):
                issues.append(desc)

        passed = len(issues) == 0
        return CheckResult(
            passed=passed,
            name="compliance",
            details="; ".join(issues) if issues else "合规检查通过",
            severity="error" if not passed else "info",
        )

    def check_semantics(self, expr: str) -> CheckResult:
        """语义检查：常见语义错误

        规则来源：alpha_checks.py + 实践经验总结
        """
        issues = []

        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError:
            return CheckResult(
                passed=True,
                name="semantics",
                details="语法错误已在 syntax 检查中报告",
                severity="info",
            )

        # 收集所有函数调用及其参数
        calls_info = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                op_name = node.func.id
                args_str = []
                for arg in node.args[:3]:  # 只看前 3 个参数
                    if isinstance(arg, ast.Constant):
                        args_str.append(str(arg.value))
                    elif isinstance(arg, ast.Name):
                        args_str.append(arg.id)
                    elif isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                        # 处理负数字面量：-5 -> "-5"
                        if isinstance(arg.operand, ast.Constant):
                            args_str.append(f"-{arg.operand.value}")
                        else:
                            args_str.append("...")
                    else:
                        args_str.append("...")
                calls_info.append((op_name, args_str))

        # 1. 除零风险检测
        div_calls = [(name, args) for name, args in calls_info if name == "divide"]
        for _, args in div_calls:
            if len(args) >= 2 and args[1] in {"0", "0.0"}:
                issues.append("除零风险：divide 第二个参数为 0")

        # 2. log 负数风险
        log_calls = [(name, args) for name, args in calls_info if name == "log"]
        for _, args in log_calls:
            if len(args) >= 1:
                # 如果参数是负数字面量
                try:
                    val = float(args[0])
                    if val <= 0:
                        issues.append(f"log 负数风险：参数 {args[0]} <= 0")
                except ValueError:
                    pass  # 参数是变量名，无法静态判断

        # 3. sqrt 负数风险
        sqrt_calls = [(name, args) for name, args in calls_info if name == "sqrt"]
        for _, args in sqrt_calls:
            if len(args) >= 1:
                try:
                    val = float(args[0])
                    if val < 0:
                        issues.append(f"sqrt 负数风险：参数 {args[0]} < 0")
                except ValueError:
                    pass

        # 4. ts_delta/ts_rank 窗口参数合理性
        for op_name, args in calls_info:
            if op_name in {"ts_delta", "ts_rank"} and len(args) >= 2:
                try:
                    window = int(float(args[1]))
                    if window < 2:
                        issues.append(f"{op_name} 窗口参数过小: {window} (建议 >= 2)")
                    if window > 252:
                        issues.append(f"{op_name} 窗口参数过大: {window} (建议 <= 252)")
                except (ValueError, TypeError):
                    pass

        # 5. 缺少时间序列或横截面变换（来自 expression_engine.py _wq_semantic_issues）
        operators_used = {name for name, _ in calls_info}
        has_time_series = any(op.startswith("ts_") for op in operators_used)
        has_cross_sectional = operators_used & {"rank", "zscore", "scale", "group_rank", "group_zscore"}
        if not has_time_series and not has_cross_sectional:
            issues.append("缺少明确的时间序列或横截面变换")

        passed = len(issues) == 0
        return CheckResult(
            passed=passed,
            name="semantics",
            details="; ".join(issues) if issues else "语义检查通过",
            severity="warning" if issues else "info",
        )

    # ══════════════════════════════════════════════════════════════════════
    # 辅助方法
    # ══════════════════════════════════════════════════════════════════════

    def _compute_complexity(self, expr: str) -> float:
        """计算表达式复杂度分数（来自 expression_engine.py complexity_score）"""
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError:
            return 100.0  # 无法解析时返回最高复杂度

        node_count = sum(1 for _ in ast.walk(tree))

        operators = set()
        fields = set()
        windows = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                operators.add(node.func.id)
                # 收集窗口参数
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    if isinstance(node.args[1].value, (int, float)):
                        windows.append(node.args[1].value)
            elif isinstance(node, ast.Name) and node.id not in operators:
                if node.id not in GROUP_KEYS and node.id not in {"True", "False", "None", "nan", "inf", "std"}:
                    fields.add(node.id)

        # 嵌套深度
        depth = 0
        max_depth = 0

        class DepthVisitor(ast.NodeVisitor):
            nonlocal depth, max_depth

            def visit_Call(self, node: ast.Call) -> None:
                nonlocal depth, max_depth
                depth += 1
                max_depth = max(max_depth, depth)
                self.generic_visit(node)
                depth -= 1

        DepthVisitor().visit(tree)

        raw = node_count * 1.0 + max_depth * 4.0 + len(operators) * 2.0 + len(fields) * 1.5 + len(windows) * 0.75
        return round(min(100.0, raw), 2)

    def _extract_semantic_tags(self, expr: str) -> list[str]:
        """提取语义标签（来自 expression_engine.py semantic_tags）"""
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError:
            return []

        operators = set()
        windows = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                operators.add(node.func.id)
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    if isinstance(node.args[1].value, (int, float)):
                        windows.append(node.args[1].value)

        tags = []

        if any(op.startswith("ts_") for op in operators):
            tags.append("time_series")
        if operators & {"rank", "zscore", "scale", "group_rank", "group_zscore"}:
            tags.append("cross_sectional")
        if operators & {"group_rank", "group_zscore", "group_neutralize", "regression_neut"}:
            tags.append("group_aware")
        if operators & {"winsorize", "truncate", "scale", "group_neutralize"}:
            tags.append("risk_control")
        if any(w <= 7 for w in windows):
            tags.append("short_horizon")
        if any(8 <= w <= 60 for w in windows):
            tags.append("medium_horizon")
        if any(w > 60 for w in windows):
            tags.append("long_horizon")

        return tags
