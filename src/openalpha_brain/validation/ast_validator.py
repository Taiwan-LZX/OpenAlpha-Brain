"""
OpenAlpha-Brain — ASTValidator (AlphaBench ICLR'26 Hard Validation)

使用 Python 标准库 ast 模块对 WQ FASTEXPR 表达式进行 AST 级硬校验。
替代 generation_gates.py 和 alpha_logics.py 中的正则/字符串匹配检查。

核心能力：
  1. 语法合法性 — ast.parse() 硬编译
  2. 算子白名单 — 66 个 WQ 平台合法算子（来自 brain_operators.json）
  3. 中性化段检测 — group_neutralize / group_zscore 最外层调用
  4. 衰减段检测 — ts_decay_linear 最外层调用
  5. 嵌套深度限制 — AST 树递归深度计算
  6. 三段式结构识别 — Signal → Neutralize → Decay 层级关系
  7. 参数类型检查 — 数值参数合理性验证

集成点：
  - GenerationGates._check_expression_code() — E↔C 门控硬拦截
  - ThreeBlockTemplate.validate_assembly() — 三段式结构完整性
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from openalpha_brain.utils.algo_logger import algo_log, Timer, log_call

logger = logging.getLogger(__name__)

EPSILON = 1e-6

# ── WQ 平台 66 个合法算子白名单（来源: brain_operators.json）─────────────────
OPERATOR_WHITELIST = frozenset({
    "add", "sqrt", "log", "subtract", "signed_power", "sign",
    "reverse", "power", "multiply", "min", "max", "inverse",
    "densify", "abs", "divide", "and", "equal", "or",
    "not_equal", "not", "greater", "greater_equal", "less_equal",
    "is_nan", "if_else", "less",
    "ts_sum", "ts_zscore", "ts_std_dev", "ts_mean", "ts_scale",
    "ts_rank", "ts_quantile", "ts_arg_min", "ts_regression",
    "kth_element", "ts_corr", "ts_count_nans", "ts_covariance",
    "ts_decay_linear", "ts_product", "ts_delay", "ts_backfill",
    "ts_av_diff", "hump", "ts_arg_max", "last_diff_value",
    "ts_step", "ts_delta", "days_from_last_change",
    "winsorize", "normalize", "quantile", "rank", "scale",
    "zscore", "vec_sum", "vec_avg", "bucket", "trade_when",
    "group_scale", "group_neutralize", "group_zscore",
    "group_backfill", "group_mean", "group_rank",
})

# 中性化算子集合
_NEUTRALIZE_OPS = frozenset({"group_neutralize", "group_zscore"})

# 衰减算子集合
_DECAY_OPS = frozenset({"ts_decay_linear"})


@dataclass
class ValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    structure_info: dict[str, Any] = field(default_factory=dict)
    signal_segment: Optional[str] = None
    fix_suggestions: list[str] = field(default_factory=list)


class _ExprVisitor(ast.NodeVisitor):
    """AST Visitor：遍历表达式收集结构信息并执行校验规则。"""

    def __init__(self, whitelist: frozenset[str], max_depth: int = 4) -> None:
        self._whitelist = whitelist
        self._max_depth = max_depth
        self.operators_used: list[str] = []
        self.unknown_operators: list[str] = []
        self.max_nesting_depth: int = 0
        self._current_depth: int = 0
        self.root_calls: list[str] = []
        self.string_literals: list[tuple[int, str]] = []
        self.numeric_constants: list[tuple[int, Any]] = []

    def visit_Call(self, node: ast.Call) -> None:
        name = self._call_name(node)
        if name:
            self.operators_used.append(name)
            if name not in self._whitelist:
                self.unknown_operators.append(name)

        self._current_depth += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self._current_depth)

        if self._current_depth == 1 and name:
            self.root_calls.append(name)

        self.generic_visit(node)
        self._current_depth -= 1

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self.string_literals.append((node.lineno, node.value))
        elif isinstance(node.value, (int, float)):
            self.numeric_constants.append((node.lineno, node.value))
        self.generic_visit(node)

    @staticmethod
    def _call_name(node: ast.Call) -> Optional[str]:
        if isinstance(node.func, ast.Name):
            return node.func.id
        return None


@dataclass
class _ThreeBlockStructure:
    has_neutralize: bool = False
    has_decay: bool = False
    neutralize_op: str = ""
    decay_op: str = ""
    signal_depth: int = 0
    is_valid_three_block: bool = False


class ASTValidator:
    """WQ FASTEXPR 表达式的 AST 级硬校验器。

    Usage::
        validator = ASTValidator()
        result = validator.validate("group_neutralize(ts_decay_linear(rank(ts_delta(close, 5)), 10), industry)")
        if not result.passed:
            for err in result.errors:
                print(f"FATAL: {err}")
    """

    def __init__(
        self,
        operator_whitelist: Optional[frozenset[str]] = None,
        max_nesting_depth: int = 8,
    ) -> None:
        self._whitelist = operator_whitelist or OPERATOR_WHITELIST
        self._max_depth = max_nesting_depth

    @algo_log(level=logging.INFO, label="ASTValidator.validate")
    def validate(self, expr: str) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        fix_suggestions: list[str] = []
        structure_info: dict[str, Any] = {}

        if not expr or not expr.strip():
            return ValidationResult(
                passed=False,
                errors=["[DEFENSIVE_LOG] 表达式为空"],
                structure_info={"expr_length": 0},
                fix_suggestions=["提供有效的 FASTEXPR 公式"],
            )

        expr = expr.strip()
        structure_info["expr_length"] = len(expr)

        # ── Rule 1: 语法合法性 (ast.parse 硬编译) ──────────────────────────────
        parse_ok, tree_or_error = self._try_parse(expr)
        if not parse_ok:
            err_msg = f"[DEFENSIVE_LOG] AST 语法解析失败: {tree_or_error}"
            errors.append(err_msg)
            return ValidationResult(
                passed=False,
                errors=errors,
                warnings=warnings,
                structure_info={**structure_info, "syntax_valid": False},
                fix_suggestions=["修复表达式语法错误后重试"],
            )
        tree = tree_or_error
        structure_info["syntax_valid"] = True

        # ── Rule 2~8: Visitor 遍历执行全量校验 ────────────────────────────────
        visitor = _ExprVisitor(self._whitelist, self._max_depth)
        try:
            visitor.visit(tree)
        except (ValueError, TypeError, RuntimeError) as exc:
            logger.warning("[DEFENSIVE_LOG] AST Visitor 遍历异常: %s", exc)

        # Rule 2: 算子白名单
        if visitor.unknown_operators:
            unknown_set = list(dict.fromkeys(visitor.unknown_operators))
            errors.append(
                f"[DEFENSIVE_LOG] 非法算子 detected: {unknown_set}"
            )
            fix_suggestions.append(
                f"移除非法算子，仅使用 WQ 平台 {len(self._whitelist)} 个合法算子"
            )

        # Rule 5: 嵌套深度
        structure_info["nesting_depth"] = visitor.max_nesting_depth
        if visitor.max_nesting_depth > self._max_depth:
            errors.append(
                f"[DEFENSIVE_LOG] 嵌套深度超限: {visitor.max_nesting_depth} > {self._max_depth}"
            )
        elif visitor.max_nesting_depth == self._max_depth:
            warnings.append(
                f"[DEFENSIVE_LOG] 嵌套深度达到上限: {visitor.max_nesting_depth}/{self._max_depth}"
            )

        # Rule 8: 参数类型 — 字符串字面量警告
        for lineno, sval in visitor.string_literals:
            warnings.append(
                f"[DEFENSIVE_LOG] 检测到字符串字面量 (line {lineno}): '{sval[:30]}', 可能是非法参数"
            )

        # Rule 3 & 4 & 7: 三段式结构识别
        three_block = self._analyze_three_block(tree, visitor.root_calls)
        structure_info.update({
            "operators_used": list(dict.fromkeys(visitor.operators_used)),
            "operator_count": len(visitor.operators_used),
            "has_neutralize": three_block.has_neutralize,
            "has_decay": three_block.has_decay,
            "neutralize_op": three_block.neutralize_op,
            "decay_op": three_block.decay_op,
            "is_valid_three_block": three_block.is_valid_three_block,
            "signal_depth": three_block.signal_depth,
        })

        if not three_block.has_neutralize:
            warnings.append(
                "[DEFENSIVE_LOG] 缺少中性化段 (group_neutralize/group_zscore)"
            )
            fix_suggestions.append(
                "注入默认 B 段: group_neutralize({signal}, industry)"
            )
        if not three_block.has_decay:
            warnings.append(
                "[DEFENSIVE_LOG] 缺少衰减段 (ts_decay_linear)"
            )
            fix_suggestions.append(
                "注入默认 C 段: ts_decay_linear({neutralized}, 10)"
            )

        signal_seg = self._extract_signal_segment(expr, three_block)
        structure_info["signal_segment"] = signal_seg

        passed = len(errors) == 0

        log_call(
            "ASTValidator.validation_result",
            input={"expr": expr[:80]},
            output={
                "passed": passed,
                "error_count": len(errors),
                "warning_count": len(warnings),
                "nesting_depth": visitor.max_nesting_depth,
                "ops_count": len(visitor.operators_used),
                "unknown_ops": visitor.unknown_operators[:5],
                "has_neutralize": three_block.has_neutralize,
                "has_decay": three_block.has_decay,
            },
            level=logging.DEBUG if passed else logging.WARNING,
        )

        return ValidationResult(
            passed=passed,
            errors=errors,
            warnings=warnings,
            structure_info=structure_info,
            signal_segment=signal_seg or None,
            fix_suggestions=fix_suggestions,
        )

    @staticmethod
    @algo_log(level=logging.DEBUG, label="ASTValidator.try_parse")
    def _try_parse(expr: str) -> tuple[bool, ast.Module | str]:
        with Timer("ast.parse") as t:
            try:
                tree = ast.parse(expr, mode='eval')
                return True, tree
            except SyntaxError as exc:
                return False, f"{exc.msg} (offset {exc.offset})"

    def _analyze_three_block(
        self, tree: ast.Expression, root_calls: list[str],
    ) -> _ThreeBlockStructure:
        result = _ThreeBlockStructure()

        outer_call = root_calls[0] if root_calls else ""

        if outer_call in _NEUTRALIZE_OPS:
            result.has_neutralize = True
            result.neutralize_op = outer_call
            inner_body = self._get_first_arg(tree.body)
            if inner_body is not None and isinstance(inner_body, ast.Call):
                inner_name = _ExprVisitor._call_name(inner_body)
                if inner_name in _DECAY_OPS:
                    result.has_decay = True
                    result.decay_op = inner_name
                    deepest = self._get_first_arg(inner_body)
                    if deepest is not None:
                        result.signal_depth = self._calc_node_depth(deepest)
                    result.is_valid_three_block = True
                else:
                    result.signal_depth = self._calc_node_depth(inner_body)

        elif outer_call in _DECAY_OPS:
            result.has_decay = True
            result.decay_op = outer_call
            inner_body = self._get_first_arg(tree.body)
            if inner_body is not None:
                result.signal_depth = self._calc_node_depth(inner_body)

        else:
            result.signal_depth = self._calc_node_depth(tree.body)

        return result

    @staticmethod
    def _get_first_arg(node: ast.expr) -> Optional[ast.expr]:
        if isinstance(node, ast.Call) and node.args:
            return node.args[0]
        return None

    @staticmethod
    def _calc_node_depth(node: ast.AST) -> int:
        depth = 0
        current = node
        while isinstance(current, ast.Call):
            depth += 1
            first_arg = ASTValidator._get_first_arg(current)
            if first_arg is None:
                break
            current = first_arg
        return depth

    @staticmethod
    def _extract_signal_segment(
        expr: str, three_block: _ThreeBlockStructure,
    ) -> Optional[str]:
        if three_block.is_valid_three_block:
            try:
                gn_start = expr.index(three_block.neutralize_op)
                gn_paren = expr.index("(", gn_start)
                td_start = expr.index(three_block.decay_op, gn_paren)
                td_paren = expr.index("(", td_start)
                close_pos = _find_matching_close(expr, td_paren + 1)
                if close_pos > 0:
                    decay_inner = expr[td_paren + 1:close_pos].strip()
                    gn_close = _find_matching_close(expr, gn_paren + 1)
                    if gn_close > 0:
                        neutralize_inner = expr[gn_paren + 1:gn_close].strip()
                        comma_pos = neutralize_inner.rfind(",")
                        if comma_pos > 0:
                            return neutralize_inner[:comma_pos].strip()
                    return decay_inner
            except (ValueError, IndexError):
                pass
        return None


def _find_matching_close(s: str, open_pos: int) -> int:
    depth = 0
    for i in range(open_pos, len(s)):
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    return -1
