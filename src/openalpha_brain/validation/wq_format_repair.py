"""WQ BRAIN Format Error Auto-Repair Mechanism

Automatically diagnoses and repairs common WQ BRAIN expression format errors:
- Missing lookback parameters for time-series operators
- Unknown/invalid field names
- Unexpected characters in expressions
- Parse errors (unbalanced parentheses)
- Invalid string literals
- Decay parameter issues
- Missing neutralization parameters

Integration Point: Called by feedback_orchestrator.py REPAIR_AND_RETRY decision.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RepairDiagnosis:
    """Diagnosis result for WQ format error"""

    error_type: str
    error_message: str
    affected_operators: list[str] = field(default_factory=list)
    repair_strategy: str = ""
    confidence: float = 0.0
    original_expression: str = ""


class WQFormatRepair:
    """WQ BRAIN Expression Format Error Auto-Repairer

    Automatically identifies and fixes common format errors in WQ BRAIN expressions.
    All repair operations are pure functions (input expression → output repaired expression).
    Original expression is preserved; returns a repaired copy.

    Usage:
        repairer = WQFormatRepair()
        diagnosis = repairer.diagnose(error_msg="Required attribute 'lookback' must have a value",
                                     expression="ts_mean(close)")
        if diagnosis.confidence > 0.7:
            repaired_expr = repairer.repair(diagnosis)
            is_valid, warnings = repairer.validate_repaired(repaired_expr)
    """

    TIME_SERIES_OPERATORS_NEEDING_LOOKBACK = {
        "ts_mean",
        "ts_std_dev",
        "ts_zscore",
        "ts_rank",
        "ts_decay_linear",
        "ts_delay",
        "ts_sum",
        "ts_corr",
        "ts_delta",
        "ts_arg_max",
        "ts_arg_min",
        "ts_av_diff",
        "ts_quantile",
        "ts_regression",
        "ts_std",
        "ts_max",
        "ts_min",
        "ts_shift",
        "ts_product",
        "ts_skewness",
        "ts_kurtosis",
        "ts_covariance",
        "ts_cov",
        "ts_ir",
        "ts_backfill",
        "decay_linear",
    }

    CROSS_SECTIONAL_OPERATORS = {
        "rank",
        "zscore",
        "scale",
        "tanh",
        "sigmoid",
        "sign_power",
        "log",
        "abs",
        "max",
        "min",
        "sign",
        "sqrt",
        "exp",
        "clip",
    }

    GROUP_OPERATORS = {"group_neutralize", "group_rank", "group_zscore", "group_mean", "group_vector_neut"}

    CONDITIONAL_OPERATORS = {"trade_when", "where"}

    ALL_VALID_OPERATORS = (
        TIME_SERIES_OPERATORS_NEEDING_LOOKBACK
        | CROSS_SECTIONAL_OPERATORS
        | GROUP_OPERATORS
        | CONDITIONAL_OPERATORS
        | {
            "power",
            "indneutralize",
            "normalize",
            "quantile",
            "bucket",
            "pasteurize",
            "vec_avg",
            "vec_sum",
            "vec_norm",
            "vec_choose",
        }
    )

    SAFE_FIELD_SUBSTITUTES = {
        "field_3921": "close",
        "price": "close",
        "vol": "volume",
        "ret": "returns",
        "mkt_cap": "cap",
        "liq": "volume",
        "amt": "vwap",
    }

    ERROR_PATTERNS = {
        "lookback": {
            "pattern": r"(?i)(lookback|window|must have a value).*?(\w+)",
            "root_cause": "Time-series operator missing required lookback/window parameter",
            "strategy": "Insert default lookback value (usually 20) after operator",
            "confidence": 0.95,
        },
        "unknown variable": {
            "pattern": r'(?i)(unknown variable|undefined|not found).*?[\'"](\w+)[\'"]',
            "root_cause": "Reference to non-existent field name",
            "strategy": "Replace with nearest valid field substitute",
            "confidence": 0.85,
        },
        "unexpected character": {
            "pattern": r"(?i)(unexpected character|illegal character|invalid token)",
            "root_cause": "Expression contains illegal characters",
            "strategy": "Remove non-alphanumeric characters (except ()_,./-*+)",
            "confidence": 0.80,
        },
        "parse error": {
            "pattern": r"(?i)(parse error|syntax error|unexpected token)",
            "root_cause": "Unbalanced parentheses or syntax error",
            "strategy": "Attempt to balance parentheses or fix syntax",
            "confidence": 0.75,
        },
        "string literal": {
            "pattern": r"(?i)(string literal|expected number|type error)",
            "root_cause": "String parameter used where integer expected",
            "strategy": "Replace string with appropriate integer value",
            "confidence": 0.85,
        },
        "decay": {
            "pattern": r"(?i)(decay.*invalid|decay.*error|decay.*parameter)",
            "root_cause": "Invalid decay parameter value",
            "strategy": "Set decay to standard value (5)",
            "confidence": 0.90,
        },
        "neutralization": {
            "pattern": r"(?i)(neutraliz.*missing|neutraliz.*required|group.*argument)",
            "root_cause": "Missing neutralization group parameter",
            "strategy": "Wrap with group_neutralize(expr, industry)",
            "confidence": 0.88,
        },
    }

    DEFAULT_LOOKBACK = 20
    DEFAULT_DECAY_WINDOW = 5

    def __init__(self, default_lookback: int = 20):
        """Initialize WQ Format Repairer

        Args:
            default_lookback: Default window size for time-series operators (default: 20)
        """
        self.default_lookback = default_lookback

    def diagnose(self, error_message: str, expression: str) -> RepairDiagnosis:
        """Analyze WQ error message and return diagnosis result

        Args:
            error_message: Error message returned by WQ BRAIN
            expression: The original expression that caused the error

        Returns:
            RepairDiagnosis with error type, affected operators, and repair strategy
        """
        if not error_message or not expression:
            return RepairDiagnosis(
                error_type="empty_input",
                error_message=error_message or "Empty error message",
                original_expression=expression or "",
                confidence=0.0,
            )

        logger.info(
            "[DEFENSIVE_LOG] WQFormatRepair.diagnose: analyzing error='%s' expr='%s'",
            error_message[:100],
            expression[:60],
        )

        for error_type, pattern_info in self.ERROR_PATTERNS.items():
            match = re.search(pattern_info["pattern"], error_message)
            if match:
                affected_ops = self._extract_affected_operators(expression, error_type)

                diagnosis = RepairDiagnosis(
                    error_type=error_type,
                    error_message=error_message,
                    affected_operators=affected_ops,
                    repair_strategy=pattern_info["strategy"],
                    confidence=pattern_info["confidence"],
                    original_expression=expression,
                )

                logger.info(
                    "[DEFENSIVE_LOG] WQFormatRepair.diagnose: diagnosed as '%s' confidence=%.2f affected_ops=%s",
                    error_type,
                    diagnosis.confidence,
                    affected_ops,
                )

                return diagnosis

        logger.warning("[DEFENSIVE_LOG] WQFormatRepair.diagnose: unrecognized error pattern: %s", error_message[:100])

        return RepairDiagnosis(
            error_type="unknown",
            error_message=error_message,
            repair_strategy="No automatic repair available",
            confidence=0.0,
            original_expression=expression,
        )

    def repair(self, diagnosis: RepairDiagnosis) -> str:
        """Repair expression based on diagnosis result

        Args:
            diagnosis: Diagnosis result from diagnose() method

        Returns:
            Repaired expression string (original unchanged if no repair possible)
        """
        if not diagnosis or diagnosis.confidence < 0.5:
            logger.warning(
                "[DEFENSIVE_LOG] WQFormatRepair.repair: low confidence (%.2f), skipping",
                diagnosis.confidence if diagnosis else 0,
            )
            return diagnosis.original_expression if diagnosis else ""

        expression = diagnosis.original_expression
        error_type = diagnosis.error_type

        logger.info(
            "[DEFENSIVE_LOG] WQFormatRepair.repair: applying strategy for '%s' on expr='%s'",
            error_type,
            expression[:60],
        )

        repair_methods = {
            "lookback": self._repair_missing_lookback,
            "unknown variable": self._repair_unknown_variable,
            "unexpected character": self._repair_unexpected_character,
            "parse error": self._repair_parse_error,
            "string literal": self._repair_string_literal,
            "decay": self._repair_decay_related,
            "neutralization": self._repair_neutralization_missing,
        }

        repair_fn = repair_methods.get(error_type)
        if repair_fn:
            try:
                repaired = repair_fn(expression, diagnosis.error_message)
                logger.info(
                    "[DEFENSIVE_LOG] WQFormatRepair.repair: repaired successfully: '%s' → '%s'",
                    expression[:50],
                    repaired[:50],
                )
                return repaired
            except (ValueError, TypeError, RuntimeError) as exc:
                logger.error("[DEFENSIVE_LOG] WQFormatRepair.repair: repair failed for '%s': %s", error_type, exc)
                return expression

        logger.warning("[DEFENSIVE_LOG] WQFormatRepair.repair: no repair method for '%s'", error_type)
        return expression

    def validate_repaired(self, expression: str) -> tuple[bool, list[str]]:
        """Validate repaired expression for basic syntax correctness

        Checks:
        - Parentheses balance
        - Operator whitelist compliance
        - Parameter type validation (lookback must be positive integer)
        - group_neutralize must have exactly 2 arguments

        Args:
            expression: Expression to validate

        Returns:
            Tuple of (is_valid: bool, warnings: list[str])
        """
        warnings = []

        if not expression or not expression.strip():
            warnings.append("Expression is empty")
            return False, warnings

        expr = expression.strip()

        open_count = expr.count("(")
        close_count = expr.count(")")
        if open_count != close_count:
            warnings.append(f"Unbalanced parentheses: {open_count} opening vs {close_count} closing")
            return False, warnings

        depth = 0
        max_depth = 0
        for ch in expr:
            if ch == "(":
                depth += 1
                max_depth = max(max_depth, depth)
            elif ch == ")":
                depth -= 1

        if max_depth > 8:
            warnings.append(f"Nesting depth too high: {max_depth} > 8")

        tokens = re.findall(r"\b([a-zA-Z_]\w*)\s*\(", expr)
        unknown_ops = [t for t in tokens if t not in self.ALL_VALID_OPERATORS]
        if unknown_ops:
            warnings.append(f"Unknown operators detected: {unknown_ops[:3]}")

        ts_op_pattern = (
            r"(?:"
            + "|".join(re.escape(op) for op in self.TIME_SERIES_OPERATORS_NEEDING_LOOKBACK)
            + r")\s*\([^,]+,\s*([^\)]+)\)"
        )

        for match in re.finditer(ts_op_pattern, expr):
            param = match.group(1).strip()
            try:
                val = int(param)
                if val <= 0:
                    warnings.append(f"Non-positive lookback parameter: {val}")
                elif val > 500:
                    warnings.append(f"Extremely large lookback parameter: {val}")
            except ValueError:
                warnings.append(f"Non-integer lookback parameter: {param}")

        neut_match = re.search(r"group_neutralize\s*\(", expr)
        if neut_match:
            start = neut_match.start()
            depth = 0
            arg_count = 0
            has_comma = False
            for i in range(start, len(expr)):
                ch = expr[i]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        break
                elif ch == "," and depth == 1:
                    has_comma = True
                    arg_count += 1

            if not has_comma or arg_count < 1:
                warnings.append("group_neutralize missing second argument (group column)")

        is_valid = len(warnings) == 0

        if is_valid:
            logger.debug("[DEFENSIVE_LOG] WQFormatRepair.validate_repaired: PASSED for expr='%s'", expr[:60])
        else:
            logger.warning("[DEFENSIVE_LOG] WQFormatRepair.validate_repaired: FAILED warnings=%s", warnings)

        return is_valid, warnings

    def _extract_affected_operators(self, expression: str, error_type: str) -> list[str]:
        """Extract operators affected by the error from expression"""
        operators_found = re.findall(r"\b([a-zA-Z_]\w*)\s*\(", expression)

        if error_type == "lookback":
            return [op for op in operators_found if op in self.TIME_SERIES_OPERATORS_NEEDING_LOOKBACK]
        elif error_type == "unknown variable":
            return list(set(operators_found))
        elif error_type == "neutralization":
            return [op for op in operators_found if op in self.GROUP_OPERATORS]

        return list(set(operators_found))[:5]

    def _repair_missing_lookback(self, expression: str, error_msg: str) -> str:
        """Repair missing lookback parameters in time-series operators

        Find patterns like ts_xxx(field) where field is not a number,
        and insert default lookback value.
        """
        pattern = (
            r"\b("
            + "|".join(re.escape(op) for op in sorted(self.TIME_SERIES_OPERATORS_NEEDING_LOOKBACK))
            + r")\s*\(([^)]+)\)"
        )

        def replace_missing_lookback(match):
            op_name = match.group(1)
            args_str = match.group(2).strip()

            parts = self._split_top_level_args(args_str)

            if len(parts) == 1:
                single_arg = parts[0].strip()
                try:
                    int(single_arg)
                    return match.group(0)
                except ValueError:
                    return f"{op_name}({single_arg}, {self.default_lookback})"

            return match.group(0)

        repaired = re.sub(pattern, replace_missing_lookback, expression)

        logger.info("[DEFENSIVE_LOG] _repair_missing_lookback: applied default_lookback=%d", self.default_lookback)

        return repaired

    def _repair_unknown_variable(self, expression: str, error_msg: str) -> str:
        """Repair unknown variable references by substituting valid field names"""
        var_match = re.search(r'[\'"](\w+)[\'"]', error_msg)
        unknown_var = var_match.group(1) if var_match else None

        repaired = expression

        if unknown_var and unknown_var.lower() in self.SAFE_FIELD_SUBSTITUTES:
            substitute = self.SAFE_FIELD_SUBSTITUTES[unknown_var.lower()]
            pattern = r"\b" + re.escape(unknown_var) + r"\b"
            repaired = re.sub(pattern, substitute, expression, flags=re.IGNORECASE)

            logger.info("[DEFENSIVE_LOG] _repair_unknown_variable: substituted '%s' → '%s'", unknown_var, substitute)

            return repaired

        generic_bad_fields = re.findall(r"\b(field_\d+|var_\d+|col_\d+)\b", expression, re.IGNORECASE)
        for bad_field in generic_bad_fields:
            if bad_field.lower() not in self.SAFE_FIELD_SUBSTITUTES:
                self.SAFE_FIELD_SUBSTITUTES[bad_field.lower()] = "close"

            substitute = self.SAFE_FIELD_SUBSTITUTES[bad_field.lower()]
            pattern = r"\b" + re.escape(bad_field) + r"\b"
            repaired = re.sub(pattern, substitute, repaired, flags=re.IGNORECASE)

            logger.info(
                "[DEFENSIVE_LOG] _repair_unknown_variable: substituted generic '%s' → '%s'", bad_field, substitute
            )

        return repaired

    def _repair_unexpected_character(self, expression: str, error_msg: str) -> str:
        """Remove unexpected/illegal characters from expression"""
        cleaned = re.sub(r"[^a-zA-Z0-9_(),.\-+*/\s]", "", expression)

        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        logger.info("[DEFENSIVE_LOG] _repair_unexpected_character: removed illegal chars")

        return cleaned

    def _repair_parse_error(self, expression: str, error_msg: str) -> str:
        """Attempt to fix parse errors (primarily unbalanced parentheses)"""
        repaired = expression.strip()

        open_count = repaired.count("(")
        close_count = repaired.count(")")

        if open_count > close_count:
            repaired += ")" * (open_count - close_count)
            logger.info("[DEFENSIVE_LOG] _repair_parse_error: added %d closing parentheses", open_count - close_count)
        elif close_count > open_count:
            repaired = "(" * (close_count - open_count) + repaired
            logger.info("[DEFENSIVE_LOG] _repair_parse_error: added %d opening parentheses", close_count - open_count)

        trailing_comma_match = re.search(r"(,+)\s*$", repaired)
        if trailing_comma_match:
            repaired = repaired[: trailing_comma_match.start()]
            logger.info("[DEFENSIVE_LOG] _repair_parse_error: removed trailing commas")

        double_operator_match = re.search(r"\)\s*\(", repaired)
        if (
            double_operator_match
            and "*" not in repaired[double_operator_match.start() : double_operator_match.end() + 1]
        ):
            pos = double_operator_match.end() - 1
            repaired = repaired[:pos] + " * " + repaired[pos:]
            logger.info("[DEFENSIVE_LOG] _repair_parse_error: inserted * between function calls")

        return repaired

    def _repair_string_literal(self, expression: str, error_msg: str) -> str:
        """Replace string parameters with integer values"""
        string_params = re.findall(r'([a-zA-Z_]\w*)\s*\(\s*([^)]*?)([\'"][^\'"]*[\'"])([^)]*?)\)', expression)

        repaired = expression
        for match in string_params:
            full_match = match[0]
            before_str = match[1]
            str_literal = match[2]
            after_str = match[3]

            str_value = str_literal.strip("\"'")

            if str_value.lower() in ("industry", "sector", "subindustry"):
                continue

            try:
                int_val = int(str_value)
                replacement = f"{before_str}{int_val}{after_str}"
            except ValueError:
                if str_value.lower() in ("true", "yes"):
                    replacement = f"{before_str}1{after_str}"
                elif str_value.lower() in ("false", "no"):
                    replacement = f"{before_str}0{after_str}"
                else:
                    replacement = f"{before_str}1{after_str}"

            repaired = repaired.replace(full_match, replacement, 1)

            logger.info("[DEFENSIVE_LOG] _repair_string_literal: replaced '%s' with numeric", str_literal)

        return repaired

    def _repair_decay_related(self, expression: str, error_msg: str) -> str:
        """Fix invalid decay parameter values"""
        decay_pattern = r"(ts_decay_linear|decay_linear)\s*\([^,]+,\s*(\d+)"

        def replace_decay_window(match):
            op = match.group(1)
            current_window = int(match.group(2))

            if current_window < 2 or current_window > 30:
                new_window = self.DEFAULT_DECAY_WINDOW
                logger.info(
                    "[DEFENSIVE_LOG] _repair_decay_related: changed decay window %d → %d", current_window, new_window
                )
                return f"{op}(X_PLACEHOLDER, {new_window})"

            return match.group(0)

        repaired = re.sub(decay_pattern, replace_decay_window, expression)
        repaired = repaired.replace("X_PLACEHOLDER", "X")

        if "ts_decay_linear" not in expression and "decay_linear" not in expression:
            if "group_neutralize" in expression:
                inner_match = re.search(r"group_neutralize\s*\((.+?)\s*,\s*\w+\s*\)", expression, re.DOTALL)
                if inner_match:
                    inner_expr = inner_match.group(1)
                    wrapped = f"ts_decay_linear({inner_expr}, {self.DEFAULT_DECAY_WINDOW})"
                    repaired = expression.replace(inner_expr, wrapped, 1)

                    logger.info(
                        "[DEFENSIVE_LOG] _repair_decay_related: wrapped with ts_decay_linear(window=%d)",
                        self.DEFAULT_DECAY_WINDOW,
                    )

        return repaired

    def _repair_neutralization_missing(self, expression: str, error_msg: str) -> str:
        """Add missing neutralization wrapper or fix group_neutralize parameters"""
        if "group_neutralize" not in expression:
            repaired = f"group_neutralize({expression}, industry)"

            logger.info("[DEFENSIVE_LOG] _repair_neutralization_missing: wrapped with group_neutralize")

            return repaired

        neut_pattern = r"group_neutralize\s*\(\s*([^,]+?)\s*\)"
        match = re.search(neut_pattern, expression)

        if match:
            inner_expr = match.group(1).strip()
            repaired = expression.replace(match.group(0), f"group_neutralize({inner_expr}, industry)", 1)

            logger.info("[DEFENSIVE_LOG] _repair_neutralization_missing: added 'industry' parameter")

            return repaired

        return expression

    @staticmethod
    def _split_top_level_args(args_str: str) -> list[str]:
        """Split argument string by commas at top level (respecting nested parentheses)"""
        parts = []
        depth = 0
        current = []

        for ch in args_str:
            if ch == "(":
                depth += 1
                current.append(ch)
            elif ch == ")":
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(ch)

        if current:
            parts.append("".join(current))

        return parts


def create_wq_format_repairer(default_lookback: int = 20) -> WQFormatRepair:
    """Factory function to create WQFormatRepair instance

    Args:
        default_lookback: Default window size for time-series operators

    Returns:
        Configured WQFormatRepair instance
    """
    return WQFormatRepair(default_lookback=default_lookback)


def auto_repair_wq_expression(
    error_message: str, expression: str, default_lookback: int = 20
) -> tuple[str, RepairDiagnosis, bool]:
    """Convenience function for one-shot auto-repair workflow

    Args:
        error_message: Error message from WQ BRAIN
        expression: Original expression that failed
        default_lookback: Default lookback window size

    Returns:
        Tuple of (repaired_expression, diagnosis, was_repaired)
    """
    repairer = create_wq_format_repairer(default_lookback)

    diagnosis = repairer.diagnose(error_message, expression)

    if diagnosis.confidence >= 0.7:
        repaired = repairer.repair(diagnosis)
        is_valid, warnings = repairer.validate_repaired(repaired)

        if is_valid:
            logger.info(
                "[DEFENSIVE_LOG] auto_repair_wq_expression: SUCCESS expr='%s' → '%s'", expression[:50], repaired[:50]
            )
            return repaired, diagnosis, True
        else:
            logger.warning("[DEFENSIVE_LOG] auto_repair_wq_expression: repaired but validation failed: %s", warnings)
            return repaired, diagnosis, False

    logger.info(
        "[DEFENSIVE_LOG] auto_repair_wq_expression: LOW CONFIDENCE (%.2f), no repair attempted", diagnosis.confidence
    )
    return expression, diagnosis, False
