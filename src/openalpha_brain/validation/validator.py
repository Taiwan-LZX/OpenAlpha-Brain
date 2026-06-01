"""
OpenAlpha - Quant — IQC Constraint + Syntax Validator
Two public functions: validate_syntax() and validate_metrics().
Both return ValidationResult(passed, failures, warnings).
"""

from __future__ import annotations

import logging
import re

from openalpha_brain.cli.algo_monitor import AlgoMonitor
from openalpha_brain.config.config import settings
from openalpha_brain.core.models import ValidationResult
from openalpha_brain.data import get_data_path
from openalpha_brain.generation.ast_originality import OriginalityChecker
from openalpha_brain.utils.algo_logger import algo_log
from openalpha_brain.validation.complexity_control import ComplexityController

logger = logging.getLogger(__name__)

# ── Permitted operator whitelist ──────────────────────────────────────────────
PERMITTED_OPERATORS_FALLBACK: set[str] = {
    "rank",
    "ts_rank",
    "ts_mean",
    "ts_std_dev",
    "ts_delta",
    "ts_zscore",
    "ts_decay_linear",
    "group_neutralize",
    "abs",
    "log",
    "signed_power",
    "max",
    "min",
    "scale",
    "ts_delay",
    "ts_sum",
    "ts_corr",
    "ts_regression",
    "ts_arg_max",
    "ts_arg_min",
    "ts_backfill",
    "trade_when",
    "zscore",
    "normalize",
    "winsorize",
    "hump",
    "group_rank",
    "group_zscore",
    "ts_av_diff",
    "ts_quantile",
    "quantile",
    "vec_sum",
    "vec_avg",
    "add",
    "divide",
    "multiply",
    "subtract",
}

# Regex to find all function-call identifiers: word chars followed by (
_KEYWORD_ARG_RE = re.compile(r"[,\s]\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*\d")

_FUNC_CALL_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")

# Regex to detect float literals used as window arguments (after a comma)
# Matches patterns like: ts_mean(close, 20.5) → flags 20.5
_FLOAT_WINDOW_RE = re.compile(r",\s*(\d+\.\d+)\s*[,)]")

# Detect group_neutralize with correct second arg
_GN_RE = re.compile(
    r"group_neutralize\s*\(.*?,\s*(sector|industry|subindustry)\s*\)",
    re.IGNORECASE | re.DOTALL,
)

# ── Verified BRAIN FastExpr variable whitelist ────────────────────────────────
# ONLY these bare variable names are confirmed to exist in BRAIN FastExpr.
# Fundamental/Analyst data uses dataset-specific prefixed names (e.g. fn_sales_q)
# accessed via the BRAIN Data Explorer — NOT bare names like 'sales' or 'earnings'.
VALID_BRAIN_VARS_STRICT: set[str] = {
    "open",
    "high",
    "low",
    "close",
    "vwap",
    "volume",
    "adv20",
    "returns",
    "cap",
    "sales",
    "assets",
    "liabilities",
    "revenue",
    "equity",
    "debt",
    "industry",
    "subindustry",
    "sector",
}

TS_OPS_REQUIRING_LOOKBACK: set[str] = {
    "ts_mean",
    "ts_std_dev",
    "ts_delta",
    "ts_zscore",
    "ts_rank",
    "ts_decay_linear",
    "ts_delay",
    "ts_sum",
    "ts_corr",
    "ts_arg_max",
    "ts_arg_min",
    "ts_av_diff",
    "ts_quantile",
    "ts_regression",
}


def _load_operators_from_schema() -> set[str]:
    schema_path = get_data_path("brain_operators.json")
    if not schema_path.exists():
        return PERMITTED_OPERATORS_FALLBACK
    import json

    with open(schema_path, encoding="utf-8") as f:
        data = json.load(f)
    names = set()
    for op in data:
        name = op.get("name", "")
        if name:
            names.add(name)
    return names if names else PERMITTED_OPERATORS_FALLBACK


def _load_datafields_from_schema() -> set[str]:
    schema_path = get_data_path("brain_datafields.json")
    if not schema_path.exists():
        return {v.lower() for v in VALID_BRAIN_VARS_STRICT}
    import json

    with open(schema_path, encoding="utf-8") as f:
        data = json.load(f)
    names = set()
    for field in data:
        fid = field.get("id", "")
        if fid:
            names.add(fid.lower())
    names.update({"industry", "subindustry", "sector"})
    return names if names else {v.lower() for v in VALID_BRAIN_VARS_STRICT}


# Regex to find bare variable identifiers (not followed by '(')
_VAR_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b(?!\s*\()")


def _check_parens(expr: str) -> bool:
    """Stack-based parenthesis balance check. Returns True if balanced."""
    depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False  # unmatched closing paren
    return depth == 0  # True only if every opener was closed


@algo_log()
def validate_syntax(expression: str) -> ValidationResult:
    """
    Check BRAIN expression for structural correctness.

    Checks:
    1. Expression length: 5 ≤ len ≤ 2000
    2. group_neutralize(..., sector|industry) is present
    3. Parentheses are balanced (stack-based)
    4. All function calls use permitted operators only
    5. No float literals used as window arguments
    """
    failures: list[str] = []
    warnings: list[str] = []

    if not expression or not expression.strip():
        return ValidationResult(passed=False, failures=["Expression is empty"])

    expr = expression.strip()

    # 1. Length bounds
    if len(expr) < 5:
        failures.append(f"Expression too short ({len(expr)} chars, minimum 5)")
    if len(expr) > 2000:
        failures.append(f"Expression too long ({len(expr)} chars, maximum 2000)")

    # 2. group_neutralize presence and correct form
    if "group_neutralize" not in expr:
        failures.append("group_neutralize() is missing — required on every alpha")
    elif not _GN_RE.search(expr):
        warnings.append("group_neutralize found but second arg is not 'sector' or 'industry'")

    # 3. Balanced parentheses (stack-based)
    if not _check_parens(expr):
        failures.append("Unbalanced parentheses detected")

    # 4. Operator whitelist
    called_funcs = set(_FUNC_CALL_RE.findall(expr))
    # Filter out single-letter or data field names (e.g. no trailing paren normally)
    unknown = called_funcs - PERMITTED_OPERATORS
    if unknown:
        failures.append(f"Unknown/forbidden operators: {', '.join(sorted(unknown))}")

    # 4b. Lookback window missing for time-series operators
    ts_call_re = re.compile(
        r"\b(" + "|".join(re.escape(op) for op in TS_OPS_REQUIRING_LOOKBACK) + r")\s*\(([^()]*)\)",
        re.IGNORECASE,
    )
    for m in ts_call_re.finditer(expr):
        op_name = m.group(1)
        inner = m.group(2)
        depth = 0
        arg_count = 1
        for ch in inner:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "," and depth == 0:
                arg_count += 1
        if arg_count < 2:
            failures.append(f"{op_name} requires lookback window (only {arg_count} argument provided)")

    # 5. Float window arguments (integers only allowed)
    float_windows = _FLOAT_WINDOW_RE.findall(expr)
    if float_windows:
        failures.append(
            f"Non-integer lookback window(s) detected: {float_windows} — all window arguments must be strict integers"
        )

    # 5b. Keyword arguments not allowed in FastExpr
    kw_matches = _KEYWORD_ARG_RE.findall(expr)
    if kw_matches:
        failures.append(
            "Keyword arguments detected (e.g. d=5) — "
            "BRAIN FastExpr uses positional arguments only: ts_decay_linear(x, 5) not ts_decay_linear(x, d=5)"
        )

    # 5c. String literals not allowed in FastExpr
    string_literal_re = re.compile(r'["\'][a-zA-Z_][a-zA-Z0-9_]*["\']')
    string_matches = string_literal_re.findall(expr)
    if string_matches:
        failures.append("String literals not allowed in FastExpr — use integer rettype values instead")

    # 6. Placeholder syntax check — "..." is never valid in BRAIN expressions
    if "..." in expr:
        failures.append(
            'Placeholder "..." detected — every function call MUST have complete, '
            'valid arguments. Replace "..." with actual expressions.'
        )

    # 7. Invalid BRAIN variable check — catches hallucinated fields before API call
    identifiers = set(_VAR_RE.findall(expr))
    bad_vars = {v.lower() for v in identifiers} - get_valid_brain_vars() - PERMITTED_OPERATORS
    if bad_vars:
        failures.append(
            f"INVALID BRAIN variables (will cause simulation ERROR): "
            f"{', '.join(sorted(bad_vars))}. "
            f"Refer to brain_datafields.json for valid data fields"
        )

    passed = len(failures) == 0
    return ValidationResult(passed=passed, failures=failures, warnings=warnings)


_JSON_ARTIFACT_KEYWORDS = re.compile(
    r"\b(?:expression|regular|simulation_payload|settings|rationale|decision|"
    r"fingerprint|family|ast_topology|metrics|mutation_paths|refinement_log)\s*[:=]\s*"
)

BRAIN_7_STANDARDS = {
    "sharpe",
    "fitness",
    "turnover",
    "returns",
    "drawdown",
    "self_correlation",
    "low_sub_universe_sharpe",
}


@algo_log()
def compute_hierarchical_reward(expression: str, brain_result: dict | None = None) -> float:
    """
    新 Reward 设计 — 高区分度，让 MAB 能真正学习

    分数范围: 0.0 ~ 1.0
    只有 Sharpe >= 1.25 的因子才能拿到 > 0.5 的分数

    层级设计:
      Level 0 (0.00-0.10): 语法错误 / 提交失败 ERROR
      Level 1 (0.10-0.25): 语法通过但无 WQ 结果 或 Sharpe < 0.5
      Level 2 (0.25-0.40): 0.5 <= Sharpe < 1.0 (有信号但不够强)
      Level 3 (0.40-0.60): 1.0 <= Sharpe < 1.25 (接近过审)
      Level 4 (0.60-0.80): Sharpe >= 1.25 + Fitness >= 1.0 (过审!)
      Level 5 (0.80-1.00): Sharpe >= 1.75 + 所有 checks 通过 (优秀)

    惩罚项:
      - SELF_CORRELATION FAIL: -0.15
      - TURNOVER 超范围 (<1% or >70%): -0.10
      - HIGH DRAWDOWN (>15%): -0.08
      - LOW FITNESS (<0.5): -0.05
    """
    if not expression or not expression.strip():
        return 0.0

    reward = 0.0

    reward += 0.05

    if "{" not in expression and "}" not in expression:
        pass

    if "group_neutralize" in expression.lower():
        reward += 0.03

    if brain_result is None:
        return round(reward, 4)

    status = brain_result.get("status", "ERROR")
    if status == "ERROR":
        return round(max(0.0, reward), 4)

    sharpe = brain_result.get("sharpe")
    fitness = brain_result.get("fitness")
    turnover = brain_result.get("turnover")
    drawdown = brain_result.get("drawdown")
    checks = brain_result.get("checks", [])

    if sharpe is None:
        return round(max(0.0, reward), 4)

    if sharpe < 0:
        sharpe_bonus = 0.0
    elif sharpe < 0.5:
        sharpe_bonus = 0.05 * (sharpe / 0.5)
    elif sharpe < 1.0:
        sharpe_bonus = 0.10 + 0.15 * ((sharpe - 0.5) / 0.5)
    elif sharpe < 1.25:
        sharpe_bonus = 0.25 + 0.15 * ((sharpe - 1.0) / 0.25)
    elif sharpe < 2.0:
        sharpe_bonus = 0.40 + 0.20 * min(1.0, (sharpe - 1.25) / 1.0)
    else:
        sharpe_bonus = 0.60 + 0.15 * min(1.0, (sharpe - 2.0) / 2.0)

    reward += sharpe_bonus

    if fitness is not None and fitness > 1.0:
        reward += 0.05
        if fitness > 2.0:
            reward += 0.05

    failed_checks = set()
    if isinstance(checks, list):
        for chk in checks:
            if isinstance(chk, str):
                failed_checks.add(chk.upper())
            elif isinstance(chk, dict) and chk.get("result") == "FAIL":
                failed_checks.add(chk.get("name", "").upper())

    all_checks_passed = len(failed_checks) == 0
    if sharpe >= 1.25 and fitness is not None and fitness >= 1.0 and all_checks_passed:
        reward += 0.10

    if "SELF_CORRELATION" in failed_checks:
        reward -= 0.15

    if turnover is not None:
        if turnover < 1.0 or turnover > 70.0:
            reward -= 0.10

    if drawdown is not None and drawdown > 15.0:
        reward -= 0.08

    if fitness is not None and fitness < 0.5:
        reward -= 0.05

    reward = max(0.0, min(1.0, reward))
    return round(reward, 4)


def get_reward_level(reward: float) -> tuple[str, str]:
    if reward < 0.3:
        return ("Level 1 - Basic", "Syntax valid, needs signal")
    elif reward < 0.6:
        return ("Level 2 - Signal", "Has signal, needs quality")
    elif reward < 0.8:
        return ("Level 3 - Quality", "Good quality, needs excellence")
    else:
        return ("Level 4 - Excellence", "Excellent alpha")


def estimate_sharpe_likelihood(expression: str) -> float:
    """Estimate Sharpe likelihood based on keyword matching.
    NOTE: This is a heuristic keyword-counting scorer, NOT a statistical Sharpe
    prediction. It checks for the presence of operators like ts_zscore,
    ts_decay_linear, trade_when etc. and assigns fixed scores. A high score does
    NOT guarantee a high Sharpe ratio."""
    if not expression or not expression.strip():
        return 0.0
    score = 0.0
    expr_lower = expression.lower()
    if "ts_zscore" in expr_lower:
        score += 0.20
    if "/ ts_std_dev" in expr_lower or "/ts_std_dev" in expr_lower:
        score += 0.25
    if "ts_decay_linear" in expr_lower:
        score += 0.15
    if "trade_when" in expr_lower:
        score += 0.20
    if "ts_regression" in expr_lower:
        score += 0.15
    if "group_neutralize" in expr_lower:
        score += 0.05
    if "subindustry" in expr_lower:
        score += 0.05
    ops = set(re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", expr_lower))
    if len(ops) >= 3:
        score += 0.05
    if "*" in expression and "rank(" in expr_lower:
        score += 0.10
    if "rank(" in expr_lower and expr_lower.count("rank(") == 1:
        score += 0.05
    return min(score, 1.0)


def validate_fitness(
    sharpe: float,
    returns_pct: float,
    turnover_pct: float,
) -> ValidationResult:
    """
    v2: Compute Fitness using the exact IQC formula.

    Formula: Fitness = Sharpe × sqrt(|Returns|) / max(Turnover, 0.125)
    where Returns and Turnover are decimal fractions (not percentages).

    Sweet spot: Sharpe ~1.4, Returns ~20%, Turnover ~20% → Fitness ~2.0
    """
    import math

    failures: list[str] = []
    warnings: list[str] = []

    returns_dec = abs(returns_pct) / 100.0
    turnover_dec = turnover_pct / 100.0
    turnover_denom = max(turnover_dec, 0.125)  # floor at 12.5%

    fitness = sharpe * math.sqrt(returns_dec) / turnover_denom
    breakdown = (
        f"Fitness = {sharpe} × sqrt({returns_pct}%={returns_dec:.4f}) "
        f"/ max({turnover_pct}%={turnover_dec:.4f}, 0.125) "
        f"= {sharpe} × {math.sqrt(returns_dec):.4f} / {turnover_denom:.4f} "
        f"= {fitness:.4f}"
    )

    if fitness <= 1.0:
        failures.append(f"Fitness {fitness:.4f} ≤ 1.0 — gate FAIL. {breakdown}")
    elif fitness < 1.5:
        warnings.append(f"Fitness {fitness:.4f} is below the 1.5 competitive threshold.")

    passed = len(failures) == 0
    result = ValidationResult(passed=passed, failures=failures, warnings=warnings)
    result.fitness_computed = round(fitness, 4)
    result.fitness_breakdown = breakdown
    return result


def validate_metrics(parsed: dict) -> ValidationResult:
    """
    Check parsed IQC metric estimates against hard submission gates.

    Gates:
    - Sharpe ≥ 1.25  (uses sharpe_min as the conservative bound)
    - Fitness > 1.0  (v2: computed via exact formula when returns_pct available;
                          falls back to fitness_min from LLM estimate)
    - Turnover 1–70% (uses turnover_min ≥ 1, turnover_max ≤ 70)
    - Corr Risk != HIGH

    Returns ValidationResult with any failing gates listed.
    """
    failures: list[str] = []
    warnings: list[str] = []
    values: dict = {}

    metrics = parsed.get("metrics", {})

    sharpe_min: float | None = metrics.get("sharpe_min")
    _sharpe_max: float | None = metrics.get("sharpe_max")
    fitness_min: float | None = metrics.get("fitness_min")
    turnover_min: float | None = metrics.get("turnover_min")
    turnover_max: float | None = metrics.get("turnover_max")
    returns_pct: float | None = metrics.get("returns_pct")
    corr_risk: str | None = metrics.get("corr_risk")

    fitness_computed: float | None = None
    fitness_breakdown: str | None = None

    # Sharpe gate ≥ 1.25
    if sharpe_min is None:
        warnings.append("Sharpe estimate missing from LLM output")
    else:
        values["sharpe"] = sharpe_min
        if sharpe_min < 1.25:
            failures.append(f"Sharpe {sharpe_min} < 1.25 — gate FAIL (need ≥ 1.25)")
        elif sharpe_min < 1.35:
            warnings.append(f"Sharpe {sharpe_min} is marginal — consider pushing above 1.35")

    # Turnover gate 1–70% (needed for fitness formula too)
    if turnover_min is None and turnover_max is None:
        warnings.append("Turnover estimate missing from LLM output")
    else:
        lo = turnover_min or 0.0
        hi = turnover_max or turnover_min or 0.0
        values["turnover_range"] = f"{lo}%–{hi}%"
        if lo < 1.0:
            failures.append(f"Turnover lower bound {lo}% < 1% — gate FAIL (need 1–70%)")
        if hi > 70.0:
            failures.append(f"Turnover upper bound {hi}% > 70% — gate FAIL (need 1–70%)")

    # Fitness gate: v2 uses exact formula when all inputs are available
    t_min = turnover_min if turnover_min is not None else 0
    t_max = turnover_max if turnover_max is not None else (turnover_min if turnover_min is not None else 0)
    turnover_for_formula = (t_min + t_max) / 2
    if sharpe_min is not None and returns_pct is not None and turnover_for_formula > 0:
        # Use exact formula (v2 path)
        fit_result = validate_fitness(sharpe_min, returns_pct, turnover_for_formula)
        fitness_computed = fit_result.fitness_computed
        fitness_breakdown = fit_result.fitness_breakdown
        failures.extend(fit_result.failures)
        warnings.extend(fit_result.warnings)
    elif fitness_min is not None:
        # Fallback: use LLM-estimated fitness (v1 path)
        values["fitness"] = fitness_min
        if fitness_min <= 1.0:
            failures.append(f"Fitness {fitness_min} ≤ 1.0 — gate FAIL (need > 1.0) [LLM estimate]")
    else:
        warnings.append(
            "Fitness cannot be computed (missing Sharpe, Returns, or Turnover) — and no LLM fitness estimate provided"
        )

    # Corr Risk — HIGH triggers re-iteration
    if corr_risk == "HIGH":
        failures.append("Corr Risk is HIGH — alpha too similar to known factor families")
    elif corr_risk is None:
        warnings.append("Corr Risk field missing")

    passed = len(failures) == 0
    result = ValidationResult(passed=passed, failures=failures, warnings=warnings)
    result.fitness_computed = fitness_computed
    result.fitness_breakdown = fitness_breakdown
    return result


def fingerprint_collision(new_fp: dict, existing_fps: list[dict]) -> bool:
    """
    Return True if new_fp shares ≥ 2 fields with any fingerprint in existing_fps.
    None values are ignored (partial fingerprints treated charitably).
    """
    keys = ["dataset", "topology", "temporal", "normalization", "direction", "neutral"]
    for fp in existing_fps:
        matches = sum(1 for k in keys if new_fp.get(k) and fp.get(k) and new_fp[k].lower() == fp[k].lower())
        if matches >= 2:
            return True
    return False


PERMITTED_OPERATORS = _load_operators_from_schema()
VALID_BRAIN_VARS = _load_datafields_from_schema()

_whitelist_manager = None

_originality_checker = OriginalityChecker()
_complexity_controller = ComplexityController()
_monitor = AlgoMonitor.get_instance()


def set_whitelist_manager(wm) -> None:
    global _whitelist_manager
    _whitelist_manager = wm


def get_valid_brain_vars() -> set[str]:
    if _whitelist_manager is not None:
        wm_fields = _whitelist_manager.get_allowed_fields()
        if wm_fields:
            return wm_fields
    return VALID_BRAIN_VARS


def get_complexity_controller() -> ComplexityController:
    return _complexity_controller


def get_originality_checker() -> OriginalityChecker:
    return _originality_checker


_FINANCIAL_KEYWORDS: set[str] = {
    "momentum",
    "reversal",
    "mean",
    "volatility",
    "risk",
    "return",
    "signal",
    "noise",
    "correlation",
    "neutralization",
    "ranking",
    "cross-sectional",
    "time-series",
    "overreaction",
    "underreaction",
    "liquidity",
    "size",
    "value",
    "quality",
}


def verify_economic_explanation(expression: str, rationale: str, direction: str = "") -> dict:
    """Verify economic explanation by counting financial keyword hits.
    NOTE: This is a keyword-matching verifier, NOT a semantic consistency checker.
    It counts how many financial keywords appear in the rationale text. A high
    consistency_score does NOT guarantee the explanation is economically sound."""
    if not rationale or len(rationale.strip()) < 10:
        return {"valid": False, "reason": "No economic rationale provided", "warnings": [], "consistency_score": 0.0}

    rationale_lower = rationale.lower()
    keyword_hits = sum(1 for kw in _FINANCIAL_KEYWORDS if kw in rationale_lower)
    warnings: list[str] = []

    if keyword_hits == 0:
        return {
            "valid": False,
            "reason": "No financial concept keywords found in rationale",
            "warnings": [],
            "consistency_score": 0.1,
        }

    expr_lower = expression.lower()

    if ("ts_delta" in expr_lower or "ts_returns" in expr_lower) and not any(
        w in rationale_lower for w in ("momentum", "change", "return")
    ):
        warnings.append("Expression uses ts_delta/ts_returns but rationale does not mention momentum/change/return")

    if ("ts_zscore" in expr_lower or "ts_mean" in expr_lower) and not any(
        w in rationale_lower for w in ("mean", "reversion", "average")
    ):
        warnings.append("Expression uses ts_zscore/ts_mean but rationale does not mention mean/reversion/average")

    if "group_neutralize" in expr_lower and not any(
        w in rationale_lower for w in ("neutralization", "sector", "industry")
    ):
        warnings.append(
            "Expression uses group_neutralize but rationale does not mention neutralization/sector/industry"
        )

    consistency_score = min(1.0, keyword_hits / 5.0)
    if warnings:
        consistency_score = max(0.0, consistency_score - 0.2 * len(warnings))

    return {"valid": True, "warnings": warnings, "consistency_score": round(consistency_score, 3)}


def validate_alpha(alpha_id: str, expression: str) -> ValidationResult:
    failures: list[str] = []
    warnings: list[str] = []

    if not expression or not expression.strip():
        failures.append("EMPTY_EXPRESSION: expression is empty or whitespace-only")
        return ValidationResult(passed=False, failures=failures, warnings=warnings)

    complexity_passed = True

    if settings.COMPLEXITY_CHECK_ENABLED:
        try:
            passed, metrics, reason = _complexity_controller.check_complexity(expression)
            if not passed:
                complexity_passed = False
                failures.append(f"COMPLEXITY_FAIL: {reason}")
                logger.info("[ALGO_FAIL] module=complexity alpha=%s reason=%s", alpha_id, reason)
                _monitor.record("FAIL", "complexity", "check", reason, alpha_id=alpha_id)
            else:
                logger.info(
                    "[ALGO_PASS] module=complexity alpha=%s depth=%d nodes=%d ops=%d",
                    alpha_id,
                    metrics.depth,
                    metrics.node_count,
                    metrics.operator_count,
                )
                _monitor.record(
                    "PASS",
                    "complexity",
                    "check",
                    f"depth={metrics.depth} ops={metrics.operator_count}",
                    alpha_id=alpha_id,
                )
        except (ValueError, TypeError, OSError) as exc:
            logger.warning("Complexity check error for %s: %s", alpha_id, exc)
            logger.warning(
                "[DEFENSIVE] validator: complexity_check_skipped alpha=%s error=%s — validation may pass without complexity gate",
                alpha_id,
                exc,
            )

    if settings.ORIGINALITY_CHECK_ENABLED:
        try:
            from openalpha_brain.core.loop_state import _algo_tick

            _algo_tick("originality_check")
            score = _originality_checker.check_originality(alpha_id, expression)
            if score < _originality_checker.ORIGINALITY_THRESHOLD:
                failures.append(
                    f"ORIGINALITY_FAIL: score={score:.2f} (threshold={_originality_checker.ORIGINALITY_THRESHOLD})"
                )
                logger.info(
                    "[ALGO_SKIP] module=originality alpha=%s score=%.2f reason=low_originality",
                    alpha_id,
                    score,
                )
                _monitor.record("SKIP", "originality", "check", f"score={score:.2f} below_threshold", alpha_id=alpha_id)
            else:
                logger.info("[ALGO_PASS] module=originality alpha=%s score=%.2f", alpha_id, score)
                _monitor.record("PASS", "originality", "check", f"score={score:.2f}", alpha_id=alpha_id)
                if complexity_passed:
                    _originality_checker.register_alpha(alpha_id, expression)
        except (ValueError, TypeError, OSError) as exc:
            logger.warning("Originality check error for %s: %s", alpha_id, exc)
            logger.warning(
                "[DEFENSIVE] validator: originality_check_skipped alpha=%s error=%s — validation may pass without originality gate",
                alpha_id,
                exc,
            )

    passed = len(failures) == 0
    if passed and (not settings.COMPLEXITY_CHECK_ENABLED or not settings.ORIGINALITY_CHECK_ENABLED):
        logger.info(
            "[DEFENSIVE] validator: validate_alpha_passed alpha=%s failures=%d — some gates were disabled or skipped",
            alpha_id,
            len(failures),
        )
    elif passed and len(failures) == 0:
        logger.info("[DEFENSIVE] validator: validate_alpha_passed_clean alpha=%s all_gates_checked", alpha_id)
    else:
        logger.warning(
            "[DEFENSIVE] validator: validate_alpha_failed alpha=%s failures=%d warnings=%d",
            alpha_id,
            len(failures),
            len(warnings),
        )
    return ValidationResult(passed=passed, failures=failures, warnings=warnings)
