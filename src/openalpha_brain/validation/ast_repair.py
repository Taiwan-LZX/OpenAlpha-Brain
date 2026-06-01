"""
OpenAlpha-Brain — AST Auto-Repair Module

Parses alpha expressions, detects invalid operators/variables,
and attempts automatic repair by finding the closest valid alternative
using string similarity (Levenshtein distance).

Also records errors to the session's hallucination_log for LLM feedback.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC

from openalpha_brain.data import get_data_path
from openalpha_brain.utils.algo_logger import algo_log

logger = logging.getLogger(__name__)

_PASCAL_CASE_MAP = {
    "TSDecayLinear": "ts_decay_linear",
    "TSDelta": "ts_delta",
    "TSZScore": "ts_zscore",
    "TSRank": "ts_rank",
    "TSStdDev": "ts_std_dev",
    "TSMean": "ts_mean",
    "TSRegression": "ts_regression",
    "TSMin": "ts_min",
    "TSMax": "ts_max",
    "TSArgMax": "ts_argmax",
    "TSArgMin": "ts_argmin",
    "TSDelay": "ts_delay",
    "TSSum": "ts_sum",
    "TSCorr": "ts_corr",
    "TSQuantile": "ts_quantile",
    "TSBackfill": "ts_backfill",
    "TSAvDiff": "ts_av_diff",
    "TSProduct": "ts_product",
    "TSScale": "ts_scale",
    "GroupNeutralize": "group_neutralize",
    "GroupZScore": "group_zscore",
    "GroupRank": "group_rank",
    "GroupMean": "group_mean",
    "GroupScale": "group_scale",
    "GroupBackfill": "group_backfill",
    "Negative": "-",
    "Reverse": "-",
    "SignedPower": "signed_power",
    "TradeWhen": "trade_when",
}

_COMMON_ENGLISH_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "need",
    "dare",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "because",
    "but",
    "and",
    "or",
    "if",
    "while",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "we",
    "our",
    "you",
    "your",
    "they",
    "them",
    "their",
    "what",
    "which",
    "who",
    "whom",
    "often",
    "captures",
    "using",
    "uses",
    "used",
    "based",
    "applies",
    "apply",
    "applied",
    "computes",
    "compute",
    "calculates",
    "calculate",
    "measures",
    "measure",
    "returns",
    "return",
    "provides",
    "provide",
    "takes",
    "take",
    "generates",
    "generate",
    "expression",
    "alpha",
    "signal",
    "value",
}


_FUNC_CALL_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
_BARE_IDENT_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b")
_DOT_METHOD_RE = re.compile(r"\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")


def _levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row: list[int] = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def _find_closest(name: str, candidates: set[str], max_distance: int = 3) -> str | None:
    best = None
    best_dist = max_distance + 1
    for candidate in candidates:
        d = _levenshtein(name, candidate)
        if d < best_dist:
            best_dist = d
            best = candidate
    return best if best_dist <= max_distance else None


def _load_valid_operators() -> set[str]:
    path = get_data_path("brain_operators.json")
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {op.get("name", "") for op in data if op.get("name")}


def _load_valid_variables() -> set[str]:
    path = get_data_path("brain_datafields.json")
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    names = {field.get("id", "") for field in data if field.get("id")}
    names.update({"industry", "subindustry", "sector"})
    return names


def _find_expr_start(s: str, pos: int) -> int:
    i = pos
    while True:
        if i < 0:
            return 0
        seg_start = i
        if s[i] == ")":
            depth = 1
            i -= 1
            while i >= 0 and depth > 0:
                if s[i] == ")":
                    depth += 1
                elif s[i] == "(":
                    depth -= 1
                i -= 1
            i += 1
            seg_start = i
            i -= 1
            if i >= 0 and (s[i].isalnum() or s[i] == "_"):
                while i >= 0 and (s[i].isalnum() or s[i] == "_"):
                    i -= 1
                i += 1
                seg_start = i
        elif s[i].isalnum() or s[i] == "_":
            while i >= 0 and (s[i].isalnum() or s[i] == "_"):
                i -= 1
            i += 1
            seg_start = i
        else:
            return i + 1
        if seg_start > 0 and s[seg_start - 1] == ".":
            i = seg_start - 2
            continue
        return seg_start


def _repair_dot_method_chains(expression: str) -> tuple[str, list[dict]]:
    entries: list[dict] = []
    result = expression
    while True:
        matches = list(_DOT_METHOD_RE.finditer(result))
        if not matches:
            break
        match = matches[-1]
        method_name = match.group(1)
        dot_pos = match.start()
        if dot_pos == 0:
            break
        open_paren_pos = match.end() - 1
        depth = 1
        close_paren_pos = open_paren_pos + 1
        while close_paren_pos < len(result) and depth > 0:
            if result[close_paren_pos] == "(":
                depth += 1
            elif result[close_paren_pos] == ")":
                depth -= 1
            close_paren_pos += 1
        close_paren_pos -= 1
        if depth != 0:
            break
        args = result[open_paren_pos + 1 : close_paren_pos].strip()
        prec_search_pos = dot_pos - 1
        while prec_search_pos >= 0 and result[prec_search_pos] in " \t":
            prec_search_pos -= 1
        if prec_search_pos < 0:
            break
        prec_start = _find_expr_start(result, prec_search_pos)
        preceding_expr = result[prec_start : prec_search_pos + 1].strip()
        if not preceding_expr:
            break
        replacement = f"{method_name}({preceding_expr}, {args})" if args else f"{method_name}({preceding_expr})"
        result = result[:prec_start] + replacement + result[close_paren_pos + 1 :]
        entries.append(
            {
                "variable": method_name,
                "error_type": "DOT_METHOD_REPAIRED",
                "error_message": f"Dot method '.{method_name}()' converted to function call '{method_name}()'",
                "source": "ast_repair",
            }
        )
    return result, entries


@algo_log()
def repair_expression(
    expression: str,
) -> tuple[str, list[dict]]:
    """
    Attempt to repair an alpha expression by replacing invalid
    operators and variables with closest valid alternatives.

    Returns (repaired_expression, new_hallucination_entries).
    """
    expression, dot_entries = _repair_dot_method_chains(expression)

    valid_ops = _VALID_OPS_CACHE
    valid_vars = _VALID_VARS_CACHE

    if not valid_ops and not valid_vars:
        if dot_entries:
            from datetime import datetime

            for entry in dot_entries:
                entry["timestamp"] = datetime.now(UTC).isoformat()
        return expression, dot_entries

    called_funcs = set(_FUNC_CALL_RE.findall(expression))
    all_idents = set(_BARE_IDENT_RE.findall(expression))
    bare_vars = all_idents - called_funcs

    new_entries: list[dict] = dot_entries
    repaired = expression

    invalid_ops = called_funcs - valid_ops if valid_ops else set()
    for op in list(invalid_ops):
        if op in _PASCAL_CASE_MAP:
            mapped = _PASCAL_CASE_MAP[op]
            if mapped == "-":
                repaired = re.sub(r"\b" + re.escape(op) + r"\s*\(", "-(", repaired)
            else:
                repaired = re.sub(r"\b" + re.escape(op) + r"\s*\(", f"{mapped}(", repaired)
            new_entries.append(
                {
                    "variable": op,
                    "error_type": "UNKNOWN_OPERATOR",
                    "error_message": f"PascalCase operator '{op}' mapped to snake_case '{mapped}'",
                    "source": "ast_repair",
                }
            )
            invalid_ops.discard(op)
            continue

        closest = _find_closest(op, valid_ops) if valid_ops else None
        if closest:
            repaired = re.sub(r"\b" + re.escape(op) + r"\s*\(", f"{closest}(", repaired)
            new_entries.append(
                {
                    "variable": op,
                    "error_type": "UNKNOWN_OPERATOR",
                    "error_message": f"Unknown operator '{op}' replaced with '{closest}'",
                    "source": "ast_repair",
                }
            )
        else:
            new_entries.append(
                {
                    "variable": op,
                    "error_type": "UNKNOWN_OPERATOR",
                    "error_message": f"Unknown operator '{op}' — no close match found",
                    "source": "ast_repair",
                }
            )

    invalid_vars = bare_vars - valid_vars if valid_vars else set()
    from openalpha_brain.validation.validator import PERMITTED_OPERATORS

    skip = PERMITTED_OPERATORS | {"True", "False", "None", "NaN", "Inf"}
    skip |= _COMMON_ENGLISH_WORDS
    invalid_vars -= skip

    for var in list(invalid_vars):
        if var in _PASCAL_CASE_MAP:
            mapped = _PASCAL_CASE_MAP[var]
            pattern = r"\b" + re.escape(var) + r"\b"
            repaired = re.sub(pattern, "-", repaired) if mapped == "-" else re.sub(pattern, mapped, repaired)
            new_entries.append(
                {
                    "variable": var,
                    "error_type": "INVALID_VAR",
                    "error_message": f"PascalCase/invalid identifier '{var}' mapped to '{mapped}'",
                    "source": "ast_repair",
                }
            )
            invalid_vars.discard(var)
            continue

        closest = _find_closest(var, valid_vars) if valid_vars else None
        if closest and closest != var:
            pattern = r"\b" + re.escape(var) + r"\b"
            repaired = re.sub(pattern, closest, repaired)
            new_entries.append(
                {
                    "variable": var,
                    "error_type": "INVALID_VAR",
                    "error_message": f"Invalid variable '{var}' replaced with '{closest}'",
                    "source": "ast_repair",
                }
            )
        elif var.lower() in _COMMON_ENGLISH_WORDS or len(var) <= 2:
            pass
        else:
            new_entries.append(
                {
                    "variable": var,
                    "error_type": "INVALID_VAR",
                    "error_message": f"Invalid variable '{var}' — no close match found",
                    "source": "ast_repair",
                }
            )

    if new_entries:
        from datetime import datetime

        for entry in new_entries:
            entry["timestamp"] = datetime.now(UTC).isoformat()

    return repaired, new_entries


_VALID_OPS_CACHE: set[str] = _load_valid_operators()
_VALID_VARS_CACHE: set[str] = _load_valid_variables()
