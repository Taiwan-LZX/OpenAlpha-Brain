"""
OpenAlpha-Brain — Adaptive Complexity Control (AlphaAgent §3.4)

Implements three-phase complexity regularization and P90 adaptive thresholds
for FASTEXPR alpha expressions. Uses regex-based estimation instead of full
AST parsing to avoid coupling with ast_originality.py (still in development).

Three-phase check order:
  Phase 1 — Structural constraint: AST depth <= max_depth
  Phase 2 — Operator constraint:   operator count <= max_operators
  Phase 3 — Parameter constraint:  total node count <= max_nodes

Adaptive thresholds: after >= 20 successful alphas are recorded, the P90
of each metric distribution replaces the default threshold.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_FUNC_CALL_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
_BARE_IDENT_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b")
_NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

_KNOWN_OPERATORS: set[str] = {
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

_KEYWORDS: set[str] = {"True", "False", "None", "NaN", "Inf"}

_ADAPT_MIN_SAMPLES = 20
_PERCENTILE = 90


@dataclass
class ComplexityMetrics:
    depth: int = 0
    node_count: int = 0
    operator_count: int = 0
    field_count: int = 0
    constant_count: int = 0


@dataclass
class ComplexityThresholds:
    max_depth: int = 5
    max_nodes: int = 30
    max_operators: int = 8


def _compute_paren_depth(expr: str) -> int:
    max_depth = 0
    current = 0
    for ch in expr:
        if ch == "(":
            current += 1
            if current > max_depth:
                max_depth = current
        elif ch == ")":
            current -= 1
    return max_depth


def _percentile(sorted_values: list[int], pct: int) -> int:
    if not sorted_values:
        return 0
    idx = math.ceil(pct / 100.0 * len(sorted_values)) - 1
    idx = max(0, min(idx, len(sorted_values) - 1))
    return sorted_values[idx]


class ComplexityController:
    def __init__(self, thresholds: ComplexityThresholds | None = None):
        self._thresholds = thresholds or ComplexityThresholds()
        self._success_history: list[ComplexityMetrics] = []

    def compute_complexity(self, expr: str) -> ComplexityMetrics:
        if not expr or not expr.strip():
            return ComplexityMetrics()

        depth = _compute_paren_depth(expr)

        called_funcs = set(_FUNC_CALL_RE.findall(expr))
        operator_count = 0
        for m in _FUNC_CALL_RE.finditer(expr):
            if m.group(1) in _KNOWN_OPERATORS:
                operator_count += 1

        all_idents = set(_BARE_IDENT_RE.findall(expr))
        bare_idents = all_idents - called_funcs - _KEYWORDS
        field_count = len(bare_idents)

        constant_count = len(_NUMERIC_RE.findall(expr))

        node_count = operator_count + field_count + constant_count

        return ComplexityMetrics(
            depth=depth,
            node_count=node_count,
            operator_count=operator_count,
            field_count=field_count,
            constant_count=constant_count,
        )

    def check_complexity(self, expr: str) -> tuple[bool, ComplexityMetrics, str]:
        metrics = self.compute_complexity(expr)
        t = self._thresholds

        if metrics.depth > t.max_depth:
            return (
                False,
                metrics,
                f"Phase 1 STRUCTURAL: depth {metrics.depth} > max_depth {t.max_depth}",
            )

        if metrics.operator_count > t.max_operators:
            return (
                False,
                metrics,
                f"Phase 2 OPERATOR: operator_count {metrics.operator_count} > max_operators {t.max_operators}",
            )

        if metrics.node_count > t.max_nodes:
            return (
                False,
                metrics,
                f"Phase 3 PARAMETER: node_count {metrics.node_count} > max_nodes {t.max_nodes}",
            )

        return True, metrics, ""

    def record_success(self, metrics: ComplexityMetrics) -> None:
        self._success_history.append(metrics)
        logger.debug(
            "Recorded success: depth=%d ops=%d nodes=%d (total=%d)",
            metrics.depth,
            metrics.operator_count,
            metrics.node_count,
            len(self._success_history),
        )

    def adapt_thresholds(self) -> bool:
        n = len(self._success_history)
        if n < _ADAPT_MIN_SAMPLES:
            logger.debug(
                "adapt_thresholds: only %d samples, need %d — skipping",
                n,
                _ADAPT_MIN_SAMPLES,
            )
            return False

        depths = sorted(m.depth for m in self._success_history)
        ops = sorted(m.operator_count for m in self._success_history)
        nodes = sorted(m.node_count for m in self._success_history)

        p90_depth = _percentile(depths, _PERCENTILE)
        p90_ops = _percentile(ops, _PERCENTILE)
        p90_nodes = _percentile(nodes, _PERCENTILE)

        old = ComplexityThresholds(
            max_depth=self._thresholds.max_depth,
            max_nodes=self._thresholds.max_nodes,
            max_operators=self._thresholds.max_operators,
        )

        self._thresholds.max_depth = max(p90_depth, 1)
        self._thresholds.max_operators = max(p90_ops, 1)
        self._thresholds.max_nodes = max(p90_nodes, 1)

        logger.info(
            "Adapted thresholds (P90 over %d samples): depth %d->%d, operators %d->%d, nodes %d->%d",
            n,
            old.max_depth,
            self._thresholds.max_depth,
            old.max_operators,
            self._thresholds.max_operators,
            old.max_nodes,
            self._thresholds.max_nodes,
        )
        return True

    @property
    def thresholds(self) -> ComplexityThresholds:
        return self._thresholds

    @property
    def success_count(self) -> int:
        return len(self._success_history)
