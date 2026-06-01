from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple


class FailureMode(Enum):
    FM1_LOW_SIGNAL_QUALITY = ("FM-1", "Low Signal Quality",
                               "Sharpe below threshold, flat PnL")
    FM2_EXCESSIVE_TURNOVER = ("FM-2", "Excessive Turnover",
                               "Turnover exceeds upper bound")
    FM3_HIGH_DRAWDOWN      = ("FM-3", "High Drawdown",
                               "Max drawdown exceeds threshold")
    FM4_SIGNAL_REVERSAL    = ("FM-4", "Signal Reversal",
                               "Negative annualized returns")
    FM5_LOW_COVERAGE       = ("FM-5", "Low Coverage",
                               "Long/short count below minimum")
    FM6_LOW_TURNOVER       = ("FM-6", "Low Turnover",
                               "Turnover below lower bound")

    @property
    def code(self) -> str:
        return self.value[0]

    @property
    def name(self) -> str:
        return self.value[1]

    @property
    def description(self) -> str:
        return self.value[2]


CHECK_TO_FM: Dict[str, FailureMode] = {
    "LOW_SHARPE":                FailureMode.FM1_LOW_SIGNAL_QUALITY,
    "LOW_FITNESS":               FailureMode.FM1_LOW_SIGNAL_QUALITY,
    "HIGH_TURNOVER":             FailureMode.FM2_EXCESSIVE_TURNOVER,
    "HIGH_DRAWDOWN":             FailureMode.FM3_HIGH_DRAWDOWN,
    "LOW_SUB_UNIVERSE_SHARPE":   FailureMode.FM5_LOW_COVERAGE,
    "LOW_TURNOVER":              FailureMode.FM6_LOW_TURNOVER,
    "CONCENTRATED_WEIGHT":       FailureMode.FM5_LOW_COVERAGE,
    "NEGATIVE_RETURNS":          FailureMode.FM4_SIGNAL_REVERSAL,
    "LOW_RETURNS":               FailureMode.FM4_SIGNAL_REVERSAL,
    "RETURNS":                   FailureMode.FM4_SIGNAL_REVERSAL,
}

CORRELATION_CHECKS = {"SELF_CORRELATION", "PROD_CORRELATION"}
BLOCKING_CHECKS = {
    "LOW_SHARPE", "LOW_FITNESS", "HIGH_TURNOVER", "LOW_TURNOVER",
    "HIGH_DRAWDOWN", "CONCENTRATED_WEIGHT", "LOW_SUB_UNIVERSE_SHARPE",
    "NEGATIVE_RETURNS", "LOW_RETURNS", "RETURNS",
}
ALL_CHECKS = BLOCKING_CHECKS | CORRELATION_CHECKS


def classify_failures(failed_checks: Sequence[str]) -> List[FailureMode]:
    seen: set[FailureMode] = set()
    result: List[FailureMode] = []
    for check in failed_checks:
        fm = CHECK_TO_FM.get(check)
        if fm is not None and fm not in seen:
            seen.add(fm)
            result.append(fm)
    return result


def primary_failure_mode(failed_checks: Sequence[str]) -> Optional[FailureMode]:
    fms = classify_failures(failed_checks)
    severity: Dict[FailureMode, int] = {
        FailureMode.FM1_LOW_SIGNAL_QUALITY: 5,
        FailureMode.FM2_EXCESSIVE_TURNOVER: 4,
        FailureMode.FM3_HIGH_DRAWDOWN:     3,
        FailureMode.FM4_SIGNAL_REVERSAL:   2,
        FailureMode.FM5_LOW_COVERAGE:      1,
        FailureMode.FM6_LOW_TURNOVER:      1,
    }
    best = None
    best_sev = -1
    for fm in fms:
        sev = severity.get(fm, 0)
        if sev > best_sev:
            best_sev = sev
            best = fm
    return best


def is_correlation_blocked(failed_checks: Sequence[str]) -> bool:
    return any(c in CORRELATION_CHECKS for c in failed_checks)


REPAIR_CATALOGUE: Dict[FailureMode, List[Tuple[str, str, str]]] = {
    FailureMode.FM1_LOW_SIGNAL_QUALITY: [
        ("neutralize_sector",      "group_neutralize(x, sector)",
         "Wrap with group_neutralize(x, sector) to remove sector beta"),
        ("neutralize_subindustry", "group_neutralize(x, subindustry)",
         "Apply group_neutralize(x, subindustry) for finer neutralization"),
        ("zscore_normalize",       "zscore(x)",
         "Apply zscore(x) cross-sectionally to reduce noise"),
        ("winsorize_outliers",     "winsorize(x, std=3)",
         "Clip outliers with winsorize(x, std=3)"),
        ("extend_lookback_2x",     "increase lookback ×2",
         "Increase all lookback parameters by 2×"),
    ],
    FailureMode.FM2_EXCESSIVE_TURNOVER: [
        ("decay_linear_5",         "ts_decay_linear(x, 5)",
         "Smooth via ts_decay_linear(x, 5) to reduce turnover"),
        ("mean_std_ratio",         "ts_mean(x,d) / ts_std_dev(x,d)",
         "Normalize as ts_mean(x,d) / ts_std_dev(x,d)"),
        ("pasteurize_add",         "pasteurize(x)",
         "Add pasteurize(x) to improve coverage"),
        ("restructure_rank",       "rank()",
         "Restructure using rank() cross-sectionally"),
    ],
    FailureMode.FM3_HIGH_DRAWDOWN: [
        ("winsorize_3std",         "winsorize(x, std=3)",
         "Clip outliers with winsorize(x, std=3) to reduce drawdown"),
        ("volatility_condition",   "condition on low vol regime",
         "Condition on low volatility regime"),
    ],
    FailureMode.FM4_SIGNAL_REVERSAL: [
        ("negate_expr",            "-1 * (expr)",
         "Negate expression to flip signal direction"),
        ("inverse_operator",       "invert core operator sign",
         "Invert core operator (positive → negative)"),
    ],
    FailureMode.FM5_LOW_COVERAGE: [
        ("pasteurize_wrap",        "pasteurize(x)",
         "Wrap with pasteurize(x) to improve coverage"),

    ],
    FailureMode.FM6_LOW_TURNOVER: [
        ("reduce_lookback",        "reduce lookback window",
         "Reduce lookback window to increase turnover"),
        ("cross_sectional_tilt",   "add cross-sectional variation",
         "Add cross-sectional variation via rank or zscore"),
    ],
}

REPAIR_SEVERITY: Dict[str, int] = {
    "neutralize_sector":      80,
    "neutralize_subindustry": 75,
    "zscore_normalize":       70,
    "winsorize_outliers":     65,
    "extend_lookback_2x":     60,
    "decay_linear_5":         85,
    "mean_std_ratio":         80,
    "pasteurize_add":         70,
    "restructure_rank":       75,
    "winsorize_3std":         60,
    "volatility_condition":   55,
    "negate_expr":            90,
    "inverse_operator":       85,
    "pasteurize_wrap":        75,
    "reduce_lookback":        70,
    "cross_sectional_tilt":   75,
}


def repair_severity(strategy_name: str) -> int:
    return REPAIR_SEVERITY.get(strategy_name, 50)
