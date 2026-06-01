import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class YearlyConsistency:
    year_sharpes: dict[int, float] = field(default_factory=dict)
    sharpe_std: float = 0.0
    sharpe_mean: float = 0.0
    is_stable: bool = True
    worst_year: int = 0
    worst_year_sharpe: float = 0.0
    best_year: int = 0
    best_year_sharpe: float = 0.0

@dataclass
class DrawdownAnalysis:
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: int = 0
    current_drawdown_pct: float = 0.0
    is_acceptable: bool = True

@dataclass
class PnLStabilityReport:
    yearly_consistency: YearlyConsistency | None = None
    drawdown_analysis: DrawdownAnalysis | None = None
    stability_score: float = 1.0
    reward_adjustment: float = 0.0
    warnings: list[str] = field(default_factory=list)

class PnLAnalyzer:
    SHARPE_STD_THRESHOLD = 1.0
    DRAWDOWN_THRESHOLD = 10.0
    YEARLY_INSTABILITY_PENALTY = 0.05
    DRAWDOWN_PENALTY = 0.03

    def analyze_yearly_consistency(self, yearly_data: list[dict]) -> YearlyConsistency:
        if not yearly_data:
            return YearlyConsistency()

        year_sharpes = {}
        for row in yearly_data:
            year = row.get("year", 0)
            sharpe = row.get("sharpe", 0.0)
            if not isinstance(sharpe, (int, float)) or not math.isfinite(sharpe):
                sharpe = 0.0
            if year > 0:
                year_sharpes[year] = sharpe

        if len(year_sharpes) < 2:
            return YearlyConsistency(
                year_sharpes=year_sharpes,
                is_stable=True,
            )

        values = [v for v in year_sharpes.values() if math.isfinite(v)]
        if len(values) < 2:
            return YearlyConsistency(
                year_sharpes=year_sharpes,
                is_stable=True,
            )

        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        std_val = math.sqrt(variance)

        worst_year = min(year_sharpes, key=year_sharpes.get)  # type: ignore[arg-type]
        best_year = max(year_sharpes, key=year_sharpes.get)  # type: ignore[arg-type]

        return YearlyConsistency(
            year_sharpes=year_sharpes,
            sharpe_std=round(std_val, 4),
            sharpe_mean=round(mean_val, 4),
            is_stable=std_val <= self.SHARPE_STD_THRESHOLD,
            worst_year=worst_year,
            worst_year_sharpe=year_sharpes[worst_year],
            best_year=best_year,
            best_year_sharpe=year_sharpes[best_year],
        )

    def analyze_drawdown(self, pnl_curve: list[float]) -> DrawdownAnalysis:
        if not pnl_curve or len(pnl_curve) < 2:
            return DrawdownAnalysis()

        pnl_curve = [v for v in pnl_curve if isinstance(v, (int, float)) and math.isfinite(v)]

        if len(pnl_curve) < 2:
            return DrawdownAnalysis()

        equity = [0.0]
        for pnl in pnl_curve:
            if not math.isfinite(pnl):
                pnl = 0.0
            equity.append(equity[-1] + pnl)

        peak = equity[0]
        max_dd = 0.0
        max_dd_duration = 0
        current_dd_start = 0
        in_drawdown = False

        for i, val in enumerate(equity):
            if val > peak:
                peak = val
                if in_drawdown:
                    duration = i - current_dd_start
                    max_dd_duration = max(max_dd_duration, duration)
                    in_drawdown = False
            elif val < peak:
                if peak > 0:
                    dd = (peak - val) / peak * 100
                else:
                    dd = 100.0
                max_dd = max(max_dd, dd)
                if not in_drawdown:
                    in_drawdown = True
                    current_dd_start = i

        if in_drawdown:
            duration = len(equity) - current_dd_start
            max_dd_duration = max(max_dd_duration, duration)

        current_dd = 0.0
        if equity:
            current_peak = max(equity)
            if current_peak > 0:
                current_dd = (current_peak - equity[-1]) / current_peak * 100
            elif equity[-1] < current_peak:
                current_dd = 100.0

        return DrawdownAnalysis(
            max_drawdown_pct=round(max_dd, 2),
            max_drawdown_duration_days=max_dd_duration,
            current_drawdown_pct=round(current_dd, 2),
            is_acceptable=max_dd <= self.DRAWDOWN_THRESHOLD,
        )

    def compute_stability_score(
        self,
        yearly_data: list[dict] | None = None,
        pnl_curve: list[float] | None = None,
        drawdown_pct: float | None = None,
        garch_persistence: float | None = None,
    ) -> float:
        score = 0.5

        if yearly_data:
            sharpes = [yd.get("sharpe", 0.0) for yd in yearly_data if isinstance(yd.get("sharpe"), (int, float)) and math.isfinite(yd.get("sharpe", 0.0))]
            if len(sharpes) >= 2:
                mean_s = sum(sharpes) / len(sharpes)
                if mean_s > 0:
                    std_s = math.sqrt(sum((s - mean_s)**2 for s in sharpes) / len(sharpes))
                    cv = std_s / mean_s
                    if cv < 0.3:
                        score += 0.2
                    elif cv < 0.5:
                        score += 0.1
                    elif cv > 1.0:
                        score -= 0.2
                    else:
                        score -= 0.1

        if drawdown_pct is not None:
            if drawdown_pct < 5.0:
                score += 0.1
            elif drawdown_pct < 10.0:
                score += 0.0
            elif drawdown_pct < 20.0:
                score -= 0.1
            else:
                score -= 0.2

        if garch_persistence is not None:
            if garch_persistence > 0.95:
                score -= 0.1
            elif garch_persistence < 0.8:
                score += 0.1

        if pnl_curve and len(pnl_curve) > 20:
            mean_pnl = sum(pnl_curve) / len(pnl_curve)
            residuals = [p - mean_pnl for p in pnl_curve]
            sq_res = [r * r for r in residuals]
            if len(sq_res) > 2:
                mean_sq = sum(sq_res) / len(sq_res)
                if mean_sq > 0:
                    cov = sum(sq_res[i] * sq_res[i-1] for i in range(1, len(sq_res))) / (len(sq_res) - 1)
                    autocorr = cov / mean_sq if mean_sq > 0 else 0
                    if autocorr > 0.3:
                        score -= 0.1
                    elif autocorr < 0.1:
                        score += 0.1

        return max(0.0, min(1.0, round(score, 4)))

    def generate_stability_report(
        self,
        yearly_data: list[dict] | None = None,
        pnl_curve: list[float] | None = None,
    ) -> PnLStabilityReport:
        consistency = None
        drawdown = None
        reward_adj = 0.0
        warnings = []

        if yearly_data:
            consistency = self.analyze_yearly_consistency(yearly_data)
            if not consistency.is_stable:
                reward_adj -= self.YEARLY_INSTABILITY_PENALTY
                warnings.append(
                    f"Yearly instability: Sharpe std={consistency.sharpe_std:.2f} > {self.SHARPE_STD_THRESHOLD}",
                )

        if pnl_curve:
            drawdown = self.analyze_drawdown(pnl_curve)
            if not drawdown.is_acceptable:
                reward_adj -= self.DRAWDOWN_PENALTY
                warnings.append(
                    f"High drawdown: {drawdown.max_drawdown_pct:.1f}% > {self.DRAWDOWN_THRESHOLD}%",
                )

        drawdown_pct_for_score = None
        if drawdown:
            drawdown_pct_for_score = drawdown.max_drawdown_pct

        stability_score = self.compute_stability_score(
            yearly_data=yearly_data,
            pnl_curve=pnl_curve,
            drawdown_pct=drawdown_pct_for_score,
        )

        return PnLStabilityReport(
            yearly_consistency=consistency,
            drawdown_analysis=drawdown,
            stability_score=stability_score,
            reward_adjustment=round(reward_adj, 4),
            warnings=warnings,
        )
