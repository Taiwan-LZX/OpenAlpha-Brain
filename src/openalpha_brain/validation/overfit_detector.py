import logging
import math
from dataclasses import dataclass, field

from openalpha_brain.config.config import settings

logger = logging.getLogger(__name__)

@dataclass
class OverfitResult:
    is_overfit: bool = False
    is_os_decay_ratio: float = 0.0
    yearly_sharpe_std: float = 0.0
    yearly_sharpe_cv: float = 0.0
    worst_year_sharpe: float = 0.0
    best_year_sharpe: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    consistency_score: float = 1.0
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self):
        pass

_IS_OS_DECAY_SEVERE = getattr(settings, 'OVERFIT_IS_OS_DECAY_SEVERE', 0.5)
_IS_OS_DECAY_WARNING = getattr(settings, 'OVERFIT_IS_OS_DECAY_WARNING', 0.7)
_YEARLY_SHARPE_CV_SEVERE = getattr(settings, 'OVERFIT_YEARLY_SHARPE_CV_SEVERE', 1.0)
_YEARLY_SHARPE_CV_WARNING = getattr(settings, 'OVERFIT_YEARLY_SHARPE_CV_WARNING', 0.5)


def detect_overfit(
    is_sharpe: float | None = None,
    os_sharpe: float | None = None,
    yearly_sharpes: list[float] | None = None,
    pnl_curve: list[float] | None = None,
) -> OverfitResult:
    """Detect overfitting using multiple signals.

    Signals:
    1. IS/OS Sharpe decay ratio (os_sharpe / is_sharpe)
    2. Yearly Sharpe consistency (coefficient of variation)
    3. Drawdown recovery analysis
    """
    result = OverfitResult()

    # Signal 1: IS/OS decay
    if is_sharpe is not None and os_sharpe is not None and is_sharpe > 0:
        result.is_os_decay_ratio = round(os_sharpe / is_sharpe, 4)
        if result.is_os_decay_ratio < _IS_OS_DECAY_SEVERE:
            result.is_overfit = True
            result.warnings.append(f"IS/OS decay ratio {result.is_os_decay_ratio:.2f} < {_IS_OS_DECAY_SEVERE}: severe overfitting")
        elif result.is_os_decay_ratio < _IS_OS_DECAY_WARNING:
            result.warnings.append(f"IS/OS decay ratio {result.is_os_decay_ratio:.2f} < {_IS_OS_DECAY_WARNING}: moderate overfitting risk")

    # Signal 2: Yearly Sharpe consistency
    if yearly_sharpes and len(yearly_sharpes) >= 2:
        valid = [s for s in yearly_sharpes if isinstance(s, (int, float)) and math.isfinite(s)]
        if len(valid) >= 2:
            mean_s = sum(valid) / len(valid)
            result.best_year_sharpe = max(valid)
            result.worst_year_sharpe = min(valid)
            if mean_s > 0:
                variance = sum((s - mean_s) ** 2 for s in valid) / len(valid)
                result.yearly_sharpe_std = math.sqrt(variance)
                result.yearly_sharpe_cv = result.yearly_sharpe_std / mean_s
                if result.yearly_sharpe_cv > _YEARLY_SHARPE_CV_SEVERE:
                    result.warnings.append(f"Yearly Sharpe CV {result.yearly_sharpe_cv:.2f} > {_YEARLY_SHARPE_CV_SEVERE}: highly inconsistent")
                    result.is_overfit = True
                elif result.yearly_sharpe_cv > _YEARLY_SHARPE_CV_WARNING:
                    result.warnings.append(f"Yearly Sharpe CV {result.yearly_sharpe_cv:.2f} > {_YEARLY_SHARPE_CV_WARNING}: inconsistent performance")

            if mean_s > 1.0 and result.worst_year_sharpe < 0:
                result.warnings.append(f"Year reversal: mean Sharpe {mean_s:.2f} but worst year {result.worst_year_sharpe:.2f}")
                result.consistency_score = 0.5

    # Signal 3: Drawdown recovery analysis
    if pnl_curve and len(pnl_curve) >= 10:
        max_drawdown = 0.0
        peak = pnl_curve[0]
        drawdown_start_idx = 0
        max_dd_duration = 0
        current_dd_start = 0
        in_drawdown = False
        for i, val in enumerate(pnl_curve):
            if val > peak:
                peak = val
                if in_drawdown:
                    dd_duration = i - current_dd_start
                    max_dd_duration = max(max_dd_duration, dd_duration)
                    in_drawdown = False
            dd = (peak - val) / peak if peak > 0 else 0.0
            if dd > max_drawdown:
                max_drawdown = dd
                if not in_drawdown:
                    current_dd_start = i
                    in_drawdown = True
        if in_drawdown:
            max_dd_duration = max(max_dd_duration, len(pnl_curve) - current_dd_start)
        if max_drawdown > 0.3:
            result.warnings.append(f"Max drawdown {max_drawdown:.1%} > 30%: severe drawdown risk")
            result.is_overfit = True
        elif max_drawdown > 0.15:
            result.warnings.append(f"Max drawdown {max_drawdown:.1%} > 15%: moderate drawdown risk")
        if max_dd_duration > len(pnl_curve) * 0.4 and max_drawdown > 0.1:
            result.warnings.append(f"Drawdown recovery took {max_dd_duration}/{len(pnl_curve)} periods: slow recovery")
            if max_drawdown > 0.2:
                result.is_overfit = True
        result.max_drawdown = max_drawdown
        result.max_drawdown_duration = max_dd_duration

    if not result.warnings:
        result.consistency_score = 1.0
    elif result.is_overfit:
        result.consistency_score = 0.3
    else:
        result.consistency_score = max(0.5, 1.0 - 0.1 * len(result.warnings))

    return result
