import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GARCHResult:
    omega: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    persistence: float = 0.0
    half_life: float = 0.0
    is_clustering: bool = False
    conditional_variances: list[float] = field(default_factory=list)

    def __post_init__(self):
        pass


def estimate_garch11(returns: list[float], max_iter: int = 100, tol: float = 1e-6) -> GARCHResult:
    if len(returns) < 20:
        return GARCHResult()

    clean = [r for r in returns if isinstance(r, (int, float)) and math.isfinite(r)]
    if len(clean) < 20:
        return GARCHResult()

    n = len(clean)
    mean_ret = sum(clean) / n
    residuals = [r - mean_ret for r in clean]

    var = sum(r * r for r in residuals) / n
    omega = var * 0.1
    alpha = 0.1
    beta = 0.85

    for _ in range(max_iter):
        cond_var = [0.0] * n
        cond_var[0] = var

        for t in range(1, n):
            cond_var[t] = omega + alpha * residuals[t - 1] ** 2 + beta * cond_var[t - 1]
            if cond_var[t] <= 0:
                cond_var[t] = 1e-10

        if alpha + beta < 1.0:
            omega_new = var * (1 - alpha - beta)
        else:
            omega_new = var * 0.01

        sq_res = [r ** 2 for r in residuals[1:]]
        sq_res_prev = [residuals[i - 1] ** 2 for i in range(1, n)]
        if len(sq_res) > 2:
            mean_sq = sum(sq_res) / len(sq_res)
            mean_sq_prev = sum(sq_res_prev) / len(sq_res_prev)
            if mean_sq_prev > 0 and mean_sq > 0:
                cov = sum(a * b for a, b in zip(sq_res, sq_res_prev)) / len(sq_res) - mean_sq * mean_sq_prev
                var_sq = sum(a ** 2 for a in sq_res_prev) / len(sq_res_prev) - mean_sq_prev ** 2
                if var_sq > 0:
                    alpha_new = max(0.01, min(0.5, cov / var_sq))
                else:
                    alpha_new = alpha
            else:
                alpha_new = alpha
        else:
            alpha_new = alpha

        beta_new = max(0.5, min(0.95, 1 - alpha_new - omega_new / var)) if var > 0 else beta

        if abs(omega_new - omega) < tol and abs(alpha_new - alpha) < tol and abs(beta_new - beta) < tol:
            omega, alpha, beta = omega_new, alpha_new, beta_new
            break

        omega, alpha, beta = omega_new, alpha_new, beta_new

    persistence = alpha + beta
    half_life = -math.log(2) / math.log(persistence) if 0 < persistence < 1 else float("inf")
    is_clustering = persistence > 0.9

    cond_var = [0.0] * n
    cond_var[0] = var
    for t in range(1, n):
        cond_var[t] = omega + alpha * residuals[t - 1] ** 2 + beta * cond_var[t - 1]
        if cond_var[t] <= 0:
            cond_var[t] = 1e-10

    return GARCHResult(
        omega=round(omega, 6),
        alpha=round(alpha, 4),
        beta=round(beta, 4),
        persistence=round(persistence, 4),
        half_life=round(half_life, 2),
        is_clustering=is_clustering,
        conditional_variances=cond_var,
    )
