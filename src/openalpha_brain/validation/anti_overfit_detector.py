"""Anti-overfit detection — lightweight & full versions

Lightweight version works with WQ BRAIN cloud metrics (no pandas required).
Full version wraps QuantGPT's original implementation (requires pandas).

Tests:
  Lightweight:
    1. Sharpe consistency with historical average
    2. Turnover sanity check
    3. Fitness efficiency analysis
    4. Drawdown stability check
    5. Failed-check pattern analysis

  Full (QuantGPT):
    1. IC Stability — yearly Spearman IC consistency
    2. Sub-sample Stress — IC across market regimes
    3. Placebo Test — random permutation + time-shift
    4. Half-life Estimation — IC decay across forward periods
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    name: str
    passed: bool
    details: dict


@dataclass
class AntiOverfitResult:
    score: float
    recommendation: str
    tests: list[TestResult] = field(default_factory=list)
    passed_count: int = 0
    total_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "recommendation": self.recommendation,
            "passed_count": self.passed_count,
            "total_count": self.total_count,
            "tests": [
                {"name": t.name, "passed": t.passed, "details": t.details}
                for t in self.tests
            ],
        }


class LightweightAntiOverfitDetector:
    """Lightweight anti-overfit detector based on WQ BRAIN cloud metrics.

    Does NOT require pandas or local backtest data.
    Works with aggregated metrics returned by WQ platform.

    Args:
        historical_sharpes: Optional list of historical Sharpe ratios for consistency checking.
    """

    def __init__(self, historical_sharpes: Optional[list[float]] = None):
        self._historical_sharpes: list[float] = list(historical_sharpes) if historical_sharpes else []

    def evaluate(self, metrics: dict) -> AntiOverfitResult:
        """Run all lightweight tests against WQ metrics.

        Args:
            metrics: Dict from WQ containing at minimum:
                - sharpe (float): Sharpe ratio
                - fitness (float): Fitness score
                - turnover (float): Turnover rate (0-1 or percentage)
                - returns (float): Total/annualized returns
                - checks (list[dict]): Validation checks with keys name, value, limit, result
                - drawdown (float, optional): Max drawdown

        Returns:
            AntiOverfitResult with composite score and recommendation.
        """
        tests = [
            self._test_sharpe_consistency(metrics.get("sharpe")),
            self._test_turnover_sanity(metrics.get("turnover")),
            self._test_fitness_efficiency(
                metrics.get("fitness"),
                metrics.get("sharpe"),
                metrics.get("turnover"),
            ),
            self._test_drawdown_stability(metrics.get("drawdown"), metrics.get("sharpe")),
            self._test_check_pattern(metrics.get("checks", [])),
        ]

        passed = sum(1 for t in tests if t.passed)
        total = len(tests)
        score = (passed / total) * 100 if total > 0 else 0.0

        if score >= 80:
            rec = "推荐"
        elif score >= 60:
            rec = "谨慎"
        elif score >= 40:
            rec = "需改进"
        else:
            rec = "不推荐"

        result = AntiOverfitResult(
            score=score,
            recommendation=rec,
            tests=tests,
            passed_count=passed,
            total_count=total,
        )

        level = logger.info if result.score >= 60 else logger.warning
        level(
            "[ANTI-FIT] 过拟合检测结果: score=%.0f/%d rec=%s 通过=%d",
            result.score,
            result.total_count,
            result.recommendation,
            result.passed_count,
        )

        return result

    def _test_sharpe_consistency(self, sharpe: Optional[float]) -> TestResult:
        """Test 1: Sharpe ratio consistency with historical average.

        If current Sharpe deviates > 2σ from historical mean → potential overfitting.
        """
        if sharpe is None or not isinstance(sharpe, (int, float)):
            return TestResult("Sharpe一致性", False, {"error": "缺少Sharpe数据"})

        sharpe = float(sharpe)

        if len(self._historical_sharpes) < 3:
            logger.debug("[ANTI-FIT] 历史Sharpe不足3条，跳过一致性检验")
            return TestResult("Sharpe一致性", True, {
                "current_sharpe": round(sharpe, 4),
                "note": "历史数据不足，默认通过",
            })

        valid = [s for s in self._historical_sharpes if isinstance(s, (int, float)) and math.isfinite(s)]
        if len(valid) < 3:
            return TestResult("Sharpe一致性", True, {
                "current_sharpe": round(sharpe, 4),
                "note": "有效历史数据不足",
            })

        mean_s = sum(valid) / len(valid)
        variance = sum((s - mean_s) ** 2 for s in valid) / len(valid)
        std_dev = math.sqrt(variance) if variance > 0 else 0.01

        deviation = abs(sharpe - mean_s) / std_dev if std_dev > 0 else 0.0
        threshold = 2.0

        passed = deviation <= threshold

        logger.debug(
            "[ANTI-FIT] Sharpe一致性: current=%.4f mean=%.4f std=%.4f deviation=%.2fσ",
            sharpe, mean_s, std_dev, deviation,
        )

        return TestResult("Sharpe一致性", passed, {
            "current_sharpe": round(sharpe, 4),
            "historical_mean": round(mean_s, 4),
            "historical_std": round(std_dev, 4),
            "deviation_sigma": round(deviation, 2),
            "threshold_sigma": threshold,
        })

    def _test_turnover_sanity(self, turnover: Optional[float]) -> TestResult:
        """Test 2: Turnover rate sanity check.

        Rules:
          TO < 1%   → over-smoothed (signal too static)
          TO > 70%  → overfitted to noise
          Ideal:    15%-50%
        """
        if turnover is None or not isinstance(turnover, (int, float)):
            return TestResult("换手率合理性", False, {"error": "缺少换手率数据"})

        turnover = float(turnover)

        if turnover <= 0:
            return TestResult("换手率合理性", False, {
                "turnover": turnover,
                "reason": "换手率为零或负数",
            })

        if turnover > 1.0:
            turnover_pct = turnover
        else:
            turnover_pct = turnover * 100

        if turnover_pct < 1.0:
            passed = False
            reason = "过低（信号过于静态，可能过平滑）"
        elif turnover_pct > 70.0:
            passed = False
            reason = "过高（可能拟合噪声）"
        elif 15.0 <= turnover_pct <= 50.0:
            passed = True
            reason = "理想范围"
        elif 1.0 <= turnover_pct < 15.0:
            passed = True
            reason = "偏低但可接受"
        else:
            passed = True
            reason = "偏高但可接受"

        logger.debug("[ANTI-FIT] 换手率检查: %.2f%% → %s (%s)", turnover_pct, "PASS" if passed else "FAIL", reason)

        return TestResult("换手率合理性", passed, {
            "turnover_percent": round(turnover_pct, 2),
            "reason": reason,
        })

    def _test_fitness_efficiency(
        self,
        fitness: Optional[float],
        sharpe: Optional[float],
        turnover: Optional[float],
    ) -> TestResult:
        """Test 3: Fitness efficiency analysis.

        Fitness = Sharpe * sqrt(|Returns|) / max(TO, 0.125)

        Anomalously high fitness with mediocre Sharpe suggests potential data leakage.
        """
        if fitness is None or not isinstance(fitness, (int, float)):
            return TestResult("Fitness效率", False, {"error": "缺少Fitness数据"})

        fitness = float(fitness)
        sharpe_val = float(sharpe) if sharpe is not None and isinstance(sharpe, (int, float)) else 0.0
        turnover_val = float(turnover) if turnover is not None and isinstance(turnover, (int, float)) else 0.3

        if turnover_val <= 0:
            turnover_val = 0.125

        if turnover_val > 1.0:
            turnover_normalized = turnover_val / 100.0
        else:
            turnover_normalized = turnover_val

        effective_to = max(turnover_normalized, 0.125)

        expected_fitness_lower = abs(sharpe_val) * 0.5 / effective_to if effective_to > 0 else 0.0
        expected_fitness_upper = abs(sharpe_val) * 3.0 / effective_to if effective_to > 0 else 10.0

        suspicious = (
            fitness > expected_fitness_upper * 1.5
            and abs(sharpe_val) < 1.0
        )

        passed = not suspicious

        logger.debug(
            "[ANTI-FIT] Fitness效率: fitness=%.4f sharpe=%.4f TO=%.4f expected=[%.4f, %.4f] suspicious=%s",
            fitness, sharpe_val, turnover_val, expected_fitness_lower, expected_fitness_upper, suspicious,
        )

        return TestResult("Fitness效率", passed, {
            "fitness": round(fitness, 4),
            "sharpe": round(sharpe_val, 4),
            "turnover": round(turnover_val, 4),
            "expected_range": [round(expected_fitness_lower, 4), round(expected_fitness_upper, 4)],
            "suspicious": suspicious,
        })

    def _test_drawdown_stability(
        self,
        drawdown: Optional[float],
        sharpe: Optional[float],
    ) -> TestResult:
        """Test 4: Drawdown stability check.

        Rules:
          Drawdown > 25% → unstable
          Drawdown < 5% but high Sharpe (>2.0) → suspicious
        """
        if drawdown is None:
            logger.debug("[ANTI-FIT] 无回撤数据，跳过回撤稳定性检验")
            return TestResult("回撤稳定性", True, {
                "note": "无回撤数据，默认通过",
            })

        if not isinstance(drawdown, (int, float)):
            return TestResult("回撤稳定性", False, {"error": "回撤数据格式错误"})

        drawdown = float(abs(drawdown))
        sharpe_val = float(sharpe) if sharpe is not None and isinstance(sharpe, (int, float)) else 0.0

        warnings = []

        if drawdown > 0.25:
            warnings.append(f"回撤{drawdown:.1%}过高")
        elif drawdown > 0.15:
            warnings.append(f"回撤{drawdown:.1%}偏高")

        if drawdown < 0.05 and abs(sharpe_val) > 2.0:
            warnings.append("低回撤+高Sharpe组合可疑")

        passed = len(warnings) == 0

        logger.debug(
            "[ANTI-FIT] 回撤稳定性: dd=%.4f sharpe=%.4f warnings=%d",
            drawdown, sharpe_val, len(warnings),
        )

        return TestResult("回撤稳定性", passed, {
            "drawdown": round(drawdown, 4),
            "sharpe": round(sharpe_val, 4),
            "warnings": warnings,
        })

    def _test_check_pattern(self, checks: Optional[list]) -> TestResult:
        """Test 5: Failed-check pattern analysis.

        Pattern detection:
          LOW_SHARPE + LOW_FITNESS simultaneously → fundamental issue
          HIGH_TURNOVER + LOW_SHARPE → signal direction may be wrong
        """
        if not checks or not isinstance(checks, list):
            logger.debug("[ANTI-FIT] 无checks数据，跳过模式分析")
            return TestResult("检查模式分析", True, {
                "note": "无检查数据，默认通过",
            })

        failed_names = set()
        for ck in checks:
            if not isinstance(ck, dict):
                continue
            result = ck.get("result")
            name = ck.get("name", "")
            if result is False or str(result).lower() in ("false", "fail", "failed"):
                failed_names.add(name.upper() if isinstance(name, str) else str(name))

        patterns = []
        severity = "none"

        if "LOW_SHARPE" in failed_names and "LOW_FITNESS" in failed_names:
            patterns.append("LOW_SHARPE+LOW_FITNESS: 基本面问题")
            severity = "high"
        if "HIGH_TURNOVER" in failed_names and ("LOW_SHARPE" in failed_names or "NEGATIVE_SHARPE" in failed_names):
            patterns.append("HIGH_TURNOVER+LOW_SHARPE: 信号方向可能错误")
            severity = "medium" if severity != "high" else severity
        if len(failed_names) >= 3:
            patterns.append(f"多项失败({len(failed_names)}项): 因子质量堪忧")
            severity = "high"

        passed = severity == "none"

        logger.debug(
            "[ANTI-FIT] 检查模式: failed=%d patterns=%d severity=%s",
            len(failed_names), len(patterns), severity,
        )

        return TestResult("检查模式分析", passed, {
            "failed_checks": sorted(failed_names),
            "patterns": patterns,
            "severity": severity,
        })


class FullAntiOverfitDetector:
    """Full anti-overfit detector wrapping QuantGPT's original implementation.

    Requires pandas DataFrame with columns:
      trade_date, stock_code, factor_value, daily_ret

    This is an optional advanced mode for when local backtest data is available.
    Falls back gracefully if pandas is not installed.
    """

    def __init__(self, factor_df: Any = None, holding_period: int = 5):
        """Initialize the full detector.

        Args:
            factor_df: pandas DataFrame with trade_date, stock_code, factor_value, daily_ret.
                       If None, detector must be initialized later via init_from_dataframe().
            holding_period: Holding period in trading days.
        """
        self._detector = None
        self._import_error: Optional[str] = None
        self._holding_period = holding_period

        if factor_df is not None:
            self.init_from_dataframe(factor_df, holding_period)

    def init_from_dataframe(self, factor_df: Any, holding_period: int = 5) -> None:
        """Initialize the underlying QuantGPT detector from a DataFrame."""
        try:
            import numpy as np
            import pandas as pd
            from scipy import stats as sp_stats
            from scipy.optimize import curve_fit

            self._detector = _QuantGPTAntiOverfitDetector(factor_df, holding_period)
            self._import_error = None
            logger.info("[ANTI-FIT] 完整版检测器初始化成功 (pandas+scipy)")
        except ImportError as e:
            self._import_error = f"Missing dependency: {e}"
            self._detector = None
            logger.warning("[ANTI-FIT] 完整版检测器不可用: %s", self._import_error)
        except (ValueError, TypeError, RuntimeError) as e:
            self._import_error = f"Initialization error: {e}"
            self._detector = None
            logger.error("[ANTI-FIT] 完整版检测器初始化失败: %s", e)

    @property
    def available(self) -> bool:
        """Check if the full detector is available (dependencies loaded)."""
        return self._detector is not None

    @property
    def import_error(self) -> Optional[str]:
        """Return import error message if any."""
        return self._import_error

    def run_all(self) -> AntiOverfitResult:
        """Run all 4 QuantGPT anti-overfit tests.

        Returns:
            AntiOverfitResult with composite score.

        Raises:
            RuntimeError: If detector is not initialized or dependencies missing.
        """
        if self._detector is None:
            raise RuntimeError(
                f"FullAntiOverfitDetector not available: "
                f"{self._import_error or 'Not initialized. Call init_from_dataframe() first.'}"
            )

        qgpt_result = self._detector.run_all()

        return AntiOverfitResult(
            score=qgpt_result.score,
            recommendation=qgpt_result.recommendation,
            tests=qgpt_result.tests,
            passed_count=qgpt_result.passed_count,
            total_count=qgpt_result.total_count,
        )


class _QuantGPTAntiOverfitDetector:
    """Internal wrapper of QuantGPT's original AntiOverfitDetector.

    This class contains the original QuantGPT implementation adapted for
    integration into OpenAlpha-Brain. It requires pandas, numpy, and scipy.
    """

    def __init__(self, factor_df: Any, holding_period: int = 5):
        import numpy as np
        import pandas as pd
        from scipy import stats as sp_stats
        from scipy.optimize import curve_fit

        if not hasattr(factor_df, 'copy') or not hasattr(factor_df, 'columns'):
            raise ValueError("factor_df must be a pandas DataFrame")

        self.df = factor_df.copy()
        self.df["trade_date"] = pd.to_datetime(self.df["trade_date"])
        self.holding_period = holding_period
        self._prepare_forward_returns()

    def _prepare_forward_returns(self):
        import pandas as pd
        self.df = self.df.sort_values(["stock_code", "trade_date"])
        self.df["fwd_ret"] = (
            self.df.groupby("stock_code")["daily_ret"]
            .transform(
                lambda s: s.shift(-1)
                .rolling(self.holding_period, min_periods=self.holding_period)
                .sum()
                .shift(-(self.holding_period - 1))
            )
        )

    def _calc_daily_ic(self, df=None):
        import numpy as np
        import pandas as pd
        from scipy import stats as sp_stats

        data = df if df is not None else self.df
        valid = data.dropna(subset=["factor_value", "fwd_ret"])
        if valid.empty:
            return pd.Series(dtype=float)

        def _spearman(g):
            if len(g) < 5 or g["factor_value"].nunique() < 2:
                return np.nan
            corr, _ = sp_stats.spearmanr(g["factor_value"], g["fwd_ret"])
            return corr if not np.isnan(corr) else 0.0

        return valid.groupby("trade_date").apply(_spearman).dropna()

    def run_all(self) -> AntiOverfitResult:
        tests = [
            self.test_ic_stability(),
            self.test_subsample_stress(),
            self.test_placebo(),
            self.test_half_life(),
        ]
        passed = sum(1 for t in tests if t.passed)
        score = (passed / 4) * 100

        if score >= 80:
            rec = "推荐"
        elif score >= 60:
            rec = "谨慎"
        elif score >= 40:
            rec = "需改进"
        else:
            rec = "不推荐"

        return AntiOverfitResult(
            score=score,
            recommendation=rec,
            tests=tests,
            passed_count=passed,
            total_count=4,
        )

    def test_ic_stability(self) -> TestResult:
        import numpy as np

        ic_series = self._calc_daily_ic()
        if len(ic_series) < 20:
            return TestResult("IC稳定性", False, {"error": "IC数据不足"})

        ic_mean = float(ic_series.mean())
        positive_rate = float((ic_series > 0).sum() / len(ic_series))

        yearly_ic = ic_series.groupby(ic_series.index.year).mean()
        overall_sign = np.sign(ic_mean)
        yearly_signs = np.sign(yearly_ic.values)
        has_reversal = bool(np.any(yearly_signs != overall_sign)) if overall_sign != 0 else True

        passed = (positive_rate >= 0.55) and (abs(ic_mean) >= 0.02) and (not has_reversal)

        return TestResult("IC稳定性", passed, {
            "ic_mean": round(ic_mean, 4),
            "positive_rate": round(positive_rate, 4),
            "yearly_ic": {str(y): round(float(v), 4) for y, v in yearly_ic.items()},
            "has_reversal": has_reversal,
        })

    def test_subsample_stress(self) -> TestResult:
        import numpy as np
        import pandas as pd

        ic_series = self._calc_daily_ic()
        if len(ic_series) < 40:
            return TestResult("子样本压力", False, {"error": "数据不足"})

        overall_sign = np.sign(ic_series.mean())
        if overall_sign == 0:
            return TestResult("子样本压力", False, {"error": "整体IC为零"})

        market_ret = self.df.groupby("trade_date")["daily_ret"].mean()
        market_ret = market_ret.reindex(ic_series.index).fillna(0)

        cum_ret_60 = market_ret.rolling(60, min_periods=30).sum()
        volatility_60 = market_ret.rolling(60, min_periods=30).std()

        sub_ics = {}
        bull_mask = cum_ret_60 > 0.05
        bear_mask = cum_ret_60 < -0.05
        sideways_mask = ~bull_mask & ~bear_mask

        for name, mask in [("bull", bull_mask), ("bear", bear_mask), ("sideways", sideways_mask)]:
            aligned_mask = mask.reindex(ic_series.index).fillna(False)
            sub = ic_series[aligned_mask]
            if len(sub) >= 10:
                sub_ics[name] = float(sub.mean())

        vol_median = volatility_60.median()
        high_vol = volatility_60 > vol_median
        low_vol = ~high_vol
        for name, mask in [("high_vol", high_vol), ("low_vol", low_vol)]:
            aligned_mask = mask.reindex(ic_series.index).fillna(False)
            sub = ic_series[aligned_mask]
            if len(sub) >= 10:
                sub_ics[name] = float(sub.mean())

        if len(sub_ics) == 0:
            return TestResult("子样本压力", False, {"error": "子样本划分失败"})

        same_sign_count = sum(1 for v in sub_ics.values() if np.sign(v) == overall_sign)
        consistency = same_sign_count / len(sub_ics)
        passed = consistency >= 0.6

        return TestResult("子样本压力", passed, {
            "overall_ic_sign": int(overall_sign),
            "sub_sample_ics": {k: round(v, 4) for k, v in sub_ics.items()},
            "consistency": round(consistency, 4),
        })

    def test_placebo(self, n_permutations: int = 20) -> TestResult:
        import numpy as np
        import pandas as pd

        ic_series = self._calc_daily_ic()
        if len(ic_series) < 20:
            return TestResult("安慰剂检验", False, {"error": "IC数据不足"})

        real_ic = float(ic_series.mean())

        perm_ics = []
        rng = np.random.RandomState(42)
        valid = self.df.dropna(subset=["factor_value", "fwd_ret"])
        sampled_dates = sorted(valid["trade_date"].unique())[::5]
        valid_sampled = valid[valid["trade_date"].isin(sampled_dates)]
        for _ in range(n_permutations):
            shuffled = valid_sampled.copy()
            shuffled["factor_value"] = shuffled.groupby("trade_date")["factor_value"].transform(
                lambda s: s.sample(frac=1, random_state=rng).values
            )
            perm_ic = self._calc_daily_ic(shuffled)
            if len(perm_ic) > 0:
                perm_ics.append(float(perm_ic.mean()))

        if len(perm_ics) < 10:
            return TestResult("安慰剂检验", False, {"error": "置换检验数据不足"})

        perm_95 = float(np.percentile(perm_ics, 95))
        perm_pass = abs(real_ic) > abs(perm_95)

        shift_ics = {}
        for shift in [5, 10, 20]:
            shifted = self.df.copy()
            shifted["factor_value"] = shifted.groupby("stock_code")["factor_value"].shift(shift)
            shift_ic = self._calc_daily_ic(shifted)
            if len(shift_ic) > 0:
                shift_ics[shift] = float(shift_ic.mean())

        decay_ok = True
        if shift_ics:
            for shift_val in shift_ics.values():
                if abs(shift_val) >= abs(real_ic):
                    decay_ok = False
                    break

        passed = perm_pass and decay_ok

        return TestResult("安慰剂检验", passed, {
            "real_ic": round(real_ic, 4),
            "perm_95th": round(perm_95, 4),
            "perm_pass": perm_pass,
            "shift_ics": {str(k): round(v, 4) for k, v in shift_ics.items()},
            "decay_ok": decay_ok,
        })

    def test_half_life(self) -> TestResult:
        import numpy as np
        import pandas as pd
        from scipy.optimize import curve_fit

        periods = [1, 2, 5, 10, 20, 40]
        period_ics = {}

        valid = self.df.dropna(subset=["factor_value"]).copy()
        valid = valid.sort_values(["stock_code", "trade_date"])
        sampled_dates = sorted(valid["trade_date"].unique())[::3]
        valid = valid[valid["trade_date"].isin(sampled_dates)]

        for p in periods:
            valid[f"fwd_ret_{p}"] = (
                valid.groupby("stock_code")["daily_ret"]
                .transform(
                    lambda s: s.shift(-1)
                    .rolling(p, min_periods=p)
                    .sum()
                    .shift(-(p - 1))
                )
            )
            sub = valid.dropna(subset=["factor_value", f"fwd_ret_{p}"])
            if sub.empty:
                continue

            def _spearman_p(g, col=f"fwd_ret_{p}"):
                from scipy import stats as sp_stats
                if len(g) < 5 or g["factor_value"].nunique() < 2:
                    return np.nan
                corr, _ = sp_stats.spearmanr(g["factor_value"], g[col])
                return corr if not np.isnan(corr) else 0.0

            ic_s = sub.groupby("trade_date").apply(_spearman_p).dropna()
            if len(ic_s) > 0:
                period_ics[p] = abs(float(ic_s.mean()))

        if len(period_ics) < 3:
            return TestResult("半衰期估计", False, {"error": "前瞻期IC数据不足"})

        x = np.array(list(period_ics.keys()), dtype=float)
        y = np.array(list(period_ics.values()), dtype=float)

        try:
            def exp_decay(t, a, b):
                return a * np.exp(-b * t)

            popt, _ = curve_fit(exp_decay, x, y, p0=[y[0], 0.05], maxfev=5000)
            a, b = popt
            half_life = float(np.log(2) / b) if b > 0 else 999.0
        except (ValueError, TypeError, RuntimeError):
            if len(period_ics) >= 2:
                sorted_p = sorted(period_ics.items())
                ic_first = sorted_p[0][1]
                ic_last = sorted_p[-1][1]
                t_span = sorted_p[-1][0] - sorted_p[0][0]
                if ic_first > 0 and ic_last > 0 and ic_last < ic_first:
                    b_est = np.log(ic_first / ic_last) / t_span
                    half_life = float(np.log(2) / b_est) if b_est > 0 else 999.0
                else:
                    half_life = 999.0
            else:
                half_life = 0.0

        passed = half_life > 5.0

        return TestResult("半衰期估计", passed, {
            "half_life_days": round(half_life, 1),
            "period_ics": {str(k): round(v, 4) for k, v in period_ics.items()},
        })
