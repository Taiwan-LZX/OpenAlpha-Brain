from __future__ import annotations

import contextlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_STATE_PATH = Path(__file__).resolve().parent / "market_state.json"

_garch_cache: dict[str, tuple] = {}


@dataclass
class MarketState:
    """[Brief description of class purpose.]"""

    year: int = 0
    momentum_sharpe: float = 0.0
    mean_reversion_sharpe: float = 0.0
    volatility_sharpe: float = 0.0
    value_sharpe: float = 0.0
    quality_sharpe: float = 0.0
    liquidity_sharpe: float = 0.0
    size_sharpe: float = 0.0
    dominant_strategy: str = ""

    def to_dict(self) -> dict:
        """[Brief description of function purpose.]

        Returns:
            dict: [Description]
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> MarketState:
        """[Brief description of function purpose.]

        Args:
            d (dict): [Description]

        Returns:
            MarketState: [Description]
        """
        return cls(
            year=d.get("year", 0),
            momentum_sharpe=d.get("momentum_sharpe", 0.0),
            mean_reversion_sharpe=d.get("mean_reversion_sharpe", 0.0),
            volatility_sharpe=d.get("volatility_sharpe", 0.0),
            value_sharpe=d.get("value_sharpe", 0.0),
            quality_sharpe=d.get("quality_sharpe", 0.0),
            liquidity_sharpe=d.get("liquidity_sharpe", 0.0),
            size_sharpe=d.get("size_sharpe", 0.0),
            dominant_strategy=d.get("dominant_strategy", ""),
        )

    def _all_sharpes(self) -> dict[str, float]:
        """[Brief description of function purpose.]

        Returns:
            dict[str, float]: [Description]
        """
        return {
            "momentum": self.momentum_sharpe,
            "mean_reversion": self.mean_reversion_sharpe,
            "volatility": self.volatility_sharpe,
            "value": self.value_sharpe,
            "quality": self.quality_sharpe,
            "liquidity": self.liquidity_sharpe,
            "size": self.size_sharpe,
        }

    def _determine_dominant(self) -> str:
        """[Brief description of function purpose.]

        Returns:
            str: [Description]
        """
        sharpes = self._all_sharpes()
        best = max(sharpes, key=lambda k: float(sharpes[k]))
        return best if sharpes[best] > 0.0 else ""


class MarketStateInferencer:
    """[Brief description of class purpose.]"""

    def __init__(self, path: str | Path | None = None):
        """[Brief description of function purpose.]

        Args:
            path (str | Path | None): [Description]
        """
        self._path = Path(path) if path else _STATE_PATH
        self._yearly_states: dict[int, MarketState] = {}
        self._direction_sharpes: dict[str, list[float]] = {}
        self._accumulated_results: list[dict] = []
        self._last_update_time: float = 0.0
        self._load()
        self._ensure_default_state()

    def _ensure_default_state(self) -> None:
        if self._yearly_states and self._direction_sharpes:
            return
        import datetime as _dt

        _current_year = _dt.date.today().year
        _default = MarketState(
            year=_current_year,
            momentum_sharpe=0.8,
            mean_reversion_sharpe=0.4,
            volatility_sharpe=0.3,
            value_sharpe=0.5,
            quality_sharpe=0.6,
            liquidity_sharpe=0.2,
            size_sharpe=0.3,
            dominant_strategy="momentum",
        )
        self._yearly_states[_current_year] = _default
        self._direction_sharpes = {
            "momentum": [0.8, 1.0, 0.6, 0.9],
            "mean_reversion": [0.4, 0.3, 0.5],
            "value": [0.5, 0.4, 0.6],
            "quality": [0.6, 0.7, 0.5],
            "low_volatility": [0.3, 0.4],
            "cross_sectional": [0.5, 0.6],
            "temporal": [0.4, 0.5],
        }
        logger.info("MarketStateInferencer: initialized with default academic priors for year %d", _current_year)

    def _load(self) -> None:
        """[Brief description of function purpose.]

        Returns:
            None: [Description]
        """
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._yearly_states = {int(k): MarketState.from_dict(v) for k, v in data.get("yearly_states", {}).items()}
            self._direction_sharpes = dict(data.get("direction_sharpes", {}).items())
            logger.info(
                "MarketStateInferencer: loaded %d yearly states, %d directions from %s",
                len(self._yearly_states),
                len(self._direction_sharpes),
                self._path,
            )
        except (ValueError, TypeError, OSError):
            logger.warning("MarketStateInferencer: failed to load from %s", self._path, exc_info=True)
            self._yearly_states = {}
            self._direction_sharpes = {}

    def _save(self) -> None:
        """[Brief description of function purpose.]

        Returns:
            None: [Description]
        """
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "yearly_states": {str(k): v.to_dict() for k, v in self._yearly_states.items()},
                "direction_sharpes": self._direction_sharpes,
            }
            self._path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            logger.warning("MarketStateInferencer: failed to save", exc_info=True)

    def infer_from_brain_results(self, brain_results: list[dict], yearly_breakdown: list[dict] | None = None) -> dict:
        """[Brief description of function purpose.]

        Args:
            brain_results (list[dict]): [Description]
            yearly_breakdown (list[dict] | None): [Description]

        Returns:
            dict: [Description]
        """
        self._accumulated_results.extend(brain_results)
        self._last_update_time = time.time()

        direction_sharpes: dict[str, list[float]] = {}
        for result in brain_results:
            direction = result.get("direction", "")
            sharpe = result.get("sharpe")
            if direction and sharpe is not None:
                direction_sharpes.setdefault(direction, []).append(sharpe)

            yearly_data = result.get("yearly_breakdown")
            if yearly_data and isinstance(yearly_data, list):
                for yd in yearly_data:
                    year = yd.get("year", 0)
                    if year:
                        ms = self._yearly_states.get(year, MarketState(year=year))
                        strat = yd.get("strategy", "")
                        y_sharpe = yd.get("sharpe", 0.0)
                        if not isinstance(y_sharpe, (int, float)) or not math.isfinite(y_sharpe):
                            y_sharpe = 0.0
                        key = self._direction_to_sharpe_key(strat)
                        if key:
                            self._set_sharpe_by_key(ms, key, y_sharpe)
                        self._yearly_states[year] = ms

        if yearly_breakdown and isinstance(yearly_breakdown, list):
            for yd in yearly_breakdown:
                year = yd.get("year", 0)
                if not year:
                    continue
                ms = self._yearly_states.get(year, MarketState(year=year))
                direction = yd.get("direction", "")
                y_sharpe = yd.get("sharpe", 0.0)
                if not isinstance(y_sharpe, (int, float)) or not math.isfinite(y_sharpe):
                    y_sharpe = 0.0
                key = self._direction_to_sharpe_key(direction)
                if key:
                    self._set_sharpe_by_key(ms, key, y_sharpe)
                else:
                    all_zero = all(v == 0.0 for v in ms._all_sharpes().values())
                    if all_zero:
                        ms.momentum_sharpe = y_sharpe
                self._yearly_states[year] = ms

        for direction, sharpes in direction_sharpes.items():
            self._direction_sharpes.setdefault(direction, []).extend(sharpes)

        friendly_years: dict[str, list[int]] = {}
        for year, ms in self._yearly_states.items():
            ms.dominant_strategy = ms._determine_dominant()
            if ms.dominant_strategy:
                friendly_years.setdefault(ms.dominant_strategy, []).append(year)

        avg_sharpes: dict[str, float] = {}
        for d, sharpes in self._direction_sharpes.items():
            if sharpes:
                avg_sharpes[d] = sum(sharpes) / len(sharpes)

        current_dominant = ""
        if avg_sharpes:
            current_dominant = max(avg_sharpes, key=avg_sharpes.get)

        self._save()

        result = {
            "current_dominant": current_dominant,
            "avg_sharpes_by_direction": avg_sharpes,
        }
        for strat, years in friendly_years.items():
            result[f"{strat}_friendly_years"] = sorted(years)

        return result

    @staticmethod
    def _direction_to_sharpe_key(direction: str) -> str:
        """[Brief description of function purpose.]

        Args:
            direction (str): [Description]

        Returns:
            str: [Description]
        """
        d = direction.lower()
        if "mean_reversion" in d or "reversion" in d:
            return "mean_reversion"
        if "momentum" in d:
            return "momentum"
        if "volatility" in d:
            return "volatility"
        if "value" in d:
            return "value"
        if "quality" in d:
            return "quality"
        if "liquidity" in d:
            return "liquidity"
        if "size" in d:
            return "size"
        return ""

    @staticmethod
    def _set_sharpe_by_key(ms: MarketState, key: str, sharpe: float) -> None:
        """[Brief description of function purpose.]

        Args:
            ms (MarketState): [Description]
            key (str): [Description]
            sharpe (float): [Description]

        Returns:
            None: [Description]
        """
        if key == "momentum":
            ms.momentum_sharpe = max(ms.momentum_sharpe, sharpe)
        elif key == "mean_reversion":
            ms.mean_reversion_sharpe = max(ms.mean_reversion_sharpe, sharpe)
        elif key == "volatility":
            ms.volatility_sharpe = max(ms.volatility_sharpe, sharpe)
        elif key == "value":
            ms.value_sharpe = max(ms.value_sharpe, sharpe)
        elif key == "quality":
            ms.quality_sharpe = max(ms.quality_sharpe, sharpe)
        elif key == "liquidity":
            ms.liquidity_sharpe = max(ms.liquidity_sharpe, sharpe)
        elif key == "size":
            ms.size_sharpe = max(ms.size_sharpe, sharpe)

    def infer_from_yearly_data(self, yearly_data: list[dict], direction: str) -> dict:
        """[Brief description of function purpose.]

        Args:
            yearly_data (list[dict]): [Description]
            direction (str): [Description]

        Returns:
            dict: [Description]
        """
        if not yearly_data:
            return {"years_updated": 0, "direction": direction}

        self._last_update_time = time.time()

        years_updated = 0
        for yd in yearly_data:
            year = yd.get("year", 0)
            if not year:
                continue
            sharpe = yd.get("sharpe", 0.0)
            if not isinstance(sharpe, (int, float)) or not math.isfinite(sharpe):
                sharpe = 0.0
            ms = self._yearly_states.get(year, MarketState(year=year))

            key = self._direction_to_sharpe_key(direction)
            if key:
                self._set_sharpe_by_key(ms, key, sharpe)
            else:
                all_zero = all(v == 0.0 for v in ms._all_sharpes().values())
                if all_zero:
                    ms.momentum_sharpe = max(ms.momentum_sharpe, sharpe)

            ms.dominant_strategy = ms._determine_dominant()

            self._yearly_states[year] = ms
            years_updated += 1

        self._direction_sharpes.setdefault(direction, [])
        for yd in yearly_data:
            sharpe = yd.get("sharpe")
            if sharpe is not None:
                self._direction_sharpes[direction].append(sharpe)

        self._save()

        logger.info(
            "MarketStateInferencer: infer_from_yearly_data updated %d years for direction=%s",
            years_updated,
            direction,
        )

        return {"years_updated": years_updated, "direction": direction}

    def adjust_mab_bias(self, mab_explorer, current_direction: str = "") -> None:
        """[Brief description of function purpose.]

        Args:
            mab_explorer: [Description]
            current_direction (str): [Description]

        Returns:
            None: [Description]
        """
        if mab_explorer is None:
            return

        avg_sharpes: dict[str, float] = {}
        for d, sharpes in self._direction_sharpes.items():
            if sharpes:
                avg_sharpes[d] = sum(sharpes) / len(sharpes)

        current_dominant = ""
        if avg_sharpes:
            current_dominant = max(avg_sharpes, key=avg_sharpes.get)

        if not current_dominant:
            return

        max_sharpe = max(avg_sharpes.values()) if avg_sharpes else 0.0
        if max_sharpe <= 0:
            return

        direction_boost: dict[str, float] = {}
        for d, avg_s in avg_sharpes.items():
            direction_boost[d] = avg_s / max_sharpe

        trend_adjustment = self._compute_yearly_trend_adjustment()
        for d, weight in direction_boost.items():
            adjusted = weight + trend_adjustment.get(d, 0.0)
            adjusted = max(0.1, min(adjusted, 1.5))
            with contextlib.suppress(OSError, ValueError, RuntimeError):
                mab_explorer.set_initial_bias(d, adjusted)

    def _compute_yearly_trend_adjustment(self) -> dict[str, float]:
        """[Brief description of function purpose.]

        Returns:
            dict[str, float]: [Description]
        """
        adjustment: dict[str, float] = {}
        if len(self._yearly_states) < 2:
            return adjustment

        sorted_years = sorted(self._yearly_states.keys())
        recent_years = sorted_years[-3:]

        _sharpe_keys = [
            "momentum",
            "mean_reversion",
            "volatility",
            "value",
            "quality",
            "liquidity",
            "size",
        ]

        for key in _sharpe_keys:
            sharpes = [
                getattr(self._yearly_states[y], f"{key}_sharpe", 0.0) for y in recent_years if y in self._yearly_states
            ]
            if len(sharpes) >= 3:
                if key == "momentum":
                    declining = all(sharpes[i] > sharpes[i + 1] for i in range(len(sharpes) - 1))
                    if declining:
                        adjustment[key] = -0.15
                        logger.info(
                            "MarketStateInferencer: %s Sharpe declining for %d years → bias -0.15",
                            key,
                            len(recent_years),
                        )
                elif key == "mean_reversion":
                    rising = all(sharpes[i] < sharpes[i + 1] for i in range(len(sharpes) - 1))
                    if rising:
                        adjustment[key] = 0.15
                        logger.info(
                            "MarketStateInferencer: %s Sharpe rising for %d years → bias +0.15", key, len(recent_years)
                        )
                else:
                    declining = all(sharpes[i] > sharpes[i + 1] for i in range(len(sharpes) - 1))
                    if declining:
                        adjustment[key] = -0.10
                    rising = all(sharpes[i] < sharpes[i + 1] for i in range(len(sharpes) - 1))
                    if rising:
                        adjustment[key] = adjustment.get(key, 0.0) + 0.10

        return adjustment

    def health_check(self) -> dict:
        """[Brief description of function purpose.]

        Returns:
            dict: [Description]
        """
        return {
            "module": "MarketStateInferencer",
            "status": "active",
            "last_update_time": self._last_update_time,
            "directions_tracked": len(self._direction_sharpes),
            "yearly_states_count": len(self._yearly_states),
        }

    def get_market_state_summary(self) -> dict:
        """[Brief description of function purpose.]

        Returns:
            dict: [Description]
        """
        friendly_years: dict[str, list[int]] = {}
        for year, ms in self._yearly_states.items():
            if ms.dominant_strategy:
                friendly_years.setdefault(ms.dominant_strategy, []).append(year)

        avg_sharpes: dict[str, float] = {}
        for d, sharpes in self._direction_sharpes.items():
            if sharpes:
                avg_sharpes[d] = sum(sharpes) / len(sharpes)

        current_dominant = ""
        if avg_sharpes:
            current_dominant = max(avg_sharpes, key=lambda k: float(avg_sharpes[k]))

        result: dict = {
            "current_dominant": current_dominant,
            "avg_sharpes_by_direction": avg_sharpes,
        }
        for strat, years in friendly_years.items():
            result[f"{strat}_friendly_years"] = sorted(years)

        return result

    @property
    def yearly_states(self) -> dict[int, MarketState]:
        """Return a defensive copy of yearly market states.

        Maps each year to its MarketState (dominant strategy, direction sharpes).
        Useful for analysing regime shifts and long-term strategy performance trends.
        """
        return dict(self._yearly_states)

    def infer_current_regime(self, returns_20d: list[float] | None = None) -> str:
        """Infer the current market regime based on accumulated state data.

        Returns one of: 'high_volatility', 'trending', 'low_volatility',
        'crash_risk', or 'unknown'.

        The inference uses a rule-based heuristic over recent direction Sharpes:
          - high_volatility: volatility Sharpe > 0.6 OR mean_reversion dominant
          - trending: momentum Sharpe > 1.0 AND rising trend
          - low_volatility: all Sharpes < 0.5 AND quality/value dominant
          - crash_risk: momentum collapsing (< 0.2) with high vol dominance
          - unknown: insufficient data or unclear signal

        AE-3 Enhancement: GARCH(1,1) volatility clustering detection.
        If returns_20d is provided and GARCH detects volatility clustering,
        upgrade regime to 'crash_risk' when already in high-volatility/trending state.
        """
        avg_sharpes: dict[str, float] = {}
        for d, sharpes in self._direction_sharpes.items():
            if sharpes:
                avg_sharpes[d] = sum(sharpes) / len(sharpes)

        if not avg_sharpes:
            return "unknown"

        vol_sharpe = avg_sharpes.get("volatility", 0.0)
        mom_sharpe = avg_sharpes.get("momentum", 0.0)
        mr_sharpe = avg_sharpes.get("mean_reversion", 0.0)
        qual_sharpe = avg_sharpes.get("quality", 0.0)
        val_sharpe = avg_sharpes.get("value", 0.0)

        # Crash risk: momentum very weak + volatility elevated
        if mom_sharpe < 0.2 and vol_sharpe > 0.5:
            regime = "crash_risk"
        # High volatility regime
        elif vol_sharpe > 0.6 or mr_sharpe > 0.8:
            regime = "high_volatility"
        # Trending regime: strong momentum
        elif mom_sharpe > 1.0:
            regime = "trending"
        # Low volatility: modest Sharpes across board, quality/value lead
        else:
            max_sharpe = max(avg_sharpes.values()) if avg_sharpes else 0.0
            if max_sharpe < 0.7 and (qual_sharpe >= val_sharpe >= mr_sharpe):
                regime = "low_volatility"
            else:
                regime = "unknown"

        # ── AE-3: GARCH Volatility Clustering Detection ──
        if returns_20d and len(returns_20d) >= 20:
            try:
                from openalpha_brain.utils.volatility_detector import estimate_garch11

                cache_key = "garch"
                cached, cache_time = _garch_cache.get(cache_key, (None, 0))
                if cached and (time.time() - cache_time) < 3600:
                    garch_result = cached
                else:
                    garch_result = estimate_garch11(returns_20d)
                    _garch_cache[cache_key] = (garch_result, time.time())

                if garch_result and getattr(garch_result, "is_clustering", False):
                    if regime in ("high_volatility", "trending"):
                        _old_regime = regime
                        regime = "crash_risk"
                        logger.info(
                            "[GARCH] Volatility clustering detected! persistence=%.3f "
                            "→ upgrading %s → crash_risk",
                            getattr(garch_result, "persistence", 0),
                            _old_regime,
                        )
            except (ImportError, ValueError, TypeError, OSError) as _garch_exc:
                logger.debug("[GARCH] Volatility detector unavailable: %s", _garch_exc)

        return regime

    @staticmethod
    def get_regime_parameters(regime: str) -> dict[str, Any]:
        """根据市场状态返回推荐的生成参数 (#2 数据流闭环)

        将 GARCH 检测到的 regime 映射为具体的生成参数建议，
        使得系统能够根据市场状态自动调整 decay_window、neutralize_group、turnover_limit 等关键参数。

        Args:
            regime: 市场状态字符串 (来自 infer_current_regime())
                    可选值: 'high_volatility', 'trending', 'low_volatility', 'crash_risk', 'unknown'

        Returns:
            dict: 推荐的生成参数字典，包含:
                - default_decay_window: int — 衰减窗口长度
                - default_neutralize_group: str — 中性化粒度
                - turnover_limit: float — 换手率上限
                - complexity_target_min: int — 最小复杂度目标
                - complexity_target_max: int — 最大复杂度目标
                - risk_multiplier: float — 风险乘数
        """
        params = {
            "default_decay_window": 20,
            "default_neutralize_group": "sector",
            "turnover_limit": 0.35,
            "complexity_target_min": 3,
            "complexity_target_max": 8,
            "risk_multiplier": 1.0,
        }

        if regime == "high_volatility":
            params.update({
                "default_decay_window": 10,       # 更短衰减窗口
                "default_neutralize_group": "subindustry",  # 更细粒度中性化
                "turnover_limit": 0.25,           # 更低换手限制
                "complexity_target_min": 4,
                "complexity_target_max": 7,
                "risk_multiplier": 1.5,
            })
            logger.debug("[REGIME PARAMS] High volatility regime detected → conservative params")
        elif regime == "crash_risk":
            params.update({
                "default_decay_window": 5,        # 极短衰减
                "default_neutralize_group": "industry",
                "turnover_limit": 0.15,           # 极低换手
                "complexity_target_min": 5,
                "complexity_target_max": 6,
                "risk_multiplier": 2.0,
            })
            logger.warning("[REGIME PARAMS] ⚠ Crash risk regime detected → ultra-conservative params")
        elif regime == "low_volatility":
            params.update({
                "default_decay_window": 30,       # 更长衰减窗口
                "default_neutralize_group": "sector",
                "turnover_limit": 0.45,           # 放宽换手限制
                "complexity_target_min": 2,
                "complexity_target_max": 10,
                "risk_multiplier": 0.8,
            })
            logger.debug("[REGIME PARAMS] Low volatility regime detected → aggressive params")
        elif regime == "trending":
            params.update({
                "default_decay_window": 15,
                "default_neutralize_group": "sector",
                "turnover_limit": 0.40,
                "complexity_target_min": 3,
                "complexity_target_max": 9,
                "risk_multiplier": 1.2,
            })
            logger.debug("[REGIME PARAMS] Trending regime detected → balanced params")

        return params

