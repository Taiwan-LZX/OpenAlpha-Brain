from __future__ import annotations

import contextlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_STATE_PATH = Path(__file__).resolve().parent / "market_state.json"


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
