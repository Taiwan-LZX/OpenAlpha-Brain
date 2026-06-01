"""
OpenAlpha-Brain — Three-Layer Dynamic Whitelist Manager
Manages core (permanent), dynamic (RAG-retrieved), and solidified (BRAIN-validated)
field whitelists with decay and elimination mechanisms.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from openalpha_brain.config.config import settings
from openalpha_brain.learning.mab import BetaArm

logger = logging.getLogger(__name__)

CORE_FIELDS: set[str] = {
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

ELIMINATION_THRESHOLD = getattr(settings, "ELIMINATION_THRESHOLD", 0.15)
ELIMINATION_MIN_OBS = 10
OVERUSE_PENALTY = 0.1
OVERUSE_WINDOW = getattr(settings, "OVERUSE_WINDOW", 20)
OVERUSE_THRESHOLD = getattr(settings, "OVERUSE_THRESHOLD", 8)
FIELD_DIVERSITY_MIN_DATASETS = 2


class WhitelistManager:
    """[Brief description of class purpose.]"""

    def __init__(self, core_fields: set[str] | None = None) -> None:
        """[Brief description of function purpose.]

        Args:
            core_fields (set[str] | None): [Description]

        Returns:
            None: [Description]
        """
        self._core: set[str] = core_fields if core_fields is not None else set(CORE_FIELDS)
        self._dynamic: set[str] = set()
        self._solidified: dict[str, BetaArm] = {}
        self._eliminated: dict[str, dict[str, Any]] = {}
        self._usage_counts: dict[str, int] = {}
        self._recent_usage: list[str] = []
        self._field_datasets: dict[str, str] = {}

    def get_allowed_fields(self) -> set[str]:
        """[Brief description of function purpose.]

        Returns:
            set[str]: [Description]
        """
        allowed = set(self._core)
        allowed.update(self._dynamic)
        allowed.update(self._solidified.keys())
        allowed -= set(self._eliminated.keys())
        return allowed

    def update_dynamic(self, field_ids: list[str]) -> None:
        """[Brief description of function purpose.]

        Args:
            field_ids (list[str]): [Description]

        Returns:
            None: [Description]
        """
        self._dynamic = set(field_ids) - self._eliminated.keys()
        logger.debug("Dynamic whitelist updated: %d fields", len(self._dynamic))

    def record_usage(self, field_id: str) -> None:
        """[Brief description of function purpose.]

        Args:
            field_id (str): [Description]

        Returns:
            None: [Description]
        """
        self._usage_counts[field_id] = self._usage_counts.get(field_id, 0) + 1
        self._recent_usage.append(field_id)
        if len(self._recent_usage) > OVERUSE_WINDOW * 2:
            self._recent_usage = self._recent_usage[-OVERUSE_WINDOW:]

    def register_field_dataset(self, field_id: str, dataset: str) -> None:
        """[Brief description of function purpose.]

        Args:
            field_id (str): [Description]
            dataset (str): [Description]

        Returns:
            None: [Description]
        """
        self._field_datasets[field_id] = dataset

    def detect_field_overfit(self) -> list[dict[str, Any]]:
        """[Brief description of function purpose.]

        Returns:
            list[dict[str, Any]]: [Description]
        """
        warnings: list[dict[str, Any]] = []
        if len(self._recent_usage) < OVERUSE_WINDOW:
            return warnings
        recent_window = self._recent_usage[-OVERUSE_WINDOW:]
        from collections import Counter

        counts = Counter(recent_window)
        for field_id, count in counts.items():
            if count >= OVERUSE_THRESHOLD:
                ratio = count / OVERUSE_WINDOW
                warnings.append(
                    {
                        "field_id": field_id,
                        "type": "field_overuse",
                        "count": count,
                        "window": OVERUSE_WINDOW,
                        "ratio": round(ratio, 3),
                        "message": f"Field '{field_id}' used {count}/{OVERUSE_WINDOW} times recently ({ratio:.1%}) — overfitting risk",  # noqa: E501
                    }
                )
                if field_id in self._solidified:
                    self._solidified[field_id].penalize(OVERUSE_PENALTY * 2)
        datasets_in_window = set()
        for fid in recent_window:
            ds = self._field_datasets.get(fid, "unknown")
            datasets_in_window.add(ds)
        if len(datasets_in_window) < FIELD_DIVERSITY_MIN_DATASETS:
            warnings.append(
                {
                    "type": "low_dataset_diversity",
                    "datasets_count": len(datasets_in_window),
                    "min_required": FIELD_DIVERSITY_MIN_DATASETS,
                    "message": f"Only {len(datasets_in_window)} dataset(s) in recent {OVERUSE_WINDOW} fields — diversity too low",  # noqa: E501
                }
            )
        return warnings

    def apply_overuse_penalty(self, field_id: str) -> None:
        """[Brief description of function purpose.]

        Args:
            field_id (str): [Description]

        Returns:
            None: [Description]
        """
        if field_id in self._solidified:
            self._solidified[field_id].penalize(OVERUSE_PENALTY)
        if field_id in self._usage_counts and self._usage_counts[field_id] > 5:
            if field_id in self._solidified:
                self._solidified[field_id].penalize(OVERUSE_PENALTY)
            logger.debug("Overuse penalty applied to field: %s", field_id)

    def solidify_field(self, field_id: str) -> None:
        """[Brief description of function purpose.]

        Args:
            field_id (str): [Description]

        Returns:
            None: [Description]
        """
        if field_id in self._eliminated:
            del self._eliminated[field_id]
        if field_id not in self._solidified:
            self._solidified[field_id] = BetaArm(alpha=2.0, beta=1.0)
        else:
            self._solidified[field_id].reward(1.0)
        logger.info("Field solidified: %s (E[Beta]=%.3f)", field_id, self._solidified[field_id].expectation)

    def check_eliminations(self) -> list[str]:
        """[Brief description of function purpose.]

        Returns:
            list[str]: [Description]
        """
        eliminated_now: list[str] = []
        for field_id, arm in list(self._solidified.items()):
            total = arm.alpha + arm.beta
            if total >= ELIMINATION_MIN_OBS and arm.expectation < ELIMINATION_THRESHOLD:
                self._eliminated[field_id] = {
                    "reason": "low_score",
                    "expectation": arm.expectation,
                    "eliminated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                del self._solidified[field_id]
                self._dynamic.discard(field_id)
                eliminated_now.append(field_id)
                logger.info("Field eliminated: %s (E[Beta]=%.3f, obs=%d)", field_id, arm.expectation, int(total))
        return eliminated_now

    def update_field_reward(self, field_id: str, reward: float = 0.0, penalty: float = 0.0) -> None:
        """[Brief description of function purpose.]

        Args:
            field_id (str): [Description]
            reward (float): [Description]
            penalty (float): [Description]

        Returns:
            None: [Description]
        """
        if field_id in self._solidified:
            if reward > 0:
                self._solidified[field_id].reward(reward)
            if penalty > 0:
                self._solidified[field_id].penalize(penalty)

    @property
    def core_fields(self) -> set[str]:
        """[Brief description of function purpose.]

        Returns:
            set[str]: [Description]
        """
        return set(self._core)

    @property
    def dynamic_fields(self) -> set[str]:
        """[Brief description of function purpose.]

        Returns:
            set[str]: [Description]
        """
        return set(self._dynamic)

    @property
    def solidified_fields(self) -> dict[str, BetaArm]:
        """[Brief description of function purpose.]

        Returns:
            dict[str, BetaArm]: [Description]
        """
        return dict(self._solidified)

    @property
    def eliminated_fields(self) -> dict[str, dict[str, Any]]:
        """[Brief description of function purpose.]

        Returns:
            dict[str, dict[str, Any]]: [Description]
        """
        return dict(self._eliminated)

    def health_check(self) -> dict[str, Any]:
        """[Brief description of function purpose.]

        Returns:
            dict[str, Any]: [Description]
        """
        return {
            "module": "WhitelistManager",
            "status": "active",
            "solidified_count": len(self._solidified),
            "dynamic_count": len(self._dynamic),
            "overuse_threshold": OVERUSE_THRESHOLD,
            "eliminated_count": len(self._eliminated),
        }

    def to_dict(self) -> dict[str, Any]:
        """[Brief description of function purpose.]

        Returns:
            dict[str, Any]: [Description]
        """
        return {
            "core": sorted(self._core),
            "dynamic": sorted(self._dynamic),
            "solidified": {fid: arm.to_dict() for fid, arm in self._solidified.items()},
            "eliminated": self._eliminated,
            "usage_counts": self._usage_counts,
            "recent_usage": list(self._recent_usage),
            "field_datasets": dict(self._field_datasets) if hasattr(self, "_field_datasets") else {},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WhitelistManager:
        """[Brief description of function purpose.]

        Args:
            d (dict[str, Any]): [Description]

        Returns:
            'WhitelistManager': [Description]
        """
        core = set(d.get("core", CORE_FIELDS))
        wm = cls(core_fields=core)
        wm._dynamic = set(d.get("dynamic", []))
        solidified_data = d.get("solidified", {})
        wm._solidified = {fid: BetaArm.from_dict(arm_data) for fid, arm_data in solidified_data.items()}
        wm._eliminated = d.get("eliminated", {})
        wm._usage_counts = d.get("usage_counts", {})
        wm._recent_usage = d.get("recent_usage", [])
        if "field_datasets" in d:
            wm._field_datasets = d.get("field_datasets", {})
        return wm
