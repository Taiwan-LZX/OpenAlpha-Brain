from __future__ import annotations

import logging
import random
from typing import Any

from openalpha_brain.knowledge.field_proxy_map import FieldProxyMap, get_field_proxy_map
from openalpha_brain.learning.mab import (
    _TEMPLATE_DIRECTION_MAP,
    TemplateFamilyBandit,
)
from openalpha_brain.validation.decay_detector import AlphaDecayDetector

logger = logging.getLogger(__name__)


class ExplorationScheduler:
    """v3 统一调度器: TemplateFamilyBandit (600 arm) + FeatureMap + FieldProxyMap"""

    def __init__(
        self,
        template_bandit: TemplateFamilyBandit | None = None,
        feature_map: Any | None = None,
        field_proxy_map: FieldProxyMap | None = None,
        decay_detector: AlphaDecayDetector | None = None,
    ) -> None:
        self._bandit = template_bandit or TemplateFamilyBandit()
        self._feature_map = feature_map
        self._fpm = field_proxy_map
        self._decay_detector = decay_detector
        self._initialized = False

    def initialize(
        self,
        template_ids: list[str] | None = None,
        family_ids: list[str] | None = None,
    ) -> None:
        if self._initialized:
            return
        if template_ids is None:
            template_ids = list(_TEMPLATE_DIRECTION_MAP.keys())
        if family_ids is None:
            if self._fpm is None:
                self._fpm = get_field_proxy_map()
            family_ids = [f.family_id for f in self._fpm.get_families()]
        self._bandit.initialize_arms_from_library(template_ids, family_ids)
        self._initialized = True
        logger.info(
            "ExplorationScheduler initialized: %d templates × %d families = %d arms",
            len(template_ids), len(family_ids), self._bandit.arm_count,
        )

    @property
    def bandit(self) -> TemplateFamilyBandit:
        return self._bandit

    @property
    def feature_map(self) -> Any:
        return self._feature_map

    @feature_map.setter
    def feature_map(self, fm: Any) -> None:
        self._feature_map = fm

    @property
    def field_proxy_map(self) -> FieldProxyMap | None:
        return self._fpm

    @field_proxy_map.setter
    def field_proxy_map(self, fpm: FieldProxyMap | None) -> None:
        self._fpm = fpm

    @property
    def decay_detector(self) -> AlphaDecayDetector | None:
        return self._decay_detector

    @decay_detector.setter
    def decay_detector(self, dd: AlphaDecayDetector | None) -> None:
        self._decay_detector = dd

    _BLACKLIST_MAX_RETRIES = 5

    def _get_crossover_proposals(self) -> list[dict[str, Any]]:
        try:
            from openalpha_brain.core import loop_state as _ls
            proposals = getattr(_ls, '_crossover_exploration_proposals', None)
            if proposals:
                return proposals[:5]
        except (OSError, ValueError, RuntimeError):
            pass
        return []

    def select_exploration_arm(
        self,
        focus_area: str | None = None,
        explore_mode: bool = False,
    ) -> dict[str, Any] | None:
        if not self._initialized:
            self.initialize()
        result = self._bandit.select_with_direction(
            explore_mode=explore_mode,
            focus_area=focus_area,
        )
        if result is None:
            return None
        if self._decay_detector is not None:
            for _ in range(self._BLACKLIST_MAX_RETRIES):
                direction = result.get("direction", "")
                if not direction:
                    break
                blacklisted, reason = self._decay_detector.is_blacklisted(direction)
                if not blacklisted:
                    break
                logger.info(
                    "ExplorationScheduler: direction=%s is blacklisted (%s), retrying",
                    direction, reason,
                )
                result = self._bandit.select_with_direction(
                    explore_mode=True,
                    focus_area=None,
                )
                if result is None:
                    return None
            else:
                direction = result.get("direction", "")
                blacklisted, reason = self._decay_detector.is_blacklisted(direction)
                if blacklisted:
                    logger.warning(
                        "ExplorationScheduler: all %d retries hit blacklisted directions, returning last result dir=%s",
                        self._BLACKLIST_MAX_RETRIES, direction,
                    )
        if self._feature_map is not None:
            try:
                schedule = self._feature_map.get_explore_exploit_schedule()
                if schedule.get("strategy") == "explore":
                    unexplored = self._feature_map.get_unexplored_directions()
                    if unexplored and random.random() < schedule.get("explore_weight", 0.3):
                        alt_dir = unexplored[0]
                        alt_result = self._bandit.select_with_direction(
                            explore_mode=True, focus_area=alt_dir,
                        )
                        if alt_result and alt_result.get("direction") == alt_dir:
                            result = alt_result
            except (OSError, ValueError, RuntimeError):
                pass
        crossover_proposals = self._get_crossover_proposals()
        if crossover_proposals:
            try:
                proposal_dirs = {p.get("direction", "") for p in crossover_proposals if p.get("direction")}
                if proposal_dirs:
                    if result and result.get("direction") in proposal_dirs:
                        self._bandit.update_by_direction(result["direction"], reward=0.05)
                        logger.info(
                            "select_exploration_arm: crossover proposal boosts direction=%s",
                            result["direction"],
                        )
                    else:
                        for pdir in proposal_dirs:
                            alt_result = self._bandit.select_with_direction(
                                explore_mode=True, focus_area=pdir,
                            )
                            if alt_result and alt_result.get("direction") == pdir:
                                self._bandit.update_by_direction(pdir, reward=0.05)
                                result = alt_result
                                logger.info(
                                    "select_exploration_arm: crossover proposal overrides to direction=%s",
                                    pdir,
                                )
                                break
            except (OSError, ValueError, RuntimeError):
                pass
        if self._fpm and result.get("family_id"):
            try:
                recommended = self._fpm.recommend_fields_for_template(
                    template_id=result["template_id"],
                    family_id=result["family_id"],
                    top_k=10,
                    exclude_cold=True,
                )
                result["recommended_fields"] = recommended
            except (OSError, ValueError, RuntimeError):
                result["recommended_fields"] = []
        else:
            result["recommended_fields"] = []
        return result

    def select_diverse_directions(
        self,
        num: int = 3,
        focus_area: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self._initialized:
            self.initialize()
        results = self._bandit.select_diverse_directions(num, explore_mode=False)
        if focus_area and focus_area in _TEMPLATE_DIRECTION_MAP.values():
            biased = self._bandit.select_with_direction(
                explore_mode=False, focus_area=focus_area,
            )
            if biased and biased.get("direction") == focus_area:
                existing_dirs = {r.get("direction") for r in results}
                if focus_area not in existing_dirs:
                    results.insert(0, biased)
                    results = results[:num]
        if self._feature_map is not None:
            try:
                schedule = self._feature_map.get_explore_exploit_schedule()
                unexplored = self._feature_map.get_unexplored_directions()
                if schedule.get("strategy") == "explore" and unexplored:
                    explore_weight = schedule.get("explore_weight", 0.3)
                    for i in range(min(len(unexplored), len(results))):
                        if random.random() < explore_weight:
                            alt = self._bandit.select_with_direction(
                                explore_mode=True, focus_area=unexplored[i],
                            )
                            if alt:
                                results[i] = alt
            except (OSError, ValueError, RuntimeError):
                pass
        if self._decay_detector is not None:
            filtered = []
            for r in results:
                direction = r.get("direction", "")
                if direction:
                    blacklisted, reason = self._decay_detector.is_blacklisted(direction)
                    if blacklisted:
                        logger.info(
                            "select_diverse_directions: filtering blacklisted direction=%s (%s)",
                            direction, reason,
                        )
                        continue
                filtered.append(r)
            results = filtered
        return results[:num]

    def record_result(
        self,
        template_id: str,
        family_id: str,
        direction: str,
        reward: float = 0.0,
        penalty: float = 0.0,
    ) -> None:
        self._bandit.update(template_id, family_id, reward=reward, penalty=penalty)

    def record_direction_result(
        self,
        direction: str,
        reward: float = 0.0,
        penalty: float = 0.0,
    ) -> None:
        self._bandit.update_by_direction(direction, reward=reward, penalty=penalty)

    def get_direction_stats(self) -> dict[str, dict[str, Any]]:
        return self._bandit.get_direction_stats()

    def get_arm_stats(self) -> dict[str, dict[str, Any]]:
        return self._bandit.get_stats()

    def health_check(self) -> dict[str, Any]:
        bandit_hc = self._bandit.health_check()
        fm_hc = {}
        if self._feature_map is not None and hasattr(self._feature_map, "health_check"):
            try:
                fm_hc = self._feature_map.health_check()
            except (OSError, ValueError, RuntimeError):
                fm_hc = {"status": "error"}
        return {
            "scheduler": "active",
            "initialized": self._initialized,
            "bandit": bandit_hc,
            "feature_map": fm_hc,
            "field_proxy_map_loaded": self._fpm is not None and self._fpm.is_ready,
        }

    def adjust_mab_bias(self, market_state: Any) -> None:
        if market_state is None:
            return
        try:
            if hasattr(market_state, "adjust_bandit_bias"):
                market_state.adjust_bandit_bias(self._bandit)
        except (OSError, ValueError, RuntimeError):
            pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "bandit": self._bandit.to_dict(),
            "initialized": self._initialized,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExplorationScheduler:
        bandit = TemplateFamilyBandit.from_dict(d.get("bandit", {}))
        sched = cls(template_bandit=bandit)
        sched._initialized = d.get("initialized", False)
        return sched
