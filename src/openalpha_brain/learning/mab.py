"""
OpenAlpha-Brain — Thompson Sampling Multi-Armed Bandit
Implements hierarchical MAB with Beta distribution for exploration-exploitation
balance in alpha factor mining.
"""
from __future__ import annotations

import json
import logging
import math
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from openalpha_brain.config.config import settings
from openalpha_brain.utils.algo_logger import algo_log

logger = logging.getLogger(__name__)

_MAB_STATE_PATH = Path(__file__).resolve().parent / "mab_state.json"

BETA_DECAY_FACTOR = getattr(settings, 'BETA_DECAY_FACTOR', 0.99)
BETA_DECAY_INTERVAL = getattr(settings, 'BETA_DECAY_INTERVAL', 100)


class BetaArm:
    """[Brief description of class purpose.]"""
    __slots__ = ("alpha", "beta", "_total_updates")

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        """[Brief description of function purpose.]

            Args:
                alpha (float): [Description]
                beta (float): [Description]

            Returns:
                None: [Description]
            """
        self.alpha = alpha
        self.beta = beta
        self._total_updates = 0

    @property
    def expectation(self) -> float:
        """[Brief description of function purpose.]

            Returns:
                float: [Description]
            """
        total = self.alpha + self.beta
        if total == 0:
            return 0.5
        return self.alpha / total

    def sample(self) -> float:
        """[Brief description of function purpose.]

            Returns:
                float: [Description]
            """
        return np.random.beta(max(self.alpha, 1e-6), max(self.beta, 1e-6))

    def reward(self, value: float) -> None:
        """[Brief description of function purpose.]

            Args:
                value (float): [Description]

            Returns:
                None: [Description]
            """
        value = max(0.0, value)
        self.alpha += value
        self._total_updates += 1
        if self._total_updates % BETA_DECAY_INTERVAL == 0:
            self.alpha = max(1.0, self.alpha * BETA_DECAY_FACTOR)
            self.beta = max(1.0, self.beta * BETA_DECAY_FACTOR)

    def penalize(self, value: float) -> None:
        """[Brief description of function purpose.]

            Args:
                value (float): [Description]

            Returns:
                None: [Description]
            """
        value = max(0.0, value)
        self.beta += value
        self._total_updates += 1
        if self._total_updates % BETA_DECAY_INTERVAL == 0:
            self.alpha = max(1.0, self.alpha * BETA_DECAY_FACTOR)
            self.beta = max(1.0, self.beta * BETA_DECAY_FACTOR)

    def to_dict(self) -> dict[str, float]:
        """[Brief description of function purpose.]

            Returns:
                dict[str, float]: [Description]
            """
        return {"alpha": self.alpha, "beta": self.beta, "total_updates": self._total_updates}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> BetaArm:
        """[Brief description of function purpose.]

            Args:
                d (dict[str, float]): [Description]

            Returns:
                'BetaArm': [Description]
            """
        arm = cls(alpha=d.get("alpha", 1.0), beta=d.get("beta", 1.0))
        _tu: int = int(d.get("total_updates", 0))
        arm._total_updates = _tu
        return arm


class ThompsonBandit:
    """[Brief description of class purpose.]"""
    def __init__(self) -> None:
        """[Brief description of function purpose.]

            Returns:
                None: [Description]
            """
        self._arms: dict[str, BetaArm] = {}

    def ensure_arm(self, arm_id: str) -> BetaArm:
        """[Brief description of function purpose.]

            Args:
                arm_id (str): [Description]

            Returns:
                BetaArm: [Description]
            """
        if arm_id not in self._arms:
            self._arms[arm_id] = BetaArm()
        return self._arms[arm_id]

    def select_arm(self) -> str:
        """[Brief description of function purpose.]

            Returns:
                str: [Description]
            """
        if not self._arms:
            raise ValueError("No arms available")
        samples = {aid: arm.sample() for aid, arm in self._arms.items()}
        return max(samples, key=lambda k: cast(float, samples[k]))

    def select_top_k(self, k: int) -> list[str]:
        """[Brief description of function purpose.]

            Args:
                k (int): [Description]

            Returns:
                list[str]: [Description]
            """
        if not self._arms:
            return []
        samples = {aid: arm.sample() for aid, arm in self._arms.items()}
        sorted_arms = sorted(samples, key=lambda k: cast(float, samples[k]), reverse=True)
        return sorted_arms[:k]

    def update(self, arm_id: str, reward: float = 0.0, penalty: float = 0.0) -> None:
        """[Brief description of function purpose.]

            Args:
                arm_id (str): [Description]
                reward (float): [Description]
                penalty (float): [Description]

            Returns:
                None: [Description]
            """
        arm = self.ensure_arm(arm_id)
        if reward > 0:
            arm.reward(reward)
        if penalty > 0:
            arm.penalize(penalty)

    def get_stats(self) -> dict[str, dict[str, float]]:
        """[Brief description of function purpose.]

            Returns:
                dict[str, dict[str, float]]: [Description]
            """
        return {aid: {"alpha": arm.alpha, "beta": arm.beta, "expectation": arm.expectation} for aid, arm in self._arms.items()}

    def get_arm(self, arm_id: str) -> BetaArm | None:
        """[Brief description of function purpose.]

            Args:
                arm_id (str): [Description]

            Returns:
                BetaArm | None: [Description]
            """
        return self._arms.get(arm_id)

    @property
    def arm_count(self) -> int:
        """[Brief description of function purpose.]

            Returns:
                int: [Description]
            """
        return len(self._arms)

    def to_dict(self) -> dict[str, dict[str, float]]:
        """[Brief description of function purpose.]

            Returns:
                dict[str, dict[str, float]]: [Description]
            """
        return {aid: arm.to_dict() for aid, arm in self._arms.items()}

    @classmethod
    def from_dict(cls, d: dict[str, dict[str, float]]) -> ThompsonBandit:
        """[Brief description of function purpose.]

            Args:
                d (dict[str, dict[str, float]]): [Description]

            Returns:
                'ThompsonBandit': [Description]
            """
        bandit = cls()
        for aid, params in d.items():
            bandit._arms[aid] = BetaArm.from_dict(params)
        return bandit


class SlidingWindowUCB:
    """Non-stationary UCB with sliding window + exponential decay.

    Core improvements over standard UCB:
    - Only retains the most recent W rewards (discards stale data)
    - Applies exponential decay to windowed rewards (recent = higher weight)
    - UCB formula: weighted_mean + exploration_bonus * sqrt(log(total) / window_count)

    Reference: R&D-Agent-Quant (NeurIPS 2025) non-stationary bandit design.
    """

    __slots__ = (
        "arm_id",
        "window_size",
        "decay_factor",
        "_exploration_const",
        "_rewards",
        "_total_pulls",
        "_last_reward_time",
        "_prior_weight",
    )

    def __init__(
        self,
        arm_id: str,
        window_size: int = 20,
        decay_factor: float = 0.95,
        exploration_const: float = 2.0,
    ) -> None:
        self.arm_id = arm_id
        self.window_size = window_size
        self.decay_factor = decay_factor
        self._exploration_const = exploration_const

        self._rewards: deque[tuple[float, datetime]] = deque(maxlen=window_size)
        self._total_pulls: int = 0
        self._last_reward_time: datetime | None = None
        self._prior_weight: float = 0.05

    @algo_log()
    def update(self, reward: float) -> None:
        now = datetime.now(UTC)
        self._rewards.append((reward, now))
        self._total_pulls += 1
        self._last_reward_time = now

    @property
    def weighted_mean(self) -> float:
        if not self._rewards:
            return 0.0
        from openalpha_brain.utils.algo_logger import Timer

        with Timer("sw_ucb_weighted_mean"):
            now = datetime.now(UTC)
            weights: list[float] = []
            values: list[float] = []
            for rew, ts in sorted(self._rewards, key=lambda x: x[1]):
                hours_ago = (now - ts).total_seconds() / 3600.0
                weight = self.decay_factor ** max(hours_ago, 0.0)
                weights.append(weight)
                values.append(rew)
            total_weight = sum(weights) + 1e-6
            result = sum(w * v for w, v in zip(weights, values)) / total_weight
        return result

    @property
    def ucb_score(self) -> float:
        n = len(self._rewards)
        if n == 0:
            return self._prior_weight
        mean = self.weighted_mean
        bonus = self._exploration_const * math.sqrt(
            math.log(self._total_pulls + 1) / (n + 1e-6)
        )
        return mean + bonus

    def set_prior(self, prior_score: float) -> None:
        self._prior_weight = max(0.05, min(1.0, prior_score))

    @property
    def total_pulls(self) -> int:
        return self._total_pulls

    @property
    def window_count(self) -> int:
        return len(self._rewards)

    @property
    def expectation(self) -> float:
        n = len(self._rewards)
        if n == 0:
            return self._prior_weight
        return self.weighted_mean

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "window_size": self.window_size,
            "decay_factor": self.decay_factor,
            "exploration_const": self._exploration_const,
            "prior_weight": self._prior_weight,
            "total_pulls": self._total_pulls,
            "rewards": [
                {"reward": r, "timestamp": t.isoformat()}
                for r, t in self._rewards
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SlidingWindowUCB:
        arm = cls(
            arm_id=d.get("arm_id", ""),
            window_size=d.get("window_size", 20),
            decay_factor=d.get("decay_factor", 0.95),
            exploration_const=d.get("exploration_const", 2.0),
        )
        arm._prior_weight = d.get("prior_weight", 0.05)
        arm._total_pulls = d.get("total_pulls", 0)
        for entry in d.get("rewards", []):
            try:
                ts = datetime.fromisoformat(entry["timestamp"])
                arm._rewards.append((entry["reward"], ts))
            except (ValueError, KeyError):
                pass
        return arm


class MABPriorInitializer:
    """使用 FieldProxyMap 语义相似度 + 模板经典度为 MAB arm 设置先验.

    参考: LLM-Powered MCTS (AAAI 2025) 的先验导航思想.
    先验组成（加权）:
      - 模板经典度 (0.40): 该模板在因子库中的出现频率/学术引用度
      - 字段族数据质量 (0.30): 字段数量、覆盖率、冷字段比例
      - 模板-字段族兼容性 (0.30): applicable_templates 匹配度
    """

    _TEMPLATE_CLASSICNESS: dict[str, float] = {
        "momentum_short_term_reversal": 0.80,
        "momentum_medium_term_continuation": 0.90,
        "momentum_volume_confirmed": 0.70,
        "momentum_long_term_reversal": 0.60,
        "value_regression": 0.85,
        "value_earnings_quality": 0.65,
        "quality_earnings_stability": 0.75,
        "quality_asset_turnover": 0.55,
        "size_small_cap_premium": 0.70,
        "volatility_low_vol_anomaly": 0.65,
        "volatility_change_signal": 0.50,
        "volatility_clustering": 0.45,
        "liquidity_premium": 0.65,
        "liquidity_improvement_signal": 0.50,
        "lead_lag_price_volume": 0.55,
        "lead_lag_cross_field": 0.50,
        "lead_lag_industry_rotation": 0.40,
        "mean_reversion_zscore": 0.75,
        "mean_reversion_bollinger": 0.60,
        "mean_reversion_valuation": 0.45,
    }

    def __init__(self) -> None:
        self._fpm: Any = None
        self._family_stats_cache: dict[str, dict[str, Any]] | None = None

    def set_field_proxy_map(self, fpm: Any) -> None:
        self._fpm = fpm
        if hasattr(fpm, "get_family_stats"):
            self._family_stats_cache = fpm.get_family_stats()

    @algo_log()
    def compute_prior(self, template_id: str, family_id: str) -> float:
        template_score = self._get_template_classicness(template_id)
        family_score = self._get_family_quality(family_id)
        compat_score = self._get_compatibility(template_id, family_id)

        prior = 0.40 * template_score + 0.30 * family_score + 0.30 * compat_score
        result = max(0.05, min(1.0, prior))
        logger.debug(
            "[DEFENSIVE_LOG] MABPriorInitializer::compute_prior "
            "template=%s family=%s classic=%.3f quality=%.3f compat=%.3f prior=%.3f",
            template_id, family_id, template_score, family_score, compat_score, result,
        )
        return result

    def _get_template_classicness(self, template_id: str) -> float:
        return self._TEMPLATE_CLASSICNESS.get(template_id, 0.30)

    def _get_family_quality(self, family_id: str) -> float:
        if self._family_stats_cache is None:
            return 0.40
        stats = self._family_stats_cache.get(family_id)
        if stats is None:
            return 0.30
        field_count = stats.get("field_count", 0)
        avg_coverage = stats.get("avg_coverage", 0.0)
        cold_ratio = stats.get("cold_ratio", 1.0)

        count_norm = min(field_count / 100.0, 1.0)
        coverage_score = avg_coverage
        freshness_score = 1.0 - cold_ratio

        quality = 0.4 * count_norm + 0.35 * coverage_score + 0.25 * freshness_score
        return max(0.05, min(1.0, quality))

    def _get_compatibility(self, template_id: str, family_id: str) -> float:
        try:
            from openalpha_brain.knowledge.field_proxy_map import FIELD_FAMILIES
            fam = FIELD_FAMILIES.get(family_id)
            if fam is None:
                return 0.20
            if template_id in fam.applicable_templates:
                return 0.90
            l1_match = 0
            for fid, f in FIELD_FAMILIES.items():
                if template_id in f.applicable_templates and f.l1_category == fam.l1_category:
                    l1_match += 1
            total_in_l1 = sum(
                1 for f in FIELD_FAMILIES.values()
                if f.l1_category == fam.l1_category
            )
            if total_in_l1 > 0:
                return 0.40 + 0.30 * (l1_match / total_in_l1)
            return 0.20
        except (ValueError, TypeError, OSError):
            logger.warning("[DEFENSIVE_LOG] MABPriorInitializer::_get_compatibility error")
            return 0.20


class ComputeAllocator:
    """Thompson Sampling 层做 GENERATE vs IMPROVE 算力分配.

    状态感知:
      - 近期 PASS 率高 → 多 IMPROVE（深挖有效区域）
      - 近期 PASS 率低 → 多 GENERATE（广度探索新区域）
    """

    __slots__ = (
        "_gen_alpha",
        "_gen_beta",
        "_imp_alpha",
        "_imp_beta",
        "_recent_results",
        "_window",
    )

    def __init__(self, window_size: int = 20) -> None:
        self._gen_alpha: float = 2.0
        self._gen_beta: float = 2.0
        self._imp_alpha: float = 2.0
        self._imp_beta: float = 2.0
        self._recent_results: deque[bool] = deque(maxlen=window_size)
        self._window = window_size

    @algo_log()
    def allocate(self) -> dict[str, Any]:
        gen_sample = np.random.beta(max(self._gen_alpha, 1e-6), max(self._gen_beta, 1e-6))
        imp_sample = np.random.beta(max(self._imp_alpha, 1e-6), max(self._imp_beta, 1e-6))
        total = gen_sample + imp_sample + 1e-6
        gen_ratio = gen_sample / total
        imp_ratio = imp_sample / total
        action = "IMPROVE" if imp_ratio >= gen_ratio else "GENERATE"
        return {
            "action": action,
            "generate_ratio": round(gen_ratio, 4),
            "improve_ratio": round(imp_ratio, 4),
            "gen_alpha": self._gen_alpha,
            "gen_beta": self._gen_beta,
            "imp_alpha": self._imp_alpha,
            "imp_beta": self._imp_beta,
        }

    @algo_log()
    def update(self, action: str, success: bool) -> None:
        self._recent_results.append(success)
        if action == "GENERATE":
            if success:
                self._gen_alpha += 1.0
            else:
                self._gen_beta += 1.0
        elif action == "IMPROVE":
            if success:
                self._imp_alpha += 1.0
            else:
                self._imp_beta += 1.0

    @property
    def recent_pass_rate(self) -> float:
        if not self._recent_results:
            return 0.5
        return sum(1 for r in self._recent_results if r) / len(self._recent_results)

    def health_check(self) -> dict[str, Any]:
        return {
            "module": "ComputeAllocator",
            "status": "active",
            "recent_pass_rate": self.recent_pass_rate,
            "window_size": self._window,
            "result_count": len(self._recent_results),
        }


EXPLORATION_DIRECTIONS: list[dict[str, str]] = [
    {"id": "momentum", "description": "Price momentum and trend-following strategies using time-series operators"},
    {"id": "mean_reversion", "description": "Mean reversion and contrarian strategies based on price overextension"},
    {"id": "volatility", "description": "Volatility-based strategies including breakout and risk premium"},
    {"id": "value", "description": "Fundamental value investing using financial ratios and book values"},
    {"id": "quality", "description": "Quality factor based on profitability, earnings stability, and leverage"},
    {"id": "liquidity", "description": "Liquidity-based strategies using volume and trading activity"},
    {"id": "size", "description": "Size effect and market capitalization based strategies"},
    {"id": "industry_rotation", "description": "Industry and sector rotation strategies"},
    {"id": "temporal", "description": "Time-series pattern strategies using decay, ranking, and regression"},
    {"id": "cross_sectional", "description": "Cross-sectional ranking and neutralization strategies"},
    {"id": "interaction", "description": "Multi-factor interaction strategies combining multiple signals"},
]


class HierarchicalMAB:
    """Hierarchical MAB: outer direction selection + inner operator/field selection.

    v4 upgrade: added ComputeAllocator for GENERATE vs IMPROVE allocation,
    and support for MABPriorInitializer cold-start priors.
    """
    def __init__(self) -> None:
        self._outer = ThompsonBandit()
        self._inner_ops: dict[str, ThompsonBandit] = {}
        self._inner_fields: dict[str, ThompsonBandit] = {}
        self._update_count: int = 0
        self._allocator = ComputeAllocator()
        for direction in EXPLORATION_DIRECTIONS:
            self._outer.ensure_arm(direction["id"])

    @property
    def allocator(self) -> ComputeAllocator:
        return self._allocator

    def initialize_template_family_bandit(
        self,
        tf_bandit: TemplateFamilyBandit,
        prior_initializer: MABPriorInitializer,
    ) -> None:
        tf_bandit.initialize_priors(prior_initializer)
        logger.info(
            "[DEFENSIVE_LOG] HierarchicalMAB::initialize_template_family_bandit "
            "完成先验设置，arm_count=%d",
            tf_bandit.arm_count,
        )

    def select(self, top_k_ops: int = 15, top_k_fields: int = 40,
                focus_area: str = "") -> dict[str, Any]:
        """[Brief description of function purpose.

        Args:
            top_k_ops (int): [Description]
            top_k_fields (int): [Description]
            focus_area (str): User-specified exploration direction to bias toward.

        Returns:
            dict[str, Any]: [Description]
        """
        if focus_area and focus_area in self._outer._arms:
            _direction = focus_area
            logger.debug("MAB: using focus_area bias: %s", focus_area)
        else:
            _direction = self._outer.select_arm()
        ops_bandit = self._inner_ops.get(_direction)
        fields_bandit = self._inner_fields.get(_direction)

        selected_ops: list[str] = []
        if ops_bandit and ops_bandit.arm_count > 0:
            selected_ops = ops_bandit.select_top_k(top_k_ops)

        selected_fields: list[str] = []
        if fields_bandit and fields_bandit.arm_count > 0:
            selected_fields = fields_bandit.select_top_k(top_k_fields)

        return {
            "direction": _direction,
            "operators": selected_ops,
            "fields": selected_fields,
        }

    @algo_log()
    def update(
        self,
        direction: str,
        operators: list[str],
        fields: list[str],
        reward: float = 0.0,
        penalty: float = 0.0,
    ) -> None:
        """[Brief description of function purpose.]

            Args:
                direction (str): [Description]
                operators (list[str]): [Description]
                fields (list[str]): [Description]
                reward (float): [Description]
                penalty (float): [Description]

            Returns:
                None: [Description]
            """
        self._update_count += 1
        self._outer.ensure_arm(direction)
        if reward > 0:
            self._outer.update(direction, reward=reward)
        if penalty > 0:
            self._outer.update(direction, penalty=penalty)

        if direction not in self._inner_ops:
            self._inner_ops[direction] = ThompsonBandit()
        if direction not in self._inner_fields:
            self._inner_fields[direction] = ThompsonBandit()

        ops_bandit = self._inner_ops[direction]
        fields_bandit = self._inner_fields[direction]

        for op in operators:
            ops_bandit.ensure_arm(op)
            if reward > 0:
                ops_bandit.update(op, reward=reward / max(len(operators), 1))
            if penalty > 0:
                ops_bandit.update(op, penalty=penalty / max(len(operators), 1))

        for field in fields:
            fields_bandit.ensure_arm(field)
            if reward > 0:
                fields_bandit.update(field, reward=reward / max(len(fields), 1))
            if penalty > 0:
                fields_bandit.update(field, penalty=penalty / max(len(fields), 1))

    def set_initial_bias(self, direction: str, weight: float) -> None:
        """[Brief description of function purpose.]

            Args:
                direction (str): [Description]
                weight (float): [Description]

            Returns:
                None: [Description]
            """
        arm = self._outer._arms.get(direction)
        if arm is not None:
            arm.alpha = max(arm.alpha, 1.0 + weight * 2.0)

    def get_field_expectations(self) -> dict[str, float]:
        """[Brief description of function purpose.]

            Returns:
                dict[str, float]: [Description]
            """
        result = {}
        for direction, bandit in self._inner_fields.items():
            for arm_id, arm in bandit._arms.items():
                result[arm_id] = arm.expectation
        return result

    def get_operator_expectations(self) -> dict[str, float]:
        """[Brief description of function purpose.]

            Returns:
                dict[str, float]: [Description]
            """
        result = {}
        for direction, bandit in self._inner_ops.items():
            for arm_id, arm in bandit._arms.items():
                result[arm_id] = arm.expectation
        return result

    def get_direction_stats(self) -> dict[str, dict[str, float]]:
        """[Brief description of function purpose.]

            Returns:
                dict[str, dict[str, float]]: [Description]
            """
        return self._outer.get_stats()

    def health_check(self) -> dict[str, Any]:
        inner_ops_count = sum(b.arm_count for b in self._inner_ops.values())
        inner_fields_count = sum(b.arm_count for b in self._inner_fields.values())
        return {
            "module": "HierarchicalMAB",
            "status": "active",
            "outer_arms": self._outer.arm_count,
            "inner_ops_arms": inner_ops_count,
            "inner_fields_arms": inner_fields_count,
            "update_count": self._update_count,
            "allocator": self._allocator.health_check(),
        }

    def to_dict(self) -> dict[str, Any]:
        """[Brief description of function purpose.]

            Returns:
                dict[str, Any]: [Description]
            """
        return {
            "outer": self._outer.to_dict(),
            "inner_ops": {d: b.to_dict() for d, b in self._inner_ops.items()},
            "inner_fields": {d: b.to_dict() for d, b in self._inner_fields.items()},
            "update_count": self._update_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HierarchicalMAB:
        """[Brief description of function purpose.]

            Args:
                d (dict[str, Any]): [Description]

            Returns:
                'HierarchicalMAB': [Description]
            """
        mab = cls()
        if "outer" in d:
            mab._outer = ThompsonBandit.from_dict(d["outer"])
        if "inner_ops" in d:
            mab._inner_ops = {direction: ThompsonBandit.from_dict(data) for direction, data in d["inner_ops"].items()}
        if "inner_fields" in d:
            mab._inner_fields = {direction: ThompsonBandit.from_dict(data) for direction, data in d["inner_fields"].items()}
        mab._update_count = d.get("update_count", 0)
        return mab


class AssociationMatrix:
    """[Brief description of class purpose.]"""
    def __init__(self) -> None:
        """[Brief description of function purpose.]

            Returns:
                None: [Description]
            """
        self._matrix: dict[str, dict[str, BetaArm]] = {}
        self._total_successes: int = 0

    def ensure_entry(self, operator: str, field: str) -> BetaArm:
        """[Brief description of function purpose.]

            Args:
                operator (str): [Description]
                field (str): [Description]

            Returns:
                BetaArm: [Description]
            """
        if operator not in self._matrix:
            self._matrix[operator] = {}
        if field not in self._matrix[operator]:
            self._matrix[operator][field] = BetaArm()
        return self._matrix[operator][field]

    def update(self, operator: str, field: str, reward: float = 0.0, penalty: float = 0.0) -> None:
        """[Brief description of function purpose.]

            Args:
                operator (str): [Description]
                field (str): [Description]
                reward (float): [Description]
                penalty (float): [Description]

            Returns:
                None: [Description]
            """
        arm = self.ensure_entry(operator, field)
        if reward > 0:
            arm.reward(reward)
            self._total_successes += 1
        if penalty > 0:
            arm.penalize(penalty)

    def get_top_fields(self, operator: str, top_k: int = 10) -> list[tuple[str, float]]:
        """[Brief description of function purpose.]

            Args:
                operator (str): [Description]
                top_k (int): [Description]

            Returns:
                list[tuple[str, float]]: [Description]
            """
        ops_entries = self._matrix.get(operator, {})
        if not ops_entries:
            return []
        samples = {field: arm.sample() for field, arm in ops_entries.items()}
        sorted_fields = sorted(samples, key=lambda k: cast(float, samples[k]), reverse=True)
        return [(f, samples[f]) for f in sorted_fields[:top_k]]

    def get_top_operators(self, field: str, top_k: int = 10) -> list[tuple[str, float]]:
        """[Brief description of function purpose.]

            Args:
                field (str): [Description]
                top_k (int): [Description]

            Returns:
                list[tuple[str, float]]: [Description]
            """
        results: dict[str, float] = {}
        for op, entries in self._matrix.items():
            if field in entries:
                results[op] = entries[field].sample()
        sorted_ops = sorted(results, key=lambda k: cast(float, results[k]), reverse=True)
        return [(op, results[op]) for op in sorted_ops[:top_k]]

    def _mf_predict(self, entity_id: str, n: int = 10) -> list[tuple[str, float]]:
        operators = list(self._matrix.keys())
        if not operators:
            return []
        fields: list[str] = []
        for op_entries in self._matrix.values():
            for f in op_entries:
                if f not in fields:
                    fields.append(f)
        if not fields:
            return []

        n_ops = len(operators)
        n_fields = len(fields)
        op_idx = {op: i for i, op in enumerate(operators)}
        field_idx = {f: i for i, f in enumerate(fields)}

        mat = np.zeros((n_ops, n_fields))
        mask = np.zeros((n_ops, n_fields))
        for op, entries in self._matrix.items():
            for f, arm in entries.items():
                i, j = op_idx[op], field_idx[f]
                mat[i, j] = arm.expectation
                mask[i, j] = 1.0

        try:
            k = min(5, min(n_ops, n_fields))
            U, s, Vt = np.linalg.svd(mat, full_matrices=False)
            U_k, s_k, Vt_k = U[:, :k], s[:k], Vt[:k, :]
            reconstructed = U_k @ np.diag(s_k) @ Vt_k
        except np.linalg.LinAlgError:
            col_means = np.sum(mat, axis=0) / np.maximum(np.sum(mask, axis=0), 1)
            reconstructed = np.where(mask, mat, col_means[np.newaxis, :])

        predictions: list[tuple[str, float]] = []
        if entity_id in op_idx:
            row = reconstructed[op_idx[entity_id]]
            for j, f in enumerate(fields):
                if mask[op_idx[entity_id], j] == 0:
                    score = float(np.clip(row[j], 0.0, 1.0))
                    predictions.append((f, score))
        elif entity_id in field_idx:
            col = reconstructed[:, field_idx[entity_id]]
            for i, op in enumerate(operators):
                if mask[i, field_idx[entity_id]] == 0:
                    score = float(np.clip(col[i], 0.0, 1.0))
                    predictions.append((op, score))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]

    @property
    def total_successes(self) -> int:
        """[Brief description of function purpose.]

            Returns:
                int: [Description]
            """
        return self._total_successes

    def health_check(self) -> dict[str, Any]:
        """[Brief description of function purpose.]

            Returns:
                dict[str, Any]: [Description]
            """
        return {
            "module": "AssociationMatrix",
            "status": "active",
            "total_entries": sum(len(ops) for ops in self._matrix.values()),
            "total_successes": self._total_successes,
        }

    def to_dict(self) -> dict[str, Any]:
        """[Brief description of function purpose.]

            Returns:
                dict[str, Any]: [Description]
            """
        matrix_data: dict[str, dict[str, dict[str, float]]] = {}
        for op, entries in self._matrix.items():
            matrix_data[op] = {field: arm.to_dict() for field, arm in entries.items()}
        return {
            "matrix": matrix_data,
            "total_successes": self._total_successes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AssociationMatrix:
        """[Brief description of function purpose.]

            Args:
                d (dict[str, Any]): [Description]

            Returns:
                'AssociationMatrix': [Description]
            """
        am = cls()
        matrix_data = d.get("matrix", {})
        for op, entries in matrix_data.items():
            am._matrix[op] = {field: BetaArm.from_dict(arm_data) for field, arm_data in entries.items()}
        am._total_successes = d.get("total_successes", 0)
        return am


_TEMPLATE_DIRECTION_MAP: dict[str, str] = {
    "momentum_short_term_reversal": "momentum",
    "momentum_medium_term_continuation": "momentum",
    "momentum_volume_confirmed": "momentum",
    "momentum_long_term_reversal": "momentum",
    "value_regression": "value",
    "value_earnings_quality": "value",
    "quality_earnings_stability": "quality",
    "quality_asset_turnover": "quality",
    "size_small_cap_premium": "size",
    "volatility_low_vol_anomaly": "volatility",
    "volatility_change_signal": "volatility",
    "volatility_clustering": "volatility",
    "liquidity_premium": "liquidity",
    "liquidity_improvement_signal": "liquidity",
    "lead_lag_price_volume": "lead_lag",
    "lead_lag_cross_field": "lead_lag",
    "lead_lag_industry_rotation": "lead_lag",
    "mean_reversion_zscore": "mean_reversion",
    "mean_reversion_bollinger": "mean_reversion",
    "mean_reversion_valuation": "mean_reversion",
}


class TemplateFamilyBandit:
    """v4: Template × Field Family MAB with Non-stationary UCB (SlidingWindowUCB).

    升级自 v3 Thompson Sampling:
    - 内层 arm 从 BetaArm(Thompson) → SlidingWindowUCB（滑动窗口 + 指数衰减）
    - 冷启动时使用 MABPriorInitializer 先验分数
    - 非平稳环境自适应：风格切换后旧奖励自动衰减

    Arm space: ~20 templates × ~30 field families ≈ 600 arms.
    Each arm key is "{template_id}::{family_id}".
    """

    def __init__(self, window_size: int = 20, decay_factor: float = 0.95,
                 exploration_const: float = 2.0) -> None:
        self._arms: dict[str, SlidingWindowUCB] = {}
        self._template_ids: set[str] = set()
        self._family_ids: set[str] = set()
        self._window_size = window_size
        self._decay_factor = decay_factor
        self._exploration_const = exploration_const

    def _arm_key(self, template_id: str, family_id: str) -> str:
        return f"{template_id}::{family_id}"

    def _parse_arm_key(self, key: str) -> tuple[str, str]:
        parts = key.split("::", 1)
        return parts[0], parts[1] if len(parts) > 1 else ""

    @algo_log()
    def ensure_arm(self, template_id: str, family_id: str) -> SlidingWindowUCB:
        key = self._arm_key(template_id, family_id)
        self._template_ids.add(template_id)
        self._family_ids.add(family_id)
        if key not in self._arms:
            self._arms[key] = SlidingWindowUCB(
                arm_id=key,
                window_size=self._window_size,
                decay_factor=self._decay_factor,
                exploration_const=self._exploration_const,
            )
        return self._arms[key]

    @algo_log()
    def select(self, explore_mode: bool = False) -> dict[str, Any] | None:
        if not self._arms:
            return None
        if explore_mode:
            scores = {aid: arm.ucb_score for aid, arm in self._arms.items()}
            unvisited = [aid for aid in self._arms if self._arms[aid].total_pulls == 0]
            if unvisited:
                key = unvisited[0]
            else:
                key = min(scores, key=lambda k: cast(float, scores[k]))
        else:
            scores = {aid: arm.ucb_score for aid, arm in self._arms.items()}
            key = max(scores, key=lambda k: cast(float, scores[k]))
        template_id, family_id = self._parse_arm_key(key)
        sw_arm = self._arms.get(key)
        score = sw_arm.ucb_score if sw_arm else 0.5
        return {
            "template_id": template_id,
            "family_id": family_id,
            "arm_key": key,
            "ucb_score": score,
            "visits": sw_arm.total_pulls if sw_arm else 0,
        }

    @algo_log()
    def select_top_k(self, k: int, explore_mode: bool = False) -> list[dict[str, Any]]:
        if not self._arms:
            return []
        scores = {aid: arm.ucb_score for aid, arm in self._arms.items()}
        if explore_mode:
            unvisited_keys = {aid for aid in self._arms if self._arms[aid].total_pulls == 0}
            sorted_arms = sorted(scores, key=lambda k: (0 if k in unvisited_keys else 1, -cast(float, scores[k])))
        else:
            sorted_arms = sorted(scores, key=lambda k: -cast(float, scores[k]))
        results = []
        for key in sorted_arms:
            template_id, family_id = self._parse_arm_key(key)
            sw_arm = self._arms.get(key)
            results.append({
                "template_id": template_id,
                "family_id": family_id,
                "arm_key": key,
                "ucb_score": scores[key],
                "visits": sw_arm.total_pulls if sw_arm else 0,
            })
            if len(results) >= k:
                break
        return results

    @algo_log()
    def update(self, template_id: str, family_id: str, reward: float = 0.0, penalty: float = 0.0) -> None:
        key = self._arm_key(template_id, family_id)
        sw_arm = self.ensure_arm(template_id, family_id)
        if reward > 0:
            sw_arm.update(reward)
        if penalty > 0:
            sw_arm.update(-penalty)

    def get_stats(self) -> dict[str, dict[str, Any]]:
        stats = {}
        for key, sw_arm in self._arms.items():
            template_id, family_id = self._parse_arm_key(key)
            stats[key] = {
                "template_id": template_id,
                "family_id": family_id,
                "ucb_score": sw_arm.ucb_score,
                "expectation": sw_arm.expectation,
                "visits": sw_arm.total_pulls,
                "window_count": sw_arm.window_count,
                "prior_weight": sw_arm._prior_weight,
            }
        return stats

    def get_top_arms(self, top_k: int = 10) -> list[dict[str, Any]]:
        if not self._arms:
            return []
        scores = {aid: arm.ucb_score for aid, arm in self._arms.items()}
        sorted_arms = sorted(scores, key=lambda k: -cast(float, scores[k]))
        results = []
        for key in sorted_arms[:top_k]:
            template_id, family_id = self._parse_arm_key(key)
            sw_arm = self._arms.get(key)
            results.append({
                "template_id": template_id,
                "family_id": family_id,
                "arm_key": key,
                "ucb_score": scores[key],
                "expectation": sw_arm.expectation if sw_arm else 0.0,
                "visits": sw_arm.total_pulls if sw_arm else 0,
            })
        return results

    @property
    def arm_count(self) -> int:
        return len(self._arms)

    def set_prior_for_arm(self, template_id: str, family_id: str, prior_score: float) -> None:
        sw_arm = self.ensure_arm(template_id, family_id)
        sw_arm.set_prior(prior_score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "arms": {key: arm.to_dict() for key, arm in self._arms.items()},
            "template_ids": list(self._template_ids),
            "family_ids": list(self._family_ids),
            "window_size": self._window_size,
            "decay_factor": self._decay_factor,
            "exploration_const": self._exploration_const,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TemplateFamilyBandit:
        tf = cls(
            window_size=d.get("window_size", 20),
            decay_factor=d.get("decay_factor", 0.95),
            exploration_const=d.get("exploration_const", 2.0),
        )
        arms_data = d.get("arms", {})
        for key, arm_data in arms_data.items():
            tf._arms[key] = SlidingWindowUCB.from_dict(arm_data)
        tf._template_ids = set(d.get("template_ids", []))
        tf._family_ids = set(d.get("family_ids", []))
        return tf

    def health_check(self) -> dict[str, Any]:
        return {
            "module": "TemplateFamilyBandit",
            "status": "active",
            "algorithm": "SlidingWindowUCB (Non-stationary)",
            "total_arms": len(self._arms),
            "unique_templates": len(self._template_ids),
            "unique_families": len(self._family_ids),
        }

    def select_with_direction(self, explore_mode: bool = False, focus_area: str | None = None) -> dict[str, Any] | None:
        if focus_area:
            direction_arms = [
                aid for aid in self._arms
                if _TEMPLATE_DIRECTION_MAP.get(self._parse_arm_key(aid)[0]) == focus_area
            ]
            if direction_arms:
                if explore_mode:
                    unvisited = [aid for aid in direction_arms if self._arms[aid].total_pulls == 0]
                    if unvisited:
                        key = unvisited[0]
                    else:
                        scores = {aid: self._arms[aid].ucb_score for aid in direction_arms}
                        key = min(scores, key=lambda k: cast(float, scores[k]))
                else:
                    scores = {aid: self._arms[aid].ucb_score for aid in direction_arms}
                    key = max(scores, key=lambda k: cast(float, scores[k]))
                template_id, family_id = self._parse_arm_key(key)
                sw_arm = self._arms.get(key)
                score = sw_arm.ucb_score if sw_arm else 0.5
                return {
                    "template_id": template_id,
                    "family_id": family_id,
                    "arm_key": key,
                    "ucb_score": score,
                    "visits": sw_arm.total_pulls if sw_arm else 0,
                    "direction": _TEMPLATE_DIRECTION_MAP.get(template_id, ""),
                }
        result = self.select(explore_mode)
        if result is None:
            return None
        result["direction"] = _TEMPLATE_DIRECTION_MAP.get(result["template_id"], "")
        return result

    def select_diverse_directions(self, k: int, explore_mode: bool = False) -> list[dict[str, Any]]:
        candidates = self.select_top_k(k * 3, explore_mode)
        seen_directions: set[str] = set()
        results: list[dict[str, Any]] = []
        for candidate in candidates:
            direction = _TEMPLATE_DIRECTION_MAP.get(candidate["template_id"], "")
            if direction and direction not in seen_directions:
                candidate["direction"] = direction
                seen_directions.add(direction)
                results.append(candidate)
                if len(results) >= k:
                    break
        return results

    def update_by_direction(self, direction: str, reward: float = 0.0, penalty: float = 0.0) -> None:
        matching_arms: list[tuple[str, str]] = []
        for aid in self._arms:
            template_id, family_id = self._parse_arm_key(aid)
            if _TEMPLATE_DIRECTION_MAP.get(template_id) == direction:
                matching_arms.append((template_id, family_id))
        if not matching_arms:
            return
        n = len(matching_arms)
        for template_id, family_id in matching_arms:
            self.update(template_id, family_id, reward=reward / n, penalty=penalty / n)

    def get_direction_stats(self) -> dict[str, dict[str, Any]]:
        direction_data: dict[str, list[tuple[str, str, float, int]]] = {}
        for key, sw_arm in self._arms.items():
            template_id, family_id = self._parse_arm_key(key)
            direction = _TEMPLATE_DIRECTION_MAP.get(template_id, "")
            if not direction:
                continue
            if direction not in direction_data:
                direction_data[direction] = []
            direction_data[direction].append((template_id, family_id, sw_arm.expectation, sw_arm.total_pulls))
        result: dict[str, dict[str, Any]] = {}
        for direction, arms in direction_data.items():
            total_visits = sum(a[3] for a in arms)
            avg_expectation = sum(a[2] for a in arms) / len(arms)
            top_arm = max(arms, key=lambda a: a[2])
            result[direction] = {
                "avg_expectation": avg_expectation,
                "total_visits": total_visits,
                "arm_count": len(arms),
                "top_template": top_arm[0],
                "top_family": top_arm[1],
            }
        return result

    def initialize_arms_from_library(self, template_ids: list[str], family_ids: list[str]) -> None:
        for template_id in template_ids:
            for family_id in family_ids:
                self.ensure_arm(template_id, family_id)

    def initialize_priors(self, prior_initializer: MABPriorInitializer) -> None:
        logger.info("[DEFENSIVE_LOG] TemplateFamilyBandit::initialize_priors 开始设置 %d 个 arm 的先验", len(self._arms))
        count = 0
        for key in list(self._arms.keys()):
            template_id, family_id = self._parse_arm_key(key)
            prior = prior_initializer.compute_prior(template_id, family_id)
            self._arms[key].set_prior(prior)
            count += 1
        logger.info("[DEFENSIVE_LOG] TemplateFamilyBandit::initialize_priors 完成，共设置 %d 个先验", count)


REWARD_VALIDATOR_PASS = getattr(settings, 'MAB_REWARD_VALIDATOR_PASS', 0.1)
REWARD_BRAIN_SUBMIT = getattr(settings, 'MAB_REWARD_BRAIN_SUBMIT', 0.3)
REWARD_SHARPE_05 = getattr(settings, 'MAB_REWARD_SHARPE_05', 0.5)
REWARD_SHARPE_10 = getattr(settings, 'MAB_REWARD_SHARPE_10', 1.0)
PENALTY_BRAIN_FAIL = getattr(settings, 'MAB_PENALTY_BRAIN_FAIL', 0.3)
PENALTY_BRAIN_ERROR = getattr(settings, 'MAB_PENALTY_BRAIN_ERROR', 0.5)
PENALTY_OVERUSE = getattr(settings, 'MAB_PENALTY_OVERUSE', 0.1)


def save_mab_state(
    mab: HierarchicalMAB,
    association: AssociationMatrix,
    whitelist_data: dict[str, Any],
    path: str | Path | None = None,
    arbiter_data: dict[str, Any] | None = None,
    template_family_bandit: TemplateFamilyBandit | None = None,
) -> None:
    path = Path(path) if path else _MAB_STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "hierarchical_mab": mab.to_dict(),
        "association_matrix": association.to_dict(),
        "whitelist": whitelist_data,
    }
    if arbiter_data is not None:
        state["signal_arbiter"] = arbiter_data
    if template_family_bandit is not None:
        state["template_family_bandit"] = template_family_bandit.to_dict()
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("MAB state saved → %s", path)


def load_mab_state(
    path: str | Path | None = None,
) -> tuple[HierarchicalMAB, AssociationMatrix, dict[str, Any], dict[str, Any] | None, TemplateFamilyBandit | None, dict[str, Any] | None] | None:
    path = Path(path) if path else _MAB_STATE_PATH
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        mab = HierarchicalMAB.from_dict(data.get("hierarchical_mab", {}))
        association = AssociationMatrix.from_dict(data.get("association_matrix", {}))
        whitelist_data = data.get("whitelist", {})
        arbiter_data = data.get("signal_arbiter")
        tf_bandit = None
        if "template_family_bandit" in data:
            tf_bandit = TemplateFamilyBandit.from_dict(data["template_family_bandit"])
        scheduler_data = data.get("scheduler")
        logger.info("MAB state loaded ← %s", path)
        return mab, association, whitelist_data, arbiter_data, tf_bandit, scheduler_data
    except (ValueError, TypeError, OSError) as exc:
        logger.error("Failed to load MAB state: %s", exc)
        return None
