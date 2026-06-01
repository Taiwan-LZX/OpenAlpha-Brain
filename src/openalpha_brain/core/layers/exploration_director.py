"""
OpenAlpha-Brain ExplorationDirector — Layer 1 of 6-Layer Architecture
====================================================================
探索方向选择总控模块，封装原本散落在 loop_engine.py 中的方向选择逻辑。

职责:
  1. 协调四个子算法的探索方向决策:
     - NavigationFusion (三模块投票融合)
     - HierarchicalMAB (UCB 智能选择)
     - MAP-Elites / QD (多样性探索)
     - ToTSearchStrategy (树搜索改进)
  2. 统一反馈更新接口 (MAB + Scheduler)
  3. 提供 ExplorationResult 结构化输出

架构位置 (Layer 1):
  ┌─────────────────────────────────────┐
  │        ExplorationDirector          │  ← 本模块
  │  ┌──────────┐ ┌────────┐ ┌──────┐ │
  │  │NavFusion │ │ HMAB   │ │ToT   │ │
  │  └──────────┘ └────────┘ └──────┘ │
  │  ┌──────────┐ ┌──────────────────┐ │
  │  │MAP-Elites│ │TemplateFamilyBandit│ │
  │  └──────────┘ └──────────────────┘ │
  └─────────────────┬───────────────────┘
                    ↓
         selected exploration_direction

提取来源 (loop_engine.py):
  - L387-459: 方向选择主逻辑 (MAB → MAP-Elites → 随机降级)
  - L1881-1905: HierarchicalMAB.update() 反馈
  - L1907-1950: ToT Search 近 PASS 触发
  - L2144-2272: NavigationFusion.fuse() 融合决策

Usage:
    director = ExplorationDirector(config={"fusion_override_threshold": 0.7})
    result = await director.select_direction(
        feature_map=_ls._feature_map,
        scheduler=_ls._scheduler,
        hmab=_ls._hmab,
        nav_fusion=_ls._nav_fusion,
        tot_search=_ls._tot_search,
        session_id="sess_001",
        cycle_num=42,
    )
    print(result.direction, result.confidence, result.method)

    director.update_feedback(
        direction="momentum",
        brain_result=brain_result,
        expression="ts_decay_linear(rank(close/volume), sector), 10)",
        hmab=_ls._hmab,
        scheduler=_ls._scheduler,
    )
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

from openalpha_brain.core.navigation_fusion import (
    AlignerOutput,
    ClassifierOutput,
    FusionResult,
    MABOutput,
    NavigationFusion,
)
from openalpha_brain.evolution.near_pass_improver import NEAR_PASS_SHARPE_THRESHOLD
from openalpha_brain.evolution.tot_search import ToTSearchResult, ToTSearchStrategy
from openalpha_brain.learning.mab import (
    PENALTY_BRAIN_ERROR,
    PENALTY_BRAIN_FAIL,
    HierarchicalMAB,
)
from openalpha_brain.learning.reward_updater import (
    _extract_strategy_features,
    _get_fields_from_context,
    _get_operators_from_context,
)
from openalpha_brain.services.brain_submitter import BrainSimStatus

logger = logging.getLogger(__name__)

_EPSILON = 1e-6
_AUTO_DIRECTION_VALUES = frozenset({"auto", "", "none", "default"})


@dataclass
class ExplorationResult:
    """探索方向选择的完整结果"""

    direction: str = ""
    confidence: float = 0.0
    method: str = "fallback"
    mab_output: MABOutput | None = None
    fusion_result: FusionResult | None = None
    tot_result: Any | None = None
    template_id: str = ""
    family_id: str = ""
    recommended_fields: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ExplorationDirector:
    """探索方向选择总控器 — Layer 1

    将 loop_engine.py 中分散的方向选择、MAB 反馈更新、ToT 触发、
    NavigationFusion 融合等逻辑统一收口为一个清晰的 API。

    Decision Pipeline (select_direction):
      1. Base direction ← focus_area 或 DEFAULT
      2. TemplateFamilyBandit.select_exploration_arm() ← UCB 选择
      3. MAP-Elites schedule check ← explore/exploit 切换
      4a. Explore path:
          - HierarchicalMAB.select() ← 算子/字段级 UCB
          - Unexplored directions ← 随机探索未覆盖区域
          - Explore targets ← QD 目标单元格
      4b. Exploit path ← 利用已知高绩效方向
      5. (post-cycle) NavigationFusion.fuse() ← 三模块融合覆盖
      6. (post-result) ToT search ← 近 PASS 树搜索改进

    Feedback Pipeline (update_feedback):
      1. HierarchicalMAB.update() ← reward/penalty 反向传播
      2. TemplateFamilyBandit.record_direction_result() ← 方向级奖励
      3. FeatureMap.add_candidate() ← QD 存档 (可选)
    """

    DEFAULT_CONFIG: dict[str, Any] = {
        "fusion_override_threshold": 0.7,
        "near_pass_enabled": True,
        "mab_select_timeout_sec": 5.0,
        "tot_search_timeout_sec": 300.0,
    }

    def __init__(self, config: dict[str, Any] | None = None):
        self.config: dict[str, Any] = {**self.DEFAULT_CONFIG, **(config or {})}
        self._selection_count: int = 0
        self._update_count: int = 0
        self._last_direction: str = ""
        self._method_stats: dict[str, int] = {}

    def resolve_effective_focus(self, focus_area: str | None) -> str:
        if not focus_area or focus_area.lower() in _AUTO_DIRECTION_VALUES:
            return ""
        return focus_area

    async def select_direction(
        self,
        feature_map: Any,
        scheduler: Any,
        hmab: HierarchicalMAB | None,
        nav_fusion: NavigationFusion | None,
        tot_search: ToTSearchStrategy | None,
        session_id: str,
        cycle_num: int,
        focus_area: str | None = None,
        current_direction: str = "",
        explore_weight: float = 0.3,
    ) -> ExplorationResult:
        """执行完整的探索方向选择流水线

        Args:
            feature_map: _ls._feature_map (MAP-Elites 特征图)
            scheduler: _ls._scheduler (TemplateFamilyBandit 包装器)
            hmab: _ls._hmab (HierarchicalMAB 实例, 可为 None)
            nav_fusion: _ls._nav_fusion (NavigationFusion 实例, 可为 None)
            tot_search: _ls._tot_search (ToTSearchStrategy 实例, 可为 None)
            session_id: 当前会话 ID
            cycle_num: 当前循环编号
            focus_area: 用户指定的关注领域
            current_direction: 当前正在使用的方向
            explore_weight: 探索权重 (0-1)

        Returns:
            ExplorationResult: 包含方向、置信度、决策方法等完整信息
        """
        self._selection_count += 1
        result = ExplorationResult()
        effective_focus = self.resolve_effective_focus(focus_area)

        try:
            from openalpha_brain.config.config import settings

            default_direction = effective_focus or getattr(settings, "DEFAULT_EXPLORATION_DIRECTION", "")
            mab_enabled = getattr(settings, "MAB_ENABLED", True)
        except (ImportError, AttributeError, OSError):
            default_direction = effective_focus or "momentum"
            mab_enabled = True

        exploration_direction = default_direction
        method = "fallback"
        sched_template_id = ""
        sched_family_id = ""
        sched_recommended_fields: list[str] = []

        if mab_enabled and scheduler is not None:
            try:
                sched_result = scheduler.select_exploration_arm(
                    focus_area=effective_focus,
                    explore_mode=False,
                )
                if sched_result:
                    exploration_direction = sched_result.get("direction", exploration_direction)
                    sched_template_id = sched_result.get("template_id", "")
                    sched_family_id = sched_result.get("family_id", "")
                    sched_recommended_fields = sched_result.get("recommended_fields", [])
                    method = "mab_ucb"
            except (OSError, ValueError, RuntimeError) as exc:
                logger.debug(
                    "[%s] cycle=%d scheduler select_exploration_arm failed: %s",
                    session_id,
                    cycle_num,
                    exc,
                )

        map_strategy = "exploit"
        actual_explore_weight = explore_weight
        unexplored: list[str] = []

        if feature_map is not None:
            try:
                schedule = feature_map.get_explore_exploit_schedule()
                map_strategy = schedule.get("strategy", "exploit")
                actual_explore_weight = schedule.get("explore_weight", explore_weight)
                unexplored = feature_map.get_unexplored_directions()
            except (OSError, ValueError, RuntimeError, AttributeError) as exc:
                logger.debug("[%s] cycle=%d feature_map schedule failed: %s", session_id, cycle_num, exc)

        hmab_used = False
        if map_strategy == "explore":
            if mab_enabled and hmab is not None:
                try:
                    hmab_result = hmab.select(
                        top_k_ops=15,
                        top_k_fields=40,
                        focus_area=effective_focus,
                    )
                    if hmab_result.get("direction"):
                        exploration_direction = hmab_result["direction"]
                        hmab_used = True
                        method = "hierarchical_mab"
                        logger.info(
                            "[%s] MAB-HIER: selected direction=%s (UCB) operators=%d fields=%d",
                            session_id,
                            exploration_direction,
                            len(hmab_result.get("operators", [])),
                            len(hmab_result.get("fields", [])),
                        )
                except (OSError, ValueError, RuntimeError) as exc:
                    logger.debug(
                        "[%s] cycle=%d HierarchicalMAB.select failed: %s",
                        session_id,
                        cycle_num,
                        exc,
                    )

            if not hmab_used and unexplored and random.random() < actual_explore_weight:
                exploration_direction = random.choice(unexplored)
                method = "map_elites"
                logger.info(
                    "[%s] MAP-Elites EXPLORE: pivoting to unexplored direction=%s (weight=%.2f)",
                    session_id,
                    exploration_direction,
                    actual_explore_weight,
                )

            if not hmab_used and feature_map is not None:
                try:
                    explore_targets = feature_map.get_explore_targets(top_k=3)
                    if explore_targets:
                        target = random.choice(explore_targets)
                        target_dir = target.get("direction", "")
                        if target_dir:
                            exploration_direction = target_dir
                            method = "map_elites_target"
                        logger.info(
                            "[%s] MAP-Elites EXPLORE_TARGET: cell=%s direction=%s",
                            session_id,
                            target.get("key", ""),
                            target_dir,
                        )
                except (OSError, ValueError, RuntimeError, AttributeError):
                    pass

        mab_out = MABOutput(
            selected_family=exploration_direction,
            confidence=0.5,
            exploration_rate=actual_explore_weight,
        )

        # ── MAB 方向权重融合 ──
        mab_direction_weight = 0.5
        if hmab is not None and mab_enabled:
            try:
                direction_stats = hmab.get_direction_stats()
                dir_arm_stats = direction_stats.get(exploration_direction, {})
                mab_expectation = dir_arm_stats.get("expectation", 0.5)
                mab_direction_weight = max(_EPSILON, min(1.0, mab_expectation))

                nav_fusion_score = mab_out.confidence
                final_score = nav_fusion_score * 0.6 + mab_direction_weight * 0.4
                mab_out.confidence = max(_EPSILON, min(1.0, final_score))

                logger.info(
                    "[DEFENSIVE_LOG] EXPLORATION_DIRECTOR::MAB_WEIGHT_FUSION "
                    "direction=%s nav_fusion=%.3f mab_weight=%.3f final_score=%.3f",
                    exploration_direction,
                    nav_fusion_score,
                    mab_direction_weight,
                    mab_out.confidence,
                )
            except (OSError, ValueError, RuntimeError) as exc:
                logger.debug(
                    "[%s] cycle=%d MAB direction stats fusion failed: %s",
                    session_id,
                    cycle_num,
                    exc,
                )

        result.direction = exploration_direction
        result.confidence = mab_out.confidence
        result.method = method
        result.mab_output = mab_out
        result.template_id = sched_template_id
        result.family_id = sched_family_id
        result.recommended_fields = sched_recommended_fields
        result.metadata = {
            "map_strategy": map_strategy,
            "explore_weight": actual_explore_weight,
            "hmab_used": hmab_used,
            "unexplored_count": len(unexplored),
            "effective_focus": effective_focus,
        }
        self._last_direction = exploration_direction
        self._method_stats[method] = self._method_stats.get(method, 0) + 1

        logger.info(
            "[%s] EXPLORATION_DIRECTOR: cycle=%d direction=%s method=%s confidence=%.2f "
            "template=%s family=%s fields=%d",
            session_id,
            cycle_num,
            exploration_direction,
            method,
            result.confidence,
            sched_template_id or "-",
            sched_family_id or "-",
            len(sched_recommended_fields),
        )
        return result

    def apply_fusion_override(
        self,
        current_result: ExplorationResult,
        classifier_output: ClassifierOutput | None,
        aligner_output: AlignerOutput | None,
        nav_fusion: NavigationFusion,
        session_id: str,
        cycle_num: int,
    ) -> ExplorationResult:
        """在获得 BRAIN 结果后执行 NavigationFusion 融合覆盖

        这是 post-cycle 步骤，在 select_direction 之后、下一轮之前调用。
        如果融合置信度足够高且方向不同，则覆盖当前方向。

        Args:
            current_result: 当前已选的探索结果
            classifier_output: StrategyClassifier 输出 (可为 None)
            aligner_output: HypothesisAligner 输出 (可为 None)
            nav_fusion: NavigationFusion 实例
            session_id: 会话 ID
            cycle_num: 循环编号

        Returns:
            可能被融合覆盖后的 ExplorationResult
        """
        if current_result.mab_output is None:
            current_result.mab_output = MABOutput(
                selected_family=current_result.direction,
                confidence=current_result.confidence,
                exploration_rate=current_result.metadata.get("explore_weight", 0.3),
            )
        try:
            fusion_result = nav_fusion.fuse(
                mab_output=current_result.mab_output,
                classifier_output=classifier_output,
                aligner_output=aligner_output,
                cycle_num=cycle_num,
            )
            current_result.fusion_result = fusion_result
            logger.info(
                "[%s] NAV_FUSION: direction=%s confidence=%.2f strategy=%s disagreement=%.2f votes=%s",
                session_id,
                fusion_result.final_direction,
                fusion_result.confidence,
                fusion_result.strategy.value,
                fusion_result.disagreement_score,
                fusion_result.raw_votes,
            )
            override_threshold = self.config.get("fusion_override_threshold", 0.7)
            if (
                fusion_result.confidence > override_threshold
                and fusion_result.final_direction
                and fusion_result.final_direction != current_result.direction
            ):
                logger.info(
                    "[%s] NAV_FUSION OVERRIDE: %s → %s (confidence=%.2f)",
                    session_id,
                    current_result.direction,
                    fusion_result.final_direction,
                    fusion_result.confidence,
                )
                old_direction = current_result.direction
                current_result.direction = fusion_result.final_direction
                current_result.confidence = fusion_result.confidence
                current_result.method = "nav_fusion"
                current_result.metadata["fusion_override_from"] = old_direction
        except (OSError, ValueError, RuntimeError) as exc:
            logger.warning("[%s] NavigationFusion failed: %s", session_id, exc, exc_info=True)
        return current_result

    async def trigger_tot_search_if_near_pass(
        self,
        brain_result: Any,
        expression: str,
        tot_search: ToTSearchStrategy | None,
        session_id: str,
        cycle_num: int,
    ) -> tuple[str, ToTSearchResult | None]:
        """检测近 PASS 条件并触发 ToT 树搜索

        当 alpha 未通过但 Sharpe 接近阈值时，启动树搜索尝试找到更优变体。

        Args:
            brain_result: BRAIN 平台返回结果 (需有 status, real_sharpe, real_fitness 属性)
            expression: 当前因子表达式
            tot_search: ToTSearchStrategy 实例 (可为 None)
            session_id: 会话 ID
            cycle_num: 循环编号

        Returns:
            tuple: (可能改进后的表达式, ToT 搜索结果或 None)
        """
        if tot_search is None:
            return expression, None
        if not self.config.get("near_pass_enabled", True):
            return expression, None
        is_near_pass = (
            brain_result.status != BrainSimStatus.PASS
            and brain_result.status != BrainSimStatus.ERROR
            and (brain_result.real_sharpe or 0.0) >= NEAR_PASS_SHARPE_THRESHOLD
        )
        if not is_near_pass:
            return expression, None
        try:
            logger.info(
                "[%s] TOT-TRIGGER: near-pass detected (sharpe=%.3f) starting tree search",
                session_id,
                brain_result.real_sharpe,
            )
            tot_result = await tot_search.search(
                seed_expression=expression or "",
                target_fitness=1.25,
                initial_fitness=brain_result.real_fitness or 0.0,
                context={"session_id": session_id, "cycle": cycle_num},
            )
            if tot_result.best_node is not None and (tot_result.best_fitness or 0.0) > (
                brain_result.real_fitness or 0.0
            ):
                improved_expression = tot_result.best_node.expression or expression
                logger.info(
                    "[%s] TOT-IMPROVED: fitness %.3f→%.3f nodes=%d depth=%d",
                    session_id,
                    brain_result.real_fitness or 0.0,
                    tot_result.best_fitness or 0.0,
                    tot_result.total_nodes_explored,
                    tot_result.total_depth_reached,
                )
                return improved_expression, tot_result
            logger.info(
                "[%s] TOT-NO-IMPROVEMENT: best_fitness=%.3f nodes=%d duration=%.1fs",
                session_id,
                tot_result.best_fitness or 0.0,
                tot_result.total_nodes_explored,
                tot_result.search_duration_sec,
            )
            return expression, tot_result
        except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
            logger.warning(
                "[%s] TOT search failed, using original expression: %s",
                session_id,
                exc,
                exc_info=True,
            )
            return expression, None

    def update_feedback(
        self,
        direction: str,
        brain_result: Any,
        expression: str,
        hmab: HierarchicalMAB | None,
        scheduler: Any,
        feature_map: Any | None = None,
    ) -> None:
        """将 BRAIN 结果反馈到 MAB 和 Scheduler

        封装 loop_engine.py L1881-1905 的 MAB.update() 逻辑，
        以及 scheduler.record_direction_result() 和 feature_map.add_candidate()。

        Args:
            direction: 本次使用的探索方向
            brain_result: BRAIN 返回结果 (需有 status, real_sharpe 属性)
            expression: 提交的表达式
            hmab: HierarchicalMAB 实例 (可为 None)
            scheduler: TemplateFamilyBandit 包装器 (可为 None)
            feature_map: 特征图实例 (可为 None, 用于 QD 存档)
        """
        self._update_count += 1
        if hmab is not None:
            try:
                hmab_reward = brain_result.real_sharpe or 0.0
                hmab_penalty = PENALTY_BRAIN_FAIL if brain_result.status != BrainSimStatus.PASS else 0.0
                if brain_result.status == BrainSimStatus.ERROR:
                    hmab_penalty = PENALTY_BRAIN_ERROR
                hmab_ops = _get_operators_from_context({"expression": expression}) if expression else []
                hmab_fields = _get_fields_from_context({"expression": expression}) if expression else []
                hmab.update(
                    direction=direction,
                    operators=hmab_ops or [],
                    fields=hmab_fields or [],
                    reward=hmab_reward,
                    penalty=hmab_penalty,
                )
                logger.info(
                    "MAB-HIER-UPDATE: direction=%s reward=%.3f penalty=%.3f ops=%d fields=%d",
                    direction,
                    hmab_reward,
                    hmab_penalty,
                    len(hmab_ops or []),
                    len(hmab_fields or []),
                )
            except (OSError, ValueError, RuntimeError):
                logger.debug("HierarchicalMAB.update failed", exc_info=True)

        if scheduler is not None:
            try:
                if brain_result.status == BrainSimStatus.ERROR:
                    scheduler.record_direction_result(direction, penalty=PENALTY_BRAIN_ERROR)
                elif brain_result.status != BrainSimStatus.PASS:
                    scheduler.record_direction_result(direction, penalty=PENALTY_BRAIN_FAIL)
            except (OSError, ValueError, RuntimeError):
                logger.debug("Scheduler.record_direction_result failed", exc_info=True)

        if feature_map is not None and brain_result.status == BrainSimStatus.PASS:
            try:
                feat = _extract_strategy_features(expression, direction)
                feature_map.add_candidate(
                    expression,
                    feat,
                    fitness_score=brain_result.real_fitness or 0.0,
                    sharpe=brain_result.real_sharpe or 0.0,
                    turnover=getattr(brain_result, "real_turnover", 0.0) or 0.0,
                )
            except (OSError, ValueError, RuntimeError, AttributeError):
                logger.debug("FeatureMap.add_candidate failed", exc_info=True)

    def get_stats(self) -> dict[str, Any]:
        """返回探索导演器的统计信息"""
        total = max(self._selection_count, 1)
        method_distribution = {k: round(v / total, 3) for k, v in self._method_stats.items()}
        return {
            "total_selections": self._selection_count,
            "total_updates": self._update_count,
            "last_direction": self._last_direction,
            "method_distribution": method_distribution,
            "config": {k: v for k, v in self.config.items() if k != "fusion_override_threshold"},
        }

    def reset_stats(self) -> None:
        """重置统计计数器"""
        self._selection_count = 0
        self._update_count = 0
        self._method_stats.clear()
        self._last_direction = ""
