"""
OpenAlpha-Brain NavigationFusion
================================
MAB / StrategyClassifier / HypothesisAligner 投票融合引擎

职责:
  1. 接收三个导航模块的独立输出 (MAB权重/Classifier方向/Aligner偏移)
  2. 基于动态权重进行加权融合
  3. 输出最终探索方向 + 置信度 + 融合元数据

解决的问题:
  - 原问题: 三个导航模块各自为战，缺乏协同
  - 现象: 算法利用率 <30%
  - 目标: 提升到 50%+ 通过智能融合

架构位置:
  ┌──────────┐  ┌─────────────────┐  ┌────────────────┐
  │   MAB    │  │StrategyClassifier│  │HypothesisAligner│
  └────┬─────┘  └────────┬────────┘  └────────┬────────┘
       │                │                   │
       └────────────────┼───────────────────┘
                        ↓
              ┌──────────────────┐
              │ NavigationFusion  │
              │  (本模块)         │
              └────────┬─────────┘
                       ↓
              最终探索方向 (exploration_direction)

Usage:
    fusion = NavigationFusion(config={
        "mab_weight": 0.4,
        "classifier_weight": 0.3,
        "aligner_weight": 0.3,
    })

    result = fusion.fuse(
        mab_output=MABOutput(...),
        classifier_output=ClassifierOutput(...),
        aligner_output=AlignerOutput(...),
        cycle_num=42,
    )

    print(result.final_direction)     # "momentum"
    print(result.confidence)           # 0.78
    print(result.weight_distribution)   # {"mab": 0.45, "classifier": 0.28, "aligner": 0.27}
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """融合策略"""

    WEIGHTED_AVERAGE = "weighted_average"  # 加权平均 (默认)
    MAJORITY_VOTE = "majority_vote"  # 多数投票
    CONFIDENCE_WEIGHTED = "confidence_weighted"  # 置信度加权
    WINNER_TAKES_ALL = "winner_takes_all"  # 最高置信度胜出


@dataclass
class MABOutput:
    """MAB (TemplateFamilyBandit) 输出"""

    selected_family: str = ""  # 选中的模板家族 (momentum/reversal/mean_reversion等)
    weight_vector: dict[str, float] = field(default_factory=dict)  # 完整权重向量
    confidence: float = 0.5  # UCB置信度或exploitation概率
    exploration_rate: float = 0.3  # 当前探索率

    @property
    def is_valid(self) -> bool:
        return len(self.selected_family) > 0 and self.confidence > 0


@dataclass
class ClassifierOutput:
    """StrategyClassifier 输出"""

    direction: str = ""  # 主导方向
    confidence: float = 0.5  # 分类置信度
    market_state: str = ""  # 市场状态 (trending/ranging/volatile)
    is_ambiguous: bool = False  # 是否模糊/不确定
    is_tied: bool = False  # 是否平局
    calibration_details: dict = field(default_factory=dict)  # LLM校准详情

    @property
    def is_valid(self) -> bool:
        return len(self.direction) > 0 and self.confidence > 0


@dataclass
class AlignerOutput:
    """HypothesisAligner 输出"""

    alignment_level: str = ""  # 对齐等级 (aligned/weak/contradictory)
    r2_score: float = 0.0  # R² 对齐分数
    offset_magnitude: float = 0.0  # 偏移量大小
    suggested_adjustment: str = ""  # 建议的方向调整
    diagnosis: str = ""  # 诊断信息

    @property
    def is_valid(self) -> bool:
        return len(self.alignment_level) > 0 and self.r2_score >= 0

    @property
    def is_aligned(self) -> bool:
        return self.alignment_level in ("aligned", "strongly_aligned")

    @property
    def alignment_strength(self) -> float:
        if self.r2_score >= 0.7:
            return 1.0
        elif self.r2_score >= 0.5:
            return 0.75
        elif self.r2_score >= 0.3:
            return 0.5
        else:
            return 0.25


@dataclass
class FusionResult:
    """融合结果"""

    final_direction: str = ""  # 最终探索方向
    confidence: float = 0.0  # 融合后综合置信度 (0-1)
    strategy: FusionStrategy = FusionStrategy.WEIGHTED_AVERAGE
    weight_distribution: dict[str, float] = field(default_factory=dict)
    raw_votes: dict[str, str] = field(default_factory=dict)  # 各模块原始投票
    disagreement_score: float = 0.0  # 模块间不一致程度 (0-1, 高=冲突)
    processing_time_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def has_consensus(self) -> bool:
        """三个模块是否达成一致"""
        votes = list(self.raw_votes.values())
        unique_votes = set(v for v in votes if v)
        return len(unique_votes) == 1 or self.disagreement_score < 0.3

    def to_log_dict(self) -> dict:
        return {
            "direction": self.final_direction,
            "confidence": round(self.confidence, 3),
            "weights": {k: round(v, 2) for k, v in self.weight_distribution.items()},
            "disagreement": round(self.disagreement_score, 3),
            "consensus": self.has_consensus,
            "time_ms": round(self.processing_time_ms, 2),
        }


class NavigationFusion:
    """导航融合引擎 — 三模块协同决策器

    将 MAB、StrategyClassifier、HypothesisAligner 的独立输出
    融合为统一的探索方向，通过动态权重提升算法利用率。

    Design Principles:
    - 容错性: 任一模块失效时自动降级
    - 自适应: 根据历史表现动态调整权重
    - 可观测性: 记录每个决策的完整元数据
    - 冲突检测: 当模块间不一致时发出警告
    """

    DEFAULT_CONFIG = {
        "base_weights": {
            "mab": 0.4,
            "classifier": 0.3,
            "aligner": 0.3,
        },
        "fusion_strategy": "weighted_average",
        "min_confidence_threshold": 0.3,  # 低于此值的模块降权
        "disagreement_penalty": 0.15,  # 不一致时的惩罚系数
        "history_window": 20,  # 历史成功率计算窗口
        "adaptive_enabled": True,  # 是否启用自适应权重调整
    }

    def __init__(
        self,
        config: dict | None = None,
    ):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self._success_history: dict[str, list[bool]] = {
            "mab": [],
            "classifier": [],
            "aligner": [],
        }

    def fuse(
        self,
        mab_output: MABOutput | None = None,
        classifier_output: ClassifierOutput | None = None,
        aligner_output: AlignerOutput | None = None,
        cycle_num: int = 0,
    ) -> FusionResult:
        """执行三模块融合

        Args:
            mab_output: MAB输出 (可为None，表示不可用)
            classifier_output: Classifier输出 (可为None)
            aligner_output: Aligner输出 (可为None)
            cycle_num: 当前循环编号 (用于日志)

        Returns:
            FusionResult: 融合后的最终方向和置信度
        """
        t0 = time.perf_counter()

        result = FusionResult()

        # Step 1: 收集各模块的"投票"
        votes = self._collect_votes(mab_output, classifier_output, aligner_output)
        result.raw_votes = votes

        # Step 2: 检测可用性和有效性
        availability = self._check_availability(mab_output, classifier_output, aligner_output)

        # Step 3: 计算动态权重
        weights = self._compute_dynamic_weights(mab_output, classifier_output, aligner_output, availability)
        result.weight_distribution = weights

        # Step 4: 检测不一致性
        result.disagreement_score = self._compute_disagreement(votes)

        # Step 5: 执行融合策略
        strategy = FusionStrategy(self.config.get("fusion_strategy", "weighted_average"))
        result.strategy = strategy

        if strategy == FusionStrategy.WEIGHTED_AVERAGE:
            final_dir, conf = self._weighted_average_fuse(votes, weights, availability)
        elif strategy == FusionStrategy.CONFIDENCE_WEIGHTED:
            final_dir, conf = self._confidence_weighted_fuse(
                votes, mab_output, classifier_output, aligner_output, availability
            )
        elif strategy == FusionStrategy.MAJORITY_VOTE:
            final_dir, conf = self._majority_vote_fuse(votes, availability)
        else:  # winner_takes_all
            final_dir, conf = self._winner_takes_all_fuse(
                votes, mab_output, classifier_output, aligner_output, availability
            )

        result.final_direction = final_dir
        result.confidence = conf
        result.processing_time_ms = (time.perf_counter() - t0) * 1000
        result.metadata = {
            "cycle_num": cycle_num,
            "availability": availability,
            "valid_votes_count": sum(1 for v in votes.values() if v),
        }

        logger.info(
            "[FUSION] ◆ RESULT | cycle=%d direction=%s confidence=%.3f disagreement=%.3f consensus=%s weights=%s",
            cycle_num,
            final_dir,
            conf,
            result.disagreement_score,
            result.has_consensus,
            {k: round(v, 2) for k, v in weights.items()},
        )

        if not result.has_consensus and result.disagreement_score > 0.6:
            logger.warning(
                "[FUSION] ⚠ HIGH DISAGREEMENT | cycle=%d score=%.3f votes=%s",
                cycle_num,
                result.disagreement_score,
                votes,
            )

        return result

    def _collect_votes(
        self,
        mab: MABOutput | None,
        classifier: ClassifierOutput | None,
        aligner: AlignerOutput | None,
    ) -> dict[str, str]:
        """收集各模块的投票 (提取方向字符串)"""
        votes = {}

        if mab is not None and mab.is_valid:
            votes["mab"] = mab.selected_family

        if classifier is not None and classifier.is_valid:
            votes["classifier"] = classifier.direction

        if aligner is not None and aligner.is_valid:
            if aligner.is_aligned and aligner.suggested_adjustment:
                votes["aligner"] = aligner.suggested_adjustment
            elif aligner.is_aligned:
                votes["aligner"] = "continue_current"  # 对齐良好，保持当前方向
            else:
                votes["aligner"] = f"adjust_{aligner.alignment_level}"

        return votes

    def _check_availability(
        self,
        mab: MABOutput | None,
        classifier: ClassifierOutput | None,
        aligner: AlignerOutput | None,
    ) -> dict[str, bool]:
        """检查各模块是否可用且有效"""
        return {
            "mab": mab is not None and mab.is_valid,
            "classifier": classifier is not None and classifier.is_valid,
            "aligner": aligner is not None and aligner.is_valid,
        }

    def _compute_dynamic_weights(
        self,
        mab: MABOutput | None,
        classifier: ClassifierOutput | None,
        aligner: AlignerOutput | None,
        availability: dict[str, bool],
    ) -> dict[str, float]:
        """计算动态权重 (基础权重 × 可用性 × 置信度调整 × 历史成功率)"""
        base = self.config["base_weights"].copy()

        for module_name in ["mab", "classifier", "aligner"]:
            if not availability.get(module_name, False):
                base[module_name] = 0.0
                continue

            # 置信度调整
            conf = 0.5  # 默认中等置信度
            if module_name == "mab" and mab is not None:
                conf = max(mab.confidence, 0.3)
            elif module_name == "classifier" and classifier is not None:
                conf = max(classifier.confidence, 0.3)
            elif module_name == "aligner" and aligner is not None:
                conf = max(aligner.alignment_strength, 0.3)

            min_conf = self.config.get("min_confidence_threshold", 0.3)
            if conf < min_conf:
                base[module_name] *= conf / min_conf  # 低置信度降权

            # 历史成功率调整 (如果启用)
            if self.config.get("adaptive_enabled", True):
                success_rate = self._get_success_rate(module_name)
                if success_rate is not None:
                    base[module_name] *= 0.5 + success_rate  # 成功率高则增权

        # 归一化
        total = sum(base.values())
        if total > 0:
            base = {k: v / total for k, v in base.items()}

        return base

    def _compute_disagreement(self, votes: dict[str, str]) -> float:
        """计算模块间的不一致程度 (0=完全一致, 1=完全不同)"""
        valid_votes = [v for v in votes.values() if v]
        if len(valid_votes) <= 1:
            return 0.0

        from collections import Counter

        counts = Counter(valid_votes)
        most_common_count = counts.most_common(1)[0][1]

        # 不一致度 = 1 - (最多投票数 / 总投票数)
        return 1.0 - (most_common_count / len(valid_votes))

    def _weighted_average_fuse(
        self,
        votes: dict[str, str],
        weights: dict[str, float],
        availability: dict[str, bool],
    ) -> tuple[str, float]:
        """加权平均融合"""
        direction_scores: dict[str, float] = {}

        for module_name, direction in votes.items():
            w = weights.get(module_name, 0)
            direction_scores[direction] = direction_scores.get(direction, 0) + w

        if not direction_scores:
            return "", 0.0

        best_dir = max(direction_scores.items(), key=lambda x: x[1])[0]
        best_score = direction_scores[best_dir]

        # 置信度 = 最高分 × 一致性加成
        confidence = best_score
        if len(direction_scores) == 1:
            confidence = min(confidence * 1.2, 1.0)  # 一致性加成

        return best_dir, confidence

    def _confidence_weighted_fuse(
        self,
        votes: dict[str, str],
        mab: MABOutput | None,
        classifier: ClassifierOutput | None,
        aligner: AlignerOutput | None,
        availability: dict[str, bool],
    ) -> tuple[str, float]:
        """基于各模块自身置信度的加权融合"""
        confidences = {}

        if availability.get("mab", False) and mab is not None:
            confidences["mab"] = mab.confidence
        if availability.get("classifier", False) and classifier is not None:
            confidences["classifier"] = classifier.confidence
        if availability.get("aligner", False) and aligner is not None:
            confidences["aligner"] = aligner.alignment_strength

        if not confidences:
            return self._weighted_average_fuse(votes, {"mab": 0.34, "classifier": 0.33, "aligner": 0.33}, availability)

        total_conf = sum(confidences.values())
        weights = {k: v / total_conf for k, v in confidences.items()}

        return self._weighted_average_fuse(votes, weights, availability)

    def _majority_vote_fuse(
        self,
        votes: dict[str, str],
        availability: dict[str, bool],
    ) -> tuple[str, float]:
        """多数投票融合"""
        from collections import Counter

        counter = Counter(votes.values())

        if not counter:
            return "", 0.0

        best_dir, count = counter.most_common(1)[0]
        total = len(votes)

        confidence = count / total
        if count == total:
            confidence = 0.95  # 全一致给高置信度

        return best_dir, confidence

    def _winner_takes_all_fuse(
        self,
        votes: dict[str, str],
        mab: MABOutput | None,
        classifier: ClassifierOutput | None,
        aligner: AlignerOutput | None,
        availability: dict[str, bool],
    ) -> tuple[str, float]:
        """最高置信度者胜出"""
        candidates = []

        if availability.get("mab", False) and mab is not None and mab.is_valid:
            candidates.append(("mab", mab.selected_family, mab.confidence))
        if availability.get("classifier", False) and classifier is not None and classifier.is_valid:
            candidates.append(("classifier", classifier.direction, classifier.confidence))
        if availability.get("aligner", False) and aligner is not None and aligner.is_valid:
            candidates.append(("aligner", aligner.suggested_adjustment or "continue", aligner.alignment_strength))

        if not candidates:
            return "", 0.0

        winner = max(candidates, key=lambda x: x[2])
        return winner[1], winner[2]

    def record_success(self, module_name: str, success: bool) -> None:
        """记录某模块的成功/失败 (用于自适应权重调整)"""
        if module_name not in self._success_history:
            return

        history = self._success_history[module_name]
        history.append(success)

        window = self.config.get("history_window", 20)
        if len(history) > window:
            self._success_history[module_name] = history[-window:]

    def _get_success_rate(self, module_name: str) -> float | None:
        """获取某模块的历史成功率"""
        history = self._success_history.get(module_name, [])
        if len(history) < 5:
            return None  # 数据不足，不调整

        return sum(history) / len(history)
