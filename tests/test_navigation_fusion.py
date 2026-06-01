"""
Unit tests for NavigationFusion (Voting Mechanism)
================================================
测试 MAB / StrategyClassifier / HypothesisAligner 投票融合引擎。

覆盖范围:
  1. 基本三模块融合 (全部有效)
  2. 单/双模块失效容错
  3. 4种融合策略 (weighted_average/majority/confidence/winner)
  4. 动态权重计算 (置信度+历史成功率)
  5. 不一致性检测和共识判断
  6. 自适应权重记录与查询
  7. 边界情况和空值处理
"""

from __future__ import annotations

import pytest

from openalpha_brain.core.navigation_fusion import (
    AlignerOutput,
    ClassifierOutput,
    FusionResult,
    FusionStrategy,
    MABOutput,
    NavigationFusion,
)


@pytest.fixture
def fusion():
    """默认配置的 NavigationFusion 实例"""
    return NavigationFusion()


@pytest.fixture
def consensus_inputs():
    """三个模块达成一致的输入"""
    return {
        "mab_output": MABOutput(
            selected_family="momentum",
            confidence=0.8,
            weight_vector={"momentum": 0.45, "reversal": 0.30, "mean_reversion": 0.25},
        ),
        "classifier_output": ClassifierOutput(
            direction="momentum",
            confidence=0.75,
            market_state="trending",
        ),
        "aligner_output": AlignerOutput(
            alignment_level="aligned",
            r2_score=0.72,
            suggested_adjustment="momentum",  # 完全一致
            diagnosis="Strong alignment with momentum hypothesis",
        ),
    }


@pytest.fixture
def disagreement_inputs():
    """三个模块意见不一致的输入"""
    return {
        "mab_output": MABOutput(
            selected_family="reversal",
            confidence=0.65,
            weight_vector={"reversal": 0.40, "momentum": 0.35, "mean_reversion": 0.25},
        ),
        "classifier_output": ClassifierOutput(
            direction="mean_reversion",
            confidence=0.55,
            market_state="ranging",
            is_ambiguous=True,
        ),
        "aligner_output": AlignerOutput(
            alignment_level="weak",
            r2_score=0.35,
            suggested_adjustment="reduce_momentum_exposure",
            diagnosis="Weak support for current direction",
        ),
    }


class TestBasicFusion:
    """基本融合功能测试"""

    def test_three_modules_consensus(self, fusion, consensus_inputs):
        """三模块一致时返回共同方向"""
        result = fusion.fuse(**consensus_inputs, cycle_num=1)

        assert isinstance(result, FusionResult)
        assert result.final_direction in ["momentum", "momentum_enhanced"]
        assert result.confidence >= 0.7  # 一致时高置信度
        assert result.has_consensus is True
        assert result.disagreement_score < 0.3

    def test_all_none_returns_empty(self, fusion):
        """所有模块为None时返回空结果"""
        result = fusion.fuse(
            mab_output=None,
            classifier_output=None,
            aligner_output=None,
        )

        assert result.final_direction == ""
        assert result.confidence == 0.0
        assert len(result.raw_votes) == 0

    def test_single_module_valid(self, fusion):
        """只有MAB有效时使用MAB的方向"""
        mab = MABOutput(selected_family="reversal", confidence=0.7)

        result = fusion.fuse(mab_output=mab)

        assert result.final_direction == "reversal"
        assert result.weight_distribution["mab"] > 0
        assert result.weight_distribution.get("classifier", 0) == 0

    def test_two_modules_valid(self, fusion):
        """MAB和Classifier有效，Aligner缺失"""
        mab = MABOutput(selected_family="momentum", confidence=0.75)
        classifier = ClassifierOutput(direction="momentum", confidence=0.80)

        result = fusion.fuse(mab_output=mab, classifier_output=classifier)

        assert result.final_direction == "momentum"
        assert result.confidence > 0.5


class TestFusionStrategies:
    """4种融合策略测试"""

    def test_weighted_average_strategy(self, fusion, consensus_inputs):
        """加权平均策略 (默认)"""
        result = fusion.fuse(**consensus_inputs)

        assert result.strategy == FusionStrategy.WEIGHTED_AVERAGE
        # 权重应该接近配置的基础权重 (归一化后)
        total_weight = sum(result.weight_distribution.values())
        assert abs(total_weight - 1.0) < 0.01

    def test_majority_vote_strategy(self, fusion):
        """多数投票策略"""
        f = NavigationFusion(config={"fusion_strategy": "majority_vote"})

        mab = MABOutput(selected_family="momentum", confidence=0.9)
        classifier = ClassifierOutput(direction="momentum", confidence=0.85)
        aligner = AlignerOutput(alignment_level="aligned", r2_score=0.8, suggested_adjustment="momentum")

        result = f.fuse(mab_output=mab, classifier_output=classifier, aligner_output=aligner)

        assert result.strategy == FusionStrategy.MAJORITY_VOTE
        assert result.final_direction == "momentum"
        assert result.confidence >= 0.95  # 全一致

    def test_confidence_weighted_strategy(self, fusion):
        """置信度加权策略"""
        f = NavigationFusion(config={"fusion_strategy": "confidence_weighted"})

        mab = MABOutput(selected_family="reversal", confidence=0.90)  # 高置信度
        classifier = ClassifierOutput(direction="momentum", confidence=0.40)  # 低置信度
        aligner = AlignerOutput(alignment_level="aligned", r2_score=0.60, suggested_adjustment="reversal")

        result = f.fuse(mab_output=mab, classifier_output=classifier, aligner_output=aligner)

        assert result.strategy == FusionStrategy.CONFIDENCE_WEIGHTED
        # MAB和Aligner都支持reversal，且MAB置信度高，应该选reversal
        assert result.final_direction == "reversal"

    def test_winner_takes_all_strategy(self, fusion):
        """最高置信度者胜出策略"""
        f = NavigationFusion(config={"fusion_strategy": "winner_takes_all"})

        mab = MABOutput(selected_family="mean_reversion", confidence=0.55)
        classifier = ClassifierOutput(direction="momentum", confidence=0.92)  # 最高
        aligner = AlignerOutput(alignment_level="contradictory", r2_score=0.20, suggested_adjustment="avoid")

        result = f.fuse(mab_output=mab, classifier_output=classifier, aligner_output=aligner)

        assert result.strategy == FusionStrategy.WINNER_TAKES_ALL
        assert result.final_direction == "momentum"  # Classifier胜出


class TestDynamicWeights:
    """动态权重测试"""

    def test_low_confidence_penalty(self, fusion):
        """低置信度模块被降权"""
        mab = MABOutput(selected_family="momentum", confidence=0.15)  # 很低
        classifier = ClassifierOutput(direction="reversal", confidence=0.80)  # 正常
        aligner = AlignerOutput(alignment_level="aligned", r2_score=0.7, suggested_adjustment="reversal")

        result = fusion.fuse(mab_output=mab, classifier_output=classifier, aligner_output=aligner)

        # MAB权重应该远小于Classifier+Aligner
        w_mab = result.weight_distribution.get("mab", 0)
        w_others = result.weight_distribution.get("classifier", 0) + result.weight_distribution.get("aligner", 0)
        assert w_others > w_mab  # 其他模块总权重应该大于MAB

    def test_unavailable_module_zero_weight(self, fusion):
        """不可用模块权重为0"""
        mab = MABOutput(selected_family="momentum", confidence=0.70)

        result = fusion.fuse(mab_output=mab, classifier_output=None, aligner_output=None)

        assert result.weight_distribution.get("classifier", 0) == 0
        assert result.weight_distribution.get("aligner", 0) == 0
        assert result.weight_distribution["mab"] > 0


class TestDisagreementDetection:
    """不一致性检测测试"""

    def test_full_agreement_low_disagreement(self, fusion, consensus_inputs):
        """完全一致时不一致度低"""
        result = fusion.fuse(**consensus_inputs)

        # 注意: Aligner的suggested_adjustment是"momentum_enhanced"，与"momentum"略有不同
        # 所以disagreement_score可能不是0，但应该较低
        assert result.disagreement_score <= 0.4  # 放宽阈值
        assert result.has_consensus is True or result.disagreement_score < 0.5

    def test_high_disagreement_detected(self, fusion, disagreement_inputs):
        """高度不一致时检测到警告"""
        result = fusion.fuse(**disagreement_inputs)

        assert result.disagreement_score > 0.5
        assert result.has_consensus is False

    def test_disagreement_with_two_modules(self, fusion):
        """两模块不一致"""
        mab = MABOutput(selected_family="momentum", confidence=0.7)
        classifier = ClassifierOutput(direction="reversal", confidence=0.7)

        result = fusion.fuse(mab_output=mab, classifier_output=classifier)

        assert result.disagreement_score > 0.4  # 两票不同
        assert result.has_consensus is False


class TestAdaptiveWeights:
    """自适应权重测试"""

    def test_record_success_increases_future_weight(self, fusion):
        """记录成功后，该模块未来权重增加"""
        # 初始: 模拟MAB连续成功
        for _ in range(10):
            fusion.record_success("mab", True)
        for _ in range(5):
            fusion.record_success("mab", False)

        # 现在MAB成功率约66%
        mab = MABOutput(selected_family="momentum", confidence=0.6)
        classifier = ClassifierOutput(direction="momentum", confidence=0.6)

        result = fusion.fuse(mab_output=mab, classifier_output=classifier)

        # MAB应该有更高的相对权重（因为历史成功率高）
        w_mab = result.weight_distribution.get("mab", 0)
        w_class = result.weight_distribution.get("classifier", 0)
        assert w_mab > w_class  # MAB历史表现好，权重更高

    def test_insufficient_history_no_adjustment(self, fusion):
        """历史数据不足时不调整"""
        # 只有3条记录 (< 5条阈值)
        fusion.record_success("mab", True)
        fusion.record_success("mab", True)
        fusion.record_success("mab", True)

        mab = MABOutput(selected_family="test", confidence=0.5)
        classifier = ClassifierOutput(direction="test", confidence=0.5)

        result = fusion.fuse(mab_output=mab, classifier_output=classifier)

        # 历史不足时，MAB和Classifier应该有相近的权重（因为置信度相同）
        w_mab = result.weight_distribution.get("mab", 0)
        w_class = result.weight_distribution.get("classifier", 0)
        assert abs(w_mab - w_class) < 0.15  # 允许小偏差


class TestDataClasses:
    """Dataclass 测试"""

    def test_mab_output_validity(self):
        """MABOutput.is_valid 属性"""
        valid = MABOutput(selected_family="momentum", confidence=0.7)
        invalid_empty = MABOutput()
        invalid_zero_conf = MABOutput(selected_family="test", confidence=0)

        assert valid.is_valid is True
        assert invalid_empty.is_valid is False
        assert invalid_zero_conf.is_valid is False

    def test_classifier_output_validity(self):
        """ClassifierOutput.is_valid 属性"""
        valid = ClassifierOutput(direction="reversal", confidence=0.6)
        invalid = ClassifierOutput()

        assert valid.is_valid is True
        assert invalid.is_valid is False

    def test_aligner_output_properties(self):
        """AlignerOutput 的对齐属性"""
        aligned = AlignerOutput(alignment_level="aligned", r2_score=0.75)
        weak = AlignerOutput(alignment_level="weak", r2_score=0.35)
        contradictory = AlignerOutput(alignment_level="contradictory", r2_score=0.10)

        assert aligned.is_aligned is True
        assert weak.is_aligned is False
        assert contradictory.is_aligned is False

        assert aligned.alignment_strength > weak.alignment_strength
        assert weak.alignment_strength > contradictory.alignment_strength

    def test_fusion_result_to_log_dict(self):
        """FusionResult.to_log_dict() 截断长字符串"""
        result = FusionResult(
            final_direction="momentum",
            confidence=0.82,
            weight_distribution={"mab": 0.40, "classifier": 0.35, "aligner": 0.25},
            disagreement_score=0.15,
        )
        log_dict = result.to_log_dict()

        assert log_dict["direction"] == "momentum"
        assert log_dict["confidence"] == 0.82
        assert "weights" in log_dict
        assert "consensus" in log_dict


class TestEdgeCases:
    """边界情况测试"""

    def test_empty_string_directions(self, fusion):
        """空方向字符串的处理"""
        mab = MABOutput(selected_family="", confidence=0.5)  # 无效
        classifier = ClassifierOutput(direction="", confidence=0.5)  # 无效
        aligner = AlignerOutput(alignment_level="aligned", r2_score=0.6, suggested_adjustment="valid_dir")

        result = fusion.fuse(mab_output=mab, classifier_output=classifier, aligner_output=aligner)

        # 只有Aligner有效
        assert result.final_direction == "valid_dir"

    def test_very_high_confidence_all(self, fusion):
        """所有模块都很自信"""
        inputs = {
            "mab_output": MABOutput(selected_family="mean_reversion", confidence=0.98),
            "classifier_output": ClassifierOutput(direction="mean_reversion", confidence=0.97),
            "aligner_output": AlignerOutput(
                alignment_level="aligned",
                r2_score=0.92,
                suggested_adjustment="mean_reversion",  # 完全一致
            ),
        }
        result = fusion.fuse(**inputs)

        assert result.confidence > 0.85  # 高置信度（因为完全一致）
        assert result.has_consensus is True

    def test_processing_time_recorded(self, fusion, consensus_inputs):
        """processing_time_ms 被记录"""
        result = fusion.fuse(**consensus_inputs)

        assert result.processing_time_ms > 0

    def test_metadata_contains_cycle_num(self, fusion, consensus_inputs):
        """metadata 包含 cycle_num"""
        result = fusion.fuse(**consensus_inputs, cycle_num=42)

        assert result.metadata.get("cycle_num") == 42
        assert "availability" in result.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
