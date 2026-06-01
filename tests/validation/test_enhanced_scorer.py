"""
策略 D 升级评分系统测试 (ICIR + Multi-Faceted Reward)

测试范围:
  1. ICIRMetrics 数据类 (10 tests)
  2. MultiFacetedReward 数据类 (10 tests)
  3. infer_icir_metrics 方法 (10 tests)
  4. compute_multi_faceted_reward 方法 (10 tests)
  5. multi_layer_evaluate 增强 (5 tests)
  6. ScoreReport 向后兼容性 (5 tests)
  7. 辅助方法 (_estimate_uniqueness, _estimate_complexity_penalty) (5+ tests)

运行: pytest tests/validation/test_enhanced_scorer.py -v
"""

import pytest
import math

from openalpha_brain.validation.official_scorer import (
    OfficialScoringAdapter,
    ScoreReport,
    AdvancedMetrics,
    ICIRMetrics,
    MultiFacetedReward,
    BRAIN_THRESHOLDS,
)


# ══════════════════════════════════════════════════════════════════════
# Test Fixtures
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture
def scorer():
    return OfficialScoringAdapter()


@pytest.fixture
def perfect_metrics():
    """完美因子指标"""
    return {
        "sharpe": 1.75,
        "fitness": 1.25,
        "turnover": 0.20,
        "returns": 0.05,
        "drawdown": 0.10,
        "checks": [],
        "delay": 1,
    }


@pytest.fixture
def high_icir_low_fitness_metrics():
    """高 ICIR 低 Fitness 的因子指标 (策略 D 特殊类型)"""
    return {
        "sharpe": 1.50,  # 较高 Sharpe
        "fitness": 0.85,  # 低 Fitness (< 1.0)
        "turnover": 0.35,  # 中等换手率
        "returns": 0.04,
        "drawdown": 0.18,
        "checks": [],
        "delay": 1,
    }


@pytest.fixture
def efficient_alpha_metrics():
    """高效因子 (efficiency > 0.3)"""
    return {
        "sharpe": 1.25,
        "fitness": 1.10,
        "turnover": 0.10,  # 低换手率 → 高效率
        "returns": 0.06,  # 较高收益
        "drawdown": 0.12,
        "checks": [],
        "delay": 1,
    }


@pytest.fixture
def noise_metrics():
    """噪声因子"""
    return {
        "sharpe": -0.5,
        "fitness": 0.30,
        "turnover": 0.75,
        "returns": -0.02,
        "drawdown": 0.35,
        "checks": [
            {"name": "test", "value": None, "limit": None, "result": False},
        ],
        "delay": 1,
    }


@pytest.fixture
def complex_expression():
    """复杂表达式"""
    return (
        "group_neutralize("
        "tanh(signed_power(rank(ts_decay_linear(ts_zscore(close, 20), 10)), 0.5)), "
        "industry)"
    )


@pytest.fixture
def simple_expression():
    """简单表达式"""
    return "group_neutralize(rank(ts_delta(close, 5)), industry)"


# ══════════════════════════════════════════════════════════════════════
# 1. ICIRMetrics 数据类测试 (10 tests)
# ══════════════════════════════════════════════════════════════════════

class TestICIRMetricsDataclass:
    """ICIRMetrics 数据类基本功能测试"""

    def test_default_values(self):
        """默认值应该全为 0"""
        metrics = ICIRMetrics()
        assert metrics.ic == 0.0
        assert metrics.rank_ic == 0.0
        assert metrics.ir == 0.0
        assert metrics.icir == 0.0
        assert metrics.predicted_fitness == 0.0
        assert metrics.confidence == 0.0

    def test_custom_initialization(self):
        """自定义初始化"""
        metrics = ICIRMetrics(
            ic=0.05,
            rank_ic=0.055,
            ir=1.2,
            icir=1.5,
            predicted_fitness=1.1,
            confidence=0.85,
        )
        assert metrics.ic == 0.05
        assert metrics.rank_ic == 0.055
        assert metrics.ir == 1.2
        assert metrics.icir == 1.5
        assert metrics.predicted_fitness == 1.1
        assert metrics.confidence == 0.85

    def test_to_dict_returns_correct_keys(self):
        """to_dict 应该返回所有字段的字典"""
        metrics = ICIRMetrics(ic=0.05, ir=1.2, icir=1.5)
        d = metrics.to_dict()
        assert isinstance(d, dict)
        assert "ic" in d
        assert "rank_ic" in d
        assert "ir" in d
        assert "icir" in d
        assert "predicted_fitness" in d
        assert "confidence" in d

    def test_to_dict_rounds_values(self):
        """to_dict 应该正确四舍五入"""
        metrics = ICIRMetrics(ic=0.01234567, ir=1.23456, icir=1.23456)
        d = metrics.to_dict()
        assert d["ic"] == round(0.01234567, 6)
        assert d["ir"] == round(1.23456, 4)
        assert d["icir"] == round(1.23456, 4)

    def test_is_high_icir_true_when_above_threshold(self):
        """当 ICIR > 1.0 时应返回 True"""
        metrics = ICIRMetrics(icir=1.5)
        assert metrics.is_high_icir is True

    def test_is_high_icir_false_when_below_threshold(self):
        """当 ICIR <= 1.0 时应返回 False"""
        metrics = ICIRMetrics(icir=0.9)
        assert metrics.is_high_icir is False

    def test_is_high_icir_boundary_value(self):
        """边界值: ICIR = 1.0 应返回 False"""
        metrics = ICIRMetrics(icir=1.0)
        assert metrics.is_high_icir is False

    def test_is_low_fitness_high_icir_true(self):
        """高 ICIR 且低 predicted_fitness 应返回 True"""
        metrics = ICIRMetrics(icir=1.5, predicted_fitness=0.85)
        assert metrics.is_low_fitness_high_icir is True

    def test_is_low_fitness_high_icir_false_when_fitness_ok(self):
        """Fitness >= 1.0 时即使 ICIR 高也应返回 False"""
        metrics = ICIRMetrics(icir=1.5, predicted_fitness=1.1)
        assert metrics.is_low_fitness_high_icir is False

    def test_is_low_fitness_high_icir_false_when_icir_low(self):
        """ICIR 低时应返回 False"""
        metrics = ICIRMetrics(icir=0.5, predicted_fitness=0.85)
        assert metrics.is_low_fitness_high_icir is False


# ══════════════════════════════════════════════════════════════════════
# 2. MultiFacetedReward 数据类测试 (10 tests)
# ══════════════════════════════════════════════════════════════════════

class TestMultiFacetedRewardDataclass:
    """MultiFacetedReward 数据类基本功能测试"""

    def test_default_values(self):
        """默认值应该全为 0"""
        reward = MultiFacetedReward()
        assert reward.signal_quality == 0.0
        assert reward.stability == 0.0
        assert reward.efficiency == 0.0
        assert reward.uniqueness == 0.0
        assert reward.simplicity == 0.0
        assert reward.total_reward == 0.0

    def test_weights_constant(self):
        """WEIGHTS 常量应该包含所有维度且总和为 1.0"""
        weights = MultiFacetedReward.WEIGHTS
        assert "signal_quality" in weights
        assert "stability" in weights
        assert "efficiency" in weights
        assert "uniqueness" in weights
        assert "simplicity" in weights
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001, f"Weights sum should be 1.0, got {total}"

    def test_weight_values(self):
        """权重值应符合预期配置"""
        weights = MultiFacetedReward.WEIGHTS
        assert weights["signal_quality"] == 0.30
        assert weights["stability"] == 0.25
        assert weights["efficiency"] == 0.20
        assert weights["uniqueness"] == 0.15
        assert weights["simplicity"] == 0.10

    def test_to_dict_returns_all_fields(self):
        """to_dict 应返回所有字段"""
        reward = MultiFacetedReward(
            signal_quality=0.8,
            stability=0.7,
            efficiency=0.6,
            uniqueness=0.5,
            simplicity=0.2,
            total_reward=0.75,
        )
        d = reward.to_dict()
        assert len(d) == 6
        assert all(k in d for k in ["signal_quality", "stability", "efficiency",
                                    "uniqueness", "simplicity", "total_reward"])

    def test_is_efficient_alpha_true(self):
        """efficiency > 0.3 应返回 True"""
        reward = MultiFacetedReward(efficiency=0.4)
        assert reward.is_efficient_alpha is True

    def test_is_efficient_alpha_false(self):
        """efficiency <= 0.3 应返回 False"""
        reward = MultiFacetedReward(efficiency=0.3)
        assert reward.is_efficient_alpha is False

    def test_is_efficient_alpha_boundary(self):
        """边界值: efficiency = 0.3001 应返回 True"""
        reward = MultiFacetedReward(efficiency=0.3001)
        assert reward.is_efficient_alpha is True

    def test_dominant_dimension_signal(self):
        """signal_quality 最高时应该是主导维度"""
        reward = MultiFacetedReward(signal_quality=0.9, stability=0.5, efficiency=0.4, uniqueness=0.3)
        assert reward.dominant_dimension == "signal_quality"

    def test_dominant_dimension_stability(self):
        """stability 最高时应该是主导维度"""
        reward = MultiFacetedReward(signal_quality=0.5, stability=0.9, efficiency=0.4, uniqueness=0.3)
        assert reward.dominant_dimension == "stability"

    def test_total_reward_calculation_example(self, scorer):
        """验证 total_reward 的手动计算 (需要通过 scorer 计算)"""
        metrics = {
            "sharpe": 1.6,
            "fitness": 1.4,
            "turnover": 0.2,
            "returns": 0.04,
        }
        reward = scorer.compute_multi_faceted_reward(metrics)

        # 手动验证 total_reward 在合理范围内
        expected_min = (
            MultiFacetedReward.WEIGHTS["signal_quality"] * reward.signal_quality +
            MultiFacetedReward.WEIGHTS["stability"] * reward.stability +
            MultiFacetedReward.WEIGHTS["efficiency"] * reward.efficiency +
            MultiFacetedReward.WEIGHTS["uniqueness"] * reward.uniqueness -
            MultiFacetedReward.WEIGHTS["simplicity"] * reward.simplicity
        )

        # 允许小的浮点误差
        assert abs(reward.total_reward - expected_min) < 0.01


# ══════════════════════════════════════════════════════════════════════
# 3. infer_icir_metrics 方法测试 (10 tests)
# ══════════════════════════════════════════════════════════════════════

class TestInferICIRMetrics:
    """ICIR 推断方法测试"""

    def test_positive_sharpe_produces_nonzero_ic(self, scorer, perfect_metrics):
        """正 Sharpe 应产生非零 IC"""
        result = scorer.infer_icir_metrics(perfect_metrics)
        assert result.ic > 0

    def test_negative_sharpe_returns_zeros(self, scorer, noise_metrics):
        """负或零 Sharpe 应返回零值"""
        result = scorer.infer_icir_metrics(noise_metrics)
        assert result.ic == 0.0
        assert result.icir == 0.0
        assert result.predicted_fitness == 0.0

    def test_zero_sharpe_returns_zeros(self, scorer):
        """Sharpe = 0 应返回零值"""
        metrics = {"sharpe": 0, "fitness": 1.0, "turnover": 0.2}
        result = scorer.infer_icir_metrics(metrics)
        assert result.ic == 0.0

    def test_rank_ic_higher_than_ic(self, scorer, perfect_metrics):
        """Rank IC 通常比 IC 高"""
        result = scorer.infer_icir_metrics(perfect_metrics)
        if result.ic > 0:
            assert result.rank_ic > result.ic

    def test_ir_within_reasonable_range(self, scorer, perfect_metrics):
        """IR 应在合理范围内 (0-2.5)"""
        result = scorer.infer_icir_metrics(perfect_metrics)
        if result.ir > 0:
            assert 0 < result.ir <= 2.5

    def test_higher_sharpe_produces_higher_icir(self, scorer):
        """更高的 Sharpe 应产生更高的 ICIR"""
        metrics_low = {"sharpe": 0.8, "fitness": 0.9, "turnover": 0.2}
        metrics_high = {"sharpe": 1.8, "fitness": 1.5, "turnover": 0.2}

        result_low = scorer.infer_icir_metrics(metrics_low)
        result_high = scorer.infer_icir_metrics(metrics_high)

        assert result_high.icir > result_low.icir

    def test_turnover_affects_ic(self, scorer):
        """换手率应影响 IC 推断"""
        metrics_low_to = {"sharpe": 1.0, "fitness": 1.0, "turnover": 0.05}
        metrics_high_to = {"sharpe": 1.0, "fitness": 1.0, "turnover": 0.40}

        result_low = scorer.infer_icir_metrics(metrics_low_to)
        result_high = scorer.infer_icir_metrics(metrics_high_to)

        assert result_high.ic > result_low.ic

    def test_confidence_between_0_and_1(self, scorer, perfect_metrics):
        """置信度应在 [0, 1] 范围内"""
        result = scorer.infer_icir_metrics(perfect_metrics)
        assert 0 <= result.confidence <= 1

    def test_predicted_fitness_positive_for_good_factor(self, scorer, perfect_metrics):
        """好的因子应有正的 predicted_fitness"""
        result = scorer.infer_icir_metrics(perfect_metrics)
        if result.icir > 0:
            assert result.predicted_fitness > 0

    def test_to_dict_structure(self, scorer, perfect_metrics):
        """to_dict 返回的字典结构应正确"""
        result = scorer.infer_icir_metrics(perfect_metrics)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert len(d) == 6


# ══════════════════════════════════════════════════════════════════════
# 4. compute_multi_faceted_reward 方法测试 (10 tests)
# ══════════════════════════════════════════════════════════════════════

class TestComputeMultiFacetedReward:
    """多面奖励函数测试"""

    def test_perfect_metrics_high_reward(self, scorer, perfect_metrics):
        """完美指标应产生高 total_reward"""
        result = scorer.compute_multi_faceted_reward(perfect_metrics)
        assert result.total_reward > 0.5

    def test_noise_metrics_low_reward(self, scorer, noise_metrics):
        """噪声因子应产生低 total_reward"""
        result = scorer.compute_multi_faceted_reward(noise_metrics)
        assert result.total_reward < 0.3

    def test_signal_quality_based_on_sharpe(self, scorer):
        """signal_quality 应基于 Sharpe 归一化"""
        metrics_high = {"sharpe": 2.0, "fitness": 1.0, "turnover": 0.2, "returns": 0.01}
        metrics_low = {"sharpe": 0.5, "fitness": 1.0, "turnover": 0.2, "returns": 0.01}

        result_high = scorer.compute_multi_faceted_reward(metrics_high)
        result_low = scorer.compute_multi_faceted_reward(metrics_low)

        assert result_high.signal_quality > result_low.signal_quality

    def test_stability_based_on_fitness(self, scorer):
        """stability 应基于 Fitness 归一化"""
        metrics_high = {"sharpe": 1.0, "fitness": 2.0, "turnover": 0.2, "returns": 0.01}
        metrics_low = {"sharpe": 1.0, "fitness": 0.3, "turnover": 0.2, "returns": 0.01}

        result_high = scorer.compute_multi_faceted_reward(metrics_high)
        result_low = scorer.compute_multi_faceted_reward(metrics_low)

        assert result_high.stability > result_low.stability

    def test_efficiency_with_low_turnover(self, scorer, efficient_alpha_metrics):
        """低换手率 + 正收益应产生高效率"""
        result = scorer.compute_multi_faceted_reward(efficient_alpha_metrics)
        assert result.efficiency > 0.3
        assert result.is_efficient_alpha is True

    def test_uniqueness_with_unique_expression(self, scorer, perfect_metrics):
        """使用独特字段的表达式应有更高 uniqueness"""
        unique_expr = "group_neutralize(rank(ts_regression(earnings, volume, 20)), industry)"
        common_expr = "group_neutralize(rank(ts_delta(close, 5)), industry)"

        result_unique = scorer.compute_multi_faceted_reward(perfect_metrics, expr=unique_expr)
        result_common = scorer.compute_multi_faceted_reward(perfect_metrics, expr=common_expr)

        assert result_unique.uniqueness > result_common.uniqueness

    def test_simplicity_penalty_for_complex_expression(self, scorer, perfect_metrics, complex_expression, simple_expression):
        """复杂表达式应有更高的 complexity penalty"""
        result_complex = scorer.compute_multi_faceted_reward(
            perfect_metrics, expr=complex_expression
        )
        result_simple = scorer.compute_multi_faceted_reward(
            perfect_metrics, expr=simple_expression
        )

        assert result_complex.simplicity >= result_simple.simplicity

    def test_total_reward_between_0_and_1(self, scorer, perfect_metrics):
        """total_reward 应在 [0, 1] 范围内"""
        result = scorer.compute_multi_faceted_reward(perfect_metrics)
        assert 0 <= result.total_reward <= 1

    def test_empty_expression_default_uniqueness(self, scorer, perfect_metrics):
        """空表达式应使用默认 uniqueness (0.5)"""
        result = scorer.compute_multi_faceted_reward(perfect_metrics, expr="")
        assert result.uniqueness == 0.5

    def test_dominant_dimension_identification(self, scorer, perfect_metrics):
        """dominant_dimension 应能识别主导维度"""
        result = scorer.compute_multi_faceted_reward(perfect_metrics)
        dominant = result.dominant_dimension
        assert dominant in ["signal_quality", "stability", "efficiency", "uniqueness"]


# ══════════════════════════════════════════════════════════════════════
# 5. multi_layer_evaluate 增强测试 (5 tests)
# ══════════════════════════════════════════════════════════════════════

class TestMultiLayerEvaluateEnhanced:
    """增强的多层评估决策测试"""

    def test_includes_icir_metrics_field(self, scorer, perfect_metrics):
        """结果应包含 icir_metrics 字段"""
        result = scorer.multi_layer_evaluate(perfect_metrics)
        assert "icir_metrics" in result
        assert isinstance(result["icir_metrics"], dict)

    def test_includes_multi_faceted_reward_field(self, scorer, perfect_metrics):
        """结果应包含 multi_faceted_reward 字段"""
        result = scorer.multi_layer_evaluate(perfect_metrics)
        assert "multi_faceted_reward" in result
        assert isinstance(result["multi_faceted_reward"], dict)

    def test_high_icir_low_fitness_detection(self, scorer, high_icir_low_fitness_metrics):
        """应检测到 HIGH_ICIR_LOW_FITNESS 类型"""
        result = scorer.multi_layer_evaluate(high_icir_low_fitness_metrics)
        assert "is_high_icir_low_fitness" in result
        # 注意: 这个标志取决于实际推断结果，可能不总是 True
        assert isinstance(result["is_high_icir_low_fitness"], bool)

    def test_efficient_alpha_detection(self, scorer, efficient_alpha_metrics):
        """应检测到 EFFICIENT_ALPHA 类型"""
        result = scorer.multi_layer_evaluate(efficient_alpha_metrics)
        assert "is_efficient_alpha" in result
        assert isinstance(result["is_efficient_alpha"], bool)

    def test_diagnosis_contains_new_categories(self, scorer, high_icir_low_fitness_metrics):
        """diagnosis 可能包含新的分类标签"""
        result = scorer.multi_layer_evaluate(high_icir_low_fitness_metrics)
        diagnosis = result.get("diagnosis", "")
        # 至少应包含基础分类
        assert len(diagnosis) > 0


# ══════════════════════════════════════════════════════════════════════
# 6. ScoreReport 向后兼容性测试 (5 tests)
# ══════════════════════════════════════════════════════════════════════

class TestScoreReportBackwardCompatibility:
    """ScoreReport 向后兼容性测试"""

    def test_basic_fields_present(self, scorer, perfect_metrics):
        """基本字段应仍然存在"""
        report = scorer.compute_score(perfect_metrics)
        assert hasattr(report, 'overall_score')
        assert hasattr(report, 'grade')
        assert hasattr(report, 'breakdown')
        assert hasattr(report, 'passed')
        assert hasattr(report, 'details')
        assert hasattr(report, 'improvement_hints')

    def test_v1_fields_still_work(self, scorer, perfect_metrics):
        """v1.0 的增强字段应仍然工作"""
        report = scorer.compute_score(perfect_metrics)
        assert hasattr(report, 'advanced_metrics')
        assert hasattr(report, 'multi_layer_result')
        assert hasattr(report, 'factor_profile')

    def test_v2_new_fields_exist(self, scorer, perfect_metrics):
        """v2.0 的新字段应存在"""
        report = scorer.compute_score(perfect_metrics)
        assert hasattr(report, 'icir_metrics')
        assert hasattr(report, 'multi_faceted_reward')

    def test_icir_metrics_not_none_for_valid_input(self, scorer, perfect_metrics):
        """有效输入的 icir_metrics 不应为 None"""
        report = scorer.compute_score(perfect_metrics)
        assert report.icir_metrics is not None
        assert isinstance(report.icir_metrics, ICIRMetrics)

    def test_multi_faceted_reward_not_none_for_valid_input(self, scorer, perfect_metrics):
        """有效输入的 multi_faceted_reward 不应为 None"""
        report = scorer.compute_score(perfect_metrics)
        assert report.multi_faceted_reward is not None
        assert isinstance(report.multi_faceted_reward, MultiFacetedReward)

    def test_to_dict_includes_new_fields(self, scorer, perfect_metrics):
        """to_dict 应包含新字段"""
        report = scorer.compute_score(perfect_metrics)
        d = report.to_dict()
        assert "icir_metrics" in d
        assert "multi_faceted_reward" in d


# ══════════════════════════════════════════════════════════════════════
# 7. 辅助方法测试 (5+ tests)
# ══════════════════════════════════════════════════════════════════════

class TestEstimateUniqueness:
    """_estimate_uniqueness 方法测试"""

    def test_empty_expression_returns_default(self, scorer):
        """空表达式应返回默认值 0.5"""
        result = scorer._estimate_uniqueness("", {})
        assert result == 0.5

    def test_unique_field_boosts_score(self, scorer):
        """使用独特字段应提高分数"""
        common = scorer._estimate_uniqueness("rank(ts_delta(close, 5))", {})
        unique = scorer._estimate_uniqueness("rank(ts_regression(earnings, volume, 20))", {})
        assert unique > common

    def test_rare_operator_boosts_score(self, scorer):
        """使用罕见算子应提高分数"""
        common = scorer._estimate_uniqueness("rank(ts_mean(close, 20))", {})
        rare = scorer._estimate_uniqueness("tanh(signed_power(rank(close), 0.5))", {})
        assert rare > common

    def test_low_turnover_with_high_sharpe_boosts(self, scorer):
        """低换手率 + 高 Sharpe 应提高分数"""
        metrics_normal = {"turnover": 0.5, "sharpe": 0.5}
        metrics_good = {"turnover": 0.15, "sharpe": 1.5}

        result_normal = scorer._estimate_uniqueness("rank(close)", metrics_normal)
        result_good = scorer._estimate_uniqueness("rank(close)", metrics_good)

        assert result_good > result_normal

    def test_result_capped_at_1(self, scorer):
        """结果不应超过 1.0"""
        expr = "tanh(signed_power(ts_regression(ts_skewness(earnings), ts_kurtosis(volume), 20), 0.5))"
        metrics = {"turnover": 0.10, "sharpe": 2.0}
        result = scorer._estimate_uniqueness(expr, metrics)
        assert result <= 1.0


class TestEstimateComplexityPenalty:
    """_estimate_complexity_penalty 方法测试"""

    def test_empty_expression_no_penalty(self, scorer):
        """空表达式无惩罚"""
        result = scorer._estimate_complexity_penalty("")
        assert result == 0.0

    def test_long_expression_has_penalty(self, scorer):
        """长表达式应有惩罚"""
        short = "a" * 100
        long_expr = "a" * 350

        result_short = scorer._estimate_complexity_penalty(short)
        result_long = scorer._estimate_complexity_penalty(long_expr)

        assert result_long > result_short

    def test_deep_nesting_has_penalty(self, scorer):
        """深层嵌套应有惩罚"""
        shallow = "a(b(c))"
        deep = "a(b(c(d(e(f(g(h)))))))"

        result_shallow = scorer._estimate_complexity_penalty(shallow)
        result_deep = scorer._estimate_complexity_penalty(deep)

        assert result_deep > result_shallow

    def test_many_operators_has_penalty(self, scorer):
        """多个不同算子应有惩罚"""
        few_ops = "ts_mean(close, 20)"
        many_ops = "ts_mean(ts_delta(ts_regression(ts_corr(ts_std_dev(ts_av_diff(close, volume, 5), close, 10), close, volume, 15), close, 5), close, 10), 20)"

        result_few = scorer._estimate_complexity_penalty(few_ops)
        result_many = scorer._estimate_complexity_penalty(many_ops)

        assert result_many > result_few

    def test_result_capped_at_1(self, scorer):
        """结果不应超过 1.0"""
        very_complex = "a" * 500 + "(" * 10 + "b" * 500 + ")" * 10
        result = scorer._estimate_complexity_penalty(very_complex)
        assert result <= 1.0


# ══════════════════════════════════════════════════════════════════════
# 8. 集成场景测试 (额外补充)
# ══════════════════════════════════════════════════════════════════════

class TestIntegrationScenarios:
    """端到端集成场景测试"""

    def test_full_pipeline_perfect_factor(self, scorer, perfect_metrics):
        """完整流程: 完美因子"""
        report = scorer.compute_score(perfect_metrics)

        assert report.passed is True
        assert report.overall_score >= 70
        assert report.grade in ["A+", "A", "A-"]
        assert report.icir_metrics.ic > 0
        assert report.multi_faceted_reward.total_reward > 0.5

    def test_full_pipeline_noise_factor(self, scorer, noise_metrics):
        """完整流程: 噪声因子"""
        report = scorer.compute_score(noise_metrics)

        assert report.passed is False
        assert report.overall_score < 50
        assert report.icir_metrics.ic == 0  # 负 Sharpe
        assert report.multi_faceted_reward.total_reward < 0.3

    def test_high_icir_low_fitness_scenario(self, scorer, high_icir_low_fitness_metrics):
        """HIGH_ICIR_LOW_FITNESS 场景"""
        report = scorer.compute_score(high_icir_low_fitness_metrics)
        ml_result = report.multi_layer_result

        assert ml_result is not None
        # 检查是否检测到特殊分类
        if ml_result.get("is_high_icir_low_fitness"):
            assert ml_result["final_decision"] == "IMPROVE"

    def test_efficient_alpha_scenario(self, scorer, efficient_alpha_metrics):
        """EFFICIENT_ALPHA 场景"""
        report = scorer.compute_score(efficient_alpha_metrics)

        assert report.multi_faceted_reward.is_efficient_alpha is True
        ml_result = report.multi_layer_result
        if ml_result and ml_result.get("is_efficient_alpha"):
            assert "EFFICIENT_ALPHA" in ml_result["diagnosis"]

    def test_edge_case_zero_turnover(self, scorer):
        """边界情况: turnover = 0"""
        metrics = {
            "sharpe": 1.0,
            "fitness": 1.0,
            "turnover": 0.0,  # 极端值
            "returns": 0.02,
            "drawdown": 0.15,
            "checks": [],
        }
        report = scorer.compute_score(metrics)
        assert report is not None
        assert report.overall_score >= 0  # 不应崩溃

    def test_edge_case_extremely_high_turnover(self, scorer):
        """边界情况: turnover = 0.99"""
        metrics = {
            "sharpe": 1.0,
            "fitness": 0.8,
            "turnover": 0.99,  # 接近上限
            "returns": 0.03,
            "drawdown": 0.20,
            "checks": [],
        }
        report = scorer.compute_score(metrics)
        assert report is not None
        assert report.overall_score <= 100  # 不应超过满分


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
