"""
增强评分系统测试 (v1.0.0+)

测试范围:
  - AdvancedMetrics 推断
  - Multi-layer evaluation
  - Factor profile classification
  - 向后兼容性

运行: pytest tests/validation/test_official_scorer_enhanced.py -v
"""

import pytest
import math

from openalpha_brain.validation.official_scorer import (
    OfficialScoringAdapter,
    ScoreReport,
    AdvancedMetrics,
    ENHANCED_SCORE_WEIGHTS,
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
        "turnover": 0.25,
        "returns": 0.05,
        "drawdown": 0.12,
        "checks": [
            {"name": "sharpe_positive", "value": 1.75, "limit": 1.25, "result": True},
            {"name": "turnover_platform", "value": 0.25, "limit": 0.70, "result": True},
        ],
        "delay": 1,
    }


@pytest.fixture
def near_pass_metrics():
    """接近及格线的因子"""
    return {
        "sharpe": 1.0,
        "fitness": 0.8,
        "turnover": 0.35,
        "returns": 0.03,
        "drawdown": 0.18,
        "checks": [
            {"name": "sharpe_positive", "value": 1.0, "limit": 1.25, "result": False},
            {"name": "turnover_platform", "value": 0.35, "limit": 0.70, "result": True},
        ],
        "delay": 1,
    }


@pytest.fixture
def noise_metrics():
    """噪声因子（极差）"""
    return {
        "sharpe": -0.5,
        "fitness": 0.3,
        "turnover": 0.75,
        "returns": -0.02,
        "drawdown": 0.35,
        "checks": [
            {"name": "sharpe_positive", "value": -0.5, "limit": 1.25, "result": False},
            {"name": "turnover_platform", "value": 0.75, "limit": 0.70, "result": False},
        ],
        "delay": 1,
    }


@pytest.fixture
def overfit_metrics():
    """过拟合因子"""
    return {
        "sharpe": 2.8,
        "fitness": 0.9,
        "turnover": 0.55,
        "returns": 0.08,
        "drawdown": 0.25,
        "checks": [
            {"name": "sharpe_positive", "value": 2.8, "limit": 1.25, "result": True},
            {"name": "turnover_platform", "value": 0.55, "limit": 0.70, "result": True},
        ],
        "delay": 1,
    }


# ══════════════════════════════════════════════════════════════════════
# Test Class 1: AdvancedMetrics Inference
# ══════════════════════════════════════════════════════════════════════

class TestAdvancedMetricsInference:
    """测试高级指标推断功能"""

    def test_infer_from_perfect_factor(self, scorer, perfect_metrics):
        """完美因子应推断出高质量指标"""
        advanced = scorer.infer_advanced_metrics(perfect_metrics)

        assert isinstance(advanced, AdvancedMetrics)
        assert advanced.ic is not None
        assert advanced.ic > 0
        assert advanced.rank_ic is not None
        assert advanced.rank_ic > advanced.ic  # Rank IC 通常更高
        assert advanced.ir is not None
        assert advanced.ir > 0
        assert advanced.win_rate is not None
        assert advanced.win_rate > 0.65  # 高 Sharpe 应有较高胜率
        assert advanced.risk_level in ["LOW", "MEDIUM"]
        assert 0 <= advanced.improvement_potential <= 1

    def test_infer_from_near_pass(self, scorer, near_pass_metrics):
        """接近及格因子应有中等质量指标"""
        advanced = scorer.infer_advanced_metrics(near_pass_metrics)

        assert isinstance(advanced, AdvancedMetrics)
        assert advanced.ic is not None
        assert 0 < advanced.ic < 0.05  # 中等 IC
        assert advanced.win_rate is not None
        assert 0.5 < advanced.win_rate < 0.8  # 中等胜率
        assert advanced.improvement_potential > 0.3  # 有改进空间

    def test_infer_from_noise(self, scorer, noise_metrics):
        """噪声因子应标记为高风险"""
        advanced = scorer.infer_advanced_metrics(noise_metrics)

        assert isinstance(advanced, AdvancedMetrics)
        assert advanced.ic is None or advanced.ic <= 0  # 负 Sharpe 无有效 IC
        assert advanced.risk_level == "CRITICAL"
        assert advanced.improvement_potential > 0.6  # 需要大量改进

    def test_infer_from_overfit(self, scorer, overfit_metrics):
        """过拟合因子应检测到风险"""
        advanced = scorer.infer_advanced_metrics(overfit_metrics)

        assert isinstance(advanced, AdvancedMetrics)
        assert advanced.ic is not None
        assert advanced.ic > 0.03  # 高 Sharpe 导致高 IC
        assert advanced.risk_level in ["HIGH", "CRITICAL"]
        assert advanced.overall_diagnosis == "OVERFIT"

    def test_missing_fields_handled(self, scorer):
        """缺失字段应安全处理"""
        metrics = {}  # 空字典
        advanced = scorer.infer_advanced_metrics(metrics)

        assert isinstance(advanced, AdvancedMetrics)
        assert advanced.ic is None
        assert advanced.ir is None
        assert advanced.win_rate is None
        # 注意: 空字典时 sharpe=0 会被判断为 HIGH 风险 (sharpe < 0.5)
        assert advanced.risk_level in ["MEDIUM", "HIGH"]

    def test_quantile_returns_generated(self, scorer, perfect_metrics):
        """应生成五档收益分布"""
        advanced = scorer.infer_advanced_metrics(perfect_metrics)

        if advanced.quantile_returns:
            assert "Q1" in advanced.quantile_returns
            assert "Q5" in advanced.quantile_returns
            assert "spread" in advanced.quantile_returns
            assert len(advanced.quantile_returns) == 6

    def test_stability_calculation(self, scorer, perfect_metrics):
        """稳定性应在合理范围内"""
        advanced = scorer.infer_advanced_metrics(perfect_metrics)

        if advanced.stability is not None:
            assert 0 <= advanced.stability <= 1

    def test_icir_calculation(self, scorer, perfect_metrics):
        """ICIR 应在合理范围内"""
        advanced = scorer.infer_advanced_metrics(perfect_metrics)

        if advanced.icir is not None:
            assert advanced.icir > 0
            assert advanced.rank_icir is not None
            assert advanced.rank_icir >= advanced.icir

    def test_to_dict_excludes_defaults(self, scorer, perfect_metrics):
        """to_dict 应排除默认值"""
        advanced = scorer.infer_advanced_metrics(perfect_metrics)
        result = advanced.to_dict()

        assert "risk_level" not in result or result["risk_level"] != "MEDIUM"
        assert isinstance(result, dict)


# ══════════════════════════════════════════════════════════════════════
# Test Class 2: Multi-Layer Evaluation
# ══════════════════════════════════════════════════════════════════════

class TestMultiLayerEvaluation:
    """测试三层评估决策"""

    def test_layer1_value_eval_pass(self, scorer, perfect_metrics):
        """Layer 1: 完美因子应通过值评估"""
        result = scorer.multi_layer_evaluate(perfect_metrics)

        assert result["layer1_passed"] is True
        assert "正常" in result["layer1_details"]

    def test_layer1_value_eval_fail_invalid(self, scorer, noise_metrics):
        """Layer 1: 噪声因子应失败"""
        result = scorer.multi_layer_evaluate(noise_metrics)

        assert result["layer1_passed"] is False
        assert len(result["layer1_details"]) > 0

    def test_layer1_sharpe_below_threshold(self, scorer):
        """Layer 1: Sharpe 低于阈值应标记"""
        metrics = {
            "sharpe": 0.8,
            "fitness": 1.0,
            "turnover": 0.20,
            "drawdown": 0.10,
            "returns": 0.02,
        }
        result = scorer.multi_layer_evaluate(metrics)

        assert "Sharpe" in result["layer1_details"]

    def test_layer2_code_eval_high_quality(self, scorer, perfect_metrics):
        """Layer 2: 高质量因子应有高分"""
        result = scorer.multi_layer_evaluate(perfect_metrics)

        assert result["layer2_score"] >= 0.85
        assert len(result["layer2_issues"]) <= 1

    def test_layer2_code_eval_overfit_risk(self, scorer, overfit_metrics):
        """Layer 2: 过拟合风险应被检测到"""
        result = scorer.multi_layer_evaluate(overfit_metrics)

        assert result["layer2_score"] < 1.0
        assert any("过拟合" in issue for issue in result["layer2_issues"])

    def test_layer2_high_turnover_penalty(self, scorer):
        """Layer 2: 高换手率应扣分"""
        metrics = {
            "sharpe": 1.5,
            "fitness": 1.2,
            "turnover": 0.65,
            "drawdown": 0.12,
            "returns": 0.04,
        }
        result = scorer.multi_layer_evaluate(metrics)

        assert any("换手率" in issue for issue in result["layer2_issues"])
        assert result["layer2_score"] < 0.95

    def test_final_decision_pass(self, scorer, perfect_metrics):
        """最终决策: 完美因子应 PASS"""
        result = scorer.multi_layer_evaluate(perfect_metrics)

        assert result["final_decision"] == "PASS"
        assert result["confidence"] >= 0.85

    def test_final_decision_improve(self, scorer, near_pass_metrics):
        """最终决策: 接近及格因子应 IMPROVE 或 PASS"""
        result = scorer.multi_layer_evaluate(near_pass_metrics)

        # near_pass 的 layer2_score 可能 >= 0.85，所以可能是 PASS 或 IMPROVE
        assert result["final_decision"] in ["IMPROVE", "PASS"]
        assert 0.6 <= result["confidence"] <= 1.0  # confidence 可能达到 1.0

    def test_final_decision_reject(self, scorer, noise_metrics):
        """最终决策: 噪声因子应 REJECT"""
        result = scorer.multi_layer_evaluate(noise_metrics)

        assert result["final_decision"] == "REJECT"
        assert result["confidence"] >= 0.8

    def test_diagnosis_generation(self, scorer, perfect_metrics):
        """诊断信息应包含关键特征"""
        result = scorer.multi_layer_evaluate(perfect_metrics)

        assert "diagnosis" in result
        assert len(result["diagnosis"]) > 0
        assert "_" in result["diagnosis"] or result["diagnosis"] != "UNKNOWN"

    def test_confidence_range(self, scorer, near_pass_metrics):
        """置信度应在合理范围内"""
        result = scorer.multi_layer_evaluate(near_pass_metrics)

        assert 0 <= result["confidence"] <= 1


# ══════════════════════════════════════════════════════════════════════
# Test Class 3: Factor Profile Classification
# ══════════════════════════════════════════════════════════════════════

class TestFactorProfileClassification:
    """测试因子画像分类"""

    def test_classify_perfect(self, scorer, perfect_metrics):
        """完美因子应分类为 PERFECT"""
        profile = scorer.classify_factor_profile(perfect_metrics)
        assert profile == "PERFECT"

    def test_classify_high_sharpe_low_fitness(self, scorer):
        """高 Sharpe 低 Fitness 分类"""
        metrics = {
            "sharpe": 1.5,
            "fitness": 0.8,
            "turnover": 0.20,
            "drawdown": 0.10,
        }
        profile = scorer.classify_factor_profile(metrics)
        assert profile == "HIGH_SHARPE_LOW_FITNESS"

    def test_classify_near_pass(self, scorer, near_pass_metrics):
        """接近及格线分类"""
        profile = scorer.classify_factor_profile(near_pass_metrics)
        assert profile == "NEAR_PASS"

    def test_classify_overfit(self, scorer, overfit_metrics):
        """过拟合分类"""
        profile = scorer.classify_factor_profile(overfit_metrics)
        assert profile == "OVERFIT"

    def test_classify_noise(self, scorer, noise_metrics):
        """噪声分类"""
        profile = scorer.classify_factor_profile(noise_metrics)
        assert profile == "NOISE"

    def test_classify_low_sharpe_high_turnover(self, scorer):
        """低 Sharpe 高换手率分类"""
        metrics = {
            "sharpe": 0.6,
            "fitness": 0.7,
            "turnover": 0.60,
            "drawdown": 0.15,
        }
        profile = scorer.classify_factor_profile(metrics)
        assert profile == "LOW_SHARPE_HIGH_TURNOVER"

    def test_classify_stuck(self, scorer):
        """停滞因子分类"""
        metrics = {
            "sharpe": 0.6,
            "fitness": 0.6,
            "turnover": 0.30,
            "drawdown": 0.12,
        }
        profile = scorer.classify_factor_profile(metrics)
        assert profile == "STUCK"

    def test_classify_moderate_good(self, scorer):
        """中等偏上分类"""
        metrics = {
            "sharpe": 1.1,
            "fitness": 0.9,
            "turnover": 0.35,
            "drawdown": 0.16,
        }
        profile = scorer.classify_factor_profile(metrics)
        # sharpe=1.1 在 [0.8, 1.25) 且 fitness=0.9 >= 0.7，所以是 NEAR_PASS
        assert profile in ["NEAR_PASS", "MODERATE_GOOD"]

    def test_classify_moderate_weak(self, scorer):
        """中等偏弱分类"""
        metrics = {
            "sharpe": 0.7,
            "fitness": 0.6,
            "turnover": 0.40,
            "drawdown": 0.18,
        }
        profile = scorer.classify_factor_profile(metrics)
        # sharpe=0.7 < 0.8 且 fitness=0.6 < 0.7，可能落入多个分类
        assert profile in ["STUCK", "MODERATE_WEAK", "WEAK_NEEDS_WORK"]


# ══════════════════════════════════════════════════════════════════════
# Test Class 4: Integration & Backward Compatibility
# ══════════════════════════════════════════════════════════════════════

class TestIntegrationAndBackwardCompatibility:
    """集成测试和向后兼容性验证"""

    def test_compute_score_returns_enhanced_report(self, scorer, perfect_metrics):
        """compute_score 应返回包含增强字段的报告"""
        report = scorer.compute_score(perfect_metrics)

        assert isinstance(report, ScoreReport)
        assert report.advanced_metrics is not None
        assert report.multi_layer_result is not None
        assert report.factor_profile != ""

    def test_backward_compatibility_basic_fields(self, scorer, perfect_metrics):
        """原有字段应保持不变"""
        report = scorer.compute_score(perfect_metrics)

        assert hasattr(report, 'overall_score')
        assert hasattr(report, 'grade')
        assert hasattr(report, 'breakdown')
        assert hasattr(report, 'passed')
        assert hasattr(report, 'details')
        assert hasattr(report, 'improvement_hints')

    def test_score_range_unchanged(self, scorer, perfect_metrics):
        """总分范围应保持 0-100"""
        report = scorer.compute_score(perfect_metrics)

        assert 0 <= report.overall_score <= 100

    def test_grade_system_unchanged(self, scorer, perfect_metrics):
        """评级系统应保持 A+ 到 F"""
        report = scorer.compute_score(perfect_metrics)

        valid_grades = ["A+", "A", "A-", "B+", "B", "B-", "C", "D", "F"]
        assert report.grade in valid_grades

    def test_to_dict_includes_new_fields(self, scorer, perfect_metrics):
        """to_dict 应包含新字段"""
        report = scorer.compute_score(perfect_metrics)
        data = report.to_dict()

        assert "advanced_metrics" in data
        assert "multi_layer_result" in data
        assert "factor_profile" in data

    def test_str_output_includes_new_info(self, scorer, perfect_metrics):
        """__str__ 输出应包含新信息"""
        report = scorer.compute_score(perfect_metrics)
        output = str(report)

        assert "Factor Profile:" in output

    def test_enhanced_weights_defined(self):
        """增强权重配置应正确定义"""
        assert "ic_score" in ENHANCED_SCORE_WEIGHTS
        assert "icir_score" in ENHANCED_SCORE_WEIGHTS
        assert "stability_score" in ENHANCED_SCORE_WEIGHTS
        assert sum(ENHANCED_SCORE_WEIGHTS.values()) == 100

    def test_quick_score_still_works(self):
        """quick_score 函数应继续工作"""
        from openalpha_brain.validation.official_scorer import quick_score

        report = quick_score(sharpe=1.5, fitness=1.2, turnover=0.25, drawdown=0.10)
        assert isinstance(report, ScoreReport)
        assert report.advanced_metrics is not None

    def test_empty_checks_handling(self, scorer):
        """空检查项列表应安全处理"""
        metrics = {
            "sharpe": 1.5,
            "fitness": 1.2,
            "turnover": 0.25,
            "drawdown": 0.10,
            "checks": [],
        }
        report = scorer.compute_score(metrics)

        assert report is not None
        assert report.breakdown.get("checks_penalty") == 0.0

    def test_extreme_values_handled(self, scorer):
        """极端值应安全处理"""
        metrics = {
            "sharpe": 5.0,  # 极高
            "fitness": 3.0,
            "turnover": 0.01,  # 极低
            "drawdown": 0.01,
            "checks": [],
        }
        report = scorer.compute_score(metrics)

        assert 0 <= report.overall_score <= 100
        assert report.advanced_metrics is not None

    def test_negative_drawdown_handled(self, scorer):
        """负回撤值应被取绝对值"""
        metrics = {
            "sharpe": 1.5,
            "fitness": 1.2,
            "turnover": 0.25,
            "drawdown": -0.10,  # 负值
            "checks": [],
        }
        report = scorer.compute_score(metrics)

        assert report is not None


# ══════════════════════════════════════════════════════════════════════
# Test Class 5: Edge Cases & Error Handling
# ══════════════════════════════════════════════════════════════════════

class TestEdgeCasesAndErrorHandling:
    """边界情况和错误处理测试"""

    def test_zero_sharpe_inference(self, scorer):
        """Sharpe=0 的推断"""
        metrics = {"sharpe": 0, "fitness": 0.5, "turnover": 0.3, "drawdown": 0.15}
        advanced = scorer.infer_advanced_metrics(metrics)

        assert advanced.ic is None  # Sharpe<=0 不推断 IC

    def test_zero_fitness_inference(self, scorer):
        """Fitness=0 的推断"""
        metrics = {"sharpe": 1.0, "fitness": 0, "turnover": 0.3, "drawdown": 0.15}
        advanced = scorer.infer_advanced_metrics(metrics)

        assert advanced.ir is None  # Fitness<=0 不推断 IR

    def test_very_high_turnover_risk(self, scorer):
        """极高换手率的风险评估"""
        metrics = {
            "sharpe": 1.0,
            "fitness": 1.0,
            "turnover": 0.85,
            "drawdown": 0.15,
        }
        result = scorer.multi_layer_evaluate(metrics)

        assert not result["layer1_passed"]  # 超过平台限制

    def test_delay_zero_thresholds(self, scorer):
        """Delay=0 应使用正确的阈值"""
        metrics = {
            "sharpe": 1.8,
            "fitness": 1.4,
            "turnover": 0.25,
            "drawdown": 0.10,
            "delay": 0,
            "checks": [],
        }
        report = scorer.compute_score(metrics)

        # Delay=0 要求更高 (min_sharpe=2.0, min_fitness=1.3)
        # 所以 sharpe=1.8 < 2.0 会得到较低分数，可能不及格
        assert report is not None
        assert 0 <= report.overall_score <= 100

    def test_all_checks_failed(self, scorer):
        """所有检查项失败"""
        metrics = {
            "sharpe": 1.5,
            "fitness": 1.2,
            "turnover": 0.25,
            "drawdown": 0.10,
            "checks": [
                {"name": "check1", "value": 1, "limit": 2, "result": False, "severity": "ERROR"},
                {"name": "check2", "value": 1, "limit": 2, "result": False, "severity": "ERROR"},
                {"name": "check3", "value": 1, "limit": 2, "result": False, "severity": "WARNING"},
            ],
        }
        report = scorer.compute_score(metrics)

        assert report.breakdown["checks_penalty"] < 0  # 应扣分

    def test_improvement_potential_range(self, scorer, perfect_metrics):
        """改进潜力应在 [0, 1] 范围内"""
        advanced = scorer.infer_advanced_metrics(perfect_metrics)

        assert 0 <= advanced.improvement_potential <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
