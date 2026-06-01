"""
OfficialScoringAdapter 单元测试

覆盖评分计算、评级、改进建议等核心功能
"""

import pytest

from openalpha_brain.validation.official_scorer import (
    GRADE_THRESHOLDS,
    OfficialScoringAdapter,
    ScoreReport,
    evaluate_alpha_quality,
    quick_score,
)
from openalpha_brain.validation.wq_expression_validator import ValidationResult


class TestOfficialScoringAdapter:
    """官方评分适配器测试"""

    def setup_method(self):
        self.scorer = OfficialScoringAdapter()

    def test_basic_score_calculation(self):
        """基本评分计算"""
        metrics = {
            "sharpe": 1.75,
            "fitness": 1.25,
            "turnover": 0.25,
            "returns": 0.05,
            "drawdown": 0.15,
            "checks": [],
        }
        report = self.scorer.compute_score(metrics)

        assert isinstance(report, ScoreReport)
        assert 0 <= report.overall_score <= 100
        assert report.grade in [g for _, g in GRADE_THRESHOLDS]
        assert isinstance(report.breakdown, dict)

    def test_perfect_metrics(self):
        """完美指标应得高分"""
        metrics = {
            "sharpe": 2.5,
            "fitness": 2.5,
            "turnover": 0.20,  # 理想范围
            "returns": 0.10,
            "drawdown": 0.05,  # 很低
            "checks": [],
        }
        report = self.scorer.compute_score(metrics)

        assert report.overall_score >= 85
        assert report.grade in ["A+", "A", "A-"]
        assert report.passed is True

    def test_poor_metrics(self):
        """差指标应得低分"""
        metrics = {
            "sharpe": 0.5,  # 很低
            "fitness": 0.3,  # 很低
            "turnover": 0.80,  # 超过平台限制
            "returns": -0.02,
            "drawdown": 0.40,  # 很高
            "checks": [
                {"name": "test1", "value": None, "limit": None, "result": False, "severity": "ERROR"},
                {"name": "test2", "value": None, "limit": None, "result": False, "severity": "ERROR"},
                {"name": "test3", "value": None, "limit": None, "result": False, "severity": "WARNING"},
            ],
        }
        report = self.scorer.compute_score(metrics)

        assert report.overall_score < 50
        assert report.grade in ["D", "F"]
        assert report.passed is False


class TestSharpeScore:
    """Sharpe 分数计算测试"""

    def test_sharpe_above_2(self):
        """Sharpe >= 2.0 应得满分 (40)"""
        score = OfficialScoringAdapter._calc_sharpe_score(2.5)
        assert score == 40.0

    def test_sharpe_at_threshold(self):
        """Sharpe 在阈值处应得基础分 (20)"""
        score = OfficialScoringAdapter._calc_sharpe_score(1.25)
        assert score == 20.0

    def test_sharpe_below_threshold(self):
        """Sharpe 低于阈值应得低分"""
        score = OfficialScoringAdapter._calc_sharpe_score(0.8)
        assert 0 < score < 20

    def test_negative_sharpe(self):
        """负 Sharpe 应得 0 分"""
        score = OfficialScoringAdapter._calc_sharpe_score(-0.5)
        assert score == 0.0

    def test_zero_sharpe(self):
        """零 Sharpe 应得 0 分"""
        score = OfficialScoringAdapter._calc_sharpe_score(0.0)
        assert score == 0.0

    def test_delay0_threshold(self):
        """Delay=0 时使用更高的 Sharpe 阈值"""
        score_normal = OfficialScoringAdapter._calc_sharpe_score(1.75, delay=1)
        score_delay0 = OfficialScoringAdapter._calc_sharpe_score(1.75, delay=0)

        # Delay=0 的阈值更高，所以相同 Sharpe 得分应该更低或相等
        assert score_delay0 <= score_normal


class TestFitnessScore:
    """Fitness 分数计算测试"""

    def test_fitness_above_2(self):
        """Fitness >= 2.0 应得满分 (30)"""
        score = OfficialScoringAdapter._calc_fitness_score(2.5)
        assert score == 30.0

    def test_fitness_at_threshold(self):
        """Fitness 在阈值处应得基础分 (15)"""
        score = OfficialScoringAdapter._calc_fitness_score(1.0)
        assert score == 15.0

    def test_fitness_below_threshold(self):
        """Fitness 低于阈值应得低分"""
        score = OfficialScoringAdapter._calc_fitness_score(0.6)
        assert 0 < score < 15

    def test_negative_fitness(self):
        """负 Fitness 应得 0 分"""
        score = OfficialScoringAdapter._calc_fitness_score(-0.3)
        assert score == 0.0


class TestTurnoverScore:
    """换手率分数计算测试"""

    def test_ideal_turnover_range(self):
        """理想换手率范围 (5%-30%) 应得满分 (15)"""
        for t in [0.05, 0.10, 0.20, 0.30]:
            score = OfficialScoringAdapter._calc_turnover_score(t)
            assert score == 15.0, f"turnover={t} should get 15, got {score}"

    def test_acceptable_turnover_range(self):
        """可接受换手率范围 (30%-50%) 应得中等分数"""
        for t in [0.35, 0.40, 0.45, 0.50]:
            score = OfficialScoringAdapter._calc_turnover_score(t)
            assert 10 <= score <= 15, f"turnover={t} should get 10-15, got {score}"

    def test_high_turnover_near_limit(self):
        """接近平台上限的换手率 (50%-70%) 应得低分"""
        for t in [0.55, 0.60, 0.65, 0.70]:
            score = OfficialScoringAdapter._calc_turnover_score(t)
            assert 3 <= score <= 10, f"turnover={t} should get 3-10, got {score}"

    def test_excessive_turnover(self):
        """超过平台限制的换手率应得极低分"""
        score = OfficialScoringAdapter._calc_turnover_score(0.80)
        assert score < 5

    def test_low_turnover(self):
        """过低换手率 (< 1%) 应得低分"""
        score = OfficialScoringAdapter._calc_turnover_score(0.005)
        assert 0 <= score < 12

    def test_zero_turnover(self):
        """零换手率应得 0 分"""
        score = OfficialScoringAdapter._calc_turnover_score(0.0)
        assert score == 0.0


class TestDrawdownScore:
    """回撤分数计算测试"""

    def test_very_low_drawdown(self):
        """极低回撤 (< 5%) 应得高分"""
        for dd in [0.01, 0.03, 0.05]:
            score = OfficialScoringAdapter._calc_drawdown_score(dd)
            assert score >= 9.0, f"drawdown={dd} should get >=9, got {score}"

    def test_moderate_drawdown(self):
        """中等回撤 (10%-20%) 应得中等分数"""
        for dd in [0.12, 0.15, 0.18, 0.20]:
            score = OfficialScoringAdapter._calc_drawdown_score(dd)
            assert 5 <= score <= 9, f"drawdown={dd} should get 5-9, got {score}"

    def test_high_drawdown(self):
        """高回撤 (> 25%) 应得低分"""
        for dd in [0.25, 0.30, 0.35]:
            score = OfficialScoringAdapter._calc_drawdown_score(dd)
            assert score <= 5, f"drawdown={dd} should get <=5, got {score}"

    def test_extreme_drawdown(self):
        """极端回撤 (> 35%) 应得极低分"""
        score = OfficialScoringAdapter._calc_drawdown_score(0.50)
        assert score <= 2


class TestChecksPenalty:
    """检查项扣分测试"""

    def test_all_checks_passed(self):
        """所有检查通过不扣分"""
        checks = [
            {"name": "c1", "value": None, "limit": None, "result": True},
            {"name": "c2", "value": None, "limit": None, "result": True},
        ]
        penalty = OfficialScoringAdapter._calc_checks_penalty(checks)
        assert penalty == 0.0

    def test_error_failures(self):
        """ERROR 级别失败每个扣 1 分"""
        checks = [
            {"name": "c1", "value": None, "limit": None, "result": False, "severity": "ERROR"},
            {"name": "c2", "value": None, "limit": None, "result": False, "severity": "ERROR"},
        ]
        penalty = OfficialScoringAdapter._calc_checks_penalty(checks)
        assert penalty == -2.0

    def test_warning_failures(self):
        """WARNING 级别失败每个扣 0.5 分"""
        checks = [
            {"name": "c1", "value": None, "limit": None, "result": False, "severity": "WARNING"},
            {"name": "c2", "value": None, "limit": None, "result": False, "severity": "WARNING"},
        ]
        penalty = OfficialScoringAdapter._calc_checks_penalty(checks)
        assert penalty == -1.0

    def test_max_penalty_capped(self):
        """最大扣分为 5 分"""
        checks = [
            {"name": f"c{i}", "value": None, "limit": None, "result": False, "severity": "ERROR"} for i in range(10)
        ]
        penalty = OfficialScoringAdapter._calc_checks_penalty(checks)
        assert penalty == -5.0

    def test_info_no_penalty(self):
        """INFO 级别失败不扣分"""
        checks = [
            {"name": "c1", "value": None, "limit": None, "result": False, "severity": "INFO"},
        ]
        penalty = OfficialScoringAdapter._calc_checks_penalty(checks)
        assert penalty == 0.0

    def test_mixed_severities(self):
        """混合严重级别"""
        checks = [
            {"name": "c1", "value": None, "limit": None, "result": False, "severity": "ERROR"},
            {"name": "c2", "value": None, "limit": None, "result": False, "severity": "WARNING"},
            {"name": "c3", "value": None, "limit": None, "result": False, "severity": "INFO"},
        ]
        penalty = OfficialScoringAdapter._calc_checks_penalty(checks)
        assert penalty == -1.5  # -1 + (-0.5) + 0


class TestGradeSystem:
    """评级系统测试"""

    def test_a_plus_grade(self):
        """90+ 分应为 A+"""
        grade = OfficialScoringAdapter().grade(95)
        assert grade == "A+"

    def test_a_grade(self):
        """80-89 分应为 A"""
        grade = OfficialScoringAdapter().grade(85)
        assert grade == "A"

    def test_a_minus_grade(self):
        """70-79 分应为 A-"""
        grade = OfficialScoringAdapter().grade(75)
        assert grade == "A-"

    def test_b_plus_grade(self):
        """60-69 分应为 B+"""
        grade = OfficialScoringAdapter().grade(62)
        assert grade == "B+"

    def test_b_grade(self):
        """50-59 分应为 B"""
        grade = OfficialScoringAdapter().grade(55)
        assert grade == "B"

    def test_b_minus_grade(self):
        """40-49 分应为 B-"""
        grade = OfficialScoringAdapter().grade(42)
        assert grade == "B-"

    def test_c_grade(self):
        """30-39 分应为 C"""
        grade = OfficialScoringAdapter().grade(33)
        assert grade == "C"

    def test_d_grade(self):
        """20-29 分应为 D"""
        grade = OfficialScoringAdapter().grade(25)
        assert grade == "D"

    def test_f_grade(self):
        """低于 20 分应为 F"""
        grade = OfficialScoringAdapter().grade(10)
        assert grade == "F"
        grade = OfficialScoringAdapter().grade(0)
        assert grade == "F"


class TestImprovementHints:
    """改进建议生成测试"""

    def test_hints_for_low_sharpe(self):
        """低 Sharpe 应生成相关建议"""
        scorer = OfficialScoringAdapter()
        metrics = {
            "sharpe": 0.8,
            "fitness": 1.5,
            "turnover": 0.20,
            "drawdown": 0.10,
            "checks": [],
        }
        breakdown = {"sharpe_score": 12.8}
        hints = scorer._generate_improvement_hints(metrics, breakdown)

        assert len(hints) > 0
        assert any("Sharpe" in h for h in hints)

    def test_hints_for_high_turnover(self):
        """高换手率应生成相关建议"""
        scorer = OfficialScoringAdapter()
        metrics = {
            "sharpe": 1.75,
            "fitness": 1.25,
            "turnover": 0.60,
            "drawdown": 0.10,
            "checks": [],
        }
        breakdown = {"turnover_score": 7}
        hints = scorer._generate_improvement_hints(metrics, breakdown)

        assert any("换手率" in h for h in hints)

    def test_hints_for_failed_checks(self):
        """失败的检查项应生成建议"""
        scorer = OfficialScoringAdapter()
        metrics = {
            "sharpe": 1.75,
            "fitness": 1.25,
            "turnover": 0.20,
            "drawdown": 0.10,
            "checks": [
                {"name": "self_correlation", "value": 0.80, "limit": 0.70, "result": False},
            ],
        }
        breakdown = {}
        hints = scorer._generate_improvement_hints(metrics, breakdown)

        assert any("自相关" in h for h in hints)


class TestScoreReport:
    """评分报告测试"""

    def test_report_to_dict(self):
        """报告序列化为字典"""
        scorer = OfficialScoringAdapter()
        metrics = {
            "sharpe": 1.5,
            "fitness": 1.2,
            "turnover": 0.25,
            "returns": 0.04,
            "drawdown": 0.12,
            "checks": [],
        }
        report = scorer.compute_score(metrics)
        d = report.to_dict()

        assert isinstance(d, dict)
        assert "overall_score" in d
        assert "grade" in d
        assert "passed" in d
        assert "breakdown" in d
        assert "improvement_hints" in d

    def test_report_str_representation(self):
        """报告的字符串表示"""
        scorer = OfficialScoringAdapter()
        metrics = {
            "sharpe": 1.5,
            "fitness": 1.2,
            "turnover": 0.25,
            "returns": 0.04,
            "drawdown": 0.12,
            "checks": [],
        }
        report = scorer.compute_score(metrics)
        s = str(report)

        assert "Score Report" in s
        assert report.grade in s
        assert ("PASS" if report.passed else "FAIL") in s


class TestQuickScore:
    """快速评分函数测试"""

    def test_quick_score_basic(self):
        """快速评分基本功能"""
        report = quick_score(sharpe=1.75, fitness=1.25, turnover=0.20)

        assert isinstance(report, ScoreReport)
        assert 0 <= report.overall_score <= 100

    def test_quick_score_with_kwargs(self):
        """快速评分支持额外参数"""
        report = quick_score(
            sharpe=1.5,
            fitness=1.0,
            turnover=0.25,
            drawdown=0.15,
            delay=1,
        )

        assert isinstance(report, ScoreReport)
        assert "drawdown_score" in report.breakdown


class TestEvaluateAlphaQuality:
    """完整评估流程测试"""

    def test_evaluate_alpha_quality_integration(self):
        """验证和评分集成测试"""
        expression = "group_neutralize(ts_decay_linear(rank(close), 5), industry)"
        metrics = {
            "sharpe": 1.75,
            "fitness": 1.25,
            "turnover": 0.20,
            "returns": 0.05,
            "drawdown": 0.12,
            "checks": [],
        }

        validation_result, score_report = evaluate_alpha_quality(expression, metrics)

        assert isinstance(validation_result, ValidationResult)
        assert isinstance(score_report, ScoreReport)

    def test_custom_validator(self):
        """使用自定义验证器"""
        from openalpha_brain.validation.wq_expression_validator import WQExpressionValidator

        custom_validator = WQExpressionValidator(max_depth=3)
        expression = "rank(close)"
        metrics = {"sharpe": 1.0, "fitness": 0.8, "turnover": 0.15}

        validation_result, _ = evaluate_alpha_quality(expression, metrics, validator=custom_validator)

        assert isinstance(validation_result, ValidationResult)


class TestCustomConfiguration:
    """自定义配置测试"""

    def test_custom_pass_threshold(self):
        """自定义通过阈值"""
        scorer = OfficialScoringAdapter(pass_threshold=70.0)
        metrics = {
            "sharpe": 1.5,
            "fitness": 1.2,
            "turnover": 0.25,
            "drawdown": 0.15,
            "checks": [],
        }
        report = scorer.compute_score(metrics)

        # 65 分左右，如果阈值是 70 则不应通过
        assert report.passed == (report.overall_score >= 70.0)

    def test_custom_weights(self):
        """自定义评分权重（暂未实现，预留接口）"""
        # 这个功能目前只是预留，实际使用默认权重
        scorer = OfficialScoringAdapter(custom_weights=None)
        assert scorer is not None


class TestEdgeCases:
    """边界情况测试"""

    def test_missing_optional_fields(self):
        """缺少可选字段应使用默认值"""
        scorer = OfficialScoringAdapter()
        metrics = {
            "sharpe": 1.5,
            "fitness": 1.0,
            "turnover": 0.20,
            # 缺少 returns, drawdown, checks
        }
        report = scorer.compute_score(metrics)

        assert isinstance(report, ScoreReport)
        assert 0 <= report.overall_score <= 100

    def test_extreme_values(self):
        """极端值处理"""
        scorer = OfficialScoringAdapter()
        metrics = {
            "sharpe": 100.0,  # 极高的 Sharpe
            "fitness": 50.0,  # 极高的 Fitness
            "turnover": 0.001,  # 极低的 turnover
            "drawdown": 0.99,  # 极高的 drawdown
            "checks": [],
        }
        report = scorer.compute_score(metrics)

        # 所有值应该在合理范围内
        assert 0 <= report.overall_score <= 100
        for key, value in report.breakdown.items():
            if isinstance(value, float):
                assert -10 <= value <= 50  # 允许一定的扣分范围

    def test_empty_checks_list(self):
        """空检查列表"""
        scorer = OfficialScoringAdapter()
        metrics = {
            "sharpe": 1.5,
            "fitness": 1.2,
            "turnover": 0.25,
            "checks": [],  # 空列表
        }
        report = scorer.compute_score(metrics)

        assert report.breakdown["checks_penalty"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
