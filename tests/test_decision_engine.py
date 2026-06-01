"""
Unit tests for DecisionEngine (Pipeline Stage 2)
===============================================
测试因子改进决策引擎的5档决策矩阵、Anti-Fit调节和策略D增强。

覆盖范围:
  1. 5档决策矩阵基础测试
  2. Anti-Overfit 惩罚降级
  3. ICIR/MFR 策略D增强
  4. 边界值测试 (精确阈值)
  5. decide_from_parsed_result 便捷方法
  6. DecisionContext / DecisionOutcome dataclass
  7. 自定义配置
  8. 负 Sharpe 和噪音因子
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from openalpha_brain.core.decision_engine import (
    DecisionEngine,
    DecisionContext,
    DecisionOutcome,
    DecisionAction,
)
from openalpha_brain.core.result_router import ParsedWQResult


@pytest.fixture
def default_engine():
    """使用默认配置的 DecisionEngine"""
    return DecisionEngine()


@pytest.fixture
def custom_engine():
    """使用自定义阈值的 DecisionEngine"""
    return DecisionEngine(config={
        "sharpe_pass_threshold": 1.5,
        "sharpe_improve_threshold": 1.0,
        "weak_improve_threshold": 0.2,
        "repair_retry_threshold": -0.3,
        "anti_fit_penalty_threshold": 35,
    })


@pytest.fixture
def parsed_result_high_sharpe():
    """高 Sharpe 的 ParsedWQResult"""
    return ParsedWQResult(
        task_id="t1",
        expression="rank(volume)",
        raw_sharpe=1.35,
        passed=True,
        wq_metrics={"fitness": 1.15, "turnover": 0.16},
        anti_fit_score=72.0,
        official_score=88.0,
        official_grade="A",
    )


@pytest.fixture
def parsed_result_mid_sharpe():
    """中等 Sharpe 的 ParsedWQResult"""
    return ParsedWQResult(
        task_id="t2",
        expression="ts_mean(close, 20)",
        raw_sharpe=1.0,
        passed=False,
        wq_metrics={"fitness": 0.85, "turnover": 0.22},
        anti_fit_score=55.0,
        official_score=65.0,
        official_grade="B",
    )


@pytest.fixture
def parsed_result_low_sharpe():
    """低 Sharpe 的 ParsedWQResult"""
    return ParsedWQResult(
        task_id="t3",
        expression="ts_decay_linear(rank(low), 5)",
        raw_sharpe=0.4,
        passed=False,
        wq_metrics={"fitness": 0.45, "turnover": 0.35},
        anti_fit_score=30.0,
        official_score=40.0,
        official_grade="C",
    )


class TestDecisionMatrixBasic:
    """5档决策矩阵基础测试"""

    def test_success_pass_high_sharpe(self, default_engine):
        """Sharpe >= 1.25 且 passed → SUCCESS_PASS"""
        ctx = DecisionContext(sharpe=1.5, passed=True)
        outcome = default_engine.decide(ctx)

        assert outcome.action == DecisionAction.SUCCESS_PASS
        assert outcome.should_improve is False
        assert outcome.improvement_priority == 0

    def test_success_pass_exact_threshold(self, default_engine):
        """Sharpe == 1.25 且 passed → SUCCESS_PASS (边界)"""
        ctx = DecisionContext(sharpe=1.25, passed=True)
        outcome = default_engine.decide(ctx)

        assert outcome.action == DecisionAction.SUCCESS_PASS

    def test_improve_strong_signal(self, default_engine):
        """Sharpe ∈ [0.8, 1.25) → IMPROVE_AND_RESUBMIT"""
        ctx = DecisionContext(sharpe=1.1, passed=False)
        outcome = default_engine.decide(ctx)

        assert outcome.action == DecisionAction.IMPROVE_AND_RESUBMIT
        assert outcome.should_improve is True
        assert outcome.improvement_priority == 4

    def test_weak_improve(self, default_engine):
        """Sharpe ∈ [0, 0.8) → WEAK_IMPROVE"""
        ctx = DecisionContext(sharpe=0.5, passed=False)
        outcome = default_engine.decide(ctx)

        assert outcome.action == DecisionAction.WEAK_IMPROVE
        assert outcome.should_improve is True
        assert outcome.improvement_priority == 2

    def test_repair_and_retry(self, default_engine):
        """Sharpe ∈ [-0.5, 0) → REPAIR_AND_RETRY"""
        ctx = DecisionContext(sharpe=-0.3, passed=False)
        outcome = default_engine.decide(ctx)

        assert outcome.action == DecisionAction.REPAIR_AND_RETRY
        assert outcome.improvement_priority == 1

    def test_noise_penalty(self, default_engine):
        """Sharpe < -0.5 → NOISE_PENALIZE"""
        ctx = DecisionContext(sharpe=-0.8, passed=False)
        outcome = default_engine.decide(ctx)

        assert outcome.action == DecisionAction.NOISE_PENALIZE
        assert outcome.should_improve is False
        assert outcome.is_noise is True

    def test_noise_exact_boundary(self, default_engine):
        """Sharpe == -0.5 → REPAIR_AND_RETRY (边界，不在噪音区)"""
        ctx = DecisionContext(sharpe=-0.5, passed=False)
        outcome = default_engine.decide(ctx)

        assert outcome.action == DecisionAction.REPAIR_AND_RETRY


class TestAntiFitPenalty:
    """Anti-Overfit 惩罚降级测试"""

    def test_anti_fit_downgrades_improve_to_weak(self, default_engine):
        """Anti-Fit score < 40: IMPROVE → WEAK_IMPROVE"""
        ctx = DecisionContext(
            sharpe=1.1,
            passed=False,
            anti_fit_score=35.0,  # < 40 阈值
        )
        outcome = default_engine.decide(ctx)

        assert outcome.action == DecisionAction.WEAK_IMPROVE
        assert outcome.anti_fit_penalty_applied is True
        assert outcome.improvement_priority == 2  # 降级后优先级降低

    def test_anti_fit_downgrades_weak_to_repair(self, default_engine):
        """Anti-Fit score < 40: WEAK_IMPROVE → REPAIR_AND_RETRY"""
        ctx = DecisionContext(
            sharpe=0.5,
            passed=False,
            anti_fit_score=30.0,
        )
        outcome = default_engine.decide(ctx)

        assert outcome.action == DecisionAction.REPAIR_AND_RETRY
        assert outcome.anti_fit_penalty_applied is True

    def test_anti_fit_no_effect_on_success(self, default_engine):
        """Anti-Fit 不影响 SUCCESS_PASS (只记录警告)"""
        ctx = DecisionContext(
            sharpe=1.5,
            passed=True,
            anti_fit_score=25.0,
        )
        outcome = default_engine.decide(ctx)

        assert outcome.action == DecisionAction.SUCCESS_PASS
        assert outcome.anti_fit_penalty_applied is True
        assert "ANTI-FIT" in outcome.reason  # 但仍然通过

    def test_anti_fit_above_threshold_no_penalty(self, default_engine):
        """Anti-Fit score >= 40: 无惩罚"""
        ctx = DecisionContext(
            sharpe=1.1,
            passed=False,
            anti_fit_score=65.0,  # > 40
        )
        outcome = default_engine.decide(ctx)

        assert outcome.action == DecisionAction.IMPROVE_AND_RESUBMIT
        assert outcome.anti_fit_penalty_applied is False

    def test_anti_fit_none_no_penalty(self, default_engine):
        """Anti-Fit score 为 None: 无惩罚"""
        ctx = DecisionContext(
            sharpe=1.1,
            passed=False,
            anti_fit_score=None,
        )
        outcome = default_engine.decide(ctx)

        assert outcome.action == DecisionAction.IMPROVE_AND_RESUBMIT
        assert outcome.anti_fit_penalty_applied is False


class TestStrategyDEnhancement:
    """策略D (ICIR/MFR) 增强测试"""

    def test_high_icir_low_fitness_forces_improve(self, default_engine):
        """HIGH_ICIR_LOW_FITNESS 强制改进路径"""
        ctx = DecisionContext(
            sharpe=0.7,  # < improve_threshold (0.8)，但 > 0.8*0.8=0.64
            passed=False,
            is_high_icir_low_fitness=True,
        )
        outcome = default_engine.decide(ctx)

        assert outcome.action == DecisionAction.IMPROVE_AND_RESUBMIT
        assert outcome.strategy_d_boost is True
        assert outcome.improvement_priority == 5
        assert "HIGH_ICIR_LOW_FITNESS" in outcome.reason

    def test_efficient_alpha_boosts_priority(self, default_engine):
        """EFFICIENT_ALPHA 提升 IMPROVE 优先级到最高"""
        ctx = DecisionContext(
            sharpe=1.1,
            passed=False,
            is_efficient_alpha=True,
        )
        outcome = default_engine.decide(ctx)

        assert outcome.action == DecisionAction.IMPROVE_AND_RESUBMIT
        assert outcome.strategy_d_boost is True
        assert outcome.improvement_priority == 5
        assert "EFFICIENT_ALPHA" in outcome.reason

    def test_efficient_alpha_with_success(self, default_engine):
        """EFFICIENT_ALPHA + SUCCESS_PASS: 记录但不改变决策"""
        ctx = DecisionContext(
            sharpe=1.5,
            passed=True,
            is_efficient_alpha=True,
        )
        outcome = default_engine.decide(ctx)

        assert outcome.action == DecisionAction.SUCCESS_PASS
        assert outcome.strategy_d_boost is True  # 标记了boost
        assert "EFFICIENT_ALPHA" in outcome.reason

    def test_both_icir_and_efficient(self, default_engine):
        """同时有 ICIR 和 EFFICIENT: ICIR 优先级更高"""
        ctx = DecisionContext(
            sharpe=0.9,
            passed=False,
            is_high_icir_low_fitness=True,
            is_efficient_alpha=True,
        )
        outcome = default_engine.decide(ctx)

        # ICIR 条件先匹配，所以走 ICIR 路径
        assert outcome.action == DecisionAction.IMPROVE_AND_RESUBMIT
        assert "HIGH_ICIR_LOW_FITNESS" in outcome.reason


class TestBoundaryValues:
    """边界值精确测试"""

    def test_exact_improve_threshold(self, default_engine):
        """Sharpe == 0.8 (improve_threshold 下界)"""
        ctx = DecisionContext(sharpe=0.8, passed=False)
        outcome = default_engine.decide(ctx)

        assert outcome.action == DecisionAction.IMPROVE_AND_RESUBMIT

    def test_exact_weak_threshold(self, default_engine):
        """Sharpe == 0.0 (weak_threshold 下界)"""
        ctx = DecisionContext(sharpe=0.0, passed=False)
        outcome = default_engine.decide(ctx)

        assert outcome.action == DecisionAction.WEAK_IMPROVE

    def test_zero_sharpe(self, default_engine):
        """Sharpe == 0.0 (刚好在 weak 区间)"""
        ctx = DecisionContext(sharpe=0.0, passed=False)
        outcome = default_engine.decide(ctx)

        assert outcome.action == DecisionAction.WEAK_IMPROVE

    def test_very_small_positive(self, default_engine):
        """Sharpe = 0.001 (接近0但为正)"""
        ctx = DecisionContext(sharpe=0.001, passed=False)
        outcome = default_engine.decide(ctx)

        assert outcome.action == DecisionAction.WEAK_IMPROVE

    def test_very_small_negative(self, default_engine):
        """Sharpe = -0.001 (接近0但为负)"""
        ctx = DecisionContext(sharpe=-0.001, passed=False)
        outcome = default_engine.decide(ctx)

        assert outcome.action == DecisionAction.REPAIR_AND_RETRY


class TestDecideFromParsedResult:
    """decide_from_parsed_result 便捷方法测试"""

    def test_basic_usage(self, default_engine, parsed_result_mid_sharpe):
        """基本用法：从 ParsedWQResult 直接决策"""
        outcome = default_engine.decide_from_parsed_result(parsed_result_mid_sharpe)

        assert isinstance(outcome, DecisionOutcome)
        assert outcome.action == DecisionAction.IMPROVE_AND_RESUBMIT

    def test_override_anti_fit_score(self, default_engine, parsed_result_mid_sharpe):
        """覆盖 anti_fit_score"""
        outcome = default_engine.decide_from_parsed_result(
            parsed_result_mid_sharpe,
            anti_fit_score=30.0,  # 覆盖原来的 55.0
        )

        assert outcome.anti_fit_penalty_applied is True
        assert outcome.action == DecisionAction.WEAK_IMPROVE  # 被降级

    def test_high_sharpe_passed(self, default_engine, parsed_result_high_sharpe):
        """高Sharpe + passed → SUCCESS_PASS"""
        outcome = default_engine.decide_from_parsed_result(parsed_result_high_sharpe)

        assert outcome.action == DecisionAction.SUCCESS_PASS
        assert outcome.is_success is True

    def test_low_sharpe_with_icir_flag(self, default_engine):
        """低Sharpe + ICIR标志 → 强制改进"""
        result = ParsedWQResult(
            raw_sharpe=0.7,
            passed=False,
            is_high_icir_low_fitness=True,
        )
        outcome = default_engine.decide_from_parsed_result(result)

        assert outcome.action == DecisionAction.IMPROVE_AND_RESUBMIT
        assert outcome.strategy_d_boost is True


class TestDecisionOutcome:
    """DecisionOutcome dataclass 测试"""

    def test_default_values(self):
        """默认初始化"""
        outcome = DecisionOutcome()

        assert outcome.action == DecisionAction.NOISE_PENALIZE
        assert outcome.reason == ""
        assert outcome.should_improve is False
        assert outcome.improvement_priority == 0

    def test_is_success_property(self):
        """is_success 快捷属性"""
        outcome = DecisionOutcome(action=DecisionAction.SUCCESS_PASS)
        assert outcome.is_success is True

        outcome_fail = DecisionOutcome(action=DecisionAction.IMPROVE_AND_RESUBMIT)
        assert outcome_fail.is_success is False

    def test_is_noise_property(self):
        """is_noise 快捷属性"""
        outcome = DecisionOutcome(action=DecisionAction.NOISE_PENALIZE)
        assert outcome.is_noise is True

        outcome_not_noise = DecisionOutcome(action=DecisionAction.WEAK_IMPROVE)
        assert outcome_not_noise.is_noise is False

    def test_to_log_dict_truncation(self):
        """to_log_dict 截断长reason"""
        outcome = DecisionOutcome(
            action=DecisionAction.IMPROVE_AND_RESUBMIT,
            reason="A" * 150,
            improvement_priority=4,
        )
        log_dict = outcome.to_log_dict()

        assert len(log_dict["reason"]) <= 103  # 100 + "..."
        assert log_dict["action"] == "improve_and_resubmit"
        assert log_dict["priority"] == 4


class TestCustomConfig:
    """自定义配置测试"""

    def test_custom_thresholds(self, custom_engine):
        """自定义阈值生效"""
        ctx = DecisionContext(sharpe=1.2, passed=True)
        outcome = custom_engine.decide(ctx)

        # pass_threshold=1.5, 所以 1.2 < 1.5 → 不是SUCCESS_PASS
        assert outcome.action != DecisionAction.SUCCESS_PASS
        # improve_threshold=1.0, 所以 1.2 >= 1.0 → IMPROVE
        assert outcome.action == DecisionAction.IMPROVE_AND_RESUBMIT

    def test_custom_anti_fit_threshold(self, custom_engine):
        """自定义 anti-fit 阈值"""
        ctx = DecisionContext(
            sharpe=1.2,
            passed=False,
            anti_fit_score=37.0,  # > 默认40, 但 > 自定义35
        )
        outcome = custom_engine.decide(ctx)

        assert outcome.anti_fit_penalty_applied is False  # 37 >= 35, 不惩罚


class TestEdgeCases:
    """边缘情况和异常测试"""

    def test_negative_sharpe_strong(self, default_engine):
        """强负 Sharpe → NOISE_PENALIZE"""
        ctx = DecisionContext(sharpe=-2.0, passed=False)
        outcome = default_engine.decide(ctx)

        assert outcome.action == DecisionAction.NOISE_PENALIZE

    def test_very_high_sharpe(self, default_engine):
        """极高 Sharpe → SUCCESS_PASS"""
        ctx = DecisionContext(sharpe=5.0, passed=True)
        outcome = default_engine.decide(ctx)

        assert outcome.action == DecisionAction.SUCCESS_PASS

    def test_processing_time_recorded(self, default_engine):
        """processing_time_ms 被记录"""
        ctx = DecisionContext(sharpe=1.0, passed=False)
        outcome = default_engine.decide(ctx)

        assert outcome.processing_time_ms > 0

    def test_reason_always_non_empty(self, default_engine):
        """reason 字段永远不为空"""
        for sharpe in [2.0, 1.3, 1.0, 0.5, 0.0, -0.3, -1.0]:
            ctx = DecisionContext(sharpe=sharpe, passed=False)
            outcome = default_engine.decide(ctx)
            assert len(outcome.reason) > 0, f"Reason empty for sharpe={sharpe}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
