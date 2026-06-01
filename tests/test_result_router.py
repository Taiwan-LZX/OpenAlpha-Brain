"""
Unit tests for ResultRouter (Pipeline Stage 1)
==============================================
测试 WQ 结果接收、解析与评估的独立模块。

覆盖范围:
  1. ParsedWQResult dataclass 字段和属性
  2. ResultRouter.route() 基本流程
  3. 稳定性评估集成
  4. 过拟合检测集成
  5. 官方评分计算集成
  6. 异常处理和容错
  7. AdaptiveNeutralizer 记录
  8. 边界情况和空值处理
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from openalpha_brain.core.result_router import ParsedWQResult, ResultRouter


@pytest.fixture
def mock_slot_info():
    """创建模拟的 SlotInfo 对象"""
    slot = MagicMock()
    slot.slot_id = 42
    slot.task_name = "test_task_001"
    slot.expression = "ts_decay_linear(rank(close/volume), 10)"
    return slot


@pytest.fixture
def mock_brain_result():
    """创建模拟的 WQ brain_result 对象"""
    result = MagicMock()
    result.sharpe = 1.35
    result.turnover = 0.18
    result.fitness = 1.12
    result.drawdown = -0.05
    result.brain_checks = ["check1", "check2"]
    result.passed = True
    return result


@pytest.fixture
def base_router():
    """创建基础 ResultRouter 实例 (无外部依赖)"""
    return ResultRouter(
        stability_guard=None,
        anti_overfit=None,
        scorer=None,
        adaptive_neutralizer=None,
    )


class TestParsedWQResult:
    """ParsedWQResult dataclass 测试"""

    def test_default_initialization(self):
        """测试默认初始化"""
        result = ParsedWQResult()
        assert result.task_id == ""
        assert result.expression == ""
        assert result.raw_sharpe == 0.0
        assert result.passed is False
        assert result.wq_metrics == {}
        assert result.stability_result is None
        assert result.is_stable is True
        assert result.should_restrict is False
        assert result.anti_fit_score == 50.0
        assert result.anti_fit_recommendation == "CAUTIOUS"
        assert result.error is None

    def test_custom_initialization(self):
        """测试自定义字段初始化"""
        result = ParsedWQResult(
            task_id="task_123",
            expression="rank(volume)",
            raw_sharpe=1.5,
            passed=True,
            wq_metrics={"sharpe": 1.5, "fitness": 1.2},
            official_score=85.0,
            official_grade="A",
        )
        assert result.task_id == "task_123"
        assert result.expression == "rank(volume)"
        assert result.raw_sharpe == 1.5
        assert result.passed is True
        assert result.wq_metrics["sharpe"] == 1.5
        assert result.official_score == 85.0
        assert result.official_grade == "A"

    def test_fitness_property(self):
        """测试 fitness 快捷属性"""
        result = ParsedWQResult(wq_metrics={"fitness": 1.25})
        assert result.fitness == 1.25

        result_empty = ParsedWQResult(wq_metrics={})
        assert result_empty.fitness is None

    def test_turnover_property(self):
        """测试 turnover 快捷属性"""
        result = ParsedWQResult(wq_metrics={"turnover": 0.15})
        assert result.turnover == 0.15

    def test_drawdown_property(self):
        """测试 drawdown 快捷属性"""
        result = ParsedWQResult(wq_metrics={"drawdown": -0.08})
        assert result.drawdown == -0.08

    def test_to_log_dict_truncation(self):
        """测试 to_log_dict 自动截断长表达式"""
        long_expr = "a" * 100
        result = ParsedWQResult(expression=long_expr, task_id="test")
        log_dict = result.to_log_dict()

        assert len(log_dict["expr"]) <= 83  # 80 + "..."
        assert log_dict["expr"].endswith("...")
        assert log_dict["task_id"] == "test"

    def test_to_log_dict_short_expression(self):
        """测试短表达式不被截断"""
        short_expr = "rank(volume)"
        result = ParsedWQResult(expression=short_expr)
        log_dict = result.to_log_dict()

        assert log_dict["expr"] == short_expr

    def test_to_log_dict_completeness(self):
        """测试 to_log_dict 包含所有关键字段"""
        result = ParsedWQResult(
            task_id="t1",
            expression="expr1",
            raw_sharpe=1.2,
            passed=True,
            wq_metrics={"fitness": 1.1},
            is_stable=True,
            anti_fit_score=75.0,
            official_score=80.0,
            error="test error",
        )
        log_dict = result.to_log_dict()

        expected_keys = {
            "task_id",
            "expr",
            "sharpe",
            "passed",
            "fitness",
            "is_stable",
            "anti_fit_score",
            "official_score",
            "error",
        }
        assert set(log_dict.keys()) >= expected_keys
        assert len(log_dict) >= 13


class TestResultRouterBasic:
    """ResultRouter 基本路由功能测试"""

    @pytest.mark.asyncio
    async def test_route_basic_parsing(self, base_router, mock_slot_info, mock_brain_result):
        """测试基本路由和解析功能"""
        result = await base_router.route(mock_slot_info, mock_brain_result)

        assert isinstance(result, ParsedWQResult)
        assert result.task_id == "test_task_001"
        assert result.expression == "ts_decay_linear(rank(close/volume), 10)"
        assert result.raw_sharpe == 1.35
        assert result.passed is True
        assert result.error is None
        assert result.processing_time_sec > 0

    @pytest.mark.asyncio
    async def test_route_wq_metrics_extraction(self, base_router, mock_brain_result):
        """测试 WQ 指标提取"""
        slot = MagicMock()
        slot.slot_id = 1
        slot.task_name = ""
        slot.expression = "test"

        result = await base_router.route(slot, mock_brain_result)

        assert result.wq_metrics["sharpe"] == 1.35
        assert result.wq_metrics["turnover"] == 0.18
        assert result.wq_metrics["fitness"] == 1.12
        assert result.wq_metrics["drawdown"] == -0.05
        assert len(result.wq_metrics["checks"]) == 2

    @pytest.mark.asyncio
    async def test_route_none_values(self, base_router):
        """测试空值/None值的容错处理"""
        slot = MagicMock()
        slot.slot_id = 99
        slot.task_name = None
        slot.expression = ""

        result_mock = MagicMock()
        result_mock.sharpe = None
        result_mock.turnover = None
        result_mock.fitness = None
        result_mock.drawdown = None
        result_mock.brain_checks = None
        result_mock.passed = False

        result = await base_router.route(slot, result_mock)

        assert result.raw_sharpe == 0.0  # None 被转换为 0.0
        assert result.passed is False
        assert result.wq_metrics["sharpe"] == 0.0
        assert result.wq_metrics["turnover"] is None
        assert result.error is None  # 不应该报错

    @pytest.mark.asyncio
    async def test_route_task_id_fallback(self, base_router):
        """测试 task_name 为空时使用 slot_id 回退"""
        slot = MagicMock()
        slot.slot_id = 77
        slot.task_name = None  # 或空字符串
        slot.expression = "test_expr"

        result_mock = MagicMock()
        result_mock.sharpe = 1.0
        result_mock.passed = True

        result = await base_router.route(slot, result_mock)

        assert "slot_77" in result.task_id


class TestResultRouterWithStabilityGuard:
    """稳定性评估集成测试"""

    @pytest.mark.asyncio
    async def test_stability_guard_called(self, mock_slot_info, mock_brain_result):
        """测试 StabilityGuard 被正确调用"""
        mock_guard = MagicMock()
        mock_guard.evaluate_and_guard.return_value = {
            "stability_score": 0.85,
            "is_stable": True,
            "instability_type": None,
            "severity": 0.1,
            "should_restrict": False,
        }

        router = ResultRouter(stability_guard=mock_guard)
        result = await router.route(mock_slot_info, mock_brain_result)

        mock_guard.evaluate_and_guard.assert_called_once()
        assert result.is_stable is True
        assert result.should_restrict is False
        assert result.stability_result is not None
        assert result.stability_result["stability_score"] == 0.85

    @pytest.mark.asyncio
    async def test_stability_restriction(self, mock_slot_info, mock_brain_result):
        """测试稳定性限制触发"""
        mock_guard = MagicMock()
        mock_guard.evaluate_and_guard.return_value = {
            "stability_score": 0.3,
            "is_stable": False,
            "instability_type": "high_volatility",
            "severity": 0.9,
            "should_restrict": True,
            "diagnosis": "因子波动过大",
        }

        router = ResultRouter(stability_guard=mock_guard)
        result = await router.route(mock_slot_info, mock_brain_result)

        assert result.should_restrict is True
        assert result.is_stable is False
        assert result.wq_metrics.get("stability_restricted") is True

    @pytest.mark.asyncio
    async def test_stability_guard_exception(self, mock_slot_info, mock_brain_result):
        """测试 StabilityGuard 异常时的容错"""
        mock_guard = MagicMock()
        mock_guard.evaluate_and_guard.side_effect = RuntimeError("Guard crashed")

        router = ResultRouter(stability_guard=mock_guard)
        result = await router.route(mock_slot_info, mock_brain_result)

        assert result.stability_result is None  # 异常时返回 None
        assert result.is_stable is True  # 默认值
        assert result.error is None  # 不应该影响整体路由

    @pytest.mark.asyncio
    async def test_none_stability_guard(self, base_router, mock_slot_info, mock_brain_result):
        """测试 stability_guard 为 None 时正常工作"""
        result = await base_router.route(mock_slot_info, mock_brain_result)

        assert result.stability_result is None
        assert result.is_stable is True


class TestResultRouterWithAntiOverfit:
    """过拟合检测集成测试"""

    @pytest.mark.asyncio
    async def test_anti_overfit_called(self, mock_slot_info, mock_brain_result):
        """测试 AntiOverfitDetector 被正确调用"""
        mock_detector = MagicMock()
        mock_detector.evaluate.return_value = MagicMock(
            score=72.0,
            recommendation="AGGRESSIVE",
            passed_count=4,
            total_count=5,
        )

        router = ResultRouter(anti_overfit=mock_detector)
        result = await router.route(mock_slot_info, mock_brain_result)

        mock_detector.evaluate.assert_called_once()
        assert result.anti_fit_score == 72.0
        assert result.anti_fit_recommendation == "AGGRESSIVE"

    @pytest.mark.asyncio
    async def test_anti_overfit_exception(self, mock_slot_info, mock_brain_result):
        """测试 AntiOverfitDetector 异常时的容错"""
        mock_detector = MagicMock()
        mock_detector.evaluate.side_effect = ValueError("Detection failed")

        router = ResultRouter(anti_overfit=mock_detector)
        result = await router.route(mock_slot_info, mock_brain_result)

        assert result.anti_fit_score == 50.0  # 默认回退值
        assert result.anti_fit_recommendation == "CAUTIOUS"

    @pytest.mark.asyncio
    async def test_none_anti_overfit(self, base_router, mock_slot_info, mock_brain_result):
        """测试 anti_overfit 为 None 时返回默认值"""
        result = await base_router.route(mock_slot_info, mock_brain_result)

        assert result.anti_fit_score == 50.0
        assert result.anti_fit_recommendation == "CAUTIOUS"


class TestResultRouterWithScorer:
    """官方评分计算集成测试"""

    @pytest.mark.asyncio
    async def test_scorer_called_with_metrics(self, mock_slot_info, mock_brain_result):
        """测试 Scorer 使用正确的 wq_metrics 调用"""
        mock_scorer = MagicMock()
        mock_report = MagicMock()
        mock_report.overall_score = 88.0
        mock_report.grade = "A"
        mock_report.passed = True
        mock_report.factor_profile = "momentum"
        mock_report.improvement_hints = ["increase turnover margin"]
        mock_report.icir_metrics = MagicMock(ic=0.08, ir=1.5, icir=0.12, predicted_fitness=1.2, confidence=0.85)
        mock_report.multi_faceted_reward = MagicMock(
            signal_quality=0.9,
            stability=0.8,
            efficiency=0.7,
            uniqueness=0.85,
            total_reward=0.81,
            is_efficient_alpha=True,
        )
        mock_report.multi_layer_result = {
            "is_high_icir_low_fitness": False,
            "is_efficient_alpha": True,
        }
        mock_scorer.compute_score.return_value = mock_report

        router = ResultRouter(scorer=mock_scorer)
        result = await router.route(mock_slot_info, mock_brain_result)

        mock_scorer.compute_score.assert_called_once()
        call_args = mock_scorer.compute_score.call_args[0][0]
        assert call_args["sharpe"] == 1.35  # 验证传入的 metrics 正确

        assert result.official_score == 88.0
        assert result.official_grade == "A"
        assert result.icir_info is not None
        assert result.mfr_info is not None
        assert result.is_efficient_alpha is True
        assert result.is_high_icir_low_fitness is False

    @pytest.mark.asyncio
    async def test_scorer_exception(self, mock_slot_info, mock_brain_result):
        """测试 Scorer 异常时的容错"""
        mock_scorer = MagicMock()
        mock_scorer.compute_score.side_effect = Exception("Scoring failed")

        router = ResultRouter(scorer=mock_scorer)
        result = await router.route(mock_slot_info, mock_brain_result)

        assert result.score_report is None
        assert result.official_score == 0.0
        assert result.official_grade == "F"

    @pytest.mark.asyncio
    async def test_none_scorer(self, base_router, mock_slot_info, mock_brain_result):
        """测试 scorer 为 None 时返回默认值"""
        result = await base_router.route(mock_slot_info, mock_brain_result)

        assert result.score_report is None
        assert result.official_score == 0.0
        assert result.official_grade == "F"


class TestResultRouterErrorHandling:
    """异常处理和容错测试"""

    @pytest.mark.asyncio
    async def test_brain_result_missing_attributes(self, base_router):
        """测试 brain_result 缺少属性时的容错"""
        slot = MagicMock()
        slot.slot_id = 1
        slot.task_name = "task"

        class MinimalResult:
            pass

        result_obj = MinimalResult()  # 没有任何预定义属性

        routed = await base_router.route(slot, result_obj)

        assert routed.raw_sharpe == 0.0  # getattr 默认值
        assert routed.passed is False
        assert routed.error is None  # 不应该崩溃

    @pytest.mark.asyncio
    async def test_slot_info_minimal(self, base_router):
        """测试最小化 slot_info 的容错"""

        class MinimalSlot:
            expression = "min_expr"

        slot = MinimalSlot()
        result_mock = MagicMock(sharpe=0.5, passed=False)

        routed = await base_router.route(slot, result_mock)

        assert routed.expression == "min_expr"
        assert "slot_" in routed.task_id or routed.task_id == ""


class TestResultRouterNeutralizerRecording:
    """AdaptiveNeutralizer 记录测试"""

    def test_record_neutralizer_outcome_success(self, base_router):
        """测试成功因子的中性化记录"""
        mock_neutralizer = MagicMock()
        router = ResultRouter(adaptive_neutralizer=mock_neutralizer)

        parsed = ParsedWQResult(
            task_id="t1",
            expression="rank(volume)",
            raw_sharpe=1.4,
            passed=True,
            wq_metrics={"fitness": 1.15, "turnover": 0.16},
        )

        router.record_neutralizer_outcome(
            parsed,
            category="momentum",
            level="standard",
        )

        mock_neutralizer.record_outcome.assert_called_once()
        call_kwargs = mock_neutralizer.record_outcome.call_args[1]
        assert call_kwargs["expression"] == "rank(volume)"
        assert call_kwargs["category"] == "momentum"
        assert call_kwargs["level"] == "standard"
        assert call_kwargs["result_metrics"]["outcome"] == "success"

    def test_record_neutralizer_outcome_failure(self, base_router):
        """测试失败因子的中性化记录"""
        mock_neutralizer = MagicMock()
        router = ResultRouter(adaptive_neutralizer=mock_neutralizer)

        parsed = ParsedWQResult(
            task_id="t2",
            expression="ts_mean(close, 20)",
            raw_sharpe=0.3,
            passed=False,
            wq_metrics={"fitness": 0.4, "turnover": 0.25},
        )

        router.record_neutralizer_outcome(parsed, category="reversal")

        call_kwargs = mock_neutralizer.record_outcome.call_args[1]
        assert call_kwargs["result_metrics"]["outcome"] == "failure"

    def test_record_neutralizer_none(self, base_router):
        """测试 adaptive_neutralizer 为 None 时不崩溃"""
        parsed = ParsedWQResult(task_id="t3", expression="test")

        def _should_not_raise():
            base_router.record_neutralizer_outcome(parsed)
        _should_not_raise()  # 不应该抛出异常

    def test_record_neutralizer_exception(self, base_router):
        """测试 neutralizer 异常时的容错"""
        mock_neutralizer = MagicMock()
        mock_neutralizer.record_outcome.side_effect = RuntimeError("Neutralizer error")

        router = ResultRouter(adaptive_neutralizer=mock_neutralizer)
        parsed = ParsedWQResult(task_id="t4", expression="test")

        def _should_not_raise():
            router.record_neutralizer_outcome(parsed)
        _should_not_raise()  # 不应该抛出异常


class TestResultRouterCycleNum:
    """循环编号更新测试"""

    def test_update_cycle_num(self, base_router):
        """测试 cycle_num 更新"""
        assert base_router._cycle_num == 0

        base_router.update_cycle_num(5)
        assert base_router._cycle_num == 5

        base_router.update_cycle_num(10)
        assert base_router._cycle_num == 10

    @pytest.mark.asyncio
    async def test_cycle_num_passed_to_stability(self, mock_slot_info, mock_brain_result):
        """测试 cycle_num 传递给 StabilityGuard"""
        mock_guard = MagicMock()
        mock_guard.evaluate_and_guard.return_value = {
            "stability_score": 0.9,
            "is_stable": True,
            "should_restrict": False,
        }

        router = ResultRouter(stability_guard=mock_guard, cycle_num=7)
        _result = await router.route(mock_slot_info, mock_brain_result)

        call_kwargs = mock_guard.evaluate_and_guard.call_args[1]
        assert call_kwargs["cycle"] == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
