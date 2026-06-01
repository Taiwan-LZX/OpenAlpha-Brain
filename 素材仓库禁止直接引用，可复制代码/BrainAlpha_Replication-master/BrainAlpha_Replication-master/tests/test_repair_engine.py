import unittest
from typing import Any, Dict, List, Optional

from alpha_agent.config import ModelConfig
from alpha_agent.llm_client import LLMClient
from alpha_agent.repair_engine import (
    ConvergenceGovernance,
    RepairEngine,
    RepairResult,
    wrap_expression,
)
from alpha_agent.failure_taxonomy import FailureMode, classify_failures
from alpha_agent.sao import SignalAlphaObject


class TestWrapExpression(unittest.TestCase):
    def test_direct_wrapper(self) -> None:
        result = wrap_expression("rank(close)", "group_neutralize(x, sector)")
        self.assertEqual(result, "group_neutralize(rank(close), sector)")

    def test_negate(self) -> None:
        result = wrap_expression("rank(close)", "-1 * (expr)")
        self.assertEqual(result, "-1 * (rank(close))")

    def test_increase_lookback_doubles_window(self) -> None:
        result = wrap_expression("ts_mean(close, 5)", "increase lookback ×2")
        self.assertEqual(result, "ts_mean(close, 10)")

    def test_zscore_wrapper(self) -> None:
        result = wrap_expression("rank(close)", "zscore(x)")
        self.assertEqual(result, "zscore(rank(close))")

    def test_simple_wrapper_with_parens(self) -> None:
        result = wrap_expression("ts_mean(close, 5)", "rank(x)")
        self.assertEqual(result, "rank(ts_mean(close, 5))")


class TestConvergenceGovernance(unittest.TestCase):
    def setUp(self) -> None:
        self.gov = ConvergenceGovernance(
            max_iterations=5, sharpe_threshold=2.0, delta_threshold=0.05,
        )

    def test_max_iterations(self) -> None:
        converged, reason = self.gov.check_convergence(5, [0.5, 0.6])
        self.assertTrue(converged)
        self.assertIn("Max iterations", reason or "")

    def test_sharpe_threshold_met(self) -> None:
        converged, reason = self.gov.check_convergence(1, [0.5, 2.5])
        self.assertTrue(converged)
        self.assertIn("threshold", reason or "")

    def test_not_converged(self) -> None:
        converged, reason = self.gov.check_convergence(1, [0.5, 0.6])
        self.assertFalse(converged)
        self.assertIsNone(reason)

    def test_delta_convergence(self) -> None:
        converged, reason = self.gov.check_convergence(3, [0.5, 0.51, 0.52])
        self.assertTrue(converged)
        self.assertIn("Converged", reason or "")

    def test_no_sharpe_history(self) -> None:
        converged, reason = self.gov.check_convergence(0, [])
        self.assertFalse(converged)
        self.assertIsNone(reason)


class TestRepairEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = RepairEngine()

    def test_repair_low_sharpe(self) -> None:
        sao = SignalAlphaObject(
            expression="rank(close)",
            sharpe=0.3,
        )
        result = self.engine.execute(sao, ["LOW_SHARPE"])
        self.assertIsInstance(result, RepairResult)
        self.assertIn("expression", result.to_dict())
        self.assertEqual(result.strategy_name, "neutralize_sector")

    def test_repair_correlation_blocked(self) -> None:
        sao = SignalAlphaObject(
            expression="rank(close)",
            sharpe=0.8,
        )
        result = self.engine.execute(sao, ["SELF_CORRELATION"])
        self.assertEqual(result.strategy_name, "neutralize_sector")
        self.assertIn("group_neutralize", result.expression)

    def test_repair_high_turnover(self) -> None:
        sao = SignalAlphaObject(
            expression="rank(close)",
            sharpe=0.5,
        )
        result = self.engine.execute(sao, ["HIGH_TURNOVER"])
        self.assertEqual(result.strategy_name, "decay_linear_5")

    def test_repair_multiple_failures_uses_primary(self) -> None:
        sao = SignalAlphaObject(
            expression="rank(close)",
            sharpe=0.2,
        )
        result = self.engine.execute(sao, ["LOW_SHARPE", "HIGH_TURNOVER"])
        fms = classify_failures(["LOW_SHARPE", "HIGH_TURNOVER"])
        primary = fms[0]
        self.assertEqual(primary, FailureMode.FM1_LOW_SIGNAL_QUALITY)

    def test_no_untried_strategies(self) -> None:
        sao = SignalAlphaObject(
            expression="rank(close)",
            sharpe=0.3,
            tried_strategies=["neutralize_sector", "zscore_normalize",
                              "winsorize_outliers", "extend_lookback_2x",
                              "neutralize_subindustry"],
        )
        gov = ConvergenceGovernance(max_iterations=10, sharpe_threshold=2.0)
        engine = RepairEngine(governance=gov)
        result = engine.execute(sao, ["LOW_SHARPE"])
        self.assertEqual(result.strategy_name, "none_available")

    def test_converged_immediately(self) -> None:
        gov = ConvergenceGovernance(max_iterations=5, sharpe_threshold=0.1)
        engine = RepairEngine(governance=gov)
        sao = SignalAlphaObject(
            expression="rank(close)",
            sharpe=5.0,
            sharpe_history=[5.0],
        )
        result = engine.execute(sao, ["LOW_SHARPE"])
        self.assertTrue(result.converged)
        self.assertIn("threshold", result.stop_reason or "")

    def test_compute_repair_effectiveness_improved(self) -> None:
        sao = SignalAlphaObject(
            expression="rank(close)",
            sharpe=0.5,
            sharpe_history=[0.3, 0.5, 0.8],
        )
        eff = self.engine.compute_repair_effectiveness(sao)
        self.assertEqual(eff["direction"], "improved")
        self.assertAlmostEqual(eff["delta"], 0.5, places=4)

    def test_compute_repair_effectiveness_flat(self) -> None:
        sao = SignalAlphaObject(
            expression="rank(close)",
            sharpe=0.5,
            sharpe_history=[0.5],
        )
        eff = self.engine.compute_repair_effectiveness(sao)
        self.assertEqual(eff["direction"], "flat")

    def test_phase_assignment(self) -> None:
        self.assertEqual(self.engine._phase_for_iteration(0), 1)
        self.assertEqual(self.engine._phase_for_iteration(1), 1)
        self.assertEqual(self.engine._phase_for_iteration(2), 2)
        self.assertEqual(self.engine._phase_for_iteration(3), 2)
        self.assertEqual(self.engine._phase_for_iteration(4), 3)

    def test_repair_result_to_dict(self) -> None:
        result = RepairResult(
            expression="rank(close)",
            strategy_name="test",
            strategy_index=0,
            phase=1,
            converged=False,
        )
        d = result.to_dict()
        self.assertIn("expression", d)
        self.assertIn("strategy_name", d)
        self.assertIn("phase", d)
        self.assertIn("converged", d)


class TestApplyMetaInstruction(unittest.TestCase):
    def test_extend_lookback_doubles_window(self) -> None:
        expr = "ts_mean(close, 5)"
        result = RepairEngine.apply_meta_instruction(expr, "extend_lookback_2x")
        self.assertEqual(result, "ts_mean(close, 10)")

    def test_extend_lookback_multiple_windows(self) -> None:
        expr = "ts_mean(ts_delta(close, 3), 5)"
        result = RepairEngine.apply_meta_instruction(expr, "extend_lookback_2x")
        self.assertEqual(result, "ts_mean(ts_delta(close, 6), 10)")

    def test_reduce_lookback_halves_window(self) -> None:
        expr = "ts_mean(close, 10)"
        result = RepairEngine.apply_meta_instruction(expr, "reduce_lookback_half")
        self.assertEqual(result, "ts_mean(close, 5)")

    def test_reduce_lookback_min_floor(self) -> None:
        expr = "ts_mean(close, 3)"
        result = RepairEngine.apply_meta_instruction(expr, "reduce_lookback_half")
        self.assertEqual(result, "ts_mean(close, 2)")

    def test_no_ts_operator_returns_unchanged(self) -> None:
        expr = "rank(close)"
        result = RepairEngine.apply_meta_instruction(expr, "extend_lookback_2x")
        self.assertEqual(result, "rank(close)")

    def test_unknown_instruction_returns_unchanged(self) -> None:
        expr = "ts_mean(close, 5)"
        result = RepairEngine.apply_meta_instruction(expr, "invert_sign")
        self.assertEqual(result, "ts_mean(close, 5)")

    def test_increase_prefix_doubles_window(self) -> None:
        expr = "ts_mean(close, 5)"
        result = RepairEngine.apply_meta_instruction(expr, "increase lookback ×2")
        self.assertEqual(result, "ts_mean(close, 10)")

    def test_reduce_prefix_halves_window(self) -> None:
        expr = "ts_mean(close, 5)"
        result = RepairEngine.apply_meta_instruction(expr, "reduce lookback ×0.5")
        self.assertEqual(result, "ts_mean(close, 2)")

    def test_zero_value_not_modified(self) -> None:
        expr = "if_else(adv20 > ts_mean(adv20, 60), rank(close), 0)"
        result = RepairEngine.apply_meta_instruction(expr, "reduce_lookback_half")
        self.assertEqual(result, "if_else(adv20 > ts_mean(adv20, 30), rank(close), 0)")

    def test_zero_value_not_modified_extend(self) -> None:
        expr = "if_else(condition, expr, 0)"
        result = RepairEngine.apply_meta_instruction(expr, "extend_lookback_2x")
        self.assertEqual(result, "if_else(condition, expr, 0)")


class MockLLMClient(LLMClient):
    def __init__(self, response: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(ModelConfig(provider="test"))
        self._response = response if response is not None else {"expression": "rank(ts_mean(close, 60))", "reasoning": "Extended lookback"}

    def request_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._response


class TestRepairEngineWithLLM(unittest.TestCase):
    def setUp(self) -> None:
        self.gov = ConvergenceGovernance(max_iterations=5, sharpe_threshold=2.0)

    def test_llm_rewrite_returns_expression(self) -> None:
        mock = MockLLMClient({"expression": "rank(ts_mean(close, 60))", "reasoning": "Extended lookback"})
        engine = RepairEngine(governance=self.gov, llm_client=mock)
        sao = SignalAlphaObject(
            expression="rank(ts_mean(close, 20))",
            sharpe=0.3,
            failed_checks=["LOW_SHARPE"],
        )
        result = engine.execute(sao, ["LOW_SHARPE"])
        self.assertEqual(result.expression, "rank(ts_mean(close, 60))")
        self.assertEqual(result.strategy_name, "neutralize_sector")

    def test_llm_fallback_to_wrap_on_failure(self) -> None:
        class FailingMock(MockLLMClient):
            def request_json(self, **kwargs: Any) -> Dict[str, Any]:
                raise RuntimeError("LLM unavailable")

        mock = FailingMock()
        engine = RepairEngine(governance=self.gov, llm_client=mock)
        sao = SignalAlphaObject(
            expression="rank(close)",
            sharpe=0.3,
            failed_checks=["LOW_SHARPE"],
        )
        result = engine.execute(sao, ["LOW_SHARPE"])
        self.assertIn("group_neutralize", result.expression)

    def test_no_llm_client_uses_wrap(self) -> None:
        engine = RepairEngine(governance=self.gov)
        sao = SignalAlphaObject(
            expression="rank(close)",
            sharpe=0.3,
            failed_checks=["LOW_SHARPE"],
        )
        result = engine.execute(sao, ["LOW_SHARPE"])
        self.assertIn("group_neutralize", result.expression)

    def test_llm_rewrite_empty_expression_falls_back(self) -> None:
        mock = MockLLMClient({})
        engine = RepairEngine(governance=self.gov, llm_client=mock)
        sao = SignalAlphaObject(
            expression="rank(close)",
            sharpe=0.3,
            failed_checks=["LOW_SHARPE"],
        )
        result = engine.execute(sao, ["LOW_SHARPE"])
        self.assertIn("group_neutralize", result.expression)

    def test_llm_rewrite_missing_expression_key_falls_back(self) -> None:
        mock = MockLLMClient({"reasoning": "some reasoning"})
        engine = RepairEngine(governance=self.gov, llm_client=mock)
        sao = SignalAlphaObject(
            expression="rank(close)",
            sharpe=0.3,
            failed_checks=["LOW_SHARPE"],
        )
        result = engine.execute(sao, ["LOW_SHARPE"])
        self.assertIn("group_neutralize", result.expression)


if __name__ == "__main__":
    unittest.main()
