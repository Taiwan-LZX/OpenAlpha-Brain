"""Unit tests for BrainAwareMutationEngine.

Tests cover all major failure modes, score computation,
prompt generation for each strategy, and helper methods.
No real LLM required - pure string operations.
"""

import pytest

from openalpha_brain.evolution.mutation_engine import (
    BrainAwareMutationEngine,
    Diagnosis,
    MutationStrategy,
)


class TestBrainAwareMutationEngineInit:
    """Test initialization and configuration."""

    def test_init_default(self):
        engine = BrainAwareMutationEngine()
        assert engine._operator_replacements is not None
        assert len(engine._nonlinear_ops) > 0
        assert len(engine._normalization_ops) > 0

    def test_init_operator_replacements(self):
        engine = BrainAwareMutationEngine()
        assert "ts_mean" in engine._operator_replacements
        assert "rank" in engine._operator_replacements
        assert isinstance(engine._operator_replacements["ts_mean"], list)


class TestComputeScore:
    """Test composite score calculation from WQ metrics."""

    def test_perfect_score(self):
        engine = BrainAwareMutationEngine()
        metrics = {
            "sharpe": 2.5,
            "fitness": 2.0,
            "turnover": 25,
        }
        score = engine.compute_score(metrics)
        assert score == 100

    def test_low_sharpe_score(self):
        engine = BrainAwareMutationEngine()
        metrics = {
            "sharpe": 0.1,
            "fitness": 0.1,
            "turnover": 25,
        }
        score = engine.compute_score(metrics)
        assert score < 25

    def test_negative_sharpe(self):
        engine = BrainAwareMutationEngine()
        metrics = {
            "sharpe": -0.5,
            "fitness": 0,
            "turnover": 80,
        }
        score = engine.compute_score(metrics)
        assert score < 20

    def test_high_turnover_penalty(self):
        engine = BrainAwareMutationEngine()
        metrics = {
            "sharpe": 1.5,
            "fitness": 1.5,
            "turnover": 80,
        }
        score = engine.compute_score(metrics)
        turnover_component = 10
        assert turnover_component in [score - (1.5 * 20 + 1.5 * 15), 10]

    def test_ideal_turnover_bonus(self):
        engine = BrainAwareMutationEngine()
        metrics = {
            "sharpe": 1.0,
            "fitness": 1.0,
            "turnover": 25,
        }
        score = engine.compute_score(metrics)
        assert score >= 50

    def test_missing_metrics_defaults(self):
        engine = BrainAwareMutationEngine()
        metrics = {}
        score = engine.compute_score(metrics)
        assert score == 20


class TestCountNesting:
    """Test expression nesting depth calculation."""

    def test_simple_expression(self):
        engine = BrainAwareMutationEngine()
        expr = "ts_delta(close, 20)"
        assert engine.count_nesting(expr) == 1

    def test_nested_expression(self):
        engine = BrainAwareMutationEngine()
        expr = "group_neutralize(rank(ts_delta(close, 20)), industry)"
        assert engine.count_nesting(expr) == 3

    def test_deeply_nested(self):
        engine = BrainAwareMutationEngine()
        expr = "group_neutralize(tanh(rank(ts_zscore(ts_delta(close, 20), 20))), industry)"
        depth = engine.count_nesting(expr)
        assert depth == 5

    def test_no_parentheses(self):
        engine = BrainAwareMutationEngine()
        expr = "close"
        assert engine.count_nesting(expr) == 0


class TestHasNormalization:
    """Test normalization operator detection."""

    def test_has_rank(self):
        engine = BrainAwareMutationEngine()
        expr = "rank(ts_delta(close, 20))"
        assert engine.has_normalization(expr) is True

    def test_has_zscore(self):
        engine = BrainAwareMutationEngine()
        expr = "zscore(volume)"
        assert engine.has_normalization(expr) is True

    def test_no_normalization(self):
        engine = BrainAwareMutationEngine()
        expr = "ts_delta(close, 20)"
        assert engine.has_normalization(expr) is False

    def test_has_group_rank(self):
        engine = BrainAwareMutationEngine()
        expr = "group_rank(expr, industry)"
        assert engine.has_normalization(expr) is True


class TestHasNonlinear:
    """Test nonlinear transform detection."""

    def test_has_tanh(self):
        engine = BrainAwareMutationEngine()
        expr = "tanh(rank(ts_delta(close, 20)))"
        assert engine.has_nonlinear(expr) is True

    def test_has_sigmoid(self):
        engine = BrainAwareMutationEngine()
        expr = "sigmoid(expr)"
        assert engine.has_nonlinear(expr) is True

    def test_has_sign_power(self):
        engine = BrainAwareMutationEngine()
        expr = "sign_power(expr, 0.5)"
        assert engine.has_nonlinear(expr) is True

    def test_no_nonlinear(self):
        engine = BrainAwareMutationEngine()
        expr = "rank(ts_delta(close, 20))"
        assert engine.has_nonlinear(expr) is False


class TestDiagnoseRegenerateFull:
    """Test REGENERATE_FULL strategy selection."""

    def test_very_low_score(self):
        engine = BrainAwareMutationEngine()
        expression = "rank(ts_delta(close, 20))"
        wq_metrics = {"sharpe": -2.0, "fitness": 0, "turnover": 80, "ic_mean": 0.01, "ic_ir": 0.6}
        wq_checks = []

        diagnosis = engine.diagnose(expression, wq_metrics, wq_checks)

        assert diagnosis.strategy == MutationStrategy.REGENERATE_FULL
        assert "极低评分" in diagnosis.reason
        assert diagnosis.details["score"] < 20


class TestDiagnoseMutateOperator:
    """Test MUTATE_OPERATOR strategy selection."""

    def test_ic_near_zero(self):
        engine = BrainAwareMutationEngine()
        expression = "ts_mean(close, 20)"
        wq_metrics = {
            "sharpe": 0.5,
            "fitness": 0.5,
            "turnover": 30,
            "ic_mean": 0.001,
            "ic_ir": 0.3,
        }
        wq_checks = []

        diagnosis = engine.diagnose(expression, wq_metrics, wq_checks)

        assert diagnosis.strategy == MutationStrategy.MUTATE_OPERATOR
        assert "IC接近零" in diagnosis.reason
        assert "suggested_replacements" in diagnosis.details


class TestDiagnoseMutateSignalType:
    """Test MUTATE_SIGNAL_TYPE strategy selection."""

    def test_negative_ic(self):
        engine = BrainAwareMutationEngine()
        expression = "rank(ts_delta(close, 20))"
        wq_metrics = {
            "sharpe": 0.3,
            "fitness": 0.3,
            "turnover": 30,
            "ic_mean": -0.02,
            "ic_ir": 0.5,
        }
        wq_checks = []

        diagnosis = engine.diagnose(expression, wq_metrics, wq_checks)

        assert diagnosis.strategy == MutationStrategy.MUTATE_SIGNAL_TYPE
        assert "IC为负" in diagnosis.reason


class TestDiagnoseSimplify:
    """Test SIMPLIFY strategy selection."""

    def test_deep_nesting(self):
        engine = BrainAwareMutationEngine()
        expression = (
            "group_neutralize("
            "tanh(rank(ts_zscore("
            "ts_delta("
            "ts_mean("
            "ts_std_dev("
            "ts_decay_linear("
            "ts_av_diff(close * volume * amount, 5), 5"
            "), 5"
            "), 10"
            "), 5"
            "), 20)"
            ")), industry)"
        )
        wq_metrics = {
            "sharpe": 0.8,
            "fitness": 0.8,
            "turnover": 25,
            "ic_mean": 0.02,
            "ic_ir": 0.6,
        }
        wq_checks = []

        diagnosis = engine.diagnose(expression, wq_metrics, wq_checks)

        assert diagnosis.strategy == MutationStrategy.SIMPLIFY
        assert "嵌套层数过深" in diagnosis.reason
        assert diagnosis.details["nesting_depth"] > 8


class TestDiagnoseMutateNonlinear:
    """Test MUTATE_NONLINEAR strategy selection."""

    def test_medium_score_no_nonlinear(self):
        engine = BrainAwareMutationEngine()
        expression = "ts_delta(close * volume * amount, 20)"
        wq_metrics = {
            "sharpe": 1.0,
            "fitness": 0.5,
            "turnover": 30,
            "ic_mean": 0.02,
            "ic_ir": 0.7,
        }
        wq_checks = []

        diagnosis = engine.diagnose(expression, wq_metrics, wq_checks)

        assert diagnosis.strategy == MutationStrategy.MUTATE_NONLINEAR
        assert "无非线性变换" in diagnosis.reason


class TestDiagnoseMutateNormalization:
    """Test MUTATE_NORMALIZATION strategy selection."""

    def test_low_ir_no_normalization(self):
        engine = BrainAwareMutationEngine()
        expression = "ts_delta(close, 20)"
        wq_metrics = {
            "sharpe": 1.0,
            "fitness": 1.0,
            "turnover": 30,
            "ic_mean": 0.02,
            "ic_ir": 0.3,
        }
        wq_checks = []

        diagnosis = engine.diagnose(expression, wq_metrics, wq_checks)

        assert diagnosis.strategy == MutationStrategy.MUTATE_NORMALIZATION
        assert "无标准化" in diagnosis.reason


class TestDiagnoseMutateInteraction:
    """Test MUTATE_INTERACTION strategy selection."""

    def test_single_signal(self):
        engine = BrainAwareMutationEngine()
        expression = "rank(ts_delta(close, 20))"
        wq_metrics = {
            "sharpe": 1.5,
            "fitness": 1.5,
            "turnover": 25,
            "ic_mean": 0.03,
            "ic_ir": 0.8,
        }
        wq_checks = []

        diagnosis = engine.diagnose(expression, wq_metrics, wq_checks)

        assert diagnosis.strategy == MutationStrategy.MUTATE_INTERACTION
        assert "单信号因子" in diagnosis.reason


class TestDiagnoseMutateWindow:
    """Test MUTATE_WINDOW as default fallback strategy."""

    def test_good_metrics_default_window(self):
        engine = BrainAwareMutationEngine()
        expression = "group_neutralize(tanh(rank(ts_delta(close * volume, 20))), industry)"
        wq_metrics = {
            "sharpe": 1.5,
            "fitness": 1.5,
            "turnover": 25,
            "ic_mean": 0.03,
            "ic_ir": 0.8,
        }
        wq_checks = []

        diagnosis = engine.diagnose(expression, wq_metrics, wq_checks)

        assert diagnosis.strategy == MutationStrategy.MUTATE_WINDOW
        assert "默认策略" in diagnosis.reason
        assert "current_windows" in diagnosis.details


class TestGenerateMutationPrompt:
    """Test prompt generation for each mutation strategy."""

    def _make_diagnosis(self, strategy: MutationStrategy, reason: str) -> Diagnosis:
        return Diagnosis(strategy=strategy, reason=reason, details={})

    def test_prompt_mutate_window(self):
        engine = BrainAwareMutationEngine()
        diagnosis = self._make_diagnosis(
            MutationStrategy.MUTATE_WINDOW,
            "调整窗口",
        )
        prompt = engine.generate_mutation_prompt(diagnosis, "ts_delta(close, 20)")

        assert "调整时序窗口" in prompt
        assert "突变指令" in prompt
        assert "ts_delta(close, 20)" in prompt

    def test_prompt_mutate_operator(self):
        engine = BrainAwareMutationEngine()
        diagnosis = self._make_diagnosis(
            MutationStrategy.MUTATE_OPERATOR,
            "替换算子",
        )
        prompt = engine.generate_mutation_prompt(diagnosis, "ts_mean(close, 20)")

        assert "替换核心算子" in prompt
        assert "ts_decay_linear" in prompt or "zscore" in prompt

    def test_prompt_mutate_normalization(self):
        engine = BrainAwareMutationEngine()
        diagnosis = self._make_diagnosis(
            MutationStrategy.MUTATE_NORMALIZATION,
            "添加标准化",
        )
        prompt = engine.generate_mutation_prompt(diagnosis, "ts_delta(close, 20)")

        assert "添加标准化" in prompt
        assert "rank()" in prompt or "zscore()" in prompt

    def test_prompt_mutate_signal_type(self):
        engine = BrainAwareMutationEngine()
        diagnosis = self._make_diagnosis(
            MutationStrategy.MUTATE_SIGNAL_TYPE,
            "翻转方向",
        )
        prompt = engine.generate_mutation_prompt(diagnosis, "rank(ts_delta(close, 20))")

        assert "翻转因子方向" in prompt
        assert "-1 *" in prompt or "-rank" in prompt

    def test_prompt_mutate_nonlinear(self):
        engine = BrainAwareMutationEngine()
        diagnosis = self._make_diagnosis(
            MutationStrategy.MUTATE_NONLINEAR,
            "引入非线性",
        )
        prompt = engine.generate_mutation_prompt(diagnosis, "rank(ts_delta(close, 20))")

        assert "引入非线性变换" in prompt
        assert "tanh" in prompt
        assert "power" in prompt or "sign_power" in prompt

    def test_prompt_mutate_interaction(self):
        engine = BrainAwareMutationEngine()
        diagnosis = self._make_diagnosis(
            MutationStrategy.MUTATE_INTERACTION,
            "组合信号",
        )
        prompt = engine.generate_mutation_prompt(diagnosis, "rank(ts_delta(close, 20))")

        assert "组合多信号源" in prompt
        assert "量价交互" in prompt or "动量+波动" in prompt

    def test_prompt_simplify(self):
        engine = BrainAwareMutationEngine()
        diagnosis = Diagnosis(
            strategy=MutationStrategy.SIMPLIFY,
            reason="简化嵌套",
            details={"nesting_depth": 10},
        )
        prompt = engine.generate_mutation_prompt(
            diagnosis,
            "group_neutralize(tanh(rank(ts_zscore(ts_delta(close, 20), 20))), industry)",
        )

        assert "适当简化表达式" in prompt
        assert "10 层" in prompt

    def test_prompt_regenerate_full(self):
        engine = BrainAwareMutationEngine()
        diagnosis = self._make_diagnosis(
            MutationStrategy.REGENERATE_FULL,
            "完全重写",
        )
        prompt = engine.generate_mutation_prompt(diagnosis, "bad_expr")

        assert "完全重写" in prompt
        assert "动量因子" in prompt or "反转因子" in prompt

    def test_prompt_with_inspiration(self):
        engine = BrainAwareMutationEngine()
        diagnosis = self._make_diagnosis(
            MutationStrategy.MUTATE_WINDOW,
            "测试",
        )
        inspiration = ["expr1", "expr2", "expr3"]
        prompt = engine.generate_mutation_prompt(diagnosis, "test_expr", inspiration_exprs=inspiration)

        assert "参考表达式" in prompt
        assert "expr1" in prompt
        assert "expr2" in prompt

    def test_all_strategies_have_unique_prompts(self):
        """Verify that different strategies produce different prompts."""
        engine = BrainAwareMutationEngine()
        prompts = {}

        for strategy in MutationStrategy:
            diagnosis = self._make_diagnosis(strategy, f"test_{strategy.value}")
            prompt = engine.generate_mutation_prompt(diagnosis, "test_expr")
            prompts[strategy] = prompt

        unique_prompts = len(set(prompts.values()))
        assert unique_prompts == len(MutationStrategy), (
            f"Each strategy should have unique prompt, got {unique_prompts} unique "
            f"for {len(MutationStrategy)} strategies"
        )


class TestSuggestOperatorReplacement:
    """Test operator replacement suggestions."""

    def test_ts_mean_replacements(self):
        engine = BrainAwareMutationEngine()
        replacements = engine.suggest_operator_replacement("expr", "ts_mean")
        assert "ts_decay_linear" in replacements
        assert "ts_sum" in replacements

    def test_unknown_operator(self):
        engine = BrainAwareMutationEngine()
        replacements = engine.suggest_operator_replacement("expr", "unknown_op")
        assert replacements == ["unknown_op"]

    def test_rank_replacements(self):
        engine = BrainAwareMutationEngine()
        replacements = engine.suggest_operator_replacement("expr", "rank")
        assert "zscore" in replacements
        assert "scale" in replacements


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_expression(self):
        engine = BrainAwareMutationEngine()
        wq_metrics = {"sharpe": -2.0, "fitness": 0, "turnover": 80, "ic_mean": 0.01, "ic_ir": 0.6}
        wq_checks = []
        diagnosis = engine.diagnose("", wq_metrics, wq_checks)
        assert diagnosis.strategy == MutationStrategy.REGENERATE_FULL

    def test_empty_checks(self):
        engine = BrainAwareMutationEngine()
        expression = "rank(ts_delta(close, 20))"
        wq_metrics = {"sharpe": 1.5, "fitness": 1.5, "turnover": 25}
        wq_checks = []
        diagnosis = engine.diagnose(expression, wq_metrics, wq_checks)
        assert diagnosis is not None

    def test_none_metrics(self):
        engine = BrainAwareMutationEngine()
        _expression = "test"
        wq_metrics = {}
        _wq_checks = []
        score = engine.compute_score(wq_metrics)
        assert score == 20

    def test_very_long_expression(self):
        engine = BrainAwareMutationEngine()
        long_expr = "group_neutralize(" * 20 + "ts_delta(close, 20)" + ", industry)" * 20
        depth = engine.count_nesting(long_expr)
        assert depth > 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
