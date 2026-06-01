import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openalpha_brain.core.feedback_orchestrator import (
    FeedbackLoopOrchestrator,
    ImprovementChain,
)
from openalpha_brain.learning.reflection_engine import (
    CritiqueResult,
    ReflectionEngine,
    ReflectionResult,
)
from openalpha_brain.services.brain_submitter import ReflexionEngine


class TestReflectionEngineInitialization:
    """Test ReflectionEngine initialization in __init__."""

    def test_reflection_engine_initialized_successfully(self):
        """Test that ReflectionEngine is initialized when import succeeds."""
        with patch.dict("sys.modules", {
            "openalpha_brain.learning.reflection_engine": MagicMock(
                ReflectionEngine=MagicMock(return_value=MagicMock())
            ),
            "openalpha_brain.knowledge.field_proxy_map": MagicMock(),
            "openalpha_brain.evolution.fitness_boost": MagicMock(),
            "openalpha_brain.validation.stability_guard": MagicMock(),
            "openalpha_brain.core.official_scoring_adapter": MagicMock(),
            "openalpha_brain.evolution.turnover_optimizer": MagicMock(),
            "openalpha_brain.generation.template_reasoning_generator": MagicMock(),
            "openalpha_brain.generation.alpha_logics": MagicMock(),
        }):
            import sys
            original_modules = sys.modules.copy()

            mock_re_module = MagicMock()
            mock_re_instance = MagicMock()
            mock_re_module.ReflectionEngine.return_value = mock_re_instance
            sys.modules["openalpha_brain.learning.reflection_engine"] = mock_re_module

            try:
                orchestrator = FeedbackLoopOrchestrator(
                    cookies=MagicMock(),
                    slot_manager=MagicMock(),
                )

                assert hasattr(orchestrator, "_reflection_engine")
            finally:
                sys.modules.clear()
                sys.modules.update(original_modules)

    def test_reflection_engine_init_failure_graceful_degradation(self):
        """Test graceful degradation when ReflectionEngine init fails."""
        import sys

        def failing_import(*args, **kwargs):
            raise Exception("Import error")

        original_modules = sys.modules.copy()
        mock_re_module = MagicMock()
        mock_re_module.ReflectionEngine.side_effect = failing_import
        sys.modules["openalpha_brain.learning.reflection_engine"] = mock_re_module

        with patch.dict("sys.modules", {
            "openalpha_brain.knowledge.field_proxy_map": MagicMock(),
            "openalpha_brain.evolution.fitness_boost": MagicMock(),
            "openalpha_brain.validation.stability_guard": MagicMock(),
            "openalpha_brain.core.official_scoring_adapter": MagicMock(),
            "openalpha_brain.evolution.turnover_optimizer": MagicMock(),
            "openalpha_brain.generation.template_reasoning_generator": MagicMock(),
            "openalpha_brain.generation.alpha_logics": MagicMock(),
        }):
            try:
                orchestrator = FeedbackLoopOrchestrator(
                    cookies=MagicMock(),
                    slot_manager=MagicMock(),
                )

                assert orchestrator._reflection_engine is None
            finally:
                sys.modules.clear()
                sys.modules.update(original_modules)

    def test_reflection_engine_attribute_exists(self):
        """Test _reflection_engine attribute exists after initialization."""
        with patch.dict("sys.modules", {
            "openalpha_brain.learning.reflection_engine": MagicMock(
                ReflectionEngine=MagicMock(return_value=MagicMock())
            ),
            "openalpha_brain.knowledge.field_proxy_map": MagicMock(),
            "openalpha_brain.evolution.fitness_boost": MagicMock(),
            "openalpha_brain.validation.stability_guard": MagicMock(),
            "openalpha_brain.core.official_scoring_adapter": MagicMock(),
            "openalpha_brain.evolution.turnover_optimizer": MagicMock(),
            "openalpha_brain.generation.template_reasoning_generator": MagicMock(),
            "openalpha_brain.generation.alpha_logics": MagicMock(),
        }):
            orchestrator = FeedbackLoopOrchestrator(
                cookies=MagicMock(),
                slot_manager=MagicMock(),
            )
            assert hasattr(orchestrator, "_reflection_engine")


class TestReflectionEngineIntegrationInHandleImprovement:
    """Test ReflectionEngine integration in _handle_improvement method."""

    @pytest.fixture
    def setup_orchestrator_with_mock(self):
        """Create orchestrator with mocked dependencies."""
        orchestrator = FeedbackLoopOrchestrator(
            cookies=MagicMock(),
            slot_manager=MagicMock(),
        )
        orchestrator._reflection_engine = MagicMock(spec=ReflectionEngine)
        orchestrator._reflexion_engine = MagicMock(spec=ReflexionEngine)
        orchestrator._reflexion_engine.reflect_and_improve = AsyncMock(
            return_value=MagicMock(
                passed_proxy=True,
                final_expr="group_neutralize(ts_decay_linear(rank(close), 10), industry)",
                final_verdict=MagicMock(overall_score=0.8),
            )
        )
        orchestrator._graph_db = MagicMock()
        orchestrator.config["llm_improve_fn"] = AsyncMock()
        orchestrator.config["max_improvement_generations"] = 3
        return orchestrator

    @pytest.mark.asyncio
    async def test_reflect_on_failure_called(self, setup_orchestrator_with_mock):
        """Test that reflect_on_failure is called with correct parameters."""
        orch = setup_orchestrator_with_mock
        orch._reflection_engine.reflect_on_failure = AsyncMock(return_value=ReflectionResult(
            failure_stage="parameters",
            failure_reason="Weak signal",
            suggested_fix="Add normalization",
            confidence=0.7,
        ))

        brain_result = MagicMock()
        brain_result.sharpe = 0.9
        brain_result.fitness = 0.8
        brain_result.turnover = 0.2
        brain_result.brain_checks = []

        slot_info = MagicMock()
        expression = "group_neutralize(rank(close), industry)"

        await orch._handle_improvement(slot_info, brain_result, expression, "task_1")

        orch._reflection_engine.reflect_on_failure.assert_called_once()
        call_args = orch._reflection_engine.reflect_on_failure.call_args
        assert call_args[1]["expression"] == expression
        assert "brain_result" in call_args[1]

    @pytest.mark.asyncio
    async def test_get_recent_reflections_called(self, setup_orchestrator_with_mock):
        """Test that get_recent_reflections is called to retrieve history."""
        orch = setup_orchestrator_with_mock
        orch._reflection_engine.reflect_on_failure = AsyncMock(
            side_effect=[
                None,
                ReflectionResult(
                    failure_stage="turnover",
                    failure_reason="High turnover",
                    suggested_fix="Apply decay",
                    confidence=0.8,
                ),
            ]
        )
        orch._reflection_engine.get_recent_reflections = MagicMock(return_value=[
            {"failure_stage": "turnover", "failure_reason": "test"},
        ])
        orch._reflection_engine.get_failure_patterns = MagicMock(return_value={"turnover": 5})
        orch._reflection_engine.self_critique = AsyncMock(return_value=CritiqueResult(
            consistency_score=0.8,
            issues=["issue1"],
            suggestions=["suggestion1"],
            critique_available=True,
        ))
        orch.config["llm_improve_fn"].return_value = "group_neutralize(ts_decay_linear(rank(close), 10), industry)"

        brain_result = MagicMock()
        brain_result.sharpe = 1.0
        brain_result.fitness = 0.7
        brain_result.turnover = 0.4
        brain_result.brain_checks = []

        slot_info = MagicMock()
        expression = "group_neutralize(rank(close), industry)"

        await orch._handle_improvement(slot_info, brain_result, expression, "task_1")

        orch._reflection_engine.get_recent_reflections.assert_called_once_with(n=5)
        orch._reflection_engine.get_failure_patterns.assert_called_once()

    @pytest.mark.asyncio
    async def test_high_icir_low_fitness_specialized_analysis(self, setup_orchestrator_with_mock):
        """Test specialized analysis for HIGH_ICIR_LOW_FITNESS factors."""
        orch = setup_orchestrator_with_mock
        orch._reflection_engine.reflect_on_failure = AsyncMock(
            side_effect=[
                None,
                ReflectionResult(
                    failure_stage="turnover",
                    failure_reason="Good Sharpe but low fitness",
                    suggested_fix="Reduce turnover",
                    confidence=0.85,
                ),
            ]
        )
        orch._reflection_engine.self_critique = AsyncMock(return_value=CritiqueResult(
            consistency_score=0.9,
            issues=["High turnover detected"],
            suggestions=["Apply ts_decay_linear"],
            critique_available=True,
        ))
        orch._reflection_engine.get_recent_reflections = MagicMock(return_value=[])
        orch._reflection_engine.get_failure_patterns = MagicMock(return_value={})
        orch.config["llm_improve_fn"].return_value = "group_neutralize(ts_decay_linear(rank(ts_zscore(close, 20)), 15), industry)"

        brain_result = MagicMock()
        brain_result.sharpe = 1.1
        brain_result.fitness = 0.6
        brain_result.turnover = 0.45
        brain_result.brain_checks = []

        slot_info = MagicMock()
        expression = "group_neutralize(rank(ts_delta(close, 5)), industry)"

        await orch._handle_improvement(slot_info, brain_result, expression, "task_high_icir")

        orch._reflection_engine.self_critique.assert_called_once()
        call_args = orch._reflection_engine.self_critique.call_args
        assert "HIGH_ICIR" in call_args[1]["hypothesis"] or "high-frequency" in call_args[1]["hypothesis"]

    @pytest.mark.asyncio
    async def test_standard_reflection_for_normal_factors(self, setup_orchestrator_with_mock):
        """Test standard reflection path for normal (non-specialized) factors."""
        orch = setup_orchestrator_with_mock
        orch._reflection_engine.reflect_on_failure = AsyncMock(
            side_effect=[
                None,
                ReflectionResult(
                    failure_stage="parameters",
                    failure_reason="Weak Sharpe",
                    suggested_fix="Enhance signal",
                    confidence=0.7,
                ),
            ]
        )
        orch._reflection_engine.self_critique = AsyncMock(return_value=CritiqueResult(
            consistency_score=0.75,
            issues=[],
            suggestions=[],
            critique_available=True,
        ))
        orch._reflection_engine.get_recent_reflections = MagicMock(return_value=[])
        orch._reflection_engine.get_failure_patterns = MagicMock(return_value={})
        orch.config["llm_improve_fn"].return_value = "group_neutralize(tanh(rank(ts_zscore(close, 20))), industry)"

        brain_result = MagicMock()
        brain_result.sharpe = 0.9
        brain_result.fitness = 0.8
        brain_result.turnover = 0.25
        brain_result.brain_checks = []

        slot_info = MagicMock()
        expression = "group_neutralize(rank(close), industry)"

        await orch._handle_improvement(slot_info, brain_result, expression, "task_normal")

        orch._reflection_engine.self_critique.assert_called_once()

    @pytest.mark.asyncio
    async def test_reflection_skipped_when_confidence_low(self, setup_orchestrator_with_mock):
        """Test that reflection-enhanced LLM is skipped when confidence < 0.6."""
        orch = setup_orchestrator_with_mock
        orch._reflection_engine.reflect_on_failure = AsyncMock(return_value=ReflectionResult(
            failure_stage="unknown",
            failure_reason="Unknown issue",
            suggested_fix="Try different approach",
            confidence=0.3,
        ))

        brain_result = MagicMock()
        brain_result.sharpe = 0.5
        brain_result.fitness = 0.5
        brain_result.turnover = 0.3
        brain_result.brain_checks = []

        slot_info = MagicMock()
        expression = "group_neutralize(rank(close), industry)"

        with patch.object(orch, "_improve_and_resubmit", new_callable=AsyncMock) as mock_improve:
            mock_improve.return_value = "improved_expr"
            await orch._handle_improvement(slot_info, brain_result, expression, "task_low_conf")

            mock_improve.assert_called_once()
            orch._reflection_engine.self_critique.assert_not_called()


class TestBuildReflectionContext:
    """Test _build_reflection_context method."""

    @pytest.fixture
    def setup_orchestrator(self):
        orchestrator = FeedbackLoopOrchestrator(
            cookies=MagicMock(),
            slot_manager=MagicMock(),
        )
        return orchestrator

    def test_build_context_with_basic_data(self, setup_orchestrator):
        """Test context building with basic reflection data."""
        orch = setup_orchestrator
        reflection_result = ReflectionResult(
            failure_stage="turnover",
            failure_reason="High turnover rate",
            suggested_fix="Apply ts_decay_linear",
            confidence=0.8,
            avoid_patterns=["high frequency signals"],
        )
        critique_result = CritiqueResult(
            consistency_score=0.85,
            issues=["Turnover too high"],
            suggestions=["Add smoothing"],
            critique_available=True,
        )

        context = orch._build_reflection_context(
            reflection_result=reflection_result,
            critique_result=critique_result,
            recent_reflections=[],
            failure_patterns={"turnover": 3},
            is_specialized=False,
        )

        assert "turnover" in context.lower()
        assert "0.8" in context or "0.80" in context
        assert "ts_decay_linear" in context
        assert "HIGH_ICIR" not in context

    def test_build_context_specialized_mode(self, setup_orchestrator):
        """Test context building in specialized HIGH_ICIR_LOW_FITNESS mode."""
        orch = setup_orchestrator
        reflection_result = ReflectionResult(
            failure_stage="turnover",
            failure_reason="Good signal but high cost",
            suggested_fix="Reduce frequency",
            confidence=0.9,
        )
        critique_result = CritiqueResult(critique_available=True)

        context = orch._build_reflection_context(
            reflection_result=reflection_result,
            critique_result=critique_result,
            recent_reflections=[],
            failure_patterns={},
            is_specialized=True,
        )

        assert "HIGH_ICIR" in context
        assert "专门诊断" in context

    def test_build_context_with_history(self, setup_orchestrator):
        """Test context building includes historical reflections."""
        orch = setup_orchestrator
        recent_refs = [
            {"failure_stage": "turnover", "failure_reason": "High TO", "suggested_fix": "Use decay"},
            {"failure_stage": "hypothesis", "failure_reason": "Wrong direction", "suggested_fix": "Reverse"},
        ]

        context = orch._build_reflection_context(
            reflection_result=ReflectionResult(),
            critique_result=None,
            recent_reflections=recent_refs,
            failure_patterns={"turnover": 5, "hypothesis": 2},
            is_specialized=False,
        )

        assert "历史类似案例" in context
        assert "turnover" in context
        assert "5 次" in context


class TestCallLLMWithReflectionContext:
    """Test _call_llm_with_reflection_context method."""

    @pytest.fixture
    def setup_orchestrator(self):
        orchestrator = FeedbackLoopOrchestrator(
            cookies=MagicMock(),
            slot_manager=MagicMock(),
        )
        orchestrator.config["llm_improve_fn"] = AsyncMock()
        return orchestrator

    @pytest.mark.asyncio
    async def test_successful_llm_call_returns_expression(self, setup_orchestrator):
        """Test that valid LLM response returns improved expression."""
        orch = setup_orchestrator
        orch.config["llm_improve_fn"].return_value = "group_neutralize(ts_decay_linear(rank(close), 10), industry)"

        result = await orch._call_llm_with_reflection_context(
            expression="group_neutralize(rank(close), industry)",
            wq_feedback={"sharpe": 0.9, "turnover": 0.4, "checks": []},
            reflection_context="## Analysis\n- Fix: Add decay",
            improvement_gen=1,
            original_task_id="task_1",
        )

        assert result is not None
        assert "ts_decay_linear" in result
        orch.config["llm_improve_fn"].assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_response_returns_none(self, setup_orchestrator):
        """Test that empty/invalid LLM response returns None."""
        orch = setup_orchestrator
        orch.config["llm_improve_fn"].return_value = ""

        result = await orch._call_llm_with_reflection_context(
            expression="group_neutralize(rank(close), industry)",
            wq_feedback={"sharpe": 0.9, "turnover": 0.4, "checks": []},
            reflection_context="## Analysis",
            improvement_gen=1,
            original_task_id="task_1",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_code_block_format_handled(self, setup_orchestrator):
        """Test that code block format in LLM response is handled correctly."""
        orch = setup_orchestrator
        orch.config["llm_improve_fn"].return_value = "```\ngroup_neutralize(tanh(rank(close)), industry)\n```"

        result = await orch._call_llm_with_reflection_context(
            expression="group_neutralize(rank(close), industry)",
            wq_feedback={"sharpe": 0.9, "turnover": 0.4, "checks": []},
            reflection_context="## Analysis",
            improvement_gen=1,
            original_task_id="task_1",
        )

        assert result is not None
        assert "tanh" in result
        assert not result.startswith("```")

    @pytest.mark.asyncio
    async def test_same_expression_returns_none(self, setup_orchestrator):
        """Test that unchanged expression returns None."""
        orch = setup_orchestrator
        original_expr = "group_neutralize(rank(close), industry)"
        orch.config["llm_improve_fn"].return_value = original_expr

        result = await orch._call_llm_with_reflection_context(
            expression=original_expr,
            wq_feedback={"sharpe": 0.9, "turnover": 0.4, "checks": []},
            reflection_context="## Analysis",
            improvement_gen=1,
            original_task_id="task_1",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_exception_handling_returns_none(self, setup_orchestrator):
        """Test that LLM exceptions are handled gracefully."""
        orch = setup_orchestrator
        orch.config["llm_improve_fn"].side_effect = Exception("LLM error")

        result = await orch._call_llm_with_reflection_context(
            expression="group_neutralize(rank(close), industry)",
            wq_feedback={"sharpe": 0.9, "turnover": 0.4, "checks": []},
            reflection_context="## Analysis",
            improvement_gen=1,
            original_task_id="task_1",
        )

        assert result is None


class TestRecordReflectionOutcome:
    """Test _record_reflection_outcome method."""

    @pytest.fixture
    def setup_orchestrator_with_graph_db(self):
        orchestrator = FeedbackLoopOrchestrator(
            cookies=MagicMock(),
            slot_manager=MagicMock(),
        )
        orchestrator._graph_db = MagicMock()
        return orchestrator

    def test_record_success_logs_correctly(self, setup_orchestrator_with_graph_db):
        """Test that success recording logs appropriately."""
        orch = setup_orchestrator_with_graph_db

        orch._record_reflection_outcome(
            expression="group_neutralize(rank(close), industry)",
            success=True,
            new_expression="group_neutralize(tanh(rank(close)), industry)",
            strategy="standard_reflection",
        )

    def test_record_failure_logs_correctly(self, setup_orchestrator_with_graph_db):
        """Test that failure recording logs appropriately."""
        orch = setup_orchestrator_with_graph_db

        orch._record_reflection_outcome(
            expression="group_neutralize(rank(close), industry)",
            success=False,
            new_expression=None,
            strategy="specialized_high_icir_low_fitness",
        )

    def test_record_without_graph_db_no_error(self):
        """Test recording works even without graph DB."""
        orchestrator = FeedbackLoopOrchestrator(
            cookies=MagicMock(),
            slot_manager=MagicMock(),
        )
        orchestrator._graph_db = None

        orchestrator._record_reflection_outcome(
            expression="group_neutralize(rank(close), industry)",
            success=True,
            new_expression="improved",
            strategy="test_strategy",
        )


class TestReflectionIntegrationEdgeCases:
    """Test edge cases and error handling in ReflectionEngine integration."""

    @pytest.fixture
    def setup_orchestrator(self):
        orchestrator = FeedbackLoopOrchestrator(
            cookies=MagicMock(),
            slot_manager=MagicMock(),
        )
        orchestrator._reflection_engine = MagicMock(spec=ReflectionEngine)
        orchestrator.config["llm_improve_fn"] = AsyncMock()
        orchestrator.config["max_improvement_generations"] = 3
        return orchestrator

    @pytest.mark.asyncio
    async def test_reflection_engine_none_skips_integration(self, setup_orchestrator):
        """Test that None _reflection_engine skips to standard path."""
        orch = setup_orchestrator
        orch._reflection_engine = None

        brain_result = MagicMock()
        brain_result.sharpe = 0.9
        brain_result.fitness = 0.8
        brain_result.turnover = 0.25
        brain_result.brain_checks = []

        with patch.object(orch, "_improve_and_resubmit", new_callable=AsyncMock) as mock_improve:
            mock_improve.return_value = "improved"
            await orch._handle_improvement(
                MagicMock(), brain_result,
                "group_neutralize(rank(close), industry)", "task_1"
            )
            mock_improve.assert_called_once()

    @pytest.mark.asyncio
    async def test_reflect_on_failure_exception_fallback(self, setup_orchestrator):
        """Test exception in reflect_on_failure falls back gracefully."""
        orch = setup_orchestrator
        orch._reflection_engine.reflect_on_failure = AsyncMock(
            side_effect=Exception("Reflection failed")
        )

        brain_result = MagicMock()
        brain_result.sharpe = 0.9
        brain_result.fitness = 0.8
        brain_result.turnover = 0.25
        brain_result.brain_checks = []

        with patch.object(orch, "_improve_and_resubmit", new_callable=AsyncMock) as mock_improve:
            mock_improve.return_value = "improved"
            await orch._handle_improvement(
                MagicMock(), brain_result,
                "group_neutralize(rank(close), industry)", "task_1"
            )
            mock_improve.assert_called_once()

    @pytest.mark.asyncio
    async def test_max_generations_limit_respected(self, setup_orchestrator):
        """Test that max_improvement_generations limit is respected."""
        orch = setup_orchestrator
        orch.config["max_improvement_generations"] = 1

        chain = ImprovementChain(
            original_task_id="task_1",
            original_expression="expr",
        )
        chain.current_generation = 1
        orch._improvement_chains["task_1"] = chain

        brain_result = MagicMock()
        brain_result.sharpe = 0.9
        brain_result.fitness = 0.8
        brain_result.turnover = 0.25
        brain_result.brain_checks = []

        await orch._handle_improvement(
            MagicMock(), brain_result, "expr", "task_1"
        )

        assert chain.current_generation == 1

    @pytest.mark.asyncio
    async def test_improvement_chain_tracking(self, setup_orchestrator):
        """Test that improvements are properly tracked in chain."""
        orch = setup_orchestrator
        orch._reflection_engine.reflect_on_failure = AsyncMock(return_value=ReflectionResult(
            failure_stage="parameters",
            failure_reason="Weak signal",
            suggested_fix="Enhance",
            confidence=0.7,
        ))
        orch._reflection_engine.self_critique = AsyncMock(return_value=CritiqueResult(
            consistency_score=0.8,
            critique_available=True,
        ))
        orch._reflection_engine.get_recent_reflections = MagicMock(return_value=[])
        orch._reflection_engine.get_failure_patterns = MagicMock(return_value={})
        orch.config["llm_improve_fn"].return_value = "improved_expression"

        brain_result = MagicMock()
        brain_result.sharpe = 0.9
        brain_result.fitness = 0.8
        brain_result.turnover = 0.25
        brain_result.brain_checks = []

        await orch._handle_improvement(
            MagicMock(), brain_result,
            "original_expr", "task_chain_test"
        )

        assert "task_chain_test" in orch._improvement_chains
        chain = orch._improvement_chains["task_chain_test"]
        assert len(chain.improvements) > 0
        assert chain.improvements[-1]["expression"] == "improved_expression"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
