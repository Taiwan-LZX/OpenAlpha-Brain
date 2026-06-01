import json
import unittest
from typing import Any, Dict, Optional

from alpha_agent.config import LLMGenConfig, ModelConfig
from alpha_agent.exploration_grid import (
    ExplorationGrid, GridCell,
)
from alpha_agent.expression_generator import (
    ExpressionGenerator,
    GeneratedCandidate,
    JaccardDiversityGate,
    build_template_expression,
    jaccard_diversity,
    tokenize_expression,
    OPERATOR_TEMPLATES,
    DEFAULT_WINDOWS,
)
from alpha_agent.llm_client import LLMClient


class TestTokenizeExpression(unittest.TestCase):
    def test_basic(self) -> None:
        tokens = tokenize_expression("rank(ts_mean(close, 20))")
        self.assertIn("rank", tokens)
        self.assertIn("ts_mean", tokens)
        self.assertIn("close", tokens)
        self.assertNotIn("20", tokens)

    def test_empty(self) -> None:
        self.assertEqual(tokenize_expression(""), set())


class TestJaccardDiversity(unittest.TestCase):
    def test_empty_frontier(self) -> None:
        score = jaccard_diversity({"close", "rank"}, [])
        self.assertEqual(score, 1.0)

    def test_empty_candidate(self) -> None:
        score = jaccard_diversity(set(), [{"close"}])
        self.assertEqual(score, 0.0)

    def test_identical(self) -> None:
        score = jaccard_diversity({"close", "rank"}, [{"close", "rank"}])
        self.assertAlmostEqual(score, 0.0)

    def test_completely_different(self) -> None:
        score = jaccard_diversity({"close", "rank"}, [{"volume", "zscore"}])
        self.assertAlmostEqual(score, 1.0)

    def test_partial_overlap(self) -> None:
        score = jaccard_diversity({"close", "rank"}, [{"close", "volume"}])
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.0)


class TestJaccardDiversityGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = JaccardDiversityGate(threshold=0.3)

    def test_score_drops_with_overlap(self) -> None:
        self.gate.seed_frontier(["rank(close)"])
        score = self.gate.score("rank(close)")
        self.assertLess(score, 0.3)

    def test_is_diverse(self) -> None:
        self.gate.seed_frontier(["rank(close)"])
        self.assertTrue(self.gate.is_diverse("zscore(volume)"))

    def test_filter_diverse(self) -> None:
        candidates = [
            GeneratedCandidate(expression="rank(close)", idea_name="a", family="value"),
            GeneratedCandidate(expression="zscore(volume)", idea_name="b", family="value"),
        ]
        self.gate.seed_frontier(["rank(close)"])
        result = self.gate.filter_diverse(candidates)
        self.assertLessEqual(len(result), 2)

    def test_add_to_frontier(self) -> None:
        self.gate.add_to_frontier("rank(close)")
        self.assertEqual(len(self.gate._frontier), 1)


class TestBuildTemplateExpression(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = ExplorationGrid()

    def test_fundamental_cross_sectional(self) -> None:
        cell = self.grid.get_cell("fundamental_cross_sectional_medium")
        assert cell is not None
        expr = build_template_expression(cell)
        self.assertIsInstance(expr, str)
        self.assertGreater(len(expr), 0)

    def test_price_volume_time_series(self) -> None:
        cell = self.grid.get_cell("price_volume_time_series_short")
        assert cell is not None
        expr = build_template_expression(cell)
        self.assertIn("ts_mean", expr)

    def test_group_operator(self) -> None:
        cell = self.grid.get_cell("fundamental_group_medium")
        assert cell is not None
        expr = build_template_expression(cell)
        self.assertIn("group", expr)


class TestExpressionGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = ExplorationGrid()
        self.generator = ExpressionGenerator()

    def test_generate_from_cell_returns_candidates(self) -> None:
        cell = self.grid.get_cell("price_volume_time_series_short")
        assert cell is not None
        results = self.generator.generate_from_cell(cell)
        self.assertGreater(len(results), 0)
        for c in results:
            self.assertIsInstance(c, GeneratedCandidate)
            self.assertIsInstance(c.expression, str)
            self.assertGreater(len(c.expression), 0)

    def test_generate_from_cell_all_cells_have_expression(self) -> None:
        for cell in list(self.grid.cells.values())[:10]:
            results = self.generator.generate_from_cell(cell, fresh_gate=True)
            self.assertGreater(len(results), 0)
            for c in results:
                self.assertGreater(len(c.expression), 0)

    def test_generate_from_cell_respects_diversity(self) -> None:
        cell = self.grid.get_cell("price_volume_time_series_short")
        assert cell is not None
        first = self.generator.generate_from_cell(cell)
        cell2 = self.grid.get_cell("price_volume_time_series_medium")
        assert cell2 is not None
        second = self.generator.generate_from_cell(cell2, frontier_expressions=[c.expression for c in first])
        for c in second:
            self.assertGreater(c.diversity_score, 0)

    def test_candidate_to_dict(self) -> None:
        c = GeneratedCandidate(expression="rank(close)", idea_name="test", family="value")
        d = c.to_dict()
        self.assertIn("expression", d)
        self.assertIn("diversity_score", d)
        self.assertIn("validation_passed", d)

    def test_generated_candidate_default_values(self) -> None:
        c = GeneratedCandidate(expression="rank(close)", idea_name="test", family="value")
        self.assertEqual(c.diversity_score, 0.0)
        self.assertFalse(c.validation_passed)
        self.assertEqual(c.validation_errors, [])
        self.assertEqual(c.cot_reasoning, "")


class MockLLMClient(LLMClient):
    def __init__(self, response: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(ModelConfig(provider="test"))
        self._response = response if response is not None else {"expression": "rank(close)", "reasoning": "test"}

    def chat_completion(self, *, body: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "choices": [{"message": {"content": json.dumps(self._response)}}],
        }

    def request_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._response


class TestExpressionGeneratorWithLLM(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = ExplorationGrid()
        self.llm_config = LLMGenConfig(enabled=True)

    def test_llm_path_returns_candidates(self) -> None:
        mock = MockLLMClient({"expression": "rank(ts_mean(close, 20))", "reasoning": "Momentum signal"})
        gen = ExpressionGenerator(llm_client=mock, llm_gen_config=self.llm_config)
        cell = self.grid.get_cell("price_volume_time_series_short")
        assert cell is not None
        results = gen.generate_from_cell(cell, fresh_gate=True)
        self.assertGreater(len(results), 0)
        self.assertIn("rank", results[0].expression)
        self.assertEqual(results[0].cot_reasoning, "Momentum signal")

    def test_llm_fallback_to_template_on_failure(self) -> None:
        class FailingMock(MockLLMClient):
            def request_json(self, **kwargs: Any) -> Dict[str, Any]:
                raise RuntimeError("LLM unavailable")

        mock = FailingMock()
        gen = ExpressionGenerator(llm_client=mock, llm_gen_config=self.llm_config)
        cell = self.grid.get_cell("price_volume_time_series_short")
        assert cell is not None
        results = gen.generate_from_cell(cell, fresh_gate=True)
        self.assertGreater(len(results), 0)
        self.assertFalse(results[0].cot_reasoning)

    def test_llm_disabled_uses_templates(self) -> None:
        disabled_config = LLMGenConfig(enabled=False)
        mock = MockLLMClient({"expression": "rank(close)", "reasoning": "test"})
        gen = ExpressionGenerator(llm_client=mock, llm_gen_config=disabled_config)
        cell = self.grid.get_cell("price_volume_time_series_short")
        assert cell is not None
        results = gen.generate_from_cell(cell, fresh_gate=True)
        self.assertGreater(len(results), 0)
        self.assertFalse(results[0].cot_reasoning)

    def test_no_llm_client_uses_templates(self) -> None:
        gen = ExpressionGenerator()
        cell = self.grid.get_cell("price_volume_time_series_short")
        assert cell is not None
        results = gen.generate_from_cell(cell, fresh_gate=True)
        self.assertGreater(len(results), 0)
        self.assertFalse(results[0].cot_reasoning)

    def test_cot_disabled_uses_templates(self) -> None:
        config = LLMGenConfig(enabled=True, cot_enabled=False)
        mock = MockLLMClient({"expression": "rank(close)", "reasoning": "test"})
        gen = ExpressionGenerator(llm_client=mock, llm_gen_config=config)
        cell = self.grid.get_cell("price_volume_time_series_short")
        assert cell is not None
        results = gen.generate_from_cell(cell, fresh_gate=True)
        self.assertGreater(len(results), 0)
        self.assertFalse(results[0].cot_reasoning)

    def test_llm_empty_expression_falls_back(self) -> None:
        mock = MockLLMClient({})
        gen = ExpressionGenerator(llm_client=mock, llm_gen_config=self.llm_config)
        cell = self.grid.get_cell("price_volume_time_series_short")
        assert cell is not None
        results = gen.generate_from_cell(cell, fresh_gate=True)
        self.assertGreater(len(results), 0)
        self.assertFalse(results[0].cot_reasoning)


if __name__ == "__main__":
    unittest.main()
