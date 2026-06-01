"""Comprehensive tests for EASearchStrategy - EA search strategy layer.

Tests cover:
  - Population initialization (diversity checks)
  - All mutation strategies (NearPass/LLM/OperatorSwap/ParamTune)
  - Crossover operation (Block A swap correctness)
  - Selection mechanism (elitism verification)
  - Diversity computation
  - Full search loop with mocked evaluation
  - Edge cases and error handling

Run: pytest tests/test_ea_search.py -v
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from openalpha_brain.evolution.ea_search import (
    EAConfig,
    EAMutationType,
    EASearchStrategy,
    FactorIndividual,
    extract_block_a,
    extract_block_c,
    mutate_operator,
    swap_block_a,
    tune_parameter,
)

SEED_EXPR = (
    "ts_decay_linear(group_neutralize(-rank(signed_power(ts_zscore(returns / bookvalue_ps, 20), 2)), sector), 10)"
)

SIMPLE_EXPR = "ts_decay_linear(group_neutralize(rank(close/volume), sector), 10)"

SHORT_EXPR = "rank(close)"


class TestExtractBlockA:
    """Tests for extract_block_a helper function."""

    def test_extract_from_standard_expression(self):
        result = extract_block_a(SIMPLE_EXPR)
        assert result == "rank(close/volume)"

    def test_extract_from_complex_expression(self):
        result = extract_block_a(SEED_EXPR)
        assert result is not None
        assert "rank" in result
        assert "signed_power" in result

    def test_extract_returns_none_for_no_group_neutralize(self):
        result = extract_block_a("rank(close)")
        assert result is None

    def test_extract_handles_nested_calls(self):
        expr = "ts_decay_linear(group_neutralize(ts_mean(rank(close), 5), industry), 20)"
        result = extract_block_a(expr)
        assert result is not None
        assert "ts_mean" in result
        assert "rank(close)" in result


class TestExtractBlockC:
    """Tests for extract_block_c helper function."""

    def test_extract_decay_window(self):
        result = extract_block_c(SIMPLE_EXPR)
        assert result == "10"

    def test_extract_complex_decay(self):
        result = extract_block_c(SEED_EXPR)
        assert result is not None
        assert result.isdigit()

    def test_extract_returns_none_when_absent(self):
        result = extract_block_c("rank(close)")
        assert result is None


class TestSwapBlockA:
    """Tests for swap_block_a crossover helper."""

    def test_successful_swap(self):
        expr_a = "ts_decay_linear(group_neutralize(rank(close), sector), 10)"
        expr_b = "ts_decay_linear(group_neutralize(zscore(volume), sector), 20)"
        child_a, child_b = swap_block_a(expr_a, expr_b)
        assert "zscore(volume)" in child_a
        assert "rank(close)" in child_b
        assert child_a != expr_a
        assert child_b != expr_b
        # Verify decay windows are preserved from respective parents
        assert ", 10)" in child_a
        assert ", 20)" in child_b

    def test_swap_preserves_structure(self):
        expr_a = SIMPLE_EXPR
        expr_b = "ts_decay_linear(group_neutralize(ts_delta(high, 5), industry), 15)"
        child_a, _ = swap_block_a(expr_a, expr_b)
        assert "group_neutralize" in child_a
        assert "sector" in child_a or "industry" in child_a

    def test_swap_raises_on_missing_block_a(self):
        with pytest.raises(ValueError):
            swap_block_a("rank(close)", SIMPLE_EXPR)

    def test_swap_raises_when_both_missing(self):
        with pytest.raises(ValueError):
            swap_block_a("rank(close)", "zscore(volume)")


class TestMutateOperator:
    """Tests for mutate_operator helper."""

    def test_replace_rank_with_zscore(self):
        result = mutate_operator("rank(close/volume)", "rank", "zscore")
        assert result == "zscore(close/volume)"

    def test_replace_ts_mean(self):
        result = mutate_operator("ts_mean(close, 10)", "ts_mean", "ts_decay_linear")
        assert result == "ts_decay_linear(close, 10)"

    def test_no_change_when_op_not_found(self):
        result = mutate_operator("zscore(close)", "rank", "scale")
        assert result is None

    def test_only_replaces_first_occurrence(self):
        result = mutate_operator("rank(rank(close))", "rank", "zscore")
        assert result == "zscore(rank(close))"


class TestTuneParameter:
    """Tests for tune_parameter helper."""

    def test_tune_decay_window_increase(self):
        result = tune_parameter("ts_decay_linear(x, 10)", "decay_window", 5)
        assert result == "ts_decay_linear(x, 15)"

    def test_tune_decay_window_decrease(self):
        result = tune_parameter("ts_decay_linear(x, 20)", "decay_window", -5)
        assert result == "ts_decay_linear(x, 15)"

    def test_tune_power_parameter(self):
        result = tune_parameter("signed_power(x, 2.0)", "power", 0.5)
        assert result == "signed_power(x, 2.5)"

    def test_returns_none_when_param_not_found(self):
        result = tune_parameter("rank(close)", "decay_window", 5)
        assert result is None

    def test_rejects_negative_result(self):
        result = tune_parameter("ts_decay_linear(x, 3)", "decay_window", -5)
        assert result is None


class TestFactorIndividual:
    """Tests for FactorIndividual dataclass."""

    def test_auto_id_generation(self):
        ind = FactorIndividual(expression="rank(close)")
        assert len(ind.id) > 0
        assert "id" in ind.metadata

    def test_default_values(self):
        ind = FactorIndividual(expression="test")
        assert ind.fitness == 0.0
        assert ind.generation == 0
        assert ind.parent_ids == []
        assert ind.mutation_type == EAMutationType.NEAR_PASS_DETERMINISTIC

    def test_custom_metadata_preserved(self):
        ind = FactorIndividual(
            expression="test",
            metadata={"custom_key": "value"},
        )
        assert ind.metadata["custom_key"] == "value"
        assert "id" in ind.metadata


class TestEAConfig:
    """Tests for EAConfig dataclass."""

    def test_default_values(self):
        config = EAConfig()
        assert config.population_size == 8
        assert config.max_generations == 3
        assert config.elite_ratio == 0.25
        assert config.mutation_rate == 0.6
        assert config.crossover_rate == 0.3
        assert config.llm_mutation_prob == 0.3
        assert config.diversity_threshold == 0.7
        assert config.timeout_seconds == 300

    def test_custom_values(self):
        config = EAConfig(population_size=16, max_generations=5)
        assert config.population_size == 16
        assert config.max_generations == 5


class TestEASearchStrategyInit:
    """Tests for EASearchStrategy initialization."""

    def setup_method(self):
        self.strategy = EASearchStrategy()

    def test_default_config(self):
        assert self.strategy.config.population_size == 8
        assert self.strategy.config.max_generations == 3

    def test_custom_init_params(self):
        strategy = EASearchStrategy(population_size=12, max_generations=5)
        assert strategy.config.population_size == 12
        assert strategy.config.max_generations == 5

    def test_dependencies_none_by_default(self):
        assert self.strategy._near_pass is None
        assert self.strategy._llm_client is None
        assert self.strategy._slot_manager is None


class TestInitializeDependencies:
    """Tests for dependency injection."""

    def setup_method(self):
        self.strategy = EASearchStrategy()

    def test_initialize_creates_near_pass(self):
        self.strategy.initialize_dependencies()
        assert self.strategy._near_pass is not None
        from openalpha_brain.evolution.near_pass_improver import NearPassImprover

        assert isinstance(self.strategy._near_pass, NearPassImprover)

    def test_initialize_with_custom_near_pass(self):
        mock_improver = MagicMock()
        self.strategy.initialize_dependencies(near_pass_improver=mock_improver)
        assert self.strategy._near_pass is mock_improver

    def test_initialize_with_llm_client(self):
        mock_llm = MagicMock()
        self.strategy.initialize_dependencies(llm_client=mock_llm)
        assert self.strategy._llm_client is mock_llm


class TestInitializePopulation:
    """Tests for population initialization."""

    def setup_method(self):
        self.strategy = EASearchStrategy(population_size=8)
        from openalpha_brain.evolution.near_pass_improver import NearPassImprover

        self.strategy.initialize_dependencies(near_pass_improver=NearPassImprover())

    @pytest.mark.asyncio
    async def test_population_size_correct(self):
        pop = await self.strategy._initialize_population(SIMPLE_EXPR, 0.5)
        assert len(pop) == 8

    @pytest.mark.asyncio
    async def test_seed_in_population(self):
        pop = await self.strategy._initialize_population(SIMPLE_EXPR, 0.5)
        expressions = [ind.expression for ind in pop]
        assert SIMPLE_EXPR in expressions

    @pytest.mark.asyncio
    async def test_population_has_diversity(self):
        pop = await self.strategy._initialize_population(SEED_EXPR, 0.8)
        unique_exprs = set(ind.expression for ind in pop)
        assert len(unique_exprs) >= 3, "Should have at least 3 unique expressions"

    @pytest.mark.asyncio
    async def test_all_individuals_have_fitness(self):
        pop = await self.strategy._initialize_population(SIMPLE_EXPR, 0.6)
        for ind in pop:
            assert isinstance(ind.fitness, float)

    @pytest.mark.asyncio
    async def test_generation_zero_for_all(self):
        pop = await self.strategy._initialize_population(SIMPLE_EXPR, 0.5)
        for ind in pop:
            assert ind.generation == 0


class TestMutation:
    """Tests for mutation operations."""

    def setup_method(self):
        self.strategy = EASearchStrategy(llm_mutation_prob=0.0)
        from openalpha_brain.evolution.near_pass_improver import NearPassImprover

        self.strategy.initialize_dependencies(near_pass_improver=NearPassImprover())

    @pytest.mark.asyncio
    async def test_fast_mutate_produces_offspring(self):
        parent = FactorIndividual(expression=SIMPLE_EXPR, fitness=0.5)
        offspring = await self.strategy._mutate(parent)
        assert len(offspring) >= 1, "Fast mutation should produce at least one offspring"

    @pytest.mark.asyncio
    async def test_offspring_differs_from_parent(self):
        parent = FactorIndividual(expression=SIMPLE_EXPR, fitness=0.5)
        offspring = await self.strategy._mutate(parent)
        for child in offspring:
            assert child.expression != parent.expression or child.fitness != parent.fitness

    @pytest.mark.asyncio
    async def test_offspring_inherits_parent_id(self):
        parent = FactorIndividual(expression=SIMPLE_EXPR, fitness=0.5)
        offspring = await self.strategy._mutate(parent)
        for child in offspring:
            assert parent.id in child.parent_ids

    @pytest.mark.asyncio
    async def test_mutation_type_set_correctly(self):
        parent = FactorIndividual(expression=SIMPLE_EXPR, fitness=0.5)
        offspring = await self.strategy._mutate(parent)
        valid_types = {
            EAMutationType.OPERATOR_SWAP,
            EAMutationType.PARAMETER_TUNE,
            EAMutationType.NEAR_PASS_DETERMINISTIC,
        }
        for child in offspring:
            assert child.mutation_type in valid_types


class TestLLMMutation:
    """Tests for LLM semantic mutation (Tier3)."""

    def setup_method(self):
        self.strategy = EASearchStrategy(llm_mutation_prob=1.0)
        from openalpha_brain.evolution.near_pass_improver import NearPassImprover

        self.strategy.initialize_dependencies(near_pass_improver=NearPassImprover())

    @pytest.mark.asyncio
    async def test_llm_mutation_calls_client(self):
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "ts_decay_linear(group_neutralize(zscore(close/volume), sector), 15)"
        self.strategy._llm_client = mock_llm

        parent = FactorIndividual(expression=SIMPLE_EXPR, fitness=0.5)
        offspring = await self.strategy._mutate(parent)

        assert len(offspring) >= 1
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_mutation_sets_semantic_type(self):
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "zscore(ts_delta(close, 5))"
        self.strategy._llm_client = mock_llm

        parent = FactorIndividual(expression=SIMPLE_EXPR, fitness=0.5)
        offspring = await self.strategy._mutate(parent)

        semantic_children = [c for c in offspring if c.mutation_type == EAMutationType.LLM_SEMANTIC]
        assert len(semantic_children) >= 1

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_fast(self):
        mock_llm = AsyncMock(side_effect=Exception("LLM error"))
        self.strategy._llm_client = mock_llm

        parent = FactorIndividual(expression=SIMPLE_EXPR, fitness=0.5)
        offspring = await self.strategy._mutate(parent)

        assert len(offspring) >= 1, "Should fall back to fast mutation on LLM failure"


class TestCrossover:
    """Tests for crossover operation."""

    def setup_method(self):
        self.strategy = EASearchStrategy()
        from openalpha_brain.evolution.near_pass_improver import NearPassImprover

        self.strategy.initialize_dependencies(near_pass_improver=NearPassImprover())

    def test_successful_crossover(self):
        parent_a = FactorIndividual(
            expression="ts_decay_linear(group_neutralize(rank(close), sector), 10)",
            fitness=0.7,
        )
        parent_b = FactorIndividual(
            expression="ts_decay_linear(group_neutralize(zscore(volume), sector), 20)",
            fitness=0.6,
        )
        child = self.strategy._crossover(parent_a, parent_b)
        assert child is not None
        assert child.expression != parent_a.expression
        assert "zscore(volume)" in child.expression

    def test_crossover_without_group_neutralize_returns_none(self):
        parent_a = FactorIndividual(expression="rank(close)", fitness=0.5)
        parent_b = FactorIndividual(expression="zscore(volume)", fitness=0.4)
        child = self.strategy._crossover(parent_a, parent_b)
        assert child is None

    def test_child_has_both_parent_ids(self):
        parent_a = FactorIndividual(expression=SIMPLE_EXPR, fitness=0.7)
        parent_b = FactorIndividual(
            expression="ts_decay_linear(group_neutralize(ts_delta(high, 5), sector), 15)",
            fitness=0.6,
        )
        child = self.strategy._crossover(parent_a, parent_b)
        if child is not None:
            assert parent_a.id in child.parent_ids
            assert parent_b.id in child.parent_ids


class TestSelection:
    """Tests for elitist selection mechanism."""

    def setup_method(self):
        self.strategy = EASearchStrategy(population_size=6, elite_ratio=0.33)
        from openalpha_brain.evolution.near_pass_improver import NearPassImprover

        self.strategy.initialize_dependencies(near_pass_improver=NearPassImprover())

    def test_elites_preserved(self):
        population = [FactorIndividual(expression=f"expr_{i}", fitness=float(i * 0.1)) for i in range(6)]
        offspring = [FactorIndividual(expression=f"child_{i}", fitness=0.05) for i in range(4)]
        next_gen = self.strategy._select(population, offspring)
        elite_count = int(6 * 0.33)
        elites = next_gen[:elite_count]
        assert all(e.fitness >= 0.4 for e in elites), "Elites should be the fittest individuals"

    def test_population_size_maintained(self):
        population = [FactorIndividual(expression=f"p{i}", fitness=0.5) for i in range(6)]
        offspring = [FactorIndividual(expression=f"c{i}", fitness=0.3) for i in range(4)]
        next_gen = self.strategy._select(population, offspring)
        assert len(next_gen) == 6

    def test_better_offspring_can_enter_population(self):
        population = [FactorIndividual(expression=f"expr_{i}", fitness=0.1 + i * 0.05) for i in range(6)]
        strong_offspring = [FactorIndividual(expression="super", fitness=0.9)]
        next_gen = self.strategy._select(population, strong_offspring)
        best = max(next_gen, key=lambda x: x.fitness)
        assert best.expression == "super"

    def test_handles_empty_offspring(self):
        population = [FactorIndividual(expression=f"p{i}", fitness=0.5) for i in range(6)]
        next_gen = self.strategy._select(population, [])
        assert len(next_gen) == 6


class TestDiversity:
    """Tests for diversity computation."""

    def setup_method(self):
        self.strategy = EASearchStrategy()

    def test_identical_population_low_diversity(self):
        population = [FactorIndividual(expression=SIMPLE_EXPR, fitness=0.5) for _ in range(5)]
        div = self.strategy._compute_diversity(population)
        assert div < 0.1, "Identical expressions should yield near-zero diversity"

    def test_diverse_population_high_diversity(self):
        expressions = [
            "rank(close)",
            "zscore(volume)",
            "ts_delta(high, 5)",
            "ts_corr(close, volume, 20)",
            "group_neutralize(returns, sector)",
        ]
        population = [FactorIndividual(expr, 0.5) for expr in expressions]
        div = self.strategy._compute_diversity(population)
        assert div > 0.3, "Diverse expressions should yield higher diversity"

    def test_single_individual_max_diversity(self):
        population = [FactorIndividual(expression="rank(close)", fitness=0.5)]
        div = self.strategy._compute_diversity(population)
        assert div == 1.0, "Single individual has perfect diversity by definition"

    def test_empty_population(self):
        div = self.strategy._compute_diversity([])
        assert div == 1.0


class TestQuickEvaluate:
    """Tests for heuristic quick evaluation."""

    def setup_method(self):
        self.strategy = EASearchStrategy()

    def test_returns_float(self):
        score = self.strategy._quick_evaluate("rank(close)")
        assert isinstance(score, float)

    def test_score_in_valid_range(self):
        score = self.strategy._quick_evaluate("ts_decay_linear(group_neutralize(rank(close/volume), sector), 10)")
        assert 0.0 <= score <= 1.0

    def test_complex_expression_scores_higher(self):
        simple_score = self.strategy._quick_evaluate("close")
        complex_score = self.strategy._quick_evaluate(
            "ts_decay_linear(group_neutralize(rank(signed_power(returns, 2)), sector), 20)"
        )
        assert complex_score > simple_score

    def test_normalization_bonus(self):
        no_norm = self.strategy._quick_evaluate("ts_mean(close, 10)")
        with_norm = self.strategy._quick_evaluate("ts_decay_linear(group_neutralize(rank(close), sector), 10)")
        assert with_norm > no_norm

    def test_too_long_penalty(self):
        normal = self.strategy._quick_evaluate("rank(close)")
        too_long = self.strategy._quick_evaluate("x" * 250)
        assert normal >= too_long


class TestFullSearchLoop:
    """Integration tests for complete search workflow."""

    def setup_method(self):
        self.strategy = EASearchStrategy(
            population_size=6,
            max_generations=2,
            timeout_seconds=30,
        )
        from openalpha_brain.evolution.near_pass_improver import NearPassImprover

        self.strategy.initialize_dependencies(near_pass_improver=NearPassImprover())

    @pytest.mark.asyncio
    async def test_search_completes(self):
        best, history = await self.strategy.search(
            seed_expression=SIMPLE_EXPR,
            target_sharpe=2.0,
            initial_sharpe=0.5,
        )
        assert isinstance(best, FactorIndividual)
        assert isinstance(history, list)
        assert len(history) >= 6

    @pytest.mark.asyncio
    async def test_best_has_higher_or_equal_fitness(self):
        best, _ = await self.strategy.search(
            seed_expression=SIMPLE_EXPR,
            initial_sharpe=0.3,
        )
        assert best.fitness >= 0.3

    @pytest.mark.asyncio
    async def test_history_tracks_all_individuals(self):
        _, history = await self.strategy.search(
            seed_expression=SEED_EXPR,
            target_sharpe=1.5,
            initial_sharpe=0.4,
        )
        assert len(history) >= self.strategy.config.population_size

    @pytest.mark.asyncio
    async def test_stats_collected(self):
        await self.strategy.search(seed_expression=SIMPLE_EXPR, initial_sharpe=0.5)
        stats = self.strategy.get_stats()
        assert len(stats) >= 1
        assert "generation" in stats[0]
        assert "best_fitness" in stats[0]
        assert "diversity" in stats[0]

    @pytest.mark.asyncio
    async def test_reset_stats_clears(self):
        await self.strategy.search(seed_expression=SIMPLE_EXPR, initial_sharpe=0.5)
        self.strategy.reset_stats()
        assert len(self.strategy.get_stats()) == 0

    @pytest.mark.asyncio
    async def test_raises_without_initialization(self):
        bad_strategy = EASearchStrategy()
        with pytest.raises(RuntimeError, match="Dependencies not initialized"):
            await bad_strategy.search(seed_expression="rank(close)")

    @pytest.mark.asyncio
    async def test_timeout_respected(self):
        slow_strategy = EASearchStrategy(timeout_seconds=1, max_generations=100)
        from openalpha_brain.evolution.near_pass_improver import NearPassImprover

        slow_strategy.initialize_dependencies(near_pass_improver=NearPassImprover())
        best, history = await slow_strategy.search(
            seed_expression=SIMPLE_EXPR,
            initial_sharpe=0.5,
        )
        assert isinstance(best, FactorIndividual)


class TestSubmitAndEval:
    """Tests for optional WQ submission path."""

    def setup_method(self):
        self.strategy = EASearchStrategy()
        from openalpha_brain.evolution.near_pass_improver import NearPassImprover

        self.strategy.initialize_dependencies(near_pass_improver=NearPassImprover())

    @pytest.mark.asyncio
    async def test_falls_back_without_slot_manager(self):
        individual = FactorIndividual(expression=SIMPLE_EXPR, fitness=0.0)
        result = await self.strategy._submit_and_eval(individual)
        assert result.fitness > 0, "Should use quick_eval when no slot_manager"

    @pytest.mark.asyncio
    async def test_uses_slot_manager_when_available(self):
        mock_sm = AsyncMock()
        mock_sm.submit.return_value = "task_123"
        mock_sm.wait_for_completion.return_value = {"sharpe": 1.35}
        self.strategy._slot_manager = mock_sm

        individual = FactorIndividual(expression=SIMPLE_EXPR, fitness=0.0)
        result = await self.strategy._submit_and_eval(individual, priority="high")
        assert result.fitness == 1.35
        mock_sm.submit.assert_called_once()


class TestInjectRandomDiversity:
    """Tests for diversity injection on low diversity."""

    def setup_method(self):
        self.strategy = EASearchStrategy()
        from openalpha_brain.evolution.near_pass_improver import NearPassImprover

        self.strategy.initialize_dependencies(near_pass_improver=NearPassImprover())

    @pytest.mark.asyncio
    async def test_injects_variants(self):
        variants = await self.strategy._inject_random_diversity(SIMPLE_EXPR, generation=2)
        assert len(variants) == 3

    @pytest.mark.asyncio
    async def test_variants_are_diverse(self):
        variants = await self.strategy._inject_random_diversity(SIMPLE_EXPR, generation=1)
        exprs = [v.expression for v in variants]
        assert len(set(exprs)) == 3, "All variants should be unique"

    @pytest.mark.asyncio
    async def test_variants_have_correct_generation(self):
        variants = await self.strategy._inject_random_diversity(SIMPLE_EXPR, generation=5)
        for v in variants:
            assert v.generation == 5


class TestEditDistance:
    """Tests for edit distance utility."""

    def test_identical_strings(self):
        dist = EASearchStrategy._edit_distance("abc", "abc")
        assert dist == 0

    def test_completely_different(self):
        dist = EASearchStrategy._edit_distance("abc", "xyz")
        assert dist == 3

    def test_one_empty(self):
        dist = EASearchStrategy._edit_distance("", "abc")
        assert dist == 3

    def test_substitution_only(self):
        dist = EASearchStrategy._edit_distance("kitten", "sitting")
        assert dist == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
