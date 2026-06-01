"""Tests for ToTSearchStrategy and related components.

Comprehensive test suite covering:
- Data structures (ToTNode, ToTConfig, ToTSearchResult)
- Strategy initialization and dependency injection
- Node creation and tree building
- Node expansion strategies (LLM, mutation, crossover)
- Survivor selection logic
- Diversity enforcement
- Full search workflow
- Utility functions (fingerprinting, diversity computation)

Test count: 43 tests across 11 test classes.
"""
from __future__ import annotations

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from openalpha_brain.evolution.tot_search import (
    ToTNode,
    ToTNodeState,
    ToTSearchResult,
    ToTConfig,
    ToTSearchStrategy,
    extract_expression_fingerprint,
    compute_expression_diversity,
    select_diverse_subset,
)


# ==============================================================================
# Test Fixtures
# ==============================================================================

@pytest.fixture
def sample_expression():
    return "ts_decay_linear(group_neutralize(rank(close/volume), sector), 10)"


@pytest.fixture
def sample_node(sample_expression):
    return ToTNode(
        node_id="1_1",
        expression=sample_expression,
        depth=1,
        fitness=0.85,
        generation_method="seed",
    )


@pytest.fixture
def default_config():
    return ToTConfig()


@pytest.fixture
def custom_config():
    return ToTConfig(
        max_depth=4,
        branch_factor=6,
        top_k_survivors=3,
        accept_threshold=0.1,
        timeout_seconds=600,
        max_total_nodes=100,
    )


@pytest.fixture
def strategy(default_config):
    return ToTSearchStrategy(config=default_config)


@pytest.fixture
def initialized_strategy(strategy):
    """Strategy with mocked dependencies."""
    mock_near_pass = MagicMock()
    mock_near_pass.analyze.return_value = MagicMock(
        improvement_priority=[
            "increase_decay_window",
            "replace_rank_with_zscore",
            "tune_parameters",
        ]
    )
    mock_near_pass.generate_deterministic_variants.return_value = [
        MagicMock(expression="ts_decay_linear(group_neutralize(zscore(close/volume), sector), 20)"),
        MagicMock(expression="ts_decay_linear(group_neutralize(rank(close/volume), sector), 20)"),
    ]
    
    strategy.initialize_dependencies(
        near_pass_improver=mock_near_pass,
        llm_client=None,
        prefilter=None,
    )
    return strategy


@pytest.fixture
def llm_strategy(initialized_strategy):
    """Strategy with LLM client."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = '["expr1", "expr2", "expr3"]'
    
    initialized_strategy._llm_client = mock_llm
    return initialized_strategy


# ==============================================================================
# Test Class: ToTNode
# ==============================================================================

class TestToTNode:
    """Tests for ToTNode data structure."""
    
    def test_node_creation_with_defaults(self):
        """Test node creation with default values."""
        node = ToTNode(node_id="1_1", expression="test_expr", depth=1)
        
        assert node.node_id == "1_1"
        assert node.expression == "test_expr"
        assert node.depth == 1
        assert node.parent_id is None
        assert node.fitness == 0.0
        assert node.metrics == {}
        assert node.state == ToTNodeState.EXPANDING
        assert node.children_ids == []
        assert node.generation_method == "seed"
        assert node.reason == ""
        assert isinstance(node.created_at, float)
    
    def test_node_creation_with_custom_values(self):
        """Test node creation with custom values."""
        now = time.time()
        node = ToTNode(
            node_id="2_3",
            expression="complex_expr",
            depth=2,
            parent_id="1_1",
            fitness=0.95,
            metrics={"sharpe": 1.25, "turnover": 15.0},
            state=ToTNodeState.EVALUATED,
            children_ids=["3_1", "3_2"],
            generation_method="llm_expand",
            reason="Improved momentum signal",
            created_at=now,
        )
        
        assert node.parent_id == "1_1"
        assert node.fitness == 0.95
        assert node.metrics["sharpe"] == 1.25
        assert node.state == ToTNodeState.EVALUATED
        assert len(node.children_ids) == 2
        assert node.generation_method == "llm_expand"
        assert "momentum" in node.reason
    
    def test_node_state_enum_values(self):
        """Test that all expected states exist in enum."""
        expected_states = [
            "expanding", "evaluated", "pruned", "survivor", "leaf"
        ]
        actual_states = [state.value for state in ToTNodeState]
        
        for state in expected_states:
            assert state in actual_states


# ==============================================================================
# Test Class: ToTConfig
# ==============================================================================

class TestToTConfig:
    """Tests for ToTConfig configuration class."""
    
    def test_default_config_values(self):
        """Test default configuration values."""
        config = ToTConfig()
        
        assert config.max_depth == 3
        assert config.branch_factor == 4
        assert config.top_k_survivors == 2
        assert config.accept_threshold == 0.0
        assert config.llm_expand_ratio == 0.5
        assert config.mutation_ratio == 0.3
        assert config.crossover_ratio == 0.2
        assert config.timeout_seconds == 300
        assert config.max_total_nodes == 50
        assert config.enforce_diversity is True
        assert config.dedup_similarity_threshold == 0.95
    
    def test_custom_config_values(self, custom_config):
        """Test custom configuration values."""
        assert custom_config.max_depth == 4
        assert custom_config.branch_factor == 6
        assert custom_config.top_k_survivors == 3
        assert custom_config.accept_threshold == 0.1
        assert custom_config.timeout_seconds == 600
        assert custom_config.max_total_nodes == 100
    
    def test_config_ratios_sum_to_one(self, default_config):
        """Test that expand ratios sum to approximately 1.0."""
        total_ratio = (
            default_config.llm_expand_ratio +
            default_config.mutation_ratio +
            default_config.crossover_ratio
        )
        
        assert abs(total_ratio - 1.0) < 0.01


# ==============================================================================
# Test Class: ToTSearchResult
# ==============================================================================

class TestToTSearchResult:
    """Tests for ToTSearchResult data structure."""
    
    def test_result_with_best_node(self, sample_node):
        """Test result when best node exists."""
        result = ToTSearchResult(
            best_node=sample_node,
            total_nodes_explored=10,
            total_depth_reached=3,
            search_duration_sec=45.5,
        )
        
        assert result.get_best_expression() == sample_node.expression
        assert result.get_best_fitness() == sample_node.fitness
        assert result.total_nodes_explored == 10
    
    def test_result_without_best_node(self):
        """Test result when no best node (empty search)."""
        result = ToTSearchResult()
        
        assert result.get_best_expression() is None
        assert result.get_best_fitness() == 0.0
    
    def test_result_statistics_tracking(self):
        """Test that statistics are tracked correctly."""
        result = ToTSearchResult(
            nodes_per_depth={1: 1, 2: 4, 3: 8},
            survival_rate_per_depth={2: 1.0, 3: 0.5},
        )
        
        assert len(result.nodes_per_depth) == 3
        assert result.nodes_per_depth[2] == 4
        assert result.survival_rate_per_depth[3] == 0.5


# ==============================================================================
# Test Class: ToTSearchStrategy Initialization
# ==============================================================================

class TestToTSearchStrategyInit:
    """Tests for ToTSearchStrategy initialization."""
    
    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        strategy = ToTSearchStrategy()
        
        assert strategy.config is not None
        assert strategy.config.max_depth == 3
        assert len(strategy._nodes) == 0
        assert strategy._node_counter == 0
    
    def test_init_with_custom_config(self, custom_config):
        """Test initialization with custom configuration."""
        strategy = ToTSearchStrategy(config=custom_config)
        
        assert strategy.config.max_depth == 4
        assert strategy.config.branch_factor == 6
    
    def test_from_dict_configuration(self):
        """Test creating instance from dictionary."""
        config_dict = {
            "max_depth": 5,
            "branch_factor": 8,
            "top_k_survivors": 4,
            "timeout_seconds": 120,
        }
        
        strategy = ToTSearchStrategy.from_dict(config_dict)
        
        assert strategy.config.max_depth == 5
        assert strategy.config.branch_factor == 8
        assert strategy.config.top_k_survivors == 4
        assert strategy.config.timeout_seconds == 120


# ==============================================================================
# Test Class: Dependency Injection
# ==============================================================================

class TestInitializeDependencies:
    """Tests for dependency injection mechanism."""
    
    def test_initialize_with_defaults(self, strategy):
        """Test initialization creates default dependencies."""
        strategy.initialize_dependencies()
        
        assert strategy._near_pass is not None
        assert strategy._llm_client is None
        assert strategy._prefilter is None
    
    def test_initialize_with_custom_dependencies(self, strategy):
        """Test initialization with custom injected dependencies."""
        mock_near_pass = MagicMock()
        mock_llm = MagicMock()
        mock_prefilter = MagicMock()
        
        strategy.initialize_dependencies(
            near_pass_improver=mock_near_pass,
            llm_client=mock_llm,
            prefilter=mock_prefilter,
        )
        
        assert strategy._near_pass is mock_near_pass
        assert strategy._llm_client is mock_llm
        assert strategy._prefilter is mock_prefilter
    
    @pytest.mark.asyncio
    async def test_error_without_initialization(self, strategy, sample_expression):
        """Test that search raises error without initialization."""
        with pytest.raises(RuntimeError, match="Dependencies not initialized"):
            await strategy.search(seed_expression=sample_expression)


# ==============================================================================
# Test Class: Node Creation
# ==============================================================================

class TestCreateNode:
    """Tests for node creation functionality."""
    
    def test_create_root_node(self, initialized_strategy, sample_expression):
        """Test creating root node (depth=1)."""
        node = initialized_strategy._create_node(
            expression=sample_expression,
            depth=1,
            method="seed",
        )
        
        assert node.depth == 1
        assert node.expression == sample_expression
        assert node.parent_id is None
        assert node.generation_method == "seed"
        assert node.node_id.startswith("1_")
    
    def test_create_child_node(self, initialized_strategy, sample_expression):
        """Test creating child node with parent reference."""
        parent = initialized_strategy._create_node(
            expression=sample_expression,
            depth=1,
            method="seed",
        )
        
        child = initialized_strategy._create_node(
            expression="modified_expr",
            depth=2,
            parent_id=parent.node_id,
            method="mutation",
            reason="Better performance",
        )
        
        assert child.depth == 2
        assert child.parent_id == parent.node_id
        assert child.generation_method == "mutation"
        assert child.reason == "Better performance"
        assert child.node_id.startswith("2_")
    
    def test_node_counter_increments(self, initialized_strategy):
        """Test that node counter increments correctly."""
        node1 = initialized_strategy._create_node(expression="a", depth=1)
        node2 = initialized_strategy._create_node(expression="b", depth=1)
        node3 = initialized_strategy._create_node(expression="c", depth=2)
        
        id1_num = int(node1.node_id.split('_')[1])
        id2_num = int(node2.node_id.split('_')[1])
        id3_num = int(node3.node_id.split('_')[1])
        
        assert id2_num > id1_num
        assert id3_num > id2_num


# ==============================================================================
# Test Class: Node Expansion
# ==============================================================================

class TestExpandNode:
    """Tests for node expansion strategies."""
    
    @pytest.mark.asyncio
    async def test_mutation_only_expansion(self, initialized_strategy, sample_expression):
        """Test expansion using only mutation (no LLM)."""
        node = initialized_strategy._create_node(
            expression=sample_expression,
            depth=1,
            method="seed",
        )
        node.fitness = 0.8
        
        children = await initialized_strategy._expand_node(node)
        
        assert len(children) > 0
        assert all(child.depth == 2 for child in children)
        assert all(child.parent_id == node.node_id for child in children)
    
    @pytest.mark.asyncio
    async def test_llm_expand_only(self, llm_strategy, sample_expression):
        """Test expansion using LLM only."""
        node = llm_strategy._create_node(
            expression=sample_expression,
            depth=1,
            method="seed",
        )
        node.fitness = 0.8
        
        children = await llm_strategy._expand_node(node)
        
        llm_children = [c for c in children if c.generation_method == "llm_expand"]
        assert len(llm_children) > 0
    
    @pytest.mark.asyncio
    async def test_crossover_expansion(self, llm_strategy, sample_expression):
        """Test crossover expansion with sibling nodes."""
        parent = llm_strategy._create_node(
            expression=sample_expression,
            depth=1,
            method="seed",
        )
        
        sibling = llm_strategy._create_node(
            expression="ts_decay_linear(group_neutralize(zscore(close/open), industry), 15)",
            depth=2,
            parent_id=parent.node_id,
            method="mutation",
        )
        
        parent.children_ids.append(sibling.node_id)
        llm_strategy._nodes[sibling.node_id] = sibling
        
        node = llm_strategy._create_node(
            expression=sample_expression,
            depth=2,
            parent_id=parent.node_id,
            method="mutation",
        )
        parent.children_ids.append(node.node_id)
        llm_strategy._nodes[node.node_id] = node
        
        children = await llm_strategy._crossover_expand(node, n_candidates=2)
        
        assert isinstance(children, list)
    
    @pytest.mark.asyncio
    async def test_mixed_strategy_expansion(self, llm_strategy, sample_expression):
        """Test expansion using mixed strategies (LLM + mutation + crossover)."""
        node = llm_strategy._create_node(
            expression=sample_expression,
            depth=1,
            method="seed",
        )
        node.fitness = 0.8
        
        children = await llm_strategy._expand_node(node)
        
        methods_used = set(c.generation_method for c in children)
        assert len(children) > 0
        assert len(methods_used) >= 1
    
    @pytest.mark.asyncio
    async def test_empty_expansion_handling(self, initialized_strategy):
        """Test handling of expansion that produces no candidates."""
        node = initialized_strategy._create_node(
            expression="very_simple",
            depth=1,
            method="seed",
        )
        
        with patch.object(initialized_strategy, '_mutation_expand', return_value=[]):
            children = await initialized_strategy._expand_node(node)
            
            assert len(children) == 0


# ==============================================================================
# Test Class: Survivor Selection
# ==============================================================================

class TestSelectSurvivors:
    """Tests for survivor selection logic."""
    
    def test_all_beat_threshold(self, initialized_strategy, sample_node):
        """Test selection when all candidates beat threshold."""
        candidates = [
            ("expr_a", 0.9),
            ("expr_b", 0.85),
            ("expr_c", 0.8),
            ("expr_d", 0.75),
        ]
        
        survivors = initialized_strategy._select_survivors(sample_node, candidates)
        
        assert len(survivors) <= initialized_strategy.config.top_k_survivors
        assert all(s.fitness > sample_node.fitness for s in survivors)
    
    def test_partial_beat_threshold(self, initialized_strategy, sample_node):
        """Test selection when only some candidates beat threshold."""
        candidates = [
            ("good_expr", 0.9),
            ("bad_expr_1", 0.5),
            ("bad_expr_2", 0.3),
            ("bad_expr_3", 0.2),
        ]
        
        survivors = initialized_strategy._select_survivors(sample_node, candidates)
        
        assert len(survivors) >= 1
        assert survivors[0].fitness >= 0.9
    
    def test_none_beat_fallback(self, initialized_strategy, sample_node):
        """Test fallback to best-of-node when none beat threshold."""
        sample_node.fitness = 0.99
        
        candidates = [
            ("poor_1", 0.5),
            ("poor_2", 0.3),
            ("poor_3", 0.1),
        ]
        
        survivors = initialized_strategy._select_survivors(sample_node, candidates)
        
        assert len(survivors) == 1
        assert survivors[0].fitness == 0.5
    
    def test_top_k_limit_enforced(self, initialized_strategy, sample_node):
        """Test that top_k limit is strictly enforced."""
        candidates = [(f"expr_{i}", 0.9 + i * 0.01) for i in range(10)]
        
        survivors = initialized_strategy._select_survivors(sample_node, candidates)
        
        assert len(survivors) <= initialized_strategy.config.top_k_survivors
    
    def test_tie_breaking(self, initialized_strategy, sample_node):
        """Test tie-breaking behavior for equal fitness candidates."""
        candidates = [
            ("expr_a", 0.9),
            ("expr_b", 0.9),
            ("expr_c", 0.9),
            ("expr_d", 0.9),
        ]
        
        survivors = initialized_strategy._select_survivors(sample_node, candidates)
        
        assert len(survivors) <= initialized_strategy.config.top_k_survivors
        assert all(s.fitness == 0.9 for s in survivors)


# ==============================================================================
# Test Class: Diversity Check
# ==============================================================================

class TestDiversityCheck:
    """Tests for diversity enforcement."""
    
    def test_duplicate_removal(self, initialized_strategy):
        """Test removal of duplicate/similar expressions."""
        candidates = [
            "ts_decay_linear(rank(close/volume), 10)",
            "ts_decay_linear(rank(close/volume), 10)",
            "ts_decay_linear(rank(close/volume), 10)",
            "ts_decay_linear(zscore(high/low), 20)",
        ]
        
        unique = initialized_strategy._check_diversity(candidates)
        
        assert len(unique) < len(candidates)
        assert len(unique) >= 2
    
    def test_family_diversity(self, initialized_strategy):
        """Test preservation of different field families."""
        candidates = [
            "rank(ts_delta(close, 10))",
            "zscore(ts_corr(volume, close, 20))",
            "scale(ts_regression(open, high, 15))",
            "group_neutralize(rank(low/market_cap), sector)",
        ]
        
        unique = initialized_strategy._check_diversity(candidates)
        
        assert len(unique) == 4
    
    def test_operator_diversity(self, initialized_strategy):
        """Test preservation of different operator combinations."""
        candidates = [
            "ts_decay_linear(rank(close), 10)",
            "ts_mean(zscore(volume), 20)",
            "ts_std_dev(scale(high-low), 15)",
            "ts_av_diff(group_neutralize(open, sector), 30)",
        ]
        
        unique = initialized_strategy._check_diversity(candidates)
        
        operators_used = []
        for expr in unique:
            ops = extract_expression_fingerprint(expr).split('_')[0]
            operators_used.append(ops)
        
        unique_operators = set(operators_used)
        assert len(unique_operators) >= 3
    
    def test_empty_input_handling(self, initialized_strategy):
        """Test handling of empty candidate list."""
        unique = initialized_strategy._check_diversity([])
        
        assert unique == []


# ==============================================================================
# Test Class: Full ToT Search Workflow
# ==============================================================================

class TestFullToTSearch:
    """Integration tests for complete ToT search workflow."""
    
    @pytest.mark.asyncio
    async def test_single_depth_search(self, initialized_strategy, sample_expression):
        """Test search with max_depth=1 (only root evaluation)."""
        initialized_strategy.config.max_depth = 1
        
        result = await initialized_strategy.search(
            seed_expression=sample_expression,
            initial_fitness=0.8,
        )
        
        assert result is not None
        assert result.best_node is not None
        assert result.total_depth_reached <= 1
        assert result.total_nodes_explored >= 1
    
    @pytest.mark.asyncio
    async def test_multi_depth_search(self, initialized_strategy, sample_expression):
        """Test search with multiple depths."""
        initialized_strategy.config.max_depth = 3
        initialized_strategy.config.branch_factor = 2
        
        result = await initialized_strategy.search(
            seed_expression=sample_expression,
            initial_fitness=0.7,
        )
        
        assert result is not None
        assert result.best_node is not None
        assert result.total_nodes_explored >= 1
        assert result.search_duration_sec >= 0
    
    @pytest.mark.asyncio
    async def test_early_termination_target_reached(self, initialized_strategy, sample_expression):
        """Test early termination when target fitness is reached."""
        result = await initialized_strategy.search(
            seed_expression=sample_expression,
            target_fitness=0.6,
            initial_fitness=0.8,
        )
        
        assert result is not None
        assert result.best_node.fitness >= 0.8
    
    @pytest.mark.asyncio
    async def test_max_depth_reached(self, initialized_strategy, sample_expression):
        """Test that search stops at max_depth."""
        initialized_strategy.config.max_depth = 2
        
        result = await initialized_strategy.search(
            seed_expression=sample_expression,
            initial_fitness=0.5,
        )
        
        max_depth_found = max(result.nodes_per_depth.keys()) if result.nodes_per_depth else 1
        assert max_depth_found <= initialized_strategy.config.max_depth
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, initialized_strategy, sample_expression):
        """Test that search respects timeout limit."""
        initialized_strategy.config.timeout_seconds = 1
        initialized_strategy.config.max_depth = 100
        
        start = time.time()
        result = await initialized_strategy.search(
            seed_expression=sample_expression,
            initial_fitness=0.5,
        )
        elapsed = time.time() - start
        
        assert result is not None
        assert elapsed < initialized_strategy.config.timeout_seconds + 2
    
    @pytest.mark.asyncio
    async def test_no_survivors_pruning(self, initialized_strategy):
        """Test handling when no candidates survive selection."""
        with patch.object(initialized_strategy, '_expand_node', return_value=[]):
            result = await initialized_strategy.search(
                seed_expression="simple_expr",
                initial_fitness=0.99,
            )
            
            assert result is not None
            assert result.total_nodes_explored >= 1
    
    @pytest.mark.asyncio
    async def test_global_best_tracking(self, initialized_strategy, sample_expression):
        """Test that global best node is properly tracked."""
        result = await initialized_strategy.search(
            seed_expression=sample_expression,
            initial_fitness=0.5,
        )
        
        assert result.best_node is not None
        all_fitnesses = [node.fitness for node in result.tree_nodes]
        
        if all_fitnesses:
            assert result.get_best_fitness() == max(all_fitnesses)


# ==============================================================================
# Test Class: Fingerprint and Diversity Utilities
# ==============================================================================

class TestFingerprintAndDiversity:
    """Tests for utility functions."""
    
    def test_fingerprint_extraction(self):
        """Test fingerprint extraction from expressions."""
        expr = "rank(ts_decay_linear(close/volume, 10))"
        fp = extract_expression_fingerprint(expr)
        
        assert "rank" in fp.lower()
        assert "ts_decay_linear" in fp.lower()
        assert "close" in fp or "volume" in fp
    
    def test_fingerprint_uniqueness(self):
        """Test that different expressions produce different fingerprints."""
        exprs = [
            "rank(close/volume)",
            "zscore(high-low)",
            "scale(ts_mean(open, 20))",
        ]
        
        fingerprints = [extract_expression_fingerprint(e) for e in exprs]
        unique_fps = set(fingerprints)
        
        assert len(unique_fps) == len(exprs)
    
    def test_diversity_computation(self):
        """Test diversity score calculation."""
        diverse_exprs = [
            "rank(ts_delta(close, 10))",
            "zscore(ts_corr(volume, returns, 20))",
            "scale(ts_regression(open, high, 15))",
        ]
        
        similar_exprs = [
            "rank(ts_delta(close, 10))",
            "rank(ts_delta(close, 15))",
            "rank(ts_delta(close, 20))",
        ]
        
        diverse_score = compute_expression_diversity(diverse_exprs)
        similar_score = compute_expression_diversity(similar_exprs)
        
        assert diverse_score > similar_score
    
    def test_diverse_subset_selection(self):
        """Test diverse subset selection algorithm."""
        candidates = [
            ("rank(close)", 0.9),
            ("rank(volume)", 0.88),
            ("zscore(high)", 0.85),
            ("zscore(low)", 0.82),
            ("scale(open)", 0.80),
            ("scale(close)", 0.78),
        ]
        
        selected = select_diverse_subset(candidates, k=3)
        
        assert len(selected) == 3
        selected_exprs = [expr for expr, _ in selected]
        selected_fps = [extract_expression_fingerprint(e) for e in selected_exprs]
        
        unique_fps = set(selected_fps)
        assert len(unique_fps) >= 2
    
    def test_edge_cases_empty_input(self):
        """Test edge cases with empty inputs."""
        assert compute_expression_diversity([]) == 1.0
        assert select_diverse_subset([], 5) == []
        assert select_diverse_subset([("a", 0.5)], 0) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
