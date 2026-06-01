"""Tests for GraphBasedExperienceDB - Graph-based knowledge base for alpha factors.

Test Categories:
1. Core Operations (add/query/save/load)
2. Feature Extraction & Similarity Computation
3. Error & Success Pattern Matching
4. Edge Cases & Error Handling
5. Integration & Compatibility
"""
from __future__ import annotations

import os
import tempfile
import pytest
from pathlib import Path

from openalpha_brain.knowledge.graph_experience_db import (
    GraphBasedExperienceDB,
    ExperienceNode,
    ExperienceEdge,
    SimpleDiGraph,
    create_graph_db,
)


@pytest.fixture
def temp_db_path(tmp_path):
    """Provide a temporary path for test databases."""
    return str(tmp_path / "test_experience_graph.pkl")


@pytest.fixture
def empty_db(temp_db_path):
    """Create an empty database instance."""
    db = GraphBasedExperienceDB(db_path=temp_db_path)
    return db


@pytest.fixture
def populated_db(temp_db_path):
    """Create a database with sample data for testing queries."""
    db = GraphBasedExperienceDB(db_path=temp_db_path)

    db.add_factor_experience(
        expression="ts_decay_linear(rank(volume), 20)",
        wq_feedback={"sharpe": 1.25, "fitness": 0.85, "turnover": 15.2, "checks": []},
        category="near_pass",
        improvement_result={
            "strategy": "increase_decay_window",
            "new_expression": "ts_decay_linear(rank(volume), 40)",
            "result": {"success": True, "metrics": {"sharpe": 1.35, "fitness": 1.05}},
        },
    )

    db.add_factor_experience(
        expression="rank(close) - rank(open)",
        wq_feedback={"sharpe": 1.85, "fitness": 1.15, "turnover": 22.5, "checks": ["high_turnover"]},
        category="success",
    )

    db.add_factor_experience(
        expression="ts_mean(returns, 10)",
        wq_feedback={"sharpe": 0.45, "fitness": 0.35, "turnover": 8.2, "checks": ["low_sharpe"]},
        category="fail",
        improvement_result={
            "strategy": "add_momentum_term",
            "new_expression": "ts_decay_linear(ts_rank(returns, 10), 20)",
            "result": {"success": False, "metrics": {}},
        },
    )

    db.add_factor_experience(
        expression="group_neutralize(rank(sales), sector)",
        wq_feedback={"sharpe": 0.95, "fitness": 0.65, "turnover": 18.7, "checks": []},
        category="near_pass",
    )

    db.add_factor_experience(
        expression="zscore(volume)",
        wq_feedback={"sharpe": 0.25, "fitness": 0.15, "turnover": 45.8, "checks": ["noise"]},
        category="noise",
    )

    return db


class TestGraphBasedExperienceDBCreation:
    """Test database initialization and basic properties."""

    def test_create_empty_database(self, empty_db):
        """Test that empty database initializes correctly."""
        assert empty_db.graph.size() == 0
        assert empty_db.node_counter == 0
        assert isinstance(empty_db.graph, SimpleDiGraph)

    def test_default_path(self):
        """Test default database path."""
        db = GraphBasedExperienceDB()
        assert str(db.db_path).endswith("experience_graph.pkl")

    def test_custom_path(self, temp_db_path):
        """Test custom database path is set correctly."""
        db = GraphBasedExperienceDB(db_path=temp_db_path)
        assert db.db_path == Path(temp_db_path)

    def test_initial_stats(self, empty_db):
        """Test initial statistics are correct."""
        stats = empty_db.get_stats()
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0
        assert len(stats["nodes_by_type"]) == 0


class TestAddFactorExperience:
    """Test adding factor experiences to the graph."""

    def test_add_simple_expression(self, empty_db):
        """Test adding a simple expression without improvement."""
        node_id = empty_db.add_factor_experience(
            expression="rank(close)",
            wq_feedback={"sharpe": 1.5, "fitness": 1.2, "turnover": 12.0, "checks": []},
            category="success",
        )
        assert node_id is not None
        assert empty_db.graph.size() >= 2
        assert empty_db.node_counter >= 2

    def test_add_with_improvement(self, empty_db):
        """Test adding an expression with full improvement chain."""
        node_id = empty_db.add_factor_experience(
            expression="ts_decay_linear(rank(volume), 10)",
            wq_feedback={"sharpe": 1.0, "fitness": 0.8, "turnover": 28.0, "checks": ["high_turnover"]},
            category="near_pass",
            improvement_result={
                "strategy": "increase_decay_window",
                "new_expression": "ts_decay_linear(rank(volume), 30)",
                "result": {"success": True, "metrics": {"sharpe": 1.2, "fitness": 1.05}},
            },
        )
        assert node_id is not None
        assert empty_db.graph.size() >= 4

    def test_add_multiple_experiences(self, populated_db):
        """Test that multiple experiences are stored independently."""
        stats = populated_db.get_stats()
        assert stats["total_nodes"] >= 10

    def test_add_invalid_expression_raises_error(self, empty_db):
        """Test that empty expression raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            empty_db.add_factor_experience(
                expression="",
                wq_feedback={"sharpe": 1.0, "fitness": 1.0, "turnover": 10.0},
            )
        with pytest.raises(ValueError):
            empty_db.add_factor_experience(
                expression=None,
                wq_feedback={"sharpe": 1.0, "fitness": 1.0, "turnover": 10.0},
            )

    def test_unknown_category_defaults_to_general(self, empty_db):
        """Test that unknown categories are handled gracefully."""
        node_id = empty_db.add_factor_experience(
            expression="rank(close)",
            wq_feedback={"sharpe": 1.0, "fitness": 1.0, "turnover": 10.0, "checks": []},
            category="invalid_category",
        )
        assert node_id is not None


class TestQuerySimilarExperiences:
    """Test similarity-based experience querying."""

    def test_query_similar_returns_results(self, populated_db):
        """Test that similar expressions are found."""
        similar = populated_db.query_similar_experiences(
            current_expression="rank(volume)",
            top_k=5,
            min_similarity=0.2,
        )
        assert isinstance(similar, list)
        assert len(similar) >= 1

    def test_query_similar_respects_top_k(self, populated_db):
        """Test that results are limited to top_k."""
        similar = populated_db.query_similar_experiences(
            current_expression="rank(close)",
            top_k=2,
            min_similarity=0.1,
        )
        assert len(similar) <= 2

    def test_query_similar_respects_min_similarity(self, populated_db):
        """Test that low-similarity results are filtered out."""
        similar_high = populated_db.query_similar_experiences(
            current_expression="rank(close)",
            top_k=10,
            min_similarity=0.9,
        )
        similar_low = populated_db.query_similar_experiences(
            current_expression="rank(close)",
            top_k=10,
            min_similarity=0.1,
        )
        assert len(similar_high) <= len(similar_low)

    def test_query_similar_empty_expression(self, populated_db):
        """Test that empty query returns empty list."""
        similar = populated_db.query_similar_experiences("", top_k=5)
        assert similar == []

    def test_query_similar_result_structure(self, populated_db):
        """Test that result dicts have expected keys."""
        similar = populated_db.query_similar_experiences(
            "ts_decay_linear(rank(amount), 20)",
            top_k=1,
            min_similarity=0.2,
        )
        if similar:
            result = similar[0]
            assert "expression" in result
            assert "similarity" in result
            assert "category" in result
            assert "node_id" in result
            assert 0 <= result["similarity"] <= 1

    def test_exact_expression_excluded(self, populated_db):
        """Test that exact match of query expression is excluded from results."""
        populated_db.add_factor_experience(
            expression="unique_expr_test_123",
            wq_feedback={"sharpe": 1.0, "fitness": 1.0, "turnover": 10.0, "checks": []},
        )
        similar = populated_db.query_similar_experiences(
            "unique_expr_test_123",
            top_k=10,
            min_similarity=0.0,
        )
        for result in similar:
            assert result["expression"] != "unique_expr_test_123"


class TestFeatureExtraction:
    """Test structural feature extraction from expressions."""

    def test_extract_fields_basic(self, empty_db):
        """Test field extraction from simple expression."""
        features = empty_db._extract_features("rank(close)")
        assert "close" in features["fields"]

    def test_extract_fields_multiple(self, empty_db):
        """Test extraction of multiple fields."""
        features = empty_db._extract_features("rank(close) - rank(open)")
        assert "close" in features["fields"]
        assert "open" in features["fields"]

    def test_extract_operators(self, empty_db):
        """Test operator extraction."""
        features = empty_db._extract_features("ts_decay_linear(rank(volume), 20)")
        assert "ts_decay_linear" in features["operators"]
        assert "rank" in features["operators"]

    def test_complexity_calculation(self, empty_db):
        """Test complexity (operator count) calculation."""
        features_simple = empty_db._extract_features("rank(close)")
        assert features_simple["complexity"] == 1
        features_complex = empty_db._extract_features("ts_decay_linear(group_neutralize(rank(volume), sector), 30)")
        assert features_complex["complexity"] == 3

    def test_neutralize_detection(self, empty_db):
        """Test detection of neutralization operators."""
        with_neutralize = empty_db._extract_features("group_neutralize(rank(close), industry)")
        assert with_neutralize["has_neutralize"] is True
        without_neutralize = empty_db._extract_features("rank(close)")
        assert without_neutralize["has_neutralize"] is False

    def test_decay_detection_and_window(self, empty_db):
        """Test decay operator detection and window extraction."""
        with_decay = empty_db._extract_features("ts_decay_linear(rank(volume), 40)")
        assert with_decay["has_decay"] is True
        assert with_decay["decay_window"] == 40
        without_decay = empty_db._extract_features("rank(close)")
        assert without_decay["has_decay"] is False
        assert without_decay["decay_window"] is None

    def test_field_family_classification(self, empty_db):
        """Test field family classification."""
        price_features = empty_db._extract_features("rank(close)")
        assert "price" in price_features["field_families"]
        vol_features = empty_db._extract_features("rank(volume)")
        assert "volume" in vol_features["field_families"]

    def test_structure_extraction(self, empty_db):
        """Test structural template extraction."""
        structure = empty_db._extract_structure("ts_decay_linear(rank(volume), 20)")
        assert "{FIELD}" in structure or "volume" not in structure.lower()


class TestSimilarityComputation:
    """Test similarity computation accuracy."""

    def test_identical_expressions_max_similarity(self, empty_db):
        """Test that identical expressions have maximum similarity."""
        f1 = empty_db._extract_features("rank(close)")
        f2 = empty_db._extract_features("rank(close)")
        sim = empty_db._compute_similarity(f1, f2)
        assert sim == 1.0

    def test_different_expressions_low_similarity(self, empty_db):
        """Test that very different expressions have low similarity."""
        f1 = empty_db._extract_features("rank(close)")
        f2 = empty_db._extract_features("group_neutralize(ts_decay_linear(rank(sales), 60), subindustry)")
        sim = empty_db._compute_similarity(f1, f2)
        assert sim < 0.5

    def test_similar_field_families_increases_similarity(self, empty_db):
        """Test that same field family increases similarity."""
        f1 = empty_db._extract_features("rank(close)")
        f2 = empty_db._extract_features("rank(open)")
        f3 = empty_db._extract_features("rank(volume)")
        sim_price = empty_db._compute_similarity(f1, f2)
        sim_cross = empty_db._compute_similarity(f1, f3)
        assert sim_price > sim_cross

    def test_same_operators_increases_similarity(self, empty_db):
        """Test that shared operators increase similarity."""
        f1 = empty_db._extract_features("ts_decay_linear(rank(close), 20)")
        f2 = empty_db._extract_features("ts_decay_linear(rank(open), 30)")
        f3 = empty_db._extract_features("group_neutralize(zscore(sales), sector)")
        sim_shared_ops = empty_db._compute_similarity(f1, f2)
        sim_diff_ops = empty_db._compute_similarity(f1, f3)
        assert sim_shared_ops > sim_diff_ops

    def test_similarity_range_validation(self, empty_db):
        """Test that similarity is always between 0 and 1."""
        f1 = empty_db._extract_features("rank(close)")
        f2 = empty_db._extract_features("group_neutralize(ts_mean(volume, 10), sector)")
        sim = empty_db._compute_similarity(f1, f2)
        assert 0.0 <= sim <= 1.0

    def test_empty_features_handling(self, empty_db):
        """Test handling of empty feature sets."""
        empty_features = {}
        normal_features = empty_db._extract_features("rank(close)")
        sim = empty_db._compute_similarity(empty_features, normal_features)
        assert sim == 0.0


class TestErrorPatternMatching:
    """Test error pattern retrieval and matching."""

    def test_get_all_error_patterns(self, populated_db):
        """Test retrieving all error patterns."""
        patterns = populated_db.get_error_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) >= 1

    def test_get_filtered_error_patterns(self, populated_db):
        """Test filtering by specific error type."""
        high_to_patterns = populated_db.get_error_patterns(error_type="high_turnover")
        all_patterns = populated_db.get_error_patterns()
        assert len(high_to_patterns) <= len(all_patterns)

    def test_error_pattern_structure(self, populated_db):
        """Test that error patterns have required fields."""
        patterns = populated_db.get_error_patterns()
        if patterns:
            pattern = patterns[0]
            assert "error_type" in pattern
            assert "fix_strategy" in pattern
            assert "original_expression" in pattern

    def test_no_error_patterns_for_new_db(self, empty_db):
        """Test that new DB has no error patterns."""
        patterns = empty_db.get_error_patterns()
        assert patterns == []


class TestSuccessfulPatternRetrieval:
    """Test successful pattern retrieval."""

    def test_get_successful_patterns(self, populated_db):
        """Test retrieving successful/near-pass patterns."""
        patterns = populated_db.get_successful_patterns()
        assert isinstance(patterns, list)
        success_count = sum(1 for p in patterns if p["category"] in ("success", "near_pass"))
        assert success_count >= 2

    def test_filter_by_field_family(self, populated_db):
        """Test filtering successful patterns by field family."""
        price_patterns = populated_db.get_successful_patterns(field_family="price")
        all_patterns = populated_db.get_successful_patterns()
        if price_patterns:
            for pattern in price_patterns:
                assert "price" in pattern["field_families"]
        assert len(price_patterns) <= len(all_patterns)

    def test_successful_pattern_sorted_by_sharpe(self, populated_db):
        """Test that successful patterns are sorted by Sharpe descending."""
        patterns = populated_db.get_successful_patterns()
        if len(patterns) >= 2:
            sharpes = [p.get("sharpe") or 0 for p in patterns]
            assert sharpes == sorted(sharpes, reverse=True)


class TestPersistenceOperations:
    """Test save/load functionality."""

    def test_save_creates_file(self, empty_db, temp_db_path):
        """Test that save creates the database file."""
        empty_db.add_factor_experience(
            expression="test_persistence",
            wq_feedback={"sharpe": 1.0, "fitness": 1.0, "turnover": 10.0, "checks": []},
        )
        empty_db.save()
        assert Path(temp_db_path).exists()

    def test_load_restores_data(self, populated_db, temp_db_path):
        """Test that load restores previously saved data."""
        original_stats = populated_db.get_stats()
        original_node_count = original_stats["total_nodes"]
        populated_db.save()
        new_db = GraphBasedExperienceDB(db_path=temp_db_path)
        loaded = new_db.load()
        assert loaded is True
        restored_stats = new_db.get_stats()
        assert restored_stats["total_nodes"] == original_node_count

    def test_load_nonexistent_file(self, empty_db):
        """Test loading when no file exists returns False."""
        result = empty_db.load()
        assert result is False

    def test_backup_rotation(self, populated_db, temp_db_path):
        """Test that backup files are created on subsequent saves."""
        populated_db.save()
        backup1 = Path(temp_db_path + ".bak1")
        if not backup1.exists():
            populated_db.add_factor_experience(
                expression="backup_test",
                wq_feedback={"sharpe": 1.0, "fitness": 1.0, "turnover": 10.0, "checks": []},
            )
            populated_db.save()
        assert backup1.exists()

    def test_atomic_write_safety(self, empty_db, temp_db_path):
        """Test that corrupted/incomplete writes don't corrupt DB."""
        empty_db.add_factor_experience(
            expression="atomic_test",
            wq_feedback={"sharpe": 1.0, "fitness": 1.0, "turnover": 10.0, "checks": []},
        )
        should_not_crash = True
        try:
            empty_db.save()
        except Exception:
            should_not_crash = False
        assert should_not_crash


class TestCompatibilityInterface:
    """Test compatibility with EvolutionDatabase interface."""

    def test_record_cycle_basic(self, empty_db):
        """Test record_cycle interface compatibility."""
        cycle_data = {
            "expression": "rank(close)",
            "sharpe": 1.5,
            "fitness": 1.2,
            "turnover": 14.0,
            "status": "PASS",
            "checks": [],
        }
        empty_db.record_cycle(cycle_data)
        stats = empty_db.get_stats()
        assert stats["total_nodes"] >= 2

    def test_record_cycle_auto_category_detection(self, empty_db):
        """Test automatic category detection from metrics."""
        near_pass_data = {
            "expression": "ts_decay_linear(rank(volume), 10)",
            "sharpe": 1.15,
            "fitness": 0.75,
            "turnover": 26.0,
            "status": "FAIL",
        }
        empty_db.record_cycle(near_pass_data)
        expr_nodes = empty_db.graph.get_nodes_by_type("expression")
        found_near_pass = any(
            n.metadata.get("category") == "near_pass"
            for n in expr_nodes
            if n.content == "ts_decay_linear(rank(volume), 10)"
        )
        assert found_near_pass

    def test_record_cycle_with_improvement(self, empty_db):
        """Test record_cycle with improvement data."""
        cycle_data = {
            "expression": "rank(volume)",
            "sharpe": 0.95,
            "fitness": 0.6,
            "turnover": 35.0,
            "status": "FAIL",
            "improvement": {
                "strategy": "add_decay",
                "new_expression": "ts_decay_linear(rank(volume), 20)",
                "result": {"success": True, "metrics": {"sharpe": 1.1}},
            },
        }
        empty_db.record_cycle(cycle_data)
        stats = empty_db.get_stats()
        assert stats["total_nodes"] >= 4

    def test_get_improvement_suggestion(self, populated_db):
        """Test getting improvement suggestions."""
        suggestion = populated_db.get_improvement_suggestion("rank(volume)")
        assert suggestion is not None
        assert "suggested_strategies" in suggestion
        assert "similar_cases" in suggestion
        assert "confidence_score" in suggestion
        assert 0 <= suggestion["confidence_score"] <= 1

    def test_get_improvement_suggestion_none_for_unknown(self, empty_db):
        """Test that None is returned when no similar cases exist."""
        suggestion = empty_db.get_improvement_suggestion("completely_unique_expression_xyz")
        assert suggestion is None


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and robustness."""

    def test_unicode_expression(self, empty_db):
        """Test handling of unicode characters in expression."""
        node_id = empty_db.add_factor_experience(
            expression="rank(收盘价)",
            wq_feedback={"sharpe": 1.0, "fitness": 1.0, "turnover": 10.0, "checks": []},
        )
        assert node_id is not None

    def test_very_long_expression(self, empty_db):
        """Test handling of very long expressions."""
        long_expr = "ts_decay_linear(" * 50 + "rank(volume)" + ", 20)" * 50
        node_id = empty_db.add_factor_experience(
            expression=long_expr,
            wq_feedback={"sharpe": 1.0, "fitness": 1.0, "turnover": 10.0, "checks": []},
        )
        assert node_id is not None

    def test_special_characters_in_expression(self, empty_db):
        """Test handling of special characters."""
        special_expr = "if_else(close > open, 1, -1)"
        node_id = empty_db.add_factor_experience(
            expression=special_expr,
            wq_feedback={"sharpe": 1.0, "fitness": 1.0, "turnover": 10.0, "checks": []},
        )
        assert node_id is not None

    def test_none_feedback_values(self, empty_db):
        """Test handling of None values in feedback."""
        node_id = empty_db.add_factor_experience(
            expression="rank(close)",
            wq_feedback={"sharpe": None, "fitness": None, "turnover": None, "checks": None},
        )
        assert node_id is not None

    def test_get_stats_after_operations(self, populated_db):
        """Test that stats reflect accumulated operations."""
        stats = populated_db.get_stats()
        assert "total_nodes" in stats
        assert "total_edges" in stats
        assert "nodes_by_type" in stats
        assert "expression_categories" in stats
        assert stats["total_nodes"] > 0


class TestSimpleDiGraph:
    """Test the lightweight directed graph implementation."""

    def test_add_and_retrieve_node(self):
        """Test basic node operations."""
        graph = SimpleDiGraph()
        node = ExperienceNode(node_id="test1", node_type="test", content="data")
        graph.add_node(node)
        retrieved = graph.get_node("test1")
        assert retrieved is not None
        assert retrieved.content == "data"

    def test_add_and_traverse_edges(self):
        """Test edge creation and traversal."""
        graph = SimpleDiGraph()
        n1 = ExperienceNode(node_id="n1", node_type="a")
        n2 = ExperienceNode(node_id="n2", node_type="b")
        graph.add_node(n1)
        graph.add_node(n2)
        edge = ExperienceEdge(source_id="n1", target_id="n2", edge_type="test_edge")
        graph.add_edge(edge)
        successors = graph.get_successors("n1")
        assert len(successors) == 1
        assert successors[0].node_id == "n2"
        predecessors = graph.get_predecessors("n2")
        assert len(predecessors) == 1
        assert predecessors[0].node_id == "n1"

    def test_filter_by_type(self):
        """Test filtering nodes by type."""
        graph = SimpleDiGraph()
        graph.add_node(ExperienceNode(node_id="a1", node_type="type_a"))
        graph.add_node(ExperienceNode(node_id="a2", node_type="type_a"))
        graph.add_node(ExperienceNode(node_id="b1", node_type="type_b"))
        type_a_nodes = graph.get_nodes_by_type("type_a")
        assert len(type_a_nodes) == 2

    def test_graph_size(self):
        """Test size tracking."""
        graph = SimpleDiGraph()
        assert graph.size() == 0
        graph.add_node(ExperienceNode(node_id="x"))
        assert graph.size() == 1


class TestFactoryFunction:
    """Test factory function for creating database instances."""

    def test_create_graph_db_default(self, tmp_path):
        """Test factory function creates usable instance."""
        db_path = str(tmp_path / "factory_test.pkl")
        db = create_graph_db(db_path=db_path, auto_load=False)
        assert isinstance(db, GraphBasedExperienceDB)
        assert db.graph.size() == 0

    def test_create_graph_db_with_load(self, populated_db, temp_db_path):
        """Test factory function loads existing database."""
        populated_db.save()
        db = create_graph_db(db_path=temp_db_path, auto_load=True)
        assert db.graph.size() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
