"""Graph-Based Experience Database for Alpha Factor Mining.

Implements a directed graph knowledge base inspired by RD-Agent's CoSTEERKnowledgeBaseV2,
tailored for WorldQuant BRAIN alpha factor evolution cycles.

Core Design:
- Nodes represent: factor expressions, feedback metrics, improvement strategies, error patterns
- Edges represent: causal relationships (expression → feedback → improvement → new_expression)
- Similarity computation uses structural feature extraction (fields, operators, complexity)

Usage:
    from openalpha_brain.knowledge.graph_experience_db import GraphBasedExperienceDB

    db = GraphBasedExperienceDB("data/experience_graph.pkl")
    node_id = db.add_factor_expression(
        expression="ts_decay_linear(rank(volume), 20)",
        wq_feedback={"sharpe": 1.25, "fitness": 0.85, "turnover": 15.2},
        category="near_pass"
    )
    similar = db.query_similar_expressions("rank(close)", top_k=5)
"""

from __future__ import annotations

import logging
import os
import pickle
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openalpha_brain.utils.algo_logger import algo_log

logger = logging.getLogger(__name__)

FIELD_FAMILY_MAP = {
    "price": {"close", "open", "high", "low", "vwap", "midpoint", "returns"},
    "volume": {"volume", "amount", "sharesout", "vwap"},
    "fundamental": {"sales", "earnings", "assets", "capex", "ebitda", "book_value"},
    "technical": {"rsi", "macd", "adx", "atr", "bbands"},
    "analyst": {"estimates", "revisions", "surprise"},
}

OPERATOR_CATEGORIES = {
    "time_series": {
        "ts_mean",
        "ts_std_dev",
        "ts_sum",
        "ts_product",
        "ts_min",
        "ts_max",
        "ts_rank",
        "ts_delta",
        "ts_decay_linear",
        "ts_regression",
        "ts_av_diff",
        "ts_skewness",
        "ts_kurtosis",
        "ts_corr",
        "ts_covariance",
    },
    "cross_section": {
        "rank",
        "zscore",
        "scale",
        "normalize",
        "winsorize",
        "group_rank",
        "group_zscore",
        "group_neutralize",
        "group_mean",
    },
    "math": {"abs", "log", "sign", "sqrt", "power", "signed_power", "max", "min"},
    "logical": {"if_else", "when", "switch"},
}


@dataclass
class ExperienceNode:
    """Node in the experience graph.

    Attributes:
        node_id: Unique identifier
        node_type: Type of node (expression/feedback/improvement/error_pattern)
        content: Node payload (expression string or dict)
        features: Extracted structural features (for expression nodes)
        metadata: Additional attributes (timestamp, category, etc.)
        created_at: Unix timestamp
    """

    node_id: str = ""
    node_type: str = ""
    content: Any = None
    features: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    created_at: float = 0.0

    def __post_init__(self):
        if not self.node_id:
            self.node_id = uuid.uuid4().hex[:12]
        if not self.created_at:
            self.created_at = time.time()


@dataclass
class ExperienceEdge:
    """Directed edge in the experience graph.

    Attributes:
        source_id: Source node ID
        target_id: Target node ID
        edge_type: Relationship type (produces/leads_to/fixed_by/similar_to)
        weight: Edge weight (similarity score or confidence)
        metadata: Additional info
    """

    source_id: str = ""
    target_id: str = ""
    edge_type: str = ""
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)


class SimpleDiGraph:
    """Lightweight directed graph implementation using nested dicts.

    Avoids networkx dependency while providing essential graph operations.
    Structure: {node_id: ExperienceNode}
    Adjacency: {source_id: {target_id: ExperienceEdge}}
    """

    def __init__(self):
        self._nodes: dict[str, ExperienceNode] = {}
        self._adj: dict[str, dict[str, ExperienceEdge]] = {}
        self._pred: dict[str, dict[str, ExperienceEdge]] = {}

    def add_node(self, node: ExperienceNode) -> None:
        if node.node_id not in self._nodes:
            self._nodes[node.node_id] = node
            self._adj.setdefault(node.node_id, {})
            self._pred.setdefault(node.node_id, {})

    def add_edge(self, edge: ExperienceEdge) -> None:
        self.add_node(self._nodes.get(edge.source_id) or ExperienceNode(node_id=edge.source_id, node_type="unknown"))
        self.add_node(self._nodes.get(edge.target_id) or ExperienceNode(node_id=edge.target_id, node_type="unknown"))
        self._adj[edge.source_id][edge.target_id] = edge
        self._pred[edge.target_id][edge.source_id] = edge

    def get_node(self, node_id: str) -> ExperienceNode | None:
        return self._nodes.get(node_id)

    def get_successors(self, node_id: str) -> list[ExperienceNode]:
        return [self._nodes[tid] for tid in self._adj.get(node_id, {}) if tid in self._nodes]

    def get_predecessors(self, node_id: str) -> list[ExperienceNode]:
        return [self._nodes[sid] for sid in self._pred.get(node_id, {}) if sid in self._nodes]

    def get_edges_from(self, node_id: str, edge_type: str = None) -> list[ExperienceEdge]:
        edges = list(self._adj.get(node_id, {}).values())
        if edge_type:
            edges = [e for e in edges if e.edge_type == edge_type]
        return edges

    def get_nodes_by_type(self, node_type: str) -> list[ExperienceNode]:
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def size(self) -> int:
        return len(self._nodes)

    def all_nodes(self) -> list[ExperienceNode]:
        return list(self._nodes.values())

    def all_edges(self) -> list[ExperienceEdge]:
        edges = []
        for src_edges in self._adj.values():
            edges.extend(src_edges.values())
        return edges


class GraphBasedExperienceDB:
    """Directed graph-based experience database for alpha factor mining.

    Stores factor expressions, their evaluation feedback, and improvement outcomes
    as a graph structure enabling similarity queries and pattern matching.

    Architecture (inspired by RD-Agent CoSTEER):
        expression_node --[produces]--> feedback_node --[leads_to]--> improvement_node
                                                              |
                                                              |--[fixed_by]--> new_expression_node

    Features:
        - Structural feature extraction from WQ expressions
        - Jaccard-weighted similarity computation
        - Atomic persistence with automatic backups
        - Error pattern and success pattern retrieval
    """

    NODE_TYPES = {
        "expression": "expression",
        "feedback": "feedback",
        "improvement": "improvement",
        "error_pattern": "error_pattern",
        "success_pattern": "success_pattern",
    }

    EDGE_TYPES = {
        "produces": "produces",
        "leads_to": "leads_to",
        "fixed_by": "fixed_by",
        "similar_to": "similar_to",
        "belongs_to": "belongs_to",
    }

    CATEGORIES = ["general", "near_pass", "success", "fail", "noise"]

    def __init__(self, db_path: str = "data/experience_graph.pkl"):
        self.graph = SimpleDiGraph()
        self.node_counter = 0
        self.db_path = Path(db_path)
        self._backup_count = 3
        self._field_family_map = FIELD_FAMILY_MAP
        self._operator_categories = OPERATOR_CATEGORIES

    @algo_log()
    def add_factor_experience(
        self, expression: str, wq_feedback: dict, improvement_result: dict | None = None, category: str = "general"
    ) -> str:
        """Record a complete factor→feedback→improvement experience triplet.

        Args:
            expression: WQ BRAIN alpha expression string
            wq_feedback: Dict with keys {sharpe, fitness, turnover, checks}
            improvement_result: Optional dict with {strategy, new_expression, result}
            category: One of CATEGORIES (near_pass/success/fail/noise/general)

        Returns:
            The expression node ID

        Raises:
            ValueError: If expression is empty or invalid
        """
        if not expression or not isinstance(expression, str):
            raise ValueError("Expression must be a non-empty string")

        if category not in self.CATEGORIES:
            logger.warning("Unknown category '%s', defaulting to 'general'", category)
            category = "general"

        features = self._extract_features(expression)

        expr_node = ExperienceNode(
            node_type=self.NODE_TYPES["expression"],
            content=expression,
            features=features,
            metadata={"category": category},
        )
        self.graph.add_node(expr_node)
        self.node_counter += 1

        feedback_node = ExperienceNode(
            node_type=self.NODE_TYPES["feedback"],
            content=wq_feedback,
            metadata={"category": category},
        )
        self.graph.add_node(feedback_node)
        self.node_counter += 1

        expr_feedback_edge = ExperienceEdge(
            source_id=expr_node.node_id,
            target_id=feedback_node.node_id,
            edge_type=self.EDGE_TYPES["produces"],
        )
        self.graph.add_edge(expr_feedback_edge)

        if improvement_result:
            imp_node = ExperienceNode(
                node_type=self.NODE_TYPES["improvement"],
                content=improvement_result,
                metadata={"category": category},
            )
            self.graph.add_node(imp_node)
            self.node_counter += 1

            fb_imp_edge = ExperienceEdge(
                source_id=feedback_node.node_id,
                target_id=imp_node.node_id,
                edge_type=self.EDGE_TYPES["leads_to"],
            )
            self.graph.add_edge(fb_imp_edge)

            if "new_expression" in improvement_result:
                new_features = self._extract_features(improvement_result["new_expression"])
                new_expr_node = ExperienceNode(
                    node_type=self.NODE_TYPES["expression"],
                    content=improvement_result["new_expression"],
                    features=new_features,
                    metadata={
                        "category": category,
                        "parent_expression": expression,
                        "improved_from": expr_node.node_id,
                    },
                )
                self.graph.add_node(new_expr_node)
                self.node_counter += 1

                imp_new_edge = ExperienceEdge(
                    source_id=imp_node.node_id,
                    target_id=new_expr_node.node_id,
                    edge_type=self.EDGE_TYPES["fixed_by"],
                    metadata={"strategy": improvement_result.get("strategy", "unknown")},
                )
                self.graph.add_edge(imp_new_edge)

        logger.info(
            "[GRAPH-DB] Added experience: expr=%s category=%s nodes=%d",
            expression[:50],
            category,
            self.node_counter,
        )
        return expr_node.node_id

    @algo_log()
    def query_similar_experiences(
        self, current_expression: str, top_k: int = 5, min_similarity: float = 0.3
    ) -> list[dict]:
        """Query historical experiences with similar factor structures.

        Args:
            current_expression: The query expression to find similarities for
            top_k: Maximum number of results to return
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of dicts with keys:
                {expression, sharpe, fitness, turnover, improvement_strategy,
                 similarity, category, node_id}
        """
        if not current_expression:
            return []

        query_features = self._extract_features(current_expression)
        expr_nodes = self.graph.get_nodes_by_type(self.NODE_TYPES["expression"])

        scored = []
        for node in expr_nodes:
            if node.content == current_expression:
                continue

            sim = self._compute_similarity(query_features, node.features)
            if sim >= min_similarity:
                feedback = self._get_node_feedback(node.node_id)
                improvement = self._get_node_improvement(node.node_id)

                result = {
                    "expression": node.content,
                    "sharpe": feedback.get("sharpe") if feedback else None,
                    "fitness": feedback.get("fitness") if feedback else None,
                    "turnover": feedback.get("turnover") if feedback else None,
                    "checks": feedback.get("checks") if feedback else None,
                    "improvement_strategy": improvement.get("strategy") if improvement else None,
                    "new_expression": improvement.get("new_expression") if improvement else None,
                    "similarity": round(sim, 4),
                    "category": node.metadata.get("category", "general"),
                    "node_id": node.node_id,
                }
                scored.append((sim, result))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]

    @algo_log()
    def get_error_patterns(self, error_type: str = None) -> list[dict]:
        """Retrieve error patterns and their fixes from the knowledge base.

        Args:
            error_type: Optional filter by error type (e.g., 'high_turnover',
                       'low_fitness', 'low_sharpe'). If None, returns all patterns.

        Returns:
            List of dicts with keys:
                {error_type, original_expression, fix_strategy, new_expression,
                 success_rate, occurrence_count}
        """
        imp_nodes = self.graph.get_nodes_by_type(self.NODE_TYPES["improvement"])
        patterns = []

        for imp_node in imp_nodes:
            if not isinstance(imp_node.content, dict):
                continue

            strategy = imp_node.content.get("strategy", "")
            result = imp_node.content.get("result", {})
            new_expr = imp_node.content.get("new_expression", "")

            pred_nodes = self.graph.get_predecessors(imp_node.node_id)
            original_expr = None
            original_feedback = None
            for pred in pred_nodes:
                if pred.node_type == self.NODE_TYPES["feedback"]:
                    original_feedback = pred.content
                    grand_preds = self.graph.get_predecessors(pred.node_id)
                    for gp in grand_preds:
                        if gp.node_type == self.NODE_TYPES["expression"]:
                            original_expr = gp.content
                            break

            pattern_error_type = self._classify_error(original_feedback)
            if error_type and pattern_error_type != error_type:
                continue

            patterns.append(
                {
                    "error_type": pattern_error_type,
                    "original_expression": original_expr,
                    "fix_strategy": strategy,
                    "new_expression": new_expr,
                    "success": result.get("success", False),
                    "result_metrics": result.get("metrics", {}),
                    "node_id": imp_node.node_id,
                }
            )

        patterns.sort(key=lambda x: x.get("success", False), reverse=True)
        return patterns

    @algo_log()
    def get_successful_patterns(self, field_family: str = None) -> list[dict]:
        """Retrieve successful factor patterns, optionally filtered by field family.

        Args:
            field_family: Optional filter (price/volume/fundamental/technical/analyst).
                         If None, returns all successful patterns.

        Returns:
            List of dicts with keys:
                {expression, field_families, operators, sharpe, fitness,
                 complexity, category, node_id}
        """
        expr_nodes = self.graph.get_nodes_by_type(self.NODE_TYPES["expression"])

        successful = []
        for node in expr_nodes:
            category = node.metadata.get("category", "")

            if category not in ("success", "near_pass"):
                continue

            if field_family:
                families = node.features.get("field_families", set())
                if field_family not in families:
                    continue

            feedback = self._get_node_feedback(node.node_id)

            successful.append(
                {
                    "expression": node.content,
                    "field_families": list(node.features.get("field_families", set())),
                    "operators": node.features.get("operators", []),
                    "complexity": node.features.get("complexity", 0),
                    "has_neutralize": node.features.get("has_neutralize", False),
                    "has_decay": node.features.get("has_decay", False),
                    "decay_window": node.features.get("decay_window"),
                    "sharpe": feedback.get("sharpe") if feedback else None,
                    "fitness": feedback.get("fitness") if feedback else None,
                    "turnover": feedback.get("turnover") if feedback else None,
                    "category": category,
                    "node_id": node.node_id,
                }
            )

        successful.sort(key=lambda x: x.get("sharpe") or 0, reverse=True)
        return successful

    @algo_log()
    def record_cycle(self, cycle_data: dict) -> None:
        """Compatibility interface with EvolutionDatabase.record_cycle().

        Args:
            cycle_data: Dict containing:
                - expression: str
                - sharpe: float
                - fitness: float
                - turnover: float
                - status: str ('PASS'/'FAIL')
                - category: str (optional)
                - improvement: dict (optional)
        """
        expression = cycle_data.get("expression", "")
        if not expression:
            logger.warning("[GRAPH-DB] record_cycle: empty expression, skipping")
            return

        wq_feedback = {
            "sharpe": cycle_data.get("sharpe"),
            "fitness": cycle_data.get("fitness"),
            "turnover": cycle_data.get("turnover"),
            "checks": cycle_data.get("checks", []),
        }

        status = cycle_data.get("status", "FAIL")
        if status == "PASS":
            category = "success"
        elif cycle_data.get("category"):
            category = cycle_data["category"]
        else:
            sharpe = wq_feedback.get("sharpe") or 0
            fitness = wq_feedback.get("fitness") or 0
            if sharpe >= 1.0 and fitness < 1.0:
                category = "near_pass"
            elif sharpe < 0.5 and fitness < 0.5:
                category = "noise"
            else:
                category = "fail"

        improvement = cycle_data.get("improvement")

        self.add_factor_experience(
            expression=expression,
            wq_feedback=wq_feedback,
            improvement_result=improvement,
            category=category,
        )

    @algo_log()
    def get_improvement_suggestion(self, expression: str) -> dict | None:
        """Get improvement suggestion for an expression based on historical patterns.

        This method is designed to be used by A2 Prompt Injection module.

        Args:
            expression: The factor expression to get suggestions for

        Returns:
            Dict with keys:
                {suggested_strategies, similar_cases, error_diagnosis,
                 confidence_score} or None if no suggestions available
        """
        if not expression:
            return None

        similar = self.query_similar_experiences(expression, top_k=3, min_similarity=0.25)

        if not similar:
            return None

        features = self._extract_features(expression)
        error_diag = self._diagnose_potential_issues(features)

        strategies = set()
        cases = []
        for case in similar:
            if case.get("improvement_strategy"):
                strategies.add(case["improvement_strategy"])
            cases.append(
                {
                    "expression": case["expression"],
                    "similarity": case["similarity"],
                    "sharpe": case["sharpe"],
                    "outcome": case["category"],
                }
            )

        avg_similarity = sum(c["similarity"] for c in cases) / len(cases) if cases else 0

        return {
            "suggested_strategies": list(strategies)[:5],
            "similar_cases": cases[:3],
            "error_diagnosis": error_diag,
            "confidence_score": round(avg_similarity, 3),
            "total_similar_found": len(similar),
        }

    @algo_log()
    def save(self) -> None:
        """Persist graph to disk using atomic write with backup rotation."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            fd, temp_path = tempfile.mkstemp(
                suffix=".pkl.tmp",
                dir=str(self.db_path.parent),
            )
            os.close(fd)

            data = {
                "graph": self.graph,
                "node_counter": self.node_counter,
                "version": "1.0.0",
                "saved_at": time.time(),
            }

            with open(temp_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            self._rotate_backups()

            shutil.move(str(temp_path), str(self.db_path))

            logger.info(
                "[GRAPH-DB] Saved %d nodes to %s",
                self.graph.size(),
                self.db_path,
            )
        except OSError:
            if Path(temp_path).exists():
                Path(temp_path).unlink(missing_ok=True)
            raise

    @algo_log()
    def load(self) -> bool:
        """Load graph from disk.

        Returns:
            True if load succeeded, False otherwise
        """
        if not self.db_path.exists():
            logger.info("[GRAPH-DB] No existing DB at %s, starting fresh", self.db_path)
            return False

        try:
            with open(self.db_path, "rb") as f:
                data = pickle.load(f)

            if not isinstance(data, dict):
                logger.error("[GRAPH-DB] Invalid DB format at %s", self.db_path)
                return False

            self.graph = data.get("graph", SimpleDiGraph())
            self.node_counter = data.get("node_counter", 0)

            logger.info(
                "[GRAPH-DB] Loaded %d nodes from %s (version=%s)",
                self.graph.size(),
                self.db_path,
                data.get("version", "unknown"),
            )
            return True
        except OSError as e:
            logger.warning("[GRAPH-DB] Failed to load from %s: %s", self.db_path, e, exc_info=True)
            return False

    def get_stats(self) -> dict:
        """Return database statistics.

        Returns:
            Dict with total nodes, nodes by type, total edges, etc.
        """
        type_counts = {}
        for node_type in self.NODE_TYPES.values():
            count = len(self.graph.get_nodes_by_type(node_type))
            if count > 0:
                type_counts[node_type] = count

        category_counts = {}
        for node in self.graph.get_nodes_by_type(self.NODE_TYPES["expression"]):
            cat = node.metadata.get("category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        return {
            "total_nodes": self.graph.size(),
            "total_edges": len(self.graph.all_edges()),
            "nodes_by_type": type_counts,
            "expression_categories": category_counts,
            "db_path": str(self.db_path),
        }

    def _extract_features(self, expression: str) -> dict:
        """Extract structural features from a WQ alpha expression.

        Analyzes the expression to identify:
        - Used fields/data items
        - Applied operators/functions
        - Structural pattern (nested function calls)
        - Field family classification
        - Complexity metrics
        - Special patterns (neutralization, decay)

        Args:
            expression: WQ BRAIN expression string

        Returns:
            Dict of extracted features
        """
        if not expression:
            return {
                "fields": set(),
                "operators": [],
                "structure": "",
                "field_families": set(),
                "complexity": 0,
                "has_neutralize": False,
                "has_decay": False,
                "decay_window": None,
            }

        fields = self._extract_fields(expression)
        operators = self._extract_operators(expression)
        structure = self._extract_structure(expression)
        field_families = self._classify_field_families(fields)
        complexity = len(operators)

        has_neutralize = any(op in expression.lower() for op in ["group_neutralize", "group_zscore"])

        has_decay = "ts_decay_linear" in expression
        decay_window = None
        if has_decay:
            match = re.search(r"ts_decay_linear\([^,]+,\s*(\d+)\)", expression)
            if match:
                decay_window = int(match.group(1))

        return {
            "fields": fields,
            "operators": operators,
            "structure": structure,
            "field_families": field_families,
            "complexity": complexity,
            "has_neutralize": has_neutralize,
            "has_decay": has_decay,
            "decay_window": decay_window,
        }

    def _compute_similarity(self, features_a: dict, features_b: dict) -> float:
        """Compute weighted similarity between two feature sets.

        Uses Jaccard similarity for sets, with weighting:
        - Fields: 30% weight (most important for factor identity)
        - Operators: 25% weight (structural similarity)
        - Field families: 20% weight (domain similarity)
        - Structure pattern: 15% weight (template matching)
        - Special flags (decay/neutralize): 10% weight

        Args:
            features_a: First feature dict
            features_b: Second feature dict

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not features_a or not features_b:
            return 0.0

        weights = {
            "fields": 0.30,
            "operators": 0.25,
            "field_families": 0.20,
            "structure": 0.15,
            "special_flags": 0.10,
        }

        fields_a = features_a.get("fields", set())
        fields_b = features_b.get("fields", set())
        field_sim = self._jaccard(fields_a, fields_b) if (fields_a and fields_b) else 0.0

        ops_a = set(features_a.get("operators", []))
        ops_b = set(features_b.get("operators", []))
        op_sim = self._jaccard(ops_a, ops_b) if (ops_a and ops_b) else 0.0

        fam_a = features_a.get("field_families", set())
        fam_b = features_b.get("field_families", set())
        fam_sim = self._jaccard(fam_a, fam_b) if (fam_a and fam_b) else 0.0

        struct_a = features_a.get("structure", "")
        struct_b = features_b.get("structure", "")
        struct_sim = self._structure_similarity(struct_a, struct_b)

        special_sim = 0.0
        decay_match = features_a.get("has_decay") == features_b.get("has_decay") and features_a.get(
            "decay_window"
        ) == features_b.get("decay_window")
        neutralize_match = features_a.get("has_neutralize") == features_b.get("has_neutralize")
        if decay_match and neutralize_match:
            special_sim = 1.0
        elif decay_match or neutralize_match:
            special_sim = 0.5

        total = (
            weights["fields"] * field_sim
            + weights["operators"] * op_sim
            + weights["field_families"] * fam_sim
            + weights["structure"] * struct_sim
            + weights["special_flags"] * special_sim
        )

        return max(0.0, min(1.0, total))

    def _get_node_feedback(self, node_id: str) -> dict | None:
        """Get feedback data associated with an expression node."""
        successors = self.graph.get_successors(node_id)
        for succ in successors:
            if succ.node_type == self.NODE_TYPES["feedback"]:
                return succ.content
        return None

    def _get_node_improvement(self, node_id: str) -> dict | None:
        """Get improvement data associated with an expression node."""
        successors = self.graph.get_successors(node_id)
        for succ in successors:
            if succ.node_type == self.NODE_TYPES["feedback"]:
                imp_successors = self.graph.get_successors(succ.node_id)
                for imp in imp_successors:
                    if imp.node_type == self.NODE_TYPES["improvement"]:
                        return imp.content
        break_outer = False
        for succ in successors:
            if succ.node_type == self.NODE_TYPES["feedback"]:
                for imp_succ in self.graph.get_successors(succ.node_id):
                    if imp_succ.node_type == self.NODE_TYPES["improvement"]:
                        return imp_succ.content
                break_outer = True
            if break_outer:
                break
        return None

    def _classify_error(self, feedback: dict | None) -> str:
        """Classify error type from feedback metrics.

        Args:
            feedback: Feedback dict with sharpe/fitness/turnover

        Returns:
            Error type string
        """
        if not feedback:
            return "unknown"

        sharpe = feedback.get("sharpe") or 0
        fitness = feedback.get("fitness") or 0
        turnover = feedback.get("turnover") or 0

        if turnover > 25:
            return "high_turnover"
        if fitness < 0.5 and sharpe >= 1.0:
            return "low_fitness"
        if sharpe < 0.5:
            return "low_sharpe"
        if sharpe < 1.0 and fitness < 1.0:
            return "below_threshold"
        return "acceptable"

    def _diagnose_potential_issues(self, features: dict) -> list[str]:
        """Diagnose potential issues based on extracted features.

        Args:
            features: Feature dict from _extract_features()

        Returns:
            List of potential issue descriptions
        """
        issues = []

        if not features.get("has_decay"):
            issues.append("No temporal smoothing - may cause high turnover")

        if not features.get("has_neutralize"):
            issues.append("No sector/industry neutralization - may have exposure bias")

        complexity = features.get("complexity", 0)
        if complexity > 8:
            issues.append(f"High complexity ({complexity} operators) - overfitting risk")
        elif complexity < 2:
            issues.append(f"Low complexity ({complexity} operators) - may lack signal strength")

        if features.get("decay_window") and features["decay_window"] < 10:
            issues.append(f"Short decay window ({features['decay_window']}) - noisy signal")

        return issues

    def _rotate_backups(self) -> None:
        """Maintain rolling backup of recent database versions."""
        if not self.db_path.exists():
            return

        for i in range(self._backup_count - 1, 0, -1):
            src = self.db_path.with_suffix(f".pkl.bak{i}")
            dst = self.db_path.with_suffix(f".pkl.bak{i + 1}")
            if src.exists():
                shutil.move(str(src), str(dst))

        backup_1 = self.db_path.with_suffix(".pkl.bak1")
        if self.db_path.exists():
            shutil.copy2(str(self.db_path), str(backup_1))

    @staticmethod
    def _extract_fields(expression: str) -> set[str]:
        """Extract field names from expression.

        Identifies data field references like 'close', 'volume', etc.
        that are not wrapped in function calls.
        """
        func_pattern = r"\b([a-z_][a-z0-9_]*)\s*\("
        funcs = set(re.findall(func_pattern, expression, re.IGNORECASE))

        tokens = re.split(r"[^\w]", expression)
        fields = set()
        known_operators = {
            "ts_mean",
            "ts_std_dev",
            "ts_sum",
            "ts_product",
            "ts_min",
            "ts_max",
            "ts_rank",
            "ts_delta",
            "ts_decay_linear",
            "ts_regression",
            "ts_av_diff",
            "ts_skewness",
            "ts_kurtosis",
            "ts_corr",
            "ts_covariance",
            "ts_zscore",
            "rank",
            "zscore",
            "scale",
            "normalize",
            "winsorize",
            "group_rank",
            "group_zscore",
            "group_neutralize",
            "group_mean",
            "abs",
            "log",
            "sign",
            "sqrt",
            "power",
            "signed_power",
            "max",
            "min",
            "if_else",
            "when",
            "switch",
            "delay",
            "delta",
        }
        known_operators.update(funcs)

        for token in tokens:
            token_lower = token.lower().strip()
            if token_lower and token_lower not in known_operators and not token.isdigit() and len(token) > 1:
                fields.add(token_lower)

        common_fields = {
            "close",
            "open",
            "high",
            "low",
            "volume",
            "amount",
            "vwap",
            "returns",
            "sharesout",
            "sales",
            "earnings",
            "capex",
        }
        return fields & common_fields if fields else fields

    @staticmethod
    def _extract_operators(expression: str) -> list[str]:
        """Extract operator/function names from expression."""
        pattern = r"\b([a-z_][a-z0-9_]*)\s*\("
        return re.findall(pattern, expression, re.IGNORECASE)

    @staticmethod
    def _extract_structure(expression: str) -> str:
        """Extract structural template by replacing fields with placeholders."""
        structure = expression
        common_fields = [
            "close",
            "open",
            "high",
            "low",
            "volume",
            "amount",
            "vwap",
            "returns",
            "sharesout",
            "sales",
            "earnings",
            "capex",
        ]
        for field in sorted(common_fields, key=len, reverse=True):
            structure = re.sub(r"\b" + field + r"\b", "{FIELD}", structure, flags=re.IGNORECASE)

        nums = re.sub(r"\b\d+\b", "{N}", structure)
        return nums

    def _classify_field_families(self, fields: set[str]) -> set[str]:
        """Classify fields into domain families.

        Args:
            fields: Set of field names

        Returns:
            Set of family names (price/volume/fundamental/etc.)
        """
        families = set()
        for field_name in fields:
            for family, members in self._field_family_map.items():
                if field_name in members:
                    families.add(family)
                    break
        return families

    @staticmethod
    def _jaccard(set_a: set, set_b: set) -> float:
        """Compute Jaccard similarity coefficient between two sets.

        Formula: |A ∩ B| / |A ∪ B|
        """
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _structure_similarity(struct_a: str, struct_b: str) -> float:
        """Compute similarity between two structural templates.

        Simple approach: compare normalized templates character-by-character
        after basic normalization.
        """
        if not struct_a or not struct_b:
            return 0.0 if (struct_a or struct_b) else 1.0

        norm_a = re.sub(r"\s+", "", struct_a.lower())
        norm_b = re.sub(r"\s+", "", struct_b.lower())

        if norm_a == norm_b:
            return 1.0

        shorter = min(len(norm_a), len(norm_b))
        if shorter == 0:
            return 0.0

        matches = sum(1 for a, b in zip(norm_a, norm_b, strict=False) if a == b)
        return matches / max(len(norm_a), len(norm_b))


def create_graph_db(db_path: str = "data/experience_graph.pkl", auto_load: bool = True) -> GraphBasedExperienceDB:
    """Factory function to create and optionally load a GraphBasedExperienceDB instance.

    Args:
        db_path: Path to the database file
        auto_load: If True, attempt to load existing DB from disk

    Returns:
        Initialized GraphBasedExperienceDB instance
    """
    db = GraphBasedExperienceDB(db_path=db_path)
    if auto_load:
        db.load()
    return db
