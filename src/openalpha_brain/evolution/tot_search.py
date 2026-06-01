"""Tree-of-Thoughts (ToT) Tree Search Strategy for Alpha Factor Mining.

ToTSearchStrategy: Hierarchical tree-based search that explores factor space
through structured thought expansion, inspired by AlphaBench's ToTSearcher.

Architecture:
  ┌──────────────┐   ┌──────────────────┐   ┌─────────────┐
  │ LLM Expand   │   │ NearPass         │   │ Crossover    │
  │ (Semantic)   │   │ (Deterministic)  │   │ (Block A)    │
  └──────┬───────┘   └────────┬─────────┘   └──────┬───────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ↓
                   ┌──────────────────┐
                   │  ToTSearchStrategy│  ← This module
                   │  (Tree-of-Thoughts)│
                   └────────┬─────────┘
                            ↓
              ┌──────────────────────────────┐
              │  Tree Expansion → Evaluate →  │
              │  Select Survivors → Recurse   │
              └──────────────────────────────┘

Key differences from EASearchStrategy:
- EA: Population-based evolution (flat generations)
- ToT: Tree-based exploration (hierarchical depth)
- EA: Each generation replaces the population
- ToT: Preserves multiple paths through top-k selection

Performance targets:
  - Single tree expansion < 30s per level
  - Explore diverse operator families at each node
  - Find improved variant within max_depth levels

Usage:
    strategy = ToTSearchStrategy(config=ToTConfig(max_depth=3, branch_factor=4))
    strategy.initialize_dependencies(
        near_pass_improver=NearPassImprover(),
        llm_client=llm_client,
        prefilter=prefilter,
    )
    result = await strategy.search(
        seed_expression="ts_decay_linear(rank(close/volume), sector), 10)",
        target_fitness=1.25,
    )
    print(result.get_best_expression())
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum

import aiohttp

from openalpha_brain.monitoring.algorithm_telemetry import AlgorithmTelemetryCollector

logger = logging.getLogger(__name__)


class ToTNodeState(Enum):
    """State of a ToT tree node."""

    EXPANDING = "expanding"
    EVALUATED = "evaluated"
    PRUNED = "pruned"
    SURVIVOR = "survivor"
    LEAF = "leaf"


@dataclass
class ToTNode:
    """ToT tree node representing a factor expression in the search tree."""

    node_id: str
    expression: str
    depth: int
    parent_id: str | None = None
    fitness: float = 0.0
    metrics: dict = field(default_factory=dict)
    state: ToTNodeState = ToTNodeState.EXPANDING
    children_ids: list[str] = field(default_factory=list)
    generation_method: str = "seed"
    reason: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class ToTSearchResult:
    """Result of a ToT search operation."""

    best_node: ToTNode | None = None
    total_nodes_explored: int = 0
    total_depth_reached: int = 0
    search_duration_sec: float = 0.0
    nodes_per_depth: dict[int, int] = field(default_factory=dict)
    survival_rate_per_depth: dict[float, float] = field(default_factory=dict)
    tree_nodes: list[ToTNode] = field(default_factory=list)

    def get_best_expression(self) -> str | None:
        return self.best_node.expression if self.best_node else None

    def get_best_fitness(self) -> float:
        return self.best_node.fitness if self.best_node else 0.0


@dataclass
class ToTConfig:
    """Configuration for ToTSearchStrategy."""

    max_depth: int = 3
    branch_factor: int = 4
    top_k_survivors: int = 2
    accept_threshold: float = 0.0
    llm_expand_ratio: float = 0.5
    mutation_ratio: float = 0.3
    crossover_ratio: float = 0.2
    timeout_seconds: int = 300
    max_total_nodes: int = 50
    enforce_diversity: bool = True
    dedup_similarity_threshold: float = 0.95


def extract_expression_fingerprint(expr: str) -> str:
    """Extract expression fingerprint for deduplication.

    Extracts: main operators + field set + structure template.

    Args:
        expr: Factor expression string.

    Returns:
        Fingerprint string for similarity comparison.

    Examples:
        >>> extract_expression_fingerprint("rank(ts_decay_linear(close/volume, 10))")
        'rank_ts_decay_linear_close_volume'
    """
    normalized = re.sub(r"\s+", "", expr.lower())
    operators = sorted(set(re.findall(r"\b([a-z_]+)\(", normalized)))
    fields = sorted(set(re.findall(r"\b(close|open|high|low|volume|returns|market_cap)\b", normalized, re.I)))
    structure_hash = hashlib.md5(normalized.encode()).hexdigest()[:8]

    return f"{'_'.join(operators)}_{'_'.join(fields)}_{structure_hash}"


def compute_expression_diversity(expressions: list[str]) -> float:
    """Compute diversity score for a set of expressions (0-1).

    Based on average Jaccard distance between fingerprints.

    Args:
        expressions: List of factor expressions.

    Returns:
        Diversity score between 0.0 and 1.0.
    """
    n = len(expressions)
    if n < 2:
        return 1.0

    fingerprints = [extract_expression_fingerprint(expr) for expr in expressions]
    total_distance = 0.0
    pair_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            fp_i = set(fingerprints[i].split("_"))
            fp_j = set(fingerprints[j].split("_"))

            intersection = len(fp_i & fp_j)
            union = len(fp_i | fp_j)

            jaccard_sim = intersection / union if union > 0 else 0
            jaccard_dist = 1.0 - jaccard_sim

            total_distance += jaccard_dist
            pair_count += 1

    return total_distance / max(pair_count, 1)


def select_diverse_subset(candidates: list[tuple[str, float]], k: int) -> list[tuple[str, float]]:
    """Select diverse top-k subset from candidates using greedy algorithm.

    Args:
        candidates: List of (expression, fitness) tuples.
        k: Number of candidates to select.

    Returns:
        Diverse subset of size min(k, len(candidates)).
    """
    if not candidates or k <= 0:
        return []

    if len(candidates) <= k:
        return candidates.copy()

    sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    selected: list[tuple[str, float]] = [sorted_candidates[0]]
    selected_fingerprints = {extract_expression_fingerprint(selected[0][0])}

    remaining = sorted_candidates[1:]

    while len(selected) < k and remaining:
        best_idx = 0
        best_min_distance = -1

        for idx, (expr, _fitness) in enumerate(remaining):
            fp = extract_expression_fingerprint(expr)
            fp_set = set(fp.split("_"))

            min_distance = float("inf")
            for sel_fp in selected_fingerprints:
                sel_fp_set = set(sel_fp.split("_"))
                intersection = len(fp_set & sel_fp_set)
                union = len(fp_set | sel_fp_set)
                distance = 1.0 - (intersection / union if union > 0 else 0)
                min_distance = min(min_distance, distance)

            if min_distance > best_min_distance:
                best_min_distance = min_distance
                best_idx = idx

        selected.append(remaining.pop(best_idx))
        selected_fingerprints.add(extract_expression_fingerprint(selected[-1][0]))

    return selected


class HybridFactorJudge:
    """Hybrid evaluation: fast rule pre-filter + LLM quality judge.

    Two-tier system inspired by AlphaBench's FactorEval module:
    Tier 1: Rule-based pre-filter (fast, <5ms) — eliminates obviously bad expressions
    Tier 2: LLM quality judgment (slower, ~2-5s) — ranks remaining candidates

    Only triggers Tier 2 when:
    - Candidate passes Tier 1 threshold (>0.4)
    - LLM client is available
    - Evaluation budget permits (max N LLM calls per search)
    """

    _RULE_THRESHOLD = 0.4
    _LLM_TIMEOUT = 10

    _MOMENTUM_OPS = {"ts_delta", "ts_regression", "ts_rank", "ts_arg_max", "ts_arg_min"}
    _PRICE_FIELDS = {"close", "open", "high", "low"}
    _VOLUME_FIELDS = {"volume"}
    _TIME_SERIES_OPS = {
        "ts_delta",
        "ts_mean",
        "ts_std_dev",
        "ts_sum",
        "ts_product",
        "ts_decay_linear",
        "ts_regression",
        "ts_rank",
        "ts_corr",
        "ts_av_diff",
        "ts_skewness",
        "ts_kurtosis",
        "ts_min",
        "ts_max",
        "ts_arg_max",
        "ts_arg_min",
        "ts_median",
        "ts_moment",
    }
    _CROSS_SECTIONAL_OPS = {
        "rank",
        "zscore",
        "group_neutralize",
        "group_zscore",
        "scale",
        "group_rank",
        "group_mean",
        "group_median",
    }
    _NORMALIZATION_OPS = {"rank", "zscore", "group_neutralize", "group_zscore", "scale"}

    def __init__(self, llm_client=None, max_llm_judgments_per_search=15):
        self._llm = llm_client
        self._llm_call_count = 0
        self._max_llm_calls = max_llm_judgments_per_search
        self._cache: dict[str, float] = {}
        self._tel = AlgorithmTelemetryCollector.get_instance()

    def reset(self):
        """Reset LLM call budget and cache for new search."""
        self._llm_call_count = 0
        self._cache.clear()
        logger.info("[TOT-JUDGE] Reset: cache cleared, budget restored")

    async def evaluate(self, expression: str) -> float:
        """Main entry point: hybrid evaluation.

        Args:
            expression: Factor expression string.

        Returns:
            Quality score between 0.0 and 1.0.
        """
        eid = None
        try:
            fingerprint = hashlib.md5(expression.strip().lower().encode()).hexdigest()[:16]
            eid = await self._tel.record_enter("ToTSearch", cycle_id="unknown", expr_id=hash(expression) % 10000)
            t0 = time.perf_counter()

            if fingerprint in self._cache:
                cached_score = self._cache[fingerprint]
                logger.debug("[TOT-JUDGE] Cache hit for %s (score=%.3f)", fingerprint[:8], cached_score)
                ms = (time.perf_counter() - t0) * 1000
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    await self._tel.record_exit(
                        "ToTSearch",
                        eid,
                        metrics={"rule_score": 0, "llm_score": 0, "final_score": cached_score, "cache_hit": True},
                        duration_ms=ms,
                    )
                return cached_score

            rule_score, passes_threshold = self._rule_tier(expression)
            final_score = rule_score
            llm_score = 0.0

            if not passes_threshold or self._llm is None or self._llm_call_count >= self._max_llm_calls:
                self._cache[fingerprint] = final_score
                ms = (time.perf_counter() - t0) * 1000
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    await self._tel.record_exit(
                        "ToTSearch",
                        eid,
                        metrics={
                            "rule_score": round(rule_score, 3),
                            "llm_score": 0,
                            "final_score": round(final_score, 3),
                            "cache_hit": False,
                        },
                        duration_ms=ms,
                    )
                return final_score

            try:
                llm_score = await self._llm_tier(expression, rule_score)
                final_score = llm_score
            except (TimeoutError, aiohttp.ClientError, ValueError, json.JSONDecodeError) as e:
                logger.warning("[TOT-JUDGE] LLM tier failed, falling back to rule score: %s", e)
                final_score = rule_score

            self._cache[fingerprint] = final_score

            logger.info(
                "[TOT-JUDGE] Evaluated %s: rule=%.3f llm=%.3f final=%.3f",
                fingerprint[:8],
                rule_score,
                llm_score,
                final_score,
            )
            ms = (time.perf_counter() - t0) * 1000
            with contextlib.suppress(OSError, ValueError, RuntimeError):
                await self._tel.record_exit(
                    "ToTSearch",
                    eid,
                    metrics={
                        "rule_score": round(rule_score, 3),
                        "llm_score": round(llm_score, 3),
                        "final_score": round(final_score, 3),
                        "cache_hit": False,
                    },
                    duration_ms=ms,
                )
            return final_score
        except (ValueError, TypeError, OSError, RuntimeError, KeyError, AttributeError) as e:
            if eid:
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    await self._tel.record_error("ToTSearch", str(e), type(e).__name__)
            raise

    def _rule_tier(self, expression: str) -> tuple[float, bool]:
        """Fast rule-based scoring.

        Returns:
            (score, passes_threshold) tuple.
        """
        eid = None
        try:
            eid = self._tel.record_enter_sync("ToTSearch", cycle_id="unknown", expr_id=hash(expression) % 10000)
            t0 = time.perf_counter()
            expr = expression.strip()
            score = 0.50

            depth = expr.count("(")
            if 3 <= depth <= 8:
                score += 0.15
            elif depth > 8:
                score -= 0.10

            fields = set(
                re.findall(
                    r"\b(close|open|high|low|volume|returns|bookvalue|market_cap|sales)\b",
                    expr,
                    re.I,
                )
            )
            score += min(len(fields) * 0.08, 0.24)

            for op in self._NORMALIZATION_OPS:
                if rf"\b{op}\b" in expr:
                    score += 0.05
                    break

            if "ts_decay_linear" in expr:
                score += 0.05

            if len(expr) > 200:
                score -= 0.10
            elif len(expr) < 20:
                score -= 0.05

            score += self._coherence_score(expr)
            score += self._anti_pattern_penalty(expr)
            score += self._structural_balance_bonus(expr)

            score = max(0.0, min(1.0, score))
            result = (score, score > self._RULE_THRESHOLD)
            ms = (time.perf_counter() - t0) * 1000
            with contextlib.suppress(OSError, ValueError, RuntimeError):
                self._tel.record_exit_sync("ToTSearch", eid, metrics={"rule_score": round(score, 3)}, duration_ms=ms)
            return result
        except (ValueError, TypeError, OSError, RuntimeError, KeyError, AttributeError) as e:
            if eid:
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    self._tel.record_error_sync("ToTSearch", str(e), type(e).__name__)
            raise

    def _coherence_score(self, expr: str) -> float:
        """Operator-field coherence: momentum ops should use price fields."""
        bonus = 0.0
        has_momentum_op = any(op in expr for op in self._MOMENTUM_OPS)
        has_price_field = any(f in expr for f in self._PRICE_FIELDS)

        if has_momentum_op and not has_price_field:
            bonus -= 0.08
        elif has_momentum_op and has_price_field:
            bonus += 0.06

        has_volume_only_momentum = re.search(r"ts_delta\s*\(\s*volume\b", expr)
        if has_volume_only_momentum:
            bonus -= 0.04

        return bonus

    def _anti_pattern_penalty(self, expr: str) -> float:
        """Penalize known bad patterns."""
        penalty = 0.0

        if re.search(r"ts_delta\s*\(\s*volume\b", expr) and not any(f in expr for f in self._PRICE_FIELDS):
            penalty -= 0.10

        if re.search(r"group_neutralize\s*\([^)]*,\s*market\b", expr, re.I):
            momentum_present = any(op in expr for op in self._MOMENTUM_OPS)
            if momentum_present:
                penalty -= 0.08

        signed_power_depth = 0
        for m in re.finditer(r"signed_power\s*\(", expr):
            inner = expr[m.end() :]
            signed_power_depth = max(signed_power_depth, 1 + inner.count("signed_power("))
        if signed_power_depth > 3:
            penalty -= 0.12

        decay_ops = ["ts_decay_linear", "ts_decay_exp_window"]
        has_decay = any(decay_op in expr for decay_op in decay_ops)
        if not has_decay and "(" in expr:
            penalty -= 0.05

        return penalty

    def _structural_balance_bonus(self, expr: str) -> float:
        """Bonus for good mix of time-series + cross-sectional operators."""
        ts_count = sum(1 for op in self._TIME_SERIES_OPS if op in expr)
        cs_count = sum(1 for op in self._CROSS_SECTIONAL_OPS if op in expr)

        if ts_count >= 1 and cs_count >= 1:
            return min(0.08, (min(ts_count, 3) * 0.02) + (min(cs_count, 2) * 0.03))
        return 0.0

    async def _llm_tier(self, expression: str, rule_score: float) -> float:
        """LLM quality judgment for promising candidates.

        Args:
            expression: Factor expression to judge.
            rule_score: Pre-computed rule-based score.

        Returns:
            LLM-adjusted score between 0.0 and 1.0.
        """
        eid = None
        try:
            eid = await self._tel.record_enter("ToTSearch", cycle_id="unknown", expr_id=hash(expression) % 10000)
            t0 = time.perf_counter()
            self._llm_call_count += 1

            depth = expression.count("(")
            fields = sorted(
                set(
                    re.findall(
                        r"\b(close|open|high|low|volume|returns|bookvalue|market_cap|sales)\b",
                        expression,
                        re.I,
                    )
                )
            )
            ops = sorted(set(re.findall(r"\b([a-z_]+)\(", expression.lower())))

            prompt = (
                "Rate this alpha factor expression for WQ BRAIN platform quality (0.0-1.0).\n\n"
                f"Expression: {expression}\n"
                f"Rule-based score: {rule_score:.3f} (pre-evaluation)\n"
                f"Structural: depth={depth}, fields={fields}, ops={ops}\n\n"
                "Consider:\n"
                "- Signal clarity: Is the economic logic clear and sound?\n"
                "- Structural quality: Appropriate operator nesting? Not over-complicated?\n"
                "- Field relevance: Fields match the strategy implied by operators?\n"
                "- Survival likelihood: Would this pass WQ's fitness formula?\n"
                "- Uniqueness: Is this a novel combination or a trivial variant?\n\n"
                "Reply ONLY: SCORE: <float> (one-line)"
            )

            try:
                response = await asyncio.wait_for(
                    self._llm_client_generate(prompt),
                    timeout=self._LLM_TIMEOUT,
                )
            except TimeoutError:
                logger.warning("[TOT-JUDGE] LLM call timed out (%ds)", self._LLM_TIMEOUT)
                ms = (time.perf_counter() - t0) * 1000
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    await self._tel.record_exit(
                        "ToTSearch",
                        eid,
                        metrics={"llm_score": 0, "final_score": round(rule_score, 3), "timed_out": True},
                        duration_ms=ms,
                    )
                return rule_score

            parsed = self._parse_llm_response(response)
            if parsed is None:
                logger.debug("[TOT-JUDGE] Could not parse LLM response, using rule score")
                ms = (time.perf_counter() - t0) * 1000
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    await self._tel.record_exit(
                        "ToTSearch",
                        eid,
                        metrics={"llm_score": 0, "final_score": round(rule_score, 3), "parse_failed": True},
                        duration_ms=ms,
                    )
                return rule_score

            blended = rule_score * 0.35 + parsed * 0.65
            logger.info(
                "[TOT-JUDGE] LLM judgment #%d: rule=%.3f llm=%.3f → blended=%.3f [%s…]",
                self._llm_call_count,
                rule_score,
                parsed,
                blended,
                expression[:40],
            )
            ms = (time.perf_counter() - t0) * 1000
            with contextlib.suppress(OSError, ValueError, RuntimeError):
                await self._tel.record_exit(
                    "ToTSearch",
                    eid,
                    metrics={"llm_score": round(parsed, 3), "final_score": round(blended, 3)},
                    duration_ms=ms,
                )
            return max(0.0, min(1.0, blended))
        except (ValueError, TypeError, OSError, RuntimeError, KeyError, AttributeError) as e:
            if eid:
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    await self._tel.record_error("ToTSearch", str(e), type(e).__name__)
            raise

    async def _llm_client_generate(self, prompt: str) -> str:
        """Thin wrapper around LLM client generate call."""
        if hasattr(self._llm, "generate") and callable(self._llm.generate):
            result = await self._llm.generate(prompt, temperature=0.3)
            return str(result).strip() if result else ""
        raise RuntimeError("[TOT-JUDGE] LLM client lacks generate() method")

    @staticmethod
    def _parse_llm_response(response: str) -> float | None:
        """Extract numeric score from LLM response text."""
        match = re.search(r"SCORE:\s*([0-9]*\.?[0-9]+)", response, re.I)
        if match:
            try:
                val = float(match.group(1))
                if 0.0 <= val <= 1.0:
                    return val
                else:
                    return max(0.0, min(1.0, val))
            except ValueError:
                pass

        numbers = re.findall(r"0?\.\d+|[01]\.0", response)
        for num_str in numbers:
            try:
                val = float(num_str)
                if 0.0 <= val <= 1.0:
                    return val
            except ValueError:
                continue

        return None


class ToTSearchStrategy:
    """Tree-of-Thoughts (ToT) tree search strategy for alpha factor mining.

    Inspired by AlphaBench's ToTSearcher, adapted for WQ BRAIN platform.

    Relationship with EASearchStrategy:
    - EA: Population evolution (flat, each generation replaces previous)
    - ToT: Tree expansion (hierarchical, preserves multiple paths)

    Use cases:
    - Deep exploration of complex factor combination spaces
    - Complement to EA when stuck in local optima
    - Fine-grained optimization of high-potential seed factors

    Core flow:
    1. Create root node from seed (depth=1)
    2. For each surviving node:
       a. Generate N candidates (LLM expand + mutation + crossover)
       b. Evaluate all candidates
       c. Select top_k survivors (> accept_threshold + parent fitness)
       d. Recursively expand to next depth
    3. Stop at max_depth or no survivors
    4. Return global best node
    """

    def __init__(self, config: ToTConfig | None = None):
        self.config = config or ToTConfig()
        self._nodes: dict[str, ToTNode] = {}
        self._node_counter: int = 0
        self._near_pass = None
        self._llm_client = None
        self._prefilter = None
        self._judge: HybridFactorJudge | None = None

    def initialize_dependencies(
        self,
        near_pass_improver=None,
        llm_client=None,
        prefilter=None,
    ):
        """Lazily inject dependencies (called from main loop).

        Args:
            near_pass_improver: NearPassImprover instance (or None to create default).
            llm_client: LLM client for semantic expansion (optional).
            prefilter: SignalQualityPreFilter for quick scoring (optional).
        """
        from openalpha_brain.evolution.near_pass_improver import NearPassImprover

        self._near_pass = near_pass_improver or NearPassImprover()
        self._llm_client = llm_client
        self._prefilter = prefilter
        if llm_client is not None:
            self._judge = HybridFactorJudge(
                llm_client=llm_client,
                max_llm_judgments_per_search=15,
            )
            logger.info("[TOT-JUDGE] HybridFactorJudge initialized (max %d LLM calls/search)", 15)
        else:
            self._judge = None
            logger.info("[TOT-JUDGE] No LLM client, rule-only mode")
        logger.info("[TOT] Dependencies initialized")

    @classmethod
    def from_dict(cls, config_dict: dict) -> ToTSearchStrategy:
        """Create instance from dictionary configuration.

        Args:
            config_dict: Dictionary with config parameters.

        Returns:
            Configured ToTSearchStrategy instance.
        """
        valid_keys = {k for k in dir(ToTConfig) if not k.startswith("_")}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
        config = ToTConfig(**filtered_config)
        return cls(config=config)

    async def search(
        self,
        seed_expression: str,
        target_fitness: float = 1.25,
        initial_fitness: float = 0.0,
        context: dict | None = None,
    ) -> ToTSearchResult:
        """Execute ToT tree search.

        Args:
            seed_expression: Seed factor expression.
            target_fitness: Target fitness (Sharpe ratio).
            initial_fitness: Known fitness of seed (if available).
            context: Extra context dict (session_id, etc.).

        Returns:
            ToTSearchResult with optimal node and statistics.

        Raises:
            RuntimeError: If dependencies not initialized.
            TimeoutError: If search exceeds timeout_seconds.
        """
        if self._near_pass is None:
            raise RuntimeError("[TOT] Dependencies not initialized. Call initialize_dependencies() first.")

        ctx = context or {}
        start_time = time.time()
        result = ToTSearchResult()

        logger.info(
            "[TOT] Starting search: seed='%s…' target_fit=%.2f max_depth=%d branch=%d",
            seed_expression[:50],
            target_fitness,
            self.config.max_depth,
            self.config.branch_factor,
        )

        root_node = self._create_node(
            expression=seed_expression,
            depth=1,
            method="seed",
        )
        root_node.fitness = initial_fitness
        root_node.state = ToTNodeState.EVALUATED

        self._nodes[root_node.node_id] = root_node
        result.total_nodes_explored = 1
        result.nodes_per_depth[1] = 1
        result.tree_nodes.append(root_node)

        result.best_node = root_node

        try:
            await asyncio.wait_for(
                self._expand_tree(root_node, target_fitness, ctx, result),
                timeout=self.config.timeout_seconds,
            )
        except TimeoutError:
            logger.warning("[TOT] Timeout reached (%ds)", self.config.timeout_seconds)

        result.search_duration_sec = time.time() - start_time
        result.total_depth_reached = max(result.nodes_per_depth.keys()) if result.nodes_per_depth else 1

        for depth, count in result.nodes_per_depth.items():
            if depth > 1:
                parent_count = result.nodes_per_depth.get(depth - 1, 1)
                survival_rate = count / max(parent_count * self.config.branch_factor, 1)
                result.survival_rate_per_depth[depth] = survival_rate

        logger.info(
            "[TOT] Search finished: best_fit=%.4f nodes=%d depth=%d time=%.1fs",
            result.get_best_fitness(),
            result.total_nodes_explored,
            result.total_depth_reached,
            result.search_duration_sec,
        )

        return result

    async def _expand_tree(
        self,
        node: ToTNode,
        target_fitness: float,
        context: dict,
        result: ToTSearchResult,
    ):
        """Recursively expand tree from given node.

        Args:
            node: Current node to expand.
            target_fitness: Target fitness threshold.
            context: Execution context.
            result: Accumulated search results.
        """
        if node.depth >= self.config.max_depth:
            node.state = ToTNodeState.LEAF
            logger.info("[TOT] Max depth reached at node %s", node.node_id)
            return

        if result.total_nodes_explored >= self.config.max_total_nodes:
            logger.warning("[TOT] Node limit reached (%d)", self.config.max_total_nodes)
            return

        if node.fitness >= target_fitness:
            logger.info("[TOT] Target reached at node %s (fit=%.4f)", node.node_id, node.fitness)
            return

        logger.info(
            "[TOT] Expanding node %s (depth=%d fit=%.4f)",
            node.node_id,
            node.depth,
            node.fitness,
        )

        children = await self._expand_node(node)

        if not children:
            node.state = ToTNodeState.LEAF
            return

        evaluated_children = []
        for child in children:
            if self._judge is not None:
                child.fitness = await self._judge.evaluate(child.expression)
            else:
                child.fitness = self._quick_evaluate(child.expression)
            child.state = ToTNodeState.EVALUATED
            evaluated_children.append(child)

            self._nodes[child.node_id] = child
            result.total_nodes_explored += 1
            result.tree_nodes.append(child)

            depth = child.depth
            result.nodes_per_depth[depth] = result.nodes_per_depth.get(depth, 0) + 1

            if child.fitness > result.best_node.fitness:
                result.best_node = child
                logger.info(
                    "[TOT] New best! node=%s fit=%.4f expr='%s…'",
                    child.node_id,
                    child.fitness,
                    child.expression[:50],
                )

        survivors = self._select_survivors(node, [(c.expression, c.fitness) for c in evaluated_children])

        node.children_ids = [s.node_id for s in survivors]

        for survivor in survivors:
            survivor.state = ToTNodeState.SURVIVOR

        for child in evaluated_children:
            if child.state != ToTNodeState.SURVIVOR:
                child.state = ToTNodeState.PRUNED

        if survivors:
            logger.info(
                "[TOT] Depth %d: %d/%d survivors (best=%.4f)",
                node.depth + 1,
                len(survivors),
                len(evaluated_children),
                max(s.fitness for s in survivors),
            )

            expand_tasks = [self._expand_tree(survivor, target_fitness, context, result) for survivor in survivors]
            await asyncio.gather(*expand_tasks, return_exceptions=True)
        else:
            logger.info("[TOT] No survivors at depth %d — pruning branch", node.depth + 1)

    async def _expand_node(self, node: ToTNode) -> list[ToTNode]:
        """Expand single node generating N candidate children.

        Strategy distribution:
        - 50% LLM semantic expansion (generate N/2 variants via LLM)
        - 30% NearPass mutation (deterministic variants)
        - 20% Crossover operations (exchange Block A with other survivors)

        Args:
            node: Parent node to expand.

        Returns:
            List of candidate child nodes.
        """
        n_candidates = self.config.branch_factor
        n_llm = max(1, int(n_candidates * self.config.llm_expand_ratio))
        n_mutation = max(1, int(n_candidates * self.config.mutation_ratio))
        n_crossover = max(0, n_candidates - n_llm - n_mutation)

        candidates: list[ToTNode] = []

        try:
            llm_candidates = await self._llm_expand(node.expression, n_llm)
            for expr in llm_candidates:
                child = self._create_node(
                    expression=expr,
                    depth=node.depth + 1,
                    parent_id=node.node_id,
                    method="llm_expand",
                )
                candidates.append(child)
        except (TimeoutError, aiohttp.ClientError, ValueError, json.JSONDecodeError) as e:
            logger.warning("[TOT] LLM expand failed: %s", e)

        try:
            mutation_candidates = self._mutation_expand(node.expression, n_mutation)
            for expr in mutation_candidates:
                child = self._create_node(
                    expression=expr,
                    depth=node.depth + 1,
                    parent_id=node.node_id,
                    method="mutation",
                )
                candidates.append(child)
        except (ValueError, SyntaxError, TypeError) as e:
            logger.warning("[TOT] Mutation expand failed: %s", e)

        if n_crossover > 0:
            try:
                crossover_candidates = await self._crossover_expand(node, n_crossover)
                for expr in crossover_candidates:
                    child = self._create_node(
                        expression=expr,
                        depth=node.depth + 1,
                        parent_id=node.node_id,
                        method="crossover",
                    )
                    candidates.append(child)
            except (ValueError, SyntaxError, TypeError) as e:
                logger.warning("[TOT] Crossover expand failed: %s", e)

        if self.config.enforce_diversity:
            unique_exprs = self._check_diversity([c.expression for c in candidates])
            candidates = [c for c in candidates if c.expression in unique_exprs]

        return candidates[:n_candidates]

    async def _llm_expand(self, expression: str, n_candidates: int) -> list[str]:
        """Use LLM to generate candidate expressions.

        Constructs AlphaBench-style instruction with current expression
        and metrics, requesting N diverse candidates with different
        operator skeletons/field families.

        Args:
            expression: Parent expression to expand from.
            n_candidates: Number of candidates to generate.

        Returns:
            List of generated expression strings.
        """
        if self._llm_client is None:
            logger.debug("[TOT] No LLM client available, skipping LLM expand")
            return []

        prompt = (
            "You are an expert quantitative alpha factor researcher.\n\n"
            f"Given this base factor expression:\n{expression}\n\n"
            f"Generate exactly {n_candidates} improved candidate factors.\n\n"
            "Requirements:\n"
            "- Return ONLY a JSON array of strings (expressions)\n"
            "- Each expression must be syntactically valid\n"
            "- Use different operator combinations for diversity:\n"
            "  * Momentum: ts_delta, ts_regression, ts_rank\n"
            "  * Mean-reversion: ts_mean, ts_av_diff, ts_corr\n"
            "  * Volatility-scaled: divide by ts_std_dev\n"
            "  * Volume-conditioned: multiply by volume signals\n"
            "- Vary lookback windows: short (5,10) vs long (20,30,60)\n"
            "- Include normalization: rank(), zscore(), group_neutralize()\n"
            "- Keep decay wrapper: ts_decay_linear(..., window)\n\n"
            'Output format: ["expr1", "expr2", ...]\n'
        )

        try:
            response = await self._llm_client.generate(prompt, temperature=0.9)
            if response and isinstance(response, str):
                import json

                text = response.strip()

                if text.startswith("["):
                    expressions = json.loads(text)
                    if isinstance(expressions, list):
                        valid_exprs = [str(expr).strip().strip("'\"") for expr in expressions if expr]
                        logger.info("[TOT] LLM generated %d candidates", len(valid_exprs))
                        return valid_exprs[:n_candidates]

                lines = text.split("\n")
                exprs = []
                for line in lines:
                    line = line.strip().strip(",\"'[]{}")
                    if line and len(line) > 5:
                        exprs.append(line)
                        if len(exprs) >= n_candidates:
                            break

                return exprs
        except (TimeoutError, aiohttp.ClientError, ValueError, json.JSONDecodeError) as e:
            logger.warning("[TOT] LLM call failed: %s", e)

        return []

    def _mutation_expand(self, expression: str, n_candidates: int) -> list[str]:
        """Generate candidates using deterministic mutations via NearPassImprover.

        Args:
            expression: Parent expression.
            n_candidates: Number of candidates to generate.

        Returns:
            List of mutated expression strings.
        """
        variants = []

        analysis = self._near_pass.analyze(
            sharpe=1.0,
            fitness=0.8,
        )

        det_variants = self._near_pass.generate_deterministic_variants(
            expression,
            analysis,
            max_variants=n_candidates * 2,
        )

        seen = {expression}
        for v in det_variants:
            if v.expression not in seen and len(variants) < n_candidates:
                variants.append(v.expression)
                seen.add(v.expression)

        import re as _re

        operator_swaps = [
            ("rank", "zscore"),
            ("zscore", "scale"),
            ("ts_mean", "ts_decay_linear"),
            ("ts_decay_linear", "ts_sum"),
        ]

        for old_op, new_op in operator_swaps:
            if len(variants) >= n_candidates:
                break
            pattern = rf"\b{_re.escape(old_op)}\b"
            mutated = _re.sub(pattern, new_op, expression, count=1)
            if mutated != expression and mutated not in seen:
                variants.append(mutated)
                seen.add(mutated)

        windows_to_try = [5, 15, 20, 30]
        for w in windows_to_try:
            if len(variants) >= n_candidates:
                break
            pattern = r"ts_decay_linear\(([^,]+),\s*\d+\)"
            mutated = _re.sub(pattern, f"ts_decay_linear(\\1, {w})", expression)
            if mutated != expression and mutated not in seen:
                variants.append(mutated)
                seen.add(mutated)

        while len(variants) < n_candidates:
            random_variant = expression.replace("rank(", "zscore(", 1)
            if random_variant != expression and random_variant not in seen:
                variants.append(random_variant)
                seen.add(random_variant)
            else:
                break

        return variants[:n_candidates]

    async def _crossover_expand(self, node: ToTNode, n_candidates: int) -> list[str]:
        """Generate candidates by crossing over with other surviving nodes.

        Exchanges Block A segments between current node and siblings.

        Args:
            node: Current node to expand.
            n_candidates: Number of candidates to generate.

        Returns:
            List of crossed-over expression strings.
        """
        candidates = []

        sibling_nodes = [
            self._nodes[sib_id]
            for sib_id in node.parent_id and self._nodes.get(node.parent_id, ToTNode("", "", 0)).children_ids or []
            if sib_id != node.node_id and sib_id in self._nodes
        ]

        if not sibling_nodes:
            return candidates

        for sibling in sibling_nodes[:n_candidates]:
            try:
                from openalpha_brain.evolution.ea_search import swap_block_a

                child_expr, _ = swap_block_a(node.expression, sibling.expression)
                if child_expr != node.expression:
                    candidates.append(child_expr)
            except (ValueError, AttributeError):
                continue

        return candidates[:n_candidates]

    def _evaluate_candidates(self, candidates: list[str]) -> list[tuple[str, float]]:
        """Quick evaluation of candidate list.

        Uses local PreFilter for fast scoring when available,
        otherwise falls back to heuristic evaluation.

        Args:
            candidates: List of expression strings to evaluate.

        Returns:
            List of (expression, fitness) tuples.
        """
        results = []
        for expr in candidates:
            fitness = self._quick_evaluate(expr)
            results.append((expr, fitness))
        return results

    def _select_survivors(
        self,
        parent: ToTNode,
        candidates: list[tuple[str, float]],
    ) -> list[ToTNode]:
        """Select top-k survivors from candidates.

        Selection criteria:
        1. fitness > parent.fitness + accept_threshold
        2. Sort by fitness descending, take top_k
        3. Fallback: if none qualify, keep best 1 (best-of-node fallback)

        Args:
            parent: Parent node.
            candidates: List of (expression, fitness) tuples.

        Returns:
            List of surviving ToTNode instances.
        """
        if not candidates:
            return []

        threshold = parent.fitness + self.config.accept_threshold
        qualified = [(expr, fitness) for expr, fitness in candidates if fitness > threshold]

        if qualified:
            qualified.sort(key=lambda x: x[1], reverse=True)
            top_survivors = qualified[: self.config.top_k_survivors]
        else:
            best_candidate = max(candidates, key=lambda x: x[1])
            top_survivors = [best_candidate]
            logger.info(
                "[TOT] Best-of-node fallback: fit=%.4f (threshold was %.4f)",
                best_candidate[1],
                threshold,
            )

        if self.config.enforce_diversity:
            top_survivors = select_diverse_subset(top_survivors, self.config.top_k_survivors)

        survivors = []
        for expr, fitness in top_survivors:
            existing_node = next(
                (node for node in self._nodes.values() if node.expression == expr),
                None,
            )
            if existing_node:
                survivors.append(existing_node)
            else:
                survivor_node = self._create_node(
                    expression=expr,
                    depth=parent.depth + 1,
                    parent_id=parent.node_id,
                    method="selected",
                )
                survivor_node.fitness = fitness
                survivor_node.state = ToTNodeState.SURVIVOR
                self._nodes[survivor_node.node_id] = survivor_node
                survivors.append(survivor_node)

        return survivors

    def _check_diversity(self, candidates: list[str]) -> list[str]:
        """Enforce diversity constraints (dedup + different field families).

        Uses simple feature fingerprinting to ensure no two candidates
        share identical field family + main operator combination.

        Args:
            candidates: List of candidate expressions.

        Returns:
            Filtered list with duplicates removed.
        """
        if not candidates:
            return []

        seen_fingerprints: set[str] = set()
        unique_candidates = []

        for expr in candidates:
            fp = extract_expression_fingerprint(expr)

            is_duplicate = False
            for seen_fp in seen_fingerprints:
                fp_parts = set(fp.split("_"))
                seen_parts = set(seen_fp.split("_"))

                common_operators = fp_parts & seen_parts
                common_fields = {p for p in fp_parts if p in ["close", "open", "high", "low", "volume"]} & {
                    p for p in seen_parts if p in ["close", "open", "high", "low", "volume"]
                }

                if len(common_operators) >= 2 and len(common_fields) >= 1:
                    similarity = len(common_operators) / max(len(fp_parts), len(seen_parts))
                    if similarity > self.config.dedup_similarity_threshold:
                        is_duplicate = True
                        break

            if not is_duplicate:
                seen_fingerprints.add(fp)
                unique_candidates.append(expr)

        return unique_candidates

    def _create_node(
        self,
        expression: str,
        depth: int,
        parent_id: str | None = None,
        method: str = "mutation",
        reason: str = "",
    ) -> ToTNode:
        """Create new tree node.

        Args:
            expression: Factor expression.
            depth: Tree depth (starting from 1).
            parent_id: Parent node ID (None for root).
            method: Generation method (seed/llm_expand/mutation/crossover).
            reason: LLM-generated improvement rationale.

        Returns:
            New ToTNode instance.
        """
        self._node_counter += 1
        node_id = f"{depth}_{self._node_counter}"

        return ToTNode(
            node_id=node_id,
            expression=expression,
            depth=depth,
            parent_id=parent_id,
            generation_method=method,
            reason=reason,
        )

    def _quick_evaluate(self, expression: str) -> float:
        """Fast local evaluation without WQ submission.

        Uses heuristic scoring based on expression characteristics:
        - Complexity bonus (moderate nesting preferred)
        - Field diversity (multiple data fields is good)
        - Normalization presence (rank/zscore/group_neutralize)
        - Penalty for extremely long expressions

        Delegates to PreFilter scoring if available.

        Args:
            expression: Factor expression to evaluate.

        Returns:
            Heuristic fitness score (not real Sharpe).
        """
        if self._prefilter is not None:
            try:
                result = self._prefilter.score(expression)
                if isinstance(result, (int, float)):
                    return float(result)
            except (OSError, ValueError, RuntimeError):
                pass

        score = 0.5
        depth = expression.count("(")
        if 3 <= depth <= 8:
            score += 0.15
        elif depth > 8:
            score -= 0.1

        fields = set(
            re.findall(
                r"\b(close|open|high|low|volume|returns|bookvalue|market_cap|sales)\b",
                expression,
                re.I,
            )
        )
        score += min(len(fields) * 0.08, 0.24)

        norm_ops = ["rank", "zscore", "group_neutralize", "group_zscore", "scale"]
        for op in norm_ops:
            if rf"\b{op}\b" in expression:
                score += 0.05
                break

        if "ts_decay_linear" in expression:
            score += 0.05

        if len(expression) > 200:
            score -= 0.1
        elif len(expression) < 20:
            score -= 0.05

        return max(0.0, min(1.0, score))

    def get_tree_stats(self) -> dict:
        """Return statistics about the current tree structure.

        Returns:
            Dictionary with keys: total_nodes, max_depth,
            nodes_by_state, average_branching_factor.
        """
        if not self._nodes:
            return {"total_nodes": 0, "max_depth": 0}

        states = {}
        for node in self._nodes.values():
            state_name = node.state.value
            states[state_name] = states.get(state_name, 0) + 1

        depths = [node.depth for node in self._nodes.values()]
        branching_factors = [len(node.children_ids) for node in self._nodes.values() if node.children_ids]

        return {
            "total_nodes": len(self._nodes),
            "max_depth": max(depths) if depths else 0,
            "nodes_by_state": states,
            "average_branching_factor": (sum(branching_factors) / len(branching_factors) if branching_factors else 0),
        }

    def reset(self):
        """Clear tree state for fresh search."""
        self._nodes.clear()
        self._node_counter = 0
        if self._judge is not None:
            self._judge.reset()
