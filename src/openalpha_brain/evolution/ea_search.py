"""Evolutionary Algorithm (EA) Search Strategy for Alpha Factor Mining.

EASearchStrategy: Population-based evolutionary search that bridges
Tier2 (deterministic mutation via NearPassImprover) and Tier3 (LLM-driven
semantic mutation).  Implements multi-strategy parallel mutation, block-A
crossover, elitist selection with diversity preservation, and adaptive
parameter tuning.

Architecture:
  ┌──────────────┐   ┌──────────────────┐   ┌─────────────┐
  │ NearPass     │   │ LLM Semantic     │   │ Operator    │
  │ Deterministic│   │ Mutation          │   │ Swap / Param│
  │ (Tier2)      │   │ (Tier3)           │   │ Tune        │
  └──────┬───────┘   └────────┬─────────┘   └──────┬──────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ↓
                   ┌──────────────────┐
                   │  EASearchStrategy │  ← This module
                   │  (Population EA)  │
                   └────────┬─────────┘
                            ↓
              ┌──────────────────────────────┐
              │  Crossover (Block A swap) +  │
              │  Elitist Selection + Diversity│
              └──────────────────────────────┘

Performance targets:
  - Single generation < 60s (dominated by WQ submission latency)
  - Find better-than-seed variant within 3 generations
  - Exploration efficiency > 3x random search

Usage:
    strategy = EASearchStrategy(population_size=8, max_generations=3)
    strategy.initialize_dependencies(
        near_pass_improver=NearPassImprover(),
        llm_client=llm_client,
        slot_manager=slot_manager,
    )
    best, history = await strategy.search(
        seed_expression="ts_decay_linear(rank(close/volume), sector), 10)",
        target_sharpe=1.25,
    )
"""
from __future__ import annotations

import asyncio
import logging
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class EAMutationType(Enum):
    """Mutation strategy types used by the EA."""
    NEAR_PASS_DETERMINISTIC = "near_pass_deterministic"
    LLM_SEMANTIC = "llm_semantic"
    OPERATOR_SWAP = "operator_swap"
    PARAMETER_TUNE = "parameter_tune"


@dataclass
class FactorIndividual:
    """Single individual in the EA population.

    Attributes:
        expression: Full factor expression string.
        fitness: Fitness score (Sharpe ratio or pre-filter score).
        generation: Generation number when this individual was created.
        parent_ids: UUIDs of parent individuals for lineage tracking.
        mutation_type: Which mutation strategy produced this individual.
        metadata: Arbitrary key-value store for extra info.
    """
    expression: str
    fitness: float = 0.0
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    mutation_type: EAMutationType = EAMutationType.NEAR_PASS_DETERMINISTIC
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.metadata.get("id", "")

    def __post_init__(self):
        if "id" not in self.metadata:
            self.metadata["id"] = uuid.uuid4().hex[:12]


@dataclass
class EAConfig:
    """Configuration for EASearchStrategy.

    Attributes:
        population_size: Number of individuals per generation (default 8).
        max_generations: Maximum number of evolutionary generations (default 3).
        elite_ratio: Fraction of population kept as elites (default 0.25).
        mutation_rate: Probability of applying mutation (default 0.6).
        crossover_rate: Probability of applying crossover (default 0.3).
        llm_mutation_prob: Among mutations, fraction that use LLM (default 0.3).
        diversity_threshold: Minimum population diversity score (default 0.7).
        timeout_seconds: Total wall-clock timeout for a single search (default 300).
    """
    population_size: int = 8
    max_generations: int = 3
    elite_ratio: float = 0.25
    mutation_rate: float = 0.6
    crossover_rate: float = 0.3
    llm_mutation_prob: float = 0.3
    diversity_threshold: float = 0.7
    timeout_seconds: int = 300


def extract_block_a(expression: str) -> str | None:
    """Extract Block A (signal segment) from a three-part expression.

    Parses the common pattern:
        ts_decay_linear(group_neutralize({BLOCK_A}, {GROUP}), {BLOCK_C})

    Args:
        expression: Full factor expression string.

    Returns:
        The inner signal expression (Block A), or None if not found.

    Examples:
        >>> extract_block_a("ts_decay_linear(group_neutralize(rank(close/volume), sector), 10)")
        'rank(close/volume)'
    """
    match = re.search(
        r"group_neutralize\((.+?),\s*[^)]+\)",
        expression,
    )
    if match:
        return match.group(1).strip()
    return None


def extract_block_c(expression: str) -> str | None:
    """Extract Block C (decay window parameter) from expression.

    Args:
        expression: Full factor expression string.

    Returns:
        The decay window value as string, or None if not found.

    Examples:
        >>> extract_block_c("ts_decay_linear(..., 10)")
        '10'
    """
    match = re.search(r"ts_decay_linear\([^,]+,\s*(\d+)\)", expression)
    if match:
        return match.group(1)
    outer_match = re.search(r",\s*(\d+)\)\s*$", expression)
    if outer_match:
        return outer_match.group(1)
    return None


def swap_block_a(expr_a: str, expr_b: str) -> tuple[str, str]:
    """Swap Block A between two expressions, producing two offspring.

    Both input expressions must contain group_neutralize(...) structure.

    Args:
        expr_a: First parent expression.
        expr_b: Second parent expression.

    Returns:
        Tuple of (child_a, child_b) with swapped Block A segments.

    Raises:
        ValueError: If either expression lacks a parseable Block A.
    """
    block_a = extract_block_a(expr_a)
    block_b = extract_block_a(expr_b)
    if not block_a or not block_b:
        raise ValueError("Both expressions must contain group_neutralize(Block_A, ...)")

    child_a = expr_a.replace(block_a, block_b, 1)
    child_b = expr_b.replace(block_b, block_a, 1)
    return child_a, child_b


def mutate_operator(expression: str, old_op: str, new_op: str) -> str | None:
    """Replace the first occurrence of an operator in the expression.

    Preserves all parameters and surrounding structure.

    Args:
        expression: Source factor expression.
        old_op: Operator name to replace (e.g., 'rank').
        new_op: Replacement operator name (e.g., 'zscore').

    Returns:
        Mutated expression string, or None if old_op not found.

    Examples:
        >>> mutate_operator("rank(close/volume)", "rank", "zscore")
        'zscore(close/volume)'
    """
    pattern = rf"\b{re.escape(old_op)}\b"
    new_expr = re.sub(pattern, new_op, expression, count=1)
    return new_expr if new_expr != expression else None


def tune_parameter(expression: str, param_name: str, delta: float) -> str | None:
    """Adjust a numeric parameter in the expression by delta.

    Supports decay_window and power parameters.

    Args:
        expression: Source factor expression.
        param_name: Parameter identifier ('decay_window' or 'power').
        delta: Amount to add to the current value.

    Returns:
        Adjusted expression, or None if parameter not found or invalid.

    Examples:
        >>> tune_parameter("ts_decay_linear(x, 10)", "decay_window", 5)
        'ts_decay_linear(x, 15)'
    """
    if param_name == "decay_window":
        pattern = r"ts_decay_linear\(([^,]+),\s*(\d+)\)"
        match = re.search(pattern, expression)
        if match:
            inner = match.group(1)
            old_val = int(match.group(2))
            new_val = old_val + int(delta)
            if new_val > 0:
                return expression[:match.start()] + f"ts_decay_linear({inner}, {new_val})" + expression[match.end():]
    elif param_name == "power":
        pattern = r"signed_power\(([^,]+),\s*([\d.]+)\)"
        match = re.search(pattern, expression)
        if match:
            base = match.group(1)
            old_power = float(match.group(2))
            new_power = round(old_power + delta, 2)
            if new_power > 0:
                return expression[:match.start()] + f"signed_power({base}, {new_power})" + expression[match.end():]
    return None


class EASearchStrategy:
    """Evolutionary Algorithm search strategy for alpha factor mining.

    Bridges Tier2 (deterministic NearPassImprover) and Tier3 (LLM semantic)
    mutation layers through population-based evolution with:

    - Parallel multi-strategy mutation (deterministic + LLM + operator swap + param tune)
    - Block-A crossover between parent expressions
    - Elitist selection with diversity preservation
    - Adaptive parameter tuning based on convergence state

    Performance goals:
      - Single generation < 60s (dominated by WQ submission)
      - Find improved variant within 3 generations
      - Exploration efficiency > 3x over random search

    All I/O operations are async-compatible. Dependencies are injected
    lazily via initialize_dependencies() to avoid circular imports.

    Attributes:
        config: EAConfig instance holding all hyper-parameters.
        _near_pass: NearPassImprover instance (injected).
        _llm_client: LLM client (injected, optional).
        _prefilter: SignalQualityPreFilter (injected, optional).
        _slot_manager: SlotManager for WQ submission (injected, optional).
    """

    def __init__(
        self,
        population_size: int = 8,
        max_generations: int = 3,
        elite_ratio: float = 0.25,
        mutation_rate: float = 0.6,
        crossover_rate: float = 0.3,
        llm_mutation_prob: float = 0.3,
        diversity_threshold: float = 0.7,
        timeout_seconds: int = 300,
    ):
        self.config = EAConfig(
            population_size=population_size,
            max_generations=max_generations,
            elite_ratio=elite_ratio,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            llm_mutation_prob=llm_mutation_prob,
            diversity_threshold=diversity_threshold,
            timeout_seconds=timeout_seconds,
        )
        self._near_pass = None
        self._llm_client = None
        self._prefilter = None
        self._slot_manager = None
        self._generation_stats: list[dict] = []

    def initialize_dependencies(
        self,
        near_pass_improver=None,
        llm_client=None,
        prefilter=None,
        slot_manager=None,
    ):
        """Lazily inject dependencies (called from main loop).

        Args:
            near_pass_improver: NearPassImprover instance (or None to create default).
            llm_client: LLM client for semantic mutations (optional).
            prefilter: SignalQualityPreFilter for quick scoring (optional).
            slot_manager: SlotManager for real WQ submissions (optional).
        """
        from openalpha_brain.evolution.near_pass_improver import NearPassImprover
        self._near_pass = near_pass_improver or NearPassImprover()
        self._llm_client = llm_client
        self._prefilter = prefilter
        self._slot_manager = slot_manager
        logger.info("[EA] Dependencies initialized")

    async def search(
        self,
        seed_expression: str,
        target_sharpe: float = 1.25,
        initial_sharpe: float = 0.0,
        context: dict | None = None,
    ) -> tuple[FactorIndividual, list[FactorIndividual]]:
        """Execute full EA search loop.

        Args:
            seed_expression: Starting factor expression (seed).
            target_sharpe: Target Sharpe ratio to achieve.
            initial_sharpe: Known Sharpe of the seed (if available).
            context: Extra context dict (generation, session_id, etc.).

        Returns:
            Tuple of (best_individual, all_population_history).

        Raises:
            RuntimeError: If dependencies not initialized.
            TimeoutError: If search exceeds timeout_seconds.
        """
        if self._near_pass is None:
            raise RuntimeError("[EA] Dependencies not initialized. Call initialize_dependencies() first.")

        ctx = context or {}
        start_time = time.time()
        all_history: list[FactorIndividual] = []

        logger.info(
            "[EA] Starting search: seed='%s…' target_shar=%.2f pop=%d gen=%d",
            seed_expression[:50], target_sharpe,
            self.config.population_size, self.config.max_generations,
        )

        population = await self._initialize_population(seed_expression, initial_sharpe)
        all_history.extend(population)

        best_individual = max(population, key=lambda ind: ind.fitness)

        for gen in range(1, self.config.max_generations + 1):
            elapsed = time.time() - start_time
            if elapsed > self.config.timeout_seconds:
                logger.warning("[EA] Timeout at generation %d (%.1fs)", gen, elapsed)
                break

            logger.info("[EA] === Generation %d/%d ===", gen, self.config.max_generations)
            ctx["generation"] = gen

            offspring: list[FactorIndividual] = []

            mutation_tasks = []
            for individual in population:
                if random.random() < self.config.mutation_rate:
                    mutation_tasks.append(self._mutate(individual))

            if mutation_tasks:
                mutation_results = await asyncio.gather(*mutation_tasks, return_exceptions=True)
                for result in mutation_results:
                    if isinstance(result, Exception):
                        logger.warning("[EA] Mutation error: %s", result)
                    elif isinstance(result, list):
                        offspring.extend(result)

            crossover_count = int(self.config.population_size * self.config.crossover_rate)
            selected_for_crossover = sorted(
                population, key=lambda x: x.fitness, reverse=True
            )[:max(2, crossover_count * 2)]
            for i in range(0, len(selected_for_crossover) - 1, 2):
                child = self._crossover(selected_for_crossover[i], selected_for_crossover[i + 1])
                if child is not None:
                    child.generation = gen
                    child.parent_ids = [selected_for_crossover[i].id, selected_for_crossover[i + 1].id]
                    offspring.append(child)

            evaluated_offspring: list[FactorIndividual] = []
            for child in offspring:
                try:
                    child.fitness = self._quick_evaluate(child.expression)
                    evaluated_offspring.append(child)
                except (ValueError, TypeError, SyntaxError, RuntimeError) as e:
                    logger.warning("[EA] Quick eval failed for '%s…': %s", child.expression[:40], e)

            population = self._select(population, evaluated_offspring)
            all_history.extend(evaluated_offspring)

            current_best = max(population, key=lambda ind: ind.fitness)
            if current_best.fitness > best_individual.fitness:
                best_individual = current_best
                logger.info(
                    "[EA] New best! fitness=%.4f expr='%s…'",
                    best_individual.fitness, best_individual.expression[:50],
                )

            diversity = self._compute_diversity(population)
            logger.info(
                "[EA] Gen %d complete: pop=%d best_fit=%.4f diversity=%.3f",
                gen, len(population), best_individual.fitness, diversity,
            )

            self._generation_stats.append({
                "generation": gen,
                "population_size": len(population),
                "offspring_produced": len(offspring),
                "best_fitness": best_individual.fitness,
                "diversity": diversity,
                "elapsed_sec": time.time() - start_time,
            })

            if best_individual.fitness >= target_sharpe:
                logger.info("[EA] Target reached at generation %d!", gen)
                break

            if diversity < self.config.diversity_threshold:
                logger.info("[EA] Low diversity (%.3f) — injecting random variants", diversity)
                random_variants = await self._inject_random_diversity(seed_expression, gen)
                population.extend(random_variants[:2])

        total_time = time.time() - start_time
        logger.info(
            "[EA] Search finished: best_fit=%.4f total_time=%.1fs gens=%d",
            best_individual.fitness, total_time, len(self._generation_stats),
        )
        return best_individual, all_history

    async def _initialize_population(
        self, seed_expression: str, initial_sharpe: float
    ) -> list[FactorIndividual]:
        """Initialize diverse starting population from seed.

        Strategy distribution:
          - 1x : Seed itself
          - 3x : NearPassImprover deterministic variants (different strategies)
          - 2x : Parameter tuning (decay_window, power values)
          - 1x : Operator replacement (rank→zscore, etc.)
          - 1x : Block A sign inversion (-signal → +signal)

        Args:
            seed_expression: The seed factor expression.
            initial_sharpe: Known fitness of the seed.

        Returns:
            Initial population list of FactorIndividual.
        """
        population: list[FactorIndividual] = []
        seed_individual = FactorIndividual(
            expression=seed_expression,
            fitness=initial_sharpe,
            generation=0,
            mutation_type=EAMutationType.NEAR_PASS_DETERMINISTIC,
        )
        population.append(seed_individual)

        analysis = self._near_pass.analyze(
            sharpe=max(initial_sharpe, 0.5),
            fitness=max(initial_sharpe * 0.8, 0.4),
        )
        deterministic_variants = self._near_pass.generate_deterministic_variants(
            seed_expression, analysis, max_variants=3,
        )
        for v in deterministic_variants[:3]:
            population.append(FactorIndividual(
                expression=v.expression,
                fitness=self._quick_evaluate(v.expression),
                generation=0,
                mutation_type=EAMutationType.NEAR_PASS_DETERMINISTIC,
                metadata={"mutation_description": v.mutation_description},
            ))

        for delta in [5, -5]:
            tuned = tune_parameter(seed_expression, "decay_window", delta)
            if tuned and tuned != seed_expression:
                population.append(FactorIndividual(
                    expression=tuned,
                    fitness=self._quick_evaluate(tuned),
                    generation=0,
                    mutation_type=EAMutationType.PARAMETER_TUNE,
                ))

        swapped = mutate_operator(seed_expression, "rank", "zscore")
        if swapped and swapped != seed_expression:
            population.append(FactorIndividual(
                expression=swapped,
                fitness=self._quick_evaluate(swapped),
                generation=0,
                mutation_type=EAMutationType.OPERATOR_SWAP,
            ))

        block_a = extract_block_a(seed_expression)
        if block_a and block_a.startswith("-"):
            inverted = seed_expression.replace(block_a, block_a[1:], 1)
        elif block_a:
            inverted = seed_expression.replace(block_a, f"-({block_a})", 1)
        else:
            inverted = None
        if inverted and inverted != seed_expression:
            population.append(FactorIndividual(
                expression=inverted,
                fitness=self._quick_evaluate(inverted),
                generation=0,
                mutation_type=EAMutationType.OPERATOR_SWAP,
                metadata={"mutation_description": "Block A sign inversion"},
            ))

        while len(population) < self.config.population_size:
            dup = FactorIndividual(
                expression=seed_expression,
                fitness=self._quick_evaluate(seed_expression) * random.uniform(0.9, 1.1),
                generation=0,
                mutation_type=EAMutationType.NEAR_PASS_DETERMINISTIC,
            )
            population.append(dup)

        logger.info(
            "[EA] Initialized population: %d individuals from seed",
            len(population),
        )
        return population[:self.config.population_size]

    async def _mutate(self, individual: FactorIndividual) -> list[FactorIndividual]:
        """Apply mutation to produce offspring variants.

        Dispatches to one of four strategies based on llm_mutation_prob:
          - NEAR_PASS_DETERMINISTIC: NearPassImprover (fast, <1s)
          - LLM_SEMANTIC: LLM-driven semantic mutation (~10s)
          - OPERATOR_SWAP: Single operator replacement (<0.1s)
          - PARAMETER_TUNE: Numeric parameter adjustment (<0.1s)

        All strategies run concurrently via asyncio.gather when called
        from search().

        Args:
            individual: Parent FactorIndividual to mutate.

        Returns:
            List of mutated offspring (may be empty).
        """
        offspring: list[FactorIndividual] = []
        roll = random.random()

        if roll < self.config.llm_mutation_prob and self._llm_client is not None:
            try:
                semantic_variants = await self._llm_semantic_mutate(individual)
                if semantic_variants:
                    offspring.extend(semantic_variants)
                else:
                    logger.warning("[EA] LLM semantic mutation returned empty — falling back to fast mutate")
                    offspring.extend(self._fast_mutate(individual))
            except (ValueError, TypeError, ConnectionError, asyncio.TimeoutError, OSError, RuntimeError) as e:
                logger.warning("[EA] LLM semantic mutation failed: %s", e)
                offspring.extend(self._fast_mutate(individual))
        else:
            offspring.extend(self._fast_mutate(individual))

        for child in offspring:
            child.generation = individual.generation + 1
            child.parent_ids = [individual.id]

        return offspring

    def _fast_mutate(self, individual: FactorIndividual) -> list[FactorIndividual]:
        """Fast non-LLM mutation (operator swap or parameter tune)."""
        offspring: list[FactorIndividual] = []
        expr = individual.expression

        if random.random() < 0.5:
            replacements = [
                ("rank", "zscore"),
                ("zscore", "scale"),
                ("ts_mean", "ts_decay_linear"),
                ("ts_decay_linear", "ts_mean"),
            ]
            for old_op, new_op in replacements:
                mutated = mutate_operator(expr, old_op, new_op)
                if mutated and mutated != expr:
                    offspring.append(FactorIndividual(
                        expression=mutated,
                        fitness=self._quick_evaluate(mutated),
                        mutation_type=EAMutationType.OPERATOR_SWAP,
                    ))
                    break
        else:
            params = [("decay_window", random.choice([5, -5, 10, -10])),
                      ("power", random.choice([0.5, -0.5, 1.0]))]
            param_name, delta = random.choice(params)
            mutated = tune_parameter(expr, param_name, delta)
            if mutated and mutated != expr:
                offspring.append(FactorIndividual(
                    expression=mutated,
                    fitness=self._quick_evaluate(mutated),
                    mutation_type=EAMutationType.PARAMETER_TUNE,
                ))

        analysis = self._near_pass.analyze(sharpe=individual.fitness, fitness=individual.fitness * 0.8)
        det_variants = self._near_pass.generate_deterministic_variants(expr, analysis, max_variants=1)
        for v in det_variants[:1]:
            if v.expression != expr:
                offspring.append(FactorIndividual(
                    expression=v.expression,
                    fitness=self._quick_evaluate(v.expression),
                    mutation_type=EAMutationType.NEAR_PASS_DETERMINISTIC,
                    metadata={"mutation_description": v.mutation_description},
                ))

        if not offspring:
            offspring.append(FactorIndividual(
                expression=expr,
                fitness=self._quick_evaluate(expr) * random.uniform(0.95, 1.05),
                mutation_type=EAMutationType.NEAR_PASS_DETERMINISTIC,
                metadata={"fallback": True},
            ))

        return offspring

    async def _llm_semantic_mutate(self, individual: FactorIndividual) -> list[FactorIndividual]:
        """LLM-driven semantic mutation (Tier3).

        Calls LLM to generate semantically meaningful variations
        of the factor expression. Falls back gracefully on failure.

        Args:
            individual: Parent individual to mutate via LLM.

        Returns:
            List of LLM-generated offspring.
        """
        if self._llm_client is None:
            return []

        prompt = (
            "You are a quantitative alpha factor researcher.\n"
            f"Given this factor expression: {individual.expression}\n"
            "Generate exactly 1 improved variant that:\n"
            "- Preserves the core signal logic\n"
            "- Changes operators/parameters to potentially improve Sharpe\n"
            "- Returns ONLY the new expression string, nothing else\n"
        )
        try:
            response = await self._llm_client.generate(prompt, temperature=0.8)
            if response and isinstance(response, str):
                new_expr = response.strip().strip("'\"")
                if new_expr and new_expr != individual.expression:
                    return [FactorIndividual(
                        expression=new_expr,
                        fitness=self._quick_evaluate(new_expr),
                        mutation_type=EAMutationType.LLM_SEMANTIC,
                        metadata={"llm_response": response[:200]},
                    )]
        except (ValueError, TypeError, ConnectionError, asyncio.TimeoutError, OSError, RuntimeError) as e:
            logger.warning("[EA] LLM mutation call failed: %s", e)
        return []

    def _crossover(
        self, parent_a: FactorIndividual, parent_b: FactorIndividual
    ) -> FactorIndividual | None:
        """Crossover operation: swap Block A between two parents.

        Parses the three-part structure:
            ts_decay_linear(group_neutralize({BLOCK_A}, sector), {BLOCK_C})
        Only exchanges BLOCK_A, keeping neutralization (BLOCK_B) and
        decay (BLOCK_C) unchanged.

        Args:
            parent_a: First parent individual.
            parent_b: Second parent individual.

        Returns:
            Child individual with swapped Block A, or None if parsing fails.
        """
        try:
            child_a_expr, _ = swap_block_a(parent_a.expression, parent_b.expression)
            return FactorIndividual(
                expression=child_a_expr,
                fitness=self._quick_evaluate(child_a_expr),
                parent_ids=[parent_a.id, parent_b.id],
                mutation_type=EAMutationType.NEAR_PASS_DETERMINISTIC,
                metadata={"crossover_from": f"{parent_a.id[:6]}+{parent_b.id[:6]}"},
            )
        except ValueError as e:
            logger.debug("[EA] Crossover failed: %s", e)
            return None

    def _select(
        self,
        population: list[FactorIndividual],
        evaluated_offspring: list[FactorIndividual],
    ) -> list[FactorIndividual]:
        """Selection: elitism + diversity-preserving roulette wheel.

        Steps:
          1. Merge parents and offspring
          2. Sort by fitness descending
          3. Keep top elite_count as elites
          4. Fill remaining slots via roulette with diversity penalty

        Args:
            population: Current parent population.
            evaluated_offspring: Newly evaluated offspring.

        Returns:
            Next-generation population of size population_size.
        """
        combined = population + evaluated_offspring
        combined.sort(key=lambda x: x.fitness, reverse=True)

        elite_count = max(1, int(self.config.population_size * self.config.elite_ratio))
        next_gen = combined[:elite_count]

        remaining_slots = self.config.population_size - elite_count
        if remaining_slots <= 0:
            return next_gen[:self.config.population_size]

        candidates = combined[elite_count:]
        if not candidates:
            return next_gen

        fitness_values = [max(c.fitness, 0.001) for c in candidates]
        total_fitness = sum(fitness_values)
        if total_fitness == 0:
            weights = [1.0 / len(candidates)] * len(candidates)
        else:
            weights = [f / total_fitness for f in fitness_values]

        existing_exprs = {ind.expression for ind in next_gen}
        attempts = 0
        while len(next_gen) < self.config.population_size and attempts < remaining_slots * 3:
            chosen = random.choices(candidates, weights=weights, k=1)[0]
            if chosen.expression not in existing_exprs:
                next_gen.append(chosen)
                existing_exprs.add(chosen.expression)
            attempts += 1

        while len(next_gen) < self.config.population_size:
            next_gen.append(random.choice(combined[:elite_count]))

        return next_gen[:self.config.population_size]

    def _compute_diversity(self, population: list[FactorIndividual]) -> float:
        """Compute population diversity as average pairwise edit distance ratio.

        Uses normalized Levenshtein-like distance on expression strings.
        Returns value in [0, 1] where 1 = maximally diverse.

        Args:
            population: Current population list.

        Returns:
            Diversity score between 0.0 and 1.0.
        """
        n = len(population)
        if n < 2:
            return 1.0

        total_dist = 0.0
        pair_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._edit_distance(population[i].expression, population[j].expression)
                max_len = max(len(population[i].expression), len(population[j].expression), 1)
                total_dist += dist / max_len
                pair_count += 1

        return total_dist / max(pair_count, 1)

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance between two strings."""
        if len(s1) < len(s2):
            return EASearchStrategy._edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]

    def _quick_evaluate(self, expression: str) -> float:
        """Fast local evaluation without WQ submission.

        Uses heuristic scoring based on expression characteristics:
          - Complexity bonus (moderate nesting preferred)
          - Field diversity (using multiple data fields is good)
          - Normalization presence (rank/zscore/group_neutralize)
          - Penalty for extremely long expressions

        If prefilter is injected, delegates to its scoring logic.

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
            except (ValueError, TypeError, SyntaxError):
                pass

        score = 0.5
        depth = expression.count("(")
        if 3 <= depth <= 8:
            score += 0.15
        elif depth > 8:
            score -= 0.1

        fields = set(re.findall(r'\b(close|open|high|low|volume|returns|bookvalue|market_cap|sales)\b', expression, re.I))
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

    async def _submit_and_eval(
        self, individual: FactorIndividual, priority: str = "normal"
    ) -> FactorIndividual:
        """Submit expression to WQ via SlotManager and wait for results.

        If SlotManager is not available, falls back to quick_evaluate.

        Args:
            individual: Individual to submit.
            priority: Submission priority ('normal', 'high', 'emergency').

        Returns:
            Updated individual with real fitness from WQ.
        """
        if self._slot_manager is None:
            individual.fitness = self._quick_evaluate(individual.expression)
            return individual

        try:
            task_id = await self._slot_manager.submit(
                expression=individual.expression,
                priority=priority,
            )
            logger.info("[EA] Submitted '%s…' as %s (priority=%s)", individual.expression[:40], task_id, priority)
            result = await self._slot_manager.wait_for_completion(task_id, timeout=120)
            if result and isinstance(result, dict):
                individual.fitness = float(result.get("sharpe", individual.fitness))
                individual.metadata["wq_result"] = result
            else:
                individual.fitness = self._quick_evaluate(individual.expression)
        except (ValueError, TypeError, ConnectionError, asyncio.TimeoutError, OSError, RuntimeError) as e:
            logger.warning("[EA] WQ submission failed: %s — using quick eval", e)
            individual.fitness = self._quick_evaluate(individual.expression)

        return individual

    async def _inject_random_diversity(self, seed_expression: str, generation: int) -> list[FactorIndividual]:
        """Inject random variants when population diversity drops too low.

        Generates diverse mutants to prevent premature convergence.

        Args:
            seed_expression: Original seed for reference.
            generation: Current generation number.

        Returns:
            List of diverse random variants.
        """
        variants: list[FactorIndividual] = []
        seen_exprs: set[str] = set()
        ops_to_try = ["ts_rank", "ts_av_diff", "ts_regression", "ts_corr"]
        fields_to_try = ["close/volume", "high-low", "returns", "open/close"]
        max_attempts = 10

        for _ in range(max_attempts):
            if len(variants) >= 3:
                break
            op = random.choice(ops_to_try)
            field = random.choice(fields_to_try)
            window = random.choice([5, 10, 20])
            random_expr = f"ts_decay_linear(group_neutralize({op}({field}), sector), {window})"
            if random_expr not in seen_exprs:
                seen_exprs.add(random_expr)
                variants.append(FactorIndividual(
                    expression=random_expr,
                    fitness=self._quick_evaluate(random_expr),
                    generation=generation,
                    mutation_type=EAMutationType.OPERATOR_SWAP,
                ))

        return variants

    def get_stats(self) -> list[dict]:
        """Return per-generation statistics collected during search.

        Returns:
            List of dicts with keys: generation, population_size,
            offspring_produced, best_fitness, diversity, elapsed_sec.
        """
        return self._generation_stats.copy()

    def reset_stats(self):
        """Clear accumulated generation statistics."""
        self._generation_stats.clear()
