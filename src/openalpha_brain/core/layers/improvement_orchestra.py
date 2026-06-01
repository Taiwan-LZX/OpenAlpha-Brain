"""Layer 4: Improvement Orchestra — Unified improvement/improver integration hub.

This is the MOST IMPORTANT layer in OpenAlpha-Brain's 6-Layer architecture.
It encapsulates ALL improvement modules that were previously disconnected or bypassed,
providing a single entry point for post-evaluation improvement.

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │                  ImprovementOrchestra                        │
  │  ┌──────────────────┐  ┌─────────┐  ┌────────┐  ┌──────┐  │
  │  │FeedbackOrchestrator│  │EASearch │  │ExpDist │  │SemMut│  │
  │  │ (6 sub-improvers) │  │(EA strat)│  │(patterns│  │(LLM) │  │
  │  ├──────────────────┤  └─────────┘  └────────┘  └──────┘  │
  │  │ • DecisionEngine  │  ┌──────────────────────┐           │
  │  │ • ReflectionEng   │  │ QualityDiversity/QD   │           │
  │  │ • AdaptNeutralizer│  │ (MAP-Elites archive)  │           │
  │  │ • NearPassImprove │  └──────────────────────┘           │
  │  │ • FitnessBoost    │                                       │
  │  │ • TurnoverOpt     │                                       │
  │  │ • AntiOverfitDet  │                                       │
  │  └──────────────────┘                                       │
  └─────────────────────────────────────────────────────────────┘

Modules integrated:
  1. FeedbackOrchestrator — primary delegate with 6 sub-improvers + DecisionEngine
  2. EASearchStrategy — evolutionary algorithm strategy selector
  3. ExperienceDistiller — success/failure pattern extraction & retrieval
  4. SemanticMutator — LLM-driven semantic mutation & MCTS exploration
  5. QualityDiversity (FeatureMap/MAP-Elites) — QD diversity maintenance

All improvers run independently with isolated try/except — one failure never blocks others.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_EPSILON = 1e-6


@dataclass
class ImprovementResult:
    """Unified result from all improvement modules.

    Attributes:
        action: DecisionAction from decision_engine classification.
        action_reason: Human-readable reason for the action.
        sharpe: Sharpe ratio from brain_result.
        fitness: Fitness score from brain_result.
        improved_expressions: All candidate improved expressions from any source.
        improvement_sources: Which improver generated each expression (parallel list).
        ea_recommendation: EASearch strategy recommendation dict.
        distilled_patterns: ExperienceDistiller extracted patterns.
        semantic_variants: SemanticMutator generated variants.
        qd_suggestions: QD diversity maintenance suggestions.
        neutralization_rec: AdaptiveNeutralizer recommendation.
        reflection_diagnosis: ReflectionEngine diagnosis.
        near_pass_analysis: NearPassImprover analysis.
        turnover_analysis: TurnoverOptimizer analysis.
    """

    action: Any = None
    action_reason: str = ""
    sharpe: float = 0.0
    fitness: float = 0.0
    improved_expressions: list[str] = field(default_factory=list)
    improvement_sources: list[str] = field(default_factory=list)
    ea_recommendation: dict | None = None
    distilled_patterns: list | None = None
    semantic_variants: list | None = None
    qd_suggestions: list | None = None
    neutralization_rec: dict | None = None
    reflection_diagnosis: dict | None = None
    near_pass_analysis: dict | None = None
    turnover_analysis: dict | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ImprovementOrchestra:
    """Layer 4: Unified improvement orchestration hub.

    Wraps FeedbackOrchestrator's analyze_and_improve() as the PRIMARY delegate,
    then additionally invokes EASearch, ExperienceDistiller, SemanticMutator,
    and QualityDiversity in parallel/sequential isolation.

    Design principles:
      - Single entry point: analyze_and_improve()
      - Fault isolation: each improver has its own try/except
      - Lazy initialization: accept pre-created instances or create on-demand
      - Zero-blocking: one improver failure never stops others
      - Structured aggregation: all results funnel into ImprovementResult
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._feedback_orch: Any = None
        self._ea_search: Any = None
        self._experience_distiller: Any = None
        self._semantic_mutator: Any = None
        self._quality_diversity: Any = None
        self._stats: dict[str, int] = {
            "total_calls": 0,
            "fo_success": 0,
            "ea_success": 0,
            "ed_success": 0,
            "sm_success": 0,
            "qd_success": 0,
        }

    async def analyze_and_improve(
        self,
        brain_result: Any,
        expression: str,
        session_id: str = "",
        cycle_num: int = 0,
        exploration_direction: str = "",
        feedback_orch: Any = None,
        ea_search: Any = None,
        experience_distiller: Any = None,
        semantic_mutator: Any = None,
        quality_diversity: Any = None,
    ) -> ImprovementResult:
        """Analyze brain result and generate improvements from ALL improvers.

        This is the single entry point for Layer 4. It:
          1. Delegates to FeedbackOrchestrator.analyze_and_improve() (primary)
          2. Calls EASearch for evolutionary strategy recommendation
          3. Calls ExperienceDistiller for pattern extraction/retrieval
          4. Calls SemanticMutator for LLM semantic variants
          5. Calls QualityDiversity for diversity suggestions

        Args:
            brain_result: Raw WQ brain result object.
            expression: The alpha expression that was submitted.
            session_id: Session identifier for logging/tracing.
            cycle_num: Current evolution cycle number.
            exploration_direction: Current exploration direction (momentum, etc.).
            feedback_orch: Pre-created FeedbackOrchestrator instance (avoids re-init).
            ea_search: Pre-created EASearchStrategy instance.
            experience_distiller: Pre-created ExperienceDistiller instance.
            semantic_mutator: Pre-created SemanticMutator instance.
            quality_diversity: Pre-created QualityDiversity/FeatureMap instance.

        Returns:
            ImprovementResult with aggregated data from all improvers.
        """
        self._stats["total_calls"] += 1
        t_start = time.monotonic()
        result = ImprovementResult()

        sharpe = getattr(brain_result, "real_sharpe", None) or getattr(brain_result, "sharpe", 0) or 0.0
        fitness = getattr(brain_result, "real_fitness", None) or getattr(brain_result, "fitness", 0) or 0.0
        result.sharpe = float(sharpe) if sharpe else 0.0
        result.fitness = float(fitness) if fitness else 0.0

        fo = feedback_orch or self._feedback_orch
        if fo is not None:
            try:
                fo_out = await fo.analyze_and_improve(
                    brain_result=brain_result,
                    expression=expression,
                    session_id=session_id,
                    cycle_num=cycle_num,
                )
                result.action = getattr(fo_out, "action", None)
                result.action_reason = getattr(fo_out, "action_reason", "") or ""
                result.improved_expressions.extend(getattr(fo_out, "improved_expressions", []) or [])
                result.improvement_sources.extend(getattr(fo_out, "improvement_sources", []) or [])
                result.neutralization_rec = getattr(fo_out, "neutralization_recommendation", None)
                result.reflection_diagnosis = getattr(fo_out, "reflection_diagnosis", None)
                result.near_pass_analysis = getattr(fo_out, "near_pass_analysis", None)
                result.turnover_analysis = getattr(fo_out, "turnover_analysis", None)
                self._stats["fo_success"] += 1
                logger.info(
                    "[IO] FO done | action=%s variants=%d sources=%s",
                    getattr(result.action, "value", str(result.action)) if result.action else "NONE",
                    len(result.improved_expressions),
                    result.improvement_sources,
                )
            except (OSError, ValueError, RuntimeError, AttributeError) as exc:
                logger.warning("[IO] FeedbackOrchestrator failed: %s", exc)

        ea = ea_search or self._ea_search
        if ea is not None:
            try:
                target_sharpe = max(float(sharpe or 0) * 1.3, 1.25)
                ea_rec = {
                    "recommended": True,
                    "target_sharpe": target_sharpe,
                    "seed_expression": expression[:80],
                    "population_size": getattr(ea.config, "population_size", 8),
                    "max_generations": getattr(ea.config, "max_generations", 3),
                    "mutation_rate": getattr(ea.config, "mutation_rate", 0.6),
                    "diversity_threshold": getattr(ea.config, "diversity_threshold", 0.7),
                    "note": "EASearch available but full search deferred to caller "
                    "(use ea.search() directly for population-based evolution)",
                }
                result.ea_recommendation = ea_rec
                self._stats["ea_success"] += 1
                logger.info(
                    "[IO] EA ready | target_sharpe=%.2f pop=%d gen=%d",
                    target_sharpe,
                    ea_rec.get("population_size", 0),
                    ea_rec.get("max_generations", 0),
                )
            except (OSError, ValueError, RuntimeError, AttributeError) as exc:
                logger.warning("[IO] EASearch failed: %s", exc)

        ed = experience_distiller or self._experience_distiller
        if ed is not None:
            try:
                context_str = f"expression={expression[:60]} sharpe={sharpe:.3f} fitness={fitness:.3f}"
                if exploration_direction:
                    context_str += f" direction={exploration_direction}"
                cards = await ed.get_applicable_cards(context=context_str, top_k=3)
                if cards:
                    result.distilled_patterns = [
                        {
                            "rule_id": c.rule_id,
                            "failure_pattern": c.failure_pattern,
                            "fix_strategy": c.fix_strategy,
                            "confidence": round(c.confidence, 4),
                            "usage_count": c.usage_count,
                            "success_rate": round(c.success_count / max(c.usage_count, 1), 4),
                        }
                        for c in cards
                    ]
                    self._stats["ed_success"] += 1
                    logger.info(
                        "[IO] ExperienceDistiller | %d applicable cards retrieved",
                        len(cards),
                    )
            except (OSError, ValueError, RuntimeError, AttributeError) as exc:
                logger.warning("[IO] ExperienceDistiller failed: %s", exc)

        sm = semantic_mutator or self._semantic_mutator
        if sm is not None and hasattr(sm, "_llm_generate_fn") and sm._llm_generate_fn is not None:
            try:
                direction = exploration_direction or "momentum"
                variant = await sm.decode_to_expression(
                    parent_a_expr=expression,
                    parent_b_expr=expression,
                    interpolation_ratio=0.5,
                    direction=direction,
                )
                if variant and variant != expression:
                    result.semantic_variants = [variant]
                    result.improved_expressions.append(variant)
                    result.improvement_sources.append("semantic_mutator_decode")
                    self._stats["sm_success"] += 1
                    logger.info("[IO] SemanticMutator | 1 variant generated via decode_to_expression")
            except (OSError, ValueError, RuntimeError, AttributeError, ConnectionError, TimeoutError) as exc:
                logger.warning("[IO] SemanticMutator decode failed: %s", exc)

            try:
                if hasattr(sm, "explore_unexplored_regions"):
                    qd_instance = quality_diversity or self._quality_diversity
                    if qd_instance is not None and hasattr(qd_instance, "archive"):
                        mcts_variants = await sm.mcts_explore(
                            expression=expression,
                            n_simulations=min(max(int(self.config.get("mcts_simulations", 20)), 5), 50),
                            direction=exploration_direction or "momentum",
                        )
                        if mcts_variants:
                            existing_semantic = set(result.semantic_variants or [])
                            new_mcts = [v for v in mcts_variants[:3] if v not in existing_semantic and v != expression]
                            if new_mcs := new_mcts:
                                if result.semantic_variants is None:
                                    result.semantic_variants = []
                                result.semantic_variants.extend(new_mcs)
                                for v in new_mcs:
                                    result.improved_expressions.append(v)
                                    result.improvement_sources.append("semantic_mutator_mcts")
                                logger.info("[IO] SemanticMutator MCTS | %d additional variants", len(new_mcs))
            except (OSError, ValueError, RuntimeError, AttributeError) as exc:
                logger.debug("[IO] SemanticMutator MCTS explore skipped: %s", exc)

        qd = quality_diversity or self._quality_diversity
        if qd is not None:
            try:
                qd_suggestions: list[dict] = []

                unexplored = getattr(qd, "get_unexplored_directions", lambda: [])()
                if unexplored:
                    qd_suggestions.append({"type": "unexplored_directions", "items": unexplored[:5]})

                explore_targets = (
                    await qd.get_explore_targets(top_k=3)
                    if asyncio.iscoroutinefunction(getattr(qd, "get_explore_targets", None))
                    else getattr(qd, "get_explore_targets", lambda **kw: [])(top_k=3)
                )
                if explore_targets:
                    qd_suggestions.append(
                        {
                            "type": "explore_targets",
                            "items": [
                                {"direction": t.get("direction", ""), "score": t.get("score", 0)}
                                for t in explore_targets[:3]
                            ],
                        }
                    )

                frontier_targets = (
                    await qd.get_frontier_targets(top_k=2)
                    if asyncio.iscoroutinefunction(getattr(qd, "get_frontier_targets", None))
                    else getattr(qd, "get_frontier_targets", lambda **kw: [])(top_k=2)
                )
                if frontier_targets:
                    qd_suggestions.append(
                        {
                            "type": "frontier_targets",
                            "items": [
                                {"direction": f.get("direction", ""), "fitness_gap": f.get("fitness_gap", 0)}
                                for f in frontier_targets[:2]
                            ],
                        }
                    )

                div_stats = getattr(qd, "get_diversity_stats", lambda: {})()
                if div_stats:
                    qd_suggestions.append({"type": "diversity_stats", "stats": div_stats})

                if qd_suggestions:
                    result.qd_suggestions = qd_suggestions
                    self._stats["qd_success"] += 1
                    logger.info(
                        "[IO] QualityDiversity | %d suggestion groups, stats=%s",
                        len(qd_suggestions),
                        {g["type"]: len(g.get("items", [])) for g in qd_suggestions},
                    )

                current_dir = exploration_direction or "momentum"
                current_horiz = self.config.get("default_time_horizon", "medium")
                current_mech = self.config.get("default_mechanism", "signal")
                qd_add_result = getattr(qd, "add_candidate", lambda **kw: False)(
                    direction=current_dir,
                    time_horizon=current_horiz,
                    mechanism=current_mech,
                    expression=expression,
                    sharpe=float(sharpe or 0.0),
                    fitness=float(fitness or 0.0),
                    turnover=getattr(brain_result, "turnover", None),
                )
                if qd_add_result:
                    logger.debug("[IO] QD archive updated for %s/%s/%s", current_dir, current_horiz, current_mech)
            except (OSError, ValueError, RuntimeError, AttributeError) as exc:
                logger.warning("[IO] QualityDiversity failed: %s", exc)

        # ── Consume TrajectoryMutation insights (from periodic_tasks background) ──
        try:
            from openalpha_brain.core import loop_state as _ls_mod

            _mutation_insights = getattr(_ls_mod._ls, "_trajectory_mutation_insights", None)
            if _mutation_insights and len(_mutation_insights) > 0:
                _relevant_mutations = [
                    m for m in _mutation_insights
                    if m.get("direction") == exploration_direction or m.get("mutation_type") == "segment_replacement"
                ]
                if _relevant_mutations:
                    result.metadata["trajectory_mutations_available"] = len(_relevant_mutations)
                    logger.info(
                        "[IO] TrajectoryMutation Insights | %d relevant mutations consumed from background",
                        len(_relevant_mutations),
                    )

            _weak_segment_alerts = getattr(_ls_mod._ls, "_weak_segment_alerts", None)
            if _weak_segment_alerts and len(_weak_segment_alerts) > 0:
                result.metadata["weak_segments_detected"] = len(_weak_segment_alerts)
                logger.info(
                    "[IO] Weak Segment Alerts | %d alerts from trajectory mutation analysis",
                    len(_weak_segment_alerts),
                )
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            logger.debug("[IO] TrajectoryMutation insight consumption skipped: %s", exc)

        elapsed = time.monotonic() - t_start
        total_variants = len(result.improved_expressions)
        active_improvers = sum(
            1
            for k in ["ea_recommendation", "distilled_patterns", "semantic_variants", "qd_suggestions"]
            if getattr(result, k, None) is not None
        )
        logger.info(
            "[IO] COMPLETE | session=%s cycle=%d | elapsed=%.2fs | "
            "action=%s variants=%d improvers_active=%d | "
            "fo=%s ea=%s ed=%s sm=%s qd=%s",
            session_id,
            cycle_num,
            elapsed,
            getattr(result.action, "value", str(result.action)) if result.action else "NONE",
            total_variants,
            active_improvers,
            self._stats.get("fo_success", 0),
            self._stats.get("ea_success", 0),
            self._stats.get("ed_success", 0),
            self._stats.get("sm_success", 0),
            self._stats.get("qd_success", 0),
        )
        return result

    def set_feedback_orchestrator(self, fo: Any) -> None:
        self._feedback_orch = fo

    def set_ea_search(self, ea: Any) -> None:
        self._ea_search = ea

    def set_experience_distiller(self, ed: Any) -> None:
        self._experience_distiller = ed

    def set_semantic_mutator(self, sm: Any) -> None:
        self._semantic_mutator = sm

    def set_quality_diversity(self, qd: Any) -> None:
        self._quality_diversity = qd

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def reset_stats(self) -> None:
        self._stats = dict.fromkeys(self._stats, 0)
