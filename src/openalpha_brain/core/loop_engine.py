"""
OpenAlpha - Quant — Autonomous Generation Loop Engine
Runs as an asyncio background task. Drives the full LLM → Parse → Validate cycle.
Persists state after every step. Checks stop_requested every cycle.

Refactored: in-memory state within cycle, modular sub-functions,
integrated AST repair, robust BRAIN expression parsing.
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, cast

from openalpha_brain.utils.algo_logger import algo_log, Timer, log_call
from openalpha_brain.core.events import get_event_bus, EVENT_CYCLE_START, EVENT_ALPHA_GENERATED, EVENT_ALPHA_VALIDATED, EVENT_ALPHA_REJECTED, EVENT_BRAIN_SUBMIT, EVENT_BRAIN_RESULT, EVENT_MAB_FEEDBACK, EVENT_CYCLE_COMPLETE, EVENT_MINING_COMPLETE, EVENT_ERROR

from openalpha_brain.generation import alpha_parser as parser
from openalpha_brain.generation.alpha_parser import parse_alpha_json
from openalpha_brain.services import brain_client
from openalpha_brain.services import llm_client
from openalpha_brain.cli import session_manager as sm
from openalpha_brain.validation import validator as val
from openalpha_brain.validation.validator import compute_hierarchical_reward, get_reward_level
from openalpha_brain.validation.ast_repair import repair_expression
from openalpha_brain.learning.param_optimizer import expression_hash
from openalpha_brain.learning.mab import (
    REWARD_VALIDATOR_PASS,
    REWARD_SHARPE_05,
    PENALTY_BRAIN_FAIL, PENALTY_BRAIN_ERROR,
)
from openalpha_brain.core.scheduler import ExplorationScheduler
from openalpha_brain.config.config import settings
from openalpha_brain.knowledge.rag_tools import (
    RAG_TOOL_SCHEMAS,
    validate_expression_fields,
    auto_repair_expression,
)
from openalpha_brain.agents.multi_agent import MultiAgentOrchestrator, IdeaAgent, FactorAgent, EvalAgent, _check_semantic_alignment, Hypothesis
from openalpha_brain.services.brain_submitter import create_dedup_mutation_callback
from openalpha_brain.validation.decay_detector import AlphaDecayDetector, DecayLevel, AlphaDecayRecord, create_alpha_decay_handler
from openalpha_brain.evolution.crossover_mutation import GradientMutation, SemanticCrossover, CrossoverMutationEngine, AlphaTrajectory
from openalpha_brain.evolution.generation_gates import GenerationGates, GenerationGateReport, GATE_HYPOTHESIS_EXPRESSION, GATE_EXPRESSION_CODE, GATE_HOLISTIC
from openalpha_brain.evolution.hypothesis_aligner import HypothesisAligner
from openalpha_brain.utils.resilience import TaskHealthRegistry, async_timeout
from openalpha_brain.generation.alpha_logics import AlphaLogicLibrary
from openalpha_brain.core.models import (
    AlphaFingerprint, AlphaMetrics, AlphaResult,
    BrainSimStatus, BrainSubmissionResult,
    PipelineStatus, SessionStatus,
)
from openalpha_brain.core.pipeline import AlphaCachePool
from openalpha_brain.generation.prompts import (
    get_system_prompt,
    build_brain_failure_feedback,
    build_dynamic_context,
    build_failure_feedback,
    build_family_switch_warning,
    build_memory_injection,
    build_restart_trigger,
    build_start_trigger,
    build_success_feedback,
)
from openalpha_brain.core import loop_state as _ls
from openalpha_brain.core.generator import (
    _generate_one_and_enqueue, _generate_single_alpha,
    _apply_generation_gates, _generate_llm_variants,
    _filter_variants_by_field_overlap, _build_continuation_msg,
)
from openalpha_brain.generation.alpha_generator import (
    _extract_hallucinations_from_failures,
    _extract_brain_feedback_from_state,
    _extract_economic_rationale, _apply_economic_rationale_verification,
    _extract_expression_from_llm, _build_alpha,
    _handle_parse_failure, _handle_reject, _handle_iterate,
    _summarise_rejected,
)
from openalpha_brain.learning.reward_updater import (
    _get_operators_from_context, _get_fields_from_context,
    _arbiter_rerank,
    _make_tool_executor, _extract_allowed_fields_from_tool_results,
    _extract_strategy_features,
    _refill_eliminated_fields, _sync_mab_bias_from_evidence,
    _run_logic_evolution,
)
from openalpha_brain.services.brain_submitter import (
    _submit_to_brain, _brain_improvement_loop,
    _run_param_optimization,
    _build_brain_result_dict, _log_brain_result,
)
from openalpha_brain.services.brain_result_processor import (
    compute_hierarchical_reward_with_penalties,
    check_margin_efficiency,
    record_pass_feedback,
    submit_for_review,
    run_stability_analysis,
    fetch_correlation_analysis,
    record_evo_and_success,
    record_fail_feedback,
    run_post_brain_processing,
)
from openalpha_brain.core.post_processor import _post_process_brain_result, _merge_session_hallucinations
from openalpha_brain.core.periodic_tasks import _periodic_decay_check, _periodic_trajectory_crossover

logger = logging.getLogger(__name__)


async def _gate_regenerate_fn(
    expression: str, correction_prompt: str, payload: dict,
) -> tuple[str, dict]:
    prompt = (
        f"The following alpha expression failed generation gate checks:\n\n"
        f"```\n{expression}\n```\n\n"
        f"{correction_prompt}\n\n"
        f"Generate a corrected FASTEXPR alpha expression that addresses ALL issues above. "
        f"Output ONLY the corrected expression, no explanation or markdown."
    )
    try:
        response = await llm_client.generate(prompt)
        new_expr = response.strip()
        if "```" in new_expr:
            import re as _gate_re
            _code_match = _gate_re.search(
                r'```(?:fastexpr|python)?\s*\n?(.*?)```', new_expr, _gate_re.DOTALL,
            )
            if _code_match:
                new_expr = _code_match.group(1).strip()
        return new_expr, payload
    except (OSError, ValueError, RuntimeError) as exc:
        logger.warning("[DEFENSIVE_LOG] _extract_code_block_from_expr 解析异常，返回原始表达式: %s", exc)
        return expression, payload



def _family_locked(state) -> bool:
    tracker = state.family_run_tracker
    if len(tracker) < 3:
        return False
    return len(set(tracker[-3:])) == 1


def _last_family(state) -> str:
    return state.family_run_tracker[-1] if state.family_run_tracker else "Unknown"



# ── Main loop ──────────────────────────────────────────────────────────────────

@algo_log(level=logging.INFO)
async def run_loop(session_id: str) -> None:
    """
    Main async generation loop. Entry point called as asyncio.create_task().
    Holds state in memory within each cycle; persists at decision checkpoints.
    """
    logger.info("[%s] Loop engine started", session_id)

    if _ls._scheduler is None and _ls._rag_engine is None:
        _ls.init_intelligent_search()

    logger.info("[%s] Component status: Scheduler=%s RAG=%s Whitelist=%s SignalArbiter=%s AlphaChannel=%s ExperienceDistiller=%s MarketState=%s MultiAgent=%s",
        session_id,
        _ls._scheduler is not None,
        _ls._rag_engine is not None and _ls._rag_engine.is_ready if _ls._rag_engine else False,
        _ls._whitelist_mgr is not None,
        _ls._signal_arbiter is not None,
        _ls._alpha_channel is not None,
        _ls._experience_distiller is not None,
        _ls._market_state_inferencer is not None,
        settings.MULTI_AGENT_ENABLED,
    )

    orchestrator = None
    _orchestrator_variants: list[str] = []
    if settings.MULTI_AGENT_ENABLED:
        if _ls._logic_library is None:
            _ls._logic_library = AlphaLogicLibrary()
        _idea_agent = IdeaAgent(llm_client.generate, _ls._rag_engine, logic_library=_ls._logic_library, classifier=_ls._strategy_classifier, mab=_ls._scheduler, field_proxy_map=getattr(_ls._scheduler, 'field_proxy_map', None) if _ls._scheduler else None)
        _factor_agent = FactorAgent(llm_client.generate, _ls._rag_engine, logic_library=_ls._logic_library, success_lib=_ls._success_lib, feature_map=_ls._feature_map, reflection_engine=_ls._reflection_engine, whitelist_mgr=_ls._whitelist_mgr, field_proxy_map=getattr(_ls._scheduler, 'field_proxy_map', None) if _ls._scheduler else None, mab=_ls._scheduler)
        _eval_agent = EvalAgent(
            brain_client.submit_and_poll,
            mab=_ls._scheduler,
            success_lib=_ls._success_lib,
            llm_generate_fn=llm_client.generate,
            success_pool=_ls.__dict__.get("_success_pool", []),
            dedup_callback=create_dedup_mutation_callback,
        )
        orchestrator = MultiAgentOrchestrator(
            _idea_agent, _factor_agent, _eval_agent,
            originality_checker=val.get_originality_checker(),
            complexity_controller=val.get_complexity_controller(),
            rag_engine=_ls._rag_engine,
            evolution_db=_ls._evo_db,
            feature_map=_ls._feature_map,
        )
        if _ls._successful_brain_expressions:
            orchestrator.inject_successful_alphas(_ls._successful_brain_expressions)
        logger.info("[%s] Multi-agent orchestrator initialized", session_id)

    if _ls._crossover_engine is None:
        try:
            from openalpha_brain.validation.validator import get_originality_checker as _get_oc
            _oc = _get_oc()
            _engine = CrossoverMutationEngine(originality_checker=_oc, llm_generate_fn=llm_client.generate)
            _ls._crossover_engine = _engine
            logger.info("[%s] CrossoverMutationEngine initialized (standard loop)", session_id)
        except (OSError, ValueError, RuntimeError):
            _ls._crossover_engine = CrossoverMutationEngine(originality_checker=None, llm_generate_fn=llm_client.generate)
            logger.info("[%s] CrossoverMutationEngine initialized (standard loop, no originality checker)", session_id)

    if _ls._decay_detector is None:
        _decay_detector = AlphaDecayDetector()
        _ls._decay_detector = _decay_detector
        if _ls._scheduler is not None:
            _ls._scheduler.decay_detector = _decay_detector
        try:
            from openalpha_brain.services.brain_data_client import get_brain_data_client
            _bdc = get_brain_data_client()
            if _bdc is not None:
                decay_handler = await create_alpha_decay_handler(loop_state_module=_ls)
                await _decay_detector.register_instruments(
                    fetch_yearly=_bdc.get_yearly_performance,
                    fetch_pnl=_bdc.get_pnl_curve,
                    estimate_garch=None,
                    on_decay_detected=decay_handler,
                )
                logger.info("[%s] DecayDetector instruments registered (standard loop)", session_id)
        except (OSError, ValueError, RuntimeError) as _decay_init_exc:
            logger.warning("[%s] DecayDetector instrument registration failed: %s", session_id, _decay_init_exc)

    if not hasattr(_ls, '_generation_gates') or _ls._generation_gates is None:
        _hyp_aligner = HypothesisAligner()
        _ls._hypothesis_aligner = _hyp_aligner
        _gen_gates = GenerationGates(
            hypothesis_aligner=_hyp_aligner,
            llm_generate_fn=llm_client.generate,
        )
        _ls._generation_gates = _gen_gates
        logger.info("[%s] GenerationGates initialized (standard loop)", session_id)

    _consecutive_errors = 0
    _MAX_CONSECUTIVE_ERRORS = 3
    _ev = get_event_bus()

    for global_cycle in range(1, settings.MAX_CYCLES + 1):

        _ev.emit(EVENT_CYCLE_START, {"session_id": session_id, "cycle": global_cycle, "max_cycles": settings.MAX_CYCLES})

        if _ls._console_stop_event and _ls._console_stop_event.is_set():
            logger.info("[%s] Console stop requested, breaking loop", session_id)
            break
        if _ls._console_pause_event and not _ls._console_pause_event.is_set():
            logger.info("[%s] Console pause requested, waiting...", session_id)
            await _ls._console_pause_event.wait()
            logger.info("[%s] Console resumed", session_id)

        brain_result = None

        if _ls._budget_tracker:
            _ls._budget_tracker.reset()

        if _ls._heartbeat:
            _ls._heartbeat.touch(session_id)

        state = await sm.load_session(session_id)
        if state is None:
            logger.error("[%s] Session disappeared — aborting loop", session_id)
            return

        if state.stop_requested:
            logger.info("[%s] Stop requested — halting after cycle %d", session_id, global_cycle - 1)
            state.status = SessionStatus.STOPPED
            await sm.save_session(state)
            if _ls._heartbeat:
                _ls._heartbeat.remove(session_id)
            return

        state.cycle = global_cycle
        state.status = SessionStatus.GENERATING
        await sm.save_session(state)

        exploration_direction: str = _resolve_effective_focus(state.focus_area) or settings.DEFAULT_EXPLORATION_DIRECTION
        _sched_template_id: str = ""
        _sched_family_id: str = ""
        _sched_recommended_fields: list[str] = []
        if settings.MAB_ENABLED and _ls._scheduler:
            try:
                _ls._algo_tick("mab_select")
                _sched_result = _ls._scheduler.select_exploration_arm(focus_area=_resolve_effective_focus(state.focus_area), explore_mode=False)
                if _sched_result:
                    exploration_direction = _sched_result["direction"]
                    _sched_template_id = _sched_result.get("template_id", "")
                    _sched_family_id = _sched_result.get("family_id", "")
                    _sched_recommended_fields = _sched_result.get("recommended_fields", [])
            except (OSError, ValueError, RuntimeError) as exc:
                logger.debug("[%s] cycle=%d scheduler select_exploration_arm failed: %s", session_id, global_cycle, exc)
            _schedule = _ls._feature_map.get_explore_exploit_schedule()
            _map_strategy = _schedule["strategy"]
            _explore_weight = _schedule["explore_weight"]
            if _map_strategy == "explore":
                _unexplored = _ls._feature_map.get_unexplored_directions()
                if _unexplored and random.random() < _explore_weight:
                    exploration_direction = random.choice(_unexplored)
                    logger.info(
                        "[%s] MAP-Elites EXPLORE: pivoting to unexplored direction=%s (weight=%.2f coverage=%.1f%%)",
                        session_id, exploration_direction, _explore_weight, _schedule["coverage"] * 100,
                    )
                _explore_targets = _ls._feature_map.get_explore_targets(top_k=3)
                if _explore_targets:
                    _target = random.choice(_explore_targets)
                    _target_dir = _target.get("direction", "")
                    if _target_dir:
                        exploration_direction = _target_dir
                    logger.info(
                        "[%s] MAP-Elites EXPLORE_TARGET: cell=%s "
                        "direction=%s horizon=%s mechanism=%s (coverage=%.1f%%)",
                        session_id, _target.get("key", ""), _target_dir,
                        _target.get("time_horizon", ""), _target.get("mechanism", ""),
                        _schedule["coverage"] * 100,
                    )
            else:
                _exploit_cell = _ls._feature_map.get_cell(
                    exploration_direction, "medium", "signal",
                )
                if _exploit_cell is not None and _exploit_cell.elites:
                    _cell_elites = _ls._feature_map.get_cell_elites(
                        exploration_direction, "medium", "signal",
                    )
                    logger.info(
                        "[%s] MAP-Elites EXPLOIT: direction=%s (coverage=%.1f%% elite_density=%.2f cell_elites=%d)",
                        session_id, exploration_direction, _schedule["coverage"] * 100,
                        _schedule["elite_density"], len(_cell_elites),
                    )

        # ── 1. Build user message ───────────────────────────────────────────
        if global_cycle == 1:
            user_msg = build_start_trigger(global_cycle, _resolve_effective_focus(state.focus_area) or settings.DEFAULT_EXPLORATION_DIRECTION)
        else:
            user_msg = _build_continuation_msg(state, str(global_cycle))

        if _family_locked(state):
            locked_family = _last_family(state)
            user_msg += build_family_switch_warning(locked_family, global_cycle)
            logger.info("[%s] cycle=%d family lock active for '%s'", session_id, global_cycle, locked_family)

        memory_str = build_memory_injection(state)
        user_msg = memory_str + "\n" + user_msg

        if state.hallucination_log:
            _recent_repairs = [e for e in state.hallucination_log[-10:] if e.get("source") == "ast_repair"]
            if _recent_repairs:
                _repair_patterns = {}
                for _re in _recent_repairs:
                    _var = _re.get("variable", "")
                    _etype = _re.get("error_type", "")
                    if _var and _etype:
                        _repair_patterns.setdefault(_etype, set()).add(_var)
                if _repair_patterns:
                    _repair_lines = ["\n\n🔧 PAST AUTO-REPAIRS (avoid these mistakes — they were already corrected):"]
                    for _etype, _vars in list(_repair_patterns.items())[:5]:
                        _repair_lines.append(f"  - {_etype}: {', '.join(list(_vars)[:3])}")
                    user_msg += "\n".join(_repair_lines)

        cycle_exploration_direction: str = _resolve_effective_focus(state.focus_area) or settings.DEFAULT_EXPLORATION_DIRECTION
        if _ls._strategy_classifier is not None:
            try:
                _top_profiles = _ls._strategy_classifier.get_top_profiles(n=3)
                if _top_profiles:
                    _profile_lines = ["\n\nProven effective strategy directions (use as reference for new alpha ideas):"]
                    for _tp in _top_profiles:
                        _profile_lines.append(
                            f"  - direction={_tp.direction} mechanism={_tp.mechanism} "
                            f"horizon={_tp.time_horizon} sharpe={_tp.sharpe:.2f}"
                        )
                    user_msg += "\n".join(_profile_lines)
            except (OSError, ValueError, RuntimeError) as exc:
                logger.debug("[%s] cycle=%d strategy classifier profile failed: %s", session_id, global_cycle, exc)
            try:
                _ls._algo_tick("find_similar_by_embedding")
                similar_strategies = await _ls._strategy_classifier.find_similar_by_embedding(cycle_exploration_direction, top_k=3)
                if similar_strategies:
                    high_sim = [s for s in similar_strategies if s.get("similarity", 0) > 0.8]
                    if high_sim:
                        logger.warning("[%s] cycle=%d Potential strategy duplication — %d similar strategies (sim>0.8)", session_id, global_cycle, len(high_sim))
                        dup_lines = []
                        for _si in high_sim[:3]:
                            p = _si.get("profile")
                            if p:
                                dup_lines.append(f"  - direction={getattr(p, 'direction', '?')} sharpe={getattr(p, 'sharpe', 'N/A')} expr={getattr(p, 'best_expression', '?')[:60]}")
                        if dup_lines:
                            user_msg += "\n\nSimilar existing strategies to avoid duplicating:\n" + "\n".join(dup_lines)
            except (OSError, ValueError, RuntimeError) as exc:
                logger.debug("[%s] cycle=%d strategy classifier embedding search failed: %s", session_id, global_cycle, exc)

        # ── 2. LLM call ────────────────────────────────────────────────────
        _fb_list = getattr(_ls, '_brain_feedback_buffer', None)
        if _fb_list:
            for _fb_msg in _fb_list:
                state.conversation_history.append(_fb_msg)
            logger.info(
                "[%s] cycle=%d injected %d BRAIN result feedback(s) into conversation history",
                session_id, global_cycle, len(_fb_list),
            )
            _ls._brain_feedback_buffer = []
            await sm.save_session(state)

        effective_history = state.conversation_history
        if _ls._summarizer:
            _ls._algo_tick("conversation_summarizer")
            effective_history, was_summarized = await _ls._summarizer.summarize_if_needed(state.conversation_history)
            if was_summarized:
                logger.info("[%s] cycle=%d — conversation history summarized before LLM call", session_id, global_cycle)
        logger.info("[%s] cycle=%d — calling LLM", session_id, global_cycle)
        _ls._monitor.record("STEP", "llm", "generate", f"cycle={global_cycle}", session_id=session_id)
        if _ls._heartbeat:
            _ls._heartbeat.touch(session_id)
        _run_loop_tool_results: dict = {}
        rag_context = None
        _global_blacklist_entries = _ls._global_knowledge.get_rag_context_entries() if _ls._global_knowledge else None

        if settings.RAG_ENABLED and _ls._rag_engine and _ls._rag_engine.is_ready:
            try:
                _ls._algo_tick("rag_retrieve")
                retrieval = await _ls._rag_engine.retrieve(exploration_direction)
                rag_context = _ls._rag_engine.assemble_context(retrieval)
                _rag_ops = retrieval.get("operators", [])
                _rag_fields = retrieval.get("fields", [])
                logger.info("[%s] MONITOR: rag_retrieve: ops=%d fields=%d top_ops=%s top_fields=%s", session_id, len(_rag_ops), len(_rag_fields), [o.get("id", o.get("name", "")) for o in _rag_ops[:3]], [f.get("id", f.get("name", "")) for f in _rag_fields[:3]])
                rag_context = await _arbiter_rerank(retrieval, rag_context, exploration_direction)
                _ls._monitor.record("STEP", "rag", "retrieve", f"direction={exploration_direction}", session_id=session_id)
                if _ls._whitelist_mgr and rag_context.get("field_ids"):
                    _ls._algo_tick("whitelist_update")
                    _ls._whitelist_mgr.update_dynamic(rag_context["field_ids"])
                    for _rag_field in retrieval.get("fields", []):
                        _fid = _rag_field.get("id") or _rag_field.get("name")
                        _fds = _rag_field.get("dataset", "unknown")
                        if _fid:
                            _ls._whitelist_mgr.register_field_dataset(_fid, _fds)
            except (OSError, ValueError, RuntimeError) as exc:
                logger.warning("[%s] cycle=%d RAG retrieval failed: %s", session_id, global_cycle, exc)

        _exp_cards_dicts: list[dict] = []
        _exp_card_rule_ids: list[str] = []
        if _ls._experience_distiller:
            try:
                _ls._algo_tick("experience_card_retrieval")
                _exp_cards = await _ls._experience_distiller.get_applicable_cards(exploration_direction, top_k=3)
                _exp_cards_dicts = [{"failure_pattern": c.failure_pattern, "fix_strategy": c.fix_strategy, "applicable_conditions": c.applicable_conditions, "confidence": c.confidence} for c in _exp_cards]
                _exp_card_rule_ids = [c.rule_id for c in _exp_cards if c.rule_id]
            except (OSError, ValueError, RuntimeError) as exc:
                logger.debug("[%s] cycle=%d experience card retrieval failed: %s", session_id, global_cycle, exc)

        _success_context = ""
        if _ls._success_lib and settings.SUCCESS_CASE_LIBRARY_ENABLED:
            try:
                similar_cases = await _ls._success_lib.search_similar(exploration_direction, top_k=3)
                if similar_cases:
                    ref_lines = []
                    for sc in similar_cases[:3]:
                        ref_lines.append(f"  - (sharpe={sc.get('sharpe', 'N/A')}, fitness={sc.get('fitness', 'N/A')}): {sc.get('expr', '')}")
                    _success_context = "\n\nPreviously successful alphas for reference:\n" + "\n".join(ref_lines)
            except (OSError, ValueError, RuntimeError) as exc:
                logger.debug("[%s] cycle=%d success case library search failed: %s", session_id, global_cycle, exc)

        if _success_context:
            user_msg += _success_context

        parsed: dict[str, Any] | None = None
        if orchestrator is not None:
            try:
                _ls._algo_tick("multi_agent_orchestrator")
                brain_feedback_data = _extract_brain_feedback_from_state(state)
                operators_list = _get_operators_from_context(rag_context)
                fields_list = _get_fields_from_context(rag_context)

                result = await orchestrator.run_iteration(
                    direction=exploration_direction,
                    history=state.conversation_history,
                    brain_feedback=brain_feedback_data,
                    operators=operators_list,
                    fields=fields_list,
                )

                logger.info(
                    "[ALGO_STEP] module=multi_agent iteration=%d converged=%s originality=%.2f",
                    result.iterations, result.converged, result.originality_score,
                )

                _orchestrator_variants = getattr(result, 'variants', [])

                try:
                    _ls._algo_tick("llm_multi_candidate")
                    llm_variants = await _generate_llm_variants(
                        result.expression, exploration_direction, num_variants=3,
                    )
                    if llm_variants:
                        valid_variants = _filter_variants_by_field_overlap(
                            result.expression, llm_variants, max_overlap=0.7,
                        )
                        _orchestrator_variants = list(_orchestrator_variants) + valid_variants
                        logger.info(
                            "[%s] cycle=%d LLM multi-candidate: generated=%d accepted=%d (field_overlap<0.7)",
                            session_id, global_cycle, len(llm_variants), len(valid_variants),
                        )
                except (OSError, ValueError, RuntimeError) as exc:
                    logger.debug("[%s] cycle=%d LLM multi-candidate generation failed: %s", session_id, global_cycle, exc)

                if hasattr(result, 'trajectory') and result.trajectory:
                    try:
                        if not hasattr(state, 'trajectories'):
                            state.trajectories = []
                        state.trajectories.append(result.trajectory.model_dump() if hasattr(result.trajectory, 'model_dump') else vars(result.trajectory))
                    except (OSError, ValueError, RuntimeError) as exc:
                        logger.debug("[%s] cycle=%d trajectory append failed: %s", session_id, global_cycle, exc)

                parsed = {
                    "decision": "SUBMIT" if result.converged else "ITERATE",
                    "expression": result.expression,
                    "simulation_payload": result.simulation_payload,
                    "fingerprint": {},
                    "family": result.hypothesis.direction,
                    "ast_topology": None,
                    "rationale": result.hypothesis.natural_language,
                    "metrics": {},
                    "mutation_paths": [],
                    "refinement_log": None,
                }

                expression = result.expression

                _gate_ops = operators_list if 'operators_list' in locals() else None
                _gate_fields = fields_list if 'fields_list' in locals() else None
                _gates = getattr(_ls, '_generation_gates', None)
                if _gates is not None:
                    expression, _gate_report = await _gates.apply_with_retry(
                        hypothesis_direction=result.hypothesis.direction,
                        hypothesis_mechanism=result.hypothesis.mechanism,
                        hypothesis_nl=result.hypothesis.natural_language,
                        expression=expression,
                        regenerate_fn=_gate_regenerate_fn,
                        operators=_gate_ops,
                        fields=_gate_fields,
                    )
                else:
                    _gate_report = await _apply_generation_gates(
                        expression=expression,
                        hypothesis_direction=result.hypothesis.direction,
                        hypothesis_nl=result.hypothesis.natural_language,
                        hypothesis_mechanism=result.hypothesis.mechanism,
                        operators=_gate_ops,
                        fields=_gate_fields,
                    )
                if not _gate_report.passed:
                    logger.warning(
                        "[%s] cycle=%d GENERATION GATES FAILED: score=%.3f failed=%s",
                        session_id, global_cycle, _gate_report.overall_score, _gate_report.failed_gates,
                    )
                    _gate_correction = _gate_report.correction_prompt
                    state.conversation_history.append({"role": "user", "content": str(user_msg)})
                    state.conversation_history.append({
                        "role": "assistant",
                        "content": f"Expression generated: {expression}\n\n{_gate_correction}",
                    })
                    state.status = SessionStatus.ITERATING
                    await sm.save_session(state)
                    continue
                logger.info(
                    "[%s] cycle=%d GENERATION GATES PASSED: score=%.3f",
                    session_id, global_cycle, _gate_report.overall_score,
                )

                syntax_result = val.validate_syntax(expression)
                if not syntax_result.passed:
                    logger.warning(
                        "[ALGO_FAIL] module=factor_agent step=validate result=syntax_error errors=%s",
                        syntax_result.failures,
                    )
                    _ls._algo_tick("ast_repair")
                    repaired, repair_entries = repair_expression(expression)
                    if repaired and repaired != expression:
                        expression = repaired
                        if parsed is not None:
                            parsed["expression"] = repaired
                        logger.info("[ALGO_STEP] module=ast_repair step=repair result=success")
                    else:
                        _debug_recovered = False
                        if _ls._failure_lib or _ls._success_lib:
                            try:
                                from openalpha_brain.knowledge.rag_engine import auto_debug_loop as _auto_debug
                                _ls._algo_tick("auto_debug_loop")
                                debugged_expr, debug_ok = await _auto_debug(
                                    generate_fn=llm_client.generate,
                                    validate_fn=val.validate_syntax,
                                    initial_expr=expression,
                                    max_rounds=2,
                                )
                                if debug_ok and debugged_expr:
                                    expression = debugged_expr
                                    if parsed is not None:
                                        parsed["expression"] = debugged_expr
                                    syntax_result = val.validate_syntax(expression)
                                    _debug_recovered = True
                                    logger.info("[ALGO_STEP] module=auto_debug step=debug result=success expr=%s", expression[:60])
                            except (OSError, ValueError, RuntimeError):
                                logger.warning("[ALGO_STEP] module=auto_debug step=debug result=failed")
                        if not _debug_recovered:
                            logger.warning("[ALGO_SKIP] module=factor_agent step=validate reason=syntax_unrepairable")
                            state.conversation_history.append({"role": "user", "content": user_msg})
                            failure_msg = build_failure_feedback(
                                failures=syntax_result.failures,
                                expression=expression,
                                cycle=global_cycle,
                            )
                            state.conversation_history.append({"role": "assistant", "content": failure_msg})
                            state.failure_catalog.append({
                                "fingerprint": {},
                                "failure_type": "syntax_unrepairable",
                                "metric_value": syntax_result.failures[:3],
                                "mutation_tried": "repair_failed",
                            })
                            state.status = SessionStatus.ITERATING
                            await sm.save_session(state)
                            _consecutive_errors += 1
                            await asyncio.sleep(1)
                            continue

                raw_response = str(parsed)
                _consecutive_errors = 0
            except llm_client.LLMError as exc:
                logger.error("[%s] cycle=%d LLM permanent failure (multi-agent): %s", session_id, global_cycle, exc)
                _consecutive_errors += 1
                if _consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                    logger.error("[%s] cycle=%d %d consecutive LLM errors — aborting", session_id, global_cycle, _consecutive_errors)
                    state.status = SessionStatus.ERROR
                    state.error_message = str(exc)
                    await sm.save_session(state)
                    if _ls._heartbeat:
                        _ls._heartbeat.remove(session_id)
                    return
                state.conversation_history.append({"role": "user", "content": user_msg})
                state.conversation_history.append({
                    "role": "assistant",
                    "content": f"LLM call failed (cycle {global_cycle}): {exc}. Retrying next cycle.",
                })
                state.status = SessionStatus.ITERATING
                state.error_message = str(exc)
                await sm.save_session(state)
                await asyncio.sleep(2)
                continue
            except (OSError, ValueError, RuntimeError) as exc:
                logger.error("[%s] cycle=%d Multi-agent orchestrator error: %s", session_id, global_cycle, exc, exc_info=True)
                _consecutive_errors += 1
                if _consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                    logger.error("[%s] cycle=%d %d consecutive errors — aborting", session_id, global_cycle, _consecutive_errors)
                    state.status = SessionStatus.ERROR
                    state.error_message = str(exc)
                    await sm.save_session(state)
                    if _ls._heartbeat:
                        _ls._heartbeat.remove(session_id)
                    return
                state.conversation_history.append({"role": "user", "content": user_msg})
                state.conversation_history.append({
                    "role": "assistant",
                    "content": f"Orchestrator error (cycle {global_cycle}): {exc}. Retrying next cycle.",
                })
                state.status = SessionStatus.ITERATING
                state.error_message = str(exc)
                await sm.save_session(state)
                await asyncio.sleep(2)
                continue

        if parsed is None:
            try:
                response_fmt = None
                _run_loop_system_prompt = get_system_prompt() + _ls._build_global_blacklist_prompt()
                if _ls._logic_library is not None:
                    try:
                        _logic_templates = _ls._logic_library.get_templates_for_direction(exploration_direction)
                        if _logic_templates:
                            _template_lines = ["\n\n▶ LOGIC-GUIDED TEMPLATES (direction=" + exploration_direction + "):"]
                            for _lti, _lt in enumerate(_logic_templates[:5]):
                                _template_lines.append(f"  {_lti+1}. {_lt}")
                            _template_lines.append("Use these as structural starting points for your alpha expression.")
                            _run_loop_system_prompt += "\n".join(_template_lines)
                    except (OSError, ValueError, RuntimeError) as exc:
                        logger.debug("[%s] logic_library get_templates_for_direction 失败，继续使用基础 prompt: %s", session_id, exc)
                if settings.RAG_TOOL_CALL_ENABLED and _ls._rag_engine and _ls._rag_engine.is_ready:
                    _run_loop_system_prompt += "\n\nUse search_operators, search_fields, and search_financial_logic tools to actively retrieve operators and data fields beyond the core set."

                if settings.RAG_ENABLED and settings.RAG_TOOL_CALL_ENABLED and _ls._rag_engine and _ls._rag_engine.is_ready:
                    _run_loop_system_prompt += build_dynamic_context(rag_context, global_blacklist=_global_blacklist_entries, experience_cards=_exp_cards_dicts or None)
                    if _sched_recommended_fields:
                        _run_loop_system_prompt += f"\n\n▶ SCHEDULER RECOMMENDED FIELDS (prioritize these in your expression): {', '.join(_sched_recommended_fields)}"
                    if _ls._market_state_inferencer is not None:
                        try:
                            _ls._algo_tick("market_state_infer")
                            _ms_summary = _ls._market_state_inferencer.get_market_state_summary()
                            if _ms_summary.get("current_dominant"):
                                _run_loop_system_prompt += f"\n\nCurrent market state: dominant strategy is {_ms_summary['current_dominant']}. Avg Sharpes: {_ms_summary.get('avg_sharpes_by_direction', {})}"
                        except (OSError, ValueError, RuntimeError):
                            pass
                    raw_response, _run_loop_tool_results = await llm_client.generate_with_tools(
                        system_prompt=_run_loop_system_prompt,
                        history=effective_history,
                        user_msg=user_msg,
                        session_id=session_id,
                        cycle=global_cycle,
                        tools=RAG_TOOL_SCHEMAS,
                        tool_executor=_make_tool_executor(_ls._rag_engine),
                        grammar=_ls._fastexpr_grammar,
                    )
                    if _run_loop_tool_results:
                        _ls._monitor.record("STEP", "rag_tools", "tool_call", f"tools={list(_run_loop_tool_results.keys())}", session_id=session_id)
                else:
                    if rag_context:
                        _run_loop_system_prompt += build_dynamic_context(rag_context, global_blacklist=_global_blacklist_entries, experience_cards=_exp_cards_dicts or None)
                    elif _global_blacklist_entries:
                        _run_loop_system_prompt += build_dynamic_context(global_blacklist=_global_blacklist_entries, experience_cards=_exp_cards_dicts or None)
                    if _sched_recommended_fields:
                        _run_loop_system_prompt += f"\n\n▶ SCHEDULER RECOMMENDED FIELDS (prioritize these in your expression): {', '.join(_sched_recommended_fields)}"

                    try:
                        from openalpha_brain.core.loop_state import _last_unexplored_directions, _last_diversity_stats, _diversity_last_cycle
                        if (_last_unexplored_directions and global_cycle > 1
                                and _diversity_last_cycle > 0
                                and (global_cycle - _diversity_last_cycle) <= 15):
                            _unexp = _last_unexplored_directions[:6]
                            if _unexp:
                                _div_hint_lines = ["\n\n▶ FEATURE_MAP DIVERSITY GUIDANCE (bias exploration toward these under-explored regions):"]
                                for _ui, _ud in enumerate(_unexp):
                                    _div_hint_lines.append(f"  {_ui+1}. {_ud}")
                                if _last_diversity_stats:
                                    _cov = _last_diversity_stats.get("coverage", 0) * 100
                                    _div_hint_lines.append(f"  Current feature-space coverage: {_cov:.1f}%")
                                    if _cov > 85.0:
                                        _div_hint_lines.append("  ⚠️ Coverage is HIGH — strongly prefer unexplored directions above to avoid redundancy.")
                                user_msg += "\n".join(_div_hint_lines)
                                logger.info(
                                    "[%s] cycle=%d Injected %d unexplored direction hints into generation prompt",
                                    session_id, global_cycle, len(_unexp),
                                )
                    except (OSError, ValueError, RuntimeError) as exc:
                        logger.debug("[%s] dynamic_skill_library 注入失败，继续执行 LLM 调用: %s", session_id, exc)

                    raw_response = await llm_client.generate(
                        system_prompt=_run_loop_system_prompt,
                        history=effective_history,
                        user_msg=user_msg,
                        session_id=session_id,
                        cycle=global_cycle,
                        response_format=response_fmt,
                        grammar=_ls._fastexpr_grammar,
                    )
            except llm_client.LLMError as exc:
                logger.error("[%s] cycle=%d LLM permanent failure: %s", session_id, global_cycle, exc)
                state.status = SessionStatus.ERROR
                state.error_message = str(exc)
                await sm.save_session(state)
                if _ls._heartbeat:
                    _ls._heartbeat.remove(session_id)
                return

        if _ls._heartbeat:
            _ls._heartbeat.touch(session_id)

        # Grow history (in memory — no disk write)
        state.conversation_history.append({"role": "user", "content": user_msg})

        # ── 3. Parse ────────────────────────────────────────────────────────
        state.status = SessionStatus.PARSING
        await sm.save_session(state)

        if parsed is None:
            parsed = parse_alpha_json(raw_response)
            if parsed is None:
                parsed = parser.parse_alpha_output(raw_response)

            if parsed is None:
                await _handle_parse_failure(state, session_id, global_cycle, raw_response)
                await asyncio.sleep(1)
                continue

        parsed = cast(dict[str, Any], parsed)

        decision: str = parsed.get("decision", "ITERATE")
        expression = parsed.get("expression", "")
        fingerprint_dict: dict[str, Any] = cast(dict[str, Any], parsed.get("fingerprint", {}))
        family: str = parsed.get("family") or "Unknown"
        ast_topology: Any = parsed.get("ast_topology")
        ast_collision: list[Any] = parsed.get("ast_collision", [])
        simulation_payload: dict[str, Any] | None = cast(dict[str, Any] | None, parsed.get("simulation_payload"))
        logger.info("[%s] MONITOR: llm_expression: expr_len=%d expr_head=%s direction=%s", session_id, len(expression), expression[:100], exploration_direction)

        if expression and orchestrator is None:
            _gates_standalone = getattr(_ls, '_generation_gates', None)
            if _gates_standalone is not None:
                expression, _gate_standalone_report = await _gates_standalone.apply_with_retry(
                    hypothesis_direction=family or exploration_direction,
                    hypothesis_mechanism="",
                    hypothesis_nl=parsed.get("rationale", ""),
                    expression=expression,
                    regenerate_fn=_gate_regenerate_fn,
                )
            else:
                _gate_standalone_report = await _apply_generation_gates(
                    expression=expression,
                    hypothesis_direction=family or exploration_direction,
                    hypothesis_nl=parsed.get("rationale", ""),
                    hypothesis_mechanism="",
                )
            if not _gate_standalone_report.passed:
                logger.warning(
                    "[%s] cycle=%d GENERATION GATES FAILED (standalone): score=%.3f failed=%s",
                    session_id, global_cycle, _gate_standalone_report.overall_score, _gate_standalone_report.failed_gates,
                )
                _gate_correction = _gate_standalone_report.correction_prompt
                state.conversation_history.append({
                    "role": "assistant",
                    "content": f"Expression: {expression}\n\n{_gate_correction}",
                })
                state.status = SessionStatus.ITERATING
                await sm.save_session(state)
                continue
            logger.info(
                "[%s] cycle=%d GENERATION GATES PASSED (standalone): score=%.3f",
                session_id, global_cycle, _gate_standalone_report.overall_score,
            )

        if expression and _ls._strategy_classifier is not None:
            try:
                _ls._algo_tick("early_strategy_classification")
                early_profile = await _ls._strategy_classifier.classify(expression)
                logger.info(
                    "[%s] cycle=%d Early strategy classification: direction=%s time_horizon=%s mechanism=%s complexity=%d",
                    session_id, global_cycle, early_profile.direction, early_profile.time_horizon, early_profile.mechanism, early_profile.complexity,
                )
                if early_profile.direction != exploration_direction.split("_")[0] if "_" in exploration_direction else exploration_direction:
                    logger.warning(
                        "[%s] cycle=%d Direction mismatch: expression classified as '%s' but exploring '%s'",
                        session_id, global_cycle, early_profile.direction, exploration_direction,
                    )
            except (OSError, ValueError, RuntimeError) as exc:
                logger.warning("[%s] cycle=%d Early strategy classification failed: %s", session_id, global_cycle, exc)

        if expression and settings.RAG_ENABLED and settings.RAG_TOOL_CALL_ENABLED and _ls._rag_engine and _ls._rag_engine.is_ready:
            _allowed_fields = _extract_allowed_fields_from_tool_results(_run_loop_tool_results)
            if not _allowed_fields and rag_context and rag_context.get("field_ids"):
                _allowed_fields = {f.lower() for f in rag_context["field_ids"]}
            if _allowed_fields:
                _is_valid, _invalid = validate_expression_fields(expression, _allowed_fields)
                if not _is_valid:
                    logger.warning(
                        "[%s] cycle=%d Invalid fields in expression: %s — auto-repairing",
                        session_id, global_cycle, _invalid,
                    )
                    _ls._algo_tick("ast_repair")
                    _repaired = auto_repair_expression(expression, _allowed_fields, _invalid)
                    if _repaired != expression:
                        logger.info(
                            "[%s] cycle=%d Auto-repaired expression: %s -> %s",
                            session_id, global_cycle, expression, _repaired,
                        )
                        expression = _repaired
                        parsed["expression"] = _repaired

        # Update dataset_usage (in memory)
        dataset_family = fingerprint_dict.get("dataset") or "Unknown"
        state.dataset_usage[dataset_family] = state.dataset_usage.get(dataset_family, 0) + 1

        if expression and _ls._strategy_classifier is not None:
            try:
                _ls._algo_tick("strategy_dup_check")
                similar_by_expr = await _ls._strategy_classifier.find_similar_by_embedding(expression, top_k=3)
                if similar_by_expr:
                    high_sim_expr = [s for s in similar_by_expr if s.get("similarity", 0) > 0.9]
                    if high_sim_expr:
                        logger.warning("[%s] cycle=%d Expression-level strategy duplication — %d similar (sim>0.9)", session_id, global_cycle, len(high_sim_expr))
            except (OSError, ValueError, RuntimeError) as exc:
                logger.warning("[%s] cycle=%d 策略去重检查 (find_similar_by_embedding) 失败，跳过重复检测: %s", session_id, global_cycle, exc)

        logger.info(
            "[%s] cycle=%d parsed — decision=%s family=%s topology=%s",
            session_id, global_cycle, decision, family, ast_topology,
        )

        _economic_rationale = _extract_economic_rationale(raw_response)
        if not _economic_rationale:
            _economic_rationale = parsed.get("rationale", "")
        _rationale_verification = _apply_economic_rationale_verification(
            expression, _economic_rationale, raw_response, state,
            session_id, global_cycle, exploration_direction,
        )

        # ── 4. Validate ────────────────────────────────────────────────────
        state.status = SessionStatus.VALIDATING
        await sm.save_session(state)

        # Expression sanitization: if expression contains natural language,
        # try to extract just the code part
        if expression and not val.validate_syntax(expression).passed:
            clean_expr = _extract_expression_from_llm(expression)
            if clean_expr and clean_expr != expression:
                logger.info(
                    "[%s] cycle=%d sanitized expression: %s → %s",
                    session_id, global_cycle, expression[:60], clean_expr[:60],
                )
                expression = clean_expr

        syntax_result = val.validate_syntax(expression)
        metrics_result = val.validate_metrics(parsed)
        collision = val.fingerprint_collision(fingerprint_dict, state.fingerprint_memory)

        sharpe_est = 0.0
        if expression:
            try:
                sharpe_est = val.estimate_sharpe_likelihood(expression)
                logger.info("[%s] cycle=%d Sharpe likelihood estimate: %.2f for %s",
                            session_id, global_cycle, sharpe_est, expression[:60])
            except (OSError, ValueError, RuntimeError) as exc:
                logger.debug("[%s] cycle=%d Sharpe likelihood estimate failed: %s", session_id, global_cycle, exc)
        logger.info("[%s] cycle=%d ALPHA_GENERATED: expr=%s direction=%s sharpe_est=%.2f", session_id, global_cycle, expression[:200], exploration_direction, sharpe_est)
        _ev.emit(EVENT_ALPHA_GENERATED, {"session_id": session_id, "cycle": global_cycle, "expression": expression[:200], "direction": exploration_direction, "sharpe_est": sharpe_est})

        # H1: AST repair on syntax failure in main loop
        _was_repaired = False
        if not syntax_result.passed:
            _extract_hallucinations_from_failures(state, syntax_result)
            repaired_expr = expression
            repair_entries = []
            if len(expression) < 500:
                _ls._algo_tick("ast_repair")
                repaired_expr, repair_entries = repair_expression(expression)
            if repair_entries:
                state.hallucination_log.extend(repair_entries)
                logger.info(
                    "[%s] cycle=%d main loop AST repair: %s",
                    session_id, global_cycle,
                    "; ".join(e["error_message"] for e in repair_entries),
                )
                if repaired_expr != expression:
                    expression = repaired_expr
                    _was_repaired = True
                    syntax_result = val.validate_syntax(expression)
                    if syntax_result.passed:
                        logger.info("[%s] cycle=%d AST repair succeeded", session_id, global_cycle)

            if not syntax_result.passed and (_ls._failure_lib or _ls._success_lib):
                try:
                    from openalpha_brain.knowledge.rag_engine import auto_debug_loop as _auto_debug
                    _ls._algo_tick("auto_debug_loop")
                    debugged_expr, debug_ok = await _auto_debug(
                        generate_fn=llm_client.generate,
                        validate_fn=val.validate_syntax,
                        initial_expr=expression,
                        max_rounds=2,
                    )
                    if debug_ok and debugged_expr:
                        expression = debugged_expr
                        _was_repaired = True
                        syntax_result = val.validate_syntax(expression)
                        logger.info("[%s] cycle=%d auto_debug_loop succeeded: %s", session_id, global_cycle, expression[:60])
                except (OSError, ValueError, RuntimeError):
                    logger.warning("[%s] cycle=%d auto_debug_loop failed", session_id, global_cycle)
        logger.info("[%s] cycle=%d ALPHA_PARSED: valid=%s ast_repaired=%s", session_id, global_cycle, syntax_result.passed, _was_repaired)

        # AST originality + complexity check
        alpha_id_prelim = f"A{len(state.passed_alphas) + 1:03d}"
        algo_result = val.validate_alpha(alpha_id_prelim, expression)

        # Topology collision check
        topology_collision = False
        if ast_topology and ast_topology in state.topology_map:
            existing_status = state.topology_map[ast_topology]
            if existing_status in ("FAILED", "CROWDED"):
                topology_collision = True
                logger.info(
                    "[%s] cycle=%d topology collision: '%s' is %s",
                    session_id, global_cycle, ast_topology, existing_status,
                )

        # Dataset exhaustion check
        dataset_exhausted = (
            dataset_family != "Unknown"
            and state.dataset_usage.get(dataset_family, 0) >= 3
            and all(
                fc.get("fingerprint", {}).get("dataset") == dataset_family
                for fc in state.failure_catalog[-3:]
            )
        )
        if dataset_exhausted:
            logger.info(
                "[%s] cycle=%d dataset exhausted: '%s' used %d times with consistent failure",
                session_id, global_cycle, dataset_family,
                state.dataset_usage.get(dataset_family, 0),
            )

        all_failures = syntax_result.failures + metrics_result.failures + algo_result.failures
        all_warnings = syntax_result.warnings + metrics_result.warnings + algo_result.warnings

        if all_warnings:
            logger.info("[%s] cycle=%d warnings: %s", session_id, global_cycle, all_warnings)

        _vector_duplicate = False
        if expression and _ls._evo_db is not None and settings.RAG_ENABLED:
            try:
                _ls._algo_tick("vector_duplicate_check")
                similar = await _ls._evo_db.find_similar_by_embedding(expression, top_k=3)
                for s in similar[:1]:
                    if s.get("similarity", 0) > 0.95:
                        _vector_duplicate = True
                        logger.info(
                            "[%s] cycle=%d Vector duplicate detected: similarity=%.3f with record %s",
                            session_id, global_cycle, s.get("similarity", 0), s.get("record", {}).get("record_id", ""),
                        )
                        break
            except (OSError, ValueError, RuntimeError):
                pass

        _semantic_alignment_score: float | None = None
        if expression:
            try:
                _ls._algo_tick("semantic_alignment_check")
                _sa_hypothesis = Hypothesis(
                    direction=exploration_direction,
                    asset_class="equity",
                    time_horizon="medium-term",
                    mechanism="",
                    natural_language="",
                )
                _semantic_alignment_score = _check_semantic_alignment(_sa_hypothesis, expression)
                if _semantic_alignment_score < 0.3:
                    logger.warning(
                        "[%s] cycle=%d Low semantic alignment score=%.3f for expression: %s",
                        session_id, global_cycle, _semantic_alignment_score, expression[:80],
                    )
            except (OSError, ValueError, RuntimeError):
                pass

        # ── 5. Decision branching ───────────────────────────────────────────
        # Register assistant turn (in memory)
        state.conversation_history.append({"role": "assistant", "content": raw_response})

        if ast_topology:
            if ast_topology not in state.topology_map:
                state.topology_map[ast_topology] = "EXPLORING"

        if decision == "REJECT" or collision or topology_collision or dataset_exhausted:
            reason = "REJECT decision"
            if collision:          reason = "fingerprint collision"
            if topology_collision: reason = "AST topology collision"
            if dataset_exhausted:  reason = "dataset exhausted"
            _ev.emit(EVENT_ALPHA_REJECTED, {"session_id": session_id, "cycle": global_cycle, "expression": (expression or "")[:100], "reason": reason, "direction": exploration_direction})
            await _handle_reject(state, session_id, global_cycle, fingerprint_dict, ast_topology, reason)
            await asyncio.sleep(1)
            continue

        if all_failures or decision == "ITERATE":
            if decision == "ITERATE" and not all_failures and simulation_payload and simulation_payload.get("regular"):
                logger.info(
                    "[%s] cycle=%d Overriding ITERATE→SUBMIT (no real failures, valid payload)",
                    session_id, global_cycle,
                )
                decision = "SUBMIT"
            else:
                await _handle_iterate(state, session_id, global_cycle, parsed, expression,
                                      fingerprint_dict, ast_topology, all_failures, decision)
                await asyncio.sleep(1)
                continue

        # ── 6. PASS path ────────────────────────────────────────────────────
        alpha = _build_alpha(state, parsed, expression, fingerprint_dict,
                             str(family), ast_topology, ast_collision, cast(dict, simulation_payload),
                             metrics_result, global_cycle, decision,
                             economic_rationale=_economic_rationale,
                             rationale_verification=_rationale_verification)
        alpha.exploration_direction = exploration_direction
        alpha.template_id = _sched_template_id
        alpha.family_id = _sched_family_id
        alpha.semantic_alignment_score = _semantic_alignment_score

        if orchestrator is None and _ls._reflection_engine is not None and expression:
            try:
                _ls._algo_tick("self_critique")
                critique_result = await _ls._reflection_engine.self_critique(
                    hypothesis=exploration_direction,
                    expression=expression,
                )
                if critique_result and (critique_result.consistency_score or 1.0) < 0.3:
                    logger.warning("[%s] cycle=%d Self-critique consistency_score=%.2f < 0.3 — expression may not match hypothesis", session_id, global_cycle, critique_result.consistency_score or 0)
            except (OSError, ValueError, RuntimeError):
                pass

            try:
                _recent_reflections = _ls._reflection_engine.get_recent_reflections(n=3)
                if _recent_reflections:
                    _refl_lines = ["\n▶ PAST REFLECTIONS (avoid repeating these mistakes):"]
                    for _ri, _r in enumerate(_recent_reflections):
                        _stage = _r.get("failure_stage", "unknown")
                        _reason = _r.get("failure_reason", "")
                        _fix = _r.get("suggested_fix", "")
                        _refl_lines.append(f"  {_ri+1}. [{_stage}] {_reason} → Fix: {_fix}")
                    user_msg += "\n" + "\n".join(_refl_lines)
            except (OSError, ValueError, RuntimeError):
                pass

        if _ls._scheduler is not None:
            try:
                if alpha.template_id and alpha.family_id:
                    _ls._scheduler.record_result(alpha.template_id, alpha.family_id, exploration_direction, reward=REWARD_VALIDATOR_PASS)
                else:
                    _ls._scheduler.record_direction_result(exploration_direction, reward=REWARD_VALIDATOR_PASS)
            except (OSError, ValueError, RuntimeError):
                pass
        _ev.emit(EVENT_MAB_FEEDBACK, {"session_id": session_id, "cycle": global_cycle, "direction": exploration_direction, "expression": expression[:100], "reward": REWARD_VALIDATOR_PASS})

        _consecutive_errors = 0

        logger.info(
            "[%s] cycle=%d PASS — alpha_id=%s family=%s decision=%s",
            session_id, global_cycle, alpha.alpha_id, family, decision,
        )
        _ev.emit(EVENT_ALPHA_VALIDATED, {"session_id": session_id, "cycle": global_cycle, "alpha_id": alpha.alpha_id, "expression": expression[:120], "direction": exploration_direction, "family": str(family), "decision": decision})

        if global_cycle < settings.MAX_CYCLES:
            success_msg = build_success_feedback(
                alpha_id=alpha.alpha_id,
                cycle=global_cycle,
                next_cycle=global_cycle + 1,
                fingerprint=fingerprint_dict,
                all_fingerprints=state.fingerprint_memory,
            )
            state.conversation_history.append({"role": "user", "content": success_msg})

        await sm.save_session(state)

        # ── 7. BRAIN submission ─────────────────────────────────────────────
        if settings.BRAIN_SUBMIT_ENABLED:
            if not alpha.simulation_payload:
                alpha.simulation_payload = {"settings": {}, "regular": expression}
            elif not alpha.simulation_payload.get("regular"):
                alpha.simulation_payload["regular"] = expression
            state.status = SessionStatus.SUBMITTING
            state.brain_mutation_count = 0
            state.current_brain_alpha_id = None
            _ls._log(state, "SUBMIT", f"Submitting {alpha.alpha_id} to BRAIN…", {"expr": expression[:80]})
            await sm.save_session(state)

            _brain_settings = alpha.simulation_payload.get("settings", {}) if alpha.simulation_payload else {}
            logger.info("[%s] cycle=%d BRAIN_SUBMIT: expr=%s universe=%s delay=%s decay=%s", session_id, global_cycle, expression[:100], _brain_settings.get("universe", "N/A"), _brain_settings.get("delay", "N/A"), _brain_settings.get("decay", "N/A"))

            if _ls._whitelist_mgr is not None:
                try:
                    allowed_fields = _ls._whitelist_mgr.get_allowed_fields()
                    eliminated = _ls._whitelist_mgr.eliminated_fields
                    solidified = _ls._whitelist_mgr.solidified_fields
                    _ls._algo_tick("whitelist_pre_submit_check")
                    import re as _re_whitelist
                    _expr_fields_whitelist = set(_re_whitelist.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b(?!\s*\()', expression))
                    _expr_fields_whitelist -= {'and', 'or', 'not', 'if', 'else', 'true', 'false', 'nan', 'inf'}
                    _expr_fields_whitelist -= val.PERMITTED_OPERATORS
                    _invalid_fields = [f for f in _expr_fields_whitelist if f not in allowed_fields]
                    if _invalid_fields:
                        logger.warning(
                            "[%s] cycle=%d WHITELIST_FILTER: expression contains %d non-whitelisted fields: %s (allowed=%d, solidified=%d, eliminated=%d)",
                            session_id, global_cycle, len(_invalid_fields), _invalid_fields[:5],
                            len(allowed_fields), len(solidified), len(eliminated),
                        )
                    else:
                        logger.info(
                            "[%s] cycle=%d WHITELIST_PASS: all %d fields in whitelist (allowed=%d, solidified=%d, eliminated=%d)",
                            session_id, global_cycle, len(_expr_fields_whitelist),
                            len(allowed_fields), len(solidified), len(eliminated),
                        )

                    _overuse_warnings = _ls._whitelist_mgr.detect_field_overfit()
                    if _overuse_warnings:
                        for _ow in _overuse_warnings[:3]:
                            logger.warning("[%s] cycle=%d OVERUSE_DETECTED: %s", session_id, global_cycle, _ow.get("message", ""))
                except (OSError, ValueError, RuntimeError):
                    pass

            if _ls._feature_map is not None and global_cycle % 10 == 0:
                try:
                    _diversity_stats = _ls._feature_map.get_diversity_stats()
                    _unexplored = _ls._feature_map.get_unexplored_directions()
                    from openalpha_brain.core.loop_state import _last_diversity_stats, _last_unexplored_directions, _diversity_last_cycle
                    from openalpha_brain.core import loop_state as _ls_mod
                    _ls_mod._last_diversity_stats = _diversity_stats
                    _ls_mod._last_unexplored_directions = list(_unexplored) if _unexplored else []
                    _ls_mod._diversity_last_cycle = global_cycle
                    logger.info(
                        "[%s] cycle=%d FEATURE_MAP_STATS: coverage=%.2f%% (%d/%d) filled, unexplored=%s",
                        session_id, global_cycle,
                        _diversity_stats.get("coverage", 0) * 100,
                        _diversity_stats.get("filled_cells", 0),
                        _diversity_stats.get("total_cells", 0),
                        _unexplored,
                    )
                    _coverage_pct = _diversity_stats.get("coverage", 0) * 100
                    if _coverage_pct > 90.0:
                        logger.warning(
                            "[%s] cycle=%d LOW DIVERSITY WARNING: feature space coverage=%.1f%% — consider pivoting to unexplored directions: %s",
                            session_id, global_cycle, _coverage_pct, _unexplored[:5],
                        )
                except (OSError, ValueError, RuntimeError):
                    pass

                _mutator = getattr(_ls, '_semantic_mutator', None)
                if _mutator is not None and _ls._feature_map is not None and _ls._evo_db is not None:
                    _coverage = _diversity_stats.get("coverage", 1.0)
                    if _coverage < 0.5 and global_cycle % 5 == 0:
                        try:
                            _ls._algo_tick("mcts_explore_unexplored")
                            _mcts_results = await _mutator.explore_unexplored_regions(
                                _ls._feature_map, _ls._evo_db, top_k=2,
                            )
                            for _mr in _mcts_results:
                                if _mr.get("expression"):
                                    vid = f"mcts_{expression_hash(_mr['expression'])[:8]}"
                                    variant_alpha = AlphaResult(
                                        alpha_id=vid,
                                        family=alpha.family,
                                        expression=_mr["expression"],
                                        rationale=f"MCTS探索: {_mr.get('feature_description', '')}",
                                        metrics=AlphaMetrics(),
                                        fingerprint=AlphaFingerprint(),
                                        decision="SUBMIT",
                                        simulation_payload={
                                            "regular": _mr["expression"],
                                            "settings": dict(alpha.simulation_payload.get("settings", {})) if alpha.simulation_payload else {},
                                        },
                                        cycle_num=alpha.cycle_num,
                                        passed=True,
                                        exploration_direction=exploration_direction,
                                        pipeline_status=PipelineStatus.QUEUED,
                                    )
                                    await sm.add_alpha(session_id, variant_alpha)
                                    await pool.enqueue(vid, priority="low")
                                    logger.info(
                                        "[%s] MCTS explored alpha enqueued: %s (%s)",
                                        session_id, vid, _mr.get("feature_description", "")[:60],
                                    )
                        except (OSError, ValueError, RuntimeError):
                            logger.warning("[%s] MCTS explore_unexplored_regions failed", session_id, exc_info=True)

                if _ls._evo_db is not None and hasattr(_ls, '_crossover_engine') and _ls._crossover_engine is not None:
                    try:
                        _successful_records = _ls._evo_db.sample_inspiration(n=6, min_sharpe=0.3)
                        if _ls._feature_map is not None:
                            _elite_sample = _ls._feature_map.sample_elite(direction=exploration_direction)
                            if _elite_sample and _elite_sample.get("expr"):
                                _elite_fitness = _elite_sample.get("fitness", 0.0)
                                _elite_sharpe = _elite_sample.get("sharpe", 0.0)
                                if _elite_fitness > 0.3:
                                    from openalpha_brain.knowledge.evolution_db import EvolutionRecord
                                    _elite_record = EvolutionRecord(
                                        expr=_elite_sample["expr"],
                                        direction=exploration_direction,
                                        sharpe=_elite_sharpe,
                                        fitness=_elite_fitness,
                                        turnover=_elite_sample.get("turnover"),
                                    )
                                    _successful_records.insert(0, _elite_record)
                                    logger.info(
                                        "[%s] FeatureMap sample_elite injected as crossover parent: fitness=%.4f sharpe=%.4f",
                                        session_id, _elite_fitness, _elite_sharpe,
                                    )
                        if len(_successful_records) >= 2 and global_cycle % 7 == 0:
                            _ls._algo_tick("llm_semantic_crossover_explore")
                            for _i in range(min(2, len(_successful_records) - 1)):
                                _p1 = _successful_records[_i]
                                _p2 = _successful_records[_i + 1]
                                _llm_cross_results = await _ls._crossover_engine._crossover.semantic_crossover_via_llm(
                                    llm_generate=llm_client.generate,
                                    expr1=_p1.expr,
                                    expr2=_p2.expr,
                                    context1={"direction": _p1.direction, "sharpe": _p1.sharpe},
                                    context2={"direction": _p2.direction, "sharpe": _p2.sharpe},
                                )
                                for _lcr in _llm_cross_results:
                                    if _lcr.child_expression and _lcr.originality_score >= 0.4:
                                        _lcvid = f"llmcr_{expression_hash(_lcr.child_expression)[:8]}"
                                        _llmcr_alpha = AlphaResult(
                                            alpha_id=_lcvid,
                                            family=alpha.family,
                                            expression=_lcr.child_expression,
                                            rationale=f"LLM语义交叉: {_lcr.crossover_point[:80]}",
                                            metrics=AlphaMetrics(),
                                            fingerprint=AlphaFingerprint(),
                                            decision="SUBMIT",
                                            simulation_payload={
                                                "regular": _lcr.child_expression,
                                                "settings": dict(alpha.simulation_payload.get("settings", {})) if alpha.simulation_payload else {},
                                            },
                                            cycle_num=alpha.cycle_num,
                                            passed=True,
                                            exploration_direction=_p1.direction,
                                            pipeline_status=PipelineStatus.QUEUED,
                                        )
                                        await sm.add_alpha(session_id, _llmcr_alpha)
                                        await pool.enqueue(_lcvid, priority="low")
                                        logger.info(
                                            "[%s] LLM semantic crossover enqueued: %s (%s)",
                                            session_id, _lcvid, _lcr.crossover_point[:60],
                                        )
                            logger.info("[%s] LLM semantic crossover: explored %d parent pairs", session_id, min(2, len(_successful_records) - 1))
                    except (OSError, ValueError, RuntimeError):
                        logger.warning("[%s] LLM semantic crossover exploration failed", session_id, exc_info=True)

            brain_result = await _submit_to_brain(alpha, session_id, global_cycle)

            logger.info("[%s] MONITOR: brain_submit: status=%s sharpe=%s fitness=%s turnover=%s direction=%s alpha_id=%s", session_id, brain_result.status.value if brain_result.status else "ERROR", brain_result.real_sharpe, brain_result.real_fitness, brain_result.real_turnover, alpha.exploration_direction or exploration_direction, brain_result.alpha_id)
            logger.info("[%s] cycle=%d BRAIN_RESULT: status=%s sharpe=%.4f fitness=%.4f turnover=%.2f drawdown=%.2f", session_id, global_cycle, brain_result.status.value if brain_result.status else "ERROR", brain_result.real_sharpe or 0.0, brain_result.real_fitness or 0.0, brain_result.real_turnover or 0.0, brain_result.real_drawdown or 0.0)
            _ev.emit(EVENT_BRAIN_RESULT, {"session_id": session_id, "cycle": global_cycle, "alpha_id": alpha.alpha_id, "expression": expression[:120], "status": brain_result.status.value if brain_result.status else "ERROR", "sharpe": brain_result.real_sharpe, "fitness": brain_result.real_fitness, "turnover": brain_result.real_turnover, "returns": brain_result.real_returns, "drawdown": brain_result.real_drawdown, "direction": exploration_direction})

            _engine = getattr(_ls, '_crossover_engine', None)
            if _engine is not None:
                _accepted = brain_result.status == BrainSimStatus.PASS
                _reject_reason = None
                if not _accepted:
                    if brain_result.gate_failures:
                        _reject_reason = "; ".join(brain_result.gate_failures[:3])
                    elif brain_result.status == BrainSimStatus.ERROR:
                        _reject_reason = "BRAIN_ERROR"
                    else:
                        _reject_reason = "BRAIN_FAIL"
                _engine.record_alpha_outcome(
                    direction=exploration_direction,
                    sharpe=brain_result.real_sharpe or 0.0,
                    turnover=brain_result.real_turnover,
                    complexity=len(expression or ""),
                    accepted=_accepted,
                    reject_reason=_reject_reason,
                )

                if _accepted:
                    _traj = AlphaTrajectory(
                        hypothesis_direction=exploration_direction,
                        hypothesis_mechanism=alpha.hypothesis_mechanism if hasattr(alpha, 'hypothesis_mechanism') else "",
                        expression_versions=[expression] if expression else [],
                        final_status="PASS",
                        final_sharpe=brain_result.real_sharpe,
                    )
                    _traj.add_brain_feedback({
                        "status": brain_result.status.value if brain_result.status else "UNKNOWN",
                        "sharpe": brain_result.real_sharpe,
                        "fitness": brain_result.real_fitness,
                        "turnover": brain_result.real_turnover,
                    })
                    _engine.record_trajectory(_traj, {"id": brain_result.alpha_id or "", "expression": expression, "direction": exploration_direction})

            if _ls._alpha_channel is not None and brain_result is not None and brain_result.real_sharpe is not None:
                try:
                    _ls._algo_tick("alpha_channel_submit")
                    route = await _ls._alpha_channel.submit(
                        alpha_id=brain_result.alpha_id or "",
                        sharpe=brain_result.real_sharpe,
                        expression=expression or "",
                        direction=alpha.exploration_direction or exploration_direction,
                    )
                    logger.info("[%s] MONITOR: alpha_channel: route=%s sharpe=%.2f", session_id, route, brain_result.real_sharpe)
                    if route == "stream" and _ls._alpha_channel_integrator is not None:
                        _ls._algo_tick("alpha_channel_stream_process")
                        await _ls._alpha_channel_integrator.process_stream_alpha({
                            "alpha_id": brain_result.alpha_id or "",
                            "sharpe": brain_result.real_sharpe,
                            "expression": expression or "",
                            "direction": alpha.exploration_direction or exploration_direction,
                        })
                    _ls._algo_tick("alpha_channel_batch")
                    batch = await _ls._alpha_channel.get_batch()
                    if batch and _ls._alpha_channel_integrator is not None:
                        logger.info("[%s] Processing batch of %d low-Sharpe alphas", session_id, len(batch))
                        _ls._algo_tick("alpha_channel_batch_process")
                        await _ls._alpha_channel_integrator.process_batch_alphas(batch)
                except (OSError, ValueError, RuntimeError):
                    pass

            if _ls._heartbeat:
                _ls._heartbeat.touch(session_id)

            state.current_brain_alpha_id = brain_result.alpha_id
            _log_brain_result(state, brain_result, expression, attempt=0)
            await sm.save_session(state)

            if (settings.PARAM_OPTIMIZATION_ENABLED
                    and _ls._param_optimizer
                    and brain_result.status == BrainSimStatus.FAIL
                    and brain_result.real_sharpe is not None
                    and _ls._param_optimizer.should_optimize(
                        brain_result.real_sharpe, brain_client.GATE_SHARPE_MIN)):
                _ls._log(state, "PARAM_OPT",
                     f"Sharpe={brain_result.real_sharpe:.3f} near gate "
                     f"{brain_client.GATE_SHARPE_MIN} — running param optimization",
                     {"sharpe": brain_result.real_sharpe})
                await sm.save_session(state)

                _ls._algo_tick("param_optimization")
                opt_result = await _run_param_optimization(
                    expression, session_id, global_cycle,
                )

                state = await sm.load_session(session_id)
                assert state is not None
                if opt_result is not None and opt_result.real_sharpe is not None:
                    if brain_result.real_sharpe is None or opt_result.real_sharpe > brain_result.real_sharpe:
                        logger.info(
                            "[%s] cycle=%d Param optimization improved Sharpe: %.3f → %.3f",
                            session_id, global_cycle,
                            brain_result.real_sharpe or 0, opt_result.real_sharpe,
                        )
                        brain_result = opt_result
                        _log_brain_result(state, brain_result, expression, attempt=0)
                        cached_params = _ls._param_optimizer.get_cached(expression_hash(expression))
                        if cached_params and alpha.simulation_payload:
                            if "settings" not in alpha.simulation_payload:
                                alpha.simulation_payload["settings"] = {}
                            alpha.simulation_payload["settings"]["decay"] = cached_params["decay"]
                            alpha.simulation_payload["settings"]["neutralization"] = cached_params["neutralization"]
                    else:
                        _ls._log(state, "PARAM_OPT", "Param optimization did not improve Sharpe", {})
                else:
                    _ls._log(state, "PARAM_OPT", "Param optimization found no valid results", {})
                await sm.save_session(state)

            if brain_result.status in (BrainSimStatus.FAIL, BrainSimStatus.ERROR):
                _original_expr = expression
                _original_failure_type = "BRAIN_FAIL"
                if brain_result.status == BrainSimStatus.ERROR:
                    _original_failure_type = "BRAIN_SYNTAX_ERROR"
                    if _ls._scheduler is not None:
                        try:
                            _ls._scheduler.record_direction_result(exploration_direction, penalty=PENALTY_BRAIN_ERROR)
                        except (OSError, ValueError, RuntimeError):
                            pass
                brain_checks = getattr(brain_result, "brain_checks", []) or []
                if brain_checks:
                    failed_checks = [c["name"] for c in brain_checks if c.get("result") == "FAIL"]
                    _original_failure_type = "; ".join(failed_checks[:3]) if failed_checks else "BRAIN_FAIL"
                elif brain_result.gate_failures:
                    _original_failure_type = brain_result.gate_failures[0] if brain_result.gate_failures else "BRAIN_FAIL"

                brain_result = await _brain_improvement_loop(
                    initial_result=brain_result,
                    initial_expression=expression,
                    alpha=alpha,
                    session_id=session_id,
                    global_cycle=global_cycle,
                    grammar=_ls._fastexpr_grammar,
                    resource_dispatcher=None,
                    pipeline_orchestrator=None,
                )

                if _ls._failure_lib and settings.FAILURE_FIX_LIBRARY_ENABLED:
                    try:
                        if brain_result.status == BrainSimStatus.PASS:
                            _ls._algo_tick("failure_fix_add")
                            await _ls._failure_lib.add_failure(
                                expr=_original_expr,
                                failure_type=_original_failure_type,
                                fix_attempt=alpha.expression,
                                fix_success=True,
                                direction=alpha.exploration_direction or exploration_direction,
                                session_id=session_id,
                            )
                    except (OSError, ValueError, RuntimeError):
                        logger.warning("FailureFixLibrary add_failure (fix) failed", exc_info=True)

                if _ls._tool_factory:
                    try:
                        _ls._tool_factory.record_fix_pattern(
                            failure_type=_original_failure_type,
                            fix_attempt=alpha.expression or "",
                            fix_success=brain_result.status == BrainSimStatus.PASS,
                            direction=alpha.exploration_direction or exploration_direction,
                        )
                    except (OSError, ValueError, RuntimeError):
                        logger.warning("ToolFactory record_fix_pattern failed", exc_info=True)

                if _ls._heartbeat:
                    _ls._heartbeat.touch(session_id)

                _engine = getattr(_ls, '_crossover_engine', None)
                if _engine is not None:
                    _accepted = brain_result.status == BrainSimStatus.PASS
                    _reject_reason = None
                    if not _accepted:
                        if brain_result.gate_failures:
                            _reject_reason = "; ".join(brain_result.gate_failures[:3])
                        elif brain_result.status == BrainSimStatus.ERROR:
                            _reject_reason = "BRAIN_ERROR"
                        else:
                            _reject_reason = "BRAIN_FAIL"
                    _engine.record_alpha_outcome(
                        direction=exploration_direction,
                        sharpe=brain_result.real_sharpe or 0.0,
                        turnover=brain_result.real_turnover,
                        complexity=len(alpha.expression or ""),
                        accepted=_accepted,
                        reject_reason=_reject_reason,
                    )

            _consecutive_rejections[exploration_direction] = _consecutive_rejections.get(exploration_direction, 0)
            if not _accepted:
                _consecutive_rejections[exploration_direction] += 1
            else:
                _consecutive_rejections[exploration_direction] = 0

            # Save final result
            state = await sm.load_session(session_id)
            assert state is not None
            alpha.brain = brain_result

            if _ls._feature_map is not None and brain_result.status == BrainSimStatus.PASS:
                feat = _extract_strategy_features(expression, exploration_direction)
                _ls._algo_tick("feature_map_add")
                _ls._feature_map.add_candidate(expression, feat, fitness_score=brain_result.real_fitness or 0.0, sharpe=brain_result.real_sharpe or 0.0, turnover=brain_result.real_turnover)

            _classifier = getattr(_ls, '_strategy_classifier', None)
            if _classifier is not None and brain_result.status == BrainSimStatus.PASS:
                try:
                    _ls._algo_tick("strategy_classify")
                    _brain_dict = {
                        "sharpe": brain_result.real_sharpe,
                        "fitness": brain_result.real_fitness,
                        "turnover": brain_result.real_turnover,
                    }
                    await _classifier.classify(expression, brain_result=_brain_dict)
                    _complementary = _classifier.find_complementary(n=2)
                    if _complementary:
                        logger.info(
                            "[%s] StrategyClassifier: complementary suggestions: %s",
                            session_id, [(c.get("direction"), c.get("reason")) for c in _complementary],
                        )
                        for _comp in _complementary:
                            _comp_dir = _comp.get("direction", "")
                            _comp_score = _comp.get("similarity_score", 0.3)
                            if _comp_dir and _ls._scheduler is not None:
                                try:
                                    _ls._scheduler.record_direction_result(
                                        _comp_dir,
                                        reward=_comp_score * 0.15,
                                    )
                                except (OSError, ValueError, RuntimeError):
                                    pass
                except (OSError, ValueError, RuntimeError):
                    logger.warning("[%s] StrategyClassifier failed", session_id, exc_info=True)

                _current_rejections = _consecutive_rejections.get(exploration_direction, 0)
                if _current_rejections >= _rotation_threshold and _complementary:
                    _force_dir = _complementary[0].get("direction", "")
                    if _force_dir and _force_dir != exploration_direction:
                        logger.warning(
                            "[%s] FORCED ROTATION: direction=%s after %d consecutive rejections → switching to complementary=%s",
                            session_id, exploration_direction, _current_rejections, _force_dir,
                        )
                        _ls._algo_tick("forced_rotation")
                        exploration_direction = _force_dir
                        _consecutive_rejections[exploration_direction] = 0
                        if _ls._scheduler is not None:
                            try:
                                _ls._scheduler.record_direction_result(_force_dir, reward=0.20)
                            except (OSError, ValueError, RuntimeError):
                                pass

            _aligner = getattr(_ls, '_hypothesis_aligner', None)
            if _aligner is not None:
                try:
                    _ls._algo_tick("hypothesis_align")
                    _alignment = _aligner.align(expression, exploration_direction)
                    if _alignment["alignment_level"] in ("contradictory", "weak"):
                        logger.warning(
                            "[%s] HypothesisAligner: %s alignment (R²=%.3f) for direction=%s - %s",
                            session_id, _alignment["alignment_level"].upper(),
                            _alignment["r2_score"], exploration_direction,
                            _alignment["diagnosis"],
                        )
                        _alpha_extra_context = _aligner.build_alignment_feedback(_alignment)
                        _fb_buf_align_loop = getattr(_ls, '_brain_feedback_buffer', None)
                        if _fb_buf_align_loop is None:
                            _ls._brain_feedback_buffer = []
                            _fb_buf_align_loop = _ls._brain_feedback_buffer
                        _fb_buf_align_loop.append({
                            "role": "user",
                            "content": _alpha_extra_context,
                        })
                    else:
                        _alpha_extra_context = ""
                except (OSError, ValueError, RuntimeError):
                    logger.warning("[%s] HypothesisAligner failed", session_id, exc_info=True)
                    _alpha_extra_context = ""

            _engine_for_variants = getattr(_ls, '_crossover_engine', None)
            if _engine_for_variants is not None and expression and brain_result.status == BrainSimStatus.PASS:
                _variant_batch.append({
                    "expression": expression,
                    "id": alpha.alpha_id,
                    "direction": exploration_direction,
                    "fitness": brain_result.real_fitness or 0.0,
                    "sharpe": brain_result.real_sharpe or 0.0,
                    "turnover": brain_result.real_turnover or 0.5,
                    "complexity": len(expression),
                    "originality_score": getattr(brain_result, 'originality_score', 0.6),
                })
                _variant_batch = _variant_batch[-_variant_batch_max_size * 2:]
                if len(_variant_batch) >= _variant_batch_max_size:
                    try:
                        _ls._algo_tick("generate_variants")
                        _variants = _engine_for_variants.generate_variants(
                            successful_alphas=_variant_batch[-_variant_batch_max_size:],
                            direction=exploration_direction,
                            max_variants=3,
                        )
                        for _v in _variants:
                            _vid = f"var_{expression_hash(_v['expression'])[:8]}"
                            _variant_alpha = AlphaResult(
                                alpha_id=_vid,
                                family=alpha.family,
                                expression=_v["expression"],
                                rationale=f"进化变体: {_v.get('source', 'crossover')}",
                                metrics=AlphaMetrics(),
                                fingerprint=AlphaFingerprint(),
                                decision="SUBMIT",
                                simulation_payload={
                                    "regular": _v["expression"],
                                    "settings": dict(alpha.simulation_payload.get("settings", {})) if alpha.simulation_payload else {},
                                },
                                cycle_num=alpha.cycle_num,
                                passed=True,
                                exploration_direction=_v.get("direction") or exploration_direction,
                                pipeline_status=PipelineStatus.QUEUED,
                            )
                            await sm.add_alpha(session_id, _variant_alpha)
                            await pool.enqueue(_vid, priority="low")
                            logger.info(
                                "[%s] Evolution variant enqueued: %s source=%s direction=%s",
                                session_id, _vid, _v.get("source"), _v.get("direction", exploration_direction),
                            )
                        _variant_batch.clear()
                    except (OSError, ValueError, RuntimeError):
                        logger.warning("[%s] generate_variants failed", session_id, exc_info=True)

            hierarchical_reward, hierarchical_level_name = compute_hierarchical_reward_with_penalties(
                brain_result, expression, session_id,
            )
            if hierarchical_level_name:
                alpha.hierarchical_reward = hierarchical_reward
                alpha.hierarchical_level = hierarchical_level_name

            check_margin_efficiency(brain_result, alpha, session_id)

            if (brain_result.real_returns is not None
                    and brain_result.real_margin is not None
                    and brain_result.real_margin > 0
                    and hierarchical_reward < 0.3):
                logger.info("[%s] Hierarchical reward %.2f < 0.3 (below basic) — skipping BRAIN submission, going directly to improvement loop", session_id, hierarchical_reward)
                if _ls._scheduler is not None:
                    try:
                        _ls._scheduler.record_direction_result(alpha.exploration_direction, penalty=PENALTY_BRAIN_FAIL)
                    except (OSError, ValueError, RuntimeError):
                        pass
                _ls._log(state, "LOW_REWARD_SKIP",
                     f"Alpha {alpha.alpha_id} hierarchical_reward={hierarchical_reward:.2f} < 0.3 — skipping BRAIN submission",
                     {"hierarchical_reward": hierarchical_reward})
                if _ls._failure_lib and settings.FAILURE_FIX_LIBRARY_ENABLED:
                    try:
                        _ls._algo_tick("failure_fix_add")
                        await _ls._failure_lib.add_failure(
                            expr=expression,
                            failure_type="LOW_HIERARCHICAL_REWARD",
                            fix_attempt=None,
                            fix_success=False,
                            direction=alpha.exploration_direction or exploration_direction,
                            session_id=session_id,
                        )
                    except (OSError, ValueError, RuntimeError):
                        pass
                state.failure_catalog.append({
                    "fingerprint": alpha.fingerprint.model_dump() if alpha.fingerprint else {},
                    "failure_type": "low_hierarchical_reward",
                    "metric_value": hierarchical_reward,
                    "mutation_tried": "skipped_brain",
                })
                state.status = SessionStatus.ITERATING
                await sm.save_session(state)
                continue

            for i, a in enumerate(state.passed_alphas):
                if a.alpha_id == alpha.alpha_id:
                    state.passed_alphas[i] = alpha
                    break

            if brain_result.status == BrainSimStatus.PASS:
                await record_pass_feedback(
                    brain_result, alpha, expression, exploration_direction,
                    parsed, session_id, hierarchical_reward,
                    exp_card_rule_ids=_exp_card_rule_ids,
                )
                await submit_for_review(brain_result, alpha, session_id, state)

                if _ls._decay_detector is not None and brain_result.alpha_id:
                    try:
                        await _ls._decay_detector.register_alpha(
                            alpha_id=brain_result.alpha_id,
                            expression=expression,
                            fingerprint=alpha.fingerprint if hasattr(alpha, 'fingerprint') else None,
                            direction=exploration_direction,
                            initial_sharpe=brain_result.real_sharpe or 0.0,
                        )
                        logger.info(
                            "[%s] DecayDetector: registered alpha %s dir=%s sharpe=%.3f",
                            session_id, brain_result.alpha_id,
                            exploration_direction,
                            brain_result.real_sharpe or 0.0,
                        )
                    except (OSError, ValueError, RuntimeError) as exc:
                        logger.warning("[%s] DecayDetector: register_alpha failed: %s", session_id, exc)

                state.status = SessionStatus.PASS
                sharpe_str: str = f"{brain_result.real_sharpe:.3f}" if brain_result.real_sharpe is not None else "N/A"
                f_val: str = f"{brain_result.real_fitness:.3f}" if brain_result.real_fitness is not None else "N/A"
                _ls._log(state, "PASS",
                     f"Alpha {alpha.alpha_id} PASSED all BRAIN checks! "
                     f"Sharpe={sharpe_str} Fitness={f_val}",
                     {"brain_id": brain_result.alpha_id})

                hierarchical_reward, pnl_curve, yearly_data = await run_stability_analysis(
                    brain_result, alpha, session_id, state, hierarchical_reward,
                )

            hierarchical_reward = await fetch_correlation_analysis(
                brain_result, alpha, session_id, hierarchical_reward,
            )

            if brain_result.status == BrainSimStatus.PASS:
                hierarchical_reward = await record_evo_and_success(
                    brain_result, alpha, expression, exploration_direction,
                    session_id, hierarchical_reward, parsed=parsed,
                )
                if _ls._evo_db is not None and brain_result.alpha_id:
                    try:
                        _ls._algo_tick("evo_lineage")
                        _lineage = _ls._evo_db.get_lineage(brain_result.alpha_id, depth=3)
                        if _lineage and len(_lineage) > 1:
                            _lineage_dirs = [r.direction for r in _lineage if hasattr(r, 'direction')]
                            logger.info("[%s] Alpha %s lineage: %d ancestors, directions=%s", session_id, brain_result.alpha_id, len(_lineage), _lineage_dirs)
                    except (OSError, ValueError, RuntimeError):
                        pass
            else:
                await record_fail_feedback(
                    brain_result, alpha, expression, exploration_direction,
                    session_id, fingerprint_dict, global_cycle, state,
                    hierarchical_reward, exp_card_rule_ids=_exp_card_rule_ids,
                )
            await sm.save_session(state)
            await run_post_brain_processing(
                alpha, brain_result, expression, exploration_direction,
                session_id, state,
            )
            _ls._save_intelligent_search_state()

            if global_cycle % 5 == 0:
                try:
                    _ls._algo_tick("refill_eliminated_fields")
                    _eliminated = await _refill_eliminated_fields(exploration_direction)
                    if _eliminated:
                        logger.info("[%s] cycle=%d Eliminated %d low-score fields, refilled via RAG", session_id, global_cycle, len(_eliminated))
                except (OSError, ValueError, RuntimeError):
                    logger.warning("[%s] Refill eliminated fields failed", session_id, exc_info=True)

        _ls._evolution_cycle_count = await _run_logic_evolution(_ls._evolution_cycle_count, session_id, global_cycle)

        if _ls._evolution_cycle_count % 50 == 0:
            _ls._rebuild_successful_expressions()

        if _ls._evolution_cycle_count % 10 == 0:
            try:
                from openalpha_brain.cli.algo_monitor import AlgoMonitor
                _health_modules: dict[str, Any] = {}
                if _ls._signal_arbiter is not None:
                    _health_modules["signal_arbiter"] = _ls._signal_arbiter
                if _ls._scheduler is not None:
                    _health_modules["scheduler"] = _ls._scheduler
                if _ls._whitelist_mgr is not None:
                    _health_modules["whitelist"] = _ls._whitelist_mgr
                if _ls._rag_engine is not None:
                    _health_modules["rag_engine"] = _ls._rag_engine
                if _ls._market_state_inferencer is not None:
                    _health_modules["market_state"] = _ls._market_state_inferencer
                if _ls._alpha_channel is not None:
                    _health_modules["alpha_channel"] = _ls._alpha_channel
                if _ls._association is not None:
                    _health_modules["association"] = _ls._association
                _health_report = AlgoMonitor.aggregate_health_checks(_health_modules)
                _ghosts = AlgoMonitor.detect_ghost_algorithms(_health_modules)
                if _ghosts:
                    logger.warning("[%s] Ghost algorithms detected: %s", session_id, ", ".join(_ghosts))
                logger.info("[%s] Health check: %s", session_id, {k: v.get("status", "unknown") for k, v in _health_report.items()})
                if _ls._alpha_channel is not None:
                    _ch_stats = _ls._alpha_channel.get_stats()
                    logger.info("[%s] AlphaChannel stats: streamed=%d batched=%d buffer=%d avg_stream_sharpe=%.2f avg_batch_sharpe=%.2f",
                        session_id, _ch_stats.get("streamed", 0), _ch_stats.get("batched", 0),
                        _ch_stats.get("buffer_size", 0), _ch_stats.get("avg_sharpe_stream", 0), _ch_stats.get("avg_sharpe_batch", 0))
                if _ls._scheduler is not None:
                    try:
                        _dir_stats = _ls._scheduler.get_direction_stats()
                        _over_explored = [d for d, s in _dir_stats.items() if s.get("samples", 0) > 50 and s.get("expectation", 1.0) < 0.2]
                        _under_explored = [d for d, s in _dir_stats.items() if s.get("samples", 0) < 5]
                        if _over_explored or _under_explored:
                            logger.info("[%s] Scheduler direction analysis: over_explored=%s under_explored=%s", session_id, _over_explored, _under_explored)
                    except (OSError, ValueError, RuntimeError):
                        pass
                if _ls._strategy_classifier is not None:
                    try:
                        _top_profiles = _ls._strategy_classifier.get_top_profiles(n=3)
                        if _top_profiles:
                            _top_dirs = [p.direction for p in _top_profiles]
                            _top_sharpes = [f"{p.sharpe:.2f}" for p in _top_profiles]
                            logger.info("[%s] Top strategy profiles: directions=%s sharpes=%s", session_id, _top_dirs, _top_sharpes)
                        _all_directions = ["momentum", "mean_reversion", "volatility", "statistical", "volume", "interaction"]
                        for _check_dir in _all_directions:
                            _dir_profiles = _ls._strategy_classifier.get_profiles_by_direction(_check_dir)
                            if len(_dir_profiles) < 2 and _ls._scheduler is not None:
                                try:
                                    _ls._scheduler.record_direction_result(_check_dir, reward=0.10)
                                    logger.info("[%s] Direction '%s' has only %d profiles — boosting exploration weight", session_id, _check_dir, len(_dir_profiles))
                                except (OSError, ValueError, RuntimeError):
                                    pass
                    except (OSError, ValueError, RuntimeError):
                        pass
            except (OSError, ValueError, RuntimeError):
                pass

        if _ls._market_state_inferencer is not None and brain_result is not None:
            try:
                _brain_result_for_state = {
                    "direction": alpha.exploration_direction or exploration_direction,
                    "sharpe": brain_result.real_sharpe,
                    "fitness": brain_result.real_fitness,
                    "turnover": brain_result.real_turnover,
                }
                _ls._algo_tick("market_state_inference")
                _ls._market_state_inferencer.infer_from_brain_results([_brain_result_for_state])
            except (OSError, ValueError, RuntimeError):
                pass

            if brain_result.alpha_id and brain_result.status == BrainSimStatus.PASS:
                try:
                    from openalpha_brain.services.brain_data_client import get_brain_data_client
                    _bdc = get_brain_data_client()
                    if _bdc is not None:
                        _ls._algo_tick("yearly_data_fetch")
                        yearly_data = await _bdc.get_yearly_performance(brain_result.alpha_id)
                        if yearly_data:
                            _ls._market_state_inferencer.infer_from_yearly_data(
                                yearly_data,
                                alpha.exploration_direction or exploration_direction,
                            )
                            logger.info(
                                "[%s] Yearly performance data fetched for alpha %s: %d years",
                                session_id, brain_result.alpha_id, len(yearly_data),
                            )
                except (OSError, ValueError, RuntimeError):
                    logger.warning("[%s] Failed to fetch yearly data for alpha %s", session_id, brain_result.alpha_id, exc_info=True)

        if _ls._evolution_cycle_count % 5 == 0 and _ls._market_state_inferencer is not None and _ls._scheduler is not None:
            try:
                _ls._algo_tick("mab_bias_adjustment")
                _ls._scheduler.adjust_mab_bias(_ls._market_state_inferencer)
            except (OSError, ValueError, RuntimeError):
                pass

        if _ls._feature_map is not None:
            _new_gen = _ls._feature_map.advance_generation()
            logger.info("[%s] cycle=%d FeatureMap generation advanced to %d", session_id, global_cycle, _new_gen)

        _merge_session_hallucinations(session_id, state)

        await asyncio.sleep(3)

    # ── Loop completed max cycles ─────────────────────────────────────────────
    state = await sm.load_session(session_id)
    _merge_session_hallucinations(session_id, state)
    if state and not state.stop_requested:
        state.status = SessionStatus.STOPPED
        await sm.save_session(state)
    if _ls._heartbeat:
        _ls._heartbeat.remove(session_id)
    logger.info("[%s] Loop completed — %d cycles, %d alphas passed",
                session_id, settings.MAX_CYCLES, len(state.passed_alphas) if state else 0)
    logger.info("[%s] Algo call stats: %s", session_id, _ls.get_algo_call_stats())
    if _ls._alpha_channel is not None:
        try:
            _remaining = await _ls._alpha_channel.shutdown()
            if _remaining and _ls._alpha_channel_integrator is not None:
                await _ls._alpha_channel_integrator.process_batch_alphas(_remaining)
            logger.info("[%s] AlphaChannel shutdown complete: %d remaining processed", session_id, len(_remaining))
        except (OSError, ValueError, RuntimeError) as _shutdown_exc:
            logger.warning("[%s] AlphaChannel shutdown error: %s", session_id, _shutdown_exc)
    _ev.emit(EVENT_MINING_COMPLETE, {"session_id": session_id, "mode": "sequential", "cycles": settings.MAX_CYCLES, "alphas_passed": len(state.passed_alphas) if state else 0})


# ── Multi-Generator parallel helpers ────────────────────────────────────────────

_AUTO_DIRECTION_VALUES = frozenset({"auto", "", "none", "default"})


def _resolve_effective_focus(focus_area: str | None) -> str:
    if not focus_area or focus_area.lower() in _AUTO_DIRECTION_VALUES:
        return ""
    return focus_area


async def _select_diverse_directions(
    session_id: str,
    state: Any,
    num: int = 3,
) -> list[dict[str, Any]]:
    raw_focus = getattr(state, 'focus_area', None) or ""
    effective_focus = _resolve_effective_focus(raw_focus)
    default_dir: str = effective_focus or settings.DEFAULT_EXPLORATION_DIRECTION
    results: list[dict[str, Any]] = []

    if _ls._scheduler is not None and settings.MAB_ENABLED:
        try:
            results = _ls._scheduler.select_diverse_directions(num, focus_area=effective_focus)
        except (OSError, ValueError, RuntimeError):
            results = []

    if not results:
        results = [{"direction": default_dir, "template_id": "", "family_id": "", "arm_key": ""}]

    while len(results) < num:
        results.append(results[0])

    return results[:num]


# ── Pipeline mode ──────────────────────────────────────────────────────────────


@algo_log(log_args_to_skip=("self", "state"))
async def _llm_generator(session_id: str, pool: AlphaCachePool) -> None:
    logger.info("[%s] Pipeline LLM generator started", session_id)

    orchestrator = None
    _orchestrator_variants: list[str] = []
    if settings.MULTI_AGENT_ENABLED:
        if _ls._logic_library is None:
            _ls._logic_library = AlphaLogicLibrary()
        _idea_agent = IdeaAgent(llm_client.generate, _ls._rag_engine, logic_library=_ls._logic_library, classifier=_ls._strategy_classifier, mab=_ls._scheduler, field_proxy_map=getattr(_ls._scheduler, 'field_proxy_map', None) if _ls._scheduler else None)
        _factor_agent = FactorAgent(llm_client.generate, _ls._rag_engine, logic_library=_ls._logic_library, success_lib=_ls._success_lib, feature_map=_ls._feature_map, reflection_engine=_ls._reflection_engine, whitelist_mgr=_ls._whitelist_mgr, field_proxy_map=getattr(_ls._scheduler, 'field_proxy_map', None) if _ls._scheduler else None, mab=_ls._scheduler)
        _eval_agent = EvalAgent(
            None,
            mab=_ls._scheduler,
            success_lib=_ls._success_lib,
            llm_generate_fn=llm_client.generate,
            success_pool=_ls.__dict__.get("_success_pool", []),
            dedup_callback=create_dedup_mutation_callback,
        )
        orchestrator = MultiAgentOrchestrator(
            _idea_agent, _factor_agent, _eval_agent,
            originality_checker=val.get_originality_checker(),
            complexity_controller=val.get_complexity_controller(),
            rag_engine=_ls._rag_engine,
            evolution_db=_ls._evo_db,
            feature_map=_ls._feature_map,
        )
        if _ls._successful_brain_expressions:
            orchestrator.inject_successful_alphas(_ls._successful_brain_expressions)
        logger.info("[%s] Multi-agent orchestrator initialized (pipeline)", session_id)

    _consecutive_errors = 0
    _MAX_CONSECUTIVE_ERRORS = 3
    _pev = get_event_bus()

    for global_cycle in range(1, settings.MAX_CYCLES + 1):

        _pev.emit(EVENT_CYCLE_START, {"session_id": session_id, "cycle": global_cycle, "max_cycles": settings.MAX_CYCLES, "mode": "pipeline"})

        if _ls._console_stop_event and _ls._console_stop_event.is_set():
            logger.info("[%s] Console stop requested, breaking LLM generator loop", session_id)
            break
        if _ls._console_pause_event and not _ls._console_pause_event.is_set():
            logger.info("[%s] Console pause requested, waiting... (pipeline)", session_id)
            await _ls._console_pause_event.wait()
            logger.info("[%s] Console resumed (pipeline)", session_id)

        if _ls._budget_tracker:
            _ls._budget_tracker.reset()

        if _ls._heartbeat:
            _ls._heartbeat.touch(session_id)

        state = await sm.load_session(session_id)
        if state is None:
            logger.error("[%s] Session disappeared — aborting LLM generator", session_id)
            if _ls._heartbeat:
                _ls._heartbeat.remove(session_id)
            return

        if state.stop_requested:
            logger.info("[%s] Stop requested — halting LLM generator after cycle %d", session_id, global_cycle - 1)
            state.status = SessionStatus.STOPPED
            await sm.save_session(state)
            if _ls._heartbeat:
                _ls._heartbeat.remove(session_id)
            return

        state.cycle = global_cycle
        state.status = SessionStatus.GENERATING
        await sm.save_session(state)

        if global_cycle == 1:
            user_msg = build_start_trigger(global_cycle, state.focus_area)
        else:
            user_msg = _build_continuation_msg(state, str(global_cycle))

        if _family_locked(state):
            locked_family = _last_family(state)
            user_msg += build_family_switch_warning(locked_family, global_cycle)
            logger.info("[%s] cycle=%d family lock active for '%s'", session_id, global_cycle, locked_family)

        memory_str = build_memory_injection(state)
        user_msg = memory_str + "\n" + user_msg

        if _ls._strategy_classifier is not None:
            try:
                _ls._algo_tick("find_similar_by_embedding")
                similar_strategies = await _ls._strategy_classifier.find_similar_by_embedding(exploration_direction, top_k=3)
                if similar_strategies:
                    high_sim = [s for s in similar_strategies if s.get("similarity", 0) > 0.8]
                    if high_sim:
                        logger.warning("[%s] cycle=%d Potential strategy duplication (pipeline) — %d similar strategies (sim>0.8)", session_id, global_cycle, len(high_sim))
                        dup_lines = []
                        for s in high_sim[:3]:
                            p = s.get("profile")
                            if p:
                                dup_lines.append(f"  - direction={getattr(p, 'direction', '?')} sharpe={getattr(p, 'sharpe', 'N/A')} expr={getattr(p, 'best_expression', '?')[:60]}")
                        if dup_lines:
                            user_msg += "\n\nSimilar existing strategies to avoid duplicating:\n" + "\n".join(dup_lines)
            except (OSError, ValueError, RuntimeError):
                pass

        dynamic_system_prompt = get_system_prompt() + _ls._build_global_blacklist_prompt()
        if settings.RAG_TOOL_CALL_ENABLED and _ls._rag_engine and _ls._rag_engine.is_ready:
            dynamic_system_prompt += "\n\nUse search_operators, search_fields, and search_financial_logic tools to actively retrieve operators and data fields beyond the core set."
        rag_context = None
        _global_blacklist_entries = _ls._global_knowledge.get_rag_context_entries() if _ls._global_knowledge else None

        exploration_direction: str = _resolve_effective_focus(state.focus_area) or settings.DEFAULT_EXPLORATION_DIRECTION
        _pipeline_sched_template_id: str = ""
        _pipeline_sched_family_id: str = ""
        _pipeline_sched_recommended_fields: list[str] = []
        _pipeline_exp_card_rule_ids_local: list[str] = []
        _pipeline_exp_card_rule_ids2_local: list[str] = []
        if settings.RAG_ENABLED and _ls._rag_engine and _ls._rag_engine.is_ready:
            if settings.MAB_ENABLED and _ls._scheduler:
                _ls._algo_tick("mab_select")
                _sched_result = _ls._scheduler.select_exploration_arm(focus_area=_resolve_effective_focus(state.focus_area), explore_mode=False)
                if _sched_result:
                    exploration_direction = _sched_result["direction"]
                    _pipeline_sched_template_id = _sched_result.get("template_id", "")
                    _pipeline_sched_family_id = _sched_result.get("family_id", "")
                    _pipeline_sched_recommended_fields = _sched_result.get("recommended_fields", [])

            if _ls._feature_map is not None:
                _schedule = _ls._feature_map.get_explore_exploit_schedule()
                _map_strategy = _schedule["strategy"]
                _explore_weight = _schedule["explore_weight"]
                if _map_strategy == "explore":
                    _unexplored = _ls._feature_map.get_unexplored_directions()
                    if _unexplored and random.random() < _explore_weight:
                        exploration_direction = random.choice(_unexplored)
                        logger.info(
                            "[%s] MAP-Elites EXPLORE: pivoting to unexplored direction=%s (weight=%.2f coverage=%.1f%%)",
                            session_id, exploration_direction, _explore_weight, _schedule["coverage"] * 100,
                        )
                    _explore_targets = _ls._feature_map.get_explore_targets(top_k=3)
                    if _explore_targets:
                        _target = random.choice(_explore_targets)
                        _target_dir = _target.get("direction", "")
                        if _target_dir:
                            exploration_direction = _target_dir
                        logger.info(
                            "[%s] MAP-Elites EXPLORE_TARGET (pipeline): cell=%s direction=%s horizon=%s mechanism=%s",
                            session_id, _target.get("key", ""), _target_dir,
                            _target.get("time_horizon", ""), _target.get("mechanism", ""),
                        )

            try:
                _ls._algo_tick("rag_retrieve")
                retrieval = await _ls._rag_engine.retrieve(exploration_direction)
                rag_context = _ls._rag_engine.assemble_context(retrieval)
                _rag_ops = retrieval.get("operators", [])
                _rag_fields = retrieval.get("fields", [])
                logger.info("[%s] MONITOR: rag_retrieve: ops=%d fields=%d top_ops=%s top_fields=%s", session_id, len(_rag_ops), len(_rag_fields), [o.get("id", o.get("name", "")) for o in _rag_ops[:3]], [f.get("id", f.get("name", "")) for f in _rag_fields[:3]])
                rag_context = await _arbiter_rerank(retrieval, rag_context, exploration_direction)
                _ls._monitor.record("STEP", "rag", "retrieve", f"direction={exploration_direction}", session_id=session_id)
                _pipeline_exp_cards: list[dict] = []
                if _ls._experience_distiller:
                    try:
                        _ls._algo_tick("experience_card_retrieval")
                        _pec = await _ls._experience_distiller.get_applicable_cards(exploration_direction, top_k=3)
                        _pipeline_exp_cards = [{"failure_pattern": c.failure_pattern, "fix_strategy": c.fix_strategy, "applicable_conditions": c.applicable_conditions, "confidence": c.confidence} for c in _pec]
                        _pipeline_exp_card_rule_ids_local = [c.rule_id for c in _pec if c.rule_id]
                    except (OSError, ValueError, RuntimeError):
                        pass
                dynamic_system_prompt = get_system_prompt() + _ls._build_global_blacklist_prompt() + build_dynamic_context(rag_context, global_blacklist=_global_blacklist_entries, experience_cards=_pipeline_exp_cards or None)
                if _pipeline_sched_recommended_fields:
                    dynamic_system_prompt += f"\n\n▶ SCHEDULER RECOMMENDED FIELDS (prioritize these in your expression): {', '.join(_pipeline_sched_recommended_fields)}"

                if _ls._whitelist_mgr and rag_context.get("field_ids"):
                    _ls._algo_tick("whitelist_update")
                    _ls._whitelist_mgr.update_dynamic(rag_context["field_ids"])
                    for _rag_field in retrieval.get("fields", []):
                        _fid = _rag_field.get("id") or _rag_field.get("name")
                        _fds = _rag_field.get("dataset", "unknown")
                        if _fid:
                            _ls._whitelist_mgr.register_field_dataset(_fid, _fds)
            except (OSError, ValueError, RuntimeError) as exc:
                logger.warning("[%s] cycle=%d RAG retrieval failed: %s", session_id, global_cycle, exc)
        elif _global_blacklist_entries:
            _pipeline_exp_cards2: list[dict] = []
            if _ls._experience_distiller:
                try:
                    _ls._algo_tick("experience_card_retrieval")
                    _pec2 = await _ls._experience_distiller.get_applicable_cards(exploration_direction, top_k=3)
                    _pipeline_exp_cards2 = [{"failure_pattern": c.failure_pattern, "fix_strategy": c.fix_strategy, "applicable_conditions": c.applicable_conditions, "confidence": c.confidence} for c in _pec2]
                    _pipeline_exp_card_rule_ids2_local = [c.rule_id for c in _pec2 if c.rule_id]
                except (OSError, ValueError, RuntimeError):
                    pass
            dynamic_system_prompt += build_dynamic_context(global_blacklist=_global_blacklist_entries, experience_cards=_pipeline_exp_cards2 or None)

        _pipeline_success_context = ""
        if _ls._success_lib and settings.SUCCESS_CASE_LIBRARY_ENABLED:
            try:
                similar_cases = await _ls._success_lib.search_similar(exploration_direction, top_k=3)
                if similar_cases:
                    ref_lines = []
                    for sc in similar_cases[:3]:
                        ref_lines.append(f"  - (sharpe={sc.get('sharpe', 'N/A')}, fitness={sc.get('fitness', 'N/A')}): {sc.get('expr', '')}")
                    _pipeline_success_context = "\n\nPreviously successful alphas for reference:\n" + "\n".join(ref_lines)
            except (OSError, ValueError, RuntimeError):
                pass

        if _pipeline_success_context:
            user_msg += _pipeline_success_context

        if _ls._logic_library is not None and not settings.MULTI_AGENT_ENABLED:
            try:
                _ls._algo_tick("factor_template")
                _templates = _ls._logic_library.get_templates_for_direction(exploration_direction)
                if _templates:
                    _template_str = "\n\nReference expression templates (adapt, do NOT copy verbatim):\n" + "\n".join(f"  - {t}" for t in _templates)
                    user_msg += _template_str
            except (OSError, ValueError, RuntimeError):
                pass

        _fb_list_pipeline = getattr(_ls, '_brain_feedback_buffer', None)
        if _fb_list_pipeline:
            for _fb_msg in _fb_list_pipeline:
                state.conversation_history.append(_fb_msg)
            logger.info(
                "[%s] cycle=%d injected %d BRAIN result feedback(s) — pipeline",
                session_id, global_cycle, len(_fb_list_pipeline),
            )
            _ls._brain_feedback_buffer = []
            await sm.save_session(state)

        effective_history = state.conversation_history
        if _ls._summarizer:
            _ls._algo_tick("conversation_summarizer")
            effective_history, was_summarized = await _ls._summarizer.summarize_if_needed(state.conversation_history)
            if was_summarized:
                logger.info("[%s] cycle=%d — conversation history summarized before LLM call (pipeline)", session_id, global_cycle)
        # ── Launch independent streaming generation tasks (no gather barrier) ──
        logger.info("[%s] cycle=%d — launching %d streaming generation tasks",
                    session_id, global_cycle, settings.GENERATOR_PARALLEL_TASKS)
        if _ls._heartbeat:
            _ls._heartbeat.touch(session_id)

        _directions = await _select_diverse_directions(session_id, state, num=settings.GENERATOR_PARALLEL_TASKS)
        logger.info("[%s] cycle=%d diverse directions: %s", session_id, global_cycle, [d.get("direction", "") for d in _directions])

        for _gi, _dir_info in enumerate(_directions):
            _dir = _dir_info.get("direction", "") or _resolve_effective_focus(state.focus_area) or settings.DEFAULT_EXPLORATION_DIRECTION
            _dir_tid = _dir_info.get("template_id", "")
            _dir_fid = _dir_info.get("family_id", "")
            _task_state = await sm.load_session(session_id)
            if _task_state is None:
                _task_state = state
            asyncio.create_task(_generate_one_and_enqueue(
                gen_idx=_gi,
                session_id=session_id,
                global_cycle=global_cycle,
                direction=_dir,
                state=_task_state,
                orchestrator=orchestrator,
                system_prompt=dynamic_system_prompt,
                global_blacklist_entries=_global_blacklist_entries,
                conversation_summarizer=_ls._summarizer,
                pool=pool,
                pev=_pev,
                user_msg=user_msg,
                template_id=_dir_tid,
                family_id=_dir_fid,
            ))

        _consecutive_errors = 0

        # ── Event-driven scheduling: wait for generation green light ──
        got_slot = await pool.await_generation_slot(timeout=120.0)
        if got_slot:
            logger.info("[%s] cycle=%d EVENT: generation green light — pool=%d active=%d",
                        session_id, global_cycle, len(pool), 3 - pool.available_slots())
        else:
            _pool_len = len(pool)
            _active = 3 - pool.available_slots()
            logger.info("[%s] cycle=%d EVENT: green light timeout (120s) — pool=%d active=%d — continuing",
                        session_id, global_cycle, _pool_len, _active)

        # ── Auto-degrade: if BRAIN fully saturated + backlog, use LLM for improvement ──
        if pool.should_degrade_to_improvement():
            logger.info("[%s] cycle=%d DEGRADE: BRAIN saturated — switching to improvement mode (pool=%d active=%d)",
                        session_id, global_cycle, len(pool), 3 - pool.available_slots())
            try:
                state = await sm.load_session(session_id)
                if state is not None:
                    _backlogged_ids = list(pool._normal_queue)
                    if _backlogged_ids:
                        _target_id = _backlogged_ids[0]
                        _target_alpha = None
                        for a in state.passed_alphas:
                            if a.alpha_id == _target_id:
                                _target_alpha = a
                                break
                        if _target_alpha is not None and _target_alpha.expression:
                            _ls._algo_tick("degraded_improvement")
                            logger.info("[%s] cycle=%d DEGRADE_IMPROVE: running improvement on alpha %s",
                                        session_id, global_cycle, _target_id)
                            _impr_result = await _brain_improvement_loop(
                                initial_result=None,
                                session_id=session_id,
                                cycle_num=global_cycle,
                                expression=_target_alpha.expression,
                                exploration_direction=_target_alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                                alpha=_target_alpha,
                                global_cycle=global_cycle,
                            )
                            if _impr_result and getattr(_impr_result, 'status', None) == BrainSimStatus.PASS:
                                _impr_aid = f"{_target_id}_impr"
                                _impr_alpha = AlphaResult(
                                    alpha_id=_impr_aid,
                                    family=_target_alpha.family,
                                    expression=_target_alpha.expression,
                                    rationale="Degraded improvement from backlog",
                                    metrics=AlphaMetrics(),
                                    fingerprint=AlphaFingerprint(),
                                    decision="SUBMIT",
                                    simulation_payload=dict(_target_alpha.simulation_payload) if _target_alpha.simulation_payload else {"regular": _target_alpha.expression},
                                    cycle_num=global_cycle,
                                    passed=True,
                                    exploration_direction=_target_alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                                    pipeline_status=PipelineStatus.QUEUED,
                                )
                                await sm.add_alpha(session_id, _impr_alpha)
                                await pool.enqueue(_impr_aid, priority="high")
                                logger.info("[%s] DEGRADE_RESULT: improved alpha %s re-enqueued", session_id, _impr_aid)
            except (OSError, ValueError, RuntimeError) as _degrade_exc:
                logger.warning("[%s] cycle=%d DEGRADE_ERROR: %s", session_id, global_cycle, _degrade_exc)

        # ── Periodic maintenance ──
        _ls._evolution_cycle_count = await _run_logic_evolution(_ls._evolution_cycle_count, session_id, global_cycle)

        if global_cycle % 5 == 0:
            try:
                _ls._algo_tick("refill_eliminated_fields")
                _refill_dir = settings.DEFAULT_EXPLORATION_DIRECTION
                _eliminated = await _refill_eliminated_fields(_refill_dir)
                if _eliminated:
                    logger.info("[%s] cycle=%d Eliminated %d low-score fields, refilled via RAG (pipeline)", session_id, global_cycle, len(_eliminated))
            except (OSError, ValueError, RuntimeError):
                logger.warning("[%s] Refill eliminated fields failed (pipeline)", session_id, exc_info=True)

        if _ls._evolution_cycle_count % 50 == 0:
            _ls._rebuild_successful_expressions()

        if _ls._evolution_cycle_count % 10 == 0:
            try:
                from openalpha_brain.cli.algo_monitor import AlgoMonitor
                _health_modules: dict[str, Any] = {}
                if _ls._signal_arbiter is not None:
                    _health_modules["signal_arbiter"] = _ls._signal_arbiter
                if _ls._scheduler is not None:
                    _health_modules["scheduler"] = _ls._scheduler
                if _ls._whitelist_mgr is not None:
                    _health_modules["whitelist"] = _ls._whitelist_mgr
                if _ls._rag_engine is not None:
                    _health_modules["rag_engine"] = _ls._rag_engine
                if _ls._market_state_inferencer is not None:
                    _health_modules["market_state"] = _ls._market_state_inferencer
                if _ls._alpha_channel is not None:
                    _health_modules["alpha_channel"] = _ls._alpha_channel
                if _ls._association is not None:
                    _health_modules["association"] = _ls._association
                _health_report = AlgoMonitor.aggregate_health_checks(_health_modules)
                _ghosts = AlgoMonitor.detect_ghost_algorithms(_health_modules)
                if _ghosts:
                    logger.warning("[%s] Ghost algorithms detected: %s", session_id, ", ".join(_ghosts))
                logger.info("[%s] Health check: %s", session_id, {k: v.get("status", "unknown") for k, v in _health_report.items()})
            except (OSError, ValueError, RuntimeError):
                pass

        if _ls._evolution_cycle_count % 10 == 0:
            if _ls._alpha_channel is not None:
                _ch_stats = _ls._alpha_channel.get_stats()
                logger.info("[%s] AlphaChannel stats (pipeline): streamed=%d batched=%d buffer=%d avg_stream_sharpe=%.2f avg_batch_sharpe=%.2f",
                    session_id, _ch_stats.get("streamed", 0), _ch_stats.get("batched", 0),
                    _ch_stats.get("buffer_size", 0), _ch_stats.get("avg_sharpe_stream", 0), _ch_stats.get("avg_sharpe_batch", 0))
            if _ls._scheduler is not None:
                try:
                    _dir_stats = _ls._scheduler.get_direction_stats()
                    _over = [d for d, s in _dir_stats.items() if s.get("pulls", 0) > 20]
                    _under = [d for d, s in _dir_stats.items() if s.get("pulls", 0) < 3]
                    if _over or _under:
                        logger.info("[%s] Scheduler direction balance (pipeline): over_explored=%s under_explored=%s", session_id, _over[:5], _under[:5])
                except (OSError, ValueError, RuntimeError):
                    pass
            if _ls._strategy_classifier is not None:
                try:
                    _top_profiles = _ls._strategy_classifier.get_top_profiles(n=3)
                    if _top_profiles:
                        logger.info("[%s] Top strategy profiles (pipeline): %s", session_id, [(p.direction, p.sharpe) for p in _top_profiles if hasattr(p, 'direction')])
                except (OSError, ValueError, RuntimeError):
                    pass

        if _ls._feature_map is not None:
            _new_gen = _ls._feature_map.advance_generation()
            logger.info(
                "[%s] cycle=%d FeatureMap generation advanced to %d (pipeline)",
                session_id, global_cycle, _new_gen,
            )

    state = await sm.load_session(session_id)
    _merge_session_hallucinations(session_id, state)
    if state and not state.stop_requested:
        state.status = SessionStatus.STOPPED
        await sm.save_session(state)
    if _ls._heartbeat:
        _ls._heartbeat.remove(session_id)
    logger.info("[%s] LLM generator completed — %d cycles, %d alphas passed",
                session_id, settings.MAX_CYCLES, len(state.passed_alphas) if state else 0)
    logger.info("[%s] Algo call stats: %s", session_id, _ls.get_algo_call_stats())
    if _ls._alpha_channel is not None:
        try:
            _remaining = await _ls._alpha_channel.shutdown()
            if _remaining and _ls._alpha_channel_integrator is not None:
                await _ls._alpha_channel_integrator.process_batch_alphas(_remaining)
            logger.info("[%s] AlphaChannel shutdown complete (pipeline): %d remaining processed", session_id, len(_remaining))
        except (OSError, ValueError, RuntimeError) as _shutdown_exc:
            logger.warning("[%s] AlphaChannel shutdown error: %s", session_id, _shutdown_exc)
    _pev.emit(EVENT_MINING_COMPLETE, {"session_id": session_id, "mode": "pipeline", "cycles": settings.MAX_CYCLES, "alphas_passed": len(state.passed_alphas) if state else 0})



async def _brain_submitter_worker(worker_id: int, session_id: str, pool: AlphaCachePool) -> None:
    logger.info("[%s] Pipeline BRAIN submitter worker-%d started", session_id, worker_id)
    _wev = get_event_bus()

    while True:
        if _ls._heartbeat:
            _ls._heartbeat.touch(session_id)

        state = await sm.load_session(session_id)
        if state is None:
            logger.error("[%s] Session disappeared — aborting BRAIN submitter worker-%d", session_id, worker_id)
            return
        if state.stop_requested:
            logger.info("[%s] Stop requested — halting BRAIN submitter worker-%d", session_id, worker_id)
            return

        alpha_id = await pool.next_to_submit()
        if alpha_id is None:
            _ls._monitor.record("SKIP", "brain_worker", "poll", f"worker-{worker_id} no alpha (pool_size={len(pool)}, slots={pool.available_slots()})", session_id=session_id)
            await asyncio.sleep(0.5)
            continue

        state = await sm.load_session(session_id)
        if state is None:
            await pool.release_slot(alpha_id)
            return

        alpha = None
        for a in state.passed_alphas:
            if a.alpha_id == alpha_id:
                alpha = a
                break

        if alpha is None:
            logger.warning("[%s] Alpha %s not found in state — releasing slot", session_id, alpha_id)
            await pool.release_slot(alpha_id)
            continue

        alpha.pipeline_status = PipelineStatus.BRAIN_SUBMITTED
        for i, a in enumerate(state.passed_alphas):
            if a.alpha_id == alpha_id:
                state.passed_alphas[i] = alpha
                break
        await sm.save_session(state)

        if not alpha.simulation_payload:
            alpha.simulation_payload = {"settings": {}, "regular": alpha.expression}
        elif not alpha.simulation_payload.get("regular"):
            alpha.simulation_payload["regular"] = alpha.expression

        state.status = SessionStatus.SUBMITTING
        state.brain_mutation_count = 0
        state.current_brain_alpha_id = None
        _ls._log(state, "SUBMIT", f"Submitting {alpha.alpha_id} to BRAIN (pipeline)…", {"expr": alpha.expression[:80]})
        await sm.save_session(state)

        _pipeline_brain_settings = alpha.simulation_payload.get("settings", {}) if alpha.simulation_payload else {}
        logger.info("[%s] cycle=%d BRAIN_SUBMIT: expr=%s universe=%s delay=%s decay=%s", session_id, alpha.cycle_num, alpha.expression[:100], _pipeline_brain_settings.get("universe", "N/A"), _pipeline_brain_settings.get("delay", "N/A"), _pipeline_brain_settings.get("decay", "N/A"))

        if _ls._whitelist_mgr is not None:
            try:
                allowed_fields = _ls._whitelist_mgr.get_allowed_fields()
                eliminated = _ls._whitelist_mgr.eliminated_fields
                solidified = _ls._whitelist_mgr.solidified_fields
                _ls._algo_tick("whitelist_pre_submit_check")
                import re as _re_whitelist_pipe
                _expr_fields_wl = set(_re_whitelist_pipe.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b(?!\s*\()', alpha.expression))
                _expr_fields_wl -= {'and', 'or', 'not', 'if', 'else', 'true', 'false', 'nan', 'inf'}
                _expr_fields_wl -= val.PERMITTED_OPERATORS
                _invalid_fields = [f for f in _expr_fields_wl if f not in allowed_fields]
                if _invalid_fields:
                    logger.warning(
                        "[%s] cycle=%d WHITELIST_FILTER [PIPELINE]: expression contains %d non-whitelisted fields: %s (allowed=%d, solidified=%d, eliminated=%d)",
                        session_id, alpha.cycle_num, len(_invalid_fields), _invalid_fields[:5],
                        len(allowed_fields), len(solidified), len(eliminated),
                    )
                else:
                    logger.info(
                        "[%s] cycle=%d WHITELIST_PASS [PIPELINE]: all %d fields in whitelist (allowed=%d, solidified=%d, eliminated=%d)",
                        session_id, alpha.cycle_num, len(_expr_fields_wl),
                        len(allowed_fields), len(solidified), len(eliminated),
                    )
            except (OSError, ValueError, RuntimeError):
                pass

        _wev.emit(EVENT_BRAIN_SUBMIT, {"session_id": session_id, "worker_id": worker_id, "alpha_id": alpha.alpha_id, "expression": alpha.expression[:120], "cycle": alpha.cycle_num})
        brain_result = await _submit_to_brain(alpha, session_id, alpha.cycle_num)

        logger.info("[%s] MONITOR: brain_submit: status=%s sharpe=%s fitness=%s turnover=%s direction=%s alpha_id=%s", session_id, brain_result.status.value if brain_result.status else "ERROR", brain_result.real_sharpe, brain_result.real_fitness, brain_result.real_turnover, alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION, brain_result.alpha_id)
        logger.info("[%s] cycle=%d BRAIN_RESULT: status=%s sharpe=%.4f fitness=%.4f turnover=%.2f drawdown=%.2f", session_id, alpha.cycle_num, brain_result.status.value if brain_result.status else "ERROR", brain_result.real_sharpe or 0.0, brain_result.real_fitness or 0.0, brain_result.real_turnover or 0.0, brain_result.real_drawdown or 0.0)

        if brain_result is not None and brain_result.real_sharpe is not None:
            pool.record_submission_result(
                direction=alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                sharpe=brain_result.real_sharpe,
            )

        _wev.emit(EVENT_BRAIN_RESULT, {"session_id": session_id, "worker_id": worker_id, "alpha_id": alpha.alpha_id, "expression": alpha.expression[:120], "status": brain_result.status.value if brain_result.status else "ERROR", "sharpe": brain_result.real_sharpe, "fitness": brain_result.real_fitness, "turnover": brain_result.real_turnover, "returns": brain_result.real_returns, "drawdown": brain_result.real_drawdown, "direction": alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION, "mode": "pipeline"})

        await pool.release_slot(alpha_id)
        logger.info("[%s] worker-%d slot released, spawning post-processing for alpha %s", session_id, worker_id, alpha_id)

        asyncio.create_task(_post_process_brain_result(
            worker_id=worker_id,
            session_id=session_id,
            pool=pool,
            alpha_id=alpha_id,
            brain_result=brain_result,
        ))


@algo_log(level=logging.INFO)
async def run_loop_pipeline(session_id: str) -> None:
    if _ls._scheduler is None and _ls._rag_engine is None:
        _ls.init_intelligent_search()

    pool = AlphaCachePool()
    worker_count = settings.PIPELINE_MAX_SLOTS

    _decay_detector = AlphaDecayDetector()
    _ls._decay_detector = _decay_detector
    if _ls._scheduler is not None:
        _ls._scheduler.decay_detector = _decay_detector

    _engine = CrossoverMutationEngine(originality_checker=None, llm_generate_fn=llm_client.generate)
    _ls._crossover_engine = _engine
    try:
        from openalpha_brain.validation.validator import get_originality_checker
        _oc = get_originality_checker()
        if _oc is not None:
            _engine = CrossoverMutationEngine(originality_checker=_oc, llm_generate_fn=llm_client.generate)
            _ls._crossover_engine = _engine
    except (OSError, ValueError, RuntimeError):
        pass

    _hypothesis_aligner = HypothesisAligner()
    _ls._hypothesis_aligner = _hypothesis_aligner
    _gen_gates = GenerationGates(
        hypothesis_aligner=_hypothesis_aligner,
        llm_generate_fn=llm_client.generate,
    )
    _ls._generation_gates = _gen_gates

    try:
        from openalpha_brain.services.brain_data_client import get_brain_data_client
        _bdc = get_brain_data_client()
    except (OSError, ValueError, RuntimeError):
        _bdc = None
    if _bdc is not None:
        decay_handler = await create_alpha_decay_handler(
            loop_state_module=_ls,
        )
        await _decay_detector.register_instruments(
            fetch_yearly=_bdc.get_yearly_performance,
            fetch_pnl=_bdc.get_pnl_curve,
            estimate_garch=None,
            on_decay_detected=decay_handler,
        )
        logger.info("[%s] DecayDetector: instruments registered", session_id)

    workers = [
        _brain_submitter_worker(wid, session_id, pool)
        for wid in range(worker_count)
    ]
    await asyncio.gather(
        _periodic_decay_check(session_id),
        _periodic_trajectory_crossover(session_id),
        _llm_generator(session_id, pool),
        *workers,
    )
    if _ls._alpha_channel is not None:
        try:
            _remaining = await _ls._alpha_channel.shutdown()
            if _remaining and _ls._alpha_channel_integrator is not None:
                await _ls._alpha_channel_integrator.process_batch_alphas(_remaining)
            logger.info("[%s] AlphaChannel shutdown complete (pipeline_full): %d remaining processed", session_id, len(_remaining))
        except (OSError, ValueError, RuntimeError) as _shutdown_exc:
            logger.warning("[%s] AlphaChannel shutdown error: %s", session_id, _shutdown_exc)
    get_event_bus().emit(EVENT_MINING_COMPLETE, {"session_id": session_id, "mode": "pipeline_full"})
