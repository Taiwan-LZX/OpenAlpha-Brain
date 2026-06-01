from __future__ import annotations

import asyncio
import contextlib
import logging
import random

from openalpha_brain.cli import session_manager as sm
from openalpha_brain.config.config import settings
from openalpha_brain.core import loop_state as _ls
from openalpha_brain.core.models import (
    AlphaFingerprint,
    AlphaMetrics,
    AlphaResult,
    BrainSimStatus,
    BrainSubmissionResult,
    PipelineStatus,
    SessionStatus,
)
from openalpha_brain.core.pipeline import AlphaCachePool
from openalpha_brain.evolution.crossover_mutation import GradientMutation, SemanticCrossover
from openalpha_brain.generation.alpha_generator import _summarise_rejected
from openalpha_brain.generation.prompts import build_restart_trigger
from openalpha_brain.learning.mab import (
    PENALTY_BRAIN_ERROR,
    PENALTY_BRAIN_FAIL,
    REWARD_SHARPE_05,
)
from openalpha_brain.learning.param_optimizer import expression_hash
from openalpha_brain.learning.reward_updater import (
    _extract_strategy_features,
    _sync_mab_bias_from_evidence,
)
from openalpha_brain.services import brain_client, llm_client
from openalpha_brain.services.brain_result_processor import (
    check_margin_efficiency,
    compute_hierarchical_reward_with_penalties,
    fetch_correlation_analysis,
    record_evo_and_success,
    record_fail_feedback,
    record_pass_feedback,
    run_post_brain_processing,
    run_stability_analysis,
    submit_for_review,
)
from openalpha_brain.services.brain_submitter import (
    _brain_improvement_loop,
    _build_brain_result_dict,
    _log_brain_result,
    _run_param_optimization,
)
from openalpha_brain.validation import validator as val
from openalpha_brain.validation.validator import compute_hierarchical_reward, get_reward_level

logger = logging.getLogger(__name__)


def _merge_session_hallucinations(session_id: str, state) -> None:
    if not _ls._global_knowledge:
        return
    current_log = state.hallucination_log
    last_idx = _ls._last_merged_idx.get(session_id, 0)
    new_entries = current_log[last_idx:]
    content_list = [e.get("variable", "") for e in new_entries if e.get("variable")]
    if content_list:
        _ls._global_knowledge.merge_session(session_id, content_list)
        _ls._global_knowledge.save()
    _ls._last_merged_idx[session_id] = len(current_log)


async def _post_process_brain_result(
    worker_id: int,
    session_id: str,
    pool: AlphaCachePool,
    alpha_id: str,
    brain_result: BrainSubmissionResult,
) -> None:
    """Background post-processing after BRAIN submission result returns.

    Runs as fire-and-forget task so the worker can immediately pick up
    the next alpha from the pool and fill BRAIN slots concurrently.
    """
    try:
        state = await sm.load_session(session_id)
        if state is None:
            logger.warning("[%s] post_process worker-%d: session disappeared", session_id, worker_id)
            return

        alpha = None
        for a in state.passed_alphas:
            if a.alpha_id == alpha_id:
                alpha = a
                break
        if alpha is None:
            logger.warning("[%s] post_process worker-%d: alpha %s not found", session_id, worker_id, alpha_id)
            return

        _engine = getattr(_ls, "_crossover_engine", None)
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
                direction=alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                sharpe=brain_result.real_sharpe or 0.0,
                turnover=brain_result.real_turnover,
                complexity=len(alpha.expression or ""),
                accepted=_accepted,
                reject_reason=_reject_reason,
            )

        _classifier = getattr(_ls, "_strategy_classifier", None)
        if _classifier is not None and brain_result.status == BrainSimStatus.PASS:
            try:
                _brain_dict = {
                    "sharpe": brain_result.real_sharpe,
                    "fitness": brain_result.real_fitness,
                    "turnover": brain_result.real_turnover,
                }
                await _classifier.classify(alpha.expression or "", brain_result=_brain_dict)
            except (OSError, ValueError, RuntimeError):
                pass

        _aligner = getattr(_ls, "_hypothesis_aligner", None)
        if _aligner is not None:
            try:
                _alignment = _aligner.align(
                    alpha.expression or "",
                    alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                )
                if _alignment["alignment_level"] in ("contradictory", "weak"):
                    logger.warning(
                        "[%s] post_process worker-%d HypothesisAligner: %s alignment (R²=%.3f) - %s",
                        session_id,
                        worker_id,
                        _alignment["alignment_level"].upper(),
                        _alignment["r2_score"],
                        _alignment["diagnosis"],
                    )
                    _alignment_feedback = _aligner.build_alignment_feedback(_alignment)
                    _fb_buf_align = getattr(_ls, "_brain_feedback_buffer", None)
                    if _fb_buf_align is None:
                        _ls._brain_feedback_buffer = []
                        _fb_buf_align = _ls._brain_feedback_buffer
                    _fb_buf_align.append(
                        {
                            "role": "user",
                            "content": _alignment_feedback,
                        }
                    )
            except (OSError, ValueError, RuntimeError):
                pass

        if _ls._alpha_channel is not None and brain_result is not None and brain_result.real_sharpe is not None:
            try:
                _ls._algo_tick("alpha_channel_submit")
                _route = await _ls._alpha_channel.submit(
                    alpha_id=brain_result.alpha_id or "",
                    sharpe=brain_result.real_sharpe,
                    expression=alpha.expression or "",
                    direction=alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                )
                logger.info(
                    "[%s] post_process alpha_channel: route=%s sharpe=%.2f",
                    session_id,
                    _route,
                    brain_result.real_sharpe,
                )
                if _route == "stream" and _ls._alpha_channel_integrator is not None:
                    _ls._algo_tick("alpha_channel_stream_process")
                    await _ls._alpha_channel_integrator.process_stream_alpha(
                        {
                            "alpha_id": brain_result.alpha_id or "",
                            "sharpe": brain_result.real_sharpe,
                            "expression": alpha.expression or "",
                            "direction": alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                        }
                    )
                _batch = await _ls._alpha_channel.get_batch()
                if _batch and _ls._alpha_channel_integrator is not None:
                    _ls._algo_tick("alpha_channel_batch_process")
                    await _ls._alpha_channel_integrator.process_batch_alphas(_batch)
            except (OSError, ValueError, RuntimeError):
                pass

        # ── Rest of post-processing (same as before, now in background) ──

        state = await sm.load_session(session_id)
        if state is None:
            return
        for i, a in enumerate(state.passed_alphas):
            if a.alpha_id == alpha_id:
                alpha = a
                break

        alpha.pipeline_status = PipelineStatus.BRAIN_POLLING
        state.current_brain_alpha_id = brain_result.alpha_id
        _log_brain_result(state, brain_result, alpha.expression, attempt=0)
        for i, a in enumerate(state.passed_alphas):
            if a.alpha_id == alpha_id:
                state.passed_alphas[i] = alpha
                break
        await sm.save_session(state)

        alpha.pipeline_status = PipelineStatus.BRAIN_RESULT

        if (
            settings.PARAM_OPTIMIZATION_ENABLED
            and _ls._param_optimizer
            and brain_result.status == BrainSimStatus.FAIL
            and brain_result.real_sharpe is not None
            and _ls._param_optimizer.should_optimize(brain_result.real_sharpe, brain_client.GATE_SHARPE_MIN)
        ):
            _ls._log(
                state,
                "PARAM_OPT",
                f"Sharpe={brain_result.real_sharpe:.3f} near gate "
                f"{brain_client.GATE_SHARPE_MIN} — running param optimization (pipeline)",
                {"sharpe": brain_result.real_sharpe},
            )
            for i, a in enumerate(state.passed_alphas):
                if a.alpha_id == alpha_id:
                    state.passed_alphas[i] = alpha
                    break
            await sm.save_session(state)

            _ls._algo_tick("param_optimization")
            opt_result = await _run_param_optimization(
                alpha.expression,
                session_id,
                alpha.cycle_num,
            )

            state = await sm.load_session(session_id)
            if state is None:
                return
            for i, a in enumerate(state.passed_alphas):
                if a.alpha_id == alpha_id:
                    alpha = a
                    break
            if opt_result is not None and opt_result.real_sharpe is not None:
                if brain_result.real_sharpe is None or opt_result.real_sharpe > brain_result.real_sharpe:
                    logger.info(
                        "[%s] Param optimization improved Sharpe: %.3f → %.3f",
                        session_id,
                        brain_result.real_sharpe or 0,
                        opt_result.real_sharpe,
                    )
                    brain_result = opt_result
                    _log_brain_result(state, brain_result, alpha.expression, attempt=0)
                    cached_params = _ls._param_optimizer.get_cached(expression_hash(alpha.expression))
                    if cached_params and alpha.simulation_payload:
                        if "settings" not in alpha.simulation_payload:
                            alpha.simulation_payload["settings"] = {}
                        alpha.simulation_payload["settings"]["decay"] = cached_params["decay"]
                        alpha.simulation_payload["settings"]["neutralization"] = cached_params["neutralization"]

                    _pop_round = getattr(alpha, "_improvement_round", 0) + 1
                    alpha._improvement_round = _pop_round
                    _pop_aid = f"{alpha_id}_paramopt{_pop_round}"
                    _pop_alpha = AlphaResult(
                        alpha_id=_pop_aid,
                        family=alpha.family,
                        expression=alpha.expression,
                        rationale=f"Param optimized r{_pop_round}: decay={cached_params.get('decay') if cached_params else 'N/A'} neut={cached_params.get('neutralization') if cached_params else 'N/A'}",  # noqa: E501
                        metrics=AlphaMetrics(),
                        fingerprint=AlphaFingerprint(),
                        decision="SUBMIT",
                        simulation_payload={
                            "settings": dict(alpha.simulation_payload.get("settings", {}))
                            if alpha.simulation_payload
                            else {},
                            "regular": alpha.expression,
                        },
                        cycle_num=alpha.cycle_num,
                        passed=True,
                        exploration_direction=alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                        pipeline_status=PipelineStatus.QUEUED,
                    )
                    _pop_alpha._improvement_round = _pop_round
                    await sm.add_alpha(session_id, _pop_alpha)
                    await pool.enqueue(_pop_aid, priority="high")
                    logger.info(
                        "[%s] SELF_LEARN_PARAMOPT: alpha %s r=%d re-enqueued high — Sharpe %.3f→%.3f",
                        session_id,
                        _pop_aid,
                        _pop_round,
                        brain_result.real_sharpe or 0.0,
                        opt_result.real_sharpe,
                    )
                else:
                    _ls._log(state, "PARAM_OPT", "Param optimization did not improve Sharpe", {})
            else:
                _ls._log(state, "PARAM_OPT", "Param optimization found no valid results", {})
            for i, a in enumerate(state.passed_alphas):
                if a.alpha_id == alpha_id:
                    state.passed_alphas[i] = alpha
                    break
            await sm.save_session(state)

        if brain_result.status in (BrainSimStatus.FAIL, BrainSimStatus.ERROR):
            if _ls._experience_distiller and alpha.exp_card_rule_ids:
                try:
                    for _rid in alpha.exp_card_rule_ids:
                        _ls._experience_distiller.record_card_usage(_rid, success=False)
                except (OSError, ValueError, RuntimeError):
                    pass
            if _ls._experience_distiller and alpha.exp_card_rule_ids2:
                try:
                    for _rid in alpha.exp_card_rule_ids2:
                        _ls._experience_distiller.record_card_usage(_rid, success=False)
                except (OSError, ValueError, RuntimeError):
                    pass
            _pipeline_original_expr = alpha.expression
            _pipeline_original_failure_type = "BRAIN_FAIL"
            if brain_result.status == BrainSimStatus.ERROR:
                _pipeline_original_failure_type = "BRAIN_SYNTAX_ERROR"
            _pipeline_brain_checks = getattr(brain_result, "brain_checks", []) or []
            if _pipeline_brain_checks:
                _pipeline_failed_checks = [c["name"] for c in _pipeline_brain_checks if c.get("result") == "FAIL"]
                _pipeline_original_failure_type = (
                    "; ".join(_pipeline_failed_checks[:3]) if _pipeline_failed_checks else "BRAIN_FAIL"
                )
            elif brain_result.gate_failures:
                _pipeline_original_failure_type = (
                    brain_result.gate_failures[0] if brain_result.gate_failures else "BRAIN_FAIL"
                )

            improved_result = await _brain_improvement_loop(
                initial_result=brain_result,
                initial_expression=alpha.expression,
                alpha=alpha,
                session_id=session_id,
                global_cycle=alpha.cycle_num,
                grammar=_ls._fastexpr_grammar,
                resource_dispatcher=None,
                pipeline_orchestrator=None,
            )

            state = await sm.load_session(session_id)
            if state is None:
                return
            for i, a in enumerate(state.passed_alphas):
                if a.alpha_id == alpha_id:
                    alpha = a
                    break

            if improved_result.status == BrainSimStatus.PASS:
                brain_result = improved_result
                alpha.pipeline_status = PipelineStatus.BRAIN_RESULT
                if _ls._failure_lib and settings.FAILURE_FIX_LIBRARY_ENABLED:
                    try:
                        _ls._algo_tick("failure_fix_add")
                        await _ls._failure_lib.add_failure(
                            expr=_pipeline_original_expr,
                            failure_type=_pipeline_original_failure_type,
                            fix_attempt=alpha.expression,
                            fix_success=True,
                            direction=alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                            session_id=session_id,
                        )
                    except (OSError, ValueError, RuntimeError):
                        logger.warning("FailureFixLibrary add_failure (pipeline fix) failed", exc_info=True)
                if _ls._tool_factory:
                    try:
                        _ls._tool_factory.record_fix_pattern(
                            failure_type=_pipeline_original_failure_type,
                            fix_attempt=alpha.expression or "",
                            fix_success=True,
                            direction=alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                        )
                    except (OSError, ValueError, RuntimeError):
                        logger.warning("ToolFactory record_fix_pattern (pipeline fix) failed", exc_info=True)

                _impr_round = getattr(alpha, "_improvement_round", 0) + 1
                alpha._improvement_round = _impr_round
                _new_aid = f"{alpha_id}_impr{_impr_round}"
                _impr_alpha = AlphaResult(
                    alpha_id=_new_aid,
                    family=alpha.family,
                    expression=alpha.expression,
                    rationale=f"Self-learning improved r{_impr_round}: {_pipeline_original_failure_type}",
                    metrics=AlphaMetrics(),
                    fingerprint=AlphaFingerprint(),
                    decision="SUBMIT",
                    simulation_payload={
                        "settings": dict(alpha.simulation_payload.get("settings", {}))
                        if alpha.simulation_payload
                        else {},
                        "regular": alpha.expression,
                    },
                    cycle_num=alpha.cycle_num,
                    passed=True,
                    exploration_direction=alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                    pipeline_status=PipelineStatus.QUEUED,
                )
                _impr_alpha._improvement_round = _impr_round
                await sm.add_alpha(session_id, _impr_alpha)
                await pool.enqueue(_new_aid, priority="high")
                logger.info(
                    "[%s] SELF_LEARN_REINJECT: alpha %s r=%d re-enqueued high — improved from %s",
                    session_id,
                    _new_aid,
                    _impr_round,
                    _pipeline_original_failure_type,
                )
            else:
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    await record_fail_feedback(
                        brain_result,
                        alpha,
                        alpha.expression,
                        alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                        session_id,
                        alpha.fingerprint.model_dump() if alpha.fingerprint else {},
                        alpha.cycle_num,
                        state,
                        0.0,
                        exp_card_rule_ids=alpha.exp_card_rule_ids,
                    )
                _impr_fail_round = getattr(alpha, "_improvement_round", 0) + 1
                alpha._improvement_round = _impr_fail_round
                if _impr_fail_round < 3:
                    _retry_aid = f"{alpha_id}_retry{_impr_fail_round}"
                    _retry_alpha = AlphaResult(
                        alpha_id=_retry_aid,
                        family=alpha.family,
                        expression=alpha.expression,
                        rationale=f"Self-learning retry r{_impr_fail_round}: {_pipeline_original_failure_type}",
                        metrics=AlphaMetrics(),
                        fingerprint=AlphaFingerprint(),
                        decision="SUBMIT",
                        simulation_payload={
                            "settings": dict(alpha.simulation_payload.get("settings", {}))
                            if alpha.simulation_payload
                            else {},
                            "regular": alpha.expression,
                        },
                        cycle_num=alpha.cycle_num,
                        passed=True,
                        exploration_direction=alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                        pipeline_status=PipelineStatus.QUEUED,
                    )
                    _retry_alpha._improvement_round = _impr_fail_round
                    await sm.add_alpha(session_id, _retry_alpha)
                    await pool.enqueue(_retry_aid, priority="normal")
                    logger.info(
                        "[%s] SELF_LEARN_RETRY: alpha %s r=%d/%d re-enqueued — still failing (%s)",
                        session_id,
                        _retry_aid,
                        _impr_fail_round,
                        3,
                        _pipeline_original_failure_type,
                    )
                else:
                    alpha.pipeline_status = PipelineStatus.ABANDONED
                    logger.info(
                        "[%s] SELF_LEARN_EXHAUSTED: alpha %s r=%d — abandoned after max retries",
                        session_id,
                        alpha_id,
                        _impr_fail_round,
                    )
                pipeline_fail_reward = 0.0
                if settings.HIERARCHICAL_REWARD_ENABLED:
                    pipeline_fail_brain_dict = _build_brain_result_dict(improved_result)
                    pipeline_fail_reward = compute_hierarchical_reward(alpha.expression, pipeline_fail_brain_dict)
                    pipeline_fail_level, _ = get_reward_level(pipeline_fail_reward)
                    logger.info(
                        "[%s] Hierarchical reward (fail path): %.2f (%s) — decision point",
                        session_id,
                        pipeline_fail_reward,
                        pipeline_fail_level,
                    )
                    alpha.hierarchical_reward = pipeline_fail_reward
                    alpha.hierarchical_level = pipeline_fail_level
                if improved_result.status == BrainSimStatus.ERROR:
                    if settings.HIERARCHICAL_REWARD_ENABLED and pipeline_fail_reward >= 0.3:
                        logger.info(
                            "[%s] Pipeline fail path hierarchical reward %.2f >= 0.3 — BRAIN ERROR but soft penalty",
                            session_id,
                            pipeline_fail_reward,
                        )
                        if _ls._scheduler is not None:
                            with contextlib.suppress(OSError, ValueError, RuntimeError):
                                _ls._scheduler.record_direction_result(
                                    alpha.exploration_direction or "", penalty=PENALTY_BRAIN_FAIL
                                )
                    else:
                        if _ls._scheduler is not None:
                            with contextlib.suppress(OSError, ValueError, RuntimeError):
                                _ls._scheduler.record_direction_result(
                                    alpha.exploration_direction or "", penalty=PENALTY_BRAIN_ERROR
                                )
                else:
                    if settings.HIERARCHICAL_REWARD_ENABLED and pipeline_fail_reward >= 0.6:
                        logger.info(
                            "[%s] Pipeline fail path hierarchical reward %.2f >= 0.6 (Quality level) — soft pass, recording as quality_pass",  # noqa: E501
                            session_id,
                            pipeline_fail_reward,
                        )
                        if _ls._success_lib and settings.SUCCESS_CASE_LIBRARY_ENABLED:
                            try:
                                _ls._algo_tick("success_case_add")
                                await _ls._success_lib.add_case(
                                    expr=alpha.expression,
                                    hypothesis="[quality_pass] pipeline fail path",
                                    sharpe=improved_result.real_sharpe or 0.0,
                                    fitness=improved_result.real_fitness or 0.0,
                                    turnover=improved_result.real_turnover or 0.0,
                                    direction=alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                                    session_id=session_id,
                                )
                            except (OSError, ValueError, RuntimeError):
                                logger.warning(
                                    "SuccessCaseLibrary add_case (pipeline quality_pass) failed", exc_info=True
                                )
                        if _ls._scheduler is not None:
                            with contextlib.suppress(OSError, ValueError, RuntimeError):
                                _ls._scheduler.record_direction_result(
                                    alpha.exploration_direction or "", reward=REWARD_SHARPE_05
                                )
                    elif settings.HIERARCHICAL_REWARD_ENABLED and pipeline_fail_reward >= 0.3:
                        logger.info(
                            "[%s] Pipeline fail path hierarchical reward %.2f >= 0.3 (Basic level) — no penalty",
                            session_id,
                            pipeline_fail_reward,
                        )
                    else:
                        logger.info(
                            "[%s] Pipeline fail path hierarchical reward %.2f < 0.3 — penalty applied",
                            session_id,
                            pipeline_fail_reward,
                        )
                        if _ls._scheduler is not None:
                            with contextlib.suppress(OSError, ValueError, RuntimeError):
                                _ls._scheduler.record_direction_result(
                                    alpha.exploration_direction or "", penalty=PENALTY_BRAIN_FAIL
                                )
                if _ls._rag_engine and hasattr(_ls._rag_engine, "update_weights_from_feedback"):
                    try:
                        direction = alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION
                        brain_checks = getattr(improved_result, "brain_checks", []) or []
                        _ls._rag_engine.update_weights_from_feedback(direction, brain_checks)
                    except (OSError, ValueError, RuntimeError):
                        logger.warning("RAG feedback update failed", exc_info=True)
                if settings.EVIDENCE_RECORDING_ENABLED and _ls._logic_library:
                    try:
                        direction = alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION
                        logics = _ls._logic_library.get_logic_for_direction(direction)
                        _ls._algo_tick("evidence_recording")
                        for logic in logics[:1]:
                            _ls._logic_library.record_evidence(
                                logic.logic_id,
                                False,
                                expression=alpha.expression,
                                sharpe=improved_result.real_sharpe,
                                fitness=improved_result.real_fitness,
                                turnover=improved_result.real_turnover,
                                direction=alpha.exploration_direction or "",
                                failure_type=str(getattr(improved_result, "gate_failures", [])[:3]),
                                fix_success=False,
                            )
                    except (OSError, ValueError, RuntimeError):
                        pass
                    _sync_mab_bias_from_evidence()
                state.failure_catalog.append(
                    {
                        "fingerprint": alpha.fingerprint.model_dump() if alpha.fingerprint else {},
                        "failure_type": "API_ERROR"
                        if improved_result.status == BrainSimStatus.ERROR
                        else "BRAIN_EXHAUSTED",
                        "mutations_tried": state.brain_mutation_count,
                    }
                )
                if _ls._failure_lib and settings.FAILURE_FIX_LIBRARY_ENABLED:
                    try:
                        _pipeline_fail_type = (
                            "API_ERROR" if improved_result.status == BrainSimStatus.ERROR else "BRAIN_EXHAUSTED"
                        )
                        _ls._algo_tick("failure_fix_add")
                        await _ls._failure_lib.add_failure(
                            expr=alpha.expression,
                            failure_type=_pipeline_fail_type,
                            fix_attempt=None,
                            fix_success=False,
                            direction=alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                            session_id=session_id,
                        )
                    except (OSError, ValueError, RuntimeError):
                        logger.warning("FailureFixLibrary add_failure (pipeline abandon) failed", exc_info=True)
                if _ls._tool_factory:
                    try:
                        _pipeline_gate_failures = getattr(improved_result, "gate_failures", []) or []
                        _ls._tool_factory.record_fix_pattern(
                            failure_type=str(_pipeline_gate_failures[:2]),
                            fix_attempt="",
                            fix_success=False,
                            direction=alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                        )
                    except (OSError, ValueError, RuntimeError):
                        logger.warning("ToolFactory record_fix_pattern (pipeline abandon) failed", exc_info=True)
                if _ls._evo_db is not None:
                    _ls._algo_tick("add_record")
                    await _ls._evo_db.add_record(
                        alpha.expression,
                        sharpe=improved_result.real_sharpe,
                        fitness=improved_result.real_fitness,
                        turnover=improved_result.real_turnover,
                        direction=alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                        session_id=session_id,
                        status="FAIL",
                    )
                state.conversation_history.append(
                    {
                        "role": "user",
                        "content": build_restart_trigger(
                            alpha.cycle_num + 1,
                            _summarise_rejected(
                                [alpha.fingerprint.model_dump() if alpha.fingerprint else {}]
                                + state.rejected_motifs[-3:]
                            ),
                        ),
                    }
                )
                for i, a in enumerate(state.passed_alphas):
                    if a.alpha_id == alpha_id:
                        state.passed_alphas[i] = alpha
                        break
                await sm.save_session(state)
                _ls._save_intelligent_search_state()
                await pool.release_slot(alpha_id)
                if settings.CROSSOVER_ENABLED:
                    try:
                        _engine_local = getattr(_ls, "_crossover_engine", None)
                        if _engine_local is not None:
                            mutator = _engine_local._mutation
                            mutator.record_population_fitness(
                                [
                                    improved_result.real_sharpe or 0.0,
                                    improved_result.real_fitness or 0.0,
                                ]
                            )
                        else:
                            mutator = GradientMutation(originality_checker=val.get_originality_checker())
                        mut_brain_feedback = {
                            "sharpe": improved_result.real_sharpe,
                            "turnover": improved_result.real_turnover,
                        }
                        mutation_results = mutator.mutate(
                            alpha.expression,
                            brain_feedback=mut_brain_feedback,
                            original_id=alpha.alpha_id,
                        )
                        for mr in mutation_results:
                            vid = f"{alpha.alpha_id}_gm_{expression_hash(mr.mutated_expression)[:8]}"
                            variant_alpha = AlphaResult(
                                alpha_id=vid,
                                family=alpha.family,
                                expression=mr.mutated_expression,
                                rationale=f"GradientMutation: {mr.mutation_description}",
                                metrics=AlphaMetrics(),
                                fingerprint=AlphaFingerprint(),
                                decision="SUBMIT",
                                simulation_payload={
                                    "settings": dict(alpha.simulation_payload.get("settings", {}))
                                    if alpha.simulation_payload
                                    else {},
                                    "regular": mr.mutated_expression,
                                },
                                cycle_num=alpha.cycle_num,
                                passed=True,
                                exploration_direction=alpha.exploration_direction,
                                pipeline_status=PipelineStatus.QUEUED,
                            )
                            await sm.add_alpha(session_id, variant_alpha)
                            await pool.enqueue(vid, priority="low")
                            logger.info(
                                "[%s] GradientMutation variant enqueued: %s (%s)",
                                session_id,
                                vid,
                                mr.mutation_type,
                            )
                            if _engine_local is not None:
                                _engine_local._mutation.record_operator_result(
                                    mr.mutation_type,
                                    success=True,
                                )
                    except (ValueError, TypeError, OSError, RuntimeError):
                        logger.warning("[%s] GradientMutation failed", session_id, exc_info=True)
                if _ls._market_state_inferencer is not None:
                    try:
                        _brain_result_for_state = {
                            "direction": alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                            "sharpe": improved_result.real_sharpe,
                            "fitness": improved_result.real_fitness,
                            "turnover": improved_result.real_turnover,
                        }
                        _ls._algo_tick("market_state_inference")
                        _ls._market_state_inferencer.infer_from_brain_results([_brain_result_for_state])
                    except (OSError, ValueError, RuntimeError):
                        pass

                    if improved_result.alpha_id and improved_result.status == BrainSimStatus.PASS:
                        try:
                            from openalpha_brain.services.brain_data_client import get_brain_data_client

                            _bdc = get_brain_data_client()
                            if _bdc is not None:
                                _ls._algo_tick("yearly_data_fetch")
                                yearly_data = await _bdc.get_yearly_performance(improved_result.alpha_id)
                                if yearly_data:
                                    _ls._market_state_inferencer.infer_from_yearly_data(
                                        yearly_data,
                                        alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                                    )
                                    logger.info(
                                        "[%s] Yearly performance data fetched for alpha %s: %d years",
                                        session_id,
                                        improved_result.alpha_id,
                                        len(yearly_data),
                                    )
                        except (OSError, ValueError, RuntimeError):
                            logger.warning(
                                "[%s] Failed to fetch yearly data for alpha %s",
                                session_id,
                                improved_result.alpha_id,
                                exc_info=True,
                            )
                if (
                    _ls._evolution_cycle_count % 5 == 0
                    and _ls._market_state_inferencer is not None
                    and _ls._scheduler is not None
                ):
                    try:
                        _ls._algo_tick("mab_bias_adjustment")
                        _ls._scheduler.adjust_mab_bias(_ls._market_state_inferencer)
                    except (OSError, ValueError, RuntimeError):
                        pass
                try:
                    await run_post_brain_processing(
                        alpha,
                        improved_result,
                        alpha.expression,
                        alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                        session_id,
                        state,
                        log_prefix="Pipeline ",
                    )
                except (OSError, ValueError, RuntimeError):
                    logger.warning("[%s] post_brain_processing (fail path) failed", session_id, exc_info=True)
                _fb_buf = getattr(_ls, "_brain_feedback_buffer", None)
                if _fb_buf is None:
                    _ls._brain_feedback_buffer = []
                _ls._brain_feedback_buffer.append(
                    {
                        "role": "user",
                        "content": f"[BRAIN RESULT] alpha {alpha_id}: FAIL — {_pipeline_original_failure_type} dir={alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION}. Avoid this pattern; rotate direction if consecutive failures.",  # noqa: E501
                    }
                )
                _merge_session_hallucinations(session_id, state)
                return

        if brain_result.status == BrainSimStatus.PASS:
            _pipeline_all_card_ids = list(
                set(
                    (alpha.exp_card_rule_ids or []) + (alpha.exp_card_rule_ids2 or []),
                )
            )
            pipeline_hierarchical_reward, pipeline_hierarchical_level_name = compute_hierarchical_reward_with_penalties(
                brain_result,
                alpha.expression,
                session_id,
                log_prefix="Pipeline ",
            )
            if pipeline_hierarchical_level_name:
                alpha.hierarchical_reward = pipeline_hierarchical_reward
                alpha.hierarchical_level = pipeline_hierarchical_level_name

            try:
                _cc = val.get_complexity_controller()
                _cc_metrics = _cc.compute_complexity(alpha.expression or "")
                _cc.record_success(_cc_metrics)
                _cc.adapt_thresholds()
                logger.info(
                    "[%s] ComplexityController: recorded success depth=%d ops=%d nodes=%d (total=%d)",
                    session_id,
                    _cc_metrics.depth,
                    _cc_metrics.operator_count,
                    _cc_metrics.node_count,
                    _cc.success_count,
                )
            except (OSError, ValueError, RuntimeError) as _cc_exc:
                logger.warning(
                    "[%s] ComplexityController record_success/adapt_thresholds failed: %s", session_id, _cc_exc
                )

            if _ls._decay_detector is not None and brain_result.alpha_id:
                try:
                    await _ls._decay_detector.register_alpha(
                        alpha_id=brain_result.alpha_id,
                        expression=alpha.expression,
                        fingerprint=alpha.fingerprint if hasattr(alpha, "fingerprint") else None,
                        direction=alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                        initial_sharpe=brain_result.real_sharpe or 0.0,
                    )
                    logger.info(
                        "[%s] DecayDetector: registered alpha %s dir=%s sharpe=%.3f",
                        session_id,
                        brain_result.alpha_id,
                        alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                        brain_result.real_sharpe or 0.0,
                    )
                except (OSError, ValueError, RuntimeError) as exc:
                    logger.warning("[%s] DecayDetector: register_alpha failed: %s", session_id, exc)

            check_margin_efficiency(brain_result, alpha, session_id, log_prefix="Pipeline ")

            await record_pass_feedback(
                brain_result,
                alpha,
                alpha.expression,
                alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                None,
                session_id,
                pipeline_hierarchical_reward,
                exp_card_rule_ids=_pipeline_all_card_ids,
                log_prefix="Pipeline ",
            )
            alpha.brain = brain_result
            alpha.pipeline_status = PipelineStatus.COMPLETED

            if _ls._feature_map is not None:
                feat = _extract_strategy_features(
                    alpha.expression, alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION
                )
                _ls._algo_tick("feature_map_add")
                _ls._feature_map.add_candidate(
                    alpha.expression,
                    feat,
                    fitness_score=brain_result.real_fitness or 0.0,
                    sharpe=brain_result.real_sharpe or 0.0,
                    turnover=brain_result.real_turnover,
                )

            for i, a in enumerate(state.passed_alphas):
                if a.alpha_id == alpha_id:
                    state.passed_alphas[i] = alpha
                    break

            if pipeline_hierarchical_reward < 0.3:
                logger.info(
                    "[%s] Pipeline hierarchical reward %.2f < 0.3 (below basic) — skipping review submission, marking as low quality",  # noqa: E501
                    session_id,
                    pipeline_hierarchical_reward,
                )
                if _ls._scheduler is not None:
                    with contextlib.suppress(OSError, ValueError, RuntimeError):
                        _ls._scheduler.record_direction_result(
                            alpha.exploration_direction or "", penalty=PENALTY_BRAIN_FAIL
                        )
                _ls._log(
                    state,
                    "LOW_REWARD_SKIP",
                    f"Alpha {alpha.alpha_id} pipeline_hierarchical_reward={pipeline_hierarchical_reward:.2f} < 0.3 — skipping review submission",  # noqa: E501
                    {"hierarchical_reward": pipeline_hierarchical_reward},
                )
                if _ls._failure_lib and settings.FAILURE_FIX_LIBRARY_ENABLED:
                    try:
                        _ls._algo_tick("failure_fix_add")
                        await _ls._failure_lib.add_failure(
                            expr=alpha.expression,
                            failure_type="LOW_HIERARCHICAL_REWARD",
                            fix_attempt=None,
                            fix_success=False,
                            direction=alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                            session_id=session_id,
                        )
                    except (OSError, ValueError, RuntimeError):
                        pass
                state.failure_catalog.append(
                    {
                        "fingerprint": alpha.fingerprint.model_dump() if alpha.fingerprint else {},
                        "failure_type": "low_hierarchical_reward",
                        "metric_value": pipeline_hierarchical_reward,
                        "mutation_tried": "skipped_review",
                    }
                )
            else:
                await submit_for_review(brain_result, alpha, session_id, state)

            state.status = SessionStatus.PASS
            s = f"{brain_result.real_sharpe:.3f}" if brain_result.real_sharpe is not None else "N/A"
            f = f"{brain_result.real_fitness:.3f}" if brain_result.real_fitness is not None else "N/A"
            _ls._log(
                state,
                "PASS",
                f"Alpha {alpha.alpha_id} PASSED all BRAIN checks! Sharpe={s} Fitness={f}",
                {"brain_id": brain_result.alpha_id},
            )

            pipeline_hierarchical_reward, pnl_curve, yearly_data = await run_stability_analysis(
                brain_result,
                alpha,
                session_id,
                state,
                pipeline_hierarchical_reward,
                log_prefix="Pipeline ",
            )

            pipeline_hierarchical_reward = await fetch_correlation_analysis(
                brain_result,
                alpha,
                session_id,
                pipeline_hierarchical_reward,
                log_prefix="Pipeline ",
            )

            pipeline_hierarchical_reward = await record_evo_and_success(
                brain_result,
                alpha,
                alpha.expression,
                alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                session_id,
                pipeline_hierarchical_reward,
                log_prefix="Pipeline ",
            )

            if _ls._evo_db is not None and brain_result.alpha_id:
                try:
                    _ls._algo_tick("evo_lineage")
                    _lineage = _ls._evo_db.get_lineage(brain_result.alpha_id, depth=3)
                    if _lineage and len(_lineage) > 1:
                        _lineage_dirs = [r.direction for r in _lineage if hasattr(r, "direction")]
                        logger.info(
                            "[%s] Alpha %s lineage (pipeline): %d ancestors, directions=%s",
                            session_id,
                            brain_result.alpha_id,
                            len(_lineage),
                            _lineage_dirs,
                        )
                except (OSError, ValueError, RuntimeError):
                    pass

            _ls._successful_brain_expressions.append(alpha.expression)
            if len(_ls._successful_brain_expressions) >= 2 and settings.CROSSOVER_ENABLED:
                try:
                    _engine_local = getattr(_ls, "_crossover_engine", None)
                    _crossover = (
                        _engine_local._crossover
                        if _engine_local is not None
                        else SemanticCrossover(originality_checker=val.get_originality_checker())
                    )

                    _expr_a = _ls._successful_brain_expressions[-2]
                    _expr_b = _ls._successful_brain_expressions[-1]
                    _id_a = ""
                    _id_b = ""

                    if _ls._feature_map is not None:
                        _parent_a = _ls._feature_map.sample_parent()
                        _parent_b = _ls._feature_map.sample_distant_parent(
                            _ls._feature_map._cell_key(
                                alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                                "medium",
                                "signal",
                            ),
                            n=1,
                        )
                        if _parent_a is not None and _parent_a.best_expr:
                            _expr_a = _parent_a.best_expr
                            _id_a = f"cell_{_parent_a.direction}_{_parent_a.time_horizon}"
                        if _parent_b and _parent_b[0].best_expr:
                            _expr_b = _parent_b[0].best_expr
                            _id_b = f"cell_{_parent_b[0].direction}_{_parent_b[0].time_horizon}"

                    _ls._algo_tick("semantic_crossover")
                    crossover_results = await _crossover.crossover(_expr_a, _expr_b, _id_a, _id_b)
                    for cr in crossover_results:
                        vid = f"xo_{expression_hash(cr.child_expression)[:8]}"
                        variant_alpha = AlphaResult(
                            alpha_id=vid,
                            family=alpha.family,
                            expression=cr.child_expression,
                            rationale=f"SemanticCrossover: {cr.crossover_point}",
                            metrics=AlphaMetrics(),
                            fingerprint=AlphaFingerprint(),
                            decision="SUBMIT",
                            simulation_payload={
                                "settings": dict(alpha.simulation_payload.get("settings", {}))
                                if alpha.simulation_payload
                                else {},
                                "regular": cr.child_expression,
                            },
                            cycle_num=alpha.cycle_num,
                            passed=True,
                            exploration_direction=alpha.exploration_direction,
                            pipeline_status=PipelineStatus.QUEUED,
                        )
                        await sm.add_alpha(session_id, variant_alpha)
                        await pool.enqueue(vid, priority="normal")
                        logger.info(
                            "[%s] SemanticCrossover child enqueued: %s (orig=%.2f parents=%s,%s)",
                            session_id,
                            vid,
                            cr.originality_score,
                            _id_a or "recent",
                            _id_b or "recent",
                        )

                    if _ls._feature_map is not None and random.random() < 0.3:
                        _schedule = _ls._feature_map.get_explore_exploit_schedule()
                        if _schedule["coverage"] > 0.3 and _schedule["strategy"] == "exploit":
                            _ctx_a = {
                                "direction": alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                                "sharpe": brain_result.real_sharpe,
                                "turnover": brain_result.real_turnover,
                                "id": _id_a,
                            }
                            _ctx_b = {
                                "direction": alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                                "id": _id_b,
                            }
                            _ls._algo_tick("llm_semantic_crossover")
                            _llm_results = await _crossover.semantic_crossover_via_llm(
                                llm_client.generate,
                                _expr_a,
                                _expr_b,
                                context1=_ctx_a,
                                context2=_ctx_b,
                            )
                            for cr in _llm_results:
                                vid = f"llm_xo_{expression_hash(cr.child_expression)[:8]}"
                                variant_alpha = AlphaResult(
                                    alpha_id=vid,
                                    family=alpha.family,
                                    expression=cr.child_expression,
                                    rationale=f"LLM-SemanticCrossover: {cr.crossover_point}",
                                    metrics=AlphaMetrics(),
                                    fingerprint=AlphaFingerprint(),
                                    decision="SUBMIT",
                                    simulation_payload={
                                        "settings": dict(alpha.simulation_payload.get("settings", {}))
                                        if alpha.simulation_payload
                                        else {},
                                        "regular": cr.child_expression,
                                    },
                                    cycle_num=alpha.cycle_num,
                                    passed=True,
                                    exploration_direction=alpha.exploration_direction,
                                    pipeline_status=PipelineStatus.QUEUED,
                                )
                                await sm.add_alpha(session_id, variant_alpha)
                                await pool.enqueue(vid, priority="high")
                                logger.info(
                                    "[%s] LLM-SemanticCrossover child enqueued: %s (orig=%.2f)",
                                    session_id,
                                    vid,
                                    cr.originality_score,
                                )
                except (ValueError, TypeError, OSError, RuntimeError):
                    logger.warning("[%s] SemanticCrossover failed", session_id, exc_info=True)

        for i, a in enumerate(state.passed_alphas):
            if a.alpha_id == alpha_id:
                state.passed_alphas[i] = alpha
                break
        await sm.save_session(state)

        async def _safe_post_brain():
            try:
                await run_post_brain_processing(
                    alpha,
                    brain_result,
                    alpha.expression,
                    alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                    session_id,
                    state,
                    log_prefix="Pipeline ",
                )
            except (OSError, ValueError, RuntimeError):
                logger.warning("[%s] post_brain_processing failed", session_id, exc_info=True)

        async def _safe_market_inference():
            if _ls._market_state_inferencer is not None:
                _brain_result_for_state = {
                    "direction": alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                    "sharpe": brain_result.real_sharpe,
                    "fitness": brain_result.real_fitness,
                    "turnover": brain_result.real_turnover,
                }
                _ls._algo_tick("market_state_inference")
                _ls._market_state_inferencer.infer_from_brain_results([_brain_result_for_state])

        async def _safe_yearly_data():
            if brain_result.alpha_id and brain_result.status == BrainSimStatus.PASS:
                from openalpha_brain.services.brain_data_client import get_brain_data_client

                _bdc = get_brain_data_client()
                if _bdc is not None:
                    _ls._algo_tick("yearly_data_fetch")
                    yearly_data = await _bdc.get_yearly_performance(brain_result.alpha_id)
                    if yearly_data and _ls._market_state_inferencer is not None:
                        _ls._market_state_inferencer.infer_from_yearly_data(
                            yearly_data,
                            alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                        )
                        logger.info(
                            "[%s] Yearly performance data fetched for alpha %s: %d years",
                            session_id,
                            brain_result.alpha_id,
                            len(yearly_data),
                        )

        async def _safe_mab_bias():
            if (
                _ls._evolution_cycle_count % 5 == 0
                and _ls._market_state_inferencer is not None
                and _ls._scheduler is not None
            ):
                _ls._algo_tick("mab_bias_adjustment")
                _ls._scheduler.adjust_mab_bias(_ls._market_state_inferencer)

        _ls._save_intelligent_search_state()

        await asyncio.gather(
            _safe_post_brain(),
            _safe_market_inference(),
            _safe_yearly_data(),
            _safe_mab_bias(),
            return_exceptions=True,
        )
        logger.info("[%s] post_process worker-%d parallel post-PASS ops completed", session_id, worker_id)
        _fb_buf2 = getattr(_ls, "_brain_feedback_buffer", None)
        if _fb_buf2 is None:
            _ls._brain_feedback_buffer = []
        _s_val = f"{brain_result.real_sharpe:.3f}" if brain_result.real_sharpe is not None else "N/A"
        _f_val = f"{brain_result.real_fitness:.3f}" if brain_result.real_fitness is not None else "N/A"
        _ls._brain_feedback_buffer.append(
            {
                "role": "user",
                "content": f"[BRAIN RESULT] alpha {brain_result.alpha_id or alpha_id}: PASS — Sharpe={_s_val} Fitness={_f_val} dir={alpha.exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION}. This pattern works — exploit similar directions next cycle.",  # noqa: E501
            }
        )
        _merge_session_hallucinations(session_id, state)

    except (OSError, ValueError, RuntimeError):
        logger.exception("[%s] post_process worker-%d failed", session_id, worker_id)
