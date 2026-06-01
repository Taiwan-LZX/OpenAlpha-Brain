from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, cast

from openalpha_brain.agents.multi_agent import Hypothesis, _check_semantic_alignment, _load_brain_submit_params
from openalpha_brain.cli import session_manager as sm
from openalpha_brain.config.config import settings
from openalpha_brain.core import loop_state as _ls
from openalpha_brain.core.events import EVENT_ALPHA_VALIDATED
from openalpha_brain.core.models import (
    AlphaFingerprint,
    AlphaMetrics,
    AlphaResult,
    PipelineStatus,
)
from openalpha_brain.core.pipeline import AlphaCachePool
from openalpha_brain.evolution.generation_gates import GenerationGateReport
from openalpha_brain.generation.alpha_generator import _build_alpha
from openalpha_brain.generation.prompts import build_dynamic_context, build_success_feedback
from openalpha_brain.learning.param_optimizer import expression_hash
from openalpha_brain.learning.reward_updater import (
    _arbiter_rerank,
    _get_fields_from_context,
    _get_operators_from_context,
)
from openalpha_brain.services import llm_client
from openalpha_brain.validation import validator as val
from openalpha_brain.validation.ast_repair import repair_expression

logger = logging.getLogger(__name__)


def _build_continuation_msg(state, cycle: str) -> str:
    return f"Continue. Generate Alpha {cycle} now."


async def _apply_generation_gates(
    expression: str,
    hypothesis_direction: str,
    hypothesis_nl: str = "",
    hypothesis_mechanism: str = "",
    operators: list[str] | None = None,
    fields: list[str] | None = None,
) -> GenerationGateReport:
    gates = getattr(_ls, '_generation_gates', None)
    if gates is None:
        return GenerationGateReport(
            passed=True, results=[], overall_score=1.0,
            correction_prompt="",
        )
    try:
        report = await gates.check(
            hypothesis_direction=hypothesis_direction,
            hypothesis_mechanism=hypothesis_mechanism,
            hypothesis_nl=hypothesis_nl,
            expression=expression,
            operators=operators,
            fields=fields,
        )
        return report
    except (ValueError, TypeError, KeyError, RuntimeError) as exc:
        logger.warning("_apply_generation_gates failed: %s", exc)
        return GenerationGateReport(
            passed=True, results=[], overall_score=1.0,
            correction_prompt="",
        )


def _extract_fields(expression: str) -> set[str]:
    fields = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b(?!\s*\()', expression))
    fields -= {'and', 'or', 'not', 'if', 'else', 'true', 'false', 'nan', 'inf'}
    fields -= val.PERMITTED_OPERATORS
    return fields


def _compute_field_overlap(expr1: str, expr2: str) -> float:
    fields1 = _extract_fields(expr1)
    fields2 = _extract_fields(expr2)
    if not fields1 or not fields2:
        return 0.0
    intersection = fields1 & fields2
    union = fields1 | fields2
    return len(intersection) / len(union) if union else 0.0


async def _generate_llm_variants(
    main_expression: str,
    direction: str,
    num_variants: int = 3,
) -> list[str]:
    prompt = f"""You are an alpha expression variant generator.

Given the following alpha expression:
```
{main_expression}
```

Exploration direction: {direction}

Generate {num_variants} semantically DISTINCT variant expressions that:
1. Preserve the same signal direction and economic hypothesis
2. Use DIFFERENT operators or DIFFERENT window/lookback parameters
3. Use DIFFERENT data fields to keep field overlap low
4. Each variant must be a complete, valid WorldQuant alpha expression
5. Maintain the same asset class (equity) and medium-term horizon

Output ONLY valid JSON:
```json
{{"variants": ["variant_expr_1", "variant_expr_2", "variant_expr_3"]}}
```"""

    try:
        result = await asyncio.wait_for(
            llm_client.generate(
                system_prompt="You are a quantitative alpha expression engineer. Output valid JSON only.",
                history=[],
                user_msg=prompt,
                session_id="variant_gen",
                cycle=0,
            ),
            timeout=30.0,
        )
        result = result.strip()
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        elif "```" in result:
            result = result.split("```")[1].split("```")[0].strip()
        data = json.loads(result)
        variants = data.get("variants", [])
        return [v for v in variants if isinstance(v, str) and len(v) > 5]
    except TimeoutError:
        logger.warning("LLM variant generation timed out")
        return []
    except (aiohttp.ClientError, ValueError, json.JSONDecodeError) as exc:
        return []


def _filter_variants_by_field_overlap(
    main_expression: str,
    variants: list[str],
    max_overlap: float = 0.7,
) -> list[str]:
    accepted: list[str] = []
    for v in variants:
        overlap = _compute_field_overlap(main_expression, v)
        if overlap >= max_overlap:
            logger.info(
                "LLM variant rejected: field_overlap=%.2f >= %.2f expr=%s",
                overlap, max_overlap, v[:60],
            )
            continue
        pair_rejected = False
        for a in accepted:
            pair_overlap = _compute_field_overlap(a, v)
            if pair_overlap >= max_overlap:
                pair_rejected = True
                logger.info(
                    "LLM variant rejected: pair_overlap=%.2f >= %.2f",
                    pair_overlap, max_overlap,
                )
                break
        if not pair_rejected:
            accepted.append(v)
    return accepted


async def _generate_single_alpha(
    gen_idx: int,
    session_id: str,
    global_cycle: int,
    direction: str,
    shared_history: list[dict],
    orchestrator: Any,
    system_prompt: str,
    global_blacklist_entries: Any,
    conversation_summarizer: Any,
) -> dict:
    result: dict = {
        "success": False,
        "alpha_id": "",
        "expression": "",
        "direction": direction,
        "family": "Unknown",
        "parsed": None,
        "variants": [],
        "economic_rationale": "",
        "simulation_payload": None,
        "semantic_alignment_score": None,
        "hallucination_entries": [],
        "failure_entry": None,
        "trajectory_entry": None,
        "fingerprint_dict": {},
        "ast_topology": None,
        "error_msg": "",
        "decision": "REJECT",
        "raw_response": "",
        "conversation_additions": [],
    }

    try:
        user_msg = _build_continuation_msg(None, str(global_cycle))
        user_msg = f"[Gen-{gen_idx}] Direction: {direction}\n" + user_msg

        if conversation_summarizer:
            effective_history, was_summarized = await conversation_summarizer.summarize_if_needed(list(shared_history))
            if was_summarized:
                logger.info("[%s] gen-%d cycle=%d history summarized", session_id, gen_idx, global_cycle)
        else:
            effective_history = list(shared_history)

        rag_context = None
        rag_ops: list = []
        rag_fields: list = []
        if settings.RAG_ENABLED and _ls._rag_engine and _ls._rag_engine.is_ready:
            try:
                _ls._algo_tick("rag_retrieve")
                retrieval = await _ls._rag_engine.retrieve(direction)
                rag_context = _ls._rag_engine.assemble_context(retrieval)
                rag_ops = retrieval.get("operators", [])
                rag_fields = retrieval.get("fields", [])
                rag_context = await _arbiter_rerank(retrieval, rag_context, direction)
                _ls._monitor.record("STEP", "rag", "retrieve", f"gen-{gen_idx} dir={direction}", session_id=session_id)
            except (ValueError, KeyError, AttributeError, OSError) as exc:
                logger.warning("[%s] gen-%d cycle=%d RAG failed: %s", session_id, gen_idx, global_cycle, exc)

        dyn_prompt = system_prompt
        if rag_context:
            dyn_prompt += build_dynamic_context(rag_context, global_blacklist=global_blacklist_entries)
        elif global_blacklist_entries:
            dyn_prompt += build_dynamic_context(global_blacklist=global_blacklist_entries)

        if settings.RAG_TOOL_CALL_ENABLED and _ls._rag_engine and _ls._rag_engine.is_ready:
            dyn_prompt += "\n\nUse search_operators, search_fields, and search_financial_logic tools to actively retrieve operators and data fields beyond the core set."

        if orchestrator is not None:
            try:
                operators_list = _get_operators_from_context(rag_context)
                fields_list = _get_fields_from_context(rag_context)

                orch_result = await orchestrator.run_iteration(
                    direction=direction,
                    history=effective_history,
                    brain_feedback=None,
                    operators=operators_list,
                    fields=fields_list,
                )

                logger.info(
                    "[%s] gen-%d cycle=%d orchestrator: iterations=%d converged=%s originality=%.2f",
                    session_id, gen_idx, global_cycle,
                    orch_result.iterations, orch_result.converged, orch_result.originality_score,
                )

                variants = list(getattr(orch_result, 'variants', []))
                result["variants"] = variants

                try:
                    _ls._algo_tick("llm_multi_candidate")
                    llm_variants = await _generate_llm_variants(
                        orch_result.expression, direction, num_variants=3,
                    )
                    if llm_variants:
                        valid_variants = _filter_variants_by_field_overlap(
                            orch_result.expression, llm_variants, max_overlap=0.7,
                        )
                        result["variants"] = list(result["variants"]) + valid_variants
                except (OSError, ValueError, RuntimeError):
                    pass

                if hasattr(orch_result, 'trajectory') and orch_result.trajectory:
                    result["trajectory_entry"] = (
                        orch_result.trajectory.model_dump()
                        if hasattr(orch_result.trajectory, 'model_dump')
                        else vars(orch_result.trajectory)
                    )

                expression = orch_result.expression
                result["expression"] = expression
                result["family"] = orch_result.hypothesis.direction
                result["economic_rationale"] = orch_result.hypothesis.natural_language

                gate_ops = operators_list if 'operators_list' in dir() else None
                gate_fields = fields_list if 'fields_list' in dir() else None
                gate_report = await _apply_generation_gates(
                    expression=expression,
                    hypothesis_direction=orch_result.hypothesis.direction,
                    hypothesis_nl=orch_result.hypothesis.natural_language,
                    hypothesis_mechanism=orch_result.hypothesis.mechanism,
                    operators=gate_ops,
                    fields=gate_fields,
                )
                if not gate_report.passed:
                    logger.warning(
                        "[%s] gen-%d cycle=%d GATES FAILED: score=%.3f failed=%s",
                        session_id, gen_idx, global_cycle,
                        gate_report.overall_score, gate_report.failed_gates,
                    )
                    result["error_msg"] = f"Generation gates failed: {gate_report.failed_gates}"
                    return result
                logger.info(
                    "[%s] gen-%d cycle=%d GATES PASSED: score=%.3f",
                    session_id, gen_idx, global_cycle, gate_report.overall_score,
                )

                syntax_result = val.validate_syntax(expression)
                if not syntax_result.passed:
                    _ls._algo_tick("ast_repair")
                    repaired, repair_entries = repair_expression(expression)
                    if repaired and repaired != expression:
                        expression = repaired
                        result["expression"] = repaired
                        result["hallucination_entries"] = repair_entries
                        logger.info("[%s] gen-%d AST repair success", session_id, gen_idx)
                    else:
                        debug_recovered = False
                        if _ls._failure_lib or _ls._success_lib:
                            try:
                                from openalpha_brain.knowledge.rag_engine import auto_debug_loop as _auto_debug
                                _ls._algo_tick("auto_debug_loop")
                                debugged, debug_ok = await _auto_debug(
                                    generate_fn=llm_client.generate,
                                    validate_fn=val.validate_syntax,
                                    initial_expr=expression,
                                    max_rounds=2,
                                )
                                if debug_ok and debugged:
                                    expression = debugged
                                    result["expression"] = debugged
                                    debug_recovered = True
                            except (OSError, ValueError, RuntimeError):
                                pass
                        if not debug_recovered:
                            result["error_msg"] = "syntax_unrepairable"
                            return result

                result["parsed"] = {
                    "decision": "SUBMIT" if orch_result.converged else "ITERATE",
                    "expression": expression,
                    "simulation_payload": orch_result.simulation_payload,
                    "fingerprint": {},
                    "family": orch_result.hypothesis.direction,
                    "ast_topology": None,
                    "rationale": orch_result.hypothesis.natural_language,
                    "metrics": {},
                    "mutation_paths": [],
                    "refinement_log": None,
                }
                result["simulation_payload"] = orch_result.simulation_payload
                result["decision"] = "SUBMIT" if orch_result.converged else "ITERATE"
                result["success"] = True

            except llm_client.LLMError as exc:
                result["error_msg"] = f"LLMError: {exc}"
                logger.error("[%s] gen-%d cycle=%d LLMError: %s", session_id, gen_idx, global_cycle, exc)
                return result
            except (ValueError, TypeError, RuntimeError, OSError) as exc:
                result["error_msg"] = f"Orchestrator error: {exc}"
                logger.error("[%s] gen-%d cycle=%d error: %s", session_id, gen_idx, global_cycle, exc, exc_info=True)
                return result

            raw_response = str(result.get("parsed", ""))
            result["raw_response"] = raw_response
        else:
            try:
                raw_response = await llm_client.generate(
                    system_prompt=dyn_prompt,
                    history=effective_history,
                    user_msg=user_msg,
                    session_id=session_id,
                    cycle=global_cycle,
                    grammar=_ls._fastexpr_grammar,
                )
                result["raw_response"] = raw_response
            except llm_client.LLMError as exc:
                result["error_msg"] = f"LLMError: {exc}"
                return result

        semantic_score: float | None = None
        expression = result["expression"]
        if expression:
            try:
                _ls._algo_tick("semantic_alignment_check")
                sa_hypothesis = Hypothesis(
                    direction=direction,
                    asset_class="equity",
                    time_horizon="medium-term",
                    mechanism="",
                    natural_language="",
                )
                semantic_score = _check_semantic_alignment(sa_hypothesis, expression)
                result["semantic_alignment_score"] = semantic_score
                if semantic_score < 0.3:
                    logger.warning(
                        "[%s] gen-%d cycle=%d low semantic alignment=%.3f",
                        session_id, gen_idx, global_cycle, semantic_score,
                    )
            except (OSError, ValueError, RuntimeError):
                pass

        result["success"] = True
        return result

    except (Exception,) as exc:
        result["error_msg"] = str(exc)
        logger.error("[%s] gen-%d cycle=%d unexpected error: %s", session_id, gen_idx, global_cycle, exc, exc_info=True)
        return result


async def _generate_one_and_enqueue(
    gen_idx: int,
    session_id: str,
    global_cycle: int,
    direction: str,
    state,
    orchestrator,
    system_prompt: str,
    global_blacklist_entries,
    conversation_summarizer,
    pool: AlphaCachePool,
    pev,
    user_msg: str,
    template_id: str = "",
    family_id: str = "",
) -> None:
    try:
        gen_result = await _generate_single_alpha(
            gen_idx=gen_idx,
            session_id=session_id,
            global_cycle=global_cycle,
            direction=direction,
            shared_history=list(state.conversation_history),
            orchestrator=orchestrator,
            system_prompt=system_prompt,
            global_blacklist_entries=global_blacklist_entries,
            conversation_summarizer=conversation_summarizer,
        )
    except (ValueError, TypeError, RuntimeError, OSError) as exc:
        logger.error("[%s] gen-%d cycle=%d fatal generation error: %s", session_id, gen_idx, global_cycle, exc)
        return

    if isinstance(gen_result, Exception):
        logger.error("[%s] gen-%d cycle=%d generation exception: %s", session_id, gen_idx, global_cycle, gen_result)
        return

    gen_result = cast(dict, gen_result)
    if not gen_result.get("success"):
        _err = gen_result.get("error_msg", "unknown")
        _diag_expr = gen_result.get("expression", "")
        _diag_decision = gen_result.get("decision", "N/A")
        _diag_sa = gen_result.get("semantic_alignment_score")
        logger.warning(
            "[%s] gen-%d cycle=%d generation failed: %s | expr=%s | decision=%s | semantic_alignment=%s",
            session_id, gen_idx, global_cycle, _err,
            _diag_expr[:120] if _diag_expr else "<empty>",
            _diag_decision,
            _diag_sa if _diag_sa is not None else "N/A",
        )
        return

    expression = gen_result.get("expression", "")
    if not expression:
        return

    gen_direction = gen_result.get("direction", direction)
    parsed = gen_result.get("parsed")
    decision = gen_result.get("decision", "ITERATE")
    family = gen_result.get("family", "Unknown")
    raw_response = gen_result.get("raw_response", "")
    variants = gen_result.get("variants", [])
    fingerprint_dict = gen_result.get("fingerprint_dict", {})
    ast_topology = gen_result.get("ast_topology")
    simulation_payload = gen_result.get("simulation_payload")

    if gen_result.get("trajectory_entry"):
        if not hasattr(state, 'trajectories'):
            state.trajectories = []
        state.trajectories.append(gen_result["trajectory_entry"])
    if gen_result.get("hallucination_entries"):
        for he in gen_result["hallucination_entries"]:
            if hasattr(state, 'hallucination_log'):
                state.hallucination_log.append(he)

    state.conversation_history.append({"role": "user", "content": user_msg})

    if decision == "ITERATE" and expression:
        if not simulation_payload or not simulation_payload.get("regular"):
            _brain_params = _load_brain_submit_params()
            simulation_payload = {
                "settings": {
                    "instrumentType": _brain_params.get("instrumentType", "EQUITY"),
                    "region": "USA",
                    "universe": _brain_params.get("universe", "TOP3000"),
                    "delay": _brain_params.get("delay", 1),
                    "decay": _brain_params.get("decay", 5),
                    "neutralization": _brain_params.get("neutralization", "INDUSTRY"),
                    "truncation": _brain_params.get("truncation", 0.05),
                    "pasteurization": _brain_params.get("pasteurization", "ON"),
                    "unitHandling": "VERIFY",
                    "nanHandling": _brain_params.get("nanHandling", "ON"),
                    "language": "FASTEXPR",
                    "visualization": False,
                },
                "regular": expression,
            }
        logger.info("[%s] gen-%d cycle=%d Overriding ITERATE->SUBMIT", session_id, gen_idx, global_cycle)
        decision = "SUBMIT"

    if decision != "SUBMIT":
        state.conversation_history.append({"role": "assistant", "content": raw_response})
        return

    alpha = _build_alpha(
        state, parsed, expression, fingerprint_dict,
        str(family), ast_topology, [],
        cast(dict, simulation_payload),
        val.validate_metrics(parsed or {}),
        global_cycle, decision,
        economic_rationale=gen_result.get("economic_rationale", ""),
    )
    alpha.exploration_direction = gen_direction
    alpha.template_id = template_id
    alpha.family_id = family_id
    alpha.semantic_alignment_score = gen_result.get("semantic_alignment_score")
    alpha.pipeline_status = PipelineStatus.VALIDATED

    try:
        _ls._previous_expressions.append(expression)
    except (OSError, ValueError, RuntimeError):
        pass

    logger.info(
        "[%s] gen-%d cycle=%d STREAMING_ENQUEUE — alpha_id=%s family=%s direction=%s",
        session_id, gen_idx, global_cycle, alpha.alpha_id, family, gen_direction,
    )
    pev.emit(EVENT_ALPHA_VALIDATED, {
        "session_id": session_id, "cycle": global_cycle,
        "alpha_id": alpha.alpha_id, "expression": expression[:120],
        "direction": gen_direction, "family": str(family),
        "decision": decision, "mode": "pipeline_streaming",
    })

    state.conversation_history.append({"role": "assistant", "content": raw_response})

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

    _pool_dir = gen_direction or settings.DEFAULT_EXPLORATION_DIRECTION
    await pool.enqueue(alpha.alpha_id, priority="normal", direction=_pool_dir)
    alpha.pipeline_status = PipelineStatus.QUEUED

    for variant_expr in variants:
        try:
            vid = f"{alpha.alpha_id}_v_{expression_hash(variant_expr)[:8]}"
            variant_alpha = AlphaResult(
                alpha_id=vid,
                family=family,
                expression=variant_expr,
                rationale="MultiAgent variant",
                metrics=AlphaMetrics(),
                fingerprint=AlphaFingerprint(),
                decision="SUBMIT",
                simulation_payload={
                    "settings": dict(alpha.simulation_payload.get("settings", {})) if alpha.simulation_payload else {},
                    "regular": variant_expr,
                },
                cycle_num=global_cycle,
                passed=True,
                exploration_direction=gen_direction,
                pipeline_status=PipelineStatus.QUEUED,
            )
            await sm.add_alpha(session_id, variant_alpha)
            await pool.enqueue(vid, priority="low")
        except (OSError, ValueError, RuntimeError):
            logger.warning("[%s] gen-%d Failed to enqueue variant", session_id, gen_idx, exc_info=True)

    for i, a in enumerate(state.passed_alphas):
        if a.alpha_id == alpha.alpha_id:
            state.passed_alphas[i] = alpha
            break
    await sm.save_session(state)

    logger.info("[%s] gen-%d cycle=%d STREAMING_DONE — alpha %s enqueued, pool_size=%d active=%d",
                session_id, gen_idx, global_cycle, alpha.alpha_id, len(pool), 3 - pool.available_slots())
