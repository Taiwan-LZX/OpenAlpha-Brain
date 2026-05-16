"""
OpenAlpha - Quant — Autonomous Generation Loop Engine
Runs as an asyncio background task. Drives the full LLM → Parse → Validate cycle.
Persists state after every step. Checks stop_requested every cycle.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import httpx

import alpha_parser as parser
import brain_client
import llm_client
import session_manager as sm
import validator as val
from config import settings
from models import (
    AlphaFingerprint, AlphaMetrics, AlphaResult,
    BrainSimStatus, BrainSubmissionResult,
    SessionStatus,
)
from prompts import (
    SYSTEM_PROMPT,
    build_brain_failure_feedback,
    build_failure_feedback,
    build_family_switch_warning,
    build_memory_injection,
    build_restart_trigger,
    build_start_trigger,
    build_success_feedback,
)

MAX_BRAIN_MUTATIONS = 20   # mutations before abandoning an alpha

# Cached BRAIN auth cookies (shared across cycles in a process lifetime)
_brain_cookies: httpx.Cookies | None = None
_brain_cookies_lock = asyncio.Lock()


logger = logging.getLogger(__name__)


async def run_loop(session_id: str) -> None:
    """
    Main async generation loop. Entry point called as asyncio.create_task().
    Reads and writes session state exclusively via session_manager (no shared memory).
    """
    logger.info("[%s] Loop engine started", session_id)

    for global_cycle in range(1, settings.MAX_CYCLES + 1):

        # ── 0. Load fresh state and check stop ───────────────────────────────
        state = await sm.load_session(session_id)
        if state is None:
            logger.error("[%s] Session disappeared — aborting loop", session_id)
            return

        if state.stop_requested:
            logger.info("[%s] Stop requested — halting after cycle %d", session_id, global_cycle - 1)
            await sm.update_status(session_id, SessionStatus.STOPPED)
            return

        state.cycle = global_cycle
        state.status = SessionStatus.GENERATING
        await sm.save_session(state)

        # ── 1. Build user message for this cycle ─────────────────────────────
        if global_cycle == 1:
            user_msg = build_start_trigger(global_cycle, state.focus_area)
        else:
            user_msg = _build_continuation_msg(state, global_cycle)

        # Append family-switch warning if needed
        if _family_locked(state):
            locked_family = _last_family(state)
            user_msg += build_family_switch_warning(locked_family, global_cycle)
            logger.info("[%s] cycle=%d family lock active for '%s'", session_id, global_cycle, locked_family)

        # v2: Prepend POMDP memory injection to every user message
        memory_str = build_memory_injection(state)
        user_msg = memory_str + "\n" + user_msg

        # ── 2. LLM call ───────────────────────────────────────────────────────
        logger.info("[%s] cycle=%d — calling LLM", session_id, global_cycle)
        try:
            raw_response = await llm_client.generate(
                system_prompt=SYSTEM_PROMPT,
                history=state.conversation_history,
                user_msg=user_msg,
                session_id=session_id,
                cycle=global_cycle,
            )
        except llm_client.LLMError as exc:
            logger.error("[%s] cycle=%d LLM permanent failure: %s", session_id, global_cycle, exc)
            state = await sm.load_session(session_id)
            state.status = SessionStatus.ERROR
            state.error_message = str(exc)
            await sm.save_session(state)
            return

        # Grow conversation history (user turn)
        state = await sm.load_session(session_id)
        state.conversation_history.append({"role": "user", "content": user_msg})
        # Keep history bounded to last 20 turns to avoid token bloat
        if len(state.conversation_history) > 20:
            state.conversation_history = state.conversation_history[-20:]
        await sm.save_session(state)

        # ── 3. Parse ──────────────────────────────────────────────────────────
        state = await sm.load_session(session_id)
        state.status = SessionStatus.PARSING
        await sm.save_session(state)

        parsed = parser.parse_alpha_output(raw_response)

        if parsed is None:
            logger.warning("[%s] cycle=%d parse failed — injecting re-generation", session_id, global_cycle)
            # Append assistant turn and inject a parse-failure recovery message
            state = await sm.load_session(session_id)
            state.conversation_history.append({"role": "assistant", "content": raw_response})
            recovery_msg = (
                f"Your last response (cycle {global_cycle}) could not be parsed into "
                "the required 7-field format. Please re-output the alpha using the "
                "exact structure specified in Section 6, starting with [1] ECONOMIC RATIONALE."
            )
            state.conversation_history.append({"role": "user", "content": recovery_msg})
            state.status = SessionStatus.ITERATING
            await sm.save_session(state)
            await asyncio.sleep(1)  # brief pause before retry
            continue

        decision = parsed.get("decision", "ITERATE")
        expression = parsed.get("expression", "")
        fingerprint_dict = parsed.get("fingerprint", {})
        family = parsed.get("family") or "Unknown"
        ast_topology = parsed.get("ast_topology")
        ast_collision = parsed.get("ast_collision", [])
        simulation_payload = parsed.get("simulation_payload")

        # v2: update dataset_usage counter
        dataset_family = fingerprint_dict.get("dataset") or "Unknown"
        state = await sm.load_session(session_id)
        state.dataset_usage[dataset_family] = state.dataset_usage.get(dataset_family, 0) + 1
        await sm.save_session(state)

        logger.info(
            "[%s] cycle=%d parsed — decision=%s family=%s topology=%s",
            session_id, global_cycle, decision, family, ast_topology,
        )

        # ── 4. Validate ───────────────────────────────────────────────────────
        state = await sm.load_session(session_id)
        state.status = SessionStatus.VALIDATING
        await sm.save_session(state)

        syntax_result = val.validate_syntax(expression)
        metrics_result = val.validate_metrics(parsed)
        collision = val.fingerprint_collision(fingerprint_dict, state.fingerprint_memory)

        # v2: AST topology collision check (runs before metrics validation)
        topology_collision = False
        if ast_topology and ast_topology in state.topology_map:
            existing_status = state.topology_map[ast_topology]
            if existing_status in ("FAILED", "CROWDED"):
                topology_collision = True
                logger.info(
                    "[%s] cycle=%d topology collision: '%s' is %s",
                    session_id, global_cycle, ast_topology, existing_status,
                )

        # v2: dataset exhaustion check
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

        all_failures = syntax_result.failures + metrics_result.failures
        all_warnings = syntax_result.warnings + metrics_result.warnings

        if all_warnings:
            logger.info("[%s] cycle=%d warnings: %s", session_id, global_cycle, all_warnings)

        # ── 5. Decision branching ─────────────────────────────────────────────
        state = await sm.load_session(session_id)

        # Register assistant turn in history now that we've processed it
        state.conversation_history.append({"role": "assistant", "content": raw_response})
        if len(state.conversation_history) > 20:
            state.conversation_history = state.conversation_history[-20:]

        # v2: update topology map with current topology
        if ast_topology:
            # Will be overwritten to PASSED/FAILED/CROWDED at end of decision branching
            if ast_topology not in state.topology_map:
                state.topology_map[ast_topology] = "EXPLORING"

        if decision == "REJECT" or collision or topology_collision or dataset_exhausted:
            reason = "REJECT decision"
            if collision:          reason = "fingerprint collision"
            if topology_collision: reason = "AST topology collision"
            if dataset_exhausted:  reason = "dataset exhausted"
            logger.info("[%s] cycle=%d restart — %s", session_id, global_cycle, reason)

            state.rejected_motifs.append(fingerprint_dict)
            if ast_topology:
                state.topology_map[ast_topology] = "CROWDED"
            state.mutation_count = 0
            state.consecutive_same_decision = 0
            state.last_decision = "REJECT"
            state.status = SessionStatus.FAIL

            # v2: log to failure catalog
            state.failure_catalog.append({
                "fingerprint": fingerprint_dict,
                "failure_type": "CROWDED",
                "metric_value": reason,
                "mutation_tried": "restart",
            })

            mem_summary = _summarise_rejected(state.rejected_motifs)
            restart_msg = build_restart_trigger(global_cycle + 1, mem_summary)
            state.conversation_history.append({"role": "user", "content": restart_msg})
            await sm.save_session(state)
            await asyncio.sleep(1)
            continue

        if all_failures or decision == "ITERATE":
            # Mutation requested
            state.mutation_count += 1

            if state.mutation_count > settings.MAX_MUTATIONS:
                # Exhausted mutations — force full restart
                logger.info(
                    "[%s] cycle=%d mutation cap (%d) reached — restarting ideation",
                    session_id, global_cycle, settings.MAX_MUTATIONS,
                )
                state.rejected_motifs.append(fingerprint_dict)
                state.mutation_count = 0
                state.status = SessionStatus.FAIL
                mem_summary = _summarise_rejected(state.rejected_motifs)
                restart_msg = build_restart_trigger(global_cycle + 1, mem_summary)
                state.conversation_history.append({"role": "user", "content": restart_msg})
                await sm.save_session(state)
                await asyncio.sleep(1)
                continue

            # Track consecutive same-decision; force restart after 3 identical
            if state.last_decision == decision:
                state.consecutive_same_decision += 1
            else:
                state.consecutive_same_decision = 1
            state.last_decision = decision

            if state.consecutive_same_decision >= 3:
                logger.info(
                    "[%s] cycle=%d same decision '%s' x3 — forcing restart",
                    session_id, global_cycle, decision,
                )
                state.rejected_motifs.append(fingerprint_dict)
                state.mutation_count = 0
                state.consecutive_same_decision = 0
                state.status = SessionStatus.FAIL
                mem_summary = _summarise_rejected(state.rejected_motifs)
                restart_msg = build_restart_trigger(global_cycle + 1, mem_summary)
                state.conversation_history.append({"role": "user", "content": restart_msg})
                await sm.save_session(state)
                await asyncio.sleep(1)
                continue

            state.status = SessionStatus.ITERATING
            failure_msg = build_failure_feedback(
                failures=all_failures if all_failures else [f"Decision was {decision}"],
                expression=expression,
                cycle=global_cycle,
                values=_extract_metric_values(parsed),
            )
            # v2: log to failure catalog
            state.failure_catalog.append({
                "fingerprint": fingerprint_dict,
                "failure_type": all_failures[0] if all_failures else decision,
                "metric_value": _extract_metric_values(parsed),
                "mutation_tried": "pending",
            })
            if ast_topology:
                state.topology_map[ast_topology] = "FAILED"
            state.conversation_history.append({"role": "user", "content": failure_msg})
            await sm.save_session(state)
            await asyncio.sleep(1)
            continue

        # ── 6. PASS path ──────────────────────────────────────────────────────
        alpha_id = f"A{len(state.passed_alphas) + 1:03d}"
        # Strip returns_pct from metrics before creating AlphaMetrics
        raw_metrics = dict(parsed["metrics"])
        returns_pct = raw_metrics.pop("returns_pct", None)
        metrics_obj = AlphaMetrics(**raw_metrics)
        metrics_obj.returns_pct = returns_pct
        # Attach computed fitness if available
        if metrics_result.fitness_computed is not None:
            metrics_obj.fitness_computed = metrics_result.fitness_computed
            metrics_obj.fitness_breakdown = metrics_result.fitness_breakdown

        fp_obj = AlphaFingerprint(**{
            k: v for k, v in fingerprint_dict.items()
            if k in AlphaFingerprint.model_fields
        })

        alpha = AlphaResult(
            alpha_id=alpha_id,
            family=family,
            expression=expression,
            rationale=parsed.get("rationale", ""),
            metrics=metrics_obj,
            fingerprint=fp_obj,
            decision=decision,
            refinement_log=parsed.get("refinement_log"),
            mutation_paths=parsed.get("mutation_paths", []),
            # v2 fields
            ast_topology=ast_topology,
            ast_collision=ast_collision,
            simulation_payload=simulation_payload,
            cycle_num=global_cycle,
            passed=True,
        )

        state.passed_alphas.append(alpha)
        state.fingerprint_memory.append(fingerprint_dict)
        state.family_run_tracker.append(family)
        # v2: mark topology as PASSED
        if ast_topology:
            state.topology_map[ast_topology] = "PASSED"
        state.mutation_count = 0
        state.consecutive_same_decision = 0
        state.last_decision = decision
        state.status = SessionStatus.PASS

        logger.info(
            "[%s] cycle=%d PASS — alpha_id=%s family=%s decision=%s",
            session_id, global_cycle, alpha_id, family, decision,
        )

        # Success feedback seeds next cycle with anti-crowding context
        if global_cycle < settings.MAX_CYCLES:
            success_msg = build_success_feedback(
                alpha_id=alpha_id,
                cycle=global_cycle,
                next_cycle=global_cycle + 1,
                fingerprint=fingerprint_dict,
                all_fingerprints=state.fingerprint_memory,
            )
            state.conversation_history.append({"role": "user", "content": success_msg})

        await sm.save_session(state)

        # ── 7. BRAIN submission + 20-mutation improvement loop ───────────────
        if settings.BRAIN_SUBMIT_ENABLED and alpha.simulation_payload:
            state = await sm.load_session(session_id)
            state.status = SessionStatus.SUBMITTING
            state.brain_mutation_count = 0
            state.current_brain_alpha_id = None
            _log(state, "SUBMIT", f"Submitting {alpha_id} to BRAIN…", {"expr": expression[:80]})
            await sm.save_session(state)

            brain_result = await _submit_to_brain(alpha, session_id, global_cycle)

            state = await sm.load_session(session_id)
            state.current_brain_alpha_id = brain_result.alpha_id
            _log_brain_result(state, brain_result, expression, attempt=0)
            await sm.save_session(state)

            if brain_result.status == BrainSimStatus.FAIL:
                brain_result = await _brain_improvement_loop(
                    initial_result=brain_result,
                    initial_expression=expression,
                    alpha=alpha,
                    session_id=session_id,
                    global_cycle=global_cycle,
                )

            # Save final result
            state = await sm.load_session(session_id)
            alpha.brain = brain_result
            for i, a in enumerate(state.passed_alphas):
                if a.alpha_id == alpha_id:
                    state.passed_alphas[i] = alpha
                    break

            if brain_result.status == BrainSimStatus.PASS:
                state.status = SessionStatus.PASS
                s = f"{brain_result.real_sharpe:.3f}" if brain_result.real_sharpe is not None else "N/A"
                f = f"{brain_result.real_fitness:.3f}" if brain_result.real_fitness is not None else "N/A"
                _log(state, "PASS",
                     f"Alpha {alpha_id} PASSED all BRAIN checks! "
                     f"Sharpe={s} "
                     f"Fitness={f}",
                     {"brain_id": brain_result.alpha_id})
            else:
                state.status = SessionStatus.ITERATING
                if brain_result.status == BrainSimStatus.ERROR:
                    _log(state, "ABANDON",
                         f"Alpha {alpha_id} abandoned due to API error: {brain_result.error_message}",
                         {"failures": brain_result.gate_failures})
                else:
                    _log(state, "ABANDON",
                         f"Alpha {alpha_id} exhausted {state.brain_mutation_count} mutations — "
                         "starting fresh ideation.",
                         {"failures": brain_result.gate_failures})

                state.failure_catalog.append({
                    "fingerprint": fingerprint_dict,
                    "failure_type": "API_ERROR" if brain_result.status == BrainSimStatus.ERROR else "BRAIN_EXHAUSTED",
                    "mutations_tried": state.brain_mutation_count,
                })
                state.conversation_history.append({
                    "role": "user",
                    "content": build_restart_trigger(
                        global_cycle + 1,
                        _summarise_rejected([fingerprint_dict] + state.rejected_motifs[-3:])
                    )
                })
            await sm.save_session(state)

        await asyncio.sleep(3)

    # ── Loop completed max cycles ─────────────────────────────────────────────
    state = await sm.load_session(session_id)
    if state and not state.stop_requested:
        state.status = SessionStatus.STOPPED
        await sm.save_session(state)
    logger.info("[%s] Loop completed — %d cycles, %d alphas passed", session_id, settings.MAX_CYCLES, len(state.passed_alphas) if state else 0)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _family_locked(state) -> bool:
    """True if the last 3 generations all used the same family."""
    tracker = state.family_run_tracker
    if len(tracker) < 3:
        return False
    return len(set(tracker[-3:])) == 1


def _last_family(state) -> str:
    return state.family_run_tracker[-1] if state.family_run_tracker else "Unknown"


def _build_continuation_msg(state, cycle: int) -> str:
    """For cycles > 1 where the previous iter already appended feedback, just nudge."""
    # The last element of conversation_history is the injected feedback message.
    # We return an empty nudge so the LLM knows to continue.
    return f"Continue. Generate Alpha {cycle} now."


def _summarise_rejected(motifs: list[dict]) -> str:
    if not motifs:
        return "None yet."
    lines = []
    for i, m in enumerate(motifs, 1):
        parts = ", ".join(f"{k}={v}" for k, v in m.items() if v)
        lines.append(f"  Motif-{i}: {parts}")
    return "\n".join(lines)


def _extract_metric_values(parsed: dict) -> dict:
    """Pull numeric metric values for inclusion in failure feedback."""
    m = parsed.get("metrics", {})
    result = {}
    if m.get("sharpe_min") is not None:
        result["sharpe"] = f"{m['sharpe_min']}–{m.get('sharpe_max', m['sharpe_min'])}"
    if m.get("turnover_min") is not None:
        result["turnover"] = f"{m['turnover_min']}%–{m.get('turnover_max', m['turnover_min'])}%"
    if m.get("fitness_min") is not None:
        result["fitness"] = str(m["fitness_min"])
    if m.get("corr_risk"):
        result["corr_risk"] = m["corr_risk"]
    return result


async def _submit_to_brain(alpha, session_id: str, cycle: int) -> "BrainSubmissionResult":
    """
    Authenticate with BRAIN (cached) then submit simulation and poll.
    Returns a BrainSubmissionResult regardless of outcome.
    """
    global _brain_cookies
    from datetime import datetime as dt
    from models import BrainSimStatus, BrainSubmissionResult

    result = BrainSubmissionResult(
        status=BrainSimStatus.PENDING,
        submitted_at=dt.utcnow(),
    )

    try:
        # ── Auth (cached per process) ─────────────────────────────────────
        async with _brain_cookies_lock:
            if _brain_cookies is None:
                logger.info("[%s] cycle=%d Authenticating with BRAIN as %s",
                            session_id, cycle, settings.BRAIN_EMAIL)
                _brain_cookies = await brain_client.authenticate(
                    settings.BRAIN_EMAIL,
                    settings.BRAIN_PASSWORD,
                )

        # ── Submit + poll ─────────────────────────────────────────────────
        logger.info("[%s] cycle=%d Submitting alpha to BRAIN...", session_id, cycle)
        gate = await brain_client.submit_and_poll(
            simulation_payload=alpha.simulation_payload,
            cookies=_brain_cookies,
            max_poll_seconds=settings.BRAIN_POLL_TIMEOUT,
        )

        result.alpha_id      = gate.alpha_id
        result.real_sharpe   = gate.sharpe
        result.real_fitness  = gate.fitness
        result.real_turnover = gate.turnover
        result.real_returns  = gate.returns
        result.real_drawdown = gate.drawdown
        result.gate_failures = gate.failures
        result.gate_warnings = gate.warnings
        result.completed_at  = dt.utcnow()
        result.status        = BrainSimStatus.PASS if gate.passed else BrainSimStatus.FAIL

    except brain_client.BrainAuthError as exc:
        logger.error("[%s] cycle=%d BRAIN auth error: %s", session_id, cycle, exc)
        result.status        = BrainSimStatus.ERROR
        result.error_message = f"Auth error: {exc}"
        # Invalidate cached cookies so next cycle retries auth
        async with _brain_cookies_lock:
            _brain_cookies = None

    except (brain_client.BrainSubmitError, brain_client.BrainPollError) as exc:
        logger.error("[%s] cycle=%d BRAIN submission error: %s", session_id, cycle, exc)
        result.status        = BrainSimStatus.ERROR
        result.error_message = str(exc)

    except Exception as exc:
        logger.error("[%s] cycle=%d Unexpected BRAIN error: %s", session_id, cycle, exc, exc_info=True)
        result.status        = BrainSimStatus.ERROR
        result.error_message = f"Unexpected: {exc}"

    return result


# ── Activity log helpers ────────────────────────────────────────────────────────────────────────────────────

def _log(state, log_type: str, message: str, detail: dict | None = None) -> None:
    """Append an event to the session activity_log (max 200 entries)."""
    from datetime import datetime as dt
    entry = {
        "time": dt.utcnow().strftime("%H:%M:%S"),
        "type": log_type,
        "message": message,
    }
    if detail:
        entry["detail"] = detail
    if not hasattr(state, "activity_log") or state.activity_log is None:
        state.activity_log = []
    state.activity_log.append(entry)
    state.activity_log = state.activity_log[-200:]   # keep last 200
    logger.info("[activity] [%s] %s", log_type, message)


def _log_brain_result(state, result, expression: str, attempt: int) -> None:
    """Log a BRAIN simulation result to the activity log."""
    checks = getattr(result, "brain_checks", []) or []
    failures = [c for c in checks if c.get("result") == "FAIL"]
    passing  = [c for c in checks if c.get("result") == "PASS"]
    err_msg  = getattr(result, "gate_failures", [])

    if result.status.value == "PASS":
        s = f"{result.real_sharpe:.3f}" if result.real_sharpe is not None else "N/A"
        f = f"{result.real_fitness:.3f}" if result.real_fitness is not None else "N/A"
        t = f"{result.real_turnover:.1f}%" if result.real_turnover is not None else "N/A"
        _log(state, "BRAIN_PASS",
             f"BRAIN PASS — Sharpe={s} Fitness={f} TO={t}",
             {"brain_id": result.alpha_id, "attempt": attempt})
        return

    # Log each failing check separately so UI can show per-check status
    if failures:
        for chk in failures:
            _log(state, "BRAIN_FAIL",
                 f"BRAIN check FAILED: {chk['name']} "
                 f"(value={chk.get('value')}, limit={chk.get('limit')})",
                 {"check": chk["name"], "value": chk.get("value"),
                  "limit": chk.get("limit"), "attempt": attempt})
    elif err_msg:
        _log(state, "BRAIN_ERROR", f"BRAIN simulation ERROR: {err_msg[0][:120]}",
             {"attempt": attempt})

    _log(state, "METRICS",
         f"Real metrics — Sharpe={result.real_sharpe} "
         f"Fitness={result.real_fitness} TO={result.real_turnover}% "
         f"Returns={result.real_returns}%",
         {"sharpe": result.real_sharpe, "fitness": result.real_fitness,
          "turnover": result.real_turnover, "returns": result.real_returns})


# ── 20-mutation BRAIN improvement loop ────────────────────────────────────────────────────

async def _brain_improvement_loop(
    initial_result,
    initial_expression: str,
    alpha,
    session_id: str,
    global_cycle: int,
) -> "BrainSubmissionResult":
    """
    After an initial BRAIN FAIL, iteratively mutate and re-submit the same alpha
    up to MAX_BRAIN_MUTATIONS (20) times before giving up.

    Each iteration:
      1. Build targeted mutation prompt from real BRAIN check failures
      2. Call LLM for improved expression
      3. PATCH existing BRAIN alpha + re-simulate
      4. Log every step to activity_log
      5. Return on PASS or after 20 mutations
    """
    from datetime import datetime as dt
    from models import BrainSimStatus, BrainSubmissionResult

    current_result = initial_result
    current_expression = initial_expression
    brain_alpha_id = initial_result.alpha_id

    for attempt in range(1, MAX_BRAIN_MUTATIONS + 1):
        state = await sm.load_session(session_id)
        state.brain_mutation_count = attempt
        state.status = SessionStatus.ITERATING

        # Build targeted mutation prompt
        brain_checks = getattr(current_result, "brain_checks", []) or []
        error_msg = ""
        if not brain_checks and current_result.gate_failures:
            error_msg = current_result.gate_failures[0]

        targeted_prompt = build_brain_failure_feedback(
            brain_checks=brain_checks,
            expression=current_expression,
            cycle=global_cycle,
            real_sharpe=current_result.real_sharpe,
            real_fitness=current_result.real_fitness,
            real_turnover=current_result.real_turnover,
            real_returns=current_result.real_returns,
            brain_alpha_id=brain_alpha_id,
            mutation_attempt=attempt,
            error_message=error_msg,
        )

        failed_checks = [c["name"] for c in brain_checks if c.get("result") == "FAIL"]
        _log(state, "MUTATE",
             f"Mutation {attempt}/{MAX_BRAIN_MUTATIONS} — fixing: "
             f"{', '.join(failed_checks) or error_msg[:60]}",
             {"attempt": attempt, "checks": failed_checks})

        # Append to conversation and call LLM
        state.conversation_history.append({"role": "user", "content": targeted_prompt})
        if len(state.conversation_history) > 20:
            state.conversation_history = state.conversation_history[-20:]
        await sm.save_session(state)

        try:
            raw = await llm_client.generate(
                system_prompt=SYSTEM_PROMPT,
                history=state.conversation_history[:-1],
                user_msg=targeted_prompt,
                session_id=session_id,
                cycle=global_cycle,
            )
        except llm_client.LLMError as exc:
            logger.error("[%s] LLM error during mutation %d: %s", session_id, attempt, exc)
            state = await sm.load_session(session_id)
            _log(state, "ERROR", f"LLM error on mutation {attempt}: {exc}")
            await sm.save_session(state)
            await asyncio.sleep(5)
            continue

        # Parse just the expression from response
        import re
        expr_match = re.search(
            r"(?:expression|alpha)[:\s]+([^\n]+group_neutralize[^\n]+)",
            raw, re.IGNORECASE
        )
        if not expr_match:
            # Fall back: find any line with group_neutralize
            for line in raw.splitlines():
                if "group_neutralize" in line.lower():
                    expr_match = re.search(r"(group_neutralize\(.+\))", line)
                    if expr_match:
                        break

        if not expr_match:
            state = await sm.load_session(session_id)
            _log(state, "ERROR",
                 f"Mutation {attempt}: could not parse expression from LLM — retrying")
            await sm.save_session(state)
            await asyncio.sleep(3)
            continue

        new_expression = expr_match.group(1).strip().rstrip(",;")
        state = await sm.load_session(session_id)
        _log(state, "LLM",
             f"Mutation {attempt} expression: {new_expression[:80]}…",
             {"expression": new_expression})
        await sm.save_session(state)

        # PATCH + re-simulate
        try:
            async with _brain_cookies_lock:
                if _brain_cookies is None:
                    _brain_cookies = await brain_client.authenticate(
                        settings.BRAIN_EMAIL, settings.BRAIN_PASSWORD
                    )

            state = await sm.load_session(session_id)
            state.status = SessionStatus.SUBMITTING
            _log(state, "PATCH",
                 f"Patching BRAIN alpha {brain_alpha_id} + re-simulating…",
                 {"brain_id": brain_alpha_id, "attempt": attempt})
            await sm.save_session(state)

            gate = await brain_client.patch_alpha_and_poll(
                alpha_id=brain_alpha_id,
                new_expression=new_expression,
                simulation_payload=alpha.simulation_payload or {},
                cookies=_brain_cookies,
                max_poll_seconds=settings.BRAIN_POLL_TIMEOUT,
            )

            new_result = BrainSubmissionResult(
                status=BrainSimStatus.PASS if gate.passed else BrainSimStatus.FAIL,
                alpha_id=gate.alpha_id or brain_alpha_id,
                real_sharpe=gate.sharpe,
                real_fitness=gate.fitness,
                real_turnover=gate.turnover,
                real_returns=gate.returns,
                real_drawdown=gate.drawdown,
                gate_failures=gate.failures,
                gate_warnings=gate.warnings,
                brain_checks=gate.brain_checks,
                completed_at=dt.utcnow(),
            )

        except (brain_client.BrainSubmitError, brain_client.BrainPollError) as exc:
            logger.error("[%s] BRAIN error on mutation %d: %s", session_id, attempt, exc)
            state = await sm.load_session(session_id)
            _log(state, "ERROR", f"BRAIN error on mutation {attempt}: {exc}")
            await sm.save_session(state)
            await asyncio.sleep(5)
            continue
        except Exception as exc:
            logger.error("[%s] Unexpected error on mutation %d: %s", session_id, attempt, exc)
            await asyncio.sleep(5)
            continue

        current_expression = new_expression
        current_result = new_result

        state = await sm.load_session(session_id)
        _log_brain_result(state, new_result, new_expression, attempt=attempt)
        await sm.save_session(state)

        if new_result.status == BrainSimStatus.PASS:
            logger.info(
                "[%s] BRAIN PASS on mutation %d — sharpe=%.3f",
                session_id, attempt, new_result.real_sharpe or 0,
            )
            return new_result

        await asyncio.sleep(3)

    # Exhausted 20 mutations
    logger.warning("[%s] Exhausted %d BRAIN mutations — abandoning alpha",
                   session_id, MAX_BRAIN_MUTATIONS)
    return current_result
