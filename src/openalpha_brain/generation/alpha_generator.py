from __future__ import annotations

import logging
import re
from datetime import UTC, datetime

from openalpha_brain.cli import session_manager as sm
from openalpha_brain.config.config import settings
from openalpha_brain.core.loop_state import _failure_lib
from openalpha_brain.core.models import AlphaFingerprint, AlphaMetrics, AlphaResult, SessionStatus
from openalpha_brain.generation import alpha_parser as parser
from openalpha_brain.generation.prompts import build_failure_feedback, build_restart_trigger
from openalpha_brain.validation import validator as val
from openalpha_brain.validation.validator import verify_economic_explanation

logger = logging.getLogger(__name__)


def _summarise_rejected(motifs: list[dict]) -> str:
    """[Brief description of function purpose.]

    Args:
        motifs (list[dict]): [Description]

    Returns:
        str: [Description]
    """
    if not motifs:
        return "None yet."
    lines = []
    for i, m in enumerate(motifs, 1):
        parts = ", ".join(f"{k}={v}" for k, v in m.items() if v)
        lines.append(f"  Motif-{i}: {parts}")
    return "\n".join(lines)


def _extract_metric_values(parsed: dict) -> dict:
    """[Brief description of function purpose.]

    Args:
        parsed (dict): [Description]

    Returns:
        dict: [Description]
    """
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


def _record_hallucination(state, variable: str, error_type: str, message: str, source: str) -> None:
    """[Brief description of function purpose.]

    Args:
        state: [Description]
        variable (str): [Description]
        error_type (str): [Description]
        message (str): [Description]
        source (str): [Description]

    Returns:
        None: [Description]
    """
    state.hallucination_log.append(
        {
            "variable": variable,
            "error_type": error_type,
            "error_message": message,
            "source": source,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    )


def _extract_hallucinations_from_failures(state, syntax_result) -> None:
    """[Brief description of function purpose.]

    Args:
        state: [Description]
        syntax_result: [Description]

    Returns:
        None: [Description]
    """
    if syntax_result.passed:
        return
    for fail in syntax_result.failures:
        if "INVALID BRAIN variables" in fail:
            bad_vars = re.findall(r"([a-z_][a-z0-9_]*)", fail.split(":")[-1])
            for bv in bad_vars:
                bv = bv.strip().rstrip(",")
                if bv:
                    _record_hallucination(
                        state, bv, "INVALID_VAR", f"Variable '{bv}' is not in BRAIN data fields", "validator"
                    )
        elif "Unknown/forbidden operators" in fail:
            bad_ops = re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)", fail.split(":")[-1])
            for bo in bad_ops:
                bo = bo.strip().rstrip(",")
                if bo:
                    _record_hallucination(
                        state, bo, "UNKNOWN_OPERATOR", f"Operator '{bo}' is not in BRAIN operator schema", "validator"
                    )


def _extract_brain_hallucinations(state, error_messages: list[str]) -> None:
    """[Brief description of function purpose.]

    Args:
        state: [Description]
        error_messages (list[str]): [Description]

    Returns:
        None: [Description]
    """
    for em in error_messages[:3]:
        unknown_vars = re.findall(r"unknown variable\s+[\"']?([a-zA-Z_][a-zA-Z0-9_]*)[\"']?", em, re.IGNORECASE)
        for uv in unknown_vars:
            _record_hallucination(state, uv, "BRAIN_UNKNOWN_VAR", f"BRAIN rejected variable '{uv}': {em[:80]}", "brain")


def _extract_brain_feedback_from_state(state) -> list[dict]:
    """[Brief description of function purpose.]

    Args:
        state: [Description]

    Returns:
        list[dict]: [Description]
    """
    feedback = []
    for alpha in state.passed_alphas[-3:]:
        brain = getattr(alpha, "brain", None)
        if brain is None:
            continue
        feedback.append(
            {
                "sharpe": brain.real_sharpe,
                "status": brain.status.value if brain.status else "UNKNOWN",
                "feedback": "; ".join(brain.gate_failures[:3]) if brain.gate_failures else "",
            }
        )
    for fc in getattr(state, "failure_catalog", [])[-3:]:
        failure_type = fc.get("failure_type", "UNKNOWN")
        fingerprint = fc.get("fingerprint", {})
        mutations_tried = fc.get("mutations_tried", 0)
        fb_parts = [f"ABANDONED ({failure_type})"]
        if mutations_tried:
            fb_parts.append(f"mutations_tried={mutations_tried}")
        dataset = fingerprint.get("dataset", "")
        topology = fingerprint.get("topology", "")
        if dataset:
            fb_parts.append(f"dataset={dataset}")
        if topology:
            fb_parts.append(f"topology={topology}")
        feedback.append(
            {
                "sharpe": None,
                "status": "ABANDONED",
                "feedback": "; ".join(fb_parts),
            }
        )
    return feedback


def _is_fastexpr_line(line: str) -> bool:
    """[Brief description of function purpose.]

    Args:
        line (str): [Description]

    Returns:
        bool: [Description]
    """
    if "(" not in line or ")" not in line:
        return False
    if re.match(r"^(\*\*|##|>|-|\d+\.)\s*", line):
        return False
    nl_start_words = (
        "Rationale",
        "Hypothesis",
        "Strategy",
        "Insight",
        "Note",
        "Idea",
        "Inefficiency",
        "This",
        "The",
        "When",
        "If",
        "For",
    )
    first_word = line.split()[0].rstrip(":,") if line.split() else ""
    if first_word in nl_start_words:
        return False
    return not re.search(r":\s+[A-Z]", line)


def _extract_economic_rationale(raw: str) -> str | None:
    """[Brief description of function purpose.]

    Args:
        raw (str): [Description]

    Returns:
        str | None: [Description]
    """
    m = re.search(r"ECONOMIC[_ ]RATIONALE\s*:\s*(.+)", raw, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _apply_economic_rationale_verification(
    expression: str,
    economic_rationale: str | None,
    raw_response: str,
    state,
    session_id: str,
    global_cycle: int,
    exploration_direction: str,
) -> dict | None:
    """[Brief description of function purpose.]

    Args:
        expression (str): [Description]
        economic_rationale (str | None): [Description]
        raw_response (str): [Description]
        state: [Description]
        session_id (str): [Description]
        global_cycle (int): [Description]
        exploration_direction (str): [Description]

    Returns:
        dict | None: [Description]
    """
    if not economic_rationale:
        economic_rationale = _extract_economic_rationale(raw_response)
    if not economic_rationale:
        economic_rationale = ""
    verification = verify_economic_explanation(expression, economic_rationale, exploration_direction)
    if not verification["valid"]:
        logger.warning(
            "[%s] cycle=%d Economic explanation invalid: %s",
            session_id,
            global_cycle,
            verification.get("reason", ""),
        )
    if verification.get("consistency_score", 1.0) < 0.3:
        state.conversation_history.append(
            {
                "role": "user",
                "content": (
                    "Your previous alpha lacked a clear economic rationale. "
                    "Next time, provide ECONOMIC_RATIONALE: <explanation> "
                    "describing the market inefficiency your expression captures."
                ),
            }
        )
    return verification


def _extract_expression_from_llm(raw: str) -> str | None:
    """[Brief description of function purpose.]

    Args:
        raw (str): [Description]

    Returns:
        str | None: [Description]
    """
    import json as _json

    _JSON_ARTIFACTS_RE = re.compile(
        r"\b(?:expression|regular|simulation_payload|settings|rationale|decision|"
        r"fingerprint|family|ast_topology|metrics|mutation_paths|refinement_log)\s*[:=]\s*",
    )

    parsed = parser.parse_alpha_output(raw)
    if parsed and parsed.get("expression"):
        expr = parsed["expression"]
        if not _JSON_ARTIFACTS_RE.search(expr):
            return expr

    stripped = raw.strip()
    if stripped.startswith("{") and "}" in stripped:
        try:
            _parsed_json = _json.loads(stripped)
            if isinstance(_parsed_json, dict):
                for _key in ("expression", "regular", "alpha", "code", "fastexpr"):
                    _candidate = _parsed_json.get(_key, "")
                    if _candidate and isinstance(_candidate, str) and "(" in _candidate and ")" in _candidate:
                        if not _JSON_ARTIFACTS_RE.search(_candidate):
                            clean_candidate = _candidate.strip().rstrip(",;")
                            if val.validate_syntax(clean_candidate).passed:
                                return clean_candidate
                            return clean_candidate
        except (_json.JSONDecodeError, ValueError):
            pass

        _json_expr_match = re.search(
            r'"expression"\s*:\s*"([^"]*(?:group_neutralize|ts_delta|ts_mean|ts_rank|ts_decay_linear|rank|ts_zscore|group_zscore)[^"]*)"',
            stripped,
        )
        if _json_expr_match:
            _extracted = _json_expr_match.group(1).replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
            if "(" in _extracted and ")" in _extracted:
                return _extracted

    for line in raw.splitlines():
        line = line.strip()
        if "group_neutralize" in line.lower():
            match = re.search(r"(group_neutralize\(.+\))", line)
            if match:
                expr = match.group(1).strip().rstrip(",;")
                if val.validate_syntax(expr).passed:
                    return expr

    for line in raw.splitlines():
        line = line.strip().rstrip(",;")
        if not _is_fastexpr_line(line):
            continue
        if len(line) > 20 and any(
            op in line for op in ("ts_", "rank(", "group_neutralize(", "ts_decay_linear(", "signed_power(")
        ):
            clean = re.sub(r"^[`*\s]+", "", line)
            clean = re.sub(r"[`*\s]+$", "", clean)
            if val.validate_syntax(clean).passed:
                return clean

    expr_match = re.search(
        r"(?:expression|alpha)[:\s]+([^\n]+group_neutralize[^\n]+)",
        raw,
        re.IGNORECASE,
    )
    if expr_match:
        expr = expr_match.group(1).strip().rstrip(",;")
        if not _JSON_ARTIFACTS_RE.search(expr):
            return expr

    if stripped and "(" in stripped and ")" in stripped:
        _BARE_EXPR_OPS = (
            "rank(",
            "ts_delta(",
            "ts_mean(",
            "ts_std_dev(",
            "ts_zscore(",
            "ts_decay_linear(",
            "ts_rank(",
            "ts_delay(",
            "ts_sum(",
            "ts_corr(",
            "group_neutralize(",
            "group_rank(",
            "group_zscore(",
            "signed_power(",
            "zscore(",
            "normalize(",
            "winsorize(",
            "ts_av_diff(",
            "ts_quantile(",
            "quantile(",
        )
        if any(op in stripped for op in _BARE_EXPR_OPS):
            clean = stripped.rstrip(",;")
            clean = re.sub(r"^[`*\s]+", "", clean)
            clean = re.sub(r"[`*\s]+$", "", clean)
            if not _JSON_ARTIFACTS_RE.search(clean):
                return clean

    return None


async def _handle_parse_failure(state, session_id: str, global_cycle: int, raw_response: str) -> None:
    """[Brief description of function purpose.]

    Args:
        state: [Description]
        session_id (str): [Description]
        global_cycle (int): [Description]
        raw_response (str): [Description]

    Returns:
        None: [Description]
    """
    logger.warning("[%s] cycle=%d parse failed — injecting re-generation", session_id, global_cycle)
    state.conversation_history.append({"role": "assistant", "content": raw_response})
    recovery_msg = (
        f"Your last response (cycle {global_cycle}) could not be parsed into "
        "the required 8-field format. Please re-output the alpha using the "
        "exact structure specified in Section 9, starting with [1] ECONOMIC RATIONALE."
    )
    state.conversation_history.append({"role": "user", "content": recovery_msg})
    state.status = SessionStatus.ITERATING
    await sm.save_session(state)


async def _handle_reject(
    state, session_id: str, global_cycle: int, fingerprint_dict: dict, ast_topology, reason: str
) -> None:
    """[Brief description of function purpose.]

    Args:
        state: [Description]
        session_id (str): [Description]
        global_cycle (int): [Description]
        fingerprint_dict (dict): [Description]
        ast_topology: [Description]
        reason (str): [Description]

    Returns:
        None: [Description]
    """
    logger.info("[%s] cycle=%d restart — %s", session_id, global_cycle, reason)
    state.rejected_motifs.append(fingerprint_dict)
    if ast_topology:
        state.topology_map[ast_topology] = "CROWDED"
    state.mutation_count = 0
    state.consecutive_same_decision = 0
    state.last_decision = "REJECT"
    state.status = SessionStatus.FAIL
    state.failure_catalog.append(
        {
            "fingerprint": fingerprint_dict,
            "failure_type": "CROWDED",
            "metric_value": reason,
            "mutation_tried": "restart",
        }
    )
    try:
        from openalpha_brain.core.loop_state import _ls as _als

        if _als._experience_distiller:
            _als._experience_distiller.record_single_failure(
                failure_pattern=f"reject_{reason.lower()}",
                fix_strategy=f"Change expression structure to avoid {reason}",
                direction=state.focus_area or "",
            )
    except (OSError, ValueError, RuntimeError):
        pass
    mem_summary = _summarise_rejected(state.rejected_motifs)
    restart_msg = build_restart_trigger(global_cycle + 1, mem_summary)
    state.conversation_history.append({"role": "user", "content": restart_msg})
    await sm.save_session(state)


async def _handle_iterate(
    state,
    session_id: str,
    global_cycle: int,
    parsed: dict,
    expression: str,
    fingerprint_dict: dict,
    ast_topology,
    all_failures: list,
    decision: str,
) -> None:
    """[Brief description of function purpose.]

    Args:
        state: [Description]
        session_id (str): [Description]
        global_cycle (int): [Description]
        parsed (dict): [Description]
        expression (str): [Description]
        fingerprint_dict (dict): [Description]
        ast_topology: [Description]
        all_failures (list): [Description]
        decision (str): [Description]

    Returns:
        None: [Description]
    """
    state.mutation_count += 1

    if state.mutation_count > settings.MAX_MUTATIONS:
        logger.info(
            "[%s] cycle=%d mutation cap (%d) reached — restarting ideation",
            session_id,
            global_cycle,
            settings.MAX_MUTATIONS,
        )
        state.rejected_motifs.append(fingerprint_dict)
        state.mutation_count = 0
        state.status = SessionStatus.FAIL
        mem_summary = _summarise_rejected(state.rejected_motifs)
        restart_msg = build_restart_trigger(global_cycle + 1, mem_summary)
        state.conversation_history.append({"role": "user", "content": restart_msg})
        await sm.save_session(state)
        return

    if state.last_decision == decision:
        state.consecutive_same_decision += 1
    else:
        state.consecutive_same_decision = 1
    state.last_decision = decision

    if state.consecutive_same_decision >= 3:
        logger.info(
            "[%s] cycle=%d same decision '%s' x3 — forcing restart",
            session_id,
            global_cycle,
            decision,
        )
        state.rejected_motifs.append(fingerprint_dict)
        state.mutation_count = 0
        state.consecutive_same_decision = 0
        state.status = SessionStatus.FAIL
        mem_summary = _summarise_rejected(state.rejected_motifs)
        restart_msg = build_restart_trigger(global_cycle + 1, mem_summary)
        state.conversation_history.append({"role": "user", "content": restart_msg})
        await sm.save_session(state)
        return

    state.status = SessionStatus.ITERATING
    failure_msg = build_failure_feedback(
        failures=all_failures if all_failures else [f"Decision was {decision}"],
        expression=expression,
        cycle=global_cycle,
        values=_extract_metric_values(parsed),
    )
    state.failure_catalog.append(
        {
            "fingerprint": fingerprint_dict,
            "failure_type": all_failures[0] if all_failures else decision,
            "metric_value": _extract_metric_values(parsed),
            "mutation_tried": "pending",
        }
    )
    if ast_topology:
        state.topology_map[ast_topology] = "FAILED"
    if _failure_lib and settings.FAILURE_FIX_LIBRARY_ENABLED:
        try:
            similar_fixes = await _failure_lib.search_by_expr_similarity(expression, top_k=3)
            if similar_fixes:
                fix_hints = []
                for sf in similar_fixes[:2]:
                    if sf.get("fix_attempt"):
                        fix_hints.append(f"Previous fix: {sf['fix_attempt']}")
                if fix_hints:
                    failure_msg += "\n\nKnown fixes for similar expressions:\n" + "\n".join(fix_hints)
        except (OSError, ValueError, RuntimeError):
            pass
        state.conversation_history.append({"role": "user", "content": failure_msg})
    await sm.save_session(state)


def _build_alpha(
    state,
    parsed: dict,
    expression: str,
    fingerprint_dict: dict,
    family: str,
    ast_topology,
    ast_collision,
    simulation_payload,
    metrics_result,
    global_cycle: int,
    decision: str,
    economic_rationale: str | None = None,
    rationale_verification: dict | None = None,
) -> AlphaResult:
    """[Brief description of function purpose.]

    Args:
        state: [Description]
        parsed (dict): [Description]
        expression (str): [Description]
        fingerprint_dict (dict): [Description]
        family (str): [Description]
        ast_topology: [Description]
        ast_collision: [Description]
        simulation_payload: [Description]
        metrics_result: [Description]
        global_cycle (int): [Description]
        decision (str): [Description]
        economic_rationale (str | None): [Description]
        rationale_verification (dict | None): [Description]

    Returns:
        AlphaResult: [Description]
    """
    alpha_id = f"A{len(state.passed_alphas) + 1:03d}"
    raw_metrics = {k: v for k, v in parsed["metrics"].items() if k != "returns_pct"}
    returns_pct = parsed["metrics"].get("returns_pct")
    metrics_obj = AlphaMetrics(**raw_metrics)
    metrics_obj.returns_pct = returns_pct
    if metrics_result.fitness_computed is not None:
        metrics_obj.fitness_computed = metrics_result.fitness_computed
        metrics_obj.fitness_breakdown = metrics_result.fitness_breakdown

    fp_obj = AlphaFingerprint(**{k: v for k, v in fingerprint_dict.items() if k in AlphaFingerprint.model_fields})

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
        ast_topology=ast_topology,
        ast_collision=ast_collision,
        simulation_payload=simulation_payload,
        cycle_num=global_cycle,
        passed=True,
        economic_rationale=economic_rationale,
    )

    if rationale_verification:
        if not alpha.simulation_payload:
            alpha.simulation_payload = {}
        alpha.simulation_payload["rationale_verification"] = rationale_verification

    state.passed_alphas.append(alpha)
    state.fingerprint_memory.append(fingerprint_dict)
    state.family_run_tracker.append(family)
    if ast_topology:
        state.topology_map[ast_topology] = "PASSED"
    state.mutation_count = 0
    state.consecutive_same_decision = 0
    state.last_decision = decision
    state.status = SessionStatus.PASS

    return alpha
