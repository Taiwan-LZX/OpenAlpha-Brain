from __future__ import annotations

import contextlib
import logging
import re

from openalpha_brain.config.config import settings
from openalpha_brain.core.loop_state import (
    _FIELDS_RE,
    _OPS_RE,
    _algo_tick,
    _association,
    _logic_library,
    _mab,
    _market_state_inferencer,
    _monitor,
    _rag_engine,
    _signal_arbiter,
    _success_rate_tracker,
    _whitelist_mgr,
)
from openalpha_brain.evolution.quality_diversity import StrategyFeatures
from openalpha_brain.knowledge.rag_tools import execute_rag_tool
from openalpha_brain.services import llm_client
from openalpha_brain.utils.algo_logger import algo_log
from openalpha_brain.validation import validator as val
from openalpha_brain.validation.signal_arbiter import (
    AssociationSignalAdapter,
    MABSignalAdapter,
    MarketSignalAdapter,
    RAGSignalAdapter,
    WhitelistSignalAdapter,
)

logger = logging.getLogger(__name__)


def _sync_mab_bias_from_evidence() -> None:
    """[Brief description of function purpose.]

    Returns:
        None: [Description]
    """
    if not settings.EVIDENCE_MAB_BIAS_ENABLED:
        return
    if not _mab or not _logic_library:
        return
    try:
        direction_weights = _logic_library.get_direction_weights()
        for d, w in direction_weights.items():
            _mab.set_initial_bias(d, w)
    except (ValueError, TypeError, OSError):
        pass


def _get_operators_from_context(rag_context: dict | None) -> list[str] | None:
    """[Brief description of function purpose.]

    Args:
        rag_context (dict | None): [Description]

    Returns:
        list[str] | None: [Description]
    """
    if not rag_context:
        return None
    ops = rag_context.get("top_ops_detailed", [])
    return [op["name"] for op in ops] if ops else None


def _get_fields_from_context(rag_context: dict | None) -> list[str] | None:
    """[Brief description of function purpose.]

    Args:
        rag_context (dict | None): [Description]

    Returns:
        list[str] | None: [Description]
    """
    if not rag_context:
        return None
    return rag_context.get("field_ids")


def _extract_ops_and_fields(expression: str) -> tuple[list[str], list[str]]:
    """[Brief description of function purpose.]

    Args:
        expression (str): [Description]

    Returns:
        tuple[list[str], list[str]]: [Description]
    """
    if not expression:
        return [], []
    operators = list(set(_OPS_RE.findall(expression)))
    all_vars = set(_FIELDS_RE.findall(expression))
    fields = [v for v in all_vars if v not in operators and not v.isdigit()]
    operators = [op for op in operators if op in val.PERMITTED_OPERATORS]
    valid_fields = {f.lower() for f in val.get_valid_brain_vars()}
    fields = [f for f in fields if f.lower() in valid_fields]
    return operators, fields


async def _arbiter_rerank(retrieval: dict, rag_context: dict, exploration_direction: str) -> dict:
    if not _signal_arbiter or not retrieval:
        return rag_context
    _algo_tick("signal_arbiter_rerank")
    try:
        top_op_id = ""
        top_field_id = ""
        ops_raw = retrieval.get("operators", [])
        fields_raw = retrieval.get("fields", [])
        if ops_raw:
            top_op_id = ops_raw[0].get("id", "") or ""
        if fields_raw:
            top_field_id = fields_raw[0].get("id", "") or ""
        if not top_op_id and not top_field_id:
            return rag_context

        rag_adapter = RAGSignalAdapter(retrieval)
        mab_adapter = MABSignalAdapter(_mab)
        assoc_adapter = AssociationSignalAdapter(_association, top_op_id, top_field_id)
        wl_adapter = WhitelistSignalAdapter(_whitelist_mgr)
        field_ids_for_market = [f.get("id", "") for f in fields_raw if f.get("id")]
        op_ids_for_market = [o.get("id", "") for o in ops_raw if o.get("id")]
        market_adapter = MarketSignalAdapter(
            _market_state_inferencer, exploration_direction, field_ids=field_ids_for_market, op_ids=op_ids_for_market
        )

        field_adapters = [rag_adapter, mab_adapter, assoc_adapter, wl_adapter, market_adapter]
        op_adapters = [rag_adapter, mab_adapter, assoc_adapter, market_adapter]

        ranked_fields, ranked_ops = await _signal_arbiter.rank_with_adapters(
            field_adapters,
            op_adapters,
            top_k_fields=settings.RAG_TOP_K_FIELDS,
            top_k_ops=settings.RAG_TOP_K_OPS,
        )

        logger.info(
            "[%s] MONITOR: arbiter_rerank: ranked_fields=%d ranked_ops=%d top_field_score=%.3f top_op_score=%.3f",
            exploration_direction,
            len(ranked_fields),
            len(ranked_ops),
            ranked_fields[0].final_score if ranked_fields else 0.0,
            ranked_ops[0].final_score if ranked_ops else 0.0,
        )

        if not ranked_fields and not ranked_ops:
            return rag_context

        if ranked_fields:
            ranked_field_ids = [r.item_id for r in ranked_fields]
            rag_context["field_ids"] = ranked_field_ids

        if ranked_ops:
            ranked_op_names = [r.item_id for r in ranked_ops]
            top_ops_detailed = rag_context.get("top_ops_detailed", [])
            name_to_detail = {op["name"]: op for op in top_ops_detailed}
            new_detailed = []
            new_remaining = []
            for name in ranked_op_names:
                if name in name_to_detail:
                    new_detailed.append(name_to_detail[name])
                else:
                    new_remaining.append(name)
            remaining_original = rag_context.get("remaining_op_names", [])
            seen = set(ranked_op_names)
            for n in remaining_original:
                if n not in seen:
                    new_remaining.append(n)
            rag_context["top_ops_detailed"] = new_detailed
            rag_context["remaining_op_names"] = new_remaining

        _monitor.record(
            "STEP",
            "signal_arbiter",
            "rank",
            f"top5_fields={[(r.item_id, round(r.final_score, 3)) for r in ranked_fields[:5]]} "
            f"top5_ops={[(r.item_id, round(r.final_score, 3)) for r in ranked_ops[:5]]}",
            session_id="",
        )
    except (OSError, ValueError, RuntimeError):
        logger.warning("SignalArbiter rerank failed", exc_info=True)
    return rag_context


def _make_tool_executor(rag_engine):
    """[Brief description of function purpose.]

    Args:
        rag_engine: [Description]
    """

    async def _executor(tool_name: str, arguments: dict) -> dict:
        """[Brief description of function purpose.]

        Args:
            tool_name (str): [Description]
            arguments (dict): [Description]

        Returns:
            dict: [Description]
        """
        return await execute_rag_tool(tool_name, arguments, rag_engine)

    return _executor


def _extract_allowed_fields_from_tool_results(tool_results: dict) -> set[str]:
    """[Brief description of function purpose.]

    Args:
        tool_results (dict): [Description]

    Returns:
        set[str]: [Description]
    """
    fields = set()
    for tool_name, result in tool_results.items():
        if tool_name == "search_fields" and isinstance(result, dict):
            for f in result.get("fields", []):
                fields.add(f.lower() if isinstance(f, str) else f)
    return fields


def _extract_allowed_operators_from_tool_results(tool_results: dict) -> set[str]:
    """[Brief description of function purpose.]

    Args:
        tool_results (dict): [Description]

    Returns:
        set[str]: [Description]
    """
    ops = set()
    for tool_name, result in tool_results.items():
        if tool_name == "search_operators" and isinstance(result, dict):
            for op in result.get("operators", []):
                if isinstance(op, dict) and "name" in op:
                    ops.add(op["name"])
    return ops


def _extract_strategy_features(expr: str, direction: str) -> StrategyFeatures:
    """[Brief description of function purpose.]

    Args:
        expr (str): [Description]
        direction (str): [Description]

    Returns:
        StrategyFeatures: [Description]
    """
    time_horizon = "short"
    lookback_nums = re.findall(r"(?:ts_\w+|ts_decay_linear|rank)\s*\([^)]*?(\d+)", expr)
    if not lookback_nums:
        lookback_nums = re.findall(r",\s*(\d+)", expr)
    for num_str in lookback_nums:
        num = int(num_str)
        if num > 30:
            time_horizon = "long"
            break
        if num > 10:
            time_horizon = "medium"
    mechanism = "signal"
    if "ts_std_dev" in expr or "/ ts_std_dev" in expr:
        mechanism = "normalized"
    elif "trade_when" in expr or "greater" in expr or "less" in expr:
        mechanism = "conditional"
    elif expr.count("rank(") >= 2 and "*" in expr:
        mechanism = "interaction"
    return StrategyFeatures(direction=direction, time_horizon=time_horizon, mechanism=mechanism)


@algo_log(log_args_to_skip=("expression",))
async def _apply_mab_feedback(
    exploration_direction: str, expression: str, reward: float = 0.0, penalty: float = 0.0, _scheduler=None
) -> None:
    """[Brief description of function purpose.]

    Args:
        exploration_direction (str): [Description]
        expression (str): [Description]
        reward (float): [Description]
        penalty (float): [Description]
        _scheduler: Optional ExplorationScheduler for TemplateFamilyBandit feedback.

    Returns:
        None: [Description]
    """
    if not expression or not exploration_direction:
        return
    _monitor.record("STEP", "mab", "feedback", f"direction={exploration_direction} reward={reward}", session_id="")
    if not _mab and not _association and not _whitelist_mgr:
        return
    try:
        operators, fields = _extract_ops_and_fields(expression)
        if not operators and not fields:
            return
        if _mab:
            _algo_tick("mab_update")
            _mab.update(exploration_direction, operators, fields, reward, penalty)
            logger.info(
                "[%s] MONITOR: mab_update: direction=%s reward=%.4f",
                exploration_direction,
                exploration_direction,
                reward,
            )
        if _scheduler is not None:
            try:
                _scheduler.record_direction_result(exploration_direction, reward=reward, penalty=penalty)
            except (OSError, ValueError, RuntimeError):
                logger.debug(
                    "[%s] scheduler record_direction_result failed in _apply_mab_feedback",
                    exploration_direction,
                    exc_info=True,
                )
        if _association:
            for op in operators:
                for field in fields:
                    _association.update(op, field, reward, penalty)
        if _whitelist_mgr:
            _algo_tick("whitelist_update")
            for field in fields:
                _whitelist_mgr.record_usage(field)
            if penalty > 0:
                for field in fields:
                    _whitelist_mgr.apply_overuse_penalty(field)
            overfit_warnings = _whitelist_mgr.detect_field_overfit()
            for w in overfit_warnings:
                logger.warning("[%s] Field overfit detected: %s", exploration_direction, w.get("message", ""))
                _monitor.record("WARN", "whitelist", "field_overfit", w.get("message", ""), session_id="")
    except (OSError, ValueError, RuntimeError):
        logger.warning("MAB feedback failed for direction=%s", exploration_direction, exc_info=True)

    if _signal_arbiter and _success_rate_tracker is not None:
        _success_rate_tracker.append(1 if reward > penalty else 0)
        success_rate = sum(_success_rate_tracker) / len(_success_rate_tracker) if _success_rate_tracker else 0.0
        _signal_arbiter.adjust_weights(success_rate)


async def _refill_eliminated_fields(exploration_direction: str) -> list[str]:
    """[Brief description of function purpose.]

    Args:
        exploration_direction (str): [Description]

    Returns:
        list[str]: [Description]
    """
    if not _whitelist_mgr:
        return []
    try:
        eliminated = _whitelist_mgr.check_eliminations()
        if not eliminated:
            return []
        logger.info("Fields eliminated from whitelist: %s", eliminated)
        if _rag_engine and _rag_engine.is_ready:
            _algo_tick("rag_retrieve")
            retrieval = await _rag_engine.retrieve(exploration_direction)
            rag_context = _rag_engine.assemble_context(retrieval)
            new_field_ids = rag_context.get("field_ids", [])
            if new_field_ids:
                _whitelist_mgr.update_dynamic(new_field_ids)
                _rag_engine.set_eliminated_fields(set(_whitelist_mgr.eliminated_fields.keys()))
                logger.info(
                    "Refilled %d fields via RAG after elimination of %d fields", len(new_field_ids), len(eliminated)
                )
        return eliminated
    except (OSError, ValueError, RuntimeError):
        logger.warning("Refill eliminated fields failed", exc_info=True)
        return []


async def _run_logic_evolution(cycle_count: int, session_id: str, global_cycle: int) -> int:
    """[Brief description of function purpose.]

    Args:
        cycle_count (int): [Description]
        session_id (str): [Description]
        global_cycle (int): [Description]

    Returns:
        int: [Description]
    """
    cycle_count += 1
    if cycle_count % 10 == 0 and _logic_library:
        try:
            _algo_tick("logic_evolution")
            evo_result = _logic_library.evolve_logics()
            logger.info(
                "[%s] Logic evolution at cycle %d: split=%d, merged=%d, gaps=%s",
                session_id,
                global_cycle,
                evo_result["split"],
                evo_result["merged"],
                evo_result["gaps"],
            )
            if evo_result["gaps"]:
                _algo_tick("logic_propose")
                for gap in evo_result["gaps"][:3]:
                    with contextlib.suppress(OSError, ValueError, RuntimeError):
                        await _logic_library.propose_new_logic(gap, llm_generate_fn=llm_client.generate)
        except (OSError, ValueError, RuntimeError):
            logger.warning("[%s] Logic evolution failed", session_id, exc_info=True)
    return cycle_count
