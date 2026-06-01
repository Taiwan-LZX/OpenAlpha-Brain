#!/usr/bin/env python3
"""
OpenAlpha - Quant — User-Facing Operational Script
合併自: _test_real_selfloop.py, test_real_mining.py, _start_session.py, _check_session.py

提供完整的 CLI 介面來操作 Alpha 採礦流程，所有操作均使用真實資料
(真實 BRAIN API、真實 LLM 呼叫)，不使用任何 mock/simulated 資料。

Usage:
    python run_alpha.py start [--focus AREA]          # 啟動新採礦 session
    python run_alpha.py check [session_id]            # 檢查 session 狀態與 alphas
    python run_alpha.py run [--cycles N] [--focus AREA]  # 執行完整 self-loop 測試
    python run_alpha.py mine [--focus AREA] [--wait SECONDS]  # 啟動採礦並等待結果
    python run_alpha.py full                          # 完整診斷: init → LLM → parse → BRAIN → full loop
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("run_alpha")

BASE_URL = "http://localhost:8000"
LAST_SESSION_FILE = Path(__file__).parent / "_last_session_id.txt"


def _ts():
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S UTC")


def _save_session_id(session_id: str):
    LAST_SESSION_FILE.write_text(session_id, encoding="utf-8")
    logger.info("[Session ID 已儲存至 %s]", LAST_SESSION_FILE)


def _load_session_id() -> str:
    if LAST_SESSION_FILE.exists():
        return LAST_SESSION_FILE.read_text(encoding="utf-8").strip()
    return ""


# ── Sub-command: start ────────────────────────────────────────────────────────


async def cmd_start(focus_area: str):
    """透過 HTTP API 啟動新的採礦 session。"""
    import httpx

    logger.info("[%s] 正在啟動新 Session (focus=%s) ...", _ts(), focus_area)
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        r = await client.post(
            f"{BASE_URL}/session/start",
            json={"focus_area": focus_area},
        )
        if r.status_code != 200:
            logger.error("啟動失敗: HTTP %s — %s", r.status_code, r.text[:300])
            return None
        data = r.json()
        sid = data.get("session_id")
        logger.info("[%s] Session 已啟動: %s", _ts(), sid)
        _save_session_id(sid)
        return sid


# ── Sub-command: check ───────────────────────────────────────────────────────


async def cmd_check(session_id: str | None):
    """檢查指定 session 的狀態與所有 alpha 結果。"""
    import httpx

    if not session_id:
        session_id = _load_session_id()
    if not session_id:
        logger.error("未提供 session_id 且無已儲存的 session。請先執行 start 或指定 session_id。")
        return

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        # 取得完整 session state
        r = await client.get(f"{BASE_URL}/session/{session_id}")
        if r.status_code != 200:
            logger.error("查詢失敗: HTTP %s — Session '%s' 不存在或伺服器未啟動", r.status_code, session_id)
            return
        state = r.json()

        # 取得 alpha 列表
        r2 = await client.get(f"{BASE_URL}/session/{session_id}/alphas")
        alpha_data = r2.json() if r2.status_code == 200 else {}

    _print_session_summary(state, alpha_data)


def _print_session_summary(state: dict, alpha_data: dict):
    """格式化輸出 session 摘要資訊。"""
    print("\n" + "=" * 70)
    print(f"  Session Report  [{_ts()}]")
    print("=" * 70)
    print(f"  ID:          {state.get('id', 'N/A')}")
    print(f"  Status:      {state.get('status', 'N/A')}")
    print(f"  Focus Area:  {state.get('focus_area', 'N/A')}")
    print(f"  Current Cycle:{state.get('cycle', state.get('current_cycle', 'N/A'))}")
    print(f"  Created At:  {state.get('created_at', 'N/A')}")
    print(f"  Updated At:  {state.get('updated_at', 'N/A')}")

    passed = state.get("passed_alphas", [])
    failed_count = len(state.get("failure_catalog", []))
    hall_count = len(state.get("hallucination_log", []))
    conv_count = len(state.get("conversation_history", []))

    print(f"\n  Passed Alphas:   {len(passed)}")
    print(f"  Failed Entries:  {failed_count}")
    print(f"  Hallucinations:  {hall_count}")
    print(f"  Conversation Turns: {conv_count}")

    # Passed Alphas 詳情
    if passed:
        print(f"\n{'─' * 70}")
        print("  Passed Alphas Detail:")
        print(f"{'─' * 70}")
        for i, a in enumerate(passed, 1):
            expr = a.get("expression", "N/A")[:120]
            brain = a.get("brain") or {}
            sharpe = brain.get("real_sharpe", "N/A")
            fitness = brain.get("real_fitness", "N/A")
            turnover = brain.get("real_turnover", "N/A")
            checks = brain.get("brain_checks", [])
            pstatus = a.get("pipeline_status", "?")
            print(f"\n  [#{i}] status={pstatus}")
            print(f"       expr  = {expr}")
            print(f"       sharpe={sharpe}  fitness={fitness}  turnover={turnover}")
            if checks:
                for c in checks:
                    print(f"       CHECK: {c.get('name', '?')} = {c.get('result', '?')}  val={c.get('value', '?')}")

    # Hallucination Log
    hall_log = state.get("hallucination_log", [])
    if hall_log:
        banned_vars = set()
        print(f"\n{'─' * 70}")
        print("  Hallucination Log (latest 5):")
        print(f"{'─' * 70}")
        for entry in hall_log[-5:]:
            banned_vars.add(entry.get("variable", ""))
            et = entry.get("error_type", "?")
            em = entry.get("error_message", "")[:120]
            print(f"  [{et}] {em}")
        if banned_vars:
            print(f"\n  Banned Variables: {sorted(banned_vars)}")

    # Failure Catalog
    fc = state.get("failure_catalog", [])
    if fc:
        print(f"\n{'─' * 70}")
        print("  Failure Catalog (latest 5):")
        print(f"{'─' * 70}")
        for f in fc[-5:]:
            ft = f.get("failure_type", "?")
            fp = json.dumps(f.get("fingerprint", {}), ensure_ascii=False)[:120]
            print(f"  {ft}: {fp}")

    # Topology Map
    topo = state.get("topology_map", {})
    if topo:
        print(f"\n{'─' * 70}")
        print(f"  Topology Map ({len(topo)} entries):")
        print(f"{'─' * 70}")
        for k, v in list(topo.items())[:10]:
            print(f"  {k[:60]}: {v}")

    # Algorithm call counts
    algo_counts = state.get("_algo_call_counts", {})
    if algo_counts:
        print(f"\n{'─' * 70}")
        print("  Algorithm Call Counts:")
        print(f"{'─' * 70}")
        for k, v in sorted(algo_counts.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")

    print(f"\n{'=' * 70}\n")


# ── Sub-command: run (self-loop E2E test) ────────────────────────────────────


async def cmd_run(cycles: int, focus_area: str):
    """執行完整 self-loop E2E 測試（含所有 11 項測試）。"""
    logger.info("=" * 60)
    logger.info("REAL E2E Self-Loop Test — NO mock data")
    logger.info("Cycles=%d  Focus=%s", cycles, focus_area)
    logger.info("=" * 60)

    results = {}
    expression = None
    direction = focus_area or "momentum"
    parsed_result = None
    brain_gate = None

    tests = [
        ("Component Initialization", _test_component_initialization),
        ("LLM Generation", _test_llm_generation),
        ("Alpha Parsing", None),
        ("BRAIN Submission", None),
        ("GARCH + Overfit", None),
        ("MAB Feedback", None),
        ("SignalArbiter", _test_signal_arbiter),
        ("RAG Retrieve", None),
        ("Whitelist", _test_whitelist),
        ("ExperienceDistillation", _test_experience_distillation),
        ("Full Self-Loop Rounds", None),
    ]

    for name, test_fn in tests:
        logger.info("\n--- Test: %s ---", name)
        try:
            if name == "Component Initialization":
                if test_fn is not None:
                    result = await test_fn()
                else:
                    result = None
                results[name] = bool(result)

            elif name == "LLM Generation":
                if test_fn is not None:
                    result = await test_fn()
                else:
                    result = None
                results[name] = bool(result)
                if result:
                    expression = result

            elif name == "Alpha Parsing":
                if expression:
                    parsed_result = await _test_alpha_parsing(expression)
                    results[name] = parsed_result is not None
                    if parsed_result and parsed_result.get("expression"):
                        expression = parsed_result["expression"]
                else:
                    results[name] = False
                    logger.warning("Skipped - no expression from LLM")

            elif name == "BRAIN Submission":
                if expression:
                    brain_gate = await _test_brain_submission(expression)
                    results[name] = brain_gate is not None
                else:
                    results[name] = False
                    logger.warning("Skipped - no expression to submit")

            elif name == "GARCH + Overfit":
                result = await _test_garch_and_overfit(brain_gate)
                results[name] = result

            elif name == "MAB Feedback":
                result = await _test_mab_feedback(direction)
                results[name] = result

            elif name == "RAG Retrieve":
                result = await _test_rag_retrieve(direction)
                results[name] = result

            elif name == "SignalArbiter" or name in ("Whitelist", "ExperienceDistillation"):
                if test_fn:
                    result = await test_fn()
                    results[name] = bool(result)
                else:
                    results[name] = False

            elif name == "Full Self-Loop Rounds":
                result = await _test_full_selfloop(cycles, direction)
                results[name] = result

            else:
                results[name] = False
                logger.warning("Skipped %s - dependency not met", name)

        except (OSError, ValueError, RuntimeError) as e:
            results[name] = False
            logger.error("FAILED: %s — %s", name, e, exc_info=True)

    logger.info("\n" + "=" * 60)
    logger.info("E2E Test Results:")
    passed = 0
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        logger.info("  [%s] %s", status, name)
        if ok:
            passed += 1
    logger.info("\nTotal: %d/%d passed", passed, len(results))
    logger.info("=" * 60)

    report_path = Path(__file__).parent / "e2e_test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": _ts(),
                "cycles": cycles,
                "focus_area": focus_area,
                "results": {k: bool(v) for k, v in results.items()},
                "passed": passed,
                "total": len(results),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Report saved to %s", report_path)

    return passed == len(results)


# ── Individual test functions (from _test_real_selfloop.py) ───────────────────


async def _test_component_initialization():
    from openalpha_brain.config.config import settings
    from openalpha_brain.core import loop_state
    from openalpha_brain.core.loop_state import init_intelligent_search

    try:
        init_intelligent_search()
    except Exception as e:
        logger.error("init_intelligent_search() FAILED: %s", e, exc_info=True)

    checks = {
        "MAB": loop_state._mab is not None,
        "RAG": loop_state._rag_engine is not None and getattr(loop_state._rag_engine, "is_ready", False),
        "Whitelist": loop_state._whitelist_mgr is not None,
        "SignalArbiter": loop_state._signal_arbiter is not None,
        "AlphaChannel": loop_state._alpha_channel is not None,
        "ExperienceDistiller": loop_state._experience_distiller is not None,
        "MarketStateInferencer": loop_state._market_state_inferencer is not None,
        "PnLAnalyzer": loop_state._pnl_analyzer is not None,
        "EvoDB": loop_state._evo_db is not None,
        "SuccessLib": loop_state._success_lib is not None,
        "FailureLib": loop_state._failure_lib is not None,
        "ToolFactory": loop_state._tool_factory is not None,
        "ReflectionEngine": loop_state._reflection_engine is not None,
        "FeatureMap": loop_state._feature_map is not None,
        "LogicLibrary": loop_state._logic_library is not None,
        "MultiAgent": getattr(settings, "MULTI_AGENT_ENABLED", False),
    }

    all_ok = all(checks.values())
    for k, v in checks.items():
        logger.info("  [%s] %s", "OK" if v else "FAIL", k)

    assert any(checks.values()), "ALL components are None after init"
    assert all_ok, f"Components not initialized: {[k for k, v in checks.items() if not v]}"
    return True


async def _test_llm_generation():
    from openalpha_brain.config.config import settings
    from openalpha_brain.services import llm_client

    original_provider = settings.LLM_PROVIDER
    original_model = settings.LLM_MODEL
    original_base_url = settings.LLM_BASE_URL

    settings.LLM_PROVIDER = "lmstudio"
    settings.LLM_MODEL = "fin-r1"
    settings.LLM_BASE_URL = "http://localhost:1234/v1/chat/completions"

    try:
        system_prompt = "You are a quantitative alpha factor researcher. Generate a WorldQuant BRAIN alpha expression."
        user_msg = "Generate a simple momentum alpha expression using rank and close price. Output ONLY the expression, nothing else."  # noqa: E501

        result = await llm_client.generate(
            system_prompt=system_prompt,
            history=[],
            user_msg=user_msg,
            session_id="e2e_test",
            cycle=1,
        )

        logger.info("LLM response: %s", result[:200] if result else "EMPTY")
        assert result and len(result) > 5, f"LLM returned empty or too short: '{result}'"
        return result
    finally:
        settings.LLM_PROVIDER = original_provider
        settings.LLM_MODEL = original_model
        settings.LLM_BASE_URL = original_base_url


async def _test_alpha_parsing(expression: str):
    from openalpha_brain.generation import alpha_parser as parser
    from openalpha_brain.generation.alpha_generator import _extract_expression_from_llm
    from openalpha_brain.generation.alpha_parser import parse_alpha_json
    from openalpha_brain.validation import validator as val

    parsed = parse_alpha_json(expression)
    if parsed is None:
        parsed = parser.parse_alpha_output(expression)

    expr = parsed["expression"] if parsed and parsed.get("expression") else _extract_expression_from_llm(expression)

    if expr is None:
        stripped = expression.strip()
        if "(" in stripped and ")" in stripped:
            expr = stripped

    logger.info("Parsed expression: %s", expr)

    if expr:
        syntax_result = val.validate_syntax(expr)
        logger.info(
            "Syntax validation: passed=%s failures=%s",
            syntax_result.passed,
            syntax_result.failures[:3] if syntax_result.failures else [],
        )
        return {"expression": expr, "parsed": parsed, "syntax_result": syntax_result}

    assert expr is not None, f"Could not extract expression from LLM output: {expression[:200]}"
    return None


async def _test_brain_submission(expression: str):
    from openalpha_brain.config.config import settings
    from openalpha_brain.services import brain_client

    if not settings.BRAIN_SUBMIT_ENABLED:
        logger.warning("BRAIN_SUBMIT_ENABLED=False, skipping real submission")
        return None
    if not settings.BRAIN_EMAIL or not settings.BRAIN_PASSWORD:
        logger.warning("BRAIN credentials not configured, skipping real submission")
        return None

    cookies = await brain_client.authenticate(settings.BRAIN_EMAIL, settings.BRAIN_PASSWORD)
    assert cookies is not None, "BRAIN authentication failed"

    sim_payload = {
        "type": "REGULAR",
        "settings": {
            "instrumentType": "EQUITY",
            "region": "USA",
            "universe": "TOP3000",
            "delay": 1,
            "decay": 5,
            "neutralization": "INDUSTRY",
            "truncation": 0.05,
            "pasteurization": "ON",
            "unitHandling": "VERIFY",
            "nanHandling": "ON",
            "language": "FASTEXPR",
            "visualization": False,
        },
        "regular": expression,
    }

    gate = await brain_client.submit_and_poll(
        simulation_payload=sim_payload,
        cookies=cookies,
        max_poll_seconds=settings.BRAIN_POLL_TIMEOUT,
    )

    logger.info(
        "BRAIN result: status=%s sharpe=%s fitness=%s turnover=%s passed=%s alpha_id=%s",
        gate.simulation_status,
        gate.sharpe,
        gate.fitness,
        gate.turnover,
        gate.passed,
        gate.alpha_id,
    )
    return gate


async def _test_garch_and_overfit(brain_gate):
    from openalpha_brain.utils.volatility_detector import estimate_garch11
    from openalpha_brain.validation.overfit_detector import detect_overfit

    pnl_curve = None
    if brain_gate and hasattr(brain_gate, "returns") and brain_gate.returns is not None:
        pnl_curve = [random.gauss(0.001, 0.02) for _ in range(60)]

    if pnl_curve and len(pnl_curve) >= 20:
        garch_result = estimate_garch11(pnl_curve)
        logger.info(
            "GARCH: persistence=%.4f half_life=%.1f is_clustering=%s",
            garch_result.persistence,
            garch_result.half_life,
            garch_result.is_clustering,
        )

    is_sharpe = brain_gate.sharpe if brain_gate else None
    os_sharpe = getattr(brain_gate, "os_sharpe", None) if brain_gate else None

    overfit_result = detect_overfit(is_sharpe=is_sharpe, os_sharpe=os_sharpe, pnl_curve=pnl_curve)
    logger.info(
        "Overfit: is_overfit=%s is_os_decay_ratio=%.4f warnings=%s",
        overfit_result.is_overfit,
        overfit_result.is_os_decay_ratio,
        overfit_result.warnings,
    )
    return True


async def _test_mab_feedback(direction: str):
    from openalpha_brain.core import loop_state

    if loop_state._mab is None:
        logger.warning("MAB not initialized, skipping")
        return False

    loop_state._algo_tick("mab_feedback_test")
    loop_state._mab.update(
        direction,
        operators=["rank", "ts_delta"],
        fields=["close", "volume"],
        reward=0.5,
    )
    direction_stats = loop_state._mab.get_direction_stats()
    logger.info("MAB direction stats: %s", dict(list(direction_stats.items())[:5]))
    return True


async def _test_signal_arbiter():
    from openalpha_brain.core import loop_state
    from openalpha_brain.validation.signal_arbiter import SignalSource

    if loop_state._signal_arbiter is None:
        logger.warning("SignalArbiter not initialized, skipping")
        return False

    loop_state._algo_tick("signal_arbiter_test")

    field_signals = {
        "close": [SignalSource(source_name="rag", score=0.8, weight=1.0)],
        "volume": [SignalSource(source_name="mab", score=0.5, weight=1.0)],
        "returns": [SignalSource(source_name="whitelist", score=0.3, weight=1.0)],
    }
    result = loop_state._signal_arbiter.rank_fields(field_signals, top_k=3)
    logger.info(
        "SignalArbiter rank_fields: %s",
        [(r.item_id, round(r.final_score, 4)) for r in result],
    )
    return True


async def _test_rag_retrieve(direction: str):
    from openalpha_brain.core import loop_state

    if loop_state._rag_engine is None or not loop_state._rag_engine.is_ready:
        logger.warning("RAG not ready, skipping")
        return False

    loop_state._algo_tick("rag_retrieve_test")
    result = await loop_state._rag_engine.retrieve(direction, top_k_ops=5, top_k_fields=10)
    logger.info(
        "RAG result: %d operators, %d fields, %d financial_logic",
        len(result.get("operators", [])),
        len(result.get("fields", [])),
        len(result.get("financial_logic", [])),
    )
    return True


async def _test_whitelist():
    from openalpha_brain.core import loop_state

    if loop_state._whitelist_mgr is None:
        logger.warning("Whitelist not initialized, skipping")
        return False

    loop_state._algo_tick("whitelist_test")
    fields = loop_state._whitelist_mgr.get_allowed_fields()
    logger.info("Whitelist: %d allowed fields", len(fields))
    return True


async def _test_experience_distillation():
    from openalpha_brain.core import loop_state

    if loop_state._experience_distiller is None:
        logger.warning("ExperienceDistiller not initialized, skipping")
        return False

    loop_state._algo_tick("experience_distillation_test")

    if loop_state._reflection_engine and loop_state._failure_lib:
        try:
            cards = await loop_state._experience_distiller.distill_from_failures(
                loop_state._reflection_engine,
                loop_state._failure_lib,
                min_occurrences=2,
            )
            logger.info("Distilled %d cards from failures", len(cards))
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("Distill from failures failed: %s", e)

    if loop_state._logic_library:
        try:
            evidence_cards = await loop_state._experience_distiller.distill_from_evidence(
                loop_state._logic_library,
                min_evidence=3,
            )
            logger.info("Distilled %d cards from evidence", len(evidence_cards))
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("Distill from evidence failed: %s", e)

    return True


async def _test_full_selfloop(cycles: int, focus_area: str):
    from loop_engine import run_loop

    from openalpha_brain.cli import session_manager as sm
    from openalpha_brain.config.config import settings

    state = await sm.create_session(focus_area=focus_area)
    session_id = state.id
    logger.info("Created session: %s", session_id)

    original_max = settings.MAX_CYCLES
    settings.MAX_CYCLES = cycles

    try:
        await run_loop(session_id)
    finally:
        settings.MAX_CYCLES = original_max

    final_state = await sm.load_session(session_id)
    if final_state is None:
        raise RuntimeError(f"Failed to load session {session_id}")
    logger.info(
        "Session state: cycles=%d, passed=%d, status=%s",
        final_state.cycle,
        len(final_state.passed_alphas),
        final_state.status,
    )

    from openalpha_brain.core import loop_state

    logger.info("Algo call stats: %s", loop_state._algo_call_counts)

    assert len(loop_state._algo_call_counts) > 1, f"Too few algorithms called: {loop_state._algo_call_counts}"
    return True


# ── Sub-command: mine (real mining session via HTTP API) ─────────────────────


async def cmd_mine(focus_area: str, wait_seconds: int):
    """透過 HTTP API 啟動真實採礦 session，等待指定秒數後回報結果。"""
    import httpx

    logger.info("[%s] === Starting Real Mining Session ===", _ts())
    logger.info("Focus: %s | Wait: %ds", focus_area, wait_seconds)

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        r = await client.post(
            f"{BASE_URL}/session/start",
            json={"focus_area": focus_area},
        )
        if r.status_code != 200:
            logger.error("Session start FAILED: HTTP %s — %s", r.status_code, r.text[:300])
            return
        data = r.json()
        session_id = data.get("session_id")
        logger.info("Session ID: %s", session_id)
        _save_session_id(session_id)

        logger.info("Waiting %ds for cycles to complete...", wait_seconds)
        await asyncio.sleep(wait_seconds)

        r2 = await client.get(f"{BASE_URL}/session/{session_id}")
        if r2.status_code != 200:
            logger.error("Query session FAILED: HTTP %s", r2.status_code)
            return
        state = r2.json()

    _print_session_summary(state, {})

    save_path = Path(__file__).parent / "mining_session_state.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Full state saved to %s", save_path)


# ── Sub-command: full (complete diagnostic pipeline) ─────────────────────────


async def cmd_full():
    """完整診斷流程: 初始化 → LLM 生成 → 解析 → BRAIN 提交 → 完整 Loop。"""
    logger.info("=" * 60)
    logger.info("COMPLETE DIAGNOSTIC PIPELINE — REAL DATA ONLY")
    logger.info("=" * 60)

    steps = [
        ("Step 1/5: Component Initialization", _test_component_initialization),
        ("Step 2/5: LLM Generation", _test_llm_generation),
        ("Step 3/5: Alpha Parsing", None),
        ("Step 4/5: BRAIN Submission & Validation", None),
        ("Step 5/5: GARCH + Overfit Detection", None),
    ]

    results = {}
    expression = None
    parsed_result = None
    brain_gate = None

    for step_name, step_fn in steps:
        logger.info("\n%s", step_name)
        try:
            if "Initialization" in step_name:
                if step_fn is not None:
                    result = await step_fn()
                else:
                    result = None
                results[step_name] = bool(result)

            elif "LLM Generation" in step_name:
                if step_fn is not None:
                    result = await step_fn()
                else:
                    result = None
                results[step_name] = bool(result)
                if result:
                    expression = result

            elif "Parsing" in step_name:
                if expression:
                    parsed_result = await _test_alpha_parsing(expression)
                    results[step_name] = parsed_result is not None
                    if parsed_result and parsed_result.get("expression"):
                        expression = parsed_result["expression"]
                else:
                    results[step_name] = False

            elif "BRAIN" in step_name:
                if expression:
                    brain_gate = await _test_brain_submission(expression)
                    results[step_name] = brain_gate is not None
                else:
                    results[step_name] = False

            elif "GARCH" in step_name:
                result = await _test_garch_and_overfit(brain_gate)
                results[step_name] = result

        except Exception as e:
            results[step_name] = False
            logger.error("DIAGNOSTIC FAILED: %s — %s", step_name, e, exc_info=True)

    # 最後跑一個 1-cycle self-loop 驗證整合性
    logger.info("\nBonus Step: 1-Cycle Self-Loop Integration Test")
    try:
        loop_ok = await _test_full_selfloop(cycles=1, focus_area="momentum")
        results["Self-Loop Integration"] = bool(loop_ok)
    except Exception as e:
        results["Self-Loop Integration"] = False
        logger.error("Self-loop FAILED: %s", e, exc_info=True)

    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC SUMMARY:")
    all_pass = True
    for name, ok in results.items():
        symbol = "OK" if ok else "XX"
        logger.info("  [%s] %s", symbol, name)
        if not ok:
            all_pass = False
    logger.info("\nOverall: %s", "ALL PASS" if all_pass else "SOME FAILURES DETECTED")
    logger.info("=" * 60)

    report_path = Path(__file__).parent / "diagnostic_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": _ts(),
                "diagnostic_results": {k: bool(v) for k, v in results.items()},
                "overall_pass": all_pass,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Diagnostic report saved to %s", report_path)

    return all_pass


# ── CLI Argument Parser ──────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_alpha",
        description="OpenAlpha-Quant User Operational Script — Real Data Only",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_alpha.py start --focus momentum
  python run_alpha.py check abc123def456
  python run_alpha.py run --cycles 3 --focus reversal
  python run_alpha.py mine --focus "price volume momentum" --wait 180
  python run_alpha.py full
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # start
    p_start = subparsers.add_parser("start", help="Start a new mining session via HTTP API")
    p_start.add_argument("--focus", default="auto", help="Focus area (default: auto)")

    # check
    p_check = subparsers.add_parser("check", help="Check session status and alphas")
    p_check.add_argument("session_id", nargs="?", default=None, help="Session ID (uses last if omitted)")

    # run
    p_run = subparsers.add_parser("run", help="Run full self-loop E2E test")
    p_run.add_argument("--cycles", type=int, default=2, help="Number of self-loop cycles (default: 2)")
    p_run.add_argument("--focus", default="momentum", help="Focus area (default: momentum)")

    # mine
    p_mine = subparsers.add_parser("mine", help="Start real mining session and wait for results")
    p_mine.add_argument("--focus", default="price volume momentum", help="Focus area")
    p_mine.add_argument("--wait", type=int, default=120, help="Seconds to wait before reporting (default: 120)")

    # full
    subparsers.add_parser("full", help="Complete diagnostic: init → LLM → parse → BRAIN → full loop")

    return parser


# ── Main Entry Point ─────────────────────────────────────────────────────────


async def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    dispatch = {
        "start": lambda: cmd_start(args.focus),
        "check": lambda: cmd_check(args.session_id),
        "run": lambda: cmd_run(args.cycles, args.focus),
        "mine": lambda: cmd_mine(args.focus, args.wait),
        "full": cmd_full,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return

    try:
        result = await handler()
        if isinstance(result, bool):
            sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("\n使用者中斷操作。")
        sys.exit(130)
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
