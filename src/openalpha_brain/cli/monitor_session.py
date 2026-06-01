#!/usr/bin/env python3
"""
OpenAlpha - Quant — AI Full-Chain Monitoring Script
合併自: _e2e_test.py (HTTP API polling-based monitoring)

透過 FastAPI HTTP API 對正在運行的 Session 進行全鏈路監控。
每 10 秒輪詢一次，輸出完整的演算法鏈追蹤資訊，適合 AI 分析使用。

Usage:
    python monitor_session.py [session_id]              # 監控指定 session
    python monitor_session.py [session_id] --verbose     # 詳細 debug 模式
    python monitor_session.py [session_id] --output report.json  # 自定義輸出路徑

Features:
    - Real-time session state polling (10s interval)
    - Full algorithm chain tracing (MAB / RAG / Whitelist / FeatureMap / etc.)
    - Color-coded terminal output (PASS=green, FAIL=red, PENDING=yellow)
    - Progress spinner during active monitoring
    - Final summary with best alpha, failure patterns, performance timeline
    - Complete JSON report export for AI analysis
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

try:
    import httpx
except ImportError:
    print("ERROR: 需要安裝 httpx — pip install httpx")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("monitor")

BASE_URL = "http://localhost:8000"
POLL_INTERVAL = 10
MAX_POLLS = 600

TERMINAL_STATUS = {
    "COMPLETED",
    "ERROR",
    "FAIL",
    "STOPPED",
    "CRASHED",
}

# ── Terminal Color Helpers ───────────────────────────────────────────────────


class _C:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def _ok(msg: str) -> str:
    return f"{_C.GREEN}{msg}{_C.RESET}"


def _fail(msg: str) -> str:
    return f"{_C.RED}{msg}{_C.RESET}"


def _warn(msg: str) -> str:
    return f"{_C.YELLOW}{msg}{_C.RESET}"


def _info(msg: str) -> str:
    return f"{_C.CYAN}{msg}{_C.RESET}"


def _bold(msg: str) -> str:
    return f"{_C.BOLD}{msg}{_C.RESET}"


def _dim(msg: str) -> str:
    return f"{_C.DIM}{msg}{_C.RESET}"


# ── Spinner ──────────────────────────────────────────────────────────────────

_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class Spinner:
    def __init__(self):
        self._idx = 0
        self._running = False

    def tick(self):
        if self._running:
            frame = _SPINNER_FRAMES[self._idx % len(_SPINNER_FRAMES)]
            self._idx += 1
            print(f"\r  {frame} Monitoring...", end="", flush=True)

    def start(self):
        self._running = True
        self._idx = 0

    def stop(self):
        self._running = False
        print("\r" + " " * 30 + "\r", end="")


# ── Data Collection & Formatting ─────────────────────────────────────────────


def _ts() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S UTC")


async def fetch_session(client: httpx.AsyncClient, session_id: str) -> dict | None:
    try:
        r = await client.get(f"{BASE_URL}/session/{session_id}", timeout=15.0)
        if r.status_code == 200:
            return r.json()
        logger.warning("fetch_session HTTP %s for %s", r.status_code, session_id)
        return None
    except (ConnectionError, OSError, TimeoutError) as e:
        logger.error("fetch_session error: %s", e)
        return None


async def fetch_alphas(client: httpx.AsyncClient, session_id: str) -> dict | None:
    try:
        r = await client.get(f"{BASE_URL}/session/{session_id}/alphas", timeout=15.0)
        if r.status_code == 200:
            return r.json()
        return None
    except (ConnectionError, OSError, TimeoutError) as e:
        logger.error("fetch_alphas error: %s", e)
        return None


def _color_status(status: str) -> str:
    s = str(status).upper()
    if s in ("PASS", "COMPLETED", "VALIDATED"):
        return _ok(s)
    if s in ("FAIL", "ERROR", "ABANDONED", "CRASHED"):
        return _fail(s)
    return _warn(s)


def format_alpha_detail(alpha: dict, index: int) -> list[str]:
    """格式化單一 alpha 的完整資訊。"""
    lines = []
    expr = alpha.get("expression", "N/A")[:150]
    brain = alpha.get("brain") or {}
    sharpe = brain.get("real_sharpe")
    fitness = brain.get("real_fitness")
    turnover = brain.get("real_turnover")
    checks = brain.get("brain_checks", [])
    pstatus = alpha.get("pipeline_status", "?")

    sharpe_str = f"{sharpe:.4f}" if sharpe is not None else "N/A"
    fitness_str = f"{fitness:.4f}" if fitness is not None else "N/A"
    turnover_str = f"{turnover:.2f}" if turnover is not None else "N/A"

    lines.append(f"  #{index} {_color_status(pstatus)} | Sharpe={sharpe_str} Fitness={fitness_str} Turn={turnover_str}")
    lines.append(f"      expr: {expr}")

    if checks:
        for c in checks[:6]:
            cname = c.get("name", "?")
            cresult = c.get("result", "?")
            cval = c.get("value", "?")
            lines.append(f"      CHECK: {cname}={_color_status(str(cresult))} val={cval}")

    return lines


def format_poll_snapshot(state: dict, poll_num: int, verbose: bool) -> list[str]:
    """格式化每次輪詢的完整快照。"""
    lines = []
    status = state.get("status", "?")
    cycle = state.get("cycle", state.get("current_cycle", 0))
    n_passed = len(state.get("passed_alphas", []))
    n_failed = len(state.get("failure_catalog", []))
    hall = state.get("hallucination_log", [])
    conv = state.get("conversation_history", [])

    header = (
        f"\n{_bold('─' * 72)}\n"
        f"  POLL #{poll_num} [{_ts()}]  "
        f"Status={_color_status(status)}  "
        f"Cycle={cycle}  "
        f"Alphas={_info(str(n_passed))}  "
        f"Fails={_fail(str(n_failed))}  "
        f"Hall={len(hall)}  Conv={len(conv)}\n"
        f"{_bold('─' * 72)}"
    )
    lines.append(header)

    # Passed Alphas (latest 3)
    alphas = state.get("passed_alphas", [])
    if alphas:
        lines.append(f"\n  {_bold('Passed Alphas (latest 3):')}")
        for i, a in enumerate(alphas[-3:], start=max(1, len(alphas) - 2)):
            lines.extend(format_alpha_detail(a, i))

    # Hallucination Log (latest 5)
    if hall:
        lines.append(f"\n  {_bold('Hallucination Log (latest 5):')}")
        for h in hall[-5:]:
            et = h.get("error_type", "?")
            em = h.get("error_message", "?")[:120]
            var = h.get("variable", "")
            lines.append(f"    [{_fail(et)}] {em}" + (f"  var={var}" if var else ""))

    # Failure Catalog (latest 5)
    fc = state.get("failure_catalog", [])
    if fc:
        lines.append(f"\n  {_bold('Failure Catalog (latest 5):')}")
        for f in fc[-5:]:
            ft = f.get("failure_type", "?")
            fp = json.dumps(f.get("fingerprint", {}), ensure_ascii=False)[:120]
            lines.append(f"    {_fail(ft)}: {fp}")

    # Conversation History (latest 6)
    if conv:
        lines.append(f"\n  {_bold('Conversation History (latest 6 turns):')}")
        for turn in conv[-6:]:
            role = turn.get("role", "?")
            content = turn.get("content", "")[:200]
            role_tag = _info(role.upper()) if role == "user" else _dim(role.upper())
            lines.append(f"    [{role_tag}]: {content}")

    # Algorithm Call Counts
    algo_counts = state.get("_algo_call_counts", {})
    if algo_counts:
        lines.append(f"\n  {_bold('Algorithm Call Counts:')}")
        for k, v in sorted(algo_counts.items(), key=lambda x: -x[1])[:12]:
            lines.append(f"    {k}: {_info(str(v))}")

    # Verbose: full internal state
    if verbose:
        lines.extend(_format_verbose_state(state))

    return lines


def _format_verbose_state(state: dict) -> list[str]:
    """輸出額外的詳細內部狀態資訊。"""
    lines = []
    lines.append(f"\n  {_dim('── VERBOSE INTERNAL STATE ──')}")

    # MAB direction stats
    mab_stats = state.get("_mab_direction_stats")
    if mab_stats:
        lines.append(f"\n  {_bold('MAB Direction Stats:')}")
        if isinstance(mab_stats, dict):
            for k, v in list(mab_stats.items())[:8]:
                lines.append(f"    {k}: {v}")
        else:
            lines.append(f"    {mab_stats}")

    # Whitelist state
    wl = state.get("_whitelist_state")
    if wl:
        lines.append(f"\n  {_bold('Whitelist State:')}")
        if isinstance(wl, dict):
            n_allowed = wl.get("allowed_count", len(wl.get("allowed_fields", [])))
            n_banned = wl.get("banned_count", len(wl.get("banned_fields", [])))
            lines.append(f"    Allowed: {n_allowed} | Banned: {n_banned}")
        elif isinstance(wl, list):
            lines.append(f"    Allowed fields: {len(wl)}")

    # RAG retrieval stats
    rag_stats = state.get("_rag_retrieval_stats")
    if rag_stats:
        lines.append(f"\n  {_bold('RAG Retrieval Stats:')}")
        if isinstance(rag_stats, dict):
            for k, v in rag_stats.items():
                lines.append(f"    {k}: {v}")
        else:
            lines.append(f"    {rag_stats}")

    # Experience Distiller card count
    ed_cards = state.get("_experience_distiller_cards")
    if ed_cards is not None:
        count = len(ed_cards) if isinstance(ed_cards, list) else ed_cards
        lines.append(f"\n  {_bold('Experience Distiller Cards:')} {_info(str(count))}")

    # FeatureMap cell occupancy
    fmap = state.get("_feature_map_occupancy")
    if fmap:
        lines.append(f"\n  {_bold('FeatureMap Occupancy:')}")
        if isinstance(fmap, dict):
            total = fmap.get("total_cells", 0)
            occupied = fmap.get("occupied_cells", 0)
            pct = (occupied / total * 100) if total > 0 else 0
            lines.append(f"    Cells: {occupied}/{total} ({pct:.1f}% occupied)")
            regions = fmap.get("region_breakdown", {})
            if regions:
                for rk, rv in list(regions.items())[:6]:
                    lines.append(f"      {rk}: {rv}")
        else:
            lines.append(f"    {fmap}")

    # MarketState inference result
    ms = state.get("_market_state_inference")
    if ms:
        lines.append(f"\n  {_bold('MarketState Inference:')}")
        if isinstance(ms, dict):
            regime = ms.get("regime", "?")
            confidence = ms.get("confidence")
            vol_regime = ms.get("volatility_regime", "?")
            lines.append(f"    Regime={regime} Vol={vol_regime}" + (f" Conf={confidence}" if confidence else ""))
        else:
            lines.append(f"    {ms}")

    # Topology map summary
    topo = state.get("topology_map", {})
    if topo:
        lines.append(f"\n  {_bold('Topology Map:')} {len(topo)} entries")
        for k, v in list(topo.items())[:5]:
            lines.append(f"    {k[:50]}: {v}")

    lines.append(f"  {_dim('── END VERBOSE ──')}")
    return lines


# ── Final Summary Report ─────────────────────────────────────────────────────


def generate_final_report(
    session_id: str,
    timeline: list[dict],
    final_state: dict,
    final_alphas: dict | None,
) -> dict:
    """生成最終摘要報告（dict 格式，同時用於終端輸出和 JSON 匯出）。"""
    status = final_state.get("status", "?")
    total_cycles = final_state.get("cycle", final_state.get("current_cycle", 0))
    passed_alphas = final_state.get("passed_alphas", [])
    failed_catalog = final_state.get("failure_catalog", [])
    hall_log = final_state.get("hallucination_log", [])
    conv_history = final_state.get("conversation_history", [])
    algo_counts = final_state.get("_algo_call_counts", {})

    # Best alpha (highest sharpe)
    best_alpha = None
    best_sharpe = float("-inf")
    for a in passed_alphas:
        brain = a.get("brain") or {}
        s = brain.get("real_sharpe")
        if s is not None and s > best_sharpe:
            best_sharpe = s
            best_alpha = a

    # Top failure patterns
    failure_patterns: dict[str, int] = {}
    for f in failed_catalog:
        ft = f.get("failure_type", "unknown")
        failure_patterns[ft] = failure_patterns.get(ft, 0) + 1
    top_failures = sorted(failure_patterns.items(), key=lambda x: -x[1])[:10]

    # Performance timeline (cycle-by-cycle snapshot from polls)
    perf_timeline = []
    for snap in timeline:
        perf_timeline.append(
            {
                "poll": snap.get("poll_num"),
                "timestamp": snap.get("timestamp"),
                "status": snap.get("status"),
                "cycle": snap.get("cycle"),
                "n_passed": snap.get("n_passed"),
                "n_failed": snap.get("n_failed"),
                "n_hall": snap.get("n_hall"),
            }
        )

    report = {
        "session_id": session_id,
        "report_generated_at": _ts(),
        "final_status": status,
        "total_cycles_run": total_cycles,
        "total_alphas_generated": len(passed_alphas) + len(failed_catalog),
        "total_alphas_passed": len(passed_alphas),
        "total_alphas_failed": len(failed_catalog),
        "total_hallucinations": len(hall_log),
        "total_conversation_turns": len(conv_history),
        "best_alpha": {
            "expression": best_alpha.get("expression", "") if best_alpha else None,
            "sharpe": best_sharpe if best_sharpe > float("-inf") else None,
            "fitness": best_alpha.get("brain", {}).get("real_fitness") if best_alpha else None,
            "turnover": best_alpha.get("brain", {}).get("real_turnover") if best_alpha else None,
            "pipeline_status": best_alpha.get("pipeline_status") if best_alpha else None,
        }
        if best_alpha
        else None,
        "top_failure_patterns": [{"type": t, "count": c} for t, c in top_failures],
        "algorithm_call_counts": algo_counts,
        "performance_timeline": perf_timeline,
        "final_state_snapshot": {
            "focus_area": final_state.get("focus_area"),
            "created_at": final_state.get("created_at"),
            "updated_at": final_state.get("updated_at"),
            "error_message": final_state.get("error_message"),
        },
    }

    return report


def print_final_summary(report: dict):
    """在終端機列印最終摘要報告。"""
    print(f"\n{'=' * 76}")
    print(f"  {_bold('FINAL SESSION REPORT')}  [{report['report_generated_at']}]")
    print(f"{'=' * 76}")
    print(f"  Session ID:       {_info(report['session_id'])}")
    print(f"  Final Status:     {_color_status(report['final_status'])}")
    print(f"  Total Cycles:     {report['total_cycles_run']}")
    print(f"  Alphas Generated: {report['total_alphas_generated']}")
    print(f"  Alphas Passed:    {_ok(str(report['total_alphas_passed']))}")
    print(f"  Alphas Failed:    {_fail(str(report['total_alphas_failed']))}")
    print(f"  Hallucinations:   {_warn(str(report['total_hallucinations']))}")
    print(f"  Conversation Turns:{report['total_conversation_turns']}")

    # Best Alpha
    ba = report.get("best_alpha")
    if ba and ba.get("expression"):
        print(f"\n  {_bold('Best Alpha (highest Sharpe):')}")
        print(f"    Expression : {ba['expression'][:200]}")
        print(f"    Sharpe     : {_ok(str(ba['sharpe']))}" if ba.get("sharpe") is not None else "    Sharpe     : N/A")
        print(f"    Fitness    : {ba.get('fitness', 'N/A')}")
        print(f"    Turnover   : {ba.get('turnover', 'N/A')}")
        print(f"    Status     : {_color_status(str(ba.get('pipeline_status', '?')))}")

    # Top Failure Patterns
    patterns = report.get("top_failure_patterns", [])
    if patterns:
        print(f"\n  {_bold('Top Failure Patterns:')}")
        for p in patterns:
            print(f"    {_fail(p['type'])}: {p['count']} occurrences")

    # Algorithm Call Counts
    algo = report.get("algorithm_call_counts", {})
    if algo:
        print(f"\n  {_bold('Algorithm Call Counts:')}")
        for k, v in sorted(algo.items(), key=lambda x: -x[1])[:15]:
            print(f"    {k}: {_info(str(v))}")

    # Performance Timeline
    tl = report.get("performance_timeline", [])
    if tl:
        print(f"\n  {_bold('Performance Timeline (cycle-by-cycle):')}")
        print(f"    {'Poll':>5} {'Timestamp':>22} {'Status':>12} {'Cycle':>6} {'Pass':>5} {'Fail':>5} {'Hall':>5}")
        print(f"    {'-' * 66}")
        for entry in tl:
            print(
                f"    {entry['poll']:>5} {entry['timestamp']:>22} "
                f"{entry['status']:>12} {entry['cycle']:>6} "
                f"{entry['n_passed']:>5} {entry['n_failed']:>5} {entry['n_hall']:>5}",
            )

    print(f"\n{'=' * 76}\n")


# ── Core Monitoring Loop ─────────────────────────────────────────────────────


async def run_monitor(
    session_id: str,
    verbose: bool = False,
    output_path: str | None = None,
):
    """
    主監控迴圈：每 10 秒輪詢 session 狀態，直到 terminal status。

    Args:
        session_id: 要監控的 session ID
        verbose:   是否輸出詳細內部狀態
        output_path: 報告 JSON 匯出路徑 (預設: session_{id}_report.json)
    """
    spinner = Spinner()
    spinner.start()

    timeline: list[dict] = []
    final_state: dict | None = None
    final_alphas: dict | None = None

    print(f"\n{_bold('OpenAlpha-Quant AI Monitor — Full-Chain Session Tracing')}")
    print(f"  Session: {_info(session_id)}")
    print(f"  Server:  {BASE_URL}")
    print(f"  Polling: every {POLL_INTERVAL}s | Max polls: {MAX_POLLS}")
    print(f"  Verbose: {verbose}")
    print()

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        for poll_num in range(1, MAX_POLLS + 1):
            spinner.tick()
            await asyncio.sleep(POLL_INTERVAL)

            state = await fetch_session(client, session_id)
            if state is None:
                continue

            status = state.get("status", "?")
            cycle = state.get("cycle", state.get("current_cycle", 0))
            n_passed = len(state.get("passed_alphas", []))
            n_failed = len(state.get("failure_catalog", []))
            hall = state.get("hallucination_log", [])

            # Record timeline snapshot
            timeline.append(
                {
                    "poll_num": poll_num,
                    "timestamp": _ts(),
                    "status": status,
                    "cycle": cycle,
                    "n_passed": n_passed,
                    "n_failed": n_failed,
                    "n_hall": len(hall),
                }
            )

            # Print detailed snapshot
            spinner.stop()
            lines = format_poll_snapshot(state, poll_num, verbose)
            for line in lines:
                print(line)
            spinner.start()

            # Check terminal status
            if status in TERMINAL_STATUS:
                spinner.stop()
                final_state = state
                break

        # Fetch final alpha details
        if final_state:
            final_alphas = await fetch_alphas(client, session_id)

    if final_state is None:
        spinner.stop()
        print(f"\n{_fail('Monitoring ended without reaching terminal status.')}")
        print(f"  Total polls: {len(timeline)}")
        # Still generate partial report
        final_state = {
            "status": "TIMEOUT",
            "cycle": 0,
            "passed_alphas": [],
            "failure_catalog": [],
            "_algo_call_counts": {},
        }

    # Generate & print final report
    report = generate_final_report(session_id, timeline, final_state, final_alphas)
    print_final_summary(report)

    # Save to JSON
    out = output_path or f"session_{session_id}_report.json"
    out_path = Path(out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"{_info('Report saved to:')} {out_path.resolve()}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="monitor_session",
        description="OpenAlpha-Quant AI Full-Chain Session Monitor via HTTP API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python monitor_session.py abc123def456
  python monitor_session.py abc123def456 --verbose
  python monitor_session.py abc123def456 --output my_report.json

The script connects to FastAPI server at {BASE_URL} and polls
the session state every {POLL_INTERVAL}s until it reaches a terminal
state (COMPLETED/ERROR/FAIL/STOPPED/CRASHED).
        """,
    )

    parser.add_argument(
        "session_id",
        nargs="?",
        default=None,
        help="Session ID to monitor (required)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Show extra debug info (MAB/RAG/Whitelist/FeatureMap/MarketState internals)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Custom JSON output path (default: session_<id>_report.json)",
    )
    return parser


async def main():
    parser = build_parser()
    args = parser.parse_args()

    session_id = args.session_id
    if not session_id:
        last_file = Path(__file__).parent / "_last_session_id.txt"
        if last_file.exists():
            session_id = last_file.read_text(encoding="utf-8").strip()
            print(f"Using session ID from _last_session_id.txt: {_info(session_id)}")
        else:
            parser.print_help()
            print("\nError: session_id is required, or run _start_session.py first.")
            sys.exit(1)

    try:
        await run_monitor(
            session_id=session_id,
            verbose=args.verbose,
            output_path=args.output,
        )
    except KeyboardInterrupt:
        print(f"\n\n{_warn('Monitoring interrupted by user.')}")
        sys.exit(130)
    except (OSError, ValueError, RuntimeError) as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
