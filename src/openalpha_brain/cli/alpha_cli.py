#!/usr/bin/env python3
"""
OpenAlpha-Brain Interactive CLI Controller — v3.0 Event-Driven
==============================================================
統一控制台 — 即時事件驅動、非阻塞、流水線可視化

Usage:
    openalpha run [--focus momentum] [--cycles 5] [--no-brain]
    openalpha status
    openalpha sessions
    openalpha interactive
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import queue
import threading
import time
import traceback
from datetime import datetime
from typing import Any

import aiohttp

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)-5s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("alpha_cli")

for _n in ["httpx", "httpcore", "asyncio"]:
    logging.getLogger(_n).setLevel(logging.CRITICAL)


# ── ANSI Colors ──────────────────────────────────────────────────────────────
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"


def _c(text: str, color: str = C.RESET) -> str:
    return f"{color}{text}{C.RESET}"


# ════════════════════════════════════════════════════════════════════════════
# AlphaCLI — Event-Driven Controller
# ════════════════════════════════════════════════════════════════════════════
class AlphaCLI:
    def __init__(self):
        self.state: str = "IDLE"
        self.session_id: str | None = None
        self.current_cycle: int = 0
        self.max_cycles: int = 5
        self.focus_area: str = "momentum"
        self.brain_submit_enabled: bool = True
        self.auto_confirm: bool = False
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._pause_event.set()

        self._event_count: int = 0
        self._alpha_count: int = 0
        self._reject_count: int = 0
        self._brain_count: int = 0
        self._start_time: float = 0
        self._last_event_time: float = 0
        self._live_alphas: list[dict] = []
        self._quiet_mode: bool = False

        from openalpha_brain.core.events import (
            EVENT_ALPHA_GENERATED,
            EVENT_ALPHA_REJECTED,
            EVENT_ALPHA_VALIDATED,
            EVENT_BRAIN_RESULT,
            EVENT_BRAIN_SUBMIT,
            EVENT_CYCLE_START,
            EVENT_ERROR,
            EVENT_MAB_FEEDBACK,
            EVENT_MINING_COMPLETE,
            get_event_bus,
        )

        self._bus = get_event_bus()
        self.EVENT_CYCLE_START = EVENT_CYCLE_START
        self.EVENT_ALPHA_GENERATED = EVENT_ALPHA_GENERATED
        self.EVENT_ALPHA_VALIDATED = EVENT_ALPHA_VALIDATED
        self.EVENT_ALPHA_REJECTED = EVENT_ALPHA_REJECTED
        self.EVENT_BRAIN_SUBMIT = EVENT_BRAIN_SUBMIT
        self.EVENT_BRAIN_RESULT = EVENT_BRAIN_RESULT
        self.EVENT_MAB_FEEDBACK = EVENT_MAB_FEEDBACK
        self.EVENT_MINING_COMPLETE = EVENT_MINING_COMPLETE
        self.EVENT_ERROR = EVENT_ERROR

    # ── Banner & Status ───────────────────────────────────────────────────

    def _banner(self) -> None:
        print(f"\n{C.BOLD}{C.CYAN}{'╔' + '═' * 72 + '╗'}{C.RESET}")
        print(f"{C.CYAN}║{C.RESET}  {C.BOLD}OpenAlpha-Brain Controller v3.0{C.RESET}{' ' * 36}{C.CYAN}║{C.RESET}")
        print(
            f"{C.CYAN}║{C.RESET}  {C.DIM}Event-Driven │ Real-Time │ Non-Blocking{C.RESET}{' ' * 27}{C.CYAN}║{C.RESET}"
        )
        print(f"{C.CYAN}{'╚' + '═' * 72 + '╝'}{C.RESET}\n")

    def _status_bar(self) -> str:
        state_color = {
            "IDLE": C.DIM,
            "RUNNING": C.GREEN,
            "PAUSED": C.YELLOW,
            "STOPPING": C.RED,
            "COMPLETED": C.BLUE,
            "ERROR": C.RED,
        }.get(self.state, C.RESET)
        (self.session_id or "?")[:12]
        elapsed = time.time() - self._start_time if self._start_time else 0
        m, s = divmod(int(elapsed), 60)
        return (
            f"  {_c(self.state, state_color):<10s} "
            f"Cyc:{self.current_cycle}/{self.max_cycles} "
            f"α:{self._alpha_count} ✗:{self._reject_count} 🧠:{self._brain_count} "
            f"⏱{m:d}:{s:02d} "
            f"BRAIN:{'ON' if self.brain_submit_enabled else 'OFF'}"
        )

    # ── Event Handler (real-time display) ─────────────────────────────────

    def _on_event(self, event) -> None:
        from openalpha_brain.core.events import AlphaEvent

        if not isinstance(event, AlphaEvent):
            return
        self._event_count += 1
        self._last_event_time = time.time()

        from openalpha_brain.cli.event_adapters import (
            BrainResultAdapter,
            CycleStartAdapter,
            EventAdapterFactory,
            MiningCompleteAdapter,
        )
        from openalpha_brain.cli.cli_renderer import get_cli_renderer

        adapter = EventAdapterFactory.create(event)
        renderer = get_cli_renderer()

        output = renderer.render(adapter)
        if output:
            print(output)

        if isinstance(adapter, CycleStartAdapter):
            self.current_cycle = adapter.cycle

        elif adapter.event_type == self.EVENT_ALPHA_VALIDATED:
            self._alpha_count += 1

        elif adapter.event_type == self.EVENT_ALPHA_REJECTED:
            self._reject_count += 1

        elif adapter.event_type == self.EVENT_BRAIN_RESULT:
            self._brain_count += 1
            brain_adapter = adapter
            self._live_alphas.append(
                {
                    "alpha_id": brain_adapter.alpha_id,
                    "expression": brain_adapter.expression,
                    "status": brain_adapter.status,
                    "sharpe": brain_adapter.sharpe,
                    "fitness": brain_adapter.fitness,
                    "turnover": brain_adapter.turnover,
                    "drawdown": brain_adapter.drawdown,
                    "direction": brain_adapter.direction,
                    "timestamp": event.timestamp,
                }
            )

        elif isinstance(adapter, MiningCompleteAdapter):
            elapsed = time.time() - self._start_time if self._start_time else 0
            m, s = divmod(int(elapsed), 60)
            print(f"  Rejected : {self._reject_count}")
            print(f"  BRAIN    : {self._brain_count} results")
            print(f"  Elapsed  : {m:d}:{s:02d}")
            self.state = "COMPLETED"

    # ── System Health Check ─────────────────────────────────────────────

    async def check_health(self) -> dict[str, Any]:
        results: dict[str, Any] = {"timestamp": datetime.now().isoformat(), "checks": {}}
        checks = results["checks"]

        t0 = time.perf_counter()
        try:
            from openalpha_brain.services.llm_client import generate

            reply = await generate(system_prompt="Reply with exactly: OK", history=[], user_msg="ping")
            checks["llm"] = {
                "status": "OK",
                "response": reply.strip()[:50],
                "latency_ms": round((time.perf_counter() - t0) * 1000),
            }
        except (TimeoutError, aiohttp.ClientError, ConnectionError, OSError) as e:
            checks["llm"] = {"status": "FAIL", "error": str(e)[:80]}

        t0 = time.perf_counter()
        try:
            from openalpha_brain.services.llm_client import embed

            vec = await embed("test")
            checks["embedding"] = {
                "status": "OK",
                "dim": len(vec),
                "latency_ms": round((time.perf_counter() - t0) * 1000),
            }
        except (TimeoutError, aiohttp.ClientError, ConnectionError, OSError) as e:
            checks["embedding"] = {"status": "FAIL", "error": str(e)[:80]}

        modules_ok, modules_fail = [], []
        for m, display_name in [
            ("openalpha_brain.config", "config"),
            ("openalpha_brain.core.models", "models"),
            ("openalpha_brain.core.loop_engine", "loop_engine"),
            ("openalpha_brain.agents.multi_agent", "multi_agent"),
            ("openalpha_brain.validation.validator", "validator"),
            ("openalpha_brain.learning.mab", "mab"),
            ("openalpha_brain.utils.paper_edge_enhancements", "hypothesis_aligner"),
            ("openalpha_brain.evolution.semantic_mutator", "semantic_mutator"),
            ("openalpha_brain.services.brain_client", "brain_client"),
            ("openalpha_brain.services.brain_submitter", "brain_submitter"),
            ("openalpha_brain.cli.session_manager", "session_manager"),
            ("openalpha_brain.core.loop_state", "loop_state"),
            ("openalpha_brain.core.events", "events"),
            ("openalpha_brain.core.pipeline", "pipeline"),
        ]:
            try:
                __import__(m)
                modules_ok.append(display_name)
            except (ImportError, AttributeError, OSError) as ex:
                modules_fail.append((m, str(ex)))
        checks["modules"] = {
            "status": "OK" if not modules_fail else "PARTIAL",
            "ok_count": len(modules_ok),
            "fail_modules": [m for m, _ in modules_fail],
        }

        if self.brain_submit_enabled:
            t0 = time.perf_counter()
            try:
                from openalpha_brain.config.config import settings

                email, pwd = getattr(settings, "BRAIN_EMAIL", ""), getattr(settings, "BRAIN_PASSWORD", "")
                if email and pwd:
                    from openalpha_brain.services.brain_client import authenticate

                    cookies = await authenticate(email, pwd)
                    checks["brain_auth"] = {
                        "status": "OK",
                        "has_cookies": bool(cookies),
                        "latency_ms": round((time.perf_counter() - t0) * 1000),
                    }
                else:
                    checks["brain_auth"] = {"status": "SKIP", "reason": "No credentials"}
            except (TimeoutError, aiohttp.ClientError, ConnectionError, OSError, ValueError) as e:
                checks["brain_auth"] = {"status": "FAIL", "error": str(e)[:80]}
        else:
            checks["brain_auth"] = {"status": "DISABLED"}

        from openalpha_brain.core import loop_state as _ls

        comps = {}
        for name, obj in [
            ("MAB", "_mab"),
            ("RAG Engine", "_rag_engine"),
            ("StrategyClassifier", "_strategy_classifier"),
            ("SemanticMutator", "_semantic_mutator"),
            ("ExperienceDistiller", "_experience_distiller"),
            ("ReflectionEngine", "_reflection_engine"),
            ("ToolFactory", "_tool_factory"),
            ("MarketStateInferencer", "_market_state_inferencer"),
            ("GlobalKnowledge", "_global_knowledge"),
            ("SignalArbiter", "_signal_arbiter"),
            ("AlphaChannel", "_alpha_channel"),
        ]:
            comps[name] = "✅" if getattr(_ls, obj, None) is not None else "⛔"
        checks["components"] = comps
        return results

    def print_health(self, health: dict) -> None:
        ch = health["checks"]
        print(f"\n  {C.BOLD}{'── SYSTEM HEALTH ──':─^64}{C.RESET}\n")
        for name, info in ch.items():
            if isinstance(info, dict) and "status" in info:
                st = info["status"]
                icon = {
                    "OK": f"{C.GREEN}✅",
                    "FAIL": f"{C.RED}❌",
                    "SKIP": f"{C.YELLOW}⏭️",
                    "DISABLED": f"{C.DIM}○",
                    "PARTIAL": f"{C.YELLOW}⚠️",
                }.get(st, "?")
                print(f"  {icon}{C.RESET} {name:<20s} {st}")
                if st == "FAIL":
                    err = info.get("error", "")
                    if err:
                        print(f"       {C.DIM}{err}{C.RESET}")
                elif name == "llm":
                    print(f"       Response: {info.get('response', '?')} ({info.get('latency_ms', '?')}ms)")
                elif name == "embedding":
                    print(f"       Dim={info.get('dim', '?')} ({info.get('latency_ms', '?')}ms)")
                elif name == "brain_auth":
                    print(f"       Latency: {info.get('latency_ms', '?')}ms")
                elif name == "components":
                    for cn, cs in info.items():
                        if isinstance(cs, str):
                            print(f"         {cn:<24s} {cs}")

    # ── MAB / Session helpers ────────────────────────────────────────────

    async def get_mab_status(self) -> dict | None:
        from openalpha_brain.core import loop_state as _ls

        mab = getattr(_ls, "_mab", None)
        if not mab:
            return None
        try:
            sel = mab.select(top_k_ops=3, top_k_fields=5, focus_area=self.focus_area)
            return {
                "direction": sel.get("direction", "?"),
                "top_ops": sel.get("operators", [])[:5],
                "top_fields": sel.get("fields", [])[:8],
                "focus_area": self.focus_area,
            }
        except (ValueError, KeyError, AttributeError, RuntimeError) as e:
            return {"error": str(e)}

    async def list_sessions(self) -> list[dict]:
        from openalpha_brain.cli import session_manager as sm

        sessions = []
        for sid in (await sm.list_sessions())[:10]:
            sess = await sm.load_session(sid)
            if not sess:
                continue
            sessions.append(
                {
                    "id": sid[:16],
                    "status": getattr(sess.status, "value", "?") if hasattr(sess.status, "value") else str(sess.status),
                    "cycles": getattr(sess, "cycle", 0),
                    "passed": len(getattr(sess, "passed_alphas", [])),
                    "rejected": len(getattr(sess, "rejected_alphas", [])),
                    "generated": len(getattr(sess, "alpha_results", [])),
                    "focus": getattr(sess, "focus_area", "?"),
                }
            )
        return sessions

    async def get_session_detail(self, session_id: str) -> dict:
        from openalpha_brain.cli import session_manager as sm

        sess = await sm.load_session(session_id)
        if not sess:
            return {"error": "session not found"}
        alphas = []
        for a in getattr(sess, "passed_alphas", []):
            ad = {
                "expr": getattr(a, "expression", "?")[:100],
                "direction": getattr(a, "exploration_direction", "?"),
                "decision": getattr(a, "decision", "?"),
            }
            brain = getattr(a, "brain", None)
            if brain:
                ad.update(
                    {
                        "sharpe": getattr(brain, "real_sharpe", None),
                        "fitness": getattr(brain, "real_fitness", None),
                        "turnover": getattr(brain, "real_turnover", None),
                        "returns": getattr(brain, "real_returns", None),
                        "drawdown": getattr(brain, "real_drawdown", None),
                        "brain_status": (lambda b=brain: getattr(b, "simulation_status", lambda: getattr(b, "status", "?"))())()  # noqa: E501
                        if callable(getattr(brain, "simulation_status", None))
                        else getattr(brain, "status", "?"),
                    }
                )
            alphas.append(ad)
        return {
            "id": sess.id,
            "status": sess.status.value if hasattr(sess.status, "value") else str(sess.status),
            "cycles": getattr(sess, "cycle", 0),
            "focus_area": getattr(sess, "focus_area", "?"),
            "passed_count": len(alphas),
            "alphas": alphas,
        }

    # ── Mining Control ────────────────────────────────────────────────────

    async def start_mining(
        self, focus: str = "momentum", cycles: int = 5, brain_submit: bool = True, auto_confirm: bool = False
    ) -> None:
        if self.state == "RUNNING":
            print(f"  {C.YELLOW}Already running! Use 'stop' first.{C.RESET}")
            return

        self.focus_area = focus
        self.max_cycles = cycles
        self.brain_submit_enabled = brain_submit
        self.auto_confirm = auto_confirm
        self._stop_event.clear()
        self._pause_event.set()
        self._event_count = 0
        self._alpha_count = 0
        self._reject_count = 0
        self._brain_count = 0
        self._live_alphas = []
        self._start_time = time.time()

        from openalpha_brain.cli import session_manager as sm
        from openalpha_brain.config.config import settings
        from openalpha_brain.core import loop_state as _ls

        settings.BRAIN_SUBMIT_ENABLED = brain_submit
        settings.MAX_CYCLES = cycles
        settings.PIPELINE_MODE = True
        _ls.init_intelligent_search()

        state = await sm.create_session(focus_area=focus)
        self.session_id = state.id
        self.state = "RUNNING"

        self._bus.subscribe(self._on_event)
        from openalpha_brain.core import loop_state as _ls_mod

        _ls_mod._console_stop_event = self._stop_event
        _ls_mod._console_pause_event = self._pause_event

        print(f"\n  {C.GREEN}▶ MINING STARTED (Pipeline Mode){C.RESET}")
        print(f"  Session : {C.CYAN}{self.session_id}{C.RESET}")
        print(f"  Focus   : {focus}")
        print(f"  Cycles  : {cycles}")
        print(f"  BRAIN   : {'ON (real submit)' if brain_submit else 'OFF (dry-run)'}")
        print(f"  Mode    : {C.BOLD}ASYNC — events stream in real-time{C.RESET}")
        print(f"  Type commands anytime. Use {C.YELLOW}'status'{C.RESET} for live dashboard.\n")

        try:
            from openalpha_brain.core import loop_engine

            await loop_engine.run_loop_pipeline(self.session_id)
        except asyncio.CancelledError:
            print(f"\n  {C.YELLOW}■ Mining cancelled by user{C.RESET}")
        except (RuntimeError, OSError, KeyboardInterrupt) as exc:
            self.state = "ERROR"
            logger.error("Mining error: %s", exc, exc_info=True)
            print(f"\n  {C.RED}✖ ERROR: {exc}{C.RESET}")
            traceback.print_exc()

        self._bus.unsubscribe(self._on_event)
        if self.state != "ERROR":
            self.state = "COMPLETED"

        detail = await self.get_session_detail(self.session_id)
        self._print_final_summary(detail)

    async def stop_mining(self) -> None:
        if self.state not in ("RUNNING", "PAUSED"):
            print(f"  Not running (state={self.state})")
            return
        self.state = "STOPPING"
        self._stop_event.set()
        self._pause_event.set()
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self.state = "IDLE"
        print(f"  {C.YELLOW}■ STOPPED{C.RESET}")

    async def pause_mining(self) -> None:
        if self.state != "RUNNING":
            print("  Not running")
            return
        self.state = "PAUSED"
        self._pause_event.clear()
        print(f"  {C.YELLOW}‖ PAUSED{C.RESET}")

    async def resume_mining(self) -> None:
        if self.state != "PAUSED":
            print("  Not paused")
            return
        self.state = "RUNNING"
        self._pause_event.set()
        print(f"  {C.GREEN}▶ RESUMED{C.RESET}")

    # ── Summary Output ────────────────────────────────────────────────────

    def _print_final_summary(self, detail: dict) -> None:
        alphas = detail.get("alphas", [])
        merged = list(self._live_alphas)
        {a["alpha_id"] for a in merged}
        for a in alphas:
            if a.get("expr", "?")[:8] not in [x.get("expression", "")[:8] for x in merged]:
                merged.append(
                    {
                        "alpha_id": "?",
                        "expression": a.get("expr", ""),
                        "status": a.get("brain_status", "?"),
                        "sharpe": a.get("sharpe"),
                        "fitness": a.get("fitness"),
                        "turnover": a.get("turnover"),
                        "drawdown": a.get("drawdown"),
                        "direction": a.get("direction", ""),
                    }
                )

        print(f"\n  {C.BOLD}{'═══ FINAL SUMMARY ═══':═^68}{C.RESET}")
        print(f"  Session  : {detail.get('id', '?')}")
        print(f"  Status   : {detail.get('status', '?')}")
        print(f"  Cycles   : {detail.get('cycles', 0)}")
        print(f"  Focus    : {detail.get('focus_area', '?')}")
        print(
            f"  Alphas   : {len(merged)} total | {self._alpha_count} passed | {self._reject_count} rejected | {self._brain_count} BRAIN results\n"  # noqa: E501
        )

        if not merged:
            print(f"  {C.YELLOW}No alphas generated.{C.RESET}")
            return

        print(f"  {C.BOLD}{'#':<4s} {'Expression':<52s} {'Sharpe':>7s} {'Fit':>6s} {'TO':>6s} {'Gate':>6s}{C.RESET}")
        print(f"  {'─' * 86}")
        for i, a in enumerate(merged, 1):
            sh, fit, to, bs = a.get("sharpe"), a.get("fitness"), a.get("turnover"), a.get("status", "?")
            sh_s = f"{sh:>7.2f}" if sh is not None else "     N/A"
            fit_s = f"{fit:>6.2f}" if fit is not None else "   N/A"
            to_s = f"{to:>5.1f}%" if to is not None else "  N/A"
            sc = C.GREEN if bs == "PASS" else (C.RED if bs == "FAIL" else C.YELLOW)
            expr = (a.get("expression", "") or "?")[:51]
            print(f"  {i:<4d} {expr:<52s} {sh_s} {fit_s} {to_s} {_c(bs, sc):>6s}")

        sharpes = [a["sharpe"] for a in merged if a.get("sharpe") is not None]
        if sharpes:
            avg, best = sum(sharpes) / len(sharpes), max(sharpes)
            above_1 = sum(1 for s in sharpes if s >= 1.0)
            print(f"\n  Sharpe: mean={avg:.3f}  best={best:.3f}  ≥1.0: {above_1}/{len(sharpes)}")


# ════════════════════════════════════════════════════════════════════════════
# Non-Blocking REPL (threaded input + async command processing)
# ════════════════════════════════════════════════════════════════════════════
_HELP_TEXT = f"""{C.BOLD}Commands:{C.RESET}
  {C.GREEN}start{C.RESET} [focus] [cycles] [--no-brain] [--auto]   Start mining (async, non-blocking!)
  {C.YELLOW}stop{C.RESET}          Stop mining
  {C.YELLOW}pause{C.RESET}/{C.GREEN}resume{C.RESET}                  Pause/resume
  {C.CYAN}health{C.RESET}         System health check
  {C.MAGENTA}status{C.RESET}         Live dashboard (events + alphas)
  {C.MAGENTA}mab{C.RESET}           MAB selection status
  {C.BLUE}sessions{C.RESET}        List recent sessions
  {C.BLUE}session{C.RESET} <id>    Session detail
  {C.BLUE}brain{C.RESET} <id>      BRAIN results
  {C.RED}config{C.RESET}          Show configuration
  {C.RED}set{C.RESET} <key> <val> Set config value
  {C.DIM}help{C.RESET}           This help
  {C.DIM}quit{C.RESET}           Exit"""


def _input_thread(cmd_queue: queue.Queue, prompt: str) -> None:
    while True:
        try:
            line = input(prompt).strip()
            cmd_queue.put(line)
        except (EOFError, KeyboardInterrupt):
            cmd_queue.put("__QUIT__")
            break


async def repl(cli: AlphaCLI) -> None:
    cli._banner()
    print(_HELP_TEXT)
    print(f"  {C.DIM}Tip: Type 'start' to begin — mining runs in background, you can type commands anytime.{C.RESET}\n")

    cmd_queue: queue.Queue[str] = queue.Queue()
    prompt = f"{C.CYAN}α-cli{C.RESET}> "
    input_thread = threading.Thread(target=_input_thread, args=(cmd_queue, prompt), daemon=True)
    input_thread.start()

    loop = asyncio.get_event_loop()
    while True:
        try:
            raw = await loop.run_in_executor(None, cmd_queue.get, None)
        except (OSError, ValueError, RuntimeError, EOFError):
            break
        if raw == "__QUIT__":
            break
        if not raw:
            continue

        parts = raw.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ("quit", "exit", "q"):
            if cli.state == "RUNNING":
                print("  Mining still running. Stopping...")
                await cli.stop_mining()
            print(f"\n  {C.DIM}Goodbye! 🧠{C.RESET}\n")
            break

        if cmd in ("help", "?"):
            print(_HELP_TEXT)

        elif cmd == "health":
            h = await cli.check_health()
            cli.print_health(h)

        elif cmd == "status":
            print(cli._status_bar())
            if cli._live_alphas:
                print(f"\n  {C.BOLD}{'Live BRAIN Results':─^64}{C.RESET}")
                for i, a in enumerate(cli._live_alphas[-10:], 1):
                    sh = a.get("sharpe")
                    st = a.get("status", "?")
                    sc = C.GREEN if st == "PASS" else C.RED
                    expr = (a.get("expression", "") or "?")[:55]
                    sh_s = f"{sh:>6.2f}" if sh is not None else "    N/A"
                    print(f"  {i:<3d} {expr:<56s} {sh_s} {_c(st, sc)}")
            if cli.session_id:
                print(f"\n  Session: {cli.session_id}  Events: {cli._event_count}")

        elif cmd == "mab":
            mab = await cli.get_mab_status()
            if mab and "error" not in mab:
                print(f"\n  {C.BOLD}{'── MAB STATUS ──':─^50}{C.RESET}")
                print(f"  Direction : {_c(mab.get('direction', '?'), C.GREEN)}")
                print(f"  Focus    : {mab.get('focus_area', '?')}")
                print(f"  Top Ops  : {', '.join(mab.get('top_ops', []))}")
                print(f"  Top Fields: {', '.join(mab.get('top_fields', []))}")
            else:
                print(f"  {C.RED}MAB not available{C.RESET}")

        elif cmd == "sessions":
            sessions = await cli.list_sessions()
            print(f"\n  {C.BOLD}{'── SESSIONS ──':─^64}{C.RESET}")
            if not sessions:
                print(f"  {C.DIM}No sessions found.{C.RESET}")
            else:
                print(f"  {'ID':<18s} {'Status':<10s} {'Cyc':>4s} {'Pass':>4s} {'Rej':>4s} {'Gen':>4s} {'Focus':<12s}")
                print(f"  {'─' * 66}")
                for s in sessions:
                    print(
                        f"  {s['id']:<18s} {s['status']:<10s} {s['cycles']:>4d} {s['passed']:>4d} {s['rejected']:>4d} {s['generated']:>4d} {s['focus']:<12s}"  # noqa: E501
                    )

        elif cmd == "session":
            sid = args[0] if args else cli.session_id
            if not sid:
                print("  No active session.")
                continue
            detail = await cli.get_session_detail(sid)
            if "error" in detail:
                print(f"  {C.RED}Error: {detail['error']}{C.RESET}")
            else:
                cli._print_final_summary(detail)

        elif cmd == "brain":
            sid = args[0] if args else cli.session_id
            if not sid:
                print("  No active session.")
                continue
            detail = await cli.get_session_detail(sid)
            if "error" in detail:
                print(f"  {C.RED}Error: {detail['error']}{C.RESET}")
            elif not detail.get("alphas"):
                print(f"  {C.DIM}No BRAIN submissions yet.{C.RESET}")
            else:
                print(f"\n  {C.BOLD}{'── BRAIN RESULTS ──':─^64}{C.RESET}")
                for i, a in enumerate(detail["alphas"], 1):
                    has_brain = any(
                        v is not None
                        for k, v in a.items()
                        if k in ("sharpe", "fitness", "turnover", "returns", "drawdown", "brain_status")
                    )
                    if not has_brain:
                        print(f"  [{i}] {a.get('expr', '?')[:60]} → {C.DIM}(not submitted){C.RESET}")
                        continue
                    print(f"\n  [{i}] {C.BOLD}{a.get('expr', '?')[:70]}{C.RESET}")
                    for key, label in [
                        ("brain_status", "Gate"),
                        ("sharpe", "Sharpe"),
                        ("fitness", "Fitness"),
                        ("turnover", "Turnover"),
                        ("returns", "Returns"),
                        ("drawdown", "Drawdown"),
                    ]:
                        val = a.get(key)
                        if val is not None:
                            if label in ("Turnover", "Returns"):
                                print(f"       {label:<10s}: {val}%")
                            else:
                                print(f"       {label:<10s}: {val}")

        elif cmd == "config":
            from openalpha_brain.config.config import settings

            print(f"\n  {C.BOLD}{'── CONFIGURATION ──':─^64}{C.RESET}")
            for attr in [
                "LLM_MODEL",
                "LLM_PROVIDER",
                "LMSTUDIO_API_BASE",
                "BRAIN_SUBMIT_ENABLED",
                "MAX_CYCLES",
                "PIPELINE_MODE",
                "RAG_ENABLED",
                "MAB_ENABLED",
                "RAG_TOP_K_OPS",
                "RAG_TOP_K_FIELDS",
                "BRAIN_POLL_TIMEOUT",
                "DEFAULT_EXPLORATION_DIRECTION",
            ]:
                val = getattr(settings, attr, "(not set)")
                print(f"  {attr:<30s}: {val}")
            print("\n  CLI State:")
            print(f"    focus_area   : {cli.focus_area}")
            print(f"    max_cycles   : {cli.max_cycles}")
            print(f"    brain_submit : {cli.brain_submit_enabled}")
            print(f"    auto_confirm : {cli.auto_confirm}")
            print(f"    state        : {cli.state}")

        elif cmd == "set":
            if len(args) >= 2:
                key, val = args[0], args[1]
                if key == "focus":
                    cli.focus_area = val
                elif key == "cycles":
                    cli.max_cycles = int(val)
                elif key == "brain":
                    cli.brain_submit_enabled = val.lower() in ("on", "true", "1", "yes")
                elif key == "auto":
                    cli.auto_confirm = val.lower() in ("on", "true", "1", "yes")
                else:
                    print(f"  Unknown key: {key}. Try: focus, cycles, brain, auto")
                    continue
                print(f"  ✓ Set {key} = {val}")
            else:
                print("  Usage: set <key> <value>\n  Keys: focus, cycles, brain(on/off), auto(on/off)")

        elif cmd == "start":
            focus, cycles, brain, auto = cli.focus_area, cli.max_cycles, cli.brain_submit_enabled, cli.auto_confirm
            i = 0
            while i < len(args):
                a = args[i]
                if a in ("--no-brain", "--nobrain"):
                    brain = False
                elif a == "--auto":
                    auto = True
                elif a == "--manual":
                    auto = False
                elif not a.startswith("-") and i == 0:
                    focus = a
                elif not a.startswith("-"):
                    with contextlib.suppress(ValueError):
                        cycles = int(a)
                i += 1
            cli._task = asyncio.create_task(
                cli.start_mining(focus=focus, cycles=cycles, brain_submit=brain, auto_confirm=auto),
            )

        elif cmd == "stop":
            await cli.stop_mining()
        elif cmd == "pause":
            await cli.pause_mining()
        elif cmd == "resume":
            await cli.resume_mining()
        else:
            print(f"  Unknown command: {_c(cmd, C.RED)}. Type {_c('help', C.YELLOW)}.")


# ════════════════════════════════════════════════════════════════════════════
# Entry Point — argparse subcommands
# ════════════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="openalpha",
        description="OpenAlpha-Brain: Autonomous alpha research platform for WorldQuant BRAIN",
    )
    sub = p.add_subparsers(dest="command", help="Available commands")

    run_p = sub.add_parser("run", help="Start alpha mining")
    run_p.add_argument("--focus", default="momentum", help="Exploration direction (default: momentum)")
    run_p.add_argument("--cycles", type=int, default=5, help="Number of cycles (default: 5)")
    run_p.add_argument("--no-brain", action="store_true", help="Disable BRAIN submission")

    sub.add_parser("status", help="Check system health and MAB status")
    sub.add_parser("sessions", help="List recent mining sessions")

    brain_p = sub.add_parser("brain", help="Query BRAIN platform for an alpha")
    brain_p.add_argument("alpha_id", help="Alpha ID to query")

    sub.add_parser("interactive", help="Launch interactive REPL mode")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    cli = AlphaCLI()

    if args.command == "run":
        cli._banner()
        asyncio.run(
            cli.start_mining(
                focus=args.focus,
                cycles=args.cycles,
                brain_submit=not args.no_brain,
            )
        )
    elif args.command == "status":
        cli._banner()
        asyncio.run(_monitor_mode(cli))
    elif args.command == "sessions":
        cli._banner()
        asyncio.run(_list_sessions(cli))
    elif args.command == "brain":
        cli._banner()
        asyncio.run(_query_brain(cli, args.alpha_id))
    elif args.command == "interactive" or args.command is None:
        asyncio.run(repl(cli))
    else:
        asyncio.run(repl(cli))


async def _list_sessions(cli: AlphaCLI) -> None:
    sessions = await cli.list_sessions()
    if sessions:
        print(f"\n  {'── SESSIONS ──':─^64}")
        for s in sessions[:20]:
            print(
                f"  {s['id']:<18s} {s['status']:<10s} cyc={s['cycles']} pass={s['passed']} rej={s['rejected']} gen={s['generated']}"  # noqa: E501
            )
    else:
        print("\n  No sessions found.")


async def _query_brain(cli: AlphaCLI, alpha_id: str) -> None:
    print(f"\n  Querying BRAIN for alpha {alpha_id}...")
    result = await cli.query_brain(alpha_id)
    if result:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        print(f"  No result found for alpha {alpha_id}")


async def _monitor_mode(cli: AlphaCLI) -> None:
    print(f"  {C.YELLOW}System Status{C.RESET}\n")
    health = await cli.check_health()
    cli.print_health(health)
    sessions = await cli.list_sessions()
    if sessions:
        print(f"\n  {C.BOLD}{'── LATEST SESSIONS ──':─^64}{C.RESET}")
        for s in sessions[:5]:
            print(
                f"  {s['id']:<18s} {s['status']:<10s} cyc={s['cycles']} pass={s['passed']} rej={s['rejected']} gen={s['generated']}"  # noqa: E501
            )
    mab = await cli.get_mab_status()
    if mab:
        print(f"\n  MAB Direction: {_c(mab.get('direction', '?'), C.GREEN)}")
        print(f"  Top Ops: {', '.join(mab.get('top_ops', []))}")


if __name__ == "__main__":
    main()
