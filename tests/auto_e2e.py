"""
OpenAlpha-Brain Full Autonomous Loop E2E Test
=================================================
Real pipeline test: LLM Generate → PreFilter → WQ Submit(3-slot) →
Feedback → LLM Improve → Priority Boost Re-submit

Uses: Real LM Studio + Real WQ BRAIN Platform
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("AUTO-E2E")

# Suppress noisy httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)

START_TIME = None
ALL_RESULTS = []


async def main():
    global START_TIME
    START_TIME = time.time()

    logger.info("=" * 70)
    logger.info("OpenAlpha-Brain — FULL AUTONOMOUS LOOP E2E TEST")
    logger.info("Pipeline: LLM→PreFilter→WQ(3-slot)→Feedback→LLM Improve→Re-submit")
    logger.info("Started: %s", datetime.now().isoformat())
    logger.info("=" * 70)

    # ── Phase 1: Auth ─────────────────────────────────────
    from openalpha_brain.config.config import settings

    email = getattr(settings, "BRAIN_EMAIL", "") or ""
    password = getattr(settings, "BRAIN_PASSWORD", "") or ""

    if not email or not password:
        logger.error("[FATAL] Missing BRAIN_EMAIL/BRAIN_PASSWORD in .env")
        return False

    from openalpha_brain.services import brain_client

    cookies = await brain_client.authenticate(email, password)
    logger.info("[Phase 1] Auth OK — %s***", email[:3])

    # ── Phase 2: LLM Check ───────────────────────────────
    llm_ok = await check_llm()
    if not llm_ok:
        logger.error("[Phase 2] LLM not available — aborting")
        return False
    logger.info("[Phase 2] LLM OK")

    # ── Phase 3: Create SlotManager ────────────────────────
    from openalpha_brain.services.slot_manager import SlotManager

    slot_manager = SlotManager(
        cookies=cookies,
        max_slots=3,
        poll_interval=5.0,
        max_poll_seconds=300,
    )
    await slot_manager.start()
    logger.info("[Phase 3] SlotManager started (3 slots)")

    # Register callback to collect results
    async def on_complete(slot, result):
        entry = {
            "slot": slot.slot_id,
            "name": slot.task_name,
            "expr": slot.expression,
            "source": slot.metadata.get("source", "?") if slot.metadata else "?",
            "tier": slot.metadata.get("tier", "?") if slot.metadata else "?",
            "improvement_gen": slot.metadata.get("improvement_generation", 0) if slot.metadata else 0,
            "sharpe": result.sharpe,
            "fitness": result.fitness,
            "turnover": result.turnover,
            "passed": result.passed,
            "status": result.simulation_status,
            "elapsed": round(slot.elapsed_sec, 1),
            "failures": result.failures or [],
        }
        ALL_RESULTS.append(entry)

        icon = "✅" if result.passed else ("⚡" if entry["source"] == "improved" else "❌")
        sp = f"{result.sharpe:.3f}" if result.sharpe is not None else "---"
        src_tag = "[IMPROVED]" if entry["source"] == "improved" else "[RAW]"
        logger.info(
            "  %s [Slot%d] %s %s Sharpe=%s TO=%s (%.0fs)",
            icon,
            slot.slot_id,
            src_tag,
            (slot.task_name or "")[:22],
            sp,
            f"{result.turnover:.1f}%" if result.turnover else "---",
            slot.elapsed_sec,
        )

    slot_manager.register_callback(on_complete)

    # ── Phase 4: Create FeedbackLoopOrchestrator ───────────
    from openalpha_brain.core.feedback_orchestrator import create_orchestrator

    orchestrator = await create_orchestrator(
        cookies=cookies,
        slot_manager=slot_manager,
        config={
            "max_improvement_generations": 2,
            "sharpe_improve_threshold": 0.8,
            "sharpe_pass_threshold": 1.25,
            "generate_batch_size": 3,
            "cycle_interval_sec": 2.0,
        },
        auto_start=True,
    )
    logger.info("[Phase 4] Orchestrator started")

    # ── Phase 5: Run Autonomous Cycles ─────────────────────
    max_cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    focus_areas = ["momentum", "reversal", "volatility", "liquidity"]

    logger.info("")
    logger.info("[Phase 5] Running %d autonomous cycles...", max_cycles)
    logger.info("=" * 70)

    cycle_results = []
    for i in range(max_cycles):
        area = focus_areas[i % len(focus_areas)]
        try:
            result = await orchestrator.run_one_cycle(focus_area=area)
            cycle_results.append(result)
            logger.info(
                "[Cycle %d/%d] gen=%d filt=%d sub=%d dur=%.1fs",
                i + 1,
                max_cycles,
                result.generated,
                result.prefiltered,
                result.submitted,
                result.duration_sec,
            )
        except Exception as e:
            logger.error("[Cycle %d/%d] ERROR: %s", i + 1, max_cycles, e)
            cycle_results.append(None)

        # Brief pause between cycles
        if i < max_cycles - 1:
            await asyncio.sleep(3)

    # ── Phase 6: Wait for pending completions ─────────────
    logger.info("")
    logger.info("[Phase 6] Waiting for pending WQ simulations...")
    logger.info("(max 5 minutes)")

    waited = 0
    while waited < 300:
        await asyncio.sleep(10)
        waited += 10
        m = slot_manager.get_metrics()
        slots = await slot_manager.get_slot_status()
        busy = sum(1 for s in slots if s.state.value not in ("idle", "error"))

        if m.queue_depth == 0 and busy == 0 and m.total_completed > 0:
            logger.info("All done! completed=%d queue=0 idle=3", m.total_completed)
            break

        if waited % 30 == 0:
            logger.info(
                "  ... waiting: done=%d queue=%d busy=%d best=%.3f",
                m.total_completed,
                m.queue_depth,
                busy,
                m.best_sharpe,
            )

    # ── Phase 7: Cleanup & Report ─────────────────────────
    await orchestrator.stop()
    await slot_manager.stop()

    print_final_report(cycle_results)

    return True


async def check_llm() -> bool:
    """Check if LM Studio is running and responsive"""
    from openalpha_brain.config.config import settings

    base_url = getattr(settings, "LMSTUDIO_API_BASE", "http://localhost:1234")

    try:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base_url}/v1/models")
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("id", "?") for m in data.get("data", [])]
                logger.info("  LLM endpoint: %s | Models: %s", base_url, models[:3])
                return True
    except Exception as e:
        logger.warning("  LLM check failed: %s", e)
    return False


def print_final_report(cycle_results):
    total_time = time.time() - START_TIME

    logger.info("")
    logger.info("=" * 70)
    logger.info("AUTONOMOUS LOOP E2E — FINAL REPORT")
    logger.info("=" * 70)
    logger.info("Total Time: %.0fs (%.1f min)", total_time, total_time / 60)

    # Cycle summary
    logger.info("")
    logger.info("--- Cycle Summary ---")
    for i, cr in enumerate(cycle_results):
        if cr is None:
            logger.info("  Cycle %d: ERROR", i + 1)
        else:
            logger.info(
                "  Cycle %d: gen=%d prefilter=%d submit=%d dur=%.1fs err=%d",
                i + 1,
                cr.generated,
                cr.prefiltered,
                cr.submitted,
                cr.duration_sec,
                len(cr.errors),
            )

    # Results breakdown by source
    raw_results = [r for r in ALL_RESULTS if r.get("source") == "generated"]
    improved_results = [r for r in ALL_RESULTS if r.get("source") == "improved"]

    logger.info("")
    logger.info("--- Results by Source ---")
    logger.info(
        "  Raw generated: %d | Improved (re-submitted): %d | Total: %d",
        len(raw_results),
        len(improved_results),
        len(ALL_RESULTS),
    )

    # Sharpe distribution
    all_sharpes = [r["sharpe"] for r in ALL_RESULTS if r["sharpe"] is not None]
    if all_sharpes:
        avg = sum(all_sharpes) / len(all_sharpes)
        mx = max(all_sharpes)
        mn = min(all_sharpes)
        g125 = sum(1 for s in all_sharpes if s >= 1.25)
        g100 = sum(1 for s in all_sharpes if s >= 1.0)
        g080 = sum(1 for s in all_sharpes if s >= 0.8)
        g000 = sum(1 for s in all_sharpes if s >= 0)
        neg = sum(1 for s in all_sharpes if s < 0)

        logger.info("")
        logger.info("--- Sharpe Distribution ---")
        logger.info("  Count: %d | Avg: %.4f | Best: %.4f | Worst: %.4f", len(all_sharpes), avg, mx, mn)
        logger.info("  >= 1.25 (PASS):   %d (%.0f%%)", g125, g125 / max(len(all_sharpes), 1) * 100)
        logger.info("  >= 1.00 (near):   %d (%.0f%%)", g100, g100 / max(len(all_sharpes), 1) * 100)
        logger.info("  >= 0.80 (improve): %d (%.0f%%)", g080, g080 / max(len(all_sharpes), 1) * 100)
        logger.info("  >= 0.00 (flat):   %d (%.0f%%)", g000, g000 / max(len(all_sharpes), 1) * 100)
        logger.info("  < 0.00 (loss):    %d (%.0f%%)", neg, neg / max(len(all_sharpes), 1) * 100)

        # Improvement effectiveness
        if improved_results and raw_results:
            imp_sp = [r["sharpe"] for r in improved_results if r["sharpe"] is not None]
            raw_sp = [r["sharpe"] for r in raw_results if r["sharpe"] is not None]
            if imp_sp and raw_sp:
                imp_avg = sum(imp_sp) / len(imp_sp)
                raw_avg = sum(raw_sp) / len(raw_sp)
                delta = imp_avg - raw_avg
                logger.info("")
                logger.info("--- Improvement Effectiveness ---")
                logger.info("  Raw avg Sharpe:     %.4f (%d samples)", raw_avg, len(raw_sp))
                logger.info("  Improved avg Sharpe: %.4f (%d samples)", imp_avg, len(imp_sp))
                logger.info("  Delta: %+.4f (%s)", delta, "IMPROVEMENT" if delta > 0 else "REGRESSION")

    # Per-factor detail table
    logger.info("")
    logger.info("--- All Factor Results (sorted by Sharpe) ---")
    sorted_results = sorted(ALL_RESULTS, key=lambda x: x.get("sharpe") or -999, reverse=True)
    logger.info(
        "  %-3s %-24s %-9s %7s %6s %6s %5s  %s", "#", "Name", "Source", "Sharpe", "Fitness", "TO%", "Gen", "Status"
    )
    logger.info("  " + "-" * 90)

    for i, r in enumerate(sorted_results):
        sp = f"{r['sharpe']:.4f}" if r["sharpe"] is not None else "----"
        ft = f"{r['fitness']:.3f}" if r["fitness"] is not None else "---"
        to = f"{r['turnover']:.1f}" if r["turnover"] is not None else "---"
        src = r["source"][:8].upper() if r["source"] else "?"
        gen = str(r.get("improvement_gen", 0))
        st = "PASS" if r["passed"] else "FAIL"
        fl = "; ".join(r["failures"][:1]) if r["failures"] else ""
        logger.info(
            "  %-3d %-24s %-9s %7s %6s %6s %5s  %s %s", i + 1, (r["name"] or "")[:22], src, sp, ft, to, gen, st, fl
        )

    logger.info("")
    logger.info("=" * 70)

    # Save JSON report
    out_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"auto_e2e_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_type": "full_autonomous_loop",
                "started_at": datetime.fromtimestamp(START_TIME).isoformat() if START_TIME else None,
                "total_time_sec": round(total_time, 1),
                "total_results": len(ALL_RESULTS),
                "raw_count": len(raw_results),
                "improved_count": len(improved_results),
                "results": ALL_RESULTS,
            },
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    logger.info("Report saved: %s", out_path)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
