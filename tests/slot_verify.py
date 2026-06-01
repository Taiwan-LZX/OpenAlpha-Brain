"""
WQ BRAIN 3-Slot Concurrent Verification Test
Uses SlotManager to submit factors in parallel (3 slots simultaneously)
Real credentials, real metrics, real throughput measurement.
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
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SLOT-TEST")

TEST_FACTORS = [
    {
        "name": "Mom_CloseOpen_10d",
        "expr": "ts_decay_linear(group_neutralize(ts_rank(close - open, 10), industry), 20)",
        "s": "momentum",
    },
    {
        "name": "Rev_VolumeDelta_5d",
        "expr": "ts_decay_linear(group_neutralize(rank(ts_delta(volume / adv20, 5)), sector), 15)",
        "s": "reversal",
    },
    {
        "name": "Vol_Range_Std_15d",
        "expr": "ts_decay_linear(group_neutralize(ts_std_dev(high - low, 15), subindustry), 12)",
        "s": "volatility",
    },
    {
        "name": "Trend_Accel_12d",
        "expr": "ts_decay_linear(group_neutralize(ts_delta(ts_mean(close,5)-ts_mean(close,20),3), subindustry), 16)",
        "s": "trend",
    },
    {
        "name": "Liq_Spread_Corr_8d",
        "expr": "ts_decay_linear(group_neutralize(ts_corr(volume, close-open, 8), industry), 18)",
        "s": "liquidity",
    },
    {
        "name": "Price_ZScore_20d",
        "expr": "ts_decay_linear(group_neutralize(ts_zscore(close, 20), sector), 10)",
        "s": "mean_rev",
    },
    {
        "name": "Comp_MomRev_Hybrid",
        "expr": "ts_decay_linear(group_neutralize(rank(ts_rank(close,5)) - rank(ts_rank(close,20)), sector), 18)",
        "s": "composite",
    },
    {
        "name": "Micro_OI_Change_6d",
        "expr": "ts_decay_linear(group_neutralize(ts_av_diff(open_interest, 6), industry), 14)",
        "s": "micro",
    },
    {
        "name": "Skew_Returns_20d",
        "expr": "ts_decay_linear(group_neutralize(ts_skewness(close / delay(close,1) - 1, 20), sector), 10)",
        "s": "skewness",
    },
    {
        "name": "AccVol_Divergence",
        "expr": "ts_decay_linear(group_neutralize(ts_corr(abs(return_0), volume, 10) - ts_corr(return_0, volume, 10), industry), 15)",
        "s": "anomaly",
    },
]

results = []
results_lock = asyncio.Lock()
start_time = None


async def on_complete(slot, result):
    """SlotManager completion callback — records results in real-time"""
    global results
    entry = {
        "slot_id": slot.slot_id,
        "name": slot.task_name,
        "expr": slot.expression,
        "strategy": slot.metadata.get("strategy", "?") if slot.metadata else "?",
        "sharpe": result.sharpe,
        "fitness": result.fitness,
        "turnover": result.turnover,
        "returns": result.returns,
        "margin": result.margin,
        "passed": result.passed,
        "status": result.simulation_status or "UNKNOWN",
        "alpha_id": result.alpha_id or "",
        "failures": result.failures or [],
        "elapsed_sec": round(slot.elapsed_sec, 1),
        "poll_count": slot.poll_count,
    }
    async with results_lock:
        results.append(entry)

    icon = "✅" if result.passed else "❌"
    sp = f"{result.sharpe:.4f}" if result.sharpe is not None else "---"
    to = f"{result.turnover:.1f}%" if result.turnover is not None else "---"
    logger.info(
        "  %s [Slot %d] %s Sharpe=%s TO=%s (%.0fs, %d polls)",
        icon,
        slot.slot_id,
        slot.task_name[:24],
        sp,
        to,
        slot.elapsed_sec,
        slot.poll_count,
    )

    if result.failures:
        for f in result.failures[:2]:
            logger.info("       → FAIL: %s", f)


async def status_reporter(manager, interval=15):
    """Background task: print status every N seconds"""
    while True:
        await asyncio.sleep(interval)
        m = manager.get_metrics()
        slots = await manager.get_slot_status()
        busy = sum(1 for s in slots if s.state.value != "idle")
        logger.info(
            "--- STATUS: submitted=%d completed=%d passed=%d | slots_busy=%d/%d | queue=%d | best=%.3f ---",
            m.total_submitted,
            m.total_completed,
            m.total_passed,
            busy,
            len(slots),
            m.queue_depth,
            m.best_sharpe,
        )


async def main():
    global start_time
    start_time = time.time()

    logger.info("=" * 65)
    logger.info("WQ 3-SLOT CONCURRENT VERIFICATION TEST")
    logger.info("Factors: %d | Slots: 3 | Mode: PARALLEL", len(TEST_FACTORS))
    logger.info("Started: %s", datetime.now().isoformat())
    logger.info("=" * 65)

    # ── Auth ───────────────────────────────────────────────
    from openalpha_brain.config.config import settings

    email = getattr(settings, "BRAIN_EMAIL", "") or ""
    password = getattr(settings, "BRAIN_PASSWORD", "") or ""
    if not email or not password:
        logger.error("Missing BRAIN_EMAIL/BRAIN_PASSWORD")
        return False

    from openalpha_brain.services import brain_client

    cookies = await brain_client.authenticate(email, password)
    logger.info("[Auth] OK")

    # ── Create SlotManager ────────────────────────────────
    from openalpha_brain.services.slot_manager import SlotManager

    manager = SlotManager(
        cookies=cookies,
        max_slots=3,
        poll_interval=5.0,
        max_poll_seconds=300,
    )
    manager.register_callback(on_complete)

    # Start background status reporter
    reporter_task = asyncio.create_task(status_reporter(manager, 15))

    # Start slot workers
    await manager.start()
    logger.info("[SlotManager] STARTED — 3 slots ready")

    # ── Submit all factors (non-blocking!) ─────────────────
    logger.info("")
    logger.info("--- Submitting %d factors to queue ---", len(TEST_FACTORS))

    for i, f in enumerate(TEST_FACTORS):
        tid = await manager.submit(
            expression=f["expr"],
            name=f["name"],
            strategy=f["s"],
            priority=i,
            metadata={"strategy": f["s"]},
        )
        logger.info("  [%d/%d] Queued: %s (task=%s)", i + 1, len(TEST_FACTORS), f["name"], tid)

    # ── Wait for all completions ──────────────────────────
    logger.info("")
    logger.info("--- All submitted. Waiting for completions... ---")

    timeout_sec = 600  # 10 min max
    check_interval = 10
    elapsed = 0

    while elapsed < timeout_sec:
        await asyncio.sleep(check_interval)
        elapsed += check_interval
        m = manager.get_metrics()

        if m.total_completed >= len(TEST_FACTORS):
            logger.info("All %d factors completed!", m.total_completed)
            break

        if m.queue_depth == 0:
            busy = 0
            slots = await manager.get_slot_status()
            for s in slots:
                if s.state.value not in ("idle", "error"):
                    busy += 1
            if busy == 0 and m.total_completed > 0:
                logger.info("Queue empty + all slots idle → done")
                break

        if elapsed % 30 == 0:
            logger.info(
                "  ... waiting: %d/%d done, queue=%d, best_sharpe=%.3f",
                m.total_completed,
                len(TEST_FACTORS),
                m.queue_depth,
                m.best_sharpe,
            )

    # ── Cleanup ───────────────────────────────────────────
    reporter_task.cancel()
    try:
        await reporter_task
    except asyncio.CancelledError:
        pass

    await manager.stop()

    # ── Final Report ──────────────────────────────────────
    total_time = time.time() - start_time
    print_final_report(results, total_time, manager)

    return True


def print_final_report(results, total_time, manager):
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL RESULTS — 3-Slot Concurrent Test")
    logger.info("=" * 70)

    m = manager.get_metrics()
    total = len(results)
    _passed = sum(1 for r in results if r["passed"])
    valid_sp = [r["sharpe"] for r in results if r["sharpe"] is not None]

    logger.info("Total Time: %.0fs | Throughput: %.1f alphas/min", total_time, total / max(total_time / 60, 0.01))
    logger.info(
        "Submitted: %d | Completed: %d | Passed(Sharpe>=1.25): %d", m.total_submitted, m.total_completed, m.total_passed
    )
    logger.info("")

    if valid_sp:
        avg = sum(valid_sp) / len(valid_sp)
        mx = max(valid_sp)
        mn = min(valid_sp)
        g125 = sum(1 for s in valid_sp if s >= 1.25)
        g100 = sum(1 for s in valid_sp if s >= 1.0)
        g075 = sum(1 for s in valid_sp if s >= 0.75)
        g000 = sum(1 for s in valid_sp if s >= 0)
        neg = sum(1 for s in valid_sp if s < 0)

        logger.info("--- Sharpe Distribution ---")
        logger.info("  Count: %d | Avg: %.4f | Best: %.4f | Worst: %.4f", len(valid_sp), avg, mx, mn)
        logger.info("  >= 1.25 (PASS):  %d (%.0f%%)", g125, g125 / len(valid_sp) * 100)
        logger.info("  >= 1.00 (near):  %d (%.0f%%)", g100, g100 / len(valid_sp) * 100)
        logger.info("  >= 0.75 (weak):  %d (%.0f%%)", g075, g075 / len(valid_sp) * 100)
        logger.info("  >= 0.00 (flat):   %d (%.0f%%)", g000, g000 / len(valid_sp) * 100)
        logger.info("  < 0.00 (loss):    %d (%.0f%%)", neg, neg / len(valid_sp) * 100)

    logger.info("")
    logger.info("--- Per-Factor Results ---")
    logger.info(
        "  %-3s %-26s %-10s %8s %7s %6s %6s  %s",
        "#",
        "Name",
        "Strategy",
        "Sharpe",
        "Fitness",
        "TO%",
        "Time",
        "Status/Fails",
    )
    logger.info("  " + "-" * 95)

    for i, r in enumerate(sorted(results, key=lambda x: x.get("sharpe") or -999, reverse=True)):
        sp = f"{r['sharpe']:.4f}" if r["sharpe"] is not None else "----"
        ft = f"{r['fitness']:.3f}" if r["fitness"] is not None else "---"
        to = f"{r['turnover']:.1f}" if r["turnover"] is not None else "---"
        tm = f"{r['elapsed_sec']}s"
        st = "PASS" if r["passed"] else ("FAIL" if r["status"] else "ERR")
        fl = "; ".join(r["failures"][:1]) if r["failures"] else ""
        if len(r["failures"]) > 1:
            fl += f"(+{len(r['failures']) - 1})"

        logger.info(
            "  %-3d %-26s %-10s %8s %7s %6s %6s  %s %s",
            i + 1,
            r["name"][:24],
            r["strategy"][:8],
            sp,
            ft,
            to,
            tm,
            st,
            fl,
        )

    logger.info("")
    logger.info("=" * 70)

    # Save JSON
    out_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"slot_verify_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_type": "3-slot-concurrent",
                "started_at": datetime.fromtimestamp(start_time).isoformat() if start_time else None,
                "total_time_sec": round(total_time, 1),
                "metrics": {k: v for k, v in m.__dict__.items() if not k.startswith("_")},
                "results": results,
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
