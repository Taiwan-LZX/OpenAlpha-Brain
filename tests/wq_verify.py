"""
WQ BRAIN Real Verification Test v1.0
Purpose: Submit N factors to real WQ platform, record actual metrics,
         verify if the improved pipeline can produce passable alphas.

Uses: Real credentials from .env, real brain_client API
Output: Detailed metrics table + Sharpe distribution analysis
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
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
logger = logging.getLogger("WQ-VERIFY")

# ── Test Factor Pool (diverse strategies) ────────────────────
TEST_FACTORS = [
    {
        "name": "Momentum_CloseOpen_10d",
        "expr": "ts_decay_linear(group_neutralize(ts_rank(close - open, 10), industry), 20)",
        "strategy": "momentum",
        "expected": "moderate",
    },
    {
        "name": "Reversal_VolumeDelta_5d",
        "expr": "ts_decay_linear(group_neutralize(rank(ts_delta(volume / adv20, 5)), sector), 15)",
        "strategy": "reversal",
        "expected": "low",
    },
    {
        "name": "Volatility_Range_15d",
        "expr": "ts_decay_linear(group_neutralize(ts_std_dev(high - low, 15) / ts_mean(high - low, 60), subindustry), 12)",
        "strategy": "volatility",
        "expected": "moderate",
    },
    {
        "name": "Liquidity_Spread_8d",
        "expr": "ts_decay_linear(group_neutralize(ts_corr(ask - bid, volume, 8), industry), 18)",
        "strategy": "liquidity",
        "expected": "unknown",
    },
    {
        "name": "PriceLevel_ZScore_20d",
        "expr": "ts_decay_linear(group_neutralize(ts_zscore(close, 20), sector), 10)",
        "strategy": "mean_reversion",
        "expected": "low",
    },
    {
        "name": "Trend_AccDecel_12d",
        "expr": "ts_decay_linear(group_neutralize(ts_delta(ts_mean(close, 5) - ts_mean(close, 20), 3), subindustry), 16)",
        "strategy": "trend",
        "expected": "moderate",
    },
    {
        "name": "Microstructure_OI_6d",
        "expr": "ts_decay_linear(group_neutralize(ts_av_diff(open_interest, 6), industry), 14)",
        "strategy": "microstructure",
        "expected": "low",
    },
    {
        "name": "Composite_MomRev_10d",
        "expr": "ts_decay_linear(group_neutralize(rank(ts_rank(close, 5)) - rank(ts_rank(close, 20)), sector), 18)",
        "strategy": "composite",
        "expected": "moderate",
    },
]


@dataclass
class SubmissionResult:
    index: int
    name: str
    expr: str
    strategy: str
    status: str = ""
    passed: bool = False
    sharpe: float | None = None
    fitness: float | None = None
    turnover: float | None = None
    returns: float | None = None
    margin: float | None = None
    alpha_id: str = ""
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    submit_time_sec: float = 0.0
    error: str = ""


async def run_verification(max_submissions=8, poll_timeout=180):
    logger.info("=" * 70)
    logger.info("WQ BRAIN REAL VERIFICATION TEST")
    logger.info("Target: %d submissions, poll timeout=%ds", max_submissions, poll_timeout)
    logger.info("Started at: %s", datetime.now().isoformat())
    logger.info("=" * 70)

    results: list[SubmissionResult] = []

    # ── Phase 1: Auth ───────────────────────────────────────
    try:
        from openalpha_brain.config.config import settings

        email = getattr(settings, "BRAIN_EMAIL", "") or ""
        password = getattr(settings, "BRAIN_PASSWORD", "") or ""
        if not email or not password:
            raise ValueError("Missing BRAIN_EMAIL/BRAIN_PASSWORD in .env")

        from openalpha_brain.services import brain_client

        cookies = await brain_client.authenticate(email, password)
        logger.info("[Auth] SUCCESS — logged in as %s***", email[:3])
    except Exception as e:
        logger.error("[Auth] FAIL: %s", e)
        return results

    # ── Phase 2: Sequential Submissions ─────────────────────
    for i, factor in enumerate(TEST_FACTORS[:max_submissions]):
        result = SubmissionResult(
            index=i + 1,
            name=factor["name"],
            expr=factor["expr"],
            strategy=factor["strategy"],
        )

        logger.info("")
        logger.info("-" * 50)
        logger.info("[%d/%d] Testing: %s", i + 1, max_submissions, factor["name"])
        logger.info("  Strategy: %s | Expected: %s", factor["strategy"], factor.get("expected"))
        logger.info("  Expr: %s", factor["expr"])
        logger.info("-" * 50)

        t_start = time.time()

        try:
            from openalpha_brain.services import brain_client

            gate_result = await brain_client.submit_and_poll(
                simulation_payload={
                    "regular": factor["expr"],
                    "type": "REGULAR",
                },
                cookies=cookies,
                max_poll_seconds=poll_timeout,
            )

            result.submit_time_sec = time.time() - t_start
            result.status = gate_result.simulation_status or "UNKNOWN"
            result.passed = gate_result.passed
            result.sharpe = gate_result.sharpe
            result.fitness = gate_result.fitness
            result.turnover = gate_result.turnover
            result.returns = gate_result.returns
            result.margin = gate_result.margin
            result.alpha_id = gate_result.alpha_id or ""
            result.failures = gate_result.failures or []
            result.warnings = gate_result.warnings or []

            sharpe_str = f"{result.sharpe:.4f}" if result.sharpe is not None else "N/A"
            fitness_str = f"{result.fitness:.4f}" if result.fitness is not None else "N/A"
            to_str = f"{result.turnover:.2f}%%" if result.turnover is not None else "N/A"
            ret_str = f"{result.returns:.2f}%%" if result.returns is not None else "N/A"

            status_icon = (
                "✅ PASS" if result.passed else ("⚠️ FAIL" if result.status == "COMPLETE" else f"❌ {result.status}")
            )

            logger.info(
                "  Result: %s | Sharpe=%s Fitness=%s TO=%s Ret=%s",
                status_icon,
                sharpe_str,
                fitness_str,
                to_str,
                ret_str,
            )

            if result.failures:
                for f in result.failures:
                    logger.info("  FAIL: %s", f)

        except Exception as e:
            result.submit_time_sec = time.time() - t_start
            result.error = str(e)
            logger.error("  ERROR: %s", e)

            if "429" in str(e).upper() or "CONCURRENT" in str(e).upper():
                logger.info("  [429] Slot occupied, waiting 30s before next...")
                await asyncio.sleep(30)
            elif "Rate limit" in str(e):
                logger.info("  [Rate limit] Waiting 15s...")
                await asyncio.sleep(15)

        results.append(result)

        # Brief pause between submissions to avoid rate limiting
        if i < max_submissions - 1:
            await asyncio.sleep(5)

    # ── Phase 3: Analysis Report ───────────────────────────
    print_analysis_report(results)

    return results


def print_analysis_report(results: list[SubmissionResult]):
    logger.info("")
    logger.info("=" * 70)
    logger.info("VERIFICATION RESULTS SUMMARY")
    logger.info("=" * 70)

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed and r.status == "COMPLETE")
    errored = sum(1 for r in results if r.error)
    valid_sharpes = [r.sharpe for r in results if r.sharpe is not None]

    logger.info("Total: %d | Passed: %d | Failed: %d | Errors: %d", total, passed, failed, errored)

    if valid_sharpes:
        avg_sharpe = sum(valid_sharpes) / len(valid_sharpes)
        max_sharpe = max(valid_sharpes)
        min_sharpe = min(valid_sharpes)
        above_125 = sum(1 for s in valid_sharpes if s >= 1.25)
        above_100 = sum(1 for s in valid_sharpes if s >= 1.0)
        above_075 = sum(1 for s in valid_sharpes if s >= 0.75)
        below_05 = sum(1 for s in valid_sharpes if s < 0.5)
        negative = sum(1 for s in valid_sharpes if s < 0)

        logger.info("")
        logger.info("--- Sharpe Distribution ---")
        logger.info("  Avg: %.4f | Max: %.4f | Min: %.4f", avg_sharpe, max_sharpe, min_sharpe)
        logger.info("  >= 1.25 (PASS): %d (%.0f%%)", above_125, above_125 / len(valid_sharpes) * 100)
        logger.info("  >= 1.00 (near-pass): %d (%.0f%%)", above_100, above_100 / len(valid_sharpes) * 100)
        logger.info("  >= 0.75 (weak signal): %d (%.0f%%)", above_075, above_075 / len(valid_sharpes) * 100)
        logger.info("  < 0.50 (noise): %d (%.0f%%)", below_05, below_05 / len(valid_sharpes) * 100)
        logger.info("  < 0 (negative): %d (%.0f%%)", negative, negative / len(valid_sharpes) * 100)

    logger.info("")
    logger.info("--- Detailed Results ---")
    logger.info(
        "  %-4s %-28s %-12s %8s %8s %8s %6s  %s",
        "#",
        "Name",
        "Strategy",
        "Sharpe",
        "Fitness",
        "Turnover",
        "Status",
        "Failures",
    )
    logger.info("  " + "-" * 90)

    for r in results:
        sp = f"{r.sharpe:.4f}" if r.sharpe is not None else "----"
        ft = f"{r.fitness:.4f}" if r.fitness is not None else "----"
        to = f"{r.turnover:.1f}%" if r.turnover is not None else "---"
        st = "✅PASS" if r.passed else ("FAIL" if r.status else "ERR")
        fl = "; ".join(r.failures[:2]) if r.failures else ""
        if len(r.failures) > 2:
            fl += f"(+{len(r.failures) - 2})"

        logger.info("  %-4d %-28s %-12s %8s %8s %8s %6s  %s", r.index, r.name[:26], r.strategy, sp, ft, to, st, fl)

    logger.info("")
    logger.info("=" * 70)

    # Save JSON report
    report_path = os.path.join(
        os.path.dirname(__file__), "logs", f"wq_verify_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False, default=str)
    logger.info("Full report saved to: %s", report_path)


async def main():
    max_subs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    results = await run_verification(max_submissions=max_subs)
    return len(results) > 0 and any(r.passed for r in results)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
