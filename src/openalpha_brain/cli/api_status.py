from __future__ import annotations

import contextlib
import logging
from typing import Any

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/status", tags=["status"])


def _get_ctx():
    from openalpha_brain.core.loop_state import _ctx

    return _ctx


@router.get("/mab")
async def get_mab_status():
    ctx = _get_ctx()
    mab = ctx._mab
    if mab is None:
        return {"initialized": False, "arms": None, "health": None}

    direction_stats = mab.get_direction_stats()
    arms: list[dict[str, Any]] = []
    for arm_id, stats in direction_stats.items():
        arms.append(
            {
                "direction": arm_id,
                "alpha": stats.get("alpha", 1.0),
                "beta": stats.get("beta", 1.0),
                "expectation": stats.get("expectation", 0.5),
                "ucb_score": stats.get("expectation", 0.5),
            }
        )

    outer_stats = mab._outer.get_stats() if hasattr(mab, "_outer") else {}
    for arm_id, arm_data in outer_stats.items():
        for arm in arms:
            if arm["direction"] == arm_id:
                arm["alpha"] = arm_data.get("alpha", arm["alpha"])
                arm["beta"] = arm_data.get("beta", arm["beta"])
                arm["expectation"] = arm_data.get("expectation", arm["expectation"])
                arm["ucb_score"] = arm_data.get("expectation", arm["expectation"])
                break

    health = mab.health_check() if hasattr(mab, "health_check") else None

    return {
        "initialized": True,
        "arms": arms,
        "arm_count": len(arms),
        "health": health,
    }


@router.get("/feature-map")
async def get_feature_map_status():
    ctx = _get_ctx()
    fm = ctx._feature_map
    if fm is None:
        return {"initialized": False, "coverage": None, "cells": None, "diversity_stats": None}

    diversity_stats = fm.get_diversity_stats()
    schedule = fm.get_explore_exploit_schedule()

    cells: list[dict[str, Any]] = []
    with fm._lock:
        for key, cell in fm._cells.items():
            cells.append(
                {
                    "key": key,
                    "direction": cell.direction,
                    "time_horizon": cell.time_horizon,
                    "mechanism": cell.mechanism,
                    "best_fitness": cell.best_fitness,
                    "best_sharpe": cell.best_sharpe,
                    "update_count": cell.update_count,
                    "elite_count": len(cell.elites),
                    "decay_state": cell.decay_state,
                    "admission_paused": cell.admission_paused,
                }
            )

    return {
        "initialized": True,
        "coverage": diversity_stats.get("coverage", 0.0),
        "total_cells": diversity_stats.get("total_cells", 0),
        "filled_cells": diversity_stats.get("filled_cells", 0),
        "direction_coverage": diversity_stats.get("direction_coverage", {}),
        "decay_cells": diversity_stats.get("decay_cells", {}),
        "avg_structural_novelty": diversity_stats.get("avg_structural_novelty", 0.0),
        "explore_exploit": schedule,
        "cells": cells,
        "diversity_stats": diversity_stats,
    }


@router.get("/strategy")
async def get_strategy_status():
    ctx = _get_ctx()
    sc = ctx._strategy_classifier
    if sc is None:
        return {"initialized": False, "top_profiles": None, "direction_distribution": None}

    top_profiles = sc.get_top_profiles(n=10)
    profiles_data = []
    for p in top_profiles:
        profiles_data.append(
            {
                "expr": p.expr[:120] if p.expr else "",
                "direction": p.direction,
                "time_horizon": p.time_horizon,
                "mechanism": p.mechanism,
                "sharpe": p.sharpe,
                "fitness": p.fitness,
                "turnover": p.turnover,
                "complexity": p.complexity,
                "operators": p.operators,
            }
        )

    direction_dist: dict[str, int] = {}
    for p in sc._profiles:
        direction_dist[p.direction] = direction_dist.get(p.direction, 0) + 1

    complementary = sc.find_complementary(n=5)

    return {
        "initialized": True,
        "total_profiles": len(sc._profiles),
        "top_profiles": profiles_data,
        "direction_distribution": direction_dist,
        "complementary_suggestions": complementary,
    }


@router.get("/decay")
async def get_decay_status():
    ctx = _get_ctx()
    dd = ctx._decay_detector
    if dd is None:
        return {"initialized": False, "blacklisted_directions": None, "decay_summary": None}

    blacklisted = dd.get_blacklisted_directions()
    permanent = list(dd._permanent_blacklist) if hasattr(dd, "_permanent_blacklist") else []

    records_summary: list[dict[str, Any]] = []
    if hasattr(dd, "_records"):
        for aid, rec in dd._records.items():
            records_summary.append(
                {
                    "alpha_id": aid,
                    "direction": rec.direction,
                    "decay_level": rec.decay_level.value if hasattr(rec.decay_level, "value") else str(rec.decay_level),
                    "ewma_sharpe": rec.ewma_sharpe,
                    "composite_score": rec.composite_score,
                    "initial_sharpe": rec.initial_sharpe,
                    "peak_sharpe": rec.peak_sharpe,
                    "garch_anomaly": rec.garch_anomaly,
                    "consecutive_decay_checks": rec.consecutive_decay_checks,
                }
            )

    fingerprints: list[dict[str, Any]] = []
    if hasattr(dd, "_decay_fingerprints"):
        for fp in dd._decay_fingerprints:
            fingerprints.append(
                {
                    "direction": fp.direction,
                    "topology": fp.topology,
                    "temporal": fp.temporal,
                    "decay_level": fp.decay_level,
                    "reason": fp.reason,
                    "is_permanent": fp.is_permanent,
                }
            )

    return {
        "initialized": True,
        "blacklisted_directions": blacklisted,
        "permanent_blacklist": permanent,
        "tracked_alpha_count": len(records_summary),
        "records": records_summary,
        "decay_fingerprints": fingerprints,
    }


@router.get("/experience")
async def get_experience_status():
    ctx = _get_ctx()
    ed = ctx._experience_distiller
    if ed is None:
        return {"initialized": False, "cards": None, "total_cards": None}

    cards: list[dict[str, Any]] = []
    for c in ed._cards:
        cards.append(
            {
                "rule_id": c.rule_id,
                "failure_pattern": c.failure_pattern,
                "fix_strategy": c.fix_strategy,
                "applicable_conditions": c.applicable_conditions,
                "confidence": c.confidence,
                "usage_count": c.usage_count,
                "success_count": c.success_count,
                "last_used": c.last_used,
                "created_at": c.created_at,
            }
        )

    return {
        "initialized": True,
        "total_cards": len(cards),
        "cards": cards,
    }


@router.get("/overview")
async def get_overview():
    ctx = _get_ctx()

    total_alphas = 0
    passed_alphas = 0
    avg_sharpe = 0.0
    current_cycle = ctx._evolution_cycle_count if hasattr(ctx, "_evolution_cycle_count") else 0

    evo_db = ctx._evo_db
    if evo_db is not None and hasattr(evo_db, "_records"):
        try:
            with evo_db._lock:
                recs = list(evo_db._records.values())
            total_alphas = len(recs)
            passed = [r for r in recs if getattr(r, "status", "") == "PASS"]
            passed_alphas = len(passed)
            sharpes = [r.sharpe for r in passed if r.sharpe is not None and r.sharpe > 0]
            avg_sharpe = round(sum(sharpes) / len(sharpes), 4) if sharpes else 0.0
        except (OSError, ValueError, RuntimeError):
            logger.warning("Failed to compute overview from evolution_db", exc_info=True)

    mab_health = None
    if ctx._mab is not None and hasattr(ctx._mab, "health_check"):
        with contextlib.suppress(Exception):
            mab_health = ctx._mab.health_check()

    fm_coverage = None
    if ctx._feature_map is not None:
        with contextlib.suppress(Exception):
            stats = ctx._feature_map.get_diversity_stats()
            fm_coverage = stats.get("coverage", 0.0)

    decay_dirs: list[str] = []
    if ctx._decay_detector is not None:
        with contextlib.suppress(Exception):
            decay_dirs = ctx._decay_detector.get_blacklisted_directions()

    experience_count = 0
    if ctx._experience_distiller is not None:
        with contextlib.suppress(Exception):
            experience_count = len(ctx._experience_distiller._cards)

    strategy_count = 0
    if ctx._strategy_classifier is not None:
        with contextlib.suppress(Exception):
            strategy_count = len(ctx._strategy_classifier._profiles)

    pass_rate = round(passed_alphas / total_alphas, 4) if total_alphas > 0 else 0.0

    # Count submitted to BRAIN
    submitted_brain = 0
    if evo_db is not None and hasattr(evo_db, "_records"):
        try:
            with evo_db._lock:
                recs = list(evo_db._records.values())
            submitted_brain = sum(1 for r in recs if getattr(r, "brain", None) and getattr(r.brain, "alpha_id", None))
        except (OSError, ValueError, RuntimeError):
            pass

    # Count failed alphas
    failed_alphas = 0
    if evo_db is not None and hasattr(evo_db, "_records"):
        try:
            with evo_db._lock:
                recs = list(evo_db._records.values())
            failed_alphas = len([r for r in recs if getattr(r, "status", "") == "FAIL"])
        except (OSError, ValueError, RuntimeError):
            pass

    return {
        "total_alphas": total_alphas,
        "passed_alphas": passed_alphas,
        "failed_alphas": failed_alphas,
        "pass_rate": pass_rate,
        "avg_sharpe": avg_sharpe,
        "current_cycle": current_cycle,
        "submitted_brain": submitted_brain,
        "mab_health": mab_health,
        "feature_map_coverage": fm_coverage,
        "blacklisted_directions": decay_dirs,
        "experience_cards_count": experience_count,
        "strategy_profiles_count": strategy_count,
    }


@router.get("/models")
async def get_models_status():
    ctx = _get_ctx()
    try:
        models_info = ctx._model_manager.list_models() if hasattr(ctx, "_model_manager") else []
        return {"status": "ok", "models": models_info}
    except (ConnectionError, OSError, TimeoutError) as exc:
        return {"status": "error", "message": str(exc)}
