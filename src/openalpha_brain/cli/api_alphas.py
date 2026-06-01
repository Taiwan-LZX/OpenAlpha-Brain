from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from openalpha_brain.cli import session_manager as sm

router = APIRouter(prefix="/api/alphas", tags=["alphas"])


def _alpha_to_dict(alpha) -> dict:
    direction = (
        alpha.fingerprint.direction
        if alpha.fingerprint and alpha.fingerprint.direction
        else alpha.exploration_direction
    )
    sharpe = None
    if alpha.brain and alpha.brain.real_sharpe is not None:
        sharpe = alpha.brain.real_sharpe
    elif alpha.metrics:
        if alpha.metrics.sharpe_min is not None and alpha.metrics.sharpe_max is not None:
            sharpe = (alpha.metrics.sharpe_min + alpha.metrics.sharpe_max) / 2
        elif alpha.metrics.sharpe_min is not None:
            sharpe = alpha.metrics.sharpe_min
        elif alpha.metrics.sharpe_max is not None:
            sharpe = alpha.metrics.sharpe_max

    fitness = None
    if alpha.brain and alpha.brain.real_fitness is not None:
        fitness = alpha.brain.real_fitness
    elif alpha.metrics and alpha.metrics.fitness_computed is not None:
        fitness = alpha.metrics.fitness_computed
    elif alpha.metrics:
        if alpha.metrics.fitness_min is not None and alpha.metrics.fitness_max is not None:
            fitness = (alpha.metrics.fitness_min + alpha.metrics.fitness_max) / 2
        elif alpha.metrics.fitness_min is not None:
            fitness = alpha.metrics.fitness_min
        elif alpha.metrics.fitness_max is not None:
            fitness = alpha.metrics.fitness_max

    turnover = None
    if alpha.brain and alpha.brain.real_turnover is not None:
        turnover = alpha.brain.real_turnover
    elif alpha.metrics:
        if alpha.metrics.turnover_min is not None and alpha.metrics.turnover_max is not None:
            turnover = (alpha.metrics.turnover_min + alpha.metrics.turnover_max) / 2
        elif alpha.metrics.turnover_min is not None:
            turnover = alpha.metrics.turnover_min
        elif alpha.metrics.turnover_max is not None:
            turnover = alpha.metrics.turnover_max

    status = "pass" if alpha.passed else "fail"

    return {
        "id": alpha.alpha_id,
        "expression": alpha.expression,
        "direction": direction,
        "sharpe": sharpe,
        "fitness": fitness,
        "turnover": turnover,
        "status": status,
        "submittedAt": alpha.timestamp.isoformat() if alpha.timestamp else None,
    }


@router.get("")
async def list_alphas(
    direction: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1),
    offset: int = Query(default=0, ge=0),
):
    session_ids = await sm.list_sessions()
    all_alphas: list[dict] = []
    for sid in session_ids:
        state = await sm.load_session(sid)
        if state is None:
            continue
        for alpha in state.passed_alphas:
            all_alphas.append(_alpha_to_dict(alpha))
        for alpha in getattr(state, "failed_alphas", []):
            all_alphas.append(_alpha_to_dict(alpha))

    if direction is not None:
        all_alphas = [a for a in all_alphas if a["direction"] == direction]
    if status is not None:
        all_alphas = [a for a in all_alphas if a["status"] == status]

    total = len(all_alphas)
    all_alphas = all_alphas[offset : offset + limit]

    return {"total": total, "alphas": all_alphas}


@router.get("/{alpha_id}")
async def get_alpha(alpha_id: str):
    session_ids = await sm.list_sessions()
    for sid in session_ids:
        state = await sm.load_session(sid)
        if state is None:
            continue
        for alpha in state.passed_alphas:
            if alpha.alpha_id == alpha_id:
                result = _alpha_to_dict(alpha)
                result["pnlData"] = []
                return result
        for alpha in getattr(state, "failed_alphas", []):
            if alpha.alpha_id == alpha_id:
                result = _alpha_to_dict(alpha)
                result["pnlData"] = []
                return result
    raise HTTPException(status_code=404, detail=f"Alpha '{alpha_id}' not found")
