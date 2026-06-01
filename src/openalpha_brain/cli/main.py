"""
OpenAlpha - Quant — FastAPI Application
All routes return immediately. The generation loop runs as a background asyncio task.
"""

from __future__ import annotations

import asyncio
import logging
import logging.config
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from openalpha_brain.cli import session_manager as sm
from openalpha_brain.cli.api_alphas import router as alphas_router
from openalpha_brain.cli.api_config import router as config_router
from openalpha_brain.cli.api_status import router as status_router
from openalpha_brain.cli.ws_broadcaster import ws_broadcaster
from openalpha_brain.config.config import settings
from openalpha_brain.core import loop_engine
from openalpha_brain.core.events import AlphaEvent, get_event_bus
from openalpha_brain.core.models import SessionStatus, StartSessionRequest
from openalpha_brain.services.http_pool import close_client, get_client

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("openalpha")

# Track running loop tasks so we can check their status
_running_tasks: dict[str, asyncio.Task] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.SESSION_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("OpenAlpha - Quant started — sessions dir: %s", settings.SESSION_DIR)
    if not settings.LLM_API_KEY:
        logger.warning(
            "LLM_API_KEY is not set! Sessions will ERROR when the loop tries to call the LLM. "
            "Set it in your .env file.",
        )
    getattr(loop_engine, "init_intelligent_search", lambda: None)()

    ws_broadcaster.attach(get_event_bus())

    heartbeat = getattr(loop_engine, "_heartbeat", None)
    if heartbeat:
        await heartbeat.startup_scan(sm)
        heartbeat.start_background_task()

    yield

    ws_broadcaster.detach()

    if heartbeat:
        heartbeat.stop_background_task()

    for sid, task in list(_running_tasks.items()):
        if not task.done():
            task.cancel()
            logger.info("Cancelled running loop for session %s on shutdown", sid)
    await close_client()


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="OpenAlpha - Quant",
    version="1.0.0",
    description="Autonomous WorldQuant BRAIN Alpha Generation Engine — IQC 2026",
    lifespan=lifespan,
)

_DEV_MODE = os.getenv("DEV_MODE", "").lower() in ("1", "true", "yes")
_frontend_dist = Path(__file__).resolve().parent.parent.parent.parent / "frontend" / "dist"

_cors_origins = ["*"] if _DEV_MODE else ["http://localhost:8000", "http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(alphas_router)
app.include_router(config_router)
app.include_router(status_router)

if not _DEV_MODE and _frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(_frontend_dist / "assets")), name="frontend-assets")


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    await ws_broadcaster.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_broadcaster.disconnect(websocket)


@app.websocket("/ws/session/{session_id}")
async def websocket_session(websocket: WebSocket, session_id: str):
    await websocket.accept()
    queue: asyncio.Queue[AlphaEvent] = asyncio.Queue()
    _SESSION_EVENTS = frozenset(
        {
            "cycle_complete",
            "alpha_generated",
            "alpha_passed",
            "alpha_failed",
            "brain_submitted",
            "brain_result",
            "log",
            "session_complete",
        }
    )
    loop = asyncio.get_running_loop()

    def _on_session_event(event: AlphaEvent) -> None:
        if event.event_type not in _SESSION_EVENTS:
            return
        if event.data.get("session_id") != session_id:
            return
        loop.call_soon_threadsafe(queue.put_nowait, event)

    bus = get_event_bus()
    bus.subscribe(_on_session_event)
    try:
        while True:
            event = await queue.get()
            await websocket.send_json({"type": event.event_type, "data": event.data})
    except WebSocketDisconnect:
        pass
    finally:
        bus.unsubscribe(_on_session_event)


@app.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    await websocket.accept()
    queue: asyncio.Queue[AlphaEvent] = asyncio.Queue()
    _MONITOR_EVENTS = frozenset(
        {
            "brain_submit",
            "brain_result",
            "mab_update",
            "generator_update",
            "metrics_update",
        }
    )
    loop = asyncio.get_running_loop()

    def _on_monitor_event(event: AlphaEvent) -> None:
        if event.event_type not in _MONITOR_EVENTS:
            return
        loop.call_soon_threadsafe(queue.put_nowait, event)

    bus = get_event_bus()
    bus.subscribe(_on_monitor_event)
    try:
        while True:
            event = await queue.get()
            await websocket.send_json({"type": event.event_type, "data": event.data})
    except WebSocketDisconnect:
        pass
    finally:
        bus.unsubscribe(_on_monitor_event)


@app.get("/api/lmstudio-status")
async def lmstudio_status():
    try:
        client = get_client()
        resp = await client.get(
            f"{settings.LMSTUDIO_API_BASE}/v1/models",
            timeout=10.0,
        )
        models = resp.json()
        return {"status": "online", "models": models}
    except (ConnectionError, OSError, TimeoutError) as e:
        return {"status": "offline", "error": str(e)}


@app.get("/health")
async def health():
    return {"status": "ok", "sessions_active": len(_running_tasks)}


@app.get("/api/sessions/health")
async def sessions_health():
    heartbeat = loop_engine._heartbeat
    if heartbeat is None:
        return {"sessions": {}, "heartbeat_enabled": False}
    return {"sessions": heartbeat.get_session_health(), "heartbeat_enabled": True}


@app.get("/api/dashboard/algo-status")
async def algo_status():
    from openalpha_brain.cli.algo_monitor import AlgoMonitor

    return AlgoMonitor.get_instance().get_status()


@app.post("/session/start")
async def start_session(req: StartSessionRequest):
    """
    Create a new session and immediately fire the generation loop as a background task.
    Returns session_id without waiting for any LLM call.
    """
    state = await sm.create_session(focus_area=req.focus_area)
    sid = state.id

    # Launch loop as independent asyncio task — route returns immediately
    task = asyncio.create_task(
        _run_loop_safe(sid),
        name=f"loop-{sid}",
    )
    _running_tasks[sid] = task

    logger.info("Session %s started, focus_area=%r", sid, req.focus_area)
    return {"session_id": sid, "status": SessionStatus.IDLE}


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Return full session state snapshot with computed statistics."""
    state = await sm.load_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    data = state.model_dump(mode="json")

    # Add computed statistics for frontend
    passed_count = len(state.passed_alphas)
    failed_count = len(getattr(state, "failed_alphas", []))
    total_generated = passed_count + failed_count

    data["current_cycle"] = state.cycle
    data["total_cycles"] = getattr(state, "total_cycles", state.cycle)
    data["generated_alpha"] = total_generated
    data["submitted_brain"] = sum(1 for a in state.passed_alphas if a.brain and a.brain.alpha_id)
    data["passed_alpha"] = passed_count
    data["failed_alpha"] = failed_count
    data["pass_rate"] = round(passed_count / total_generated, 4) if total_generated > 0 else 0.0
    data["brain_slots"] = getattr(state, "brain_slots", {"used": 0, "total": 3})
    data["focus_area"] = state.focus_area
    data["status"] = state.status.value if hasattr(state.status, "value") else str(state.status)

    return data


@app.get("/session/{session_id}/alphas")
async def get_alphas(session_id: str):
    """Return only the list of passed alphas for the session."""
    state = await sm.load_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return {
        "session_id": session_id,
        "count": len(state.passed_alphas),
        "alphas": [a.model_dump(mode="json") for a in state.passed_alphas],
    }


@app.post("/session/{session_id}/stop")
async def stop_session(session_id: str):
    """
    Set stop_requested flag on the session.
    The loop checks this flag at the top of every cycle and exits cleanly.
    """
    found = await sm.request_stop(session_id)
    if not found:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return {"stopped": True, "session_id": session_id}


@app.post("/session/{session_id}/pause")
async def pause_session(session_id: str):
    from openalpha_brain.core import loop_state as _ls

    if _ls._console_pause_event is not None:
        _ls._console_pause_event.clear()
        return {"status": "paused", "session_id": session_id}
    return {"status": "error", "message": "No pause event available"}


@app.get("/api/sessions")
async def list_sessions():
    """Return list of all sessions with basic info."""
    session_ids = await sm.list_sessions()
    sessions = []
    for sid in session_ids:
        state = await sm.load_session(sid)
        if state is None:
            continue
        sessions.append(
            {
                "id": sid,
                "status": state.status.value if hasattr(state.status, "value") else str(state.status),
                "cycle": state.cycle,
                "focus_area": state.focus_area,
                "passed_count": len(state.passed_alphas),
                "failed_count": len(getattr(state, "failed_alphas", [])),
                "created_at": state.created_at.isoformat() if state.created_at else None,
                "updated_at": state.updated_at.isoformat() if state.updated_at else None,
            }
        )
    return {"sessions": sessions, "total": len(sessions)}


@app.post("/session/{session_id}/resume")
async def resume_session(session_id: str):
    from openalpha_brain.core import loop_state as _ls

    if _ls._console_pause_event is not None:
        _ls._console_pause_event.set()
        return {"status": "resumed", "session_id": session_id}
    return {"status": "error", "message": "No pause event available"}


# ── Serve dashboard at root ────────────────────────────────────────────────────
@app.get("/")
async def serve_frontend():
    from fastapi.responses import FileResponse, RedirectResponse

    if _DEV_MODE:
        return RedirectResponse(url="http://localhost:5173")
    if _frontend_dist.exists():
        return FileResponse(str(_frontend_dist / "index.html"))
    return {"message": "OpenAlpha - Quant API", "docs": "/docs"}


# ── Error handler ──────────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled exception: %s", exc, exc_info=True)

    # Return structured error response
    error_response = {
        "error": "internal_server_error",
        "detail": str(exc),
        "type": exc.__class__.__name__,
        "session_id": None,
        "cycle": None,
    }

    # Add request info for debugging
    if hasattr(request, "url"):
        error_response["path"] = str(request.url.path)
        error_response["method"] = request.method if hasattr(request, "method") else "UNKNOWN"

    return JSONResponse(
        status_code=500,
        content=error_response,
    )


# ── Safe loop wrapper ──────────────────────────────────────────────────────────
async def _run_loop_safe(session_id: str) -> None:
    """Wraps run_loop to catch any unhandled exceptions and mark session ERROR."""
    try:
        if settings.PIPELINE_MODE:
            await loop_engine.run_loop_pipeline(session_id)
        else:
            await loop_engine.run_loop(session_id)
    except asyncio.CancelledError:
        logger.info("[%s] Loop task cancelled", session_id)
        await sm.update_status(session_id, SessionStatus.STOPPED)
    except Exception as exc:
        logger.error("[%s] Unhandled loop exception: %s", session_id, exc, exc_info=True)
        state = await sm.load_session(session_id)
        if state:
            state.status = SessionStatus.ERROR
            state.error_message = str(exc)
            await sm.save_session(state)
    finally:
        heartbeat = getattr(loop_engine, "_heartbeat", None)
        if heartbeat:
            heartbeat.remove(session_id)
        _running_tasks.pop(session_id, None)
