"""
OpenAlpha - Quant — Session State Manager
All session I/O is async (aiofiles). No in-memory cache —
every read hits disk to ensure two concurrent sessions never share state.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import aiofiles

from openalpha_brain.config.config import settings
from openalpha_brain.core.models import AlphaResult, SessionState, SessionStatus

logger = logging.getLogger(__name__)

_session_locks: dict[str, asyncio.Lock] = {}


def _get_lock(session_id: str) -> asyncio.Lock:
    lock = _session_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _session_locks[session_id] = lock
    return lock


def _session_path(session_id: str) -> Path:
    return settings.SESSION_DIR / f"{session_id}.json"


def _serialize(state: SessionState) -> str:
    """Convert SessionState to JSON string with datetime handling."""
    return state.model_dump_json(indent=2)


def _deserialize(raw: str) -> SessionState:
    return SessionState.model_validate_json(raw)


async def create_session(focus_area: str = "") -> SessionState:
    """Create, persist, and return a new blank session."""
    session_id = uuid.uuid4().hex[:12]
    state = SessionState(
        id=session_id,
        focus_area=focus_area,
        status=SessionStatus.IDLE,
    )
    await save_session(state)
    logger.info("[%s] Session created, focus_area=%r", session_id, focus_area)
    return state


async def load_session(session_id: str) -> SessionState | None:
    """Load session from disk. Returns None if not found."""
    path = _session_path(session_id)
    if not path.exists():
        logger.warning("[%s] Session file not found at %s", session_id, path.resolve())
        return None
    try:
        async with aiofiles.open(path, encoding="utf-8") as f:
            raw = await f.read()
        return _deserialize(raw)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.error("[%s] Failed to load session: %s", session_id, exc)
        return None


async def save_session(state: SessionState) -> None:
    """Persist full session state to disk atomically with per-session locking."""
    lock = _get_lock(state.id)
    async with lock:
        state.updated_at = datetime.now(UTC)
        path = _session_path(state.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        try:
            async with aiofiles.open(tmp_path, "w", encoding="utf-8") as f:
                await f.write(_serialize(state))
            for _retry in range(3):
                try:
                    os.replace(tmp_path, path)
                    return
                except OSError:
                    if _retry < 2:
                        time.sleep(0.1 * (_retry + 1))
                    else:
                        try:
                            shutil.move(str(tmp_path), str(path))
                            return
                        except (OSError, PermissionError):
                            pass
            logger.warning("[%s] Failed to replace session file after retries, data written to %s", state.id, tmp_path)
        except Exception as exc:
            logger.error("[%s] Failed to save session: %s", state.id, exc)
            raise


async def update_status(session_id: str, status: SessionStatus) -> None:
    """Load → update status → save. Thin convenience wrapper."""
    state = await load_session(session_id)
    if state is None:
        logger.warning("[%s] update_status: session not found", session_id)
        return
    state.status = status
    await save_session(state)


async def add_alpha(session_id: str, alpha: AlphaResult) -> None:
    """Append a passed alpha to the session's passed_alphas list."""
    state = await load_session(session_id)
    if state is None:
        return
    state.passed_alphas.append(alpha)
    await save_session(state)


async def append_fingerprint(session_id: str, fingerprint: dict) -> None:
    """Add a structural fingerprint to the session's anti-crowding memory."""
    state = await load_session(session_id)
    if state is None:
        return
    state.fingerprint_memory.append(fingerprint)
    await save_session(state)


async def request_stop(session_id: str) -> bool:
    """Set stop_requested flag. Returns True if session exists."""
    state = await load_session(session_id)
    if state is None:
        return False
    state.stop_requested = True
    await save_session(state)
    logger.info("[%s] Stop requested", session_id)
    return True


async def list_sessions() -> list[str]:
    """Return all session IDs currently on disk."""
    sessions_dir = settings.SESSION_DIR
    if not sessions_dir.exists():
        return []
    return [p.stem for p in sessions_dir.glob("*.json")]
