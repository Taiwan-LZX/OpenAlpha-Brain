"""
OpenAlpha - Quant — Zombie Session Heartbeat Detection
Passive timeout mechanism: record last activity timestamp,
background task scans periodically and marks timed-out sessions as CRASHED.
"""

from __future__ import annotations

import asyncio
import logging
import time

from openalpha_brain.core.models import SessionStatus

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = frozenset(
    {
        SessionStatus.STOPPED,
        SessionStatus.ERROR,
        SessionStatus.CRASHED,
    }
)


class SessionHeartbeat:
    def __init__(
        self,
        timeout_seconds: int = 300,
        scan_interval_seconds: int = 60,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._scan_interval_seconds = scan_interval_seconds
        self._activity_times: dict[str, float] = {}
        self._background_task: asyncio.Task | None = None

    def touch(self, session_id: str) -> None:
        self._activity_times[session_id] = time.monotonic()

    def remove(self, session_id: str) -> None:
        self._activity_times.pop(session_id, None)

    async def scan_and_mark(self, session_manager) -> None:
        from openalpha_brain.cli import session_manager as sm

        now = time.monotonic()
        all_ids = await sm.list_sessions()

        for sid in all_ids:
            state = await sm.load_session(sid)
            if state is None:
                continue

            if state.status in _TERMINAL_STATUSES:
                self._activity_times.pop(sid, None)
                continue

            last_activity = self._activity_times.get(sid)
            if last_activity is None:
                continue

            elapsed = now - last_activity
            if elapsed > self._timeout_seconds:
                logger.warning(
                    "[%s] Heartbeat timeout: no activity for %.0fs (threshold=%ds) — marking CRASHED",
                    sid,
                    elapsed,
                    self._timeout_seconds,
                )
                state.status = SessionStatus.CRASHED
                state.error_message = (
                    f"heartbeat_timeout: no activity for {int(elapsed)}s (threshold={self._timeout_seconds}s)"
                )
                await sm.save_session(state)
                self._activity_times.pop(sid, None)

    async def startup_scan(self, session_manager) -> None:
        from openalpha_brain.cli import session_manager as sm

        all_ids = await sm.list_sessions()
        crashed_count = 0

        for sid in all_ids:
            state = await sm.load_session(sid)
            if state is None:
                continue

            if state.status not in _TERMINAL_STATUSES:
                logger.info(
                    "[%s] Startup scan: session in non-terminal state '%s' — marking CRASHED",
                    sid,
                    state.status.value,
                )
                state.status = SessionStatus.CRASHED
                state.error_message = "heartbeat_timeout: stale session from previous run"
                await sm.save_session(state)
                crashed_count += 1

        if crashed_count:
            logger.info("Startup scan: marked %d stale session(s) as CRASHED", crashed_count)
        else:
            logger.info("Startup scan: no stale sessions found")

    def get_session_health(self) -> dict:
        now = time.monotonic()
        result = {}
        for sid, last_activity in self._activity_times.items():
            elapsed = now - last_activity
            result[sid] = {
                "session_id": sid,
                "last_activity_ago_seconds": round(elapsed, 1),
                "is_alive": elapsed <= self._timeout_seconds,
                "timeout_threshold_seconds": self._timeout_seconds,
            }
        return result

    def start_background_task(self) -> None:
        if self._background_task is not None and not self._background_task.done():
            logger.warning("Heartbeat background task already running")
            return

        async def _scan_loop() -> None:
            logger.info(
                "Heartbeat background scan started (interval=%ds, timeout=%ds)",
                self._scan_interval_seconds,
                self._timeout_seconds,
            )
            while True:
                await asyncio.sleep(self._scan_interval_seconds)
                try:
                    await self.scan_and_mark(None)
                except Exception:
                    logger.error("Heartbeat scan failed", exc_info=True)

        self._background_task = asyncio.create_task(
            _scan_loop(),
            name="heartbeat-scan",
        )

    def stop_background_task(self) -> None:
        if self._background_task is not None and not self._background_task.done():
            self._background_task.cancel()
            logger.info("Heartbeat background scan stopped")
        self._background_task = None
