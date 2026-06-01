from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import WebSocket

from openalpha_brain.core.events import AlphaEvent, AlphaEventBus

logger = logging.getLogger(__name__)


class WSBroadcaster:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._bus: AlphaEventBus | None = None

    @property
    def connections(self) -> set[WebSocket]:
        return self._connections

    def attach(self, bus: AlphaEventBus) -> None:
        if self._bus is not None:
            self._bus.unsubscribe(self._on_event)
        self._bus = bus
        bus.subscribe(self._on_event)

    def detach(self) -> None:
        if self._bus is not None:
            self._bus.unsubscribe(self._on_event)
            self._bus = None

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.add(websocket)
        logger.info("WebSocket client connected, total=%d", len(self._connections))

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.discard(websocket)
        logger.info("WebSocket client disconnected, total=%d", len(self._connections))

    async def broadcast(self, event_type: str, data: dict[str, Any], timestamp: float | None = None) -> None:
        if not self._connections:
            return

        # Map event types to frontend log modules
        module_map = {
            "cycle_complete": "System",
            "alpha_generated": "FactorAgent",
            "alpha_passed": "EvalAgent",
            "alpha_failed": "EvalAgent",
            "brain_submitted": "BRAIN",
            "brain_result": "BRAIN",
            "log": "System",
            "session_complete": "System",
            "brain_submit": "BRAIN",
            "mab_update": "MAB",
            "generator_update": "FactorAgent",
            "metrics_update": "System",
        }

        message = json.dumps(
            {
                "type": event_type,
                "module": module_map.get(event_type, "System"),
                "timestamp": (
                    datetime.fromtimestamp(timestamp, tz=UTC).isoformat()
                    if timestamp
                    else datetime.now(tz=UTC).isoformat()
                ),
                "data": data,
            },
            default=str,
            ensure_ascii=False,
        )
        stale: list[WebSocket] = []
        for ws in list(self._connections):
            try:
                await ws.send_text(message)
            except (OSError, ValueError, RuntimeError):
                stale.append(ws)
        for ws in stale:
            self._connections.discard(ws)

    def _on_event(self, event: AlphaEvent) -> None:
        if not self._connections:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self.broadcast(event.event_type, event.data, event.timestamp),
            )
        except RuntimeError:
            asyncio.run(
                self.broadcast(event.event_type, event.data, event.timestamp),
            )


ws_broadcaster = WSBroadcaster()
