from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import WebSocket

from openalpha_brain.core.events import AlphaEvent, AlphaEventBus
from openalpha_brain.cli.event_adapters import EventAdapterFactory
from openalpha_brain.cli.ws_renderer import get_ws_renderer, render_to_json

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

    async def broadcast(self, event_data: dict[str, Any]) -> None:
        if not self._connections:
            return

        message = json.dumps(
            event_data,
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

        adapter = EventAdapterFactory.create(event)
        renderer = get_ws_renderer()
        rendered_data = renderer.render(adapter)

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self.broadcast(rendered_data),
            )
        except RuntimeError:
            asyncio.run(
                self.broadcast(rendered_data),
            )


ws_broadcaster = WSBroadcaster()
