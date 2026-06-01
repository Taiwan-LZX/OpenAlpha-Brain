from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from openalpha_brain.config.config import settings

logger = logging.getLogger(__name__)

STREAM_THRESHOLD = getattr(settings, "ALPHA_CHANNEL_STREAM_THRESHOLD", 1.0)
BATCH_SIZE = getattr(settings, "ALPHA_CHANNEL_BATCH_SIZE", 5)
BATCH_TIMEOUT = getattr(settings, "ALPHA_CHANNEL_BATCH_TIMEOUT", 30.0)


class AlphaChannel:
    def __init__(
        self,
        stream_threshold: float = STREAM_THRESHOLD,
        batch_size: int = BATCH_SIZE,
        batch_timeout: float = BATCH_TIMEOUT,
    ) -> None:
        self._stream_threshold = stream_threshold
        self._batch_size = batch_size
        self._batch_timeout = batch_timeout
        self._buffer: list[dict[str, Any]] = []
        self._buffer_created_at: float = time.monotonic()
        self._streamed: int = 0
        self._batched: int = 0
        self._sharpe_stream_sum: float = 0.0
        self._sharpe_batch_sum: float = 0.0

    async def submit(self, alpha_id: str, sharpe: float, expression: str, direction: str) -> str:
        if sharpe > self._stream_threshold:
            self._streamed += 1
            self._sharpe_stream_sum += sharpe
            return "stream"
        self._buffer.append(
            {
                "alpha_id": alpha_id,
                "sharpe": sharpe,
                "expression": expression,
                "direction": direction,
            }
        )
        self._batched += 1
        self._sharpe_batch_sum += sharpe
        if len(self._buffer) == 1:
            self._buffer_created_at = time.monotonic()
        return "batch"

    async def get_batch(self) -> list[dict[str, Any]]:
        if not self._buffer:
            return []
        size_ok = len(self._buffer) >= self._batch_size
        timeout_ok = (time.monotonic() - self._buffer_created_at) >= self._batch_timeout
        if size_ok or timeout_ok:
            batch = list(self._buffer)
            self._buffer.clear()
            self._buffer_created_at = time.monotonic()
            return batch
        return []

    def get_stats(self) -> dict[str, Any]:
        avg_stream = (self._sharpe_stream_sum / self._streamed) if self._streamed > 0 else 0.0
        avg_batch = (self._sharpe_batch_sum / self._batched) if self._batched > 0 else 0.0
        return {
            "streamed": self._streamed,
            "batched": self._batched,
            "buffer_size": len(self._buffer),
            "avg_sharpe_stream": avg_stream,
            "avg_sharpe_batch": avg_batch,
        }

    def health_check(self) -> dict[str, Any]:
        return {
            "module": "AlphaChannel",
            "status": "active",
            "stream_threshold": self._stream_threshold,
            "batch_size": self._batch_size,
            "buffer_size": len(self._buffer),
            "total_streamed": self._streamed,
            "total_batched": self._batched,
        }

    async def drain(self) -> list[dict[str, Any]]:
        batch = list(self._buffer)
        self._buffer.clear()
        self._buffer_created_at = time.monotonic()
        if batch:
            logger.info("AlphaChannel drained: %d buffered alphas", len(batch))
        return batch

    async def shutdown(self) -> list[dict[str, Any]]:
        remaining = await self.drain()
        logger.info(
            "AlphaChannel shutdown: streamed=%d batched=%d buffered_remaining=%d avg_stream_sharpe=%.2f avg_batch_sharpe=%.2f",  # noqa: E501
            self._streamed,
            self._batched,
            len(remaining),
            self._sharpe_stream_sum / max(self._streamed, 1),
            self._sharpe_batch_sum / max(self._batched, 1),
        )
        return remaining


class AlphaChannelIntegrator:
    def __init__(
        self,
        channel: AlphaChannel,
        mab_update_fn: Callable[..., Awaitable[None]],
        whitelist_update_fn: Callable[..., Awaitable[None]],
        success_lib_fn: Callable[..., Awaitable[None]],
    ) -> None:
        self._channel = channel
        self._mab_update_fn = mab_update_fn
        self._whitelist_update_fn = whitelist_update_fn
        self._success_lib_fn = success_lib_fn

    async def process_stream_alpha(self, alpha_result: dict[str, Any]) -> None:
        direction = alpha_result.get("direction", "unknown")
        expression = alpha_result.get("expression", "")
        sharpe = alpha_result.get("sharpe", 0.0)
        reward = sharpe
        await asyncio.gather(
            self._mab_update_fn(direction=direction, expression=expression, reward=reward),
            self._whitelist_update_fn(expression=expression, reward=reward),
            self._success_lib_fn(expression=expression, direction=direction, sharpe=sharpe),
        )

    async def process_batch_alphas(self, alphas: list[dict[str, Any]]) -> None:
        if not alphas:
            return
        total_sharpe = sum(a.get("sharpe", 0.0) for a in alphas)
        weighted_reward = total_sharpe / len(alphas)
        directions = {a.get("direction", "unknown") for a in alphas}
        expressions = [a.get("expression", "") for a in alphas if a.get("expression")]
        for direction in directions:
            await self._mab_update_fn(direction=direction, expression=";".join(expressions), reward=weighted_reward)
        for expr in expressions:
            await self._whitelist_update_fn(expression=expr, reward=weighted_reward)
        for alpha in alphas:
            await self._success_lib_fn(
                expression=alpha.get("expression", ""),
                direction=alpha.get("direction", "unknown"),
                sharpe=alpha.get("sharpe", 0.0),
            )
