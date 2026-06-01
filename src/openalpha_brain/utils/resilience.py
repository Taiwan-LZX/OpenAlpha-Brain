from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_CIRCUIT_TIMEOUT = 60.0
DEFAULT_FAILURE_THRESHOLD = 5
DEFAULT_RECOVERY_TIMEOUT = 30.0
DEFAULT_HALF_OPEN_MAX = 1


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerStats:
    state: CircuitState
    failure_count: int
    success_count: int
    last_failure_time: float | None
    last_failure_reason: str
    opened_at: float | None
    total_failures: int
    total_successes: int


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        recovery_timeout: float = DEFAULT_RECOVERY_TIMEOUT,
        half_open_max: int = DEFAULT_HALF_OPEN_MAX,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._last_failure_reason = ""
        self._opened_at: float | None = None
        self._total_failures = 0
        self._total_successes = 0
        self._half_open_count = 0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if self._opened_at is not None and time.monotonic() - self._opened_at >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_count = 0
                logger.info("[CircuitBreaker:%s] OPEN -> HALF_OPEN (recovery timeout elapsed)", self.name)
        return self._state

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    @property
    def is_half_open(self) -> bool:
        return self.state == CircuitState.HALF_OPEN

    def record_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_count += 1
            if self._half_open_count >= self.half_open_max:
                self._reset()
                logger.info("[CircuitBreaker:%s] HALF_OPEN -> CLOSED (recovered)", self.name)
                return

        self._success_count += 1
        self._total_successes += 1

        if self._state == CircuitState.CLOSED:
            self._failure_count = 0

    def record_failure(self, reason: str = "") -> None:
        self._failure_count += 1
        self._total_failures += 1
        self._last_failure_time = time.monotonic()
        self._last_failure_reason = reason[:200]

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            logger.warning(
                "[CircuitBreaker:%s] HALF_OPEN -> OPEN (trial failed): %s",
                self.name,
                reason[:100],
            )
            return

        if self._state == CircuitState.CLOSED and self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            logger.warning(
                "[CircuitBreaker:%s] CLOSED -> OPEN (%d consecutive failures): %s",
                self.name,
                self._failure_count,
                reason[:100],
            )

    def _reset(self) -> None:
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at = None
        self._half_open_count = 0

    def get_stats(self) -> CircuitBreakerStats:
        return CircuitBreakerStats(
            state=self.state,
            failure_count=self._failure_count,
            success_count=self._success_count,
            last_failure_time=self._last_failure_time,
            last_failure_reason=self._last_failure_reason,
            opened_at=self._opened_at,
            total_failures=self._total_failures,
            total_successes=self._total_successes,
        )


class _CircuitOpenError(Exception):
    pass


_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str,
    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
    recovery_timeout: float = DEFAULT_RECOVERY_TIMEOUT,
) -> CircuitBreaker:
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )
    return _circuit_breakers[name]


async def async_timeout(coro: Awaitable[T], timeout_seconds: float, name: str = "") -> T:
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except TimeoutError:
        logger.warning("[Timeout:%s] Operation timed out after %.1fs", name, timeout_seconds)
        raise


async def with_async_retry(
    coro_fn: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    circuit_breaker: CircuitBreaker | None = None,
    retry_on: tuple[type[Exception], ...] = (Exception,),
    name: str = "",
) -> T:
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        if circuit_breaker is not None and circuit_breaker.is_open:
            raise _CircuitOpenError(
                f"Circuit breaker [{circuit_breaker.name}] is OPEN: {circuit_breaker._last_failure_reason}",
            )

        try:
            result = await coro_fn()
            if circuit_breaker is not None:
                circuit_breaker.record_success()
            return result
        except _CircuitOpenError:
            raise
        except retry_on as exc:
            last_exc = exc
            if circuit_breaker is not None:
                circuit_breaker.record_failure(str(exc))
            if attempt >= max_retries:
                logger.warning("[Retry:%s] Exhausted %d retries: %s", name, max_retries, exc)
                raise

            delay = min(base_delay * (2**attempt), max_delay)
            logger.warning(
                "[Retry:%s] Attempt %d/%d failed, retrying in %.1fs: %s",
                name,
                attempt + 1,
                max_retries,
                delay,
                exc,
            )
            await asyncio.sleep(delay)

    assert last_exc is not None
    raise last_exc


@dataclass
class TaskHealth:
    name: str
    last_run_time: float | None = None
    last_success_time: float | None = None
    last_error: str = ""
    consecutive_failures: int = 0
    total_runs: int = 0
    total_failures: int = 0

    def record_success(self) -> None:
        now = time.monotonic()
        self.last_run_time = now
        self.last_success_time = now
        self.consecutive_failures = 0
        self.total_runs += 1

    def record_failure(self, error: str = "") -> None:
        now = time.monotonic()
        self.last_run_time = now
        self.last_error = error[:200]
        self.consecutive_failures += 1
        self.total_runs += 1
        self.total_failures += 1

    @property
    def is_healthy(self) -> bool:
        return self.consecutive_failures < 3

    @property
    def idle_seconds(self) -> float:
        if self.last_run_time is None:
            return float("inf")
        return time.monotonic() - self.last_run_time


class TaskHealthRegistry:
    _tasks: dict[str, TaskHealth] = {}

    @classmethod
    def get(cls, name: str) -> TaskHealth:
        if name not in cls._tasks:
            cls._tasks[name] = TaskHealth(name=name)
        return cls._tasks[name]

    @classmethod
    def get_summary(cls) -> dict[str, Any]:
        return {
            name: {
                "consecutive_failures": task.consecutive_failures,
                "is_healthy": task.is_healthy,
                "idle_seconds": task.idle_seconds,
                "total_runs": task.total_runs,
                "total_failures": task.total_failures,
                "last_error": task.last_error[:100],
            }
            for name, task in cls._tasks.items()
        }
