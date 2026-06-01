"""
OpenAlpha-Brain — Algorithm Instrumentation Logger

Provides structured logging decorators and utilities for recording
inputs, outputs, and timing of every core algorithm function.

Usage:
    from openalpha_brain.utils.algo_logger import algo_log, log_call, Timer

    @algo_log(level=logging.DEBUG)
    def my_function(arg1: str, arg2: int) -> dict:
        ...

    # Or manual instrumentation inside a function:
    with Timer("expensive_operation"):
        ...
    log_call("my_step", input={"x": x}, output={"result": y}, elapsed_ms=12.3)
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

logger = logging.getLogger(__name__)

_MAX_STR_LEN = 200
_MAX_LIST_LEN = 5


def _truncate(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value[:_MAX_STR_LEN] + ("..." if len(value) > _MAX_STR_LEN else "")
    if isinstance(value, (list, tuple)):
        truncated = [_truncate(v) for v in value[:_MAX_LIST_LEN]]
        if len(value) > _MAX_LIST_LEN:
            truncated.append(f"... (+{len(value) - _MAX_LIST_LEN} more)")
        return truncated
    if isinstance(value, dict):
        return {k: _truncate(v) for k, v in list(value.items())[:_MAX_LIST_LEN]}
    if isinstance(value, (int, float, bool)):
        return value
    return str(value)[:_MAX_STR_LEN]


def _safe_repr(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for i, v in enumerate(args):
        try:
            result[f"arg{i}"] = _truncate(v)
        except (OSError, ValueError, RuntimeError):
            result[f"arg{i}"] = "<unrepresentable>"
    for k, v in kwargs.items():
        if k not in ("self", "cls"):
            try:
                result[k] = _truncate(v)
            except (OSError, ValueError, RuntimeError):
                result[k] = "<unrepresentable>"
    return result


def log_call(
    func_name: str,
    *,
    input: dict[str, Any] | None = None,
    output: Any = None,
    elapsed_ms: float | None = None,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
    level: int = logging.DEBUG,
) -> None:
    msg_parts = [f"[ALGO] {func_name}"]
    if elapsed_ms is not None:
        msg_parts.append(f"{elapsed_ms:.1f}ms")
    if error:
        msg_parts.append(f"ERROR: {error}")
    else:
        msg_parts.append("OK")

    log_data: dict[str, Any] = {}
    if input is not None:
        log_data["input"] = input
    if output is not None and error is None:
        log_data["output"] = _truncate(output)
    if error:
        log_data["error"] = error
    if extra:
        log_data.update(extra)

    logger.log(level, " ".join(msg_parts), extra={"algo_data": log_data})


class Timer:
    __slots__ = ("_extra", "_label", "_start")

    def __init__(self, label: str, **extra: Any) -> None:
        self._label = label
        self._extra = extra
        self._start: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        logger.debug("[TIMER_START] %s", self._label)
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        log_call(
            self._label,
            elapsed_ms=elapsed_ms,
            error=None if args[0] is None else str(args[1]),
            extra=self._extra if self._extra else None,
        )


def algo_log(
    *,
    level: int = logging.DEBUG,
    log_input: bool = True,
    log_output: bool = True,
    log_error: bool = True,
    log_args_to_skip: Sequence[str] | None = None,
    label: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    skip = set(log_args_to_skip or ())

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func_label = label or f"{func.__module__}.{func.__qualname__}"

        if asyncio and inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                t0 = time.perf_counter()
                inp = None
                if log_input:
                    try:
                        inp = _safe_repr(args, {k: v for k, v in kwargs.items() if k not in skip})
                    except (OSError, ValueError, RuntimeError):
                        inp = {"_parse_error": True}
                try:
                    result = await func(*args, **kwargs)
                    elapsed = (time.perf_counter() - t0) * 1000
                    out = None
                    if log_output:
                        try:
                            out = _truncate(result)
                        except (OSError, ValueError, RuntimeError):
                            out = "<unrepresentable>"
                    log_call(func_label, input=inp, output=out, elapsed_ms=elapsed, level=level)
                    return result
                except Exception as e:
                    elapsed = (time.perf_counter() - t0) * 1000
                    if log_error:
                        log_call(func_label, input=inp, elapsed_ms=elapsed, error=str(e), level=logging.WARNING)
                    raise

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            inp = None
            if log_input:
                try:
                    inp = _safe_repr(args, {k: v for k, v in kwargs.items() if k not in skip})
                except (OSError, ValueError, RuntimeError):
                    inp = {"_parse_error": True}
            try:
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - t0) * 1000
                out = None
                if log_output:
                    try:
                        out = _truncate(result)
                    except (OSError, ValueError, RuntimeError):
                        out = "<unrepresentable>"
                log_call(func_label, input=inp, output=out, elapsed_ms=elapsed, level=level)
                return result
            except Exception as e:
                elapsed = (time.perf_counter() - t0) * 1000
                if log_error:
                    log_call(func_label, input=inp, elapsed_ms=elapsed, error=str(e), level=logging.WARNING)
                raise

        return sync_wrapper

    return decorator


try:
    import asyncio
except ImportError:
    asyncio = None
