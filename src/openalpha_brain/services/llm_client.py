"""
OpenAlpha - Quant — LLM API Client
Supports Anthropic (default) and OpenAI via httpx.AsyncClient.
All retries use tenacity with exponential backoff.
API key is never logged.
"""
from __future__ import annotations

import asyncio
import logging
import time as _time
from collections.abc import Callable
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from openalpha_brain.config.config import settings

_provider_semaphores: dict[str, asyncio.Semaphore] = {}
_semaphores_lock = asyncio.Lock()

_embed_semaphore: asyncio.Semaphore | None = None


async def _get_provider_semaphore(provider: str) -> asyncio.Semaphore:
    if provider not in _provider_semaphores:
        async with _semaphores_lock:
            if provider not in _provider_semaphores:
                _provider_semaphores[provider] = asyncio.Semaphore(settings.LLM_MAX_CONCURRENT)
    return _provider_semaphores[provider]


def _get_embed_semaphore() -> asyncio.Semaphore:
    global _embed_semaphore
    if _embed_semaphore is None:
        _embed_semaphore = asyncio.Semaphore(settings.EMBED_MAX_CONCURRENT)
    return _embed_semaphore

from openalpha_brain.services.http_pool import get_client

logger = logging.getLogger(__name__)

# HTTP status codes that warrant a retry
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}

# Groq is OpenAI-compatible — just a different base URL
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
LMSTUDIO_BASE_URL = "http://localhost:1234/v1/chat/completions"

_embed_model_loaded: bool = False
_embed_load_lock = asyncio.Lock()


async def _ensure_embed_model_loaded() -> None:
    global _embed_model_loaded
    if _embed_model_loaded:
        return
    async with _embed_load_lock:
        if _embed_model_loaded:
            return
        _embed_model_loaded = True


async def embed(text: str) -> list[float]:
    sem = _get_embed_semaphore()
    if sem._value <= 0:
        logger.info("Embed semaphore FULL — waiting for slot (avail=%d max=%d)", sem._value, settings.EMBED_MAX_CONCURRENT)
    async with sem:
        await _ensure_embed_model_loaded()
        headers = {
            "Authorization": f"Bearer {settings.LLM_API_KEY or 'lm-studio'}",
            "content-type": "application/json",
        }
        payload = {
            "model": settings.EMBED_MODEL,
            "input": text,
        }
        url = settings.EMBED_BASE_URL
        client = get_client()
        for attempt in range(3):
            try:
                resp = await client.post(url, headers=headers, json=payload, timeout=120.0)
                resp.raise_for_status()
                data = resp.json()
                return data["data"][0]["embedding"]
            except Exception as e:
                if attempt < 2:
                    logger.warning("Embed request failed (attempt %d/3): %s — retrying in 5s", attempt + 1, e)
                    await asyncio.sleep(5)
                else:
                    raise
        return []


async def embed_batch(texts: list[str]) -> list[list[float]]:
    sem = _get_embed_semaphore()
    if sem._value <= 0:
        logger.info("Embed batch semaphore FULL — waiting (avail=%d max=%d)", sem._value, settings.EMBED_MAX_CONCURRENT)
    async with sem:
        await _ensure_embed_model_loaded()
        headers = {
            "Authorization": f"Bearer {settings.LLM_API_KEY or 'lm-studio'}",
            "content-type": "application/json",
        }
        payload = {
            "model": settings.EMBED_MODEL,
            "input": texts,
        }
        url = settings.EMBED_BASE_URL
        client = get_client()
        for attempt in range(3):
            try:
                resp = await client.post(url, headers=headers, json=payload, timeout=120.0)
                resp.raise_for_status()
                data = resp.json()
                sorted_data = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in sorted_data]
            except Exception as e:
                if attempt < 2:
                    logger.warning("Embed batch request failed (attempt %d/3): %s — retrying in 5s", attempt + 1, e)
                    await asyncio.sleep(5)
                else:
                    raise
        return []


class LLMError(Exception):
    """Raised when the LLM fails permanently after all retries."""
    def __init__(self, message: str, cycle: int = 0, session_id: str = ""):
        full_msg = f"[session={session_id} cycle={cycle}] {message}"
        super().__init__(full_msg)
        self.cycle = cycle
        self.session_id = session_id


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in _RETRYABLE_STATUSES:
            return True
        if status == 400:
            body = ""
            try:
                body = exc.response.text[:200]
            except (AttributeError, ValueError):
                pass
            if "model" in body.lower() or "load" in body.lower():
                return True
        return False
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    return False


async def generate(
    system_prompt: str,
    history: list[dict],
    user_msg: str,
    session_id: str = "",
    cycle: int = 0,
    response_format: dict | None = None,
    grammar: str | None = None,
) -> str:
    """
    Call the configured LLM and return the assistant text response.

    Args:
        system_prompt: The full IQC researcher system prompt (not in history).
        history: List of {role, content} dicts from prior turns.
        user_msg: The new user message for this call.
        session_id: For logging context only.
        cycle: For logging and error context only.
        grammar: GBNF grammar string for constrained decoding (LM Studio only).

    Returns:
        Raw assistant response string.

    Raises:
        LLMError: On permanent failure after retries.
    """
    provider = settings.LLM_PROVIDER.lower()
    if grammar and provider != "lmstudio":
        logger.warning(
            "[%s] cycle=%d Grammar-constrained decoding only supported with LM Studio; ignoring for provider '%s'.",
            session_id, cycle, provider,
        )
    if not settings.LLM_API_KEY and provider != "lmstudio":
        raise LLMError(
            "LLM_API_KEY is not set. Add it to your .env file.",
            cycle=cycle,
            session_id=session_id,
        )
    sem = await _get_provider_semaphore(provider)
    if sem._value <= 0:
        logger.info(
            "[%s] cycle=%d LLM semaphore FULL — waiting for slot (avail=%d max=%d)",
            session_id, cycle, sem._value, settings.LLM_MAX_CONCURRENT,
        )
    async with sem:
        if provider == "anthropic":
            return await _call_anthropic(system_prompt, history, user_msg, session_id, cycle, response_format)
        elif provider == "openai":
            return await _call_openai(system_prompt, history, user_msg, session_id, cycle, response_format)
        elif provider == "groq":
            return await _call_groq(system_prompt, history, user_msg, session_id, cycle, response_format)
        elif provider == "lmstudio":
            return await _call_lmstudio(system_prompt, history, user_msg, session_id, cycle, response_format, grammar)
        elif provider == "gemini":
            return await _call_gemini(system_prompt, history, user_msg, session_id, cycle, response_format)
        else:
            raise LLMError(
                f"Unknown LLM_PROVIDER '{provider}'. Use 'anthropic', 'openai', 'groq', 'gemini', or 'lmstudio'.",
                cycle=cycle,
                session_id=session_id,
            )


async def generate_with_tools(
    system_prompt: str,
    history: list[dict],
    user_msg: str,
    session_id: str = "",
    cycle: int = 0,
    tools: list[dict] | None = None,
    tool_executor: Callable | None = None,
    grammar: str | None = None,
) -> tuple[str, dict]:
    import json as _json

    tool_results: dict[str, Any] = {}

    if not tools or not tool_executor:
        text = await generate(system_prompt, history, user_msg, session_id, cycle, grammar=grammar)
        return text, tool_results

    provider = settings.LLM_PROVIDER.lower()
    if provider not in ("lmstudio", "openai", "groq"):
        text = await generate(system_prompt, history, user_msg, session_id, cycle, grammar=grammar)
        return text, tool_results

    messages = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": user_msg},
    ]

    payload: dict[str, Any] = {
        "model": settings.LLM_MODEL,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "temperature": settings.LLM_TEMPERATURE,
        "messages": messages,
        "tools": tools,
    }
    if grammar is not None and provider == "lmstudio":
        payload["grammar"] = grammar

    if provider == "lmstudio":
        url = settings.LLM_BASE_URL or LMSTUDIO_BASE_URL
        headers = {
            "Authorization": f"Bearer {settings.LLM_API_KEY or 'lm-studio'}",
            "content-type": "application/json",
        }
    elif provider == "groq":
        url = GROQ_BASE_URL
        headers = {
            "Authorization": f"Bearer {settings.LLM_API_KEY}",
            "content-type": "application/json",
        }
    else:
        url = settings.LLM_BASE_URL or "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.LLM_API_KEY}",
            "content-type": "application/json",
        }

    sem = await _get_provider_semaphore(provider)
    async with sem:
        try:
            raw_text = await _post_with_retry(
                url=url,
                headers=headers,
                payload=payload,
                session_id=session_id,
                cycle=cycle,
            )
        except LLMError:
            logger.warning(
                "[%s] cycle=%d Tool call request failed, falling back to generate()",
                session_id, cycle,
            )
            text = await generate(system_prompt, history, user_msg, session_id, cycle, grammar=grammar)
            return text, tool_results

    try:
        data = _json.loads(raw_text)
    except json.JSONDecodeError:
        text = await generate(system_prompt, history, user_msg, session_id, cycle, grammar=grammar)
        return text, tool_results

    if "error" in data:
        logger.warning(
            "[%s] cycle=%d Tool call not supported by model, falling back to generate()",
            session_id, cycle,
        )
        text = await generate(system_prompt, history, user_msg, session_id, cycle, grammar=grammar)
        return text, tool_results

    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})

    tool_calls = message.get("tool_calls")

    if not tool_calls:
        content = message.get("content", "").strip()
        return content, tool_results

    messages.append(message)

    for tc in tool_calls:
        tc_id = tc.get("id", "")
        fn_info = tc.get("function", {})
        tool_name = fn_info.get("name", "")
        arguments_str = fn_info.get("arguments", "{}")

        try:
            arguments = _json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning(
                "[%s] cycle=%d Tool call arguments JSON parse failed for %s: %s",
                session_id, cycle, tool_name, exc,
            )
            arguments = {}

        try:
            result = await tool_executor(tool_name, arguments)
            tool_results[tool_name] = result
        except (OSError, ValueError, RuntimeError) as exc:
            logger.warning(
                "[%s] cycle=%d Tool execution failed for %s: %s",
                session_id, cycle, tool_name, exc,
            )
            result = {"error": str(exc)}
            tool_results[tool_name] = result

        messages.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "content": _json.dumps(result),
        })

    payload_followup: dict[str, Any] = {
        "model": settings.LLM_MODEL,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "temperature": settings.LLM_TEMPERATURE,
        "messages": messages,
    }

    try:
        sem = await _get_provider_semaphore(provider)
        async with sem:
            raw_text2 = await _post_with_retry(
                url=url,
                headers=headers,
                payload=payload_followup,
                session_id=session_id,
                cycle=cycle,
            )
        data2 = _json.loads(raw_text2)
        final_content = data2.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return final_content, tool_results
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        logger.warning(
            "[%s] cycle=%d Follow-up after tool calls failed: %s",
            session_id, cycle, exc,
        )
        text = await generate(system_prompt, history, user_msg, session_id, cycle, grammar=grammar)
        return text, tool_results


async def _call_anthropic(
    system_prompt: str,
    history: list[dict],
    user_msg: str,
    session_id: str,
    cycle: int,
    response_format: dict | None = None,
) -> str:
    """Call Anthropic /v1/messages endpoint."""
    messages = [*history, {"role": "user", "content": user_msg}]

    payload: dict[str, Any] = {
        "model": settings.LLM_MODEL,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "temperature": settings.LLM_TEMPERATURE,
        "system": system_prompt,
        "messages": messages,
    }

    headers = {
        "x-api-key": settings.LLM_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    raw_text = await _post_with_retry(
        url="https://api.anthropic.com/v1/messages",
        headers=headers,
        payload=payload,
        session_id=session_id,
        cycle=cycle,
    )

    import json
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Anthropic response JSON parse error: {exc}", cycle=cycle, session_id=session_id) from exc

    content_blocks = data.get("content", [])
    text = " ".join(
        block.get("text", "") for block in content_blocks if block.get("type") == "text"
    )

    # Warn if the model was cut off mid-response
    stop_reason = data.get("stop_reason", "")
    if stop_reason == "max_tokens":
        logger.warning(
            "[%s] cycle=%d Anthropic stop_reason=max_tokens — consider raising LLM_MAX_TOKENS",
            session_id, cycle,
        )

    return text.strip()


async def _call_openai(
    system_prompt: str,
    history: list[dict],
    user_msg: str,
    session_id: str,
    cycle: int,
    response_format: dict | None = None,
) -> str:
    """Call OpenAI /v1/chat/completions endpoint."""
    messages = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": user_msg},
    ]

    payload: dict[str, Any] = {
        "model": settings.LLM_MODEL,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "temperature": settings.LLM_TEMPERATURE,
        "messages": messages,
    }
    if response_format is not None:
        payload["response_format"] = response_format

    headers = {
        "Authorization": f"Bearer {settings.LLM_API_KEY}",
        "content-type": "application/json",
    }

    raw_text = await _post_with_retry(
        url=settings.LLM_BASE_URL or "https://api.openai.com/v1/chat/completions",
        headers=headers,
        payload=payload,
        session_id=session_id,
        cycle=cycle,
    )

    import json
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"OpenAI response JSON parse error: {exc}", cycle=cycle, session_id=session_id) from exc
    choice = data.get("choices", [{}])[0]

    finish_reason = choice.get("finish_reason", "")
    if finish_reason == "length":
        logger.warning(
            "[%s] cycle=%d OpenAI finish_reason=length — consider raising LLM_MAX_TOKENS",
            session_id, cycle,
        )

    return choice.get("message", {}).get("content", "").strip()


async def _call_groq(
    system_prompt: str,
    history: list[dict],
    user_msg: str,
    session_id: str,
    cycle: int,
    response_format: dict | None = None,
) -> str:
    """
    Call Groq /openai/v1/chat/completions endpoint.
    Groq is fully OpenAI-compatible — only the base URL and auth header differ.
    Free tier: 30 RPM, 14,400 req/day. Recommended model: llama-3.3-70b-versatile.
    Get key at: https://console.groq.com/keys
    """
    messages = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": user_msg},
    ]
    payload: dict[str, Any] = {
        "model": settings.LLM_MODEL,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "temperature": settings.LLM_TEMPERATURE,
        "messages": messages,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    headers = {
        "Authorization": f"Bearer {settings.LLM_API_KEY}",
        "content-type": "application/json",
    }
    raw_text = await _post_with_retry(
        url=GROQ_BASE_URL,
        headers=headers,
        payload=payload,
        session_id=session_id,
        cycle=cycle,
    )
    import json
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Groq response JSON parse error: {exc}", cycle=cycle, session_id=session_id) from exc
    choice = data.get("choices", [{}])[0]
    finish_reason = choice.get("finish_reason", "")
    if finish_reason == "length":
        logger.warning(
            "[%s] cycle=%d Groq finish_reason=length — consider raising LLM_MAX_TOKENS",
            session_id, cycle,
        )
    return choice.get("message", {}).get("content", "").strip()


async def _warmup_lmstudio_model(
    url: str,
    headers: dict,
    session_id: str,
    cycle: int,
) -> None:
    warmup_payload: dict[str, Any] = {
        "model": settings.LLM_MODEL,
        "max_tokens": 1,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": "hi"}],
    }
    for attempt in range(3):
        try:
            client = get_client()
            resp = await client.post(url, headers=headers, json=warmup_payload, timeout=120.0)
            if resp.status_code == 400:
                body = resp.text[:200].lower()
                if "model unloaded" in body or ("model" in body and "load" in body):
                    logger.info(
                        "[%s] cycle=%d LM Studio warm-up: model unloaded (attempt %d/3), waiting 30s for reload",
                        session_id, cycle, attempt + 1,
                    )
                    await asyncio.sleep(30)
                    continue
            logger.info(
                "[%s] cycle=%d LM Studio model warm-up succeeded (attempt %d/3)",
                session_id, cycle, attempt + 1,
            )
            return
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            logger.info(
                "[%s] cycle=%d LM Studio warm-up connection error (attempt %d/3): %s, waiting 30s",
                session_id, cycle, attempt + 1, exc,
            )
            await asyncio.sleep(30)
            continue
        except (ConnectionError, OSError, TimeoutError, RuntimeError) as exc:
            logger.warning(
                "[%s] cycle=%d LM Studio warm-up unexpected error (attempt %d/3): %s",
                session_id, cycle, attempt + 1, exc,
            )
            return
    logger.warning(
        "[%s] cycle=%d LM Studio warm-up: model still not loaded after 3 attempts, proceeding with actual request",
        session_id, cycle,
    )


async def _call_lmstudio(
    system_prompt: str,
    history: list[dict],
    user_msg: str,
    session_id: str,
    cycle: int,
    response_format: dict | None = None,
    grammar: str | None = None,
) -> str:
    """
    Call LM Studio local server via OpenAI-compatible API.
    LM Studio exposes http://localhost:1234/v1/chat/completions by default.
    Uses LLM_BASE_URL if set, otherwise falls back to LMSTUDIO_BASE_URL.
    No real API key needed — LM Studio accepts any value.
    Falls back to native /api/v1/chat endpoint on connection error.
    Supports GBNF grammar-constrained decoding via the `grammar` field.
    """
    _start_time = _time.monotonic()
    messages = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": user_msg},
    ]
    effective_rf = None
    if response_format is not None:
        logger.info(
            "[%s] cycle=%d LM Studio does not support response_format, ignoring",
            session_id, cycle,
        )
    if grammar is not None:
        logger.info(
            "[%s] cycle=%d LM Studio does not support grammar, ignoring",
            session_id, cycle,
        )
    payload: dict[str, Any] = {
        "model": settings.LLM_MODEL,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "temperature": settings.LLM_TEMPERATURE,
        "messages": messages,
    }
    headers = {
        "Authorization": f"Bearer {settings.LLM_API_KEY or 'lm-studio'}",
        "content-type": "application/json",
    }
    url = settings.LLM_BASE_URL or LMSTUDIO_BASE_URL
    await _warmup_lmstudio_model(url, headers, session_id, cycle)
    try:
        raw_text = await _post_with_retry(
            url=url,
            headers=headers,
            payload=payload,
            session_id=session_id,
            cycle=cycle,
        )
    except LLMError:
        try:
            return await _call_lmstudio_native(
                system_prompt, history, user_msg, session_id, cycle,
            )
        except Exception as native_exc:
            logger.warning(
                "[%s] cycle=%d Native LM Studio fallback also failed: %s",
                session_id, cycle, native_exc,
            )
            raise
    import json
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"LM Studio response JSON parse error: {exc}", cycle=cycle, session_id=session_id) from exc
    choice = data.get("choices", [{}])[0]
    finish_reason = choice.get("finish_reason", "")
    if finish_reason == "length":
        logger.warning(
            "[%s] cycle=%d LM Studio finish_reason=length — consider raising LLM_MAX_TOKENS",
            session_id, cycle,
        )
    _elapsed = _time.monotonic() - _start_time
    logger.info(
        "[%s] cycle=%d LM Studio call took %.1fs (finish_reason=%s)",
        session_id, cycle, _elapsed, finish_reason,
    )
    message = choice.get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        content = content.strip()
    else:
        content = str(content).strip()
    if not content:
        reasoning_content = message.get("reasoning_content") or message.get("thinking_content") or message.get("reasoning")
        if reasoning_content:
            if isinstance(reasoning_content, str):
                content = reasoning_content.strip()
            elif isinstance(reasoning_content, list):
                parts = []
                for block in reasoning_content:
                    if isinstance(block, dict):
                        parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        parts.append(block)
                content = "\n".join(parts).strip()
            else:
                content = str(reasoning_content).strip()
            logger.info(
                "[%s] cycle=%d LM Studio content empty, used reasoning_content (%d chars)",
                session_id, cycle, len(content),
            )
    if not content:
        logger.warning(
            "[%s] cycle=%d LM Studio returned empty content — full message keys: %s | finish_reason=%s",
            session_id, cycle, list(message.keys()), finish_reason,
        )
        if effective_rf is not None or grammar is not None:
            logger.info(
                "[%s] cycle=%d Retrying LM Studio without response_format/grammar",
                session_id, cycle,
            )
            retry_payload: dict[str, Any] = {
                "model": settings.LLM_MODEL,
                "max_tokens": settings.LLM_MAX_TOKENS,
                "temperature": settings.LLM_TEMPERATURE,
                "messages": messages,
            }
            try:
                raw_retry = await _post_with_retry(
                    url=url,
                    headers=headers,
                    payload=retry_payload,
                    session_id=session_id,
                    cycle=cycle,
                )
                retry_data = json.loads(raw_retry)
                retry_choice = retry_data.get("choices", [{}])[0]
                retry_message = retry_choice.get("message", {})
                retry_content = retry_message.get("content", "")
                if isinstance(retry_content, str):
                    retry_content = retry_content.strip()
                else:
                    retry_content = str(retry_content).strip()
                if not retry_content:
                    retry_reasoning = retry_message.get("reasoning_content") or retry_message.get("thinking_content") or retry_message.get("reasoning")
                    if retry_reasoning:
                        if isinstance(retry_reasoning, str):
                            retry_content = retry_reasoning.strip()
                        elif isinstance(retry_reasoning, list):
                            parts = []
                            for block in retry_reasoning:
                                if isinstance(block, dict):
                                    parts.append(block.get("text", ""))
                                elif isinstance(block, str):
                                    parts.append(block)
                            retry_content = "\n".join(parts).strip()
                        else:
                            retry_content = str(retry_reasoning).strip()
                if retry_content:
                    logger.info(
                        "[%s] cycle=%d Retry without response_format/grammar succeeded (%d chars)",
                        session_id, cycle, len(retry_content),
                    )
                    return retry_content
                logger.warning(
                    "[%s] cycle=%d Retry also returned empty content — message keys: %s",
                    session_id, cycle, list(retry_message.keys()),
                )
            except (ValueError, TypeError, OSError, RuntimeError) as retry_exc:
                logger.warning(
                    "[%s] cycle=%d Retry without response_format/grammar failed: %s",
                    session_id, cycle, retry_exc,
                )
    if not content or not content.strip():
        raise LLMError(
            f"LM Studio returned empty content after all retries (finish_reason={finish_reason})",
            cycle=cycle,
            session_id=session_id,
        )
    return str(content)


async def _call_lmstudio_native(
    system_prompt: str,
    history: list[dict],
    user_msg: str,
    session_id: str,
    cycle: int,
) -> str:
    url = f"{settings.LMSTUDIO_API_BASE}/api/v1/chat"
    input_parts: list[str] = []
    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        input_parts.append(f"[{role}]: {content}")
    input_parts.append(f"[user]: {user_msg}")
    combined_input = "\n".join(input_parts)
    payload: dict[str, Any] = {
        "model": settings.LLM_MODEL,
        "system_prompt": system_prompt,
        "input": combined_input,
        "temperature": settings.LLM_TEMPERATURE,
        "max_tokens": settings.LLM_MAX_TOKENS,
    }
    headers = {
        "Authorization": f"Bearer {settings.LLM_API_KEY or 'lm-studio'}",
        "content-type": "application/json",
    }
    client = get_client()
    resp = await client.post(url, headers=headers, json=payload, timeout=90.0)
    resp.raise_for_status()
    data = resp.json()
    return data.get("output", [{}])[0].get("content", "").strip()


async def _call_gemini(
    system_prompt: str,
    history: list[dict],
    user_msg: str,
    session_id: str,
    cycle: int,
    response_format: dict | None = None,
) -> str:
    """
    Call Google Gemini generateContent REST endpoint.
    Free tier: 15 RPM, 1M tokens/day. Recommended model: gemini-1.5-flash.
    Get key at: https://aistudio.google.com/app/apikey
    """
    import json
    # Build Gemini contents list: system injected as first user turn
    contents = []
    # Gemini doesn't have a system role — inject system prompt as first user message
    contents.append({
        "role": "user",
        "parts": [{"text": system_prompt}],
    })
    contents.append({
        "role": "model",
        "parts": [{"text": "Understood. I am OpenAlpha - Quant, ready to conduct rigorous alpha research."}],
    })
    # Map prior history
    for turn in history:
        role = "model" if turn["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": turn["content"]}]})
    # Add current user message
    contents.append({"role": "user", "parts": [{"text": user_msg}]})

    payload = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": settings.LLM_MAX_TOKENS,
            "temperature": settings.LLM_TEMPERATURE,
        },
    }
    url = GEMINI_BASE_URL.format(model=settings.LLM_MODEL)
    headers = {"content-type": "application/json"}
    # Gemini auth: key in query param, not header
    url_with_key = f"{url}?key={settings.LLM_API_KEY}"

    raw_text = await _post_with_retry(
        url=url_with_key,
        headers=headers,
        payload=payload,
        session_id=session_id,
        cycle=cycle,
    )
    data = json.loads(raw_text)
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as exc:
        logger.error("[%s] cycle=%d Gemini response parse error: %s | raw: %s",
                     session_id, cycle, exc, raw_text[:200])
        raise LLMError(f"Gemini response parse error: {exc}", cycle=cycle, session_id=session_id)


async def _post_with_retry(
    url: str,
    headers: dict,
    payload: dict,
    session_id: str,
    cycle: int,
) -> str:
    """POST with up to 5 retries and exponential backoff on retryable errors."""
    import tenacity

    from openalpha_brain.utils.resilience import get_circuit_breaker
    _llm_cb = get_circuit_breaker("llm_api", failure_threshold=5, recovery_timeout=45.0)
    if _llm_cb.is_open:
        _llm_cb.record_failure("Circuit open - skipping LLM call")
        raise LLMError(
            f"LLM circuit breaker OPEN — skipping call. Last failure: {_llm_cb._last_failure_reason[:100]}",
            cycle=cycle,
            session_id=session_id,
        )

    last_exc: Exception | None = None

    try:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception(_is_retryable),
            wait=wait_exponential(multiplier=2, min=5, max=60),
            stop=stop_after_attempt(5),
            reraise=False,
        ):
            with attempt:
                client = get_client()
                resp = await client.post(url, headers=headers, json=payload, timeout=180.0)

                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    retry_after = exc.response.headers.get("Retry-After", "unknown")
                    logger.warning(
                        "[%s] cycle=%d HTTP %d from LLM — retry-after=%s",
                        session_id, cycle, status, retry_after,
                    )
                    last_exc = exc
                    _llm_cb.record_failure(f"HTTP {status}")
                    raise

                raw = resp.text
                logger.debug("[%s] cycle=%d LLM raw response: %s", session_id, cycle, raw)
                _llm_cb.record_success()
                return raw
    except tenacity.RetryError as retry_err:
        _llm_cb.record_failure(f"Retries exhausted: {last_exc}")
        raise LLMError(
            f"LLM call failed after 5 retries (last: {last_exc})",
            cycle=cycle,
            session_id=session_id,
        ) from retry_err

    raise LLMError(
        f"LLM call failed permanently. Last error: {last_exc}",
        cycle=cycle,
        session_id=session_id,
    )
