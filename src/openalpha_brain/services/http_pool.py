"""
OpenAlpha-Brain — Shared httpx AsyncClient Pool
Provides a lazily-initialized, long-lived httpx.AsyncClient for reuse
across LLM and BRAIN API calls. Managed via FastAPI lifespan.
"""
from __future__ import annotations

import httpx

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _transport = httpx.AsyncHTTPTransport(proxy=None)
        _client = httpx.AsyncClient(
            transport=_transport,
            timeout=httpx.Timeout(120.0, connect=30.0),
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_expiry=300,
            ),
            follow_redirects=True,
        )
    return _client


async def close_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None
