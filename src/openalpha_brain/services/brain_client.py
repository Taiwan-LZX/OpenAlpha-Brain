"""
OpenAlpha - Quant — WorldQuant BRAIN API Client

Full pipeline:
  1. Authenticate (HTTP Basic Auth → session cookie)
  2. POST /simulations with alpha payload
  3. Poll Location header URL respecting Retry-After
  4. Extract real metrics (Sharpe, Fitness, Turnover, Returns, Correlation)
  5. Run IQC gate check on REAL metrics
  6. Return BrainResult with PASS/FAIL and gate details

Auth: email + password via HTTP Basic Auth.
Credentials stored in .env as BRAIN_EMAIL / BRAIN_PASSWORD.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from openalpha_brain.services.http_pool import get_client
from openalpha_brain.utils.algo_logger import Timer, algo_log

logger = logging.getLogger(__name__)

BRAIN_BASE = "https://api.worldquantbrain.com"
_AUTH_URL  = f"{BRAIN_BASE}/authentication"
_SIM_URL   = f"{BRAIN_BASE}/simulations"

# IQC real-metric hard gates
GATE_SHARPE_MIN   = 1.25
GATE_FITNESS_MIN  = 1.0
GATE_TURNOVER_MIN = 1.0
GATE_TURNOVER_MAX = 70.0

_SIM_DEFAULTS = {
    "type": "REGULAR",
    "settings": {
        "instrumentType": "EQUITY",
        "region": "USA",
        "universe": "TOP3000",
        "delay": 1,
        "decay": 5,
        "neutralization": "INDUSTRY",
        "truncation": 0.05,
        "pasteurization": "ON",
        "unitHandling": "VERIFY",
        "nanHandling": "ON",
        "language": "FASTEXPR",
        "visualization": False,
    },
}


class BrainAuthError(Exception):
    """Raised when BRAIN authentication fails."""


class BrainSubmitError(Exception):
    """Raised when simulation submission fails."""


class BrainPollError(Exception):
    """Raised when polling fails after max retries."""


@dataclass
class BrainGateResult:
    """Results from real IQC gate checks on BRAIN simulation output."""
    passed: bool
    sharpe: float | None        = None
    fitness: float | None       = None
    turnover: float | None      = None
    returns: float | None       = None
    drawdown: float | None      = None
    margin: float | None        = None
    os_sharpe: float | None       = None
    os_fitness: float | None      = None
    os_returns: float | None      = None
    is_os_decay_ratio: float | None = None
    overfitting_warning: bool        = False
    failures: list[str]            = field(default_factory=list)
    warnings: list[str]            = field(default_factory=list)
    brain_checks: list[dict]       = field(default_factory=list)  # raw checks[] from BRAIN
    alpha_id: str | None        = None   # BRAIN alpha ID after submission
    simulation_status: str         = "UNKNOWN"


async def authenticate(email: str, password: str) -> httpx.Cookies:
    """
    Authenticate with BRAIN using HTTP Basic Auth.
    BRAIN returns HTTP 201 on success with session cookies.
    """
    client = get_client()
    resp = await client.post(
        _AUTH_URL,
        auth=(email, password),
        timeout=30.0,
    )
    # BRAIN returns 201 (Created) on successful auth
    if resp.status_code in (200, 201):
        logger.info("[brain] Authenticated as %s (HTTP %d)", email, resp.status_code)
        return resp.cookies
    body = _safe_json(resp)
    msg = body.get("message", resp.text[:200]) if body else resp.text[:200]
    raise BrainAuthError(
        f"BRAIN auth failed HTTP {resp.status_code}: {msg}",
    )


@algo_log(level=logging.INFO)
async def submit_and_poll(
    simulation_payload: dict,
    cookies: httpx.Cookies,
    max_poll_seconds: int = 300,
) -> BrainGateResult:
    """
    Submit a simulation to BRAIN and poll until complete.

    Args:
        simulation_payload: The dict from alpha's simulation_payload field.
                            Must include 'settings' and 'regular' keys.
        cookies: Auth cookies from authenticate().
        max_poll_seconds: Abort polling after this many seconds.

    Returns:
        BrainGateResult with all real metrics and gate pass/fail.
    """
    # Ensure type field is present (required by BRAIN API)
    # Required BRAIN API fields with IQC-compliant defaults
    _defaults = _SIM_DEFAULTS
    # Deep-merge: simulation_payload settings override defaults
    payload: dict[str, Any] = dict(_defaults)
    payload["settings"] = {**_defaults["settings"], **simulation_payload.get("settings", {})}  # type: ignore[dict-item]
    payload["regular"] = simulation_payload.get("regular", "")
    if "type" in simulation_payload:
        payload["type"] = simulation_payload["type"]


    client = get_client()

    # ── 1. Submit simulation ────────────────────────────────────────────
    with Timer("brain_submit"):
        logger.info("[brain] Submitting simulation for expression: %s...",
                    str(payload.get("regular", ""))[:60])

        sim_resp = await client.post(_SIM_URL, json=payload, cookies=cookies, timeout=60.0)

    if sim_resp.status_code not in (200, 201, 202):
        body = _safe_json(sim_resp)
        msg = body.get("message", sim_resp.text[:300]) if body else sim_resp.text[:300]
        raise BrainSubmitError(
            f"BRAIN simulation submit failed HTTP {sim_resp.status_code}: {msg}",
        )

    # BRAIN returns 201 + Location header pointing to progress URL
    location = sim_resp.headers.get("Location")
    if not location:
        result_data = _safe_json(sim_resp) or {}
        prelim = _extract_gate_result(result_data)
        alpha_id = prelim.alpha_id
        if alpha_id:
            alpha_data = await fetch_alpha_details(alpha_id, cookies)
            if alpha_data:
                result = _extract_gate_result(alpha_data)
                if result.sharpe is None and not result.brain_checks:
                    check_data = await check_alpha(alpha_id, cookies)
                    if check_data:
                        result = _extract_gate_result(check_data)
                return result
        return prelim

    # Make location absolute if relative
    if location.startswith("/"):
        location = BRAIN_BASE + location

    logger.info("[brain] Simulation submitted → polling: %s", location)

    # ── 2. Poll until done ──────────────────────────────────────────────
    with Timer("brain_poll"):
        elapsed: float = 0
        poll_count = 0

        while elapsed < max_poll_seconds:
            poll_resp = await client.get(location, cookies=cookies, timeout=60.0)

            if poll_resp.status_code in (200, 201, 202):
                data = _safe_json(poll_resp) or {}

                # Retry-After = 0 or absent means simulation is done
                retry_after_raw = poll_resp.headers.get("Retry-After")
                if retry_after_raw is None:
                    retry_after = None
                else:
                    try:
                        retry_after = float(retry_after_raw)
                    except ValueError:
                        retry_after = None

                if retry_after is None:
                    logger.info("[brain] Simulation complete after %ds / %d polls",
                                elapsed, poll_count)
                    prelim = _extract_gate_result(data)
                    alpha_id = prelim.alpha_id
                    if prelim.simulation_status == "ERROR":
                        logger.warning("[brain] BRAIN simulation ERROR: %s", prelim.failures[0] if prelim.failures else "unknown")
                        return prelim
                    if alpha_id:
                        logger.info("[brain] Fetching alpha %s details for real metrics", alpha_id)
                        alpha_data = await fetch_alpha_details(alpha_id, cookies)
                        if alpha_data:
                            result = _extract_gate_result(alpha_data)
                            if result.sharpe is None and not result.brain_checks:
                                logger.info("[brain] No metrics in alpha details — calling check API for alpha %s", alpha_id)
                                check_data = await check_alpha(alpha_id, cookies)
                                if check_data:
                                    result = _extract_gate_result(check_data)
                            return result
                    logger.info("[brain] No alpha_id from simulation — using poll response data")
                    return prelim

                wait = max(min(retry_after, 30.0), 1.0)
                logger.info(
                    "[brain] Simulation running — Retry-After=%.0fs, elapsed=%ds",
                    retry_after, elapsed,
                )
                await asyncio.sleep(wait)
                elapsed += wait
                poll_count += 1

            elif poll_resp.status_code == 429:
                wait = float(poll_resp.headers.get("Retry-After", "10"))
                logger.warning("[brain] Rate limited while polling — waiting %.0fs", wait)
                await asyncio.sleep(wait)
                elapsed += wait

            else:
                body = _safe_json(poll_resp)
                msg = (body or {}).get("message", poll_resp.text[:200])
                raise BrainPollError(
                    f"Unexpected poll response HTTP {poll_resp.status_code}: {msg}",
                )

        raise BrainPollError(
            f"BRAIN simulation did not complete within {max_poll_seconds}s",
        )



async def fetch_alpha_details(
    alpha_id: str,
    cookies: httpx.Cookies,
) -> dict | None:
    client = get_client()
    url = f"{BRAIN_BASE}/alphas/{alpha_id}"
    try:
        resp = await client.get(url, cookies=cookies, timeout=30.0)
        if resp.status_code == 200:
            return _safe_json(resp)
        logger.warning("[brain] Failed to fetch alpha %s details: HTTP %d", alpha_id, resp.status_code)
        return None
    except (TimeoutError, aiohttp.ClientError, ConnectionError) as exc:
        logger.error("[brain] Error fetching alpha %s details: %s", alpha_id, exc)
        return None


async def check_alpha(
    alpha_id: str,
    cookies: httpx.Cookies,
    max_poll_seconds: int = 120,
) -> dict | None:
    client = get_client()
    check_url = f"{BRAIN_BASE}/alphas/{alpha_id}/check"
    elapsed: float = 0
    while elapsed < max_poll_seconds:
        try:
            resp = await client.get(check_url, cookies=cookies, timeout=30.0)
        except (TimeoutError, aiohttp.ClientError, ConnectionError) as exc:
            logger.error("[brain] Error checking alpha %s: %s", alpha_id, exc)
            return None
        if resp.status_code == 401:
            raise BrainAuthError(f"Cookie expired during check of alpha {alpha_id}")
        if resp.status_code not in (200, 201, 202):
            logger.warning("[brain] Check alpha %s returned HTTP %d", alpha_id, resp.status_code)
            return None
        retry_after_raw = resp.headers.get("Retry-After")
        if retry_after_raw is None:
            return _safe_json(resp)
        try:
            wait = min(float(retry_after_raw), 30.0)
        except ValueError:
            wait = 5.0
        await asyncio.sleep(wait)
        elapsed += wait
    logger.warning("[brain] Check did not complete within %ds for alpha %s", max_poll_seconds, alpha_id)
    return None


async def patch_properties(
    alpha_id: str,
    cookies: httpx.Cookies,
    *,
    name: str | None = None,
    category: str | None = None,
    tags: list[str] | None = None,
    color: str | None = None,
    description: str | None = None,
) -> bool:
    client = get_client()
    url = f"{BRAIN_BASE}/alphas/{alpha_id}"
    properties: dict = {}
    if name is not None:
        properties["name"] = name
    if category is not None:
        properties["category"] = category
    if tags is not None:
        properties["tags"] = tags
    if color is not None:
        properties["color"] = color
    if description is not None:
        properties["regular"] = {"description": description}
    if not properties:
        return True
    try:
        resp = await client.patch(url, json=properties, cookies=cookies, timeout=30.0)
        if resp.status_code == 200:
            logger.info("[brain] Patched properties of alpha %s", alpha_id)
            return True
        if resp.status_code == 401:
            raise BrainAuthError(f"Cookie expired during patch of alpha {alpha_id}")
        logger.warning("[brain] Patch properties of alpha %s returned HTTP %d", alpha_id, resp.status_code)
        return False
    except BrainAuthError:
        raise
    except (TimeoutError, aiohttp.ClientError, ConnectionError) as exc:
        logger.error("[brain] Error patching properties of alpha %s: %s", alpha_id, exc)
        return False


async def list_alphas(
    cookies: httpx.Cookies,
    *,
    limit: int = 50,
    offset: int = 0,
    order: str | None = None,
) -> dict | None:
    client = get_client()
    url = f"{BRAIN_BASE}/users/self/alphas"
    params: dict[str, str | int] = {"limit": min(max(limit, 1), 100), "offset": min(max(offset, 0), 9999)}
    if order:
        params["order"] = order
    try:
        resp = await client.get(url, params=params, cookies=cookies, timeout=30.0)
        if resp.status_code == 200:
            return _safe_json(resp)
        if resp.status_code == 401:
            raise BrainAuthError("Cookie expired during list_alphas")
        logger.warning("[brain] List alphas returned HTTP %d", resp.status_code)
        return None
    except BrainAuthError:
        raise
    except (TimeoutError, aiohttp.ClientError, ConnectionError) as exc:
        logger.error("[brain] Error listing alphas: %s", exc)
        return None


async def submit_alpha_for_review(
    alpha_id: str,
    cookies: httpx.Cookies,
    max_poll_seconds: int = 120,
) -> bool:
    client = get_client()
    submit_url = f"{BRAIN_BASE}/alphas/{alpha_id}/submit"
    try:
        resp = await client.post(submit_url, cookies=cookies, timeout=30.0)
    except (TimeoutError, aiohttp.ClientError, ConnectionError) as exc:
        logger.error("[brain] Error submitting alpha %s for review: %s", alpha_id, exc)
        return False
    if resp.status_code == 401:
        raise BrainAuthError(f"Cookie expired during submit of alpha {alpha_id}")
    if resp.status_code not in (200, 201, 202):
        body = _safe_json(resp) or {}
        msg = body.get("message", resp.text[:200]) if body else resp.text[:200]
        logger.warning(
            "[brain] Submit alpha %s for review returned HTTP %d: %s",
            alpha_id, resp.status_code, msg,
        )
        return False
    retry_after_raw = resp.headers.get("Retry-After")
    if retry_after_raw is None:
        logger.info("[brain] Alpha %s submitted for review successfully", alpha_id)
        return True
    location = resp.headers.get("Location")
    if location and location.startswith("/"):
        location = BRAIN_BASE + location
    poll_url = location or submit_url
    elapsed: float = 0
    while elapsed < max_poll_seconds:
        try:
            poll_resp = await client.get(poll_url, cookies=cookies, timeout=30.0)
        except (TimeoutError, aiohttp.ClientError, ConnectionError):
            await asyncio.sleep(5.0)
            elapsed += 5.0
            continue
        if poll_resp.status_code == 401:
            raise BrainAuthError(f"Cookie expired during submit poll of alpha {alpha_id}")
        retry_after_raw = poll_resp.headers.get("Retry-After")
        if retry_after_raw is None:
            logger.info("[brain] Alpha %s submitted for review successfully (after poll)", alpha_id)
            return True
        try:
            wait = min(float(retry_after_raw), 30.0)
        except ValueError:
            wait = 5.0
        await asyncio.sleep(wait)
        elapsed += wait
    logger.warning("[brain] Submit for review did not complete within %ds for alpha %s", max_poll_seconds, alpha_id)
    return False


async def fetch_yearly_performance(
    alpha_id: str,
    cookies: httpx.Cookies,
    max_poll_seconds: int = 120,
) -> list[dict] | None:
    client = get_client()
    url_stats = f"{BRAIN_BASE}/alphas/{alpha_id}/recordsets/yearly-stats"
    try:
        elapsed: float = 0
        while elapsed < 30:
            resp = await client.get(url_stats, cookies=cookies, timeout=30.0)
            if resp.status_code == 401:
                raise BrainAuthError(f"Cookie expired fetching yearly-stats for alpha {alpha_id}")
            retry_after = float(resp.headers.get("Retry-After", "0"))
            if retry_after > 0:
                wait = max(min(retry_after, 10.0), 1.0)
                await asyncio.sleep(wait)
                elapsed += wait
                continue
            if resp.status_code == 200:
                data = _safe_json(resp)
                if isinstance(data, dict):
                    records = data.get("records", [])
                    if records and isinstance(records, list):
                        schema_props = data.get("schema", {}).get("properties", [])
                        if schema_props:
                            columns = [p.get("name", f"col_{i}") for i, p in enumerate(schema_props)]
                            parsed = [dict(zip(columns, rec)) for rec in records if isinstance(rec, list)]
                            if parsed:
                                logger.info("[brain] Got yearly performance from /yearly-stats for alpha %s", alpha_id)
                                return parsed
            break
    except BrainAuthError:
        raise
    except (TimeoutError, aiohttp.ClientError, ConnectionError) as exc:
        logger.debug("[brain] /yearly-stats endpoint failed for alpha %s: %s", alpha_id, exc)

    url = f"{BRAIN_BASE}/alphas/{alpha_id}/recordsets/yearly"
    try:
        resp = await client.get(url, cookies=cookies, timeout=30.0)
        if resp.status_code == 200:
            data = _safe_json(resp)
            if isinstance(data, list):
                return data
        elif resp.status_code == 401:
            raise BrainAuthError(f"Cookie expired fetching yearly performance for alpha {alpha_id}")
    except BrainAuthError:
        raise
    except (TimeoutError, aiohttp.ClientError, ConnectionError) as exc:
        logger.error("[brain] Error fetching yearly endpoint for alpha %s: %s", alpha_id, exc)

    logger.info("[brain] Yearly endpoint not available for alpha %s — computing from PnL data", alpha_id)
    url_pnl = f"{BRAIN_BASE}/alphas/{alpha_id}/recordsets/pnl"
    pnl_elapsed: float = 0
    while pnl_elapsed < max_poll_seconds:
        try:
            resp = await client.get(url_pnl, cookies=cookies, timeout=30.0)
        except (TimeoutError, aiohttp.ClientError, ConnectionError) as exc:
            logger.error("[brain] Error fetching PnL for yearly computation of alpha %s: %s", alpha_id, exc)
            return None
        if resp.status_code == 404:
            logger.warning("[brain] PnL endpoint returned 404 for alpha %s (yearly fallback)", alpha_id)
            return None
        if resp.status_code == 401:
            raise BrainAuthError(f"Cookie expired fetching PnL for yearly of alpha {alpha_id}")
        retry_after = float(resp.headers.get("Retry-After", "0"))
        if retry_after > 0:
            wait = max(min(retry_after, 30.0), 1.0)
            logger.info("[brain] PnL data not ready for yearly computation of alpha %s — Retry-After=%.0fs", alpha_id, retry_after)
            await asyncio.sleep(wait)
            pnl_elapsed += wait
            continue
        if resp.status_code == 200:
            break
        logger.warning("[brain] Failed to fetch PnL for yearly computation of alpha %s: HTTP %d", alpha_id, resp.status_code)
        return None
    else:
        logger.warning("[brain] PnL polling timed out for yearly computation of alpha %s after %ds", alpha_id, max_poll_seconds)
        return None

    raw = _safe_json(resp)
    records = raw.get("records", []) if isinstance(raw, dict) else []
    import math
    from collections import defaultdict
    grouped: dict[int, list[float]] = defaultdict(list)
    for r in records:
        if isinstance(r, (list, tuple)) and len(r) >= 2:
            year_str = str(r[0])[:4]
            if year_str.isdigit():
                grouped[int(year_str)].append(float(r[1]))
    result = []
    for year in sorted(grouped.keys()):
        daily = grouped[year]
        n = len(daily)
        total = sum(daily)
        if n > 1:
            mean = total / n
            var = sum((x - mean) ** 2 for x in daily) / (n - 1)
            std = math.sqrt(var) if var > 0 else 0
            sharpe = round(mean / std * math.sqrt(252), 4) if std > 0 else None
        else:
            sharpe = None
        result.append({"year": year, "pnl": round(total, 2), "sharpe": sharpe, "days": n})
    return result if result else None


async def fetch_pnl_curve(
    alpha_id: str,
    cookies: httpx.Cookies,
    max_poll_seconds: int = 120,
) -> list[float] | None:
    client = get_client()
    daily_url = f"{BRAIN_BASE}/alphas/{alpha_id}/recordsets/daily-pnl"
    elapsed: float = 0
    while elapsed < max_poll_seconds:
        try:
            resp = await client.get(daily_url, cookies=cookies, timeout=30.0)
        except (TimeoutError, aiohttp.ClientError, ConnectionError) as exc:
            logger.error("[brain] Error fetching daily PnL for alpha %s: %s", alpha_id, exc)
            break
        if resp.status_code == 404:
            break
        if resp.status_code == 401:
            raise BrainAuthError(f"Cookie expired fetching daily PnL for alpha {alpha_id}")
        retry_after = float(resp.headers.get("Retry-After", "0"))
        if retry_after > 0:
            wait = max(min(retry_after, 30.0), 1.0)
            logger.info("[brain] Daily PnL data not ready for alpha %s — Retry-After=%.0fs", alpha_id, retry_after)
            await asyncio.sleep(wait)
            elapsed += wait
            continue
        if resp.status_code == 200:
            data = _safe_json(resp)
            records = data.get("records", []) if isinstance(data, dict) else []
            pnl_values = []
            for r in records:
                if isinstance(r, (list, tuple)) and len(r) >= 2:
                    pnl_values.append(float(r[1]))
            if pnl_values:
                logger.info("[brain] Got daily PnL from /recordsets/daily-pnl for alpha %s (%d values)", alpha_id, len(pnl_values))
                return pnl_values
        break

    url = f"{BRAIN_BASE}/alphas/{alpha_id}/recordsets/pnl"
    submit_elapsed: float = 0
    while submit_elapsed < max_poll_seconds:
        try:
            resp = await client.get(url, cookies=cookies, timeout=30.0)
        except (TimeoutError, aiohttp.ClientError, ConnectionError) as exc:
            logger.error("[brain] Error fetching PnL curve for alpha %s: %s", alpha_id, exc)
            return None
        if resp.status_code == 404:
            logger.warning("[brain] PnL endpoint returned 404 for alpha %s", alpha_id)
            return None
        if resp.status_code == 401:
            raise BrainAuthError(f"Cookie expired fetching PnL curve for alpha {alpha_id}")
        retry_after = float(resp.headers.get("Retry-After", "0"))
        if retry_after > 0:
            wait = max(min(retry_after, 30.0), 1.0)
            logger.info("[brain] PnL data not ready for alpha %s — Retry-After=%.0fs", alpha_id, retry_after)
            await asyncio.sleep(wait)
            submit_elapsed += wait
            continue
        if resp.status_code == 200:
            data = _safe_json(resp)
            records = data.get("records", []) if isinstance(data, dict) else []
            pnl_values = []
            for r in records:
                if isinstance(r, (list, tuple)) and len(r) >= 2:
                    pnl_values.append(float(r[1]))
            return pnl_values if pnl_values else None
        logger.warning("[brain] Failed to fetch PnL curve for alpha %s: HTTP %d", alpha_id, resp.status_code)
        return None
    logger.warning("[brain] PnL polling timed out for alpha %s after %ds", alpha_id, max_poll_seconds)
    return None


async def fetch_correlations(
    alpha_id: str,
    cookies: httpx.Cookies,
) -> dict | None:
    client = get_client()
    url = f"{BRAIN_BASE}/alphas/{alpha_id}/recordsets/correlations"
    try:
        resp = await client.get(url, cookies=cookies, timeout=30.0)
        if resp.status_code == 200:
            data = _safe_json(resp)
            if isinstance(data, dict):
                return data
        elif resp.status_code == 401:
            raise BrainAuthError(f"Cookie expired fetching correlations for alpha {alpha_id}")
    except BrainAuthError:
        raise
    except (TimeoutError, aiohttp.ClientError, ConnectionError) as exc:
        logger.error("[brain] Error fetching correlations endpoint for alpha %s: %s", alpha_id, exc)

    logger.info("[brain] Correlations endpoint not available for alpha %s — falling back to /correlations/self", alpha_id)
    self_corr_url = f"{BRAIN_BASE}/alphas/{alpha_id}/correlations/self"
    try:
        elapsed: float = 0
        while elapsed < 30:
            resp = await client.get(self_corr_url, cookies=cookies, timeout=30.0)
            if resp.status_code == 401:
                raise BrainAuthError(f"Cookie expired fetching self correlations for alpha {alpha_id}")
            retry_after = float(resp.headers.get("Retry-After", "0"))
            if retry_after > 0:
                wait = max(min(retry_after, 10.0), 1.0)
                await asyncio.sleep(wait)
                elapsed += wait
                continue
            if resp.status_code == 200:
                data = _safe_json(resp)
                if isinstance(data, dict):
                    records = data.get("records", [])
                    max_corr = data.get("max")
                    if records and isinstance(records, list):
                        corr_values = []
                        for rec in records:
                            if isinstance(rec, (list, tuple)) and len(rec) >= 3:
                                try:
                                    corr_values.append(float(rec[-1]))
                                except (ValueError, TypeError):
                                    pass
                        if corr_values:
                            return {"self_correlation": max(corr_values), "max": max_corr, "records": records}
                    if max_corr is not None:
                        return {"self_correlation": float(max_corr)}
            break
    except BrainAuthError:
        raise
    except (TimeoutError, aiohttp.ClientError, ConnectionError) as exc:
        logger.error("[brain] Error fetching /correlations/self for alpha %s: %s", alpha_id, exc)

    logger.info("[brain] /correlations/self not available for alpha %s — falling back to check API", alpha_id)
    check_url = f"{BRAIN_BASE}/alphas/{alpha_id}/check"
    try:
        resp = await client.get(check_url, cookies=cookies, timeout=30.0)
        if resp.status_code == 200:
            data = _safe_json(resp)
            checks = data if isinstance(data, list) else (data.get("checks", []) if isinstance(data, dict) else [])
            for chk in checks:
                if isinstance(chk, dict) and "SELF_CORRELATION" in str(chk.get("name", "")).upper():
                    val = chk.get("value")
                    if isinstance(val, (int, float)):
                        return {"self_correlation": float(val)}
    except (TimeoutError, aiohttp.ClientError, ConnectionError) as exc:
        logger.error("[brain] Error fetching correlations from check API for alpha %s: %s", alpha_id, exc)
    return None


async def fetch_self_correlations(
    alpha_id: str,
    cookies: httpx.Cookies,
    max_poll_seconds: int = 60,
) -> dict | None:
    client = get_client()
    url = f"{BRAIN_BASE}/alphas/{alpha_id}/correlations/self"
    elapsed: float = 0
    while elapsed < max_poll_seconds:
        try:
            resp = await client.get(url, cookies=cookies, timeout=30.0)
        except (TimeoutError, aiohttp.ClientError, ConnectionError) as exc:
            logger.error("[brain] Error fetching self correlations for alpha %s: %s", alpha_id, exc)
            return None
        if resp.status_code == 401:
            raise BrainAuthError(f"Cookie expired fetching self correlations for alpha {alpha_id}")
        if resp.status_code == 404:
            return None
        retry_after = float(resp.headers.get("Retry-After", "0"))
        if retry_after > 0:
            wait = max(min(retry_after, 15.0), 1.0)
            await asyncio.sleep(wait)
            elapsed += wait
            continue
        if resp.status_code == 200:
            data = _safe_json(resp)
            if isinstance(data, dict):
                records = data.get("records", [])
                max_corr = data.get("max")
                min_corr = data.get("min")
                return {"records": records, "max": max_corr, "min": min_corr}
            return data
        break
    return None


async def fetch_prod_correlations(
    alpha_id: str,
    cookies: httpx.Cookies,
    max_poll_seconds: int = 60,
) -> dict | None:
    client = get_client()
    url = f"{BRAIN_BASE}/alphas/{alpha_id}/correlations/prod"
    elapsed: float = 0
    while elapsed < max_poll_seconds:
        try:
            resp = await client.get(url, cookies=cookies, timeout=30.0)
        except (TimeoutError, aiohttp.ClientError, ConnectionError) as exc:
            logger.error("[brain] Error fetching prod correlations for alpha %s: %s", alpha_id, exc)
            return None
        if resp.status_code == 401:
            raise BrainAuthError(f"Cookie expired fetching prod correlations for alpha {alpha_id}")
        if resp.status_code == 404:
            return None
        retry_after = float(resp.headers.get("Retry-After", "0"))
        if retry_after > 0:
            wait = max(min(retry_after, 15.0), 1.0)
            await asyncio.sleep(wait)
            elapsed += wait
            continue
        if resp.status_code == 200:
            data = _safe_json(resp)
            if isinstance(data, dict):
                records = data.get("records", [])
                max_corr = data.get("max")
                min_corr = data.get("min")
                return {"records": records, "max": max_corr, "min": min_corr}
            return data
        break
    return None


async def fetch_yearly_stats(
    alpha_id: str,
    cookies: httpx.Cookies,
    max_poll_seconds: int = 60,
) -> list[dict] | None:
    client = get_client()
    url = f"{BRAIN_BASE}/alphas/{alpha_id}/recordsets/yearly-stats"
    elapsed: float = 0
    while elapsed < max_poll_seconds:
        try:
            resp = await client.get(url, cookies=cookies, timeout=30.0)
        except (TimeoutError, aiohttp.ClientError, ConnectionError) as exc:
            logger.error("[brain] Error fetching yearly-stats for alpha %s: %s", alpha_id, exc)
            return None
        if resp.status_code == 401:
            raise BrainAuthError(f"Cookie expired fetching yearly-stats for alpha {alpha_id}")
        if resp.status_code == 404:
            return None
        retry_after = float(resp.headers.get("Retry-After", "0"))
        if retry_after > 0:
            wait = max(min(retry_after, 15.0), 1.0)
            await asyncio.sleep(wait)
            elapsed += wait
            continue
        if resp.status_code == 200:
            data = _safe_json(resp)
            if isinstance(data, dict):
                records = data.get("records", [])
                if records and isinstance(records, list):
                    schema_props = data.get("schema", {}).get("properties", [])
                    if schema_props:
                        columns = [p.get("name", f"col_{i}") for i, p in enumerate(schema_props)]
                        return [dict(zip(columns, rec)) for rec in records if isinstance(rec, list)]
                    return [{"raw": rec} for rec in records]
            return None
        break
    return None


async def fetch_daily_pnl(
    alpha_id: str,
    cookies: httpx.Cookies,
    max_poll_seconds: int = 120,
) -> list[float] | None:
    client = get_client()
    url = f"{BRAIN_BASE}/alphas/{alpha_id}/recordsets/daily-pnl"
    elapsed: float = 0
    while elapsed < max_poll_seconds:
        try:
            resp = await client.get(url, cookies=cookies, timeout=30.0)
        except (TimeoutError, aiohttp.ClientError, ConnectionError) as exc:
            logger.error("[brain] Error fetching daily-pnl for alpha %s: %s", alpha_id, exc)
            return None
        if resp.status_code == 401:
            raise BrainAuthError(f"Cookie expired fetching daily-pnl for alpha {alpha_id}")
        if resp.status_code == 404:
            return None
        retry_after = float(resp.headers.get("Retry-After", "0"))
        if retry_after > 0:
            wait = max(min(retry_after, 30.0), 1.0)
            logger.info("[brain] Daily PnL data not ready for alpha %s — Retry-After=%.0fs", alpha_id, retry_after)
            await asyncio.sleep(wait)
            elapsed += wait
            continue
        if resp.status_code == 200:
            data = _safe_json(resp)
            records = data.get("records", []) if isinstance(data, dict) else []
            pnl_values = []
            for r in records:
                if isinstance(r, (list, tuple)) and len(r) >= 2:
                    pnl_values.append(float(r[1]))
            return pnl_values if pnl_values else None
        break
    return None


def _extract_gate_result(data: dict) -> BrainGateResult:
    """
    Parse BRAIN simulation/alpha result JSON into a BrainGateResult.
    
    BRAIN API returns simulation results with the alpha registered under data["alpha"].
    Real metrics are under data["is"] (in-sample).
    Gate checks are under data["is"]["checks"].
    """
    failures: list[str] = []
    warnings: list[str] = []

    sim_status = data.get("status", "UNKNOWN")
    # Alpha ID is in "alpha" field of simulation result, or "id" of alpha record
    alpha_raw = data.get("alpha")
    if isinstance(alpha_raw, dict):
        alpha_id = alpha_raw.get("id") or data.get("id")
    elif isinstance(alpha_raw, str):
        alpha_id = alpha_raw
    else:
        alpha_id = data.get("id")
    error_msg  = data.get("message", "")

    # ── Handle ERROR status (unknown variable, syntax error, etc.) ───────────
    if sim_status == "ERROR":
        failure_msg = f"BRAIN simulation ERROR: {error_msg[:200]}"
        logger.warning("[brain] %s", failure_msg)
        return BrainGateResult(
            passed=False,
            failures=[failure_msg],
            warnings=[],
            alpha_id=alpha_id,
            simulation_status=sim_status,
        )

    # ── Extract real metrics from "is" (in-sample) section ──────────────────
    # Confirmed field names from live BRAIN API response
    is_data = data.get("is", {}) or {}

    sharpe   = _get_float(is_data, ["sharpe"])
    fitness  = _get_float(is_data, ["fitness"])
    turnover = _get_float(is_data, ["turnover"])   # decimal (0.35 = 35%)
    returns  = _get_float(is_data, ["returns"])    # decimal (-0.12 = -12%)
    drawdown = _get_float(is_data, ["drawdown"])
    margin   = _get_float(is_data, ["margin"])

    turnover_pct = (turnover * 100) if turnover is not None else None
    returns_pct  = (returns  * 100) if returns  is not None else None
    drawdown_pct = (drawdown * 100) if drawdown is not None else None

    os_data = data.get("os", {}) or {}

    os_sharpe   = _get_float(os_data, ["sharpe"])
    os_fitness  = _get_float(os_data, ["fitness"])
    os_returns  = _get_float(os_data, ["returns"])

    _os_sharpe_pct = (os_sharpe * 100) if os_sharpe is not None else None
    os_returns_pct = (os_returns * 100) if os_returns is not None else None

    is_os_decay_ratio = None
    overfitting_warning = False
    if sharpe is not None and os_sharpe is not None and sharpe > 0:
        is_os_decay_ratio = round(os_sharpe / sharpe, 4)
        if is_os_decay_ratio < 0.5:
            overfitting_warning = True
            warnings.append(f"Overfitting detected: IS/OS Sharpe decay ratio = {is_os_decay_ratio:.2f} < 0.5")

    # ── Parse BRAIN's own gate checks[] array ────────────────────────────────
    brain_checks = is_data.get("checks", [])
    for chk in brain_checks:
        name   = chk.get("name", "")
        result = chk.get("result", "")
        value  = chk.get("value")
        limit  = chk.get("limit")

        if result == "FAIL":
            failures.append(
                f"BRAIN gate {name} FAIL: value={value} limit={limit}",
            )
        elif result == "PENDING":
            warnings.append(f"BRAIN gate {name} still PENDING (correlation check)")

    # ── Log real metrics ──────────────────────────────────────────────────────
    logger.info(
        "[brain] Simulation COMPLETE — sharpe=%.3f fitness=%.3f turnover=%.1f%% returns=%.1f%%",
        sharpe   or 0.0,
        fitness  or 0.0,
        turnover_pct or 0.0,
        returns_pct  or 0.0,
    )

    # If BRAIN's own checks had failures → mark as FAIL
    # If no checks returned (status != COMPLETE), use our gate thresholds
    if not brain_checks and sharpe is not None:
        if sharpe < GATE_SHARPE_MIN:
            failures.append(f"REAL Sharpe {sharpe:.3f} < {GATE_SHARPE_MIN}")
        if fitness is not None and fitness <= GATE_FITNESS_MIN:
            failures.append(f"REAL Fitness {fitness:.3f} ≤ {GATE_FITNESS_MIN}")
        if turnover_pct is not None:
            if turnover_pct < GATE_TURNOVER_MIN:
                failures.append(f"REAL Turnover {turnover_pct:.1f}% < {GATE_TURNOVER_MIN}%")
            if turnover_pct > GATE_TURNOVER_MAX:
                failures.append(f"REAL Turnover {turnover_pct:.1f}% > {GATE_TURNOVER_MAX}%")

    if not brain_checks and sharpe is None and fitness is None:
        logger.warning("[brain] Simulation completed but no metrics or checks returned — marking as FAIL")
        failures.append("BRAIN simulation returned no metrics or gate checks — cannot verify PASS")

    if failures:
        for f in failures:
            logger.warning("[brain] Gate FAIL: %s", f)

    passed = len(failures) == 0

    return BrainGateResult(
        passed=passed,
        sharpe=sharpe,
        fitness=fitness,
        turnover=turnover_pct,
        returns=returns_pct,
        drawdown=drawdown_pct,
        margin=margin,
        os_sharpe=os_sharpe,
        os_fitness=os_fitness,
        os_returns=os_returns_pct,
        is_os_decay_ratio=is_os_decay_ratio,
        overfitting_warning=overfitting_warning,
        failures=failures,
        warnings=warnings,
        brain_checks=brain_checks,       # pass raw checks[] through
        alpha_id=alpha_id,
        simulation_status=sim_status,
    )


def _safe_json(resp: httpx.Response) -> dict | None:
    try:
        return resp.json()
    except json.JSONDecodeError:
        return None


def _get_float(d: dict, keys: list[str]) -> float | None:
    """Try multiple key names, return first float found."""
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None
