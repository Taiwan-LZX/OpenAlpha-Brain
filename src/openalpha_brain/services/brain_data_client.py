import asyncio
import logging

import aiohttp
import httpx

from openalpha_brain.services import brain_client

logger = logging.getLogger(__name__)

_brain_data_client = None


class BrainDataClient:
    def __init__(self, email: str, password: str):
        self._email = email
        self._password = password
        self._cookies: httpx.Cookies | None = None

    async def _ensure_client(self):
        if self._cookies is None:
            self._cookies = await brain_client.authenticate(self._email, self._password)
        return self._cookies

    async def get_yearly_performance(self, alpha_id: str) -> list[dict] | None:
        try:
            cookies = await self._ensure_client()
            result = await brain_client.fetch_yearly_performance(alpha_id, cookies)
            if result is not None:
                return result
        except brain_client.BrainAuthError:
            logger.info("BrainDataClient session expired, reconnecting...")
            self._cookies = None
            cookies = await self._ensure_client()
            result = await brain_client.fetch_yearly_performance(alpha_id, cookies)
            if result is not None:
                return result
        except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError, OSError) as e:  # noqa: SIM105
            logger.warning("Failed to get yearly performance for %s: %s", alpha_id, e)

        logger.info("Falling back to alpha details for yearly performance of %s", alpha_id)
        try:
            cookies = await self._ensure_client()
            details = await brain_client.fetch_alpha_details(alpha_id, cookies)
            if details:
                is_data = details.get("is", {}) or {}
                if is_data:
                    sharpe = is_data.get("sharpe")
                    fitness = is_data.get("fitness")
                    turnover = is_data.get("turnover")
                    returns_val = is_data.get("returns")
                    drawdown = is_data.get("drawdown")
                    margin = is_data.get("margin")
                    summary = {
                        "year": "aggregate",
                        "sharpe": sharpe,
                        "fitness": fitness,
                        "turnover": (turnover * 100) if isinstance(turnover, (int, float)) else None,
                        "returns": (returns_val * 100) if isinstance(returns_val, (int, float)) else None,
                        "drawdown": (drawdown * 100) if isinstance(drawdown, (int, float)) else None,
                        "margin": margin,
                    }
                    return [summary]
                os_data = details.get("os", {}) or {}
                if os_data:
                    os_sharpe = os_data.get("sharpe")
                    os_returns = os_data.get("returns")
                    summary = {
                        "year": "aggregate_os",
                        "sharpe": os_sharpe,
                        "returns": (os_returns * 100) if isinstance(os_returns, (int, float)) else None,
                    }
                    return [summary]
        except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError, OSError) as e:
            logger.warning("Fallback to alpha details also failed for %s: %s", alpha_id, e)
        return None

    async def get_pnl_curve(self, alpha_id: str) -> list[float] | None:
        try:
            cookies = await self._ensure_client()
            return await brain_client.fetch_pnl_curve(alpha_id, cookies)
        except brain_client.BrainAuthError:
            logger.info("BrainDataClient session expired, reconnecting...")
            self._cookies = None
            cookies = await self._ensure_client()
            return await brain_client.fetch_pnl_curve(alpha_id, cookies)
        except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError, OSError) as e:
            logger.warning("Failed to get PnL curve for %s: %s", alpha_id, e)
            return None

    async def get_correlations(self, alpha_id: str) -> dict | None:
        try:
            cookies = await self._ensure_client()
            result = await brain_client.fetch_correlations(alpha_id, cookies)
            if result is not None:
                return result
        except brain_client.BrainAuthError:
            logger.info("BrainDataClient session expired, reconnecting...")
            self._cookies = None
            cookies = await self._ensure_client()
            result = await brain_client.fetch_correlations(alpha_id, cookies)
            if result is not None:
                return result
        except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError):  # noqa: SIM105
            return None

    async def get_self_correlations(self, alpha_id: str) -> dict | None:
        try:
            cookies = await self._ensure_client()
            return await brain_client.fetch_self_correlations(alpha_id, cookies)
        except brain_client.BrainAuthError:
            logger.info("BrainDataClient session expired, reconnecting...")
            self._cookies = None
            cookies = await self._ensure_client()
            return await brain_client.fetch_self_correlations(alpha_id, cookies)
        except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError, OSError) as e:
            logger.warning("Failed to get self correlations for %s: %s", alpha_id, e)
            return None

    async def get_prod_correlations(self, alpha_id: str) -> dict | None:
        try:
            cookies = await self._ensure_client()
            return await brain_client.fetch_prod_correlations(alpha_id, cookies)
        except brain_client.BrainAuthError:
            logger.info("BrainDataClient session expired, reconnecting...")
            self._cookies = None
            cookies = await self._ensure_client()
            return await brain_client.fetch_prod_correlations(alpha_id, cookies)
        except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError, OSError) as e:
            logger.warning("Failed to get prod correlations for %s: %s", alpha_id, e)
            return None

    async def get_yearly_stats(self, alpha_id: str) -> list[dict] | None:
        try:
            cookies = await self._ensure_client()
            return await brain_client.fetch_yearly_stats(alpha_id, cookies)
        except brain_client.BrainAuthError:
            logger.info("BrainDataClient session expired, reconnecting...")
            self._cookies = None
            cookies = await self._ensure_client()
            return await brain_client.fetch_yearly_stats(alpha_id, cookies)
        except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError):  # noqa: SIM105
            return None

    async def get_daily_pnl(self, alpha_id: str) -> list[float] | None:
        try:
            cookies = await self._ensure_client()
            return await brain_client.fetch_daily_pnl(alpha_id, cookies)
        except brain_client.BrainAuthError:
            logger.info("BrainDataClient session expired, reconnecting...")
            self._cookies = None
            cookies = await self._ensure_client()
            return await brain_client.fetch_daily_pnl(alpha_id, cookies)
        except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError):
            return None


def init_brain_data_client(email: str, password: str):
    global _brain_data_client
    _brain_data_client = BrainDataClient(email, password)
    logger.info("BrainDataClient initialized for %s", email[:3] + "***")


def get_brain_data_client() -> BrainDataClient | None:
    return _brain_data_client
