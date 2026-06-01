from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

from worldquant_brain_cli import BrainApiError, BrainClient
from alpha_agent.config import AuthConfig


class ErrorRecovery:
    """M3.5 Error Recovery Layer — Paper Section 3.5

    Escalating three-strategy repair sequence for API failures,
    intercepting errors before repair budget is consumed.

    Strategy 1 (attempt 0-1): retry with exponential backoff (1s → 2s → 4s)
    Strategy 2 (attempt 2-3): rebuild client, re-authenticate, retry
    Strategy 3 (attempt 4+):  clear session state, fresh login, retry
    """

    MAX_STRATEGY_1_RETRIES = 2
    MAX_STRATEGY_2_RETRIES = 2
    MAX_STRATEGY_3_RETRIES = 1
    BACKOFF_BASE = [1.0, 2.0, 4.0]

    def __init__(
        self,
        auth: AuthConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.auth = auth
        self._logger = logger or logging.getLogger(__name__)
        self._consecutive_failures = 0

    def recover(self, client: BrainClient, attempt: int) -> Tuple[Optional[BrainClient], bool]:
        if attempt < self.MAX_STRATEGY_1_RETRIES:
            return self._strategy_retry(client, attempt)
        if attempt < self.MAX_STRATEGY_1_RETRIES + self.MAX_STRATEGY_2_RETRIES:
            return self._strategy_reauth(attempt)
        return self._strategy_fresh()

    def _strategy_retry(self, client: BrainClient, attempt: int) -> Tuple[BrainClient, bool]:
        delay = self.BACKOFF_BASE[min(attempt, len(self.BACKOFF_BASE) - 1)]
        self._logger.info("M3.5 Strategy 1: retry attempt %d after %.1fs", attempt, delay)
        time.sleep(delay)
        return client, True

    def _strategy_reauth(self, attempt: int) -> Tuple[Optional[BrainClient], bool]:
        self._logger.info("M3.5 Strategy 2: re-authenticate attempt %d", attempt)
        try:
            new_client = BrainClient(
                base_url=self.auth.base_url,
                timeout=self.auth.timeout,
            )
            if self.auth.cookie_header:
                new_client = BrainClient(
                    base_url=self.auth.base_url,
                    timeout=self.auth.timeout,
                    cookie_header=self.auth.cookie_header,
                )
            elif self.auth.email and self.auth.password:
                new_client.login(self.auth.email, self.auth.password)
            else:
                return None, False
            return new_client, True
        except BrainApiError as exc:
            self._logger.warning("M3.5 re-auth failed: %s", exc)
            return None, False

    def _strategy_fresh(self) -> Tuple[Optional[BrainClient], bool]:
        self._logger.info("M3.5 Strategy 3: fresh login")
        try:
            fresh = BrainClient(
                base_url=self.auth.base_url,
                timeout=self.auth.timeout,
            )
            if self.auth.cookie_header:
                fresh = BrainClient(
                    base_url=self.auth.base_url,
                    timeout=self.auth.timeout,
                    cookie_header=self.auth.cookie_header,
                )
            elif self.auth.email and self.auth.password:
                fresh.login(self.auth.email, self.auth.password)
            else:
                return None, False
            return fresh, True
        except BrainApiError as exc:
            self._logger.error("M3.5 fresh login failed: %s", exc)
            return None, False
