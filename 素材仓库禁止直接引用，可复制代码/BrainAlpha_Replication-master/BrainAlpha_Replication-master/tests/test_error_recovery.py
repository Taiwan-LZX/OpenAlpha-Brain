import unittest
from unittest.mock import MagicMock, patch
from typing import Any, Dict, Optional

from alpha_agent.config import AuthConfig
from alpha_agent.error_recovery import ErrorRecovery


class TestErrorRecovery(unittest.TestCase):
    def setUp(self) -> None:
        self.auth = AuthConfig(
            email="test@example.com",
            password="test_pass",
            cookie_header=None,
        )
        self.recovery = ErrorRecovery(auth=self.auth)

    def test_strategy_1_retry_returns_same_client(self) -> None:
        client = MagicMock()
        new_client, recovered = self.recovery._strategy_retry(client, 0)
        self.assertIs(new_client, client)
        self.assertTrue(recovered)

    def test_strategy_1_consecutive_attempts(self) -> None:
        client = MagicMock()
        for attempt in range(3):
            new_client, recovered = self.recovery._strategy_retry(client, attempt)
            self.assertIs(new_client, client)
            self.assertTrue(recovered)

    def test_recover_routes_to_strategy_1(self) -> None:
        client = MagicMock()
        for attempt in range(self.recovery.MAX_STRATEGY_1_RETRIES):
            new_client, recovered = self.recovery.recover(client, attempt)
            self.assertIs(new_client, client)
            self.assertTrue(recovered)

    @patch("alpha_agent.error_recovery.BrainClient")
    def test_recover_routes_to_strategy_2_with_email_password(
        self, MockBrainClient: MagicMock
    ) -> None:
        mock_instance = MagicMock()
        MockBrainClient.return_value = mock_instance
        client = MagicMock()
        attempt = self.recovery.MAX_STRATEGY_1_RETRIES
        new_client, recovered = self.recovery.recover(client, attempt)
        self.assertIsNotNone(new_client)
        self.assertTrue(recovered)
        mock_instance.login.assert_called_once_with(
            self.auth.email, self.auth.password
        )

    @patch("alpha_agent.error_recovery.BrainClient")
    def test_recover_routes_to_strategy_2_with_cookie(
        self, MockBrainClient: MagicMock
    ) -> None:
        mock_instance = MagicMock()
        MockBrainClient.return_value = mock_instance
        auth = AuthConfig(
            email=None, password=None,
            cookie_header="sessionid=abc123",
        )
        recovery = ErrorRecovery(auth=auth)
        client = MagicMock()
        attempt = recovery.MAX_STRATEGY_1_RETRIES
        new_client, recovered = recovery.recover(client, attempt)
        self.assertIsNotNone(new_client)
        self.assertTrue(recovered)
        MockBrainClient.assert_called_with(
            base_url=auth.base_url,
            timeout=auth.timeout,
            cookie_header="sessionid=abc123",
        )

    def test_strategy_2_no_credentials_returns_false(self) -> None:
        auth = AuthConfig(
            email=None, password=None, cookie_header=None,
        )
        recovery = ErrorRecovery(auth=auth)
        client = MagicMock()
        attempt = recovery.MAX_STRATEGY_1_RETRIES
        new_client, recovered = recovery.recover(client, attempt)
        self.assertIsNone(new_client)
        self.assertFalse(recovered)

    @patch("alpha_agent.error_recovery.BrainClient")
    def test_recover_routes_to_strategy_3(
        self, MockBrainClient: MagicMock
    ) -> None:
        mock_instance = MagicMock()
        MockBrainClient.return_value = mock_instance
        client = MagicMock()
        attempt = (self.recovery.MAX_STRATEGY_1_RETRIES +
                   self.recovery.MAX_STRATEGY_2_RETRIES)
        new_client, recovered = self.recovery.recover(client, attempt)
        self.assertIsNotNone(new_client)
        self.assertTrue(recovered)
        mock_instance.login.assert_called_once_with(
            self.auth.email, self.auth.password
        )

    def test_strategy_3_no_credentials_returns_false(self) -> None:
        auth = AuthConfig(
            email=None, password=None, cookie_header=None,
        )
        recovery = ErrorRecovery(auth=auth)
        client = MagicMock()
        attempt = (recovery.MAX_STRATEGY_1_RETRIES +
                   recovery.MAX_STRATEGY_2_RETRIES)
        new_client, recovered = recovery.recover(client, attempt)
        self.assertIsNone(new_client)
        self.assertFalse(recovered)

    @patch("alpha_agent.error_recovery.BrainClient")
    def test_recover_past_all_strategies_still_uses_strategy_3(
        self, MockBrainClient: MagicMock
    ) -> None:
        mock_instance = MagicMock()
        MockBrainClient.return_value = mock_instance
        client = MagicMock()
        attempt = (self.recovery.MAX_STRATEGY_1_RETRIES +
                   self.recovery.MAX_STRATEGY_2_RETRIES +
                   self.recovery.MAX_STRATEGY_3_RETRIES +
                   5)
        new_client, recovered = self.recovery.recover(client, attempt)
        self.assertIsNotNone(new_client)
        self.assertTrue(recovered)


if __name__ == "__main__":
    unittest.main()
