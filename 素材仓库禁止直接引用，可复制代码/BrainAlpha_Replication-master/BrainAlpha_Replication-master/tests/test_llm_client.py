import json
import unittest
from typing import Any, Dict, Optional

from alpha_agent.config import ModelConfig
from alpha_agent.llm_client import LLMClient


class TestLLMClientExtractJsonPayload(unittest.TestCase):
    def setUp(self) -> None:
        self.client = LLMClient(ModelConfig(provider="test"))

    def test_valid_string_content(self) -> None:
        completion = {
            "choices": [{"message": {"content": json.dumps({"action": "stop"})}}],
        }
        result = self.client.extract_json_payload(completion)
        self.assertEqual(result["action"], "stop")

    def test_valid_list_content(self) -> None:
        completion = {
            "choices": [{"message": {"content": [{"text": '{"action":'}, {"text": ' "submit"}'}]}}],
        }
        result = self.client.extract_json_payload(completion)
        self.assertEqual(result["action"], "submit")

    def test_no_choices_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.client.extract_json_payload({"choices": []})
        self.assertIn("no choices", str(ctx.exception))

    def test_choices_not_list_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.client.extract_json_payload({"choices": "not-a-list"})
        self.assertIn("no choices", str(ctx.exception))

    def test_first_choice_not_dict_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.client.extract_json_payload({"choices": [None]})
        self.assertIn("Unsupported", str(ctx.exception))

    def test_content_not_string_or_list_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.client.extract_json_payload({"choices": [{"message": {"content": 123}}]})
        self.assertIn("Unsupported", str(ctx.exception))

    def test_content_is_none_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.client.extract_json_payload({"choices": [{"message": {"content": None}}]})
        self.assertIn("Unsupported", str(ctx.exception))

    def test_content_not_jsonable_string(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.client.extract_json_payload({"choices": [{"message": {"content": "not json"}}]})
        self.assertIn("invalid json", str(ctx.exception).lower())


class TestLLMClientRequestJson(unittest.TestCase):
    def setUp(self) -> None:
        self.client = LLMClient(ModelConfig(provider="test", model="test-model"))

    def test_request_json_passes_temperature_and_model(self) -> None:
        called_body: Dict[str, Any] = {}

        def mock_chat(*, body: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal called_body
            called_body = dict(body)
            return {"choices": [{"message": {"content": json.dumps({"key": "value"})}}]}

        self.client.chat_completion = mock_chat  # type: ignore[method-assign]

        result = self.client.request_json(
            system_prompt="test system",
            user_prompt="test user",
            temperature=0.5,
            model="custom-model",
        )
        self.assertEqual(result["key"], "value")
        self.assertEqual(called_body["model"], "custom-model")
        self.assertEqual(called_body["temperature"], 0.5)
        self.assertEqual(len(called_body["messages"]), 2)

    def test_request_json_uses_default_model(self) -> None:
        called_model: Optional[str] = None

        def mock_chat(*, body: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal called_model
            called_model = body["model"]
            return {"choices": [{"message": {"content": "{}"}}]}

        self.client.chat_completion = mock_chat  # type: ignore[method-assign]
        self.client.request_json(
            system_prompt="s", user_prompt="u", temperature=0.1,
        )
        self.assertEqual(called_model, "test-model")

    def test_request_json_response_format_is_json_object(self) -> None:
        called_body: Dict[str, Any] = {}

        def mock_chat(*, body: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal called_body
            called_body = dict(body)
            return {"choices": [{"message": {"content": "{}"}}]}

        self.client.chat_completion = mock_chat  # type: ignore[method-assign]
        self.client.request_json(
            system_prompt="s", user_prompt="u", temperature=0.1,
        )
        self.assertEqual(called_body.get("response_format"), {"type": "json_object"})


if __name__ == "__main__":
    unittest.main()