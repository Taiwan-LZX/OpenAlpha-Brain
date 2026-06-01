from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from alpha_agent.config import ModelConfig


class LLMClient:
    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config

    def chat_completion(self, *, body: Dict[str, Any]) -> Dict[str, Any]:
        base = self.model_config.base_url.rstrip("/")
        url = f"{base}/chat/completions"
        api_key = os.getenv(self.model_config.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"No API key found in env var: {self.model_config.api_key_env}")
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.model_config.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP error: {exc.code} {raw[:400]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM connection error: {exc.reason}") from exc
        except OSError as exc:
            raise RuntimeError(f"LLM network error: {exc}") from exc

    def extract_json_payload(self, completion: Dict[str, Any]) -> Dict[str, Any]:
        choices = completion.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("LLM response has no choices.")
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            try:
                return json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(f"LLM returned invalid JSON: {exc}") from exc
        if isinstance(content, list):
            text_parts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ]
            return json.loads("".join(text_parts))
        raise ValueError("Unsupported LLM message content format.")

    def request_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        body = {
            "model": model or self.model_config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        response = self.chat_completion(body=body)
        return self.extract_json_payload(response)
