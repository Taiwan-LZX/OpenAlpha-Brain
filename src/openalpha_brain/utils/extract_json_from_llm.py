"""Extract JSON from LLM output — handles markdown code blocks, trailing commas, etc."""
from __future__ import annotations
import json
import re
import logging

logger = logging.getLogger(__name__)


def extract_json_from_llm(raw: str) -> dict | list | None:
    """
    Extract JSON from LLM raw output.
    
    Handles:
    - ```json ... ``` code blocks
    - ``` ... ``` (no language tag)
    - Trailing commas before } or ]
    - Leading/trailing non-JSON text
    
    Args:
        raw: Raw LLM response string
        
    Returns:
        Parsed JSON object/list, or None if parsing fails
    """
    if not raw or not isinstance(raw, str):
        return None

    text = raw.strip()

    # Try 1: Extract from markdown code block
    patterns = [
        r'```(?:json)?\s*\n?(.*?)\n?```',   # ```json ... ```
        r'```\s*\n?(.*?)\n?```',            # ``` ... ```
        r'(\{.*\})',                         # Raw { ... }
        r'(\[.*\])',                         # Raw [ ... ]
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            cleaned = _clean_json(match)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue

    # Try 2: Find first { or [ and parse to matching bracket
    for start_char in ['{', '[']:
        idx = text.find(start_char)
        if idx >= 0:
            candidate = text[idx:]
            cleaned = _clean_json(candidate)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue

    logger.debug("[extract_json] Failed to parse JSON from %d chars", len(text))
    return None


def _clean_json(text: str) -> str:
    """Clean common LLM JSON formatting issues."""
    # Remove trailing commas before } or ]
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    # Remove comments (# or // style)
    text = re.sub(r'//[^\n]*', '', text)
    text = re.sub(r'#[^\n]*', '', text)
    # Remove control characters except newline
    text = ''.join(c for c in text if c >= ' ' or c in '\n\r\t')
    return text.strip()
