"""
OpenAlpha - Quant — Conversation History Intelligent Summarizer

When conversation history exceeds a threshold, this module:
1. Preserves all BRAIN feedback messages (never summarized)
2. Extracts structured info (operators, fields, errors, sharpe values)
3. Generates an LLM summary of older messages
4. Returns: brain_feedback + [summary_message] + recent_messages
"""

from __future__ import annotations

import logging
import re

from openalpha_brain.services import llm_client

logger = logging.getLogger(__name__)

_BRAIN_KEYWORDS = (
    "BRAIN PASS",
    "BRAIN FAIL",
    "BRAIN check FAILED",
    "BRAIN SIMULATION ERROR",
    "GATE FAIL",
    "GATE PASS",
    "real_sharpe",
    "real_fitness",
    "real_turnover",
    "mutation_attempt",
    "brain_alpha_id",
    "BRAIN improvement",
    "BRAIN mutation",
)

_OPS_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
_FIELDS_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b(?!\s*\()")
_SHARPE_RE = re.compile(r"[Ss]harpe[=:]\s*([0-9.\-]+)")
_ERROR_TYPE_RE = re.compile(
    r"(UNKNOWN_OPERATOR|INVALID_VAR|BRAIN_UNKNOWN_VAR|"
    r"unknown variable|INVALID BRAIN variables|"
    r"Unknown/forbidden operators|syntax error|"
    r"GATE FAIL|BRAIN SIMULATION ERROR)",
    re.IGNORECASE,
)


class ConversationSummarizer:
    def __init__(self, threshold: int = 20, keep_recent: int = 5) -> None:
        self.threshold = threshold
        self.keep_recent = keep_recent

    def _is_brain_feedback(self, msg: dict) -> bool:
        content = msg.get("content", "")
        return any(kw in content for kw in _BRAIN_KEYWORDS)

    def _extract_brain_feedback(self, messages: list[dict]) -> list[dict]:
        return [m for m in messages if self._is_brain_feedback(m)]

    def _extract_structured_info(self, messages: list[dict]) -> dict:
        operators: set[str] = set()
        fields: set[str] = set()
        error_types: set[str] = set()
        sharpe_values: list[str] = []

        for msg in messages:
            content = msg.get("content", "")
            if not content:
                continue

            found_ops = _OPS_RE.findall(content)
            operators.update(found_ops)

            all_vars = set(_FIELDS_RE.findall(content))
            fields.update(v for v in all_vars if v not in operators and not v.isdigit())

            for match in _ERROR_TYPE_RE.finditer(content):
                error_types.add(match.group(1))

            for match in _SHARPE_RE.finditer(content):
                sharpe_values.append(match.group(1))

        return {
            "operators_used": sorted(operators),
            "fields_used": sorted(fields),
            "error_types_encountered": sorted(error_types),
            "sharpe_values": sharpe_values,
        }

    async def _generate_llm_summary(self, messages: list[dict]) -> str:
        conversation_text = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            conversation_text.append(f"[{role}]: {content[:500]}")

        full_text = "\n\n".join(conversation_text)

        prompt = (
            "Summarize the following conversation about alpha factor generation, "
            "focusing on key patterns, successes, and failures. "
            "Be concise but preserve important details about what worked and what didn't.\n\n"
            f"{full_text}"
        )

        try:
            summary = await llm_client.generate(
                system_prompt="You are a concise conversation summarizer for alpha factor research.",
                history=[],
                user_msg=prompt,
                session_id="summarizer",
                cycle=0,
            )
            return summary.strip()
        except (ConnectionError, OSError, TimeoutError) as exc:
            logger.warning("LLM summary generation failed, falling back to rule-based: %s", exc)
            return ""

    def _build_rule_based_summary(self, structured_info: dict) -> str:
        parts = []

        ops = structured_info.get("operators_used", [])
        if ops:
            parts.append(f"Operators used: {', '.join(ops[:30])}")

        fields = structured_info.get("fields_used", [])
        if fields:
            parts.append(f"Fields/variables used: {', '.join(fields[:30])}")

        errors = structured_info.get("error_types_encountered", [])
        if errors:
            parts.append(f"Error types encountered: {', '.join(errors)}")

        sharpes = structured_info.get("sharpe_values", [])
        if sharpes:
            parts.append(f"Sharpe values observed: {', '.join(sharpes)}")

        if not parts:
            return "No structured information extracted from early conversation."

        return "Early conversation structured summary:\n" + "\n".join(parts)

    async def summarize_if_needed(self, messages: list[dict]) -> tuple[list[dict], bool]:
        if len(messages) <= self.threshold:
            return messages, False

        recent_messages = messages[-self.keep_recent :]
        early_messages = messages[: -self.keep_recent]

        brain_feedback = self._extract_brain_feedback(early_messages)

        non_brain_early = [m for m in early_messages if not self._is_brain_feedback(m)]

        structured_info = self._extract_structured_info(early_messages)

        llm_summary = await self._generate_llm_summary(non_brain_early)

        rule_summary = self._build_rule_based_summary(structured_info)

        if llm_summary:
            summary_content = (
                "[CONVERSATION SUMMARY]\n\n"
                f"## Natural Language Summary\n{llm_summary}\n\n"
                f"## Structured Information\n{rule_summary}"
            )
        else:
            summary_content = f"[CONVERSATION SUMMARY]\n\n{rule_summary}"

        summary_message = {
            "role": "system",
            "content": summary_content,
        }

        result = brain_feedback + [summary_message] + recent_messages

        logger.info(
            "Conversation summarized: %d early messages → %d brain feedback + 1 summary + %d recent = %d total",
            len(early_messages),
            len(brain_feedback),
            len(recent_messages),
            len(result),
        )

        return result, True
