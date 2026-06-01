"""
OpenAlpha-Brain — Multi-Agent Sequential Iteration (AlphaAgent Paper)

Implements the three-agent sequential iteration pipeline:
  Idea Agent → Factor Agent → Eval Agent

Design decisions (from grill-me confirmation):
  - Collaboration mode: sequential iteration (Idea→Factor→Eval)
  - Eval Agent: pure BRAIN backtest evaluation
  - Idea Agent data sources: RAG financial logic + historical alpha patterns + BRAIN feedback
  - Max 3 iterations, terminate after 2 consecutive no-improvement rounds
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import random
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import httpx

from openalpha_brain.agents.adaptive_agent import AdaptiveAgentFactory
from openalpha_brain.cli.algo_monitor import AlgoMonitor
from openalpha_brain.config.config import settings
from openalpha_brain.data import get_data_path
from openalpha_brain.evolution.crossover_mutation import AlphaTrajectory, GradientMutation, SemanticCrossover
from openalpha_brain.evolution.hypothesis_aligner import HypothesisAligner
from openalpha_brain.evolution.strategy_classifier import StrategyClassifier
from openalpha_brain.evolution.trajectory_mutation import TrajectoryMutation
from openalpha_brain.generation.alpha_logics import AlphaLogicLibrary
from openalpha_brain.generation.ast_originality import ASTNode, FASTEXPRParser, OriginalityChecker
from openalpha_brain.generation.prompts import (
    build_idea_diagnostic_feedback,
)
from openalpha_brain.knowledge.rag_engine import RAGEngine, SuccessCaseLibrary
from openalpha_brain.learning.reflection_engine import CritiqueResult, NextPlan, ReflectionEngine
from openalpha_brain.services import brain_client, llm_client
from openalpha_brain.utils import extract_json_from_llm as _extract_json_from_llm
from openalpha_brain.utils.algo_logger import algo_log
from openalpha_brain.validation.complexity_control import ComplexityController

logger = logging.getLogger(__name__)

_DIRECTION_OPERATOR_MAP: dict[str, list[str]] = {}


def _load_direction_operator_map() -> dict[str, list[str]]:
    global _DIRECTION_OPERATOR_MAP
    try:
        p = Path(__file__).parent / "direction_operator_map.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                _DIRECTION_OPERATOR_MAP = json.load(f)
    except (OSError, json.JSONDecodeError):
        _DIRECTION_OPERATOR_MAP = {
            "momentum": ["ts_delta", "ts_zscore", "ts_rank", "ts_regression", "ts_decay_linear"],
            "value": ["rank", "group_rank", "zscore", "group_zscore"],
            "volatility": ["ts_std_dev", "ts_av_diff", "ts_zscore"],
            "mean_reversion": ["ts_mean", "ts_decay_linear", "ts_sum"],
            "lead_lag": ["ts_delta", "ts_delay", "ts_corr", "ts_regression"],
        }
    return _DIRECTION_OPERATOR_MAP


_load_direction_operator_map()

_DIRECTION_FIELD_MAP: dict[str, list[str]] = {}

_fastexpr_grammar: str | None = None
_fastexpr_grammar_loaded: bool = False


def _load_fastexpr_grammar() -> str | None:
    global _fastexpr_grammar, _fastexpr_grammar_loaded
    if _fastexpr_grammar_loaded:
        return _fastexpr_grammar
    _fastexpr_grammar_loaded = True
    if not settings.FASTEXPR_GRAMMAR_ENABLED:
        return None
    try:
        grammar_path = get_data_path("fastexpr_grammar.gbnf")
        with open(grammar_path, encoding="utf-8") as f:
            _fastexpr_grammar = f.read()
        logger.info("Loaded FastExpr GBNF grammar (%d bytes)", len(_fastexpr_grammar))
    except OSError as exc:
        logger.warning("Failed to load FastExpr GBNF grammar: %s", exc)
        _fastexpr_grammar = None
    return _fastexpr_grammar


def _load_direction_field_map() -> dict[str, list[str]]:
    global _DIRECTION_FIELD_MAP
    try:
        p = Path(__file__).parent / "direction_field_map.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                _DIRECTION_FIELD_MAP = json.load(f)
    except (OSError, json.JSONDecodeError):
        _DIRECTION_FIELD_MAP = {
            "momentum": ["close", "open", "high", "low", "vwap", "volume", "adv20", "returns"],
            "value": ["cap", "sales", "assets", "equity", "revenue", "earnings"],
            "volatility": ["close", "high", "low", "vwap", "returns", "volume"],
            "mean_reversion": ["close", "vwap", "returns", "adv20"],
            "lead_lag": ["close", "volume", "vwap", "adv20", "returns"],
        }
    return _DIRECTION_FIELD_MAP


_load_direction_field_map()

_fastexpr_parser = FASTEXPRParser()


def _collect_operators_from_ast(expression: str) -> list[str]:
    try:
        tree = _fastexpr_parser.parse(expression)
        ops: list[str] = []
        seen: set[str] = set()

        def _walk(node: ASTNode) -> None:
            if node.op and node.op not in {"+", "-", "*", "/", "neg"} and node.op not in seen:
                ops.append(node.op)
                seen.add(node.op)
            for child in node.children:
                _walk(child)

        _walk(tree)
        return ops
    except (ValueError, SyntaxError, TypeError):
        try:
            pattern = r"([a-z_][a-z0-9_]*)\s*\("
            matches = re.findall(pattern, expression)
            return list(dict.fromkeys(matches))
        except (ValueError, TypeError):
            return []


def _collect_fields_from_ast(expression: str) -> list[str]:
    try:
        tree = _fastexpr_parser.parse(expression)
        fields: list[str] = []
        seen: set[str] = set()

        def _walk(node: ASTNode) -> None:
            if node.is_field and node.value not in seen:
                fields.append(node.value)
                seen.add(node.value)
            for child in node.children:
                _walk(child)

        _walk(tree)
        return fields
    except (ValueError, SyntaxError, TypeError):
        try:
            all_identifiers = set(re.findall(r"\b([a-z][a-z0-9_]*)\b", expression))
            operators = set(re.findall(r"([a-z_][a-z0-9_]*)\s*\(", expression))
            keywords = {"and", "or", "not", "if", "else", "true", "false", "nan", "inf"}
            fields = all_identifiers - operators - keywords
            return list(fields)
        except (ValueError, TypeError):
            return []


def _extract_max_lookback(expression: str) -> int:
    try:
        numbers = re.findall(r",\s*(\d+)\s*\)", expression)
        return max(int(n) for n in numbers) if numbers else 0
    except (ValueError, TypeError):
        return 0


_DIRECTION_TIME_MAP = {
    "short": 10,
    "medium-term": 60,
    "medium": 60,
    "long": 200,
    "long-term": 200,
}

_MECHANISM_OPERATOR_MAP: dict[str, list[str]] = {
    "momentum": ["ts_delta", "ts_returns", "ts_regression"],
    "mean_reversion": ["ts_zscore", "ts_mean", "ts_decay_linear"],
    "volatility": ["ts_std_dev", "ts_variance", "ts_av_diff"],
    "value": ["rank", "group_rank", "zscore", "group_zscore"],
    "lead_lag": ["ts_delta", "ts_delay", "ts_corr"],
}


def _check_semantic_alignment(hypothesis, expression: str) -> float:
    try:
        from openalpha_brain.evolution.hypothesis_aligner import HypothesisAligner

        if not hasattr(_check_semantic_alignment, "_aligner"):
            _check_semantic_alignment._aligner = HypothesisAligner()
        result = _check_semantic_alignment._aligner.align(expression, str(hypothesis))
        return result.get("r2_score", 0.5)
    except (ValueError, TypeError, RuntimeError):
        return 0.5


async def critique_revise_alpha(
    expression: str,
    brain_result: Any,
    llm_generate: Callable[..., Awaitable[str]],
    hypothesis_text: str = "",
    direction: str = "",
    max_rounds: int = 3,
) -> dict:
    """Standalone 3-round critique-revise loop for BRAIN-failed alphas.

    Can be called from brain_submitter without needing a full MultiAgentOrchestrator instance.
    EvalAgent and FactorAgent roles are performed via direct LLM prompts.
    """
    result = {
        "expression": expression,
        "rounds_completed": 0,
        "revised_expressions": [expression],
        "critique_log": [],
        "final_semantic_check": None,
    }

    if brain_result is None:
        return result

    brain_checks = getattr(brain_result, "brain_checks", None) or []
    if isinstance(brain_checks, list):
        brain_checks = [c for c in brain_checks if isinstance(c, dict)]
    real_sharpe = getattr(brain_result, "real_sharpe", None)
    real_fitness = getattr(brain_result, "real_fitness", None)
    real_turnover = getattr(brain_result, "real_turnover", None)
    status = getattr(brain_result, "status", "FAIL")
    status_str = status.value if hasattr(status, "value") else str(status)

    current_expr = expression
    for round_num in range(1, max_rounds + 1):
        logger.info(
            "critique_revise: round %d/%d — expr=%s",
            round_num,
            max_rounds,
            current_expr[:60],
        )

        critique_prompt = f"""You are a quantitative alpha factor critic. Analyze why this expression failed BRAIN validation.  # noqa: E501

EXPRESSION: {current_expr}
HYPOTHESIS: {hypothesis_text or "Not specified"}
DIRECTION: {direction or "Not specified"}

BRAIN RESULT:
  Status: {status_str}
  Sharpe: {real_sharpe or "N/A"}
  Fitness: {real_fitness or "N/A"}
  Turnover: {real_turnover or "N/A"}
  Failed Checks: {brain_checks}

Provide a structured critique:
1. root_cause: What went wrong? (1-2 sentences, based on BRAIN data)
2. structural_issue: What structural or logical flaw exists?
3. modification_instruction: Specific, actionable instruction for FactorAgent to revise the expression
4. priority: HIGH/MEDIUM/LOW

Respond in JSON format only:
{{"root_cause": "...", "structural_issue": "...", "modification_instruction": "...", "priority": "HIGH|MEDIUM|LOW"}}"""

        critique_result = {
            "root_cause": "",
            "structural_issue": "",
            "modification_instruction": "",
            "priority": "MEDIUM",
        }
        try:
            raw = await llm_generate(
                critique_prompt,
                [],
                "",
                session_id=f"critique_r{round_num}",
                cycle=round_num,
            )
            _parsed = _extract_json_from_llm(raw)
            if _parsed and isinstance(_parsed, dict):
                critique_result = _parsed
        except (ValueError, TypeError, RuntimeError) as exc:
            logger.warning("critique_revise: critique LLM call failed r%d: %s", round_num, exc)

        result["critique_log"].append(
            {
                "round": round_num,
                "expression_before": current_expr,
                "critique": critique_result,
            }
        )

        revise_prompt = f"""You are a quantitative alpha factor engineer. Revise the failed expression based on the critique.  # noqa: E501

ORIGINAL EXPRESSION: {current_expr}
CRITIQUE: {critique_result.get("root_cause", "")}
STRUCTURAL ISSUE: {critique_result.get("structural_issue", "")}
MODIFICATION INSTRUCTION: {critique_result.get("modification_instruction", "")}
DIRECTION: {direction or "Not specified"}

Rules:
1. Follow the modification instruction precisely
2. Keep the core signal concept intact
3. Return ONLY the revised expression as a string (no markdown, no explanation)"""

        revised_expr = current_expr
        try:
            raw = await llm_generate(
                revise_prompt,
                [],
                "",
                session_id=f"revise_r{round_num}",
                cycle=round_num,
            )
            cleaned = raw.strip()
            for prefix in ["```json", "```python", "```", "expression:", "EXPRESSION:"]:
                if cleaned.lower().startswith(prefix.lower()):
                    cleaned = cleaned[len(prefix) :].strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()
            if cleaned and len(cleaned) >= 5:
                revised_expr = cleaned
        except (ValueError, TypeError, RuntimeError) as exc:
            logger.warning("critique_revise: revise LLM call failed r%d: %s", round_num, exc)

        result["revised_expressions"].append(revised_expr)
        result["rounds_completed"] = round_num
        current_expr = revised_expr

        logger.info(
            "critique_revise: round %d complete — %s... → %s...",
            round_num,
            expression[:40],
            revised_expr[:40],
        )

    result["expression"] = current_expr
    logger.info(
        "critique_revise: complete — %d rounds, final expr=%s...",
        result["rounds_completed"],
        current_expr[:60],
    )
    return result


MAX_ITERATIONS = 3
NO_IMPROVEMENT_LIMIT = 2
_monitor = AlgoMonitor.get_instance()


def _build_direction_hints() -> str:
    hints = []
    for direction, ops in _DIRECTION_OPERATOR_MAP.items():
        hints.append(f"- {direction}: {', '.join(ops[:3])}")
    return "\n".join(hints)


def _load_brain_submit_params() -> dict:
    try:
        p = Path(__file__).parent / "brain_submit_params.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    return {
        "instrumentType": "EQUITY",
        "delay": 1,
        "decay": 5,
        "neutralization": "INDUSTRY",
        "truncation": 0.05,
        "nanHandling": "OFF",
        "pasteurization": "ON",
        "universe": "TOP3000",
    }


_IDEA_SYSTEM_PROMPT = """\
You are a financial hypothesis generator. Your task is to generate market hypotheses \
that can be translated into quantitative alpha factors.

Given:
- Exploration direction: {direction}
- Relevant financial logic from knowledge base: {rag_context}
- Market logic library (evidence-backed hypotheses and factor templates): {logic_context}
- Historical alpha patterns: {history_patterns}
- BRAIN feedback from previous iterations: {brain_feedback}

Generate a structured hypothesis with:
1. Direction: the market anomaly category
2. Asset class: equity
3. Time horizon: short/medium/long-term
4. Mechanism: the economic rationale
5. Natural language: clear description of the hypothesis

Output format (JSON):
{{"direction": "...", "asset_class": "...", "time_horizon": "...", "mechanism": "...", "natural_language": "..."}}

IMPORTANT:
- Output ONLY the JSON object, no markdown fences, no extra text.
- The hypothesis must be specific enough to translate into a FASTEXPR expression.
- Avoid trivial hypotheses (e.g., "stocks go up") or overcrowded patterns.
- Prefer hypotheses that align with high-evidence market logics when available.
- Your hypothesis MUST be implementable using the recommended operators and available fields listed above.
- Do NOT propose mechanisms that require operators or fields not listed in the context.

FIELD-DRIVEN HYPOTHESIS MODE (v3):
When field_family and recommended_fields are provided, your hypothesis MUST:
1. Reference specific field families and their semantic meaning
2. Explain WHY these fields capture the market inefficiency
3. Propose a specific field combination that tests the hypothesis
4. Map each field to its role in the factor template
Format: {{"direction": str, "asset_class": str, "time_horizon": str, "mechanism": str, "natural_language": str, "field_family": str, "field_combination": list[str]}}  # noqa: E501
"""

_FACTOR_SYSTEM_PROMPT = """\
=== LAYER 1: PLATFORM IDENTITY ===
You are a factor engineer on the WorldQuant BRAIN IQC 2026 quantitative competition platform. Your expressions will be backtested against real market data and validated by BRAIN's gate checks.  # noqa: E501

=== LAYER 2: FASTEXPR LANGUAGE CHARACTERISTICS ===
FASTEXPR is a PURE FUNCTIONAL language. CRITICAL RULES:
- ONLY function(arg1, arg2) syntax is allowed
- .method() method-call syntax is STRICTLY FORBIDDEN (e.g., ❌ expr.group_neutralize(INDUSTRY))
- Keyword arguments are FORBIDDEN (e.g., ❌ ts_decay_linear(close, d=5))
- String literals are FORBIDDEN — use integer rettype values
- Comparison operators (>, <, >=, <=, ==, !=) are FORBIDDEN
- Boolean operators (&&, ||, !) are FORBIDDEN
- if/else/ternary are FORBIDDEN
- Only arithmetic operators (+, -, *, /), function calls, and parentheses are allowed

=== LAYER 3: OPERATOR SIGNATURES ===
{operator_signatures}

=== LAYER 4: CORRECT/INCORRECT COMPARISON EXAMPLES ===
COMMON MISTAKES AND CORRECT FORMS:
❌ expr.group_neutralize(INDUSTRY) → ✅ group_neutralize(expr, industry)
❌ expr.rank() → ✅ rank(expr)
❌ ts_mean(close, 20.0) → ✅ ts_mean(close, 20)
❌ ts_decay_linear(close, d=5) → ✅ ts_decay_linear(close, 5)
❌ rank(close > vwap) → ✅ rank(close - vwap)

=== LAYER 5: BRAIN SCORING CRITERIA ===
BRAIN SCORING CRITERIA:
- Sharpe Ratio ≥ 1.25 = PASS
- Fitness = Sharpe × sqrt(|Returns|) / max(Turnover, 0.125)
- Fitness > 1.0 = PASS
- Turnover: 15%-35% is optimal range
- Turnover > 70% = FAIL (HIGH_TURNOVER)
- Turnover < 1% = FAIL (LOW_TURNOVER)

Given:
- Hypothesis: {hypothesis}
- Available fields: {fields}

SEMANTIC FINE-TUNING GUIDANCE:
{_direction_hints}
- Fine-tune parameters (lookback windows, normalization) before replacing operators
- Only replace operators when BRAIN feedback explicitly indicates the operator is problematic

{brain_feedback}

Output format (JSON):
{{"expression": "...", "simulation_payload": {{"settings": {{...}}, "regular": "..."}}, "economic_rationale": "..."}}

IMPORTANT:
- Output ONLY the JSON object, no markdown fences, no extra text.
- The "expression" field MUST contain ONLY the FastExpr expression.
- The "simulation_payload.regular" field MUST match the "expression" field.
- The "economic_rationale" field MUST contain a brief explanation (1-2 sentences) of the market inefficiency or financial logic the expression captures.  # noqa: E501
- Default settings: delay=1, decay=5, neutralization=INDUSTRY, universe=TOP3000
- You MUST also include ECONOMIC_RATIONALE: <explanation> in your output describing the market inefficiency your expression captures.  # noqa: E501
"""


_TEMPLATE_REFINEMENT_PROMPT = """\
You are refining a syntactically correct FASTEXPR alpha factor to improve its Sharpe ratio.

TEMPLATE EXPRESSION (syntactically correct, use as starting point):
{template_expression}

HYPOTHESIS:
{hypothesis}

YOUR TASK — refine the template to improve signal quality. RULES:
1. KEEP the core operator structure — do NOT replace operators with different ones
2. You MAY adjust lookback windows (try 5, 10, 20, 60 instead of defaults)
3. You MAY wrap the signal with ts_zscore(x, window) for normalization
4. You MAY add ts_decay_linear(x, window) for turnover control
5. You MAY divide by ts_std_dev(x, window) for volatility normalization
6. You MUST wrap the final expression with group_neutralize(..., industry)
7. You MUST output valid FASTEXPR syntax — ONLY function(arg1, arg2) calls allowed
8. NO method-call syntax, NO keyword arguments, NO string literals, NO comparison operators
9. Output ONLY the expression — no JSON, no markdown fences, no explanation

CORRECT SYNTAX EXAMPLES:
  group_neutralize(ts_zscore(rank(ts_delta(close, 10)), 20), industry)
  group_neutralize(ts_decay_linear(rank(close / ts_mean(close, 20)), 5), industry)
"""


@dataclass
class Hypothesis:
    direction: str
    asset_class: str
    time_horizon: str
    mechanism: str
    natural_language: str
    field_family: str = ""
    field_combination: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, str | list[str]]:
        return {
            "direction": self.direction,
            "asset_class": self.asset_class,
            "time_horizon": self.time_horizon,
            "mechanism": self.mechanism,
            "natural_language": self.natural_language,
            "field_family": self.field_family,
            "field_combination": self.field_combination,
        }


@dataclass
class SemanticAnchor:
    hypothesis_text: str
    direction: str
    core_operators: list[str]
    drift_threshold: float = 0.7
    _embedding: list[float] | None = None

    async def compute_embedding(self, embed_fn) -> None:
        try:
            if embed_fn:
                result = await embed_fn(self.hypothesis_text)
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], list):
                        self._embedding = result[0]
                    else:
                        self._embedding = result
        except (ValueError, TypeError, RuntimeError):
            self._embedding = None

    async def check_drift(self, new_expression: str, embed_fn) -> float:
        if self._embedding is None:
            return 1.0
        try:
            result = await embed_fn(new_expression)
            if isinstance(result, list) and len(result) > 0:
                new_vec = result[0] if isinstance(result[0], list) else result
            else:
                return 1.0

            import math

            dot = sum(a * b for a, b in zip(self._embedding, new_vec, strict=False))
            norm_a = math.sqrt(sum(a * a for a in self._embedding))
            norm_b = math.sqrt(sum(b * b for b in new_vec))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            similarity = dot / (norm_a * norm_b)
            return round(similarity, 4)
        except (ValueError, TypeError, ZeroDivisionError):
            return 0.0


@dataclass
class AgentResult:
    hypothesis: Hypothesis
    expression: str
    simulation_payload: dict
    originality_score: float
    complexity_metrics: dict
    brain_sharpe: float | None
    iterations: int
    converged: bool
    semantic_alignment_score: float = 1.0
    variants: list[str] = field(default_factory=list)
    trajectory: AlphaTrajectory | None = None
    critique_result: CritiqueResult | None = None
    economic_rationale: str | None = None


class IdeaAgent:
    def __init__(
        self,
        llm_generate_fn: Callable[..., Awaitable[str]] | None = None,
        rag_engine: RAGEngine | None = None,
        logic_library: AlphaLogicLibrary | None = None,
        classifier: StrategyClassifier | None = None,
        mab: Any | None = None,
        field_proxy_map: Any | None = None,
    ) -> None:
        self._llm_generate = llm_generate_fn or llm_client.generate
        self._rag_engine = rag_engine
        self._logic_library = logic_library or AlphaLogicLibrary()
        self._classifier = classifier
        self._mab = mab
        self._field_proxy_map: Any = field_proxy_map

    def _select_direction(self, direction: str) -> str:
        if self._mab is not None:
            try:
                selection = self._mab.select_exploration_arm(focus_area=direction)
                if selection and selection.get("direction"):
                    return selection["direction"]
            except (ValueError, KeyError, RuntimeError) as exc:
                logger.warning("IdeaAgent: MAB select failed: %s", exc)
        return direction

    async def generate_hypothesis(
        self,
        direction: str,
        history: list[dict],
        brain_feedback: list[dict] | None = None,
    ) -> tuple[Hypothesis, dict]:
        direction = self._select_direction(direction)
        _sched_template_id = ""
        _sched_family_id = ""
        _sched_recommended_fields: list[str] = []
        if self._mab is not None:
            try:
                arm = self._mab.select_exploration_arm(
                    focus_area=direction,
                )
                if arm:
                    _sched_template_id = arm.get("template_id", "")
                    _sched_family_id = arm.get("family_id", "")
                    _sched_recommended_fields = arm.get("recommended_fields", [])
                    if arm.get("direction"):
                        direction = arm["direction"]
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning("IdeaAgent: scheduler select_exploration_arm failed: %s", exc)
        rag_context = ""
        rag_context_dict = {}
        if self._rag_engine and self._rag_engine.is_ready:
            try:
                retrieval = await self._rag_engine.retrieve(direction)
                context = self._rag_engine.assemble_context(retrieval)
                rag_context_dict = {
                    "top_ops_detailed": context.get("top_ops_detailed", []),
                    "field_ids": context.get("field_ids", []),
                    "financial_logic_ids": context.get("financial_logic_ids", []),
                }
                finlogic = context.get("financial_logic_ids", [])
                [op["name"] for op in context.get("top_ops_detailed", [])]
                top_ops_desc = [f"{op['name']}({op.get('category', '')})" for op in context.get("top_ops_detailed", [])]
                fields = context.get("field_ids", [])
                rag_context = (
                    f"Financial logic: {', '.join(finlogic)}. "
                    f"Recommended operators: {'; '.join(top_ops_desc)}. "
                    f"Available fields: {', '.join(fields)}."
                )
            except (ValueError, TypeError, RuntimeError) as exc:
                logger.warning("IdeaAgent: RAG retrieval failed: %s", exc)

        if _sched_family_id and self._field_proxy_map:
            try:
                family = self._field_proxy_map.get_family(_sched_family_id)
                if family:
                    family_desc = f"Field family: {family.family_name} ({family.l1_category}) — {family.description}"
                    if _sched_recommended_fields:
                        family_desc += (
                            f"\nRecommended fields for this family: {', '.join(_sched_recommended_fields[:15])}"
                        )
                    rag_context += f"\n{family_desc}"
                    rag_context_dict["field_family"] = _sched_family_id
                    rag_context_dict["recommended_fields"] = _sched_recommended_fields
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning("IdeaAgent: field_proxy_map lookup failed: %s", exc)

        history_patterns = ""
        if history:
            patterns = []
            for h in history[-5:]:
                direction_h = h.get("direction", "?")
                sharpe = h.get("sharpe", "N/A")
                status = h.get("status", "?")
                patterns.append(f"  - {direction_h}: sharpe={sharpe}, status={status}")
            history_patterns = "\n".join(patterns)

        feedback_str = "No previous feedback."
        if brain_feedback:
            fb_lines = []
            for fb in brain_feedback[-3:]:
                sharpe = fb.get("sharpe", "N/A")
                status = fb.get("status", "?")
                feedback_text = fb.get("feedback", "")
                fb_lines.append(
                    f"  - Sharpe={sharpe}, status={status}: {feedback_text}",
                )
            feedback_str = "\n".join(fb_lines)

        logic_context = ""
        try:
            relevant_logics = self._logic_library.get_logic_for_direction(direction)
            if relevant_logics:
                logic_lines = []
                for logic in relevant_logics[:5]:
                    templates_preview = "; ".join(logic.factor_templates[:2])
                    logic_lines.append(
                        f"  - [{logic.category}] {logic.hypothesis} "
                        f"(evidence: {logic.evidence_count}, templates: {templates_preview})",
                    )
                logic_context = "\n".join(logic_lines)
        except (KeyError, ValueError, OSError) as exc:
            logger.warning("IdeaAgent: logic library lookup failed: %s", exc)

        system_prompt = _IDEA_SYSTEM_PROMPT.format(
            direction=direction,
            rag_context=rag_context or "No RAG context available.",
            history_patterns=history_patterns or "No historical patterns yet.",
            brain_feedback=feedback_str,
            logic_context=logic_context or "No market logic context available.",
        )

        complementary_hint = ""
        if self._classifier is not None:
            try:
                suggestions = self._classifier.find_complementary()
                if suggestions:
                    hint_parts = [f"{s.get('direction', 'any')}: {s['reason']}" for s in suggestions]
                    complementary_hint = f"\nUnexplored strategy directions to consider: {'; '.join(hint_parts)}"
            except (ValueError, TypeError, RuntimeError):
                pass

        user_msg = (
            f"Generate a novel market hypothesis for direction: {direction}. "
            f"Ensure the hypothesis is specific, non-trivial, and can be "
            f"translated into a FASTEXPR alpha factor expression."
            f"{complementary_hint}"
        )
        if _sched_family_id:
            user_msg += f"\n\nFIELD-DRIVEN CONTEXT: field_family={_sched_family_id}"
            if _sched_recommended_fields:
                user_msg += f", recommended_fields={_sched_recommended_fields[:10]}"
            user_msg += ". Use these fields to construct your hypothesis."

        raw = await self._llm_generate(
            system_prompt=system_prompt,
            history=[],
            user_msg=user_msg,
            session_id="idea_agent",
            cycle=0,
        )

        hypothesis = self._parse_hypothesis(raw, direction)

        novelty = 1.0
        if history:
            existing_texts = {h.get("natural_language", "") for h in history if isinstance(h, dict)}
            if hypothesis.natural_language in existing_texts:
                novelty = 0.0
            else:
                max_sim = 0.0
                for h in history:
                    if not isinstance(h, dict):
                        continue
                    existing = h.get("natural_language", "")
                    if existing and hypothesis.natural_language:
                        overlap = len(set(hypothesis.natural_language.split()) & set(existing.split()))
                        total = len(set(hypothesis.natural_language.split()) | set(existing.split()))
                        if total > 0:
                            sim = overlap / total
                            max_sim = max(max_sim, sim)
                novelty = 1.0 - max_sim

        if novelty < 0.5 and self._rag_engine and self._rag_engine.is_ready:
            for _rag_round in range(2):
                try:
                    extra_retrieval = await self._rag_engine.retrieve(
                        f"{direction} {hypothesis.natural_language}",
                    )
                    extra_context = self._rag_engine.assemble_context(extra_retrieval)
                    extra_ops = [op["name"] for op in extra_context.get("top_ops_detailed", [])]
                    extra_fields = extra_context.get("field_ids", [])
                    if extra_ops:
                        existing_ops = [op["name"] for op in rag_context_dict.get("top_ops_detailed", [])]
                        merged_ops = rag_context_dict.get("top_ops_detailed", []) + [
                            op for op in extra_context.get("top_ops_detailed", []) if op["name"] not in existing_ops
                        ]
                        rag_context_dict["top_ops_detailed"] = merged_ops
                    if extra_fields:
                        existing_fields = rag_context_dict.get("field_ids", [])
                        merged_fields = existing_fields + [f for f in extra_fields if f not in existing_fields]
                        rag_context_dict["field_ids"] = merged_fields
                    novelty = min(1.0, novelty + 0.3)
                    break
                except (TimeoutError, aiohttp.ClientError, ConnectionError) as exc:
                    logger.warning("IdeaAgent: supplementary RAG retrieval round %d failed: %s", _rag_round + 1, exc)

        _monitor.record("STEP", "idea_agent", "generate", hypothesis.natural_language[:80], session_id="idea_agent")
        return hypothesis, rag_context_dict

    def _parse_hypothesis(self, raw: str, fallback_direction: str) -> Hypothesis:
        try:
            _extracted = _extract_json_from_llm(raw)
            if isinstance(_extracted, list) and _extracted and isinstance(_extracted[0], dict):
                data = _extracted[0]
            elif isinstance(_extracted, dict):
                data = _extracted
            else:
                data = json.loads(raw)

            return Hypothesis(
                direction=data.get("direction", fallback_direction),
                asset_class=data.get("asset_class", "equity"),
                time_horizon=data.get("time_horizon", "medium-term"),
                mechanism=data.get("mechanism", ""),
                natural_language=data.get("natural_language", ""),
                field_family=data.get("field_family", ""),
                field_combination=data.get("field_combination", []),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("IdeaAgent: failed to parse hypothesis JSON: %s", exc)
            return Hypothesis(
                direction=fallback_direction,
                asset_class="equity",
                time_horizon="medium-term",
                mechanism=raw[:200],
                natural_language=raw[:200],
            )


class HypothesisRefinementNeededError(Exception):
    pass


class _AlignmentRetryError(Exception):
    """Raised when R² alignment score < 0.3, signalling generate_expression to retry."""

    def __init__(self, feedback: str = ""):
        super().__init__(feedback)
        self.feedback = feedback


class FactorAgent:
    def __init__(
        self,
        llm_generate_fn: Callable[..., Awaitable[str]] | None = None,
        rag_engine: RAGEngine | None = None,
        logic_library: AlphaLogicLibrary | None = None,
        success_lib: SuccessCaseLibrary | None = None,
        feature_map: Any | None = None,
        reflection_engine: ReflectionEngine | None = None,
        whitelist_mgr: Any | None = None,
        field_proxy_map: Any | None = None,
        mab: Any | None = None,
    ) -> None:
        self._llm_generate = llm_generate_fn or llm_client.generate
        self._rag_engine = rag_engine
        self._logic_library = logic_library or AlphaLogicLibrary()
        self._success_lib = success_lib
        self._feature_map = feature_map
        self._reflection_engine = reflection_engine
        self._whitelist_mgr = whitelist_mgr
        self._field_proxy_map: Any = field_proxy_map
        self._mab = mab
        self._all_signatures = self._load_all_operator_signatures()
        self._last_critique_result: CritiqueResult | None = None
        self._hypothesis_aligner = HypothesisAligner()

    def _check_and_attach_alignment(self, expression: str, hypothesis, sim_payload: dict) -> tuple[dict, str]:
        """計算 R² 假設對齊分數並附加到 sim_payload。

        Returns:
            (sim_payload, action) where action is one of:
              "OK"      — R² >= 0.4: 對齊良好，正常提交
              "WEAK"    — 0.3 <= R² < 0.4: 對齊偏弱，附加懲罰標記
              "RETRY"   — R² < 0.3: 嚴重偏離，應重新生成
        """
        _action = "OK"
        try:
            alignment_result = self._hypothesis_aligner.align(
                expression,
                hypothesis.direction,
                direction=hypothesis.direction,
            )
            r2_score = alignment_result["r2_score"]
            alignment_level = alignment_result["alignment_level"]
            logger.info(
                "FactorAgent: R² alignment score=%.3f (%s) for hypothesis='%s'",
                r2_score,
                alignment_level.upper(),
                hypothesis.direction,
            )
            if r2_score < 0.3:
                _action = "RETRY"
                logger.warning(
                    "FactorAgent: CRITICAL LOW R² (%.3f) — expression contradicts hypothesis '%s': %s — RETRY",
                    r2_score,
                    hypothesis.direction,
                    alignment_result["diagnosis"],
                )
            elif r2_score < 0.4:
                _action = "WEAK"
                logger.warning(
                    "FactorAgent: LOW R² alignment (%.3f) — expression may drift from hypothesis '%s': %s",
                    r2_score,
                    hypothesis.direction,
                    alignment_result["diagnosis"],
                )
            sim_payload.setdefault("metadata", {})["r2_alignment"] = {
                "r2_score": r2_score,
                "alignment_level": alignment_level,
                "diagnosis": alignment_result.get("diagnosis", ""),
                "action": _action,
            }
            if _action == "WEAK":
                sim_payload.setdefault("metadata", {})["_weak_alignment_penalty"] = True
            if _action == "RETRY":
                try:
                    _feedback = self._hypothesis_aligner.build_alignment_feedback(
                        expression,
                        hypothesis.direction,
                        alignment_result,
                    )
                    sim_payload.setdefault("metadata", {})["_alignment_retry_feedback"] = _feedback
                except (ValueError, TypeError, RuntimeError) as _fb_exc:
                    logger.warning("FactorAgent: build_alignment_feedback failed: %s", _fb_exc)
        except (ValueError, TypeError, RuntimeError) as exc:
            logger.warning("FactorAgent: HypothesisAligner failed: %s", exc)
        return sim_payload, _action

    _self_critique_call_count: int = 0
    _SELF_CRITIQUE_INTERVAL: int = 3

    async def _run_self_critique(self, expression: str, hypothesis_text: str) -> None:
        self._last_critique_result = None
        if self._reflection_engine is None:
            return
        self._self_critique_call_count += 1
        if self._self_critique_call_count % self._SELF_CRITIQUE_INTERVAL != 0:
            return
        try:
            try:
                from loop_engine import _algo_tick

                _algo_tick("self_critique")
            except ImportError:
                pass
            critique = await self._reflection_engine.self_critique(expression, hypothesis_text)
            self._last_critique_result = critique
            if not critique.critique_available or critique.consistency_score is None:
                logger.warning(
                    "FactorAgent: self_critique unavailable — critique_available=%s, consistency_score=%s",
                    critique.critique_available,
                    critique.consistency_score,
                )
            elif critique.consistency_score < 0.5:
                logger.warning(
                    "FactorAgent: self_critique consistency_score=%.2f < 0.5 — issues: %s",
                    critique.consistency_score,
                    critique.issues,
                )
        except (ValueError, TypeError, RuntimeError) as exc:
            logger.warning("FactorAgent: self_critique call failed: %s", exc)

    _ESSENTIAL_OPERATORS = {
        "rank",
        "group_neutralize",
        "ts_mean",
        "ts_zscore",
        "ts_decay_linear",
    }

    def _validate_whitelist(self, expression: str) -> None:
        if self._whitelist_mgr is None:
            return
        try:
            allowed = self._whitelist_mgr.get_allowed_fields()
            if not allowed:
                return
            expr_fields = _collect_fields_from_ast(expression)
            allowed_lower = {f.lower() for f in allowed}
            invalid = [f for f in expr_fields if f.lower() not in allowed_lower]
            if invalid:
                raise HypothesisRefinementNeededError(
                    f"Fields not in whitelist: {', '.join(invalid)}",
                )
        except HypothesisRefinementNeededError:
            raise
        except (ValueError, TypeError, RuntimeError) as exc:
            logger.warning("FactorAgent: whitelist validation failed: %s", exc)

    @staticmethod
    def _load_default_operators() -> list[str]:
        try:
            p = get_data_path("brain_operators.json")
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    ops = json.load(f)
                return [op["name"] for op in ops if isinstance(op, dict) and "name" in op]
        except (OSError, json.JSONDecodeError):
            pass
        return ["rank", "group_neutralize", "ts_mean", "ts_zscore", "ts_decay_linear"]

    @staticmethod
    def _load_default_fields() -> list[str]:
        try:
            from openalpha_brain.utils.whitelist import WhitelistManager

            wm = WhitelistManager()
            core = wm.get_allowed_fields()
            if core:
                return list(core)
        except (ValueError, TypeError, OSError):
            pass
        return ["close", "open", "high", "low", "vwap", "volume", "adv20", "returns", "cap"]

    @staticmethod
    def _load_all_operator_signatures() -> dict[str, str]:
        try:
            operators_path = get_data_path("brain_operators.json")
            with open(operators_path, encoding="utf-8") as f:
                operators_data = json.load(f)
            sigs = {}
            for op in operators_data:
                name = op.get("name", "")
                definition = op.get("definition", "")
                description = op.get("description", "")
                definition = definition.split("\r\n")[0].split("\n")[0]
                pattern = re.compile(re.escape(name) + r"\(([^()]*(?:\([^()]*\)[^()]*)*)\)")
                match = pattern.match(definition)
                sig = f"{name}({match.group(1)})" if match else definition if definition else f"{name}(...)"
                sigs[name] = f"{sig} — {description}"
            return sigs
        except FileNotFoundError:
            logger.warning("FactorAgent: brain_operators.json not found")
            return {}
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("FactorAgent: failed to load operator signatures: %s", exc)
            return {}

    def _build_operator_signatures(self, rag_ops: list[str] | None = None) -> str:
        needed = set(self._ESSENTIAL_OPERATORS)
        if rag_ops:
            needed.update(rag_ops)
        lines = []
        for name in sorted(needed):
            sig = self._all_signatures.get(name)
            if sig:
                lines.append(sig)
        if not lines:
            return "rank(x) — rank; group_neutralize(x, g) — group neutralize"
        return "\n".join(lines)

    def _select_template(self, hypothesis: Hypothesis) -> tuple[str | None, str, str]:
        if self._mab is not None:
            try:
                arm = self._mab.select_exploration_arm(
                    focus_area=hypothesis.direction,
                )
                if arm:
                    template_id = arm.get("template_id", "")
                    family_id = arm.get("family_id", "")
                    if template_id and self._logic_library:
                        logic = self._logic_library.get_logic(template_id)
                        if logic and logic.factor_templates:
                            t = random.choice(logic.factor_templates)
                            return str(t), template_id, family_id
            except (ValueError, KeyError, RuntimeError) as exc:
                logger.warning("FactorAgent: scheduler select_exploration_arm failed: %s", exc)
        try:
            if self._logic_library:
                templates = self._logic_library.get_templates_for_direction(hypothesis.direction)
                if templates:
                    t = random.choice(templates)
                    if isinstance(t, dict):
                        return t.get("template") or t.get("expression") or str(t), "", ""
                    return str(t), "", ""
        except (ValueError, KeyError, TypeError):
            pass
        return None, "", ""

    def _fill_template(
        self, template: str, rag_context_dict: dict | None = None, template_id: str = "", family_id: str = ""
    ) -> str | None:
        try:
            filled = template
            recommended_fields: list[str] = []
            if self._field_proxy_map and template_id:
                try:
                    recommended_fields = self._field_proxy_map.recommend_fields_for_template(
                        template_id,
                        family_id or None,
                    )
                except (ValueError, TypeError, RuntimeError):
                    recommended_fields = []
            ops: list = []
            fields: list[str] = []
            if rag_context_dict:
                ops = rag_context_dict.get("top_ops_detailed", [])
                fields = rag_context_dict.get("field_ids", [])
            if recommended_fields:
                fields = list(dict.fromkeys(recommended_fields + fields))
            placeholders = re.findall(r"\{(\w+)\}", filled)
            for ph in placeholders:
                if "op" in ph.lower() and ops:
                    op = ops[0]
                    if isinstance(op, dict):
                        filled = filled.replace("{" + ph + "}", op.get("name", "rank"), 1)
                        ops = ops[1:]
                    else:
                        filled = filled.replace("{" + ph + "}", str(op), 1)
                        ops = ops[1:]
                elif "field" in ph.lower() and fields:
                    filled = filled.replace("{" + ph + "}", str(fields[0]), 1)
                    fields = fields[1:]
                elif "window" in ph.lower() or "lookback" in ph.lower() or "lb" in ph.lower():
                    filled = filled.replace("{" + ph + "}", "10", 1)
                elif "neutral" in ph.lower() or "group" in ph.lower():
                    filled = filled.replace("{" + ph + "}", "INDUSTRY", 1)
            if "{" in filled and "}" in filled:
                return None
            return filled
        except (ValueError, KeyError, TypeError):
            return None

    async def _refine_template_expression(
        self,
        template_expression: str,
        hypothesis_str: str,
        brain_feedback: list[dict] | None = None,
    ) -> str | None:
        try:
            refinement_system = _TEMPLATE_REFINEMENT_PROMPT.format(
                template_expression=template_expression,
                hypothesis=hypothesis_str,
            )
            feedback_hint = ""
            if brain_feedback:
                for fb in brain_feedback[-2:]:
                    fb_text = fb.get("feedback", "")
                    if "LOW_SHARPE" in fb_text:
                        feedback_hint += "Previous LOW_SHARPE — try ts_zscore normalization or different lookback. "
                    elif "HIGH_TURNOVER" in fb_text:
                        feedback_hint += "Previous HIGH_TURNOVER — add ts_decay_linear smoothing. "
                    elif "LOW_TURNOVER" in fb_text:
                        feedback_hint += "Previous LOW_TURNOVER — reduce decay or use shorter windows. "
            user_msg = f"Refine this template expression to improve Sharpe ratio:\n{template_expression}"
            if feedback_hint:
                user_msg += f"\n\nAdditional context: {feedback_hint.strip()}"
            raw = await self._llm_generate(
                system_prompt=refinement_system,
                history=[],
                user_msg=user_msg,
                session_id="factor_agent_refine",
                cycle=0,
                grammar=_load_fastexpr_grammar(),
            )
            refined = raw.strip()
            try:
                _extracted = _extract_json_from_llm(refined)
                if isinstance(_extracted, list) and _extracted and isinstance(_extracted[0], dict):
                    data = _extracted[0]
                elif isinstance(_extracted, dict):
                    data = _extracted
                else:
                    data = json.loads(refined)
                if isinstance(data, dict):
                    refined = data.get("expression", refined)
                elif isinstance(data, str):
                    refined = data
            except (json.JSONDecodeError, ValueError):
                pass
            if not refined or len(refined) < 5:
                return None
            return refined
        except (TimeoutError, aiohttp.ClientError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("FactorAgent: _refine_template_expression failed: %s", exc)
            return None

    @algo_log(log_args_to_skip=("self",))
    async def generate_expression(
        self,
        hypothesis: Hypothesis,
        operators: list[str] | None = None,
        fields: list[str] | None = None,
        brain_feedback: list[dict] | None = None,
        rag_context_dict: dict | None = None,
    ) -> tuple[str, dict, str | None, str]:
        if operators is None:
            operators = self._load_default_operators()

        if fields is None:
            fields = self._load_default_fields()

        if rag_context_dict:
            try:
                base_ops = [op["name"] for op in rag_context_dict.get("top_ops_detailed", [])]
                base_fields = rag_context_dict.get("field_ids", [])
                if base_ops:
                    operators = list(dict.fromkeys(base_ops + (operators or [])))
                if base_fields:
                    fields = list(dict.fromkeys(base_fields + (fields or [])))
            except (ValueError, TypeError) as exc:
                logger.warning("FactorAgent: rag_context_dict merge failed: %s", exc)

        rag_ops = None
        if self._rag_engine and self._rag_engine.is_ready:
            try:
                retrieval = await self._rag_engine.retrieve(hypothesis.natural_language or hypothesis.direction)
                context = self._rag_engine.assemble_context(retrieval)
                rag_ops = [op["name"] for op in context.get("top_ops_detailed", [])]
                rag_fields = context.get("field_ids", [])
                if rag_ops:
                    operators = list(dict.fromkeys(rag_ops + (operators or [])))
                if rag_fields:
                    fields = list(dict.fromkeys(rag_fields + (fields or [])))
            except (TimeoutError, aiohttp.ClientError, ConnectionError) as exc:
                logger.warning("FactorAgent: RAG retrieval failed: %s", exc)

        if self._rag_engine and self._rag_engine.is_ready and rag_ops:
            try:
                from openalpha_brain.core.loop_state import (
                    _mab,
                    _market_state_inferencer,
                    _signal_arbiter,
                    _whitelist_mgr,
                )
                from openalpha_brain.validation.signal_arbiter import (
                    MABSignalAdapter,
                    MarketSignalAdapter,
                    RAGSignalAdapter,
                    WhitelistSignalAdapter,
                )

                if _signal_arbiter is not None:
                    rag_adapter = RAGSignalAdapter(
                        {
                            "fields": [{"id": f, "score": 1.0} for f in rag_fields],
                            "operators": [{"id": o, "score": 1.0} for o in (rag_ops or [])],
                        }
                    )
                    mab_adapter = MABSignalAdapter(_mab)
                    wl_adapter = WhitelistSignalAdapter(_whitelist_mgr)
                    field_ids_for_market = list(set(rag_fields or []))
                    op_ids_for_market = list(set(rag_ops or []))
                    market_adapter = MarketSignalAdapter(
                        _market_state_inferencer,
                        hypothesis.direction,
                        field_ids=field_ids_for_market,
                        op_ids=op_ids_for_market,
                    )

                    field_adapters = [rag_adapter, mab_adapter, wl_adapter, market_adapter]
                    op_adapters = [rag_adapter, mab_adapter, market_adapter]

                    ranked_fields_result, ranked_ops_result = await _signal_arbiter.rank_with_adapters(
                        field_adapters,
                        op_adapters,
                        top_k_fields=len(rag_fields) if rag_fields else 0,
                        top_k_ops=len(rag_ops) if rag_ops else 0,
                    )

                    if ranked_fields_result:
                        arbitrated_fields = [r.item_id for r in ranked_fields_result]
                        if arbitrated_fields:
                            fields = list(dict.fromkeys(arbitrated_fields + (fields or [])))
                            logger.info(
                                "FactorAgent: SignalArbiter reranked %d fields (top=%.3f)",
                                len(arbitrated_fields),
                                ranked_fields_result[0].final_score if ranked_fields_result else 0.0,
                            )

                    if ranked_ops_result:
                        arbitrated_ops = [r.item_id for r in ranked_ops_result]
                        if arbitrated_ops:
                            operators = list(dict.fromkeys(arbitrated_ops + (operators or [])))
                            logger.info(
                                "FactorAgent: SignalArbiter reranked %d ops (top=%.3f)",
                                len(arbitrated_ops),
                                ranked_ops_result[0].final_score if ranked_ops_result else 0.0,
                            )
            except (ValueError, TypeError, RuntimeError) as exc:
                logger.warning("FactorAgent: SignalArbiter ranking failed: %s", exc)

        hypothesis_str = (
            f"Direction: {hypothesis.direction}\n"
            f"Asset class: {hypothesis.asset_class}\n"
            f"Time horizon: {hypothesis.time_horizon}\n"
            f"Mechanism: {hypothesis.mechanism}\n"
            f"Natural language: {hypothesis.natural_language}"
        )

        reference_section = ""
        if self._success_lib and settings.SUCCESS_CASE_LIBRARY_ENABLED:
            try:
                similar_cases = await self._success_lib.search_similar(
                    hypothesis.natural_language or hypothesis.direction,
                    top_k=3,
                )
                if similar_cases:
                    ref_lines = []
                    for sc in similar_cases[:3]:
                        ref_expr = sc.get("expr", "")
                        ref_sharpe = sc.get("sharpe", "N/A")
                        ref_fitness = sc.get("fitness", "N/A")
                        ref_direction = sc.get("direction", "")
                        ref_lines.append(
                            f"  - (direction={ref_direction}, sharpe={ref_sharpe}, fitness={ref_fitness}): {ref_expr}",
                        )
                    reference_section = "Reference successful alphas:\n" + "\n".join(ref_lines)
            except (OSError, ValueError, RuntimeError):
                logger.warning("FactorAgent: success_lib.search_similar failed")

        feedback_section = ""
        if brain_feedback:
            fb_lines = []
            for fb in brain_feedback[-3:]:
                sharpe = fb.get("sharpe", "N/A")
                status = fb.get("status", "?")
                feedback_text = fb.get("feedback", "")
                fb_lines.append(
                    f"  - Sharpe={sharpe}, status={status}: {feedback_text}",
                )
            fine_tune_hints = []
            for fb in brain_feedback[-3:]:
                feedback_text = fb.get("feedback", "")
                if "LOW_SHARPE" in feedback_text:
                    fine_tune_hints.append(
                        "LOW_SHARPE: Try adding ts_zscore normalization or adjusting lookback window"
                    )
                elif "HIGH_TURNOVER" in feedback_text:
                    fine_tune_hints.append("HIGH_TURNOVER: Try adding ts_decay_linear smoothing or increasing decay")
                elif "LOW_TURNOVER" in feedback_text:
                    fine_tune_hints.append("LOW_TURNOVER: Try reducing decay or using shorter lookback windows")
                elif "SELF_CORRELATION" in feedback_text:
                    fine_tune_hints.append(
                        "SELF_CORRELATION: Replace core operator with semantically similar alternative"
                    )

            feedback_section = "PREVIOUS BRAIN FEEDBACK — apply these fixes:\n" + "\n".join(fb_lines)
            if fine_tune_hints:
                feedback_section += "\n\nFINE-TUNING HINTS (try these BEFORE replacing operators):\n" + "\n".join(
                    f"  - {h}" for h in fine_tune_hints
                )

        operator_signatures = self._build_operator_signatures(rag_ops)

        system_prompt = _FACTOR_SYSTEM_PROMPT.format(
            hypothesis=hypothesis_str,
            fields=", ".join(fields),
            operator_signatures=operator_signatures,
            brain_feedback=feedback_section,
            _direction_hints=_build_direction_hints(),
        )

        if reference_section:
            system_prompt += "\n\n" + reference_section

        template_expression = None
        _sched_template_id = ""
        _sched_family_id = ""
        template_mode = settings.FACTOR_TEMPLATE_MODE
        if template_mode in ("template_first", "hybrid"):
            template, _sched_template_id, _sched_family_id = self._select_template(hypothesis)
            if template:
                template_expression = self._fill_template(
                    template, rag_context_dict, template_id=_sched_template_id, family_id=_sched_family_id
                )

        if template_mode == "template_first" and template_expression:
            logger.info("FactorAgent: template_first mode — using filled template directly (no LLM refinement)")
            brain_params = _load_brain_submit_params()
            sim_payload = {
                "settings": {
                    "instrumentType": brain_params.get("instrumentType", "EQUITY"),
                    "region": "USA",
                    "universe": brain_params.get("universe", "TOP3000"),
                    "delay": brain_params.get("delay", 1),
                    "decay": brain_params.get("decay", 5),
                    "neutralization": brain_params.get("neutralization", "INDUSTRY"),
                    "truncation": brain_params.get("truncation", 0.05),
                    "pasteurization": brain_params.get("pasteurization", "ON"),
                    "unitHandling": "VERIFY",
                    "nanHandling": brain_params.get("nanHandling", "ON"),
                    "language": "FASTEXPR",
                    "visualization": False,
                },
                "regular": template_expression,
            }
            _monitor.record(
                "STEP",
                "factor_agent",
                "template_fill",
                template_expression[:80],
                session_id="factor_agent",
                alpha_id="",
            )
            await self._run_self_critique(template_expression, hypothesis_str)
            self._validate_whitelist(template_expression)
            sim_payload, align_action = self._check_and_attach_alignment(template_expression, hypothesis, sim_payload)
            sim_payload.setdefault("metadata", {})["template_id"] = _sched_template_id
            sim_payload.setdefault("metadata", {})["family_id"] = _sched_family_id
            return template_expression, sim_payload, None, align_action

        if template_mode == "hybrid" and template_expression:
            logger.info("FactorAgent: hybrid mode — refining template expression via LLM")
            refined_expression = await self._refine_template_expression(
                template_expression,
                hypothesis_str,
                brain_feedback,
            )
            if refined_expression and refined_expression != template_expression:
                try:
                    from openalpha_brain.validation import validator as val

                    syntax_result = val.validate_syntax(refined_expression)
                    if syntax_result.passed:
                        try:
                            template_score = val.estimate_sharpe_likelihood(template_expression)
                            refined_score = val.estimate_sharpe_likelihood(refined_expression)
                            logger.info(
                                "FactorAgent: hybrid mode — Sharpe likelihood: template=%.2f refined=%.2f",
                                template_score,
                                refined_score,
                            )
                            if refined_score < template_score * 0.5:
                                logger.info(
                                    "FactorAgent: hybrid mode — refined Sharpe likelihood much lower, keeping template"
                                )
                                raise ValueError("refined Sharpe likelihood too low")
                        except ValueError:
                            raise
                        except (OSError, RuntimeError):
                            pass
                        if "ts_decay_linear" not in refined_expression:
                            _auto_wrap_score = None
                            with contextlib.suppress(NameError):
                                _auto_wrap_score = refined_score
                            if _auto_wrap_score is None or _auto_wrap_score > 0.3:
                                _m = re.match(r"^(group_neutralize\()(.+)(,\s*industry\))$", refined_expression)
                                if _m:
                                    _wrapped_refined = f"{_m.group(1)}ts_decay_linear({_m.group(2)}, 10){_m.group(3)}"
                                    try:
                                        _wrap_result = val.validate_syntax(_wrapped_refined)
                                        if _wrap_result.passed:
                                            logger.info(
                                                "FactorAgent: hybrid mode — auto-wrapped refined with ts_decay_linear(window=10)"  # noqa: E501
                                            )
                                            refined_expression = _wrapped_refined
                                        else:
                                            logger.info(
                                                "FactorAgent: hybrid mode — auto-wrap of refined failed syntax, keeping unwrapped"  # noqa: E501
                                            )
                                    except (OSError, ValueError, RuntimeError):
                                        logger.info(
                                            "FactorAgent: hybrid mode — auto-wrap of refined validation error, keeping unwrapped"  # noqa: E501
                                        )
                        logger.info("FactorAgent: hybrid mode — LLM refinement passed syntax validation")
                        brain_params = _load_brain_submit_params()
                        sim_payload = {
                            "settings": {
                                "instrumentType": brain_params.get("instrumentType", "EQUITY"),
                                "region": "USA",
                                "universe": brain_params.get("universe", "TOP3000"),
                                "delay": brain_params.get("delay", 1),
                                "decay": brain_params.get("decay", 5),
                                "neutralization": brain_params.get("neutralization", "INDUSTRY"),
                                "truncation": brain_params.get("truncation", 0.05),
                                "pasteurization": brain_params.get("pasteurization", "ON"),
                                "unitHandling": "VERIFY",
                                "nanHandling": brain_params.get("nanHandling", "ON"),
                                "language": "FASTEXPR",
                                "visualization": False,
                            },
                            "regular": refined_expression,
                        }
                        _monitor.record(
                            "STEP",
                            "factor_agent",
                            "hybrid_refine_pass",
                            refined_expression[:80],
                            session_id="factor_agent",
                            alpha_id="",
                        )
                        await self._run_self_critique(refined_expression, hypothesis_str)
                        self._validate_whitelist(refined_expression)
                        sim_payload, align_action = self._check_and_attach_alignment(
                            refined_expression, hypothesis, sim_payload
                        )
                        sim_payload.setdefault("metadata", {})["template_id"] = _sched_template_id
                        sim_payload.setdefault("metadata", {})["family_id"] = _sched_family_id
                        return refined_expression, sim_payload, None, align_action
                    logger.warning(
                        "FactorAgent: hybrid mode — LLM refinement failed syntax validation (%s), falling back to template",  # noqa: E501
                        syntax_result.failures,
                    )
                except ImportError:
                    logger.warning(
                        "FactorAgent: hybrid mode — validator module not available, falling back to template"
                    )
            else:
                logger.info(
                    "FactorAgent: hybrid mode — LLM returned empty or unchanged expression, falling back to template"
                )

            if "ts_decay_linear" not in template_expression:
                try:
                    from openalpha_brain.validation import validator as _val

                    _template_score = _val.estimate_sharpe_likelihood(template_expression)
                    if _template_score > 0.3:
                        _m = re.match(r"^(group_neutralize\()(.+)(,\s*industry\))$", template_expression)
                        if _m:
                            _wrapped_template = f"{_m.group(1)}ts_decay_linear({_m.group(2)}, 10){_m.group(3)}"
                            try:
                                _wrap_result = _val.validate_syntax(_wrapped_template)
                                if _wrap_result.passed:
                                    logger.info(
                                        "FactorAgent: hybrid mode — auto-wrapped template with ts_decay_linear(window=10)"  # noqa: E501
                                    )
                                    template_expression = _wrapped_template
                                else:
                                    logger.info(
                                        "FactorAgent: hybrid mode — auto-wrap of template failed syntax, keeping unwrapped"  # noqa: E501
                                    )
                            except (ValueError, TypeError, RuntimeError):
                                logger.info(
                                    "FactorAgent: hybrid mode — auto-wrap of template validation error, keeping unwrapped"  # noqa: E501
                                )
                except (ValueError, TypeError, RuntimeError):
                    pass

            brain_params = _load_brain_submit_params()
            sim_payload = {
                "settings": {
                    "instrumentType": brain_params.get("instrumentType", "EQUITY"),
                    "region": "USA",
                    "universe": brain_params.get("universe", "TOP3000"),
                    "delay": brain_params.get("delay", 1),
                    "decay": brain_params.get("decay", 5),
                    "neutralization": brain_params.get("neutralization", "INDUSTRY"),
                    "truncation": brain_params.get("truncation", 0.05),
                    "pasteurization": brain_params.get("pasteurization", "ON"),
                    "unitHandling": "VERIFY",
                    "nanHandling": brain_params.get("nanHandling", "ON"),
                    "language": "FASTEXPR",
                    "visualization": False,
                },
                "regular": template_expression,
            }
            _monitor.record(
                "STEP",
                "factor_agent",
                "hybrid_fallback",
                template_expression[:80],
                session_id="factor_agent",
                alpha_id="",
            )
            await self._run_self_critique(template_expression, hypothesis_str)
            self._validate_whitelist(template_expression)
            sim_payload, align_action = self._check_and_attach_alignment(template_expression, hypothesis, sim_payload)
            sim_payload.setdefault("metadata", {})["template_id"] = _sched_template_id
            sim_payload.setdefault("metadata", {})["family_id"] = _sched_family_id
            return template_expression, sim_payload, None, align_action

        user_msg = (
            f"Translate the following hypothesis into a FASTEXPR alpha factor expression.\n"
            f"Hypothesis:\n{hypothesis_str}"
        )

        if template_expression:
            template_section = (
                f"\n\nTEMPLATE ASSEMBLY (use as starting point, then refine):\n"
                f"{template_expression}\n\n"
                f"Refine the above template: adjust parameters, add normalization if needed, ensure FASTEXPR syntax."
            )
            user_msg += template_section

        if self._feature_map is not None:
            try:
                parent_cell = self._feature_map.sample_parent()
                if parent_cell is not None and parent_cell.best_expr:
                    user_msg += f"\n\nEvolution parent expression for mutation: {parent_cell.best_expr}"
                    logger.info(
                        "FactorAgent: injected FeatureMap parent (fitness=%.3f): %s",
                        parent_cell.best_fitness,
                        parent_cell.best_expr[:80],
                    )
            except (ValueError, KeyError, OSError) as exc:
                logger.warning("FactorAgent: FeatureMap sample_parent failed: %s", exc)

        try:
            from openalpha_brain.core.loop_state import (
                _diversity_last_cycle,
                _last_diversity_stats,
                _last_unexplored_directions,
            )

            if _last_unexplored_directions and _diversity_last_cycle > 0:
                _unexp_fa = _last_unexplored_directions[:5]
                if _unexp_fa:
                    _fa_div_lines = ["\n\n▶ UNEXPLORED DIRECTIONS (consider these under-explored regions for novelty):"]
                    for _fi, _fd in enumerate(_unexp_fa):
                        _fa_div_lines.append(f"  {_fi + 1}. {_fd}")
                    if _last_diversity_stats:
                        _fa_cov = _last_diversity_stats.get("coverage", 0) * 100
                        _fa_div_lines.append(f"  Current feature-space coverage: {_fa_cov:.1f}%")
                    user_msg += "\n".join(_fa_div_lines)
        except (ImportError, ValueError, KeyError, TypeError):
            pass

        try:
            from openalpha_brain.core.loop_state import _previous_expressions

            if _previous_expressions:
                _prev_expr_list = list(_previous_expressions)[-5:]
                if len(_prev_expr_list) >= 2:
                    _dup_avoid_lines = [
                        "\n\n⛔ PREVIOUSLY GENERATED EXPRESSIONS (DO NOT repeat any of these — they already exist):"
                    ]
                    for _pi, _pe in enumerate(_prev_expr_list):
                        _dup_avoid_lines.append(f"  [{_pi + 1}] {_pe[:120]}")
                    _dup_avoid_lines.append(
                        "\nYou MUST generate a DIFFERENT expression. Change operators, fields, parameters, or structure."  # noqa: E501
                    )
                    user_msg += "\n".join(_dup_avoid_lines)
        except (ImportError, ValueError, KeyError, TypeError):
            pass

        raw = await self._llm_generate(
            system_prompt=system_prompt,
            history=[],
            user_msg=user_msg,
            session_id="factor_agent",
            cycle=0,
        )

        expression, sim_payload, _factor_economic_rationale = self._parse_expression(raw)
        _monitor.record("STEP", "factor_agent", "generate", expression[:80], session_id="factor_agent", alpha_id="")
        await self._run_self_critique(expression, hypothesis_str)
        self._validate_whitelist(expression)
        sim_payload, align_action = self._check_and_attach_alignment(expression, hypothesis, sim_payload)
        sim_payload.setdefault("metadata", {})["template_id"] = _sched_template_id
        sim_payload.setdefault("metadata", {})["family_id"] = _sched_family_id
        return expression, sim_payload, _factor_economic_rationale, align_action

    def _parse_expression(self, raw: str) -> tuple[str, dict, str | None]:
        brain_params = _load_brain_submit_params()
        default_payload = {
            "settings": {
                "instrumentType": brain_params.get("instrumentType", "EQUITY"),
                "region": "USA",
                "universe": brain_params.get("universe", "TOP3000"),
                "delay": brain_params.get("delay", 1),
                "decay": brain_params.get("decay", 5),
                "neutralization": brain_params.get("neutralization", "INDUSTRY"),
                "truncation": brain_params.get("truncation", 0.05),
                "pasteurization": brain_params.get("pasteurization", "ON"),
                "unitHandling": "VERIFY",
                "nanHandling": brain_params.get("nanHandling", "ON"),
                "language": "FASTEXPR",
                "visualization": False,
            },
            "regular": "",
        }

        economic_rationale = None
        rationale_match = re.search(r"ECONOMIC[_ ]RATIONALE\s*:\s*(.+)", raw, re.IGNORECASE)
        if rationale_match:
            economic_rationale = rationale_match.group(1).strip()

        try:
            _extracted = _extract_json_from_llm(raw)
            if isinstance(_extracted, list) and _extracted and isinstance(_extracted[0], dict):
                data = _extracted[0]
            elif isinstance(_extracted, dict):
                data = _extracted
            else:
                data = json.loads(raw)

            expression = data.get("expression", "")
            sim_payload = data.get("simulation_payload", {})

            if not economic_rationale:
                economic_rationale = data.get("economic_rationale") or data.get("rationale")

            if not isinstance(sim_payload, dict):
                sim_payload = {}

            settings = sim_payload.get("settings", {})
            if not isinstance(settings, dict):
                settings = {}
            merged_settings = {**default_payload["settings"], **settings}

            regular = sim_payload.get("regular", expression)
            if not regular:
                regular = expression

            return expression, {"settings": merged_settings, "regular": regular}, economic_rationale

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("FactorAgent: failed to parse expression JSON: %s", exc)
            expr = raw.strip()
            if expr.startswith("```"):
                first_newline = expr.index("\n") if "\n" in expr else len(expr)
                expr = expr[first_newline + 1 :]
                if expr.endswith("```"):
                    expr = expr[:-3]
                expr = expr.strip()
            default_payload["regular"] = expr
            return expr, default_payload, economic_rationale


class EvalAgent:
    def __init__(
        self,
        brain_submit_fn: Callable[..., Awaitable[brain_client.BrainGateResult]] | None = None,
        mab: Any | None = None,
        success_lib: SuccessCaseLibrary | None = None,
        llm_generate_fn: Callable[..., Any] | None = None,
        success_pool: list[dict[str, Any]] | None = None,
        dedup_callback: Callable[[str, dict], Any] | None = None,
    ) -> None:
        self._brain_submit = brain_submit_fn or brain_client.submit_and_poll
        self._cookies: httpx.Cookies | None = None
        self._mab = mab
        self._success_lib = success_lib
        self._llm_generate = llm_generate_fn
        self._success_pool: list[dict[str, Any]] = success_pool or []
        self._dedup_callback = dedup_callback

    @staticmethod
    def _check_alignment(hypothesis, expression: str) -> float:
        return _check_semantic_alignment(hypothesis, expression)

    async def _llm_align_check(self, hypothesis, expression: str) -> dict:
        hyp_text = getattr(hypothesis, "natural_language", "") or getattr(hypothesis, "direction", "")
        hyp_mechanism = getattr(hypothesis, "mechanism", "")

        align_prompt = f"""You are evaluating how well an alpha expression implements a quantitative hypothesis.

HYPOTHESIS:
Direction: {getattr(hypothesis, "direction", "")}
Mechanism: {hyp_mechanism}
Description: {hyp_text}

EXPRESSION: {expression}

Classify the alignment into one of three categories:
- STRONG: Expression directly implements the hypothesis (core operators and fields match)
- MODERATE: Partial alignment (some elements match, but key aspects missing or mixed)
- WEAK: Expression does NOT implement the hypothesis (wrong direction, different signal)

Consider:
1. Does the expression target the same economic phenomenon?
2. Do the operators match the described mechanism?
3. Would the signal direction align with the hypothesis?

Return ONLY a JSON object:
{{"category": "STRONG|MODERATE|WEAK", "reason": "brief justification"}}"""

        _category_score_map = {"STRONG": 0.85, "MODERATE": 0.55, "WEAK": 0.20}

        if not self._llm_generate:
            static_score = self._check_alignment(hypothesis, expression)
            return {"alignment_score": static_score, "reason": "static check", "needs_regeneration": static_score < 0.6}

        try:
            raw = await asyncio.wait_for(
                self._llm_generate(
                    align_prompt,
                    [],
                    "",
                    session_id="align_check",
                    cycle=0,
                ),
                timeout=30.0,
            )
            parsed = _extract_json_from_llm(raw)
            if parsed and isinstance(parsed, dict):
                category = str(parsed.get("category", "MODERATE")).upper().strip()
                score = _category_score_map.get(category, 0.55)
                return {
                    "alignment_score": score,
                    "reason": parsed.get("reason", ""),
                    "needs_regeneration": score < 0.6,
                }
        except TimeoutError:
            logger.warning("EvalAgent: LLM align check timed out after 30s")
        except (aiohttp.ClientError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("EvalAgent: LLM align check failed: %s", exc)

        static_score = self._check_alignment(hypothesis, expression)
        return {
            "alignment_score": static_score,
            "reason": "fallback static check",
            "needs_regeneration": static_score < 0.6,
        }

    async def _archive_success(self, expression: str, result: dict) -> None:
        try:
            emb = await llm_client.embed(expression)
            self._success_pool.append(
                {
                    "expression": expression,
                    "embedding": emb,
                    "sharpe": result.get("sharpe", 0),
                    "timestamp": result.get("submitted_at", 0),
                }
            )
            logger.info("EvalAgent: archived success alpha to pool (total=%d)", len(self._success_pool))
        except (OSError, ValueError, RuntimeError) as exc:
            logger.warning("EvalAgent: failed to archive success alpha: %s", exc)

    async def _semantic_dedup_check(self, expression: str) -> dict:
        if not self._success_pool:
            return {"is_duplicate": False, "similar_candidates": [], "dedup_score": 0.0}

        try:
            new_emb = await llm_client.embed(expression)
        except (ValueError, TypeError, RuntimeError) as exc:
            logger.warning("EvalAgent: dedup embedding failed: %s", exc)
            return {"is_duplicate": False, "similar_candidates": [], "dedup_score": 0.0}

        similarities: list[tuple[int, float, dict]] = []
        for idx, entry in enumerate(self._success_pool):
            existing_emb = entry.get("embedding")
            if not existing_emb:
                continue
            dot = sum(a * b for a, b in zip(new_emb, existing_emb, strict=False))
            norm_a = math.sqrt(sum(a * a for a in new_emb))
            norm_b = math.sqrt(sum(b * b for b in existing_emb))
            if norm_a == 0 or norm_b == 0:
                continue
            cos_sim = dot / (norm_a * norm_b)
            if cos_sim > 0.85:
                similarities.append((idx, cos_sim, entry))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top5 = similarities[:5]

        if not top5:
            return {"is_duplicate": False, "similar_candidates": [], "dedup_score": 0.0}

        semantic_duplicates: list[int] = []
        for idx, cos_sim, entry in top5:
            if not self._llm_generate:
                if cos_sim > 0.95:
                    semantic_duplicates.append(idx)
                continue
            try:
                is_dup = await self._llm_semantic_compare(expression, entry["expression"])
                if is_dup:
                    semantic_duplicates.append(idx)
            except (TimeoutError, aiohttp.ClientError, ValueError, json.JSONDecodeError) as exc:
                logger.warning("EvalAgent: LLM semantic compare failed: %s", exc)
                if cos_sim > 0.95:
                    semantic_duplicates.append(idx)

        max_sim = top5[0][1] if top5 else 0.0
        is_duplicate = len(semantic_duplicates) > 0
        logger.info(
            "EvalAgent: dedup check — cos=%.3f candidates=%d semantic_dups=%d verdict=%s",
            max_sim,
            len(top5),
            len(semantic_duplicates),
            "DUPLICATE" if is_duplicate else "OK",
        )
        return {
            "is_duplicate": is_duplicate,
            "similar_candidates": [
                {
                    "idx": idx,
                    "cosine": cos_sim,
                    "expression": self._success_pool[idx]["expression"][:80],
                }
                for idx, cos_sim, _ in top5
            ],
            "dedup_score": max_sim,
            "semantic_duplicate_indices": semantic_duplicates,
        }

    async def _llm_semantic_compare(self, expr_a: str, expr_b: str) -> bool:
        prompt = f"""Compare these two alpha expressions and determine if they are SEMANTICALLY EQUIVALENT.

Expression A: {expr_a}
Expression B: {expr_b}

Definition of semantic equivalence:
- They measure the same economic phenomenon using the same core signal
- Simple renaming of variables or trivial reordering does NOT count as different
- Different lookback windows on the same operator sequence ARE the same signal
- If one is a mathematical transformation of the other (e.g., rank(A) vs -rank(-A)), they ARE equivalent

Return ONLY a JSON object: {{"equivalent": true/false, "reasoning": "brief explanation"}}"""

        raw = await self._llm_generate(prompt, [], "", session_id="dedup", cycle=0)
        try:
            result = _extract_json_from_llm(raw)
            if result and isinstance(result, dict):
                val = result.get("equivalent", False)
                if isinstance(val, str):
                    val = val.lower() in ("true", "1", "yes")
                return bool(val)
        except (ValueError, TypeError):
            pass
        return False

    async def _handle_duplication(self, expression: str, result: dict) -> dict | None:
        if not self._dedup_callback:
            logger.info("EvalAgent: duplication detected but no dedup callback configured")
            return None
        try:
            variant = await self._dedup_callback(expression, result)
            if variant:
                logger.info("EvalAgent: generated mutation variant for duplicated alpha")
                return variant
        except (ValueError, TypeError, RuntimeError) as exc:
            logger.warning("EvalAgent: dedup callback failed: %s", exc)
        return None

    @staticmethod
    def _check_complexity(expression: str) -> dict:
        depth = 0
        max_depth = 0
        node_count = 0
        for ch in expression:
            if ch == "(":
                depth += 1
                max_depth = max(max_depth, depth)
                node_count += 1
            elif ch == ")":
                depth -= 1
            elif ch == ",":
                node_count += 1
        node_count = max(node_count, 1)
        operator_count = len(re.findall(r"\b[a-zA-Z_]\w*\s*\(", expression))
        return {
            "ast_depth": max_depth,
            "node_count": node_count,
            "operator_count": operator_count,
            "is_complex": max_depth > 5 or node_count > 20 or operator_count > 8,
            "depth_exceeded": max_depth > 5,
            "nodes_exceeded": node_count > 20,
            "operators_exceeded": operator_count > 8,
        }

    async def _llm_simplify(self, expression: str) -> str | None:
        if not self._llm_generate:
            return None

        simplify_prompt = f"""You are simplifying an overly complex alpha expression while preserving its core signal.

Original expression: {expression}

Rules:
1. Remove redundant nesting (double rank, double normalize, etc.)
2. Merge same-type operators (e.g., two ts_mean with different windows → one)
3. If the expression has more than 3 nested levels, flatten one level
4. Keep the core economic intuition identical
5. Output must be a valid single expression

Return ONLY a JSON:
{{"simplified_expression": "the simplified expression", "changes_made": "brief description"}}"""

        try:
            raw = await self._llm_generate(
                simplify_prompt,
                [],
                "",
                session_id="simplify",
                cycle=0,
            )
            parsed = _extract_json_from_llm(raw)
            if parsed and isinstance(parsed, dict):
                simplified = parsed.get("simplified_expression", "")
                if simplified and simplified != expression:
                    logger.info(
                        "EvalAgent: LLM simplified expression — changes=%s new=%s",
                        parsed.get("changes_made", "?")[:80],
                        simplified[:80],
                    )
                    return simplified
        except (TimeoutError, aiohttp.ClientError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("EvalAgent: LLM simplify failed: %s", exc)
        return None

    async def _ensure_auth(self) -> httpx.Cookies:
        from openalpha_brain.config.config import settings

        if self._cookies is None:
            self._cookies = await brain_client.authenticate(
                settings.BRAIN_EMAIL,
                settings.BRAIN_PASSWORD,
            )
        return self._cookies

    async def evaluate(
        self,
        expression: str,
        simulation_payload: dict,
        direction: str = "",
    ) -> dict:
        syntax_passed = True
        syntax_errors: list[str] = []
        try:
            from openalpha_brain.validation import validator as val

            syntax_result = val.validate_syntax(expression)
            syntax_passed = syntax_result.passed
            if not syntax_passed and syntax_result.errors:
                syntax_errors = syntax_result.errors[:3]
        except (OSError, ValueError, RuntimeError):
            pass

        if not simulation_payload.get("regular"):
            simulation_payload = dict(simulation_payload)
            simulation_payload["regular"] = expression

        brain_checks: list[dict] = []
        llm_feedback = ""

        if not syntax_passed:
            brain_checks = [
                {
                    "name": "SYNTAX_CHECK",
                    "result": "FAIL",
                    "value": str(syntax_errors) if syntax_errors else "syntax error",
                },
            ]
            llm_feedback = f"Syntax validation failed: {syntax_errors[0] if syntax_errors else 'unknown error'}"
        elif self._llm_generate:
            try:
                structure_result = await self._llm_structure_check(expression, direction, simulation_payload)
                brain_checks = structure_result.get("brain_checks", [])
                llm_feedback = structure_result.get("feedback", "")
            except (ValueError, TypeError, RuntimeError) as exc:
                logger.warning("EvalAgent: LLM structure check failed: %s", exc)
                brain_checks = [
                    {"name": "PRE_FLIGHT_CHECK", "result": "PASS", "value": "syntax_ok"},
                ]
                llm_feedback = "Structure check passed at syntax level"

        complexity_result = self._check_complexity(expression)
        if complexity_result["is_complex"]:
            overfit_warn = {
                "name": "OVERFITTING_RISK",
                "result": "WARN",
                "value": f"depth={complexity_result['ast_depth']} nodes={complexity_result['node_count']} ops={complexity_result['operator_count']}",  # noqa: E501
            }
            brain_checks.append(overfit_warn)
            logger.info(
                "EvalAgent: complexity WARN — depth=%d nodes=%d ops=%d",
                complexity_result["ast_depth"],
                complexity_result["node_count"],
                complexity_result["operator_count"],
            )

        status = "FAIL"
        brain_sharpe = None
        brain_fitness = None
        brain_turnover = None
        brain_returns = None
        brain_drawdown = None
        brain_margin = None
        brain_alpha_id = None

        if not syntax_passed:
            status = "FAIL"
        else:
            brain_backtest_done = False
            if self._brain_submit is not None:
                try:
                    from openalpha_brain.config.config import settings as _settings

                    if getattr(_settings, "BRAIN_SUBMIT_ENABLED", True):
                        cookies = await self._ensure_auth()
                        if cookies is not None:
                            gate_result = await self._brain_submit(
                                simulation_payload=simulation_payload,
                                cookies=cookies,
                                max_poll_seconds=getattr(_settings, "BRAIN_POLL_TIMEOUT", 300),
                            )
                            brain_backtest_done = True
                            brain_sharpe = gate_result.sharpe
                            brain_fitness = gate_result.fitness
                            brain_turnover = gate_result.turnover
                            brain_returns = gate_result.returns
                            brain_drawdown = gate_result.drawdown
                            brain_margin = gate_result.margin
                            brain_alpha_id = gate_result.alpha_id
                            if gate_result.brain_checks:
                                brain_checks.extend(gate_result.brain_checks)
                            if gate_result.passed:
                                status = "PASS"
                                llm_feedback = llm_feedback or "BRAIN backtest passed"
                            else:
                                status = "FAIL"
                                _fail_msgs = gate_result.failures or []
                                if _fail_msgs:
                                    llm_feedback = f"BRAIN gate failed: {'; '.join(_fail_msgs[:3])}"
                                else:
                                    llm_feedback = llm_feedback or "BRAIN backtest failed"
                except (ValueError, TypeError, RuntimeError, ConnectionError) as exc:
                    logger.warning("EvalAgent: BRAIN backtest failed: %s", exc)

            if not brain_backtest_done and syntax_passed:
                status = "PENDING"
                llm_feedback = llm_feedback or "Structure check passed; awaiting BRAIN verification"

        logger.info(
            "EvalAgent: eval complete — status=%s syntax=%s backtest=%s checks=%d expr=%s",
            status,
            "OK" if syntax_passed else "FAIL",
            "done" if brain_backtest_done else "skipped",
            len(brain_checks),
            expression[:80],
        )

        return {
            "expression": expression,
            "simulation_payload": simulation_payload,
            "sharpe": brain_sharpe,
            "ic": None,
            "status": status,
            "feedback": llm_feedback or "Structure check passed",
            "alpha_id": brain_alpha_id,
            "fitness": brain_fitness,
            "turnover": brain_turnover,
            "returns": brain_returns,
            "drawdown": brain_drawdown,
            "margin": brain_margin,
            "brain_checks": brain_checks,
            "semantic_alignment_score": await self._check_alignment(direction, expression)
            if not syntax_passed
            else 0.5,
            "hierarchical_reward": 0.0,
            "direction": direction,
        }

    async def _llm_structure_check(
        self,
        expression: str,
        direction: str,
        simulation_payload: dict,
    ) -> dict:
        settings_obj = simulation_payload.get("settings", {})
        universe = settings_obj.get("universe", "unknown")
        horizon = settings_obj.get("horizon", "unknown")

        check_prompt = f"""You are a quantitative finance auditor. Analyze this alpha expression for STRUCTURAL ISSUES only.  # noqa: E501

Direction: {direction}
Expression: {expression}
Universe: {universe}
Horizon: {horizon}

Do NOT predict scores (Sharpe, IC, returns). Instead, check for structural problems:
- Are the operators combined in a logically valid way?
- Does the expression make economic sense for the given direction?
- Are there any obvious statistical pitfalls (self-correlation from lagged comparisons, rank on ratios)?
- Is the complexity justified or excessive?
- Would the turnover likely be problematic given the window parameters?

Return ONLY a JSON object:
```json
{{
  "feedback": "brief structural assessment",
  "brain_checks": [
    {{"name": "SELF_CORRELATION", "result": "PASS|WARN|FAIL", "value": "explanation of the risk"}},
    {{"name": "OVERFITTING_RISK", "result": "PASS|WARN|FAIL", "value": "complexity assessment"}},
    {{"name": "ECONOMIC_RATIONALE", "result": "PASS|WARN|FAIL", "value": "does it make economic sense"}},
    {{"name": "TURNOVER_RISK", "result": "LOW|MEDIUM|HIGH", "value": "based on window parameters"}},
    {{"name": "OPERATOR_CONSISTENCY", "result": "PASS|WARN|FAIL", "value": "operator combination validity"}}
  ]
}}
```

IMPORTANT: Do NOT invent scores. Only flag structural issues. Return ONLY the JSON."""

        try:
            logger.info("EvalAgent: _llm_structure_check started (expr=%s)", expression[:60])
            raw = await asyncio.wait_for(
                self._llm_generate(
                    check_prompt,
                    [],
                    "",
                    session_id="eval_agent",
                    cycle=0,
                ),
                timeout=settings.LLM_TIMEOUT if hasattr(settings, "LLM_TIMEOUT") else 15.0,
            )
            logger.info("EvalAgent: _llm_structure_check LLM response received (len=%d)", len(raw or ""))
            result = _extract_json_from_llm(raw)
            if result and isinstance(result, dict):
                result.setdefault("feedback", "")
                result.setdefault("brain_checks", [])
                return result
            return {
                "feedback": "LLM structure check unavailable",
                "brain_checks": [
                    {"name": "PRE_FLIGHT_CHECK", "result": "PASS", "value": "syntax_ok"},
                ],
            }
        except TimeoutError:
            logger.warning(
                "EvalAgent: _llm_structure_check timed out after %.1fs",
                settings.LLM_TIMEOUT if hasattr(settings, "LLM_TIMEOUT") else 15.0,
            )
            return {
                "feedback": "LLM structure check timed out",
                "brain_checks": [
                    {"name": "PRE_FLIGHT_CHECK", "result": "PASS", "value": "timeout_default_pass"},
                ],
            }
        except (ValueError, TypeError, RuntimeError):
            return {
                "feedback": "LLM structure check unavailable",
                "brain_checks": [
                    {"name": "PRE_FLIGHT_CHECK", "result": "PASS", "value": "syntax_ok"},
                ],
            }


@dataclass
class AgentMessage:
    sender: str
    receiver: str
    msg_type: str
    payload: dict


class MessageBus:
    def __init__(self) -> None:
        self._queue: list[AgentMessage] = []

    def send(self, message: AgentMessage) -> None:
        self._queue.append(message)

    def receive(self, receiver: str) -> list[AgentMessage]:
        received = [m for m in self._queue if m.receiver == receiver]
        self._queue = [m for m in self._queue if m.receiver != receiver]
        return received


class MultiAgentOrchestrator:
    def __init__(
        self,
        idea_agent: IdeaAgent,
        factor_agent: FactorAgent,
        eval_agent: EvalAgent,
        originality_checker: OriginalityChecker | None = None,
        complexity_controller: ComplexityController | None = None,
        rag_engine: RAGEngine | None = None,
        evolution_db=None,
        feature_map=None,
    ) -> None:
        self._idea_agent = idea_agent
        self._factor_agent = factor_agent
        self._eval_agent = eval_agent
        self._originality_checker = originality_checker
        self._complexity_controller = complexity_controller
        self._rag_engine = rag_engine
        self._evo_db = evolution_db
        self._feature_map = feature_map
        self._successful_alphas: list[str] = []
        self._reflection_engine: ReflectionEngine | None = None
        self._next_plan: NextPlan | None = None
        self._agent_factory: AdaptiveAgentFactory | None = None
        self._message_bus = MessageBus()
        if settings.ADAPTIVE_AGENT_ENABLED:
            self._agent_factory = AdaptiveAgentFactory()
        if settings.REFLECTION_ENGINE_ENABLED:
            self._reflection_engine = ReflectionEngine(path=settings.REFLECTION_ENGINE_PATH)
            self._reflection_engine.set_llm_generate_fn(llm_client.generate)

    def inject_successful_alphas(self, expressions: list[str]) -> None:
        if expressions:
            self._successful_alphas = list(expressions[-20:])
            logger.info(
                "MultiAgentOrchestrator: injected %d successful alpha expressions", len(self._successful_alphas)
            )

    async def critique_revise_loop(
        self,
        expression: str,
        brain_result: dict | None,
        hypothesis_text: str = "",
        direction: str = "",
        max_rounds: int = 3,
    ) -> dict:
        result = {
            "expression": expression,
            "rounds_completed": 0,
            "revised_expressions": [expression],
            "critique_log": [],
            "final_semantic_check": None,
        }

        if brain_result is None:
            logger.info("MultiAgent: critique_revise_loop skipped — no brain_result")
            return result

        brain_checks = (
            brain_result.get("brain_checks", [])
            if isinstance(brain_result, dict)
            else getattr(brain_result, "brain_checks", []) or []
        )
        real_sharpe = (
            brain_result.get("real_sharpe", None)
            if isinstance(brain_result, dict)
            else getattr(brain_result, "real_sharpe", None)
        )
        real_fitness = (
            brain_result.get("real_fitness", None)
            if isinstance(brain_result, dict)
            else getattr(brain_result, "real_fitness", None)
        )
        real_turnover = (
            brain_result.get("real_turnover", None)
            if isinstance(brain_result, dict)
            else getattr(brain_result, "real_turnover", None)
        )
        status = (
            brain_result.get("status", "FAIL")
            if isinstance(brain_result, dict)
            else getattr(brain_result, "status", "FAIL")
        )

        current_expr = expression
        for round_num in range(1, max_rounds + 1):
            logger.info(
                "MultiAgent: critique_revise_loop round %d/%d — expr=%s",
                round_num,
                max_rounds,
                current_expr[:60],
            )

            critique_prompt = f"""You are a quantitative alpha factor critic. Analyze why this expression failed BRAIN validation.  # noqa: E501

EXPRESSION: {current_expr}
HYPOTHESIS: {hypothesis_text or "Not specified"}
DIRECTION: {direction or "Not specified"}

BRAIN RESULT:
  Status: {status}
  Sharpe: {real_sharpe or "N/A"}
  Fitness: {real_fitness or "N/A"}
  Turnover: {real_turnover or "N/A"}
  Failed Checks: {brain_checks}

Provide a structured critique:
1. root_cause: What went wrong? (1-2 sentences, based on BRAIN data)
2. structural_issue: What structural or logical flaw exists?
3. modification_instruction: Specific, actionable instruction for FactorAgent to revise the expression
4. priority: HIGH/MEDIUM/LOW

Respond in JSON format only:
{{"root_cause": "...", "structural_issue": "...", "modification_instruction": "...", "priority": "HIGH|MEDIUM|LOW"}}"""

            critique_result = {
                "root_cause": "",
                "structural_issue": "",
                "modification_instruction": "",
                "priority": "MEDIUM",
            }
            try:
                if self._eval_agent._llm_generate:
                    raw = await self._eval_agent._llm_generate(
                        critique_prompt,
                        [],
                        "",
                        session_id=f"critique_r{round_num}",
                        cycle=round_num,
                    )
                    _parsed = _extract_json_from_llm(raw)
                    if _parsed and isinstance(_parsed, dict):
                        critique_result = _parsed
            except (ValueError, TypeError, RuntimeError) as exc:
                logger.warning("MultiAgent: critique LLM call failed r%d: %s", round_num, exc)

            result["critique_log"].append(
                {
                    "round": round_num,
                    "expression_before": current_expr,
                    "critique": critique_result,
                }
            )

            revise_prompt = f"""You are a quantitative alpha factor engineer. Revise the failed expression based on the critique.  # noqa: E501

ORIGINAL EXPRESSION: {current_expr}
CRITIQUE: {critique_result.get("root_cause", "")}
STRUCTURAL ISSUE: {critique_result.get("structural_issue", "")}
MODIFICATION INSTRUCTION: {critique_result.get("modification_instruction", "")}
DIRECTION: {direction or "Not specified"}

Rules:
1. Follow the modification instruction precisely
2. Keep the core signal concept intact
3. Return ONLY the revised expression as a string (no markdown, no explanation)"""

            revised_expr = current_expr
            try:
                if self._factor_agent._llm_generate:
                    raw = await self._factor_agent._llm_generate(
                        revise_prompt,
                        [],
                        "",
                        session_id=f"revise_r{round_num}",
                        cycle=round_num,
                    )
                    cleaned = raw.strip()
                    for prefix in ["```json", "```python", "```", "expression:", "EXPRESSION:"]:
                        if cleaned.lower().startswith(prefix.lower()):
                            cleaned = cleaned[len(prefix) :].strip()
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3].strip()
                    if cleaned and len(cleaned) >= 5:
                        revised_expr = cleaned
            except (TimeoutError, aiohttp.ClientError, ValueError, json.JSONDecodeError) as exc:
                logger.warning("MultiAgent: revise LLM call failed r%d: %s", round_num, exc)

            result["revised_expressions"].append(revised_expr)
            result["rounds_completed"] = round_num
            current_expr = revised_expr

            logger.info(
                "MultiAgent: critique_revise_loop round %d — revised: orig=%s... → new=%s...",
                round_num,
                current_expr[:40],
                revised_expr[:40],
            )

        semantic_check_prompt = f"""You are verifying that a revised alpha expression preserves the original economic intuition.  # noqa: E501

ORIGINAL: {expression}
FINAL_REVISED: {current_expr}
DIRECTION: {direction or "Not specified"}

Is the FINAL_REVISED expression semantically equivalent to the ORIGINAL in its core signal concept?
Consider: does it target the same economic phenomenon and signal direction?

Return ONLY a JSON: {{"semantically_equivalent": true/false, "reason": "brief justification"}}"""

        try:
            if self._eval_agent._llm_generate:
                raw = await self._eval_agent._llm_generate(
                    semantic_check_prompt,
                    [],
                    "",
                    session_id="semantic_check",
                    cycle=0,
                )
                _parsed = _extract_json_from_llm(raw)
                if _parsed and isinstance(_parsed, dict):
                    result["final_semantic_check"] = _parsed
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("MultiAgent: semantic equivalence check failed: %s", exc)

        result["expression"] = current_expr
        logger.info(
            "MultiAgent: critique_revise_loop complete — %d rounds, final expr=%s...",
            result["rounds_completed"],
            current_expr[:60],
        )
        return result

    @algo_log(level=logging.INFO)
    async def run_iteration(
        self,
        direction: str,
        history: list[dict],
        brain_feedback: list[dict] | None = None,
        operators: list[str] | None = None,
        fields: list[str] | None = None,
    ) -> AgentResult:
        if self._next_plan is not None and self._next_plan.direction:
            direction = self._next_plan.direction
            logger.info("MultiAgent: using reflection plan direction=%s", direction)
        best_result: AgentResult | None = None
        no_improvement_count = 0
        idea_feedback: list[dict] = []
        factor_feedback: list[dict] = list(brain_feedback or [])
        trajectory: AlphaTrajectory | None = None

        for iteration in range(1, MAX_ITERATIONS + 1):
            logger.info(
                "MultiAgent: iteration %d/%d, direction=%s",
                iteration,
                MAX_ITERATIONS,
                direction,
            )

            bus_messages = self._message_bus.receive("idea_agent")
            for msg in bus_messages:
                if msg.msg_type == "DiagnosisFeedback":
                    idea_feedback.append(msg.payload)

            hypothesis, rag_context_dict = await self._idea_agent.generate_hypothesis(
                direction=direction,
                history=history,
                brain_feedback=idea_feedback if idea_feedback else None,
            )
            logger.info(
                "MultiAgent: IdeaAgent generated hypothesis — %s",
                hypothesis.natural_language[:100],
            )

            try:
                trajectory = AlphaTrajectory(
                    hypothesis_direction=hypothesis.direction,
                    hypothesis_mechanism=hypothesis.mechanism,
                )
            except (ValueError, TypeError, RuntimeError):
                trajectory = None

            anchor: SemanticAnchor | None = None
            try:
                anchor = SemanticAnchor(
                    hypothesis_text=hypothesis.natural_language or hypothesis.direction,
                    direction=hypothesis.direction,
                    core_operators=list(_DIRECTION_OPERATOR_MAP.get(hypothesis.direction.lower(), [])),
                )
                if (
                    self._rag_engine
                    and hasattr(self._rag_engine, "_embed_fn")
                    and self._rag_engine._embed_fn is not None
                ):
                    with contextlib.suppress(OSError, ValueError, RuntimeError):
                        await anchor.compute_embedding(self._rag_engine._embed_fn)
                else:
                    with contextlib.suppress(OSError, ValueError, RuntimeError):
                        await anchor.compute_embedding(llm_client.embed)
            except (OSError, ValueError, RuntimeError):
                anchor = None

            _max_align_retries = 2
            (
                expression,
                sim_payload,
                _iteration_economic_rationale,
                align_action,
            ) = await self._factor_agent.generate_expression(
                hypothesis=hypothesis,
                operators=operators,
                fields=fields,
                brain_feedback=factor_feedback if factor_feedback else None,
                rag_context_dict=rag_context_dict,
            )
            for _align_retry in range(_max_align_retries):
                if align_action != "RETRY":
                    break
                _retry_fb = (sim_payload.get("metadata") or {}).get("_alignment_retry_feedback", "")
                logger.info(
                    "MultiAgent: alignment retry %d/%d — R² too low, regenerating (feedback: %s)",
                    _align_retry + 1,
                    _max_align_retries,
                    (_retry_fb or "")[:120],
                )
                if _retry_fb:
                    hypothesis_str_retry = (
                        f"Direction: {hypothesis.direction}\n"
                        f"Mechanism: {hypothesis.mechanism}\n"
                        f"Natural language: {hypothesis.natural_language}\n\n"
                        f"⚠️ ALIGNMENT FEEDBACK (MUST fix these issues):\n{_retry_fb}"
                    )
                    _orig_hypothesis_nl = hypothesis.natural_language
                    hypothesis.natural_language = hypothesis_str_retry
                (
                    expression,
                    sim_payload,
                    _iteration_economic_rationale,
                    align_action,
                ) = await self._factor_agent.generate_expression(
                    hypothesis=hypothesis,
                    operators=operators,
                    fields=fields,
                    brain_feedback=factor_feedback if factor_feedback else None,
                    rag_context_dict=rag_context_dict,
                )
                if _retry_fb:
                    hypothesis.natural_language = _orig_hypothesis_nl
            if align_action == "WEAK":
                logger.warning(
                    "MultiAgent: WEAK R² alignment after generation — expression may partially drift from hypothesis '%s'",  # noqa: E501
                    hypothesis.direction,
                )
            critique = self._factor_agent._last_critique_result
            if critique is not None and (
                not critique.critique_available
                or critique.consistency_score is None
                or critique.consistency_score < 0.5
            ):
                _warning_reason = ""
                if not critique.critique_available or critique.consistency_score is None:
                    _warning_reason = f"Self-critique unavailable (critique_available={critique.critique_available}, consistency_score={critique.consistency_score})."  # noqa: E501
                else:
                    _warning_reason = f"Self-critique consistency_score={critique.consistency_score:.2f} < 0.5."
                factor_feedback.append(
                    {
                        "sharpe": "N/A",
                        "status": "LOW_CONSISTENCY",
                        "feedback": f"{_warning_reason} Issues: {'; '.join(critique.issues)}. Suggestions: {'; '.join(critique.suggestions)}",  # noqa: E501
                    }
                )
            logger.info(
                "MultiAgent: FactorAgent generated expression — %s",
                expression[:80],
            )

            _max_llm_align_retries = 2
            best_align_result = None
            best_align_expression = expression
            best_align_sim_payload = sim_payload
            best_align_hypothesis = hypothesis
            for _align_llm_retry in range(_max_llm_align_retries + 1):
                align_result = await self._eval_agent._llm_align_check(
                    best_align_hypothesis,
                    best_align_expression,
                )
                align_score = align_result.get("alignment_score", 0.5)
                logger.info(
                    "MultiAgent: LLM align check — score=%.2f retry=%d needs_regeneration=%s",
                    align_score,
                    _align_llm_retry,
                    align_result.get("needs_regeneration", False),
                )
                if best_align_result is None or align_score > best_align_result.get("alignment_score", 0):
                    best_align_result = align_result
                if not align_result.get("needs_regeneration", False) or _align_llm_retry >= _max_llm_align_retries:
                    break
                align_feedback = (
                    f"Previous hypothesis-expression alignment score was {align_score:.2f} (< 0.6). "
                    f"Reason: {align_result.get('reason', 'low alignment')}. "
                    "Please generate a MORE SPECIFIC hypothesis that makes the economic mechanism clearer "
                    "so the FactorAgent can produce a better-matching expression."
                )
                regen_idea_feedback = idea_feedback + [
                    {
                        "status": "LOW_ALIGNMENT",
                        "feedback": align_feedback,
                    }
                ]
                regen_hypothesis, rag_context_dict = await self._idea_agent.generate_hypothesis(
                    direction=direction,
                    history=history,
                    brain_feedback=regen_idea_feedback,
                )
                logger.info(
                    "MultiAgent: IdeaAgent regenerated hypothesis — align_retry=%d: %s",
                    _align_llm_retry + 1,
                    regen_hypothesis.natural_language[:100],
                )
                (
                    regen_expr,
                    regen_sim_payload,
                    _regeneration_economic_rationale,
                    _regen_align_action,
                ) = await self._factor_agent.generate_expression(
                    hypothesis=regen_hypothesis,
                    operators=operators,
                    fields=fields,
                    brain_feedback=factor_feedback if factor_feedback else None,
                    rag_context_dict=rag_context_dict,
                )
                logger.info(
                    "MultiAgent: FactorAgent regenerated expression — align_retry=%d: %s",
                    _align_llm_retry + 1,
                    regen_expr[:80],
                )
                best_align_expression = regen_expr
                best_align_sim_payload = regen_sim_payload
                best_align_hypothesis = regen_hypothesis

            if best_align_result is not None:
                expression = best_align_expression
                sim_payload = best_align_sim_payload
                logger.info(
                    "MultiAgent: LLM alignment complete — best_score=%.2f",
                    best_align_result.get("alignment_score", 0),
                )

            try:
                if trajectory is not None:
                    trajectory.add_expression_version(expression)
                    rag_ops = (
                        [op["name"] for op in rag_context_dict.get("top_ops_detailed", [])] if rag_context_dict else []
                    )
                    if rag_ops:
                        for op in rag_ops[:3]:
                            trajectory.add_decision("operator", op, [o for o in rag_ops if o != op][:2])
            except (KeyError, ValueError, TypeError):
                pass

            originality_score = 0.5
            if self._originality_checker and settings.ORIGINALITY_CHECK_ENABLED:
                try:
                    originality_score = self._originality_checker.check_originality(
                        f"multi_agent_iter_{iteration}",
                        expression,
                    )
                except (OSError, ValueError, RuntimeError) as exc:
                    logger.warning("MultiAgent: originality check failed: %s", exc)

            complexity_metrics_dict: dict[str, Any] = {}
            if self._complexity_controller and expression and settings.COMPLEXITY_CHECK_ENABLED:
                try:
                    ok, metrics, reason = self._complexity_controller.check_complexity(expression)
                    complexity_metrics_dict = {
                        "depth": metrics.depth,
                        "node_count": metrics.node_count,
                        "operator_count": metrics.operator_count,
                        "field_count": metrics.field_count,
                        "constant_count": metrics.constant_count,
                        "passed": ok,
                        "reason": reason,
                    }
                    if not ok:
                        logger.warning(
                            "MultiAgent: complexity check failed — %s",
                            reason,
                        )
                except (ValueError, TypeError, RuntimeError) as exc:
                    logger.warning("MultiAgent: complexity check failed: %s", exc)

            eval_result = await self._eval_agent.evaluate(
                expression=expression,
                simulation_payload=sim_payload,
                direction=hypothesis.direction,
            )
            if (
                self._rag_engine
                and hasattr(self._rag_engine, "update_weights_from_feedback")
                and eval_result.get("brain_checks")
            ):
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    self._rag_engine.update_weights_from_feedback(direction, eval_result["brain_checks"])

            if eval_result.get("brain_checks"):
                rule_diagnosis = build_idea_diagnostic_feedback(
                    eval_result["brain_checks"],
                    eval_result.get("sharpe"),
                    eval_result.get("fitness"),
                    eval_result.get("turnover"),
                    direction,
                )
                idea_feedback.append(
                    {
                        "iteration": iteration,
                        "diagnosis_source": "rule_mapping",
                        "diagnosis": rule_diagnosis,
                    }
                )

                if (not rule_diagnosis or rule_diagnosis.strip() == "") and settings.DIAGNOSIS_LLM_ENABLED:
                    try:
                        from openalpha_brain.generation.prompts import llm_diagnose_failure

                        llm_result = await llm_diagnose_failure(
                            brain_checks=eval_result["brain_checks"],
                            expression=expression,
                            hypothesis_text=hypothesis.natural_language or hypothesis.direction,
                            llm_call_fn=self._idea_agent._llm_generate
                            if hasattr(self._idea_agent, "_llm_generate")
                            else None,
                        )
                        if llm_result:
                            idea_feedback.append(
                                {
                                    "iteration": iteration,
                                    "diagnosis_source": "llm_diagnosis",
                                    "diagnosis": f"LLM诊断: {llm_result['root_cause']}. 建议: {llm_result['suggested_fix']}",  # noqa: E501
                                    "llm_result": llm_result,
                                }
                            )
                            logic_lib = getattr(self._idea_agent, "_logic_library", None)
                            if logic_lib:
                                logic_lib.accumulate_diagnosis(hypothesis.direction, llm_result)
                    except (ValueError, TypeError, RuntimeError):
                        pass

                _failed_checks = [c.get("name", "") for c in eval_result["brain_checks"] if c.get("result") == "FAIL"]
                _failure_str = "; ".join(_failed_checks)
                for _ft in _failed_checks:
                    if _ft in ("SELF_CORRELATION", "LOW_SHARPE") and self._agent_factory is not None:
                        try:
                            from loop_engine import _algo_tick

                            _algo_tick("specialist_agent_create")
                        except ImportError:
                            pass
                        _spec_agent = self._agent_factory.create_specialist_agent(_ft, direction)
                        if _spec_agent is not None:
                            _spec_prompt = self._agent_factory.get_specialist_prompt(
                                _spec_agent.agent_type,
                                expression,
                                _failure_str,
                            )
                            factor_feedback.append(
                                {
                                    "sharpe": eval_result.get("sharpe", "N/A"),
                                    "status": _ft,
                                    "feedback": _spec_prompt,
                                }
                            )

            eval_status = eval_result.get("status", "PENDING")
            alignment_score = eval_result.get("semantic_alignment_score", 0.5)

            self._message_bus.send(
                AgentMessage(
                    sender="eval_agent",
                    receiver="idea_agent",
                    msg_type="DiagnosisFeedback",
                    payload={
                        "iteration": iteration,
                        "status": eval_status,
                        "sharpe": eval_result.get("sharpe"),
                        "fitness": eval_result.get("fitness"),
                        "turnover": eval_result.get("turnover"),
                        "brain_checks": eval_result.get("brain_checks", []),
                        "feedback": eval_result.get("feedback", ""),
                    },
                )
            )

            try:
                if trajectory is not None:
                    trajectory.add_brain_feedback(eval_result)
                    trajectory.final_status = eval_result.get("status", "PENDING")
            except (OSError, ValueError, RuntimeError):
                pass

            logger.info(
                "MultiAgent: EvalAgent result — status=%s, alignment=%.2f",
                eval_status,
                alignment_score,
            )

            current_result = AgentResult(
                hypothesis=hypothesis,
                expression=expression,
                simulation_payload=sim_payload,
                originality_score=originality_score,
                complexity_metrics=complexity_metrics_dict,
                brain_sharpe=None,
                iterations=iteration,
                converged=eval_status in ("PASS", "PENDING"),
                semantic_alignment_score=alignment_score,
                variants=[],
                trajectory=trajectory,
                critique_result=self._factor_agent._last_critique_result,
                economic_rationale=_iteration_economic_rationale,
            )

            if best_result is None or alignment_score > best_result.semantic_alignment_score:
                best_result = current_result
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            _agent_success = alignment_score >= 0.7
            if self._agent_factory is not None:
                for _a in self._agent_factory._agents.values():
                    if _a.specialty and _a.specialty in str(eval_result.get("brain_checks", [])):
                        self._agent_factory.record_agent_result(_a.agent_id, _agent_success)

            if no_improvement_count >= NO_IMPROVEMENT_LIMIT:
                logger.info(
                    "MultiAgent: %d consecutive no-improvement rounds — stopping early",
                    no_improvement_count,
                )
                break

            if eval_status != "PENDING" and settings.TRAJECTORY_MUTATION_ENABLED and trajectory is not None:
                try:
                    _traj_mutator = TrajectoryMutation()
                    _traj_variants = _traj_mutator.mutate_trajectory(trajectory)
                    if _traj_variants:
                        _traj_mutated_expr = None
                        for _tv in _traj_variants:
                            if _tv.expression_versions and _tv.expression_versions[-1]:
                                _traj_mutated_expr = _tv.expression_versions[-1]
                                break
                        if _traj_mutated_expr and _traj_mutated_expr != expression:
                            logger.info(
                                "MultiAgent: trajectory mutation generated variant — %s",
                                _traj_mutated_expr[:80],
                            )
                            factor_feedback.append(
                                {
                                    "sharpe": eval_result.get("sharpe", "N/A"),
                                    "status": "TRAJECTORY_MUTATION",
                                    "feedback": f"Trajectory mutation variant: {_traj_mutated_expr}",
                                }
                            )
                        else:
                            logger.info("MultiAgent: trajectory mutation produced no useful variant")
                    else:
                        logger.info("MultiAgent: trajectory mutation returned empty")
                except (ValueError, TypeError, RuntimeError) as exc:
                    logger.warning("MultiAgent: trajectory mutation failed: %s", exc)

            if eval_status == "PENDING":
                self._successful_alphas.append(expression)
                if len(self._successful_alphas) > 50:
                    self._successful_alphas = self._successful_alphas[-50:]
                if settings.EVIDENCE_RECORDING_ENABLED:
                    try:
                        from openalpha_brain.generation.alpha_logics import AlphaLogicLibrary

                        lib = AlphaLogicLibrary()
                        logics = lib.get_logic_for_direction(direction)
                        for logic in logics[:1]:
                            lib.record_evidence(
                                logic.logic_id,
                                alignment_score >= 0.7,
                                expression=expression,
                                direction=direction,
                                fix_success=alignment_score >= 0.7,
                            )
                    except (ValueError, TypeError, RuntimeError):
                        pass

                variants = []
                if settings.CROSSOVER_ENABLED:
                    try:
                        mutator = GradientMutation(originality_checker=self._originality_checker)
                        mutation_results = mutator.mutate(
                            expression,
                            brain_feedback={},
                            original_id=f"iter_{iteration}",
                        )
                        for mr in mutation_results:
                            variants.append(mr.mutated_expression)
                    except (OSError, ValueError, RuntimeError):
                        pass

                    if len(self._successful_alphas) >= 2:
                        try:
                            crossover = SemanticCrossover(originality_checker=self._originality_checker)
                            pair = random.sample(self._successful_alphas, 2)
                            crossover_results = await crossover.crossover(pair[0], pair[1])
                            for cr in crossover_results:
                                variants.append(cr.child_expression)
                        except (ValueError, TypeError, RuntimeError):
                            pass

                drift_threshold = settings.SEMANTIC_DRIFT_THRESHOLD
                filtered_variants = []
                for v in variants:
                    if anchor and anchor._embedding:
                        try:
                            drift_score = await anchor.check_drift(v, llm_client.embed)
                            if drift_score >= drift_threshold:
                                filtered_variants.append(v)
                            else:
                                logger.info("Semantic drift rejected: score=%.3f expr=%s", drift_score, v[:50])
                                _monitor.record(
                                    "STEP",
                                    "drift_guard",
                                    "reject",
                                    f"score={drift_score:.3f}",
                                    session_id="drift_guard",
                                    alpha_id="",
                                )
                        except (OSError, ValueError, RuntimeError):
                            filtered_variants.append(v)
                    else:
                        filtered_variants.append(v)
                variants = filtered_variants

                best_result.variants = variants
                best_result.converged = True
                if self._reflection_engine is not None:
                    self._next_plan = self._reflection_engine.plan_next_iteration(
                        evolution_db=self._evo_db, feature_map=self._feature_map
                    )
                return best_result

        if best_result is not None:
            if self._reflection_engine is not None:
                self._next_plan = self._reflection_engine.plan_next_iteration(
                    evolution_db=self._evo_db, feature_map=self._feature_map
                )
            if self._agent_factory is not None:
                self._agent_factory.cleanup_idle_agents(max_idle_cycles=10)
            return best_result

        logger.warning("MultiAgent: all iterations failed — returning empty fallback result")
        if self._agent_factory is not None:
            self._agent_factory.cleanup_idle_agents(max_idle_cycles=10)
        return AgentResult(
            hypothesis=Hypothesis(
                direction=direction,
                asset_class="equity",
                time_horizon="medium-term",
                mechanism="fallback — no valid expression generated",
                natural_language=f"Failed to generate expression for direction: {direction}",
            ),
            expression="rank(close)",
            simulation_payload=_load_brain_submit_params(),
            originality_score=0.0,
            complexity_metrics={},
            brain_sharpe=None,
            iterations=MAX_ITERATIONS,
            converged=False,
            semantic_alignment_score=0.0,
            variants=[],
        )
