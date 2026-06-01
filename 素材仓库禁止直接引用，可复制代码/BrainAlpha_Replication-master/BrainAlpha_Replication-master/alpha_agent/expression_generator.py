from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from alpha_agent.config import LLMGenConfig
from alpha_agent.exploration_grid import GridCell
from alpha_agent.llm_client import LLMClient
from alpha_agent.operator_registry import get_operator_registry
from alpha_agent.rag_spec import DeterministicValidator, RAGSpecEncoder


@dataclass
class GeneratedCandidate:
    expression: str
    idea_name: str
    family: str
    diversity_score: float = 0.0
    validation_passed: bool = False
    validation_errors: List[str] = field(default_factory=list)
    cot_reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expression": self.expression,
            "idea_name": self.idea_name,
            "family": self.family,
            "diversity_score": self.diversity_score,
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,
            "cot_reasoning": self.cot_reasoning[:200] if self.cot_reasoning else "",
        }


OPERATOR_TEMPLATES: Dict[str, List[str]] = {
    "cross_sectional": [
        "rank({field})",
        "zscore({field})",
        "group_rank({field}, sector)",
    ],
    "time_series": [
        "ts_mean({field}, {window})",
        "ts_delta({field}, {window})",
        "ts_rank({field}, {window})",
        "ts_sum({field}, {window})",
        "ts_std_dev({field}, {window})",
    ],
    "group": [
        "group_rank({field}, sector)",
        "group_zscore({field}, sector)",
        "group_neutralize({field}, sector)",
    ],
}


DEFAULT_WINDOWS: List[int] = [5, 10, 20, 60]


def tokenize_expression(expression: str) -> Set[str]:
    raw_tokens = re.split(r"[^a-zA-Z0-9_]+", expression.lower())
    return {
        token
        for token in raw_tokens
        if token and not token.isdigit() and len(token) > 1
    }


def jaccard_diversity(
    candidate_tokens: Set[str],
    frontier_token_sets: Sequence[Set[str]],
) -> float:
    if not frontier_token_sets:
        return 1.0
    if not candidate_tokens:
        return 0.0
    max_similarity = 0.0
    for ft in frontier_token_sets:
        if not ft:
            continue
        union = len(candidate_tokens | ft)
        if union == 0:
            continue
        sim = len(candidate_tokens & ft) / union
        if sim > max_similarity:
            max_similarity = sim
    return max(0.0, 1.0 - max_similarity)


DEFAULT_COT_TEMPLATES: Dict[str, str] = {
    "analyst": "Generate an alpha using analyst estimate fields: {fields}. "
               "Use rank and zscore of estimate revisions, surprises, and target prices.",
    "fundamental": "Generate a fundamental alpha using {fields}. "
                   "Use rank and zscore of profitability, valuation, and quality metrics.",
    "model": "Generate an alpha using pre-built ML model scores: {fields}. "
             "Use rank or cross-sectional combination of factor scores.",
    "news": "Generate a news-based alpha using {fields}. "
            "Use ts_mean or ts_delta of sentiment scores with short lookbacks.",
    "option": "Generate an options-market alpha using {fields}. "
              "Use rank of implied volatility, put/call ratios, or OI changes.",
    "price_volume": "Generate a price-volume alpha using {fields}. "
                    "Use ts_mean, ts_delta, or rank of price, volume, and returns.",
    "sentiment": "Generate a sentiment alpha using {fields}. "
                 "Use rank with ts_delta or ts_mean for short horizons.",
    "social_media": "Generate a social media alpha using {fields}. "
                    "Use buzz and engagement metrics with rank or ts_mean operators.",
    "default": "Generate an alpha expression using {fields}. "
               "Use rank, zscore, ts_mean, and ts_delta operators. "
               "Keep the expression between 3-8 operators.",
}


def build_template_expression(
    cell: GridCell,
    rag_encoder: Optional[RAGSpecEncoder] = None,
) -> str:
    fields = cell.candidate_fields[:5]
    if not fields:
        return "rank(close)"
    if rag_encoder:
        query = cell.build_hypothesis()
        recs = rag_encoder.recommend_fields(query, top_k=3)
        if recs:
            fields = [f for f, _ in recs] + fields

    op_name = cell.operator_category.label
    templates = OPERATOR_TEMPLATES.get(op_name, OPERATOR_TEMPLATES["cross_sectional"])
    template = templates[0]

    if op_name == "time_series":
        window = cell.horizon.min_days if cell.horizon.min_days > 1 else DEFAULT_WINDOWS[0]
        expr = template.format(field=fields[0], window=window)
    else:
        expr = template.format(field=fields[0])

    return expr


class JaccardDiversityGate:
    DIVERSITY_THRESHOLD = 0.15

    def __init__(self, threshold: float = DIVERSITY_THRESHOLD) -> None:
        self.threshold = threshold
        self._frontier: List[Set[str]] = []

    def seed_frontier(self, frontier_expressions: Sequence[str]) -> None:
        self._frontier = [tokenize_expression(e) for e in frontier_expressions]

    def add_to_frontier(self, expression: str) -> None:
        self._frontier.append(tokenize_expression(expression))

    def score(self, expression: str) -> float:
        return jaccard_diversity(tokenize_expression(expression), self._frontier)

    def is_diverse(self, expression: str) -> bool:
        return self.score(expression) >= self.threshold

    def filter_diverse(
        self, candidates: Sequence[GeneratedCandidate],
    ) -> List[GeneratedCandidate]:
        result: List[GeneratedCandidate] = []
        for c in candidates:
            tokens = tokenize_expression(c.expression)
            div_score = jaccard_diversity(tokens, self._frontier)
            c.diversity_score = round(div_score, 4)
            if div_score >= self.threshold:
                result.append(c)
            self._frontier.append(tokens)
        return result


class ExpressionGenerator:
    def __init__(
        self,
        validator: Optional[DeterministicValidator] = None,
        rag_encoder: Optional[RAGSpecEncoder] = None,
        diversity_gate: Optional[JaccardDiversityGate] = None,
        llm_client: Optional[LLMClient] = None,
        llm_gen_config: Optional[LLMGenConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.validator = validator
        self.rag_encoder = rag_encoder
        self.diversity_gate = diversity_gate or JaccardDiversityGate()
        self.llm_client = llm_client
        self.llm_gen_config = llm_gen_config
        self._logger = logger or logging.getLogger(__name__)

    def generate_from_cell(
        self,
        cell: GridCell,
        frontier_expressions: Sequence[str] = (),
        fresh_gate: bool = False,
    ) -> List[GeneratedCandidate]:
        if fresh_gate:
            gate = JaccardDiversityGate(threshold=self.diversity_gate.threshold)
        else:
            gate = self.diversity_gate
        if frontier_expressions:
            gate.seed_frontier(frontier_expressions)

        candidates: List[GeneratedCandidate] = []

        if self.llm_client and self.llm_gen_config and self.llm_gen_config.enabled:
            try:
                llm_candidates = self._llm_generate_expressions(cell, gate, max_candidates=4)
                candidates.extend(llm_candidates)
                self._logger.info("LLM generated %d expressions for cell %s", len(llm_candidates), cell.cell_id())
            except Exception as exc:
                self._logger.warning("LLM generation failed, falling back to templates: %s", exc)

        if not candidates:
            base_expr = build_template_expression(cell, self.rag_encoder)
            candidate = self._make_candidate(base_expr, cell, "template")
            candidates.append(candidate)

            if cell.operator_category.label != "cross_sectional":
                alt = build_template_expression(cell, self.rag_encoder)
                alt_fields = cell.candidate_fields[1:3] or cell.candidate_fields
                if alt_fields:
                    base_alt = re.sub(r'\b' + re.escape(cell.candidate_fields[0]) + r'\b', alt_fields[0], base_expr) if cell.candidate_fields else base_expr
                    if base_alt != base_expr:
                        candidates.append(self._make_candidate(base_alt, cell, "alt_field"))

            for _ in range(3):
                variant = self._vary_expression(base_expr)
                if variant and variant != base_expr:
                    candidates.append(self._make_candidate(variant, cell, "variant"))

        validated = self._validate_candidates(candidates)
        diverse = gate.filter_diverse(validated)
        return diverse

    @staticmethod
    def _operator_pattern_to_categories(pattern: str) -> List[str]:
        mapping = {
            "cross_sectional": ["Cross Sectional"],
            "time_series": ["Time Series"],
            "group": ["Group"],
        }
        return mapping.get(pattern, ["Cross Sectional"])

    def _build_cot_prompt(self, cell: GridCell) -> Tuple[str, str]:
        dc_label = cell.dataset_category.label
        cot_template = DEFAULT_COT_TEMPLATES.get(dc_label, DEFAULT_COT_TEMPLATES["default"])
        fields = cell.candidate_fields[:5]
        fields_str = ", ".join(fields) if fields else "close, returns, volume"
        horizon_range = f"{cell.horizon.min_days}-{cell.horizon.max_days} days"

        system_prompt = (
            "You are a quantitative researcher generating WorldQuant FASTEXPR expressions. "
            "Ground every expression in the provided verified field list only. "
            "Return strict JSON with keys: expression, reasoning."
        )

        reg = get_operator_registry()
        operator_names = reg.get_valid_names()
        categories = self._operator_pattern_to_categories(cell.operator_category.label)
        relevant: List[str] = []
        seen: Set[str] = set()
        for cat in categories:
            for op in reg.get_operators_for_category(cat, top_k=6):
                name = op.get("name", "")
                if name not in seen:
                    seen.add(name)
                    relevant.append(name)
        if not relevant:
            relevant = sorted(operator_names)[:12]

        operators_block = reg.format_operators_for_prompt(relevant)

        thesis_block = ""
        if cell.thesis:
            thesis_lines = []
            for t in cell.thesis[:3]:
                thesis_lines.append(f"- {t.short_str()}: {t.key_finding_en}")
            thesis_block = "Academic references:\n" + "\n".join(thesis_lines) + "\n\n"

        user_prompt = (
            f"Cell: [{dc_label}, {cell.operator_category.label}, {cell.horizon.label}]\n"
            f"Hypothesis: {cell.build_hypothesis()}\n"
            f"{thesis_block}"
            f"Verified fields: {fields_str}\n"
            f"Operators: {cot_template}\n"
            f"Constraint: lookback {horizon_range}, universe=TOP3000\n\n"
            "Available operators (Brain official catalog):\n"
            f"{operators_block}\n\n"
            "Use ONLY the operators listed above. Follow their exact signatures.\n\n"
            "Step 1: Identify the core signal.\n"
            "Step 2: Apply the appropriate operator.\n"
            "Step 3: Set lookback window.\n\n"
            "Output the final expression and your reasoning."
        )
        return system_prompt, user_prompt

    def _llm_generate_expressions(
        self,
        cell: GridCell,
        gate: JaccardDiversityGate,
        max_candidates: int = 4,
    ) -> List[GeneratedCandidate]:
        if not self.llm_client or not self.llm_gen_config:
            return []
        if not self.llm_gen_config.cot_enabled:
            return []

        system_prompt, user_prompt = self._build_cot_prompt(cell)
        temperature = self.llm_gen_config.cot_temperature

        candidates: List[GeneratedCandidate] = []
        seen: Set[str] = set()

        for _ in range(max_candidates * 2):
            if len(candidates) >= max_candidates:
                break
            try:
                payload = self.llm_client.request_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                )
                expression = str(payload.get("expression", "")).strip()
                reasoning = str(payload.get("reasoning", "")).strip()
                if not expression or expression in seen:
                    continue
                seen.add(expression)
                gc = GeneratedCandidate(
                    expression=expression,
                    idea_name=f"{cell.dataset_category.label}_{cell.horizon.label}_cot",
                    family=cell.dataset_category.label,
                    cot_reasoning=reasoning,
                )
                candidates.append(gc)
            except Exception:
                break

        if self.validator:
            validated: List[GeneratedCandidate] = []
            for c in candidates:
                report = self.validator.validate(c.expression)
                c.validation_passed = report.passed
                c.validation_errors = report.errors
                if report.passed:
                    validated.append(c)
            candidates = validated

        return candidates

    def _make_candidate(
        self, expression: str, cell: GridCell, source: str,
    ) -> GeneratedCandidate:
        return GeneratedCandidate(
            expression=expression,
            idea_name=f"{cell.dataset_category.label}_{cell.horizon.label}_{source}",
            family=cell.dataset_category.label,
        )

    def _validate_candidates(
        self, candidates: Sequence[GeneratedCandidate],
    ) -> List[GeneratedCandidate]:
        if not self.validator:
            return list(candidates)
        result: List[GeneratedCandidate] = []
        for c in candidates:
            report = self.validator.validate(c.expression)
            c.validation_passed = report.passed
            c.validation_errors = report.errors
            if c.validation_passed:
                result.append(c)
        return result

    @staticmethod
    def _vary_expression(expr: str) -> Optional[str]:
        variations = [
            f"rank({expr})",
            f"zscore({expr})",
            f"ts_mean({expr}, 5)",
            f"ts_delta({expr}, 5)",
            f"group_neutralize({expr}, sector)",
        ]
        return random.choice(variations)
