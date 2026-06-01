from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from alpha_agent.datasets_loader import FieldMetadata
from alpha_agent.operator_registry import get_operator_registry


@dataclass
class ValidationReport:
    passed: bool
    rule_results: Dict[str, bool]
    errors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "rule_results": self.rule_results,
            "errors": self.errors,
        }


_VALID_OPERATORS_FALLBACK: Set[str] = {
    "rank", "zscore", "scale", "winsorize", "group_neutralize",
    "ts_sum", "ts_mean", "ts_std_dev", "ts_delta",
    "ts_arg_max", "ts_arg_min", "ts_rank", "ts_product",
    "ts_corr", "ts_covariance", "ts_regression",
    "ts_quantile", "ts_zscore",
    "group_rank", "group_mean", "group_zscore",
    "ts_decay_linear",
    "log", "abs", "sign", "sqrt", "power",
    "max", "min", "if_else",
}


def _get_valid_operators() -> Set[str]:
    reg = get_operator_registry()
    official = reg.get_valid_names()
    if official:
        return official
    return _VALID_OPERATORS_FALLBACK


VALID_OPERATORS: Set[str] = _get_valid_operators()


@dataclass
class RAGSpecEncoder:
    corpus: List[str] = field(default_factory=list)
    field_index: Dict[str, int] = field(default_factory=dict)
    vectorizer: TfidfVectorizer = field(default_factory=lambda: TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=2000,
        stop_words="english",
    ))
    _fitted: bool = False

    @classmethod
    def from_fields_summary(cls, path: Path) -> RAGSpecEncoder:
        with open(path, "r") as f:
            raw = json.load(f)
        field_ids: List[str] = [item["id"] for item in raw]
        corpus = [cls._field_document(fid) for fid in field_ids]
        field_index = {fid: i for i, fid in enumerate(field_ids)}
        encoder = cls(corpus=corpus, field_index=field_index)
        encoder._fit()
        return encoder

    @classmethod
    def from_csv_metadata(cls, metadata: Dict[str, FieldMetadata]) -> RAGSpecEncoder:
        field_ids: List[str] = list(metadata.keys())
        corpus: List[str] = []
        for fid in field_ids:
            desc = metadata[fid].description
            doc = desc if desc else cls._field_document(fid)
            corpus.append(doc)
        field_index = {fid: i for i, fid in enumerate(field_ids)}
        encoder = cls(corpus=corpus, field_index=field_index)
        encoder._fit()
        return encoder

    @staticmethod
    def _field_document(field_id: str) -> str:
        tokens = re.split(r"_|\d+", field_id)
        tokens = [t for t in tokens if t and len(t) > 1]
        return " ".join(tokens) if tokens else field_id

    def _fit(self) -> None:
        if not self._fitted and self.corpus:
            self.vectorizer.fit(self.corpus)
            self._fitted = True

    def encode(self, text: str):
        if not self._fitted:
            self._fit()
        if not self._fitted or not self.corpus:
            from scipy.sparse import csr_matrix
            return csr_matrix((1, self.vectorizer.max_features))
        return self.vectorizer.transform([text])

    def similarity(self, a: str, b: str) -> float:
        vec_a = self.encode(a)
        vec_b = self.encode(b)
        dot = (vec_a @ vec_b.T).toarray()[0, 0]
        norm_a = np.sqrt((vec_a.power(2)).sum())
        norm_b = np.sqrt((vec_b.power(2)).sum())
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def recommend_fields(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self._fitted or not self.corpus:
            return []
        q_vec = self.encode(query)
        corpus_vec = self.vectorizer.transform(self.corpus)
        scores = (corpus_vec @ q_vec.T).toarray().flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        idx_to_field = {v: k for k, v in self.field_index.items()}
        return [(idx_to_field[i], round(float(scores[i]), 4)) for i in top_indices if scores[i] > 0]

    def spec_decoder(self, spec_text: str) -> Dict[str, Any]:
        vector = self.encode(spec_text)
        feature_names = self.vectorizer.get_feature_names_out()
        dense = vector.toarray()[0]
        top_indices = dense.argsort()[::-1][:10]
        top_features = [
            {"feature": str(feature_names[i]), "weight": round(float(dense[i]), 4)}
            for i in top_indices if dense[i] > 0
        ]
        return {
            "feature_count": int((dense > 0).sum()),
            "top_features": top_features,
        }


class DeterministicValidator:
    RULES = ["D6.1", "D6.2", "D6.3", "D6.4", "D6.5", "D6.6"]

    def __init__(
        self,
        known_fields: Optional[Set[str]] = None,
        max_complexity: int = 12,
        max_tokens: int = 80,
    ) -> None:
        self.known_fields: Set[str] = known_fields or set()
        self.max_complexity = max_complexity
        self.max_tokens = max_tokens

    @classmethod
    def from_fields_summary(cls, path: Path) -> DeterministicValidator:
        with open(path, "r") as f:
            raw = json.load(f)
        known_fields = {item["id"] for item in raw}
        return cls(known_fields=known_fields)

    @classmethod
    def from_csv_metadata(cls, metadata: Dict[str, FieldMetadata]) -> DeterministicValidator:
        known_fields = set(metadata.keys())
        return cls(known_fields=known_fields)

    def parse_tokens(self, expression: str) -> List[str]:
        return re.findall(r"[a-zA-Z_]\w*", expression)

    def detect_operators(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t in VALID_OPERATORS]

    def detect_field_refs(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t in self.known_fields]

    def detect_potential_fields(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t not in VALID_OPERATORS and re.match(r"^[a-z][a-z0-9_]*$", t)]

    def check_parentheses(self, expression: str) -> bool:
        stack = 0
        for ch in expression:
            if ch == "(":
                stack += 1
            elif ch == ")":
                stack -= 1
            if stack < 0:
                return False
        return stack == 0

    def validate(self, expression: str) -> ValidationReport:
        errors: List[str] = []
        results: Dict[str, bool] = {}

        tokens = self.parse_tokens(expression)
        operators = self.detect_operators(tokens)
        known_refs = self.detect_field_refs(tokens)
        potential_fields = self.detect_potential_fields(tokens)

        is_nonempty = len(expression.strip()) > 0
        results["D6.1"] = is_nonempty
        if not is_nonempty:
            errors.append("D6.1: Expression is empty")

        has_field = len(potential_fields) > 0
        results["D6.2"] = has_field
        if not has_field:
            errors.append("D6.2: No field references found in expression")

        has_operator = len(operators) > 0
        results["D6.3"] = has_operator
        if not has_operator:
            errors.append("D6.3: No valid operators found in expression")

        within_complexity = len(operators) <= self.max_complexity and len(tokens) <= self.max_tokens
        results["D6.4"] = within_complexity
        if not within_complexity:
            errors.append(
                f"D6.4: Complexity exceeded "
                f"({len(operators)} operators > {self.max_complexity} "
                f"or {len(tokens)} tokens > {self.max_tokens})"
            )

        if self.known_fields and potential_fields:
            unknown = [t for t in potential_fields if t not in self.known_fields]
            all_fields_valid = len(unknown) == 0
            results["D6.5"] = all_fields_valid
            if not all_fields_valid:
                errors.append(f"D6.5: Unknown field references: {unknown}")
        else:
            results["D6.5"] = True

        parens_ok = self.check_parentheses(expression)
        results["D6.6"] = parens_ok
        if not parens_ok:
            errors.append("D6.6: Unmatched parentheses in expression")

        passed = all(results.values())

        return ValidationReport(passed=passed, rule_results=results, errors=errors)
