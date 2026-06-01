from __future__ import annotations

import hashlib
import logging

logger = logging.getLogger(__name__)


class ParamOptimizer:
    DECAY_VALUES = [0, 3, 5, 8, 10, 15, 20]
    NEUTRALIZATION_VALUES = ["INDUSTRY", "SUBINDUSTRY", "MARKET"]

    def __init__(self) -> None:
        self._cache: dict[str, dict] = {}

    def get_param_combinations(self) -> list[dict]:
        combos = []
        for decay in self.DECAY_VALUES:
            for neut in self.NEUTRALIZATION_VALUES:
                combos.append({"decay": decay, "neutralization": neut})
        return combos

    def build_payload(self, expression: str, decay: int, neutralization: str) -> dict:
        return {
            "settings": {
                "decay": decay,
                "neutralization": neutralization,
                "delay": 1,
                "pasteurization": "ON",
                "unitHandling": "VERIFY",
                "nanHandling": "ON",
            },
            "regular": expression,
        }

    def select_best_params(self, results: list[dict]) -> dict | None:
        if not results:
            return None
        valid = [r for r in results if r.get("sharpe") is not None]
        if not valid:
            return None
        return max(valid, key=lambda r: r["sharpe"])

    def should_optimize(self, sharpe: float, gate_sharpe: float) -> bool:
        return sharpe >= (gate_sharpe - 0.5)

    def cache_result(self, expression_hash: str, best_params: dict) -> None:
        self._cache[expression_hash] = best_params

    def get_cached(self, expression_hash: str) -> dict | None:
        return self._cache.get(expression_hash)


def expression_hash(expression: str) -> str:
    return hashlib.sha256(expression.encode()).hexdigest()[:16]
