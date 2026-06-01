from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

import aiohttp
import numpy as np

from openalpha_brain.monitoring.algorithm_telemetry import AlgorithmTelemetryCollector
from openalpha_brain.utils.algo_logger import Timer, algo_log, log_call

logger = logging.getLogger(__name__)

_DIRECTION_KEYWORDS: dict[str, list[str]] = {
    "momentum": ["ts_delta", "ts_returns", "momentum", "ts_momentum"],
    "mean_reversion": ["ts_zscore", "ts_regression", "mean_reversion", "-rank"],
    "volatility": ["ts_std_dev", "ts_variance", "volatility", "signed_power"],
    "statistical": ["ts_regression", "ts_corr", "ts_covariance", "ts_rank"],
    "volume": ["volume", "adv", "vwap", "trade_when"],
    "interaction": ["rank(", "*rank", "ts_corr"],
}

_TIME_HORIZON_PATTERNS = [
    (r"ts_\w+\([^,]+,\s*(\d+)", "lookback"),
]

_MECHANISM_KEYWORDS: dict[str, list[str]] = {
    "normalized": ["/ ts_std_dev", "/ts_std_dev", "ts_zscore", "ts_regression"],
    "conditional": ["trade_when", "greater(", "less(", "if_else"],
    "interaction": ["*rank(", "rank(", "rank("],
    "signal": [],
}

_CLASSIFICATION_CALIBRATION_PROMPT = """You are a quantitative factor strategy classifier. Verify or correct this rule-based classification.  # noqa: E501

Expression: {expression}
Rule-based classification: direction="{direction}", confidence={confidence:.2f}
Detected operators: {operators}
Inferred mechanism: {mechanism}

Questions:
1. Is the rule-based direction correct? (yes/no/partial)
2. What is the TRUE economic strategy this expression implements?
3. Rate your own confidence (0-1).

Reply format:
DIRECTION: <corrected_direction>
CONFIDENCE: <your_confidence>
REASONING: <1-2 sentence explanation>
AGREE_WITH_RULES: <yes/no>
"""


@dataclass
class StrategyProfile:
    expr: str = ""
    direction: str = "momentum"
    time_horizon: str = "medium"
    mechanism: str = "signal"
    sharpe: float = 0.0
    fitness: float = 0.0
    turnover: float = 0.0
    sub_universe_sharpe: float | None = None
    complexity: int = 0
    operators: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    llm_corrected: bool = False
    llm_confidence: float | None = None
    calibration_details: str = ""


class StrategyClassifier:
    def __init__(
        self,
        path: str = "runtime/strategy_profiles.json",
        embed_fn: Callable[..., Awaitable] | None = None,
        llm_generate_fn: Callable[..., Awaitable] | None = None,
    ):
        self._path = Path(path)
        self._profiles: list[StrategyProfile] = []
        self._embed_fn = embed_fn
        self._llm_generate_fn = llm_generate_fn
        self._tel = AlgorithmTelemetryCollector.get_instance()
        self._calibration_stats: dict[str, int] = {"total": 0, "llm_called": 0, "llm_agreed": 0, "llm_overruled": 0}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._profiles = [StrategyProfile(**p) for p in data.get("profiles", [])]
            logger.info("StrategyClassifier: loaded %d profiles", len(self._profiles))
        except (OSError, FileNotFoundError, json.JSONDecodeError, ValueError):
            logger.warning("StrategyClassifier: failed to load", exc_info=True)
            self._profiles = []

    def _save(self) -> None:
        try:
            data = {"profiles": [asdict(p) for p in self._profiles]}
            self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except (OSError, ValueError):
            logger.warning("StrategyClassifier: failed to save", exc_info=True)

    def set_embed_fn(self, fn: Callable[..., Awaitable]) -> None:
        self._embed_fn = fn

    @algo_log(log_args_to_skip=("self", "brain_result"))
    async def classify(self, expression: str, brain_result: dict | None = None) -> StrategyProfile:
        eid = None
        try:
            eid = await self._tel.record_enter(
                "StrategyClassifier", cycle_id="unknown", expr_id=hash(expression) % 10000
            )
            t0 = time.perf_counter()
            with Timer("classify"):
                direction, dir_flags = self._infer_direction(expression)
                time_horizon = self._infer_time_horizon(expression)
                mechanism = self._infer_mechanism(expression)
                operators = self._extract_operators(expression)
                complexity = expression.count("(")

                direction_confidence = self._calculate_direction_confidence(expression, direction)
                if 0.5 <= direction_confidence <= 0.65:
                    logger.warning(
                        "策略分类置信度处于边界区间 (confidence=%.4f)，direction='%s' 可能不够准确",
                        direction_confidence,
                        direction,
                    )

                embedding = None
                if self._embed_fn is not None:
                    try:
                        vec = await self._embed_fn(expression)
                        if isinstance(vec, list):
                            embedding = vec[0] if len(vec) > 0 and isinstance(vec[0], list) else vec
                        elif isinstance(vec, np.ndarray):
                            embedding = vec.tolist()
                    except (ValueError, TypeError, OSError):
                        logger.warning("StrategyClassifier: embed failed for expression")
                profile = StrategyProfile(
                    expr=expression,
                    direction=direction,
                    time_horizon=time_horizon,
                    mechanism=mechanism,
                    complexity=complexity,
                    operators=operators,
                    embedding=embedding,
                )
                if brain_result:
                    profile.sharpe = brain_result.get("sharpe", 0.0) or 0.0
                    profile.fitness = brain_result.get("fitness", 0.0) or 0.0
                    profile.turnover = brain_result.get("turnover", 0.0) or 0.0
                    profile.sub_universe_sharpe = brain_result.get("sub_universe_sharpe")

                needs_calibration = (
                    direction_confidence < 0.65 or dir_flags.get("ambiguous", False) or dir_flags.get("tied", False)
                )
                llm_calibrated = False
                if needs_calibration and self._llm_generate_fn is not None:
                    self._calibration_stats["total"] += 1
                    logger.info(
                        "[STRAT-CLASS-LLM] Triggering LLM calibration: confidence=%.3f, ambiguous=%s, tied=%s, direction='%s'",  # noqa: E501
                        direction_confidence,
                        dir_flags.get("ambiguous", False),
                        dir_flags.get("tied", False),
                        direction,
                    )
                    profile = await self._llm_calibrate_classification(expression, profile)
                    llm_calibrated = profile.llm_corrected

                log_call(
                    "classify",
                    input={
                        "expression": expression[:100] + ("..." if len(expression) > 100 else ""),
                        "expression_length": len(expression),
                    },
                    output={
                        "direction": profile.direction,
                        "time_horizon": time_horizon,
                        "mechanism": mechanism,
                        "confidence_score": round(direction_confidence, 4),
                        "complexity": complexity,
                        "operators_count": len(operators),
                        "llm_corrected": profile.llm_corrected,
                        "llm_calibrated": profile.calibration_details != "",
                    },
                    extra={
                        "classification_details": {
                            "inferred_direction": profile.direction,
                            "inferred_mechanism": mechanism,
                            "confidence_level": "high"
                            if direction_confidence > 0.65
                            else ("low" if direction_confidence < 0.5 else "boundary"),
                            "dir_flags": dir_flags,
                            "calibration_applied": profile.calibration_details,
                        }
                    },
                )

                self._profiles.append(profile)
                self._save()

            ms = (time.perf_counter() - t0) * 1000
            with contextlib.suppress(OSError, ValueError, RuntimeError):
                await self._tel.record_exit(
                    "StrategyClassifier",
                    eid,
                    metrics={
                        "direction": profile.direction,
                        "confidence": round(direction_confidence, 3),
                        "llm_calibrated": llm_calibrated,
                    },
                    duration_ms=ms,
                )
            return profile
        except (ValueError, TypeError, OSError, RuntimeError, KeyError, AttributeError) as e:
            if eid:
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    await self._tel.record_error("StrategyClassifier", str(e), type(e).__name__)
            raise

    def _calculate_direction_confidence(self, expr: str, inferred_direction: str) -> float:
        expr_lower = expr.lower()
        scores: dict[str, float] = {}
        for direction, keywords in _DIRECTION_KEYWORDS.items():
            score = sum(1.0 for kw in keywords if kw in expr_lower)
            scores[direction] = score

        total_matches = sum(scores.values())
        if total_matches == 0:
            return 0.3

        max_score = max(scores.values())
        confidence = max_score / (total_matches + 1e-6)
        return round(min(1.0, confidence), 4)

    @algo_log()
    def _infer_direction(self, expr: str) -> tuple[str, dict[str, bool]]:
        expr_lower = expr.lower()
        scores: dict[str, float] = {}
        for direction, keywords in _DIRECTION_KEYWORDS.items():
            score = sum(1.0 for kw in keywords if kw in expr_lower)
            scores[direction] = score
        if not scores or max(scores.values()) == 0:
            return "momentum", {"ambiguous": False, "tied": False}
        best_direction = max(scores, key=lambda k: cast(float, scores[k]))
        directions_with_matches = [d for d, s in scores.items() if s > 0]
        is_ambiguous = len(directions_with_matches) >= 3
        sorted_scores = sorted(scores.values(), reverse=True)
        is_tied = len(sorted_scores) >= 2 and sorted_scores[0] == sorted_scores[1] and sorted_scores[0] > 0
        return best_direction, {"ambiguous": is_ambiguous, "tied": is_tied}

    def _infer_time_horizon(self, expr: str) -> str:
        lookbacks = []
        for pattern, _ in _TIME_HORIZON_PATTERNS:
            for match in re.finditer(pattern, expr):
                lookbacks.append(int(match.group(1)))
        if not lookbacks:
            return "medium"
        max_lb = max(lookbacks)
        if max_lb > 30:
            return "long"
        if max_lb > 10:
            return "medium"
        return "short"

    def _infer_mechanism(self, expr: str) -> str:
        expr_lower = expr.lower()
        for mechanism, keywords in _MECHANISM_KEYWORDS.items():
            if mechanism == "signal":
                continue
            if any(kw in expr_lower for kw in keywords):
                if mechanism == "interaction" and expr.count("rank(") < 2:
                    continue
                return mechanism
        return "signal"

    def _extract_operators(self, expr: str) -> list[str]:
        return sorted(set(re.findall(r"\b(ts_\w+|rank|group_neutralize|trade_when|signed_power)\b", expr)))

    @algo_log(log_args_to_skip=("self",))
    async def _llm_calibrate_classification(self, expression: str, rule_result: StrategyProfile) -> StrategyProfile:
        eid = None
        try:
            eid = await self._tel.record_enter(
                "StrategyClassifier", cycle_id="unknown", expr_id=hash(expression) % 10000
            )
            t0 = time.perf_counter()

            self._calibration_stats["llm_called"] += 1
            prompt = _CLASSIFICATION_CALIBRATION_PROMPT.format(
                expression=expression,
                direction=rule_result.direction,
                confidence=self._calculate_direction_confidence(expression, rule_result.direction),
                operators=", ".join(rule_result.operators) if rule_result.operators else "none",
                mechanism=rule_result.mechanism,
            )
            try:
                llm_response = await asyncio.wait_for(
                    self._llm_generate_fn(prompt),
                    timeout=12.0,
                )
            except (TimeoutError, aiohttp.ClientError, ValueError, json.JSONDecodeError):
                logger.warning(
                    "[STRAT-CLASS-LLM] LLM call failed or timed out (12s), returning rule-based result unchanged"
                )
                ms = (time.perf_counter() - t0) * 1000
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    await self._tel.record_exit(
                        "StrategyClassifier",
                        eid,
                        metrics={"agreed": True, "error": "timeout_or_exception"},
                        duration_ms=ms,
                    )
                return rule_result
            if not isinstance(llm_response, str):
                logger.warning("[STRAT-CLASS-LLM] LLM returned non-string response, returning rule-based result")
                ms = (time.perf_counter() - t0) * 1000
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    await self._tel.record_exit(
                        "StrategyClassifier",
                        eid,
                        metrics={"agreed": True, "error": "non_string_response"},
                        duration_ms=ms,
                    )
                return rule_result
            response_text = llm_response.strip()
            direction_match = re.search(r"DIRECTION:\s*(\w+)", response_text, re.IGNORECASE)
            confidence_match = re.search(r"CONFIDENCE:\s*([\d.]+)", response_text)
            reasoning_match = re.search(
                r"REASONING:\s*(.+?)(?:\nAGREE_WITH_RULES:|$)", response_text, re.DOTALL | re.IGNORECASE
            )
            agree_match = re.search(r"AGREE_WITH_RULES:\s*(yes|no)", response_text, re.IGNORECASE)

            if not direction_match:
                logger.warning(
                    "[STRAT-CLASS-LLM] Could not parse DIRECTION from LLM response, keeping rule-based result"
                )
                ms = (time.perf_counter() - t0) * 1000
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    await self._tel.record_exit(
                        "StrategyClassifier", eid, metrics={"agreed": True, "error": "parse_failed"}, duration_ms=ms
                    )
                return rule_result

            corrected_direction = direction_match.group(1).lower().strip()
            llm_confidence_val = float(confidence_match.group(1)) if confidence_match else None
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
            agrees = agree_match and agree_match.group(1).lower() == "yes"

            valid_directions = set(_DIRECTION_KEYWORDS.keys())
            if corrected_direction not in valid_directions:
                logger.warning(
                    "[STRAT-CLASS-LLM] LLM returned unknown direction '%s', keeping rule-based result",
                    corrected_direction,
                )
                ms = (time.perf_counter() - t0) * 1000
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    await self._tel.record_exit(
                        "StrategyClassifier",
                        eid,
                        metrics={"agreed": True, "error": "unknown_direction"},
                        duration_ms=ms,
                    )
                return rule_result

            if agrees and corrected_direction == rule_result.direction:
                self._calibration_stats["llm_agreed"] += 1
                logger.info(
                    "[STRAT-CLASS-LLM] LLM agreed with rule-based classification (direction='%s', confidence boost +0.10)",  # noqa: E501
                    corrected_direction,
                )
                min(1.0, self._calculate_direction_confidence(expression, rule_result.direction) + 0.10)
                result = StrategyProfile(
                    expr=rule_result.expr,
                    direction=rule_result.direction,
                    time_horizon=rule_result.time_horizon,
                    mechanism=rule_result.mechanism,
                    sharpe=rule_result.sharpe,
                    fitness=rule_result.fitness,
                    turnover=rule_result.turnover,
                    sub_universe_sharpe=rule_result.sub_universe_sharpe,
                    complexity=rule_result.complexity,
                    operators=list(rule_result.operators),
                    embedding=list(rule_result.embedding) if rule_result.embedding else None,
                    llm_corrected=False,
                    llm_confidence=llm_confidence_val,
                    calibration_details=f"LLM agreed. {reasoning}",
                )
                ms = (time.perf_counter() - t0) * 1000
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    await self._tel.record_exit(
                        "StrategyClassifier", eid, metrics={"agreed": True, "overruled": False}, duration_ms=ms
                    )
                return result

            self._calibration_stats["llm_overruled"] += 1
            logger.info(
                "[STRAT-CLASS-LLM] LLM overruled rule-based classification: '%s' -> '%s' (reason: %s)",
                rule_result.direction,
                corrected_direction,
                reasoning,
            )
            result = StrategyProfile(
                expr=rule_result.expr,
                direction=corrected_direction,
                time_horizon=rule_result.time_horizon,
                mechanism=rule_result.mechanism,
                sharpe=rule_result.sharpe,
                fitness=rule_result.fitness,
                turnover=rule_result.turnover,
                sub_universe_sharpe=rule_result.sub_universe_sharpe,
                complexity=rule_result.complexity,
                operators=list(rule_result.operators),
                embedding=list(rule_result.embedding) if rule_result.embedding else None,
                llm_corrected=True,
                llm_confidence=llm_confidence_val,
                calibration_details=f"LLM corrected from '{rule_result.direction}' to '{corrected_direction}'. {reasoning}",  # noqa: E501
            )
            ms = (time.perf_counter() - t0) * 1000
            with contextlib.suppress(OSError, ValueError, RuntimeError):
                await self._tel.record_exit(
                    "StrategyClassifier", eid, metrics={"agreed": False, "overruled": True}, duration_ms=ms
                )
            return result
        except (ValueError, TypeError, OSError, RuntimeError, KeyError, AttributeError) as e:
            if eid:
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    await self._tel.record_error("StrategyClassifier", str(e), type(e).__name__)
            raise

    @algo_log()
    def find_complementary(self, existing: list[StrategyProfile] | None = None, n: int = 5) -> list[dict]:
        with Timer("find_complementary"):
            if existing is None:
                existing = self._profiles

            covered_directions = {p.direction for p in existing}
            covered_mechanisms = {p.mechanism for p in existing}
            covered_horizons = {p.time_horizon for p in existing}

            avg_sharpe = sum(p.sharpe for p in existing) / len(existing) if existing else 0.0
            direction_counts: dict[str, int] = {}
            for p in existing:
                direction_counts[p.direction] = direction_counts.get(p.direction, 0) + 1

            suggestions: list[dict[str, object]] = []
            all_directions = list(_DIRECTION_KEYWORDS.keys())

            for d in all_directions:
                if d not in covered_directions:
                    direction_score = 0.7
                    if avg_sharpe > 1.0:
                        direction_score += 0.15
                    count_penalty = min(
                        direction_counts.get(max(direction_counts, key=direction_counts.get, default=d), 0) * 0.02, 0.1
                    )
                    relevance_score = max(0.0, direction_score - count_penalty)
                    suggestions.append(
                        {
                            "direction": d,
                            "reason": f"direction '{d}' not yet explored",
                            "relevance_score": round(relevance_score, 4),
                            "score_breakdown": {
                                "base_direction_score": round(0.7, 4),
                                "sharpe_bonus": round(0.15 if avg_sharpe > 1.0 else 0.0, 4),
                                "concentration_penalty": round(-count_penalty, 4),
                            },
                        }
                    )

            for m in _MECHANISM_KEYWORDS:
                if m not in covered_mechanisms:
                    mechanism_score = 0.5
                    if m == "interaction" and any(d == "statistical" for d in covered_directions):
                        mechanism_score += 0.15
                    elif m == "conditional" and len(existing) >= 3:
                        mechanism_score += 0.1
                    relevance_score = min(1.0, mechanism_score)
                    suggestions.append(
                        {
                            "direction": "any",
                            "mechanism": m,
                            "reason": f"mechanism '{m}' not yet explored",
                            "relevance_score": round(relevance_score, 4),
                            "score_breakdown": {
                                "base_mechanism_score": round(0.5, 4),
                                "synergy_bonus": round(relevance_score - 0.5, 4),
                            },
                        }
                    )

            for h in ["short", "medium", "long"]:
                if h not in covered_horizons:
                    horizon_score = 0.45
                    if h == "long" and "short" in covered_horizons or h == "short" and "long" in covered_horizons:
                        horizon_score += 0.12
                    relevance_score = min(1.0, horizon_score)
                    suggestions.append(
                        {
                            "direction": "any",
                            "time_horizon": h,
                            "reason": f"horizon '{h}' not yet explored",
                            "relevance_score": round(relevance_score, 4),
                            "score_breakdown": {
                                "base_horizon_score": round(0.45, 4),
                                "diversification_bonus": round(relevance_score - 0.45, 4),
                            },
                        }
                    )

            suggestions.sort(key=lambda x: cast(float, x["relevance_score"]), reverse=True)

            top_suggestions = suggestions[:n]

            if top_suggestions:
                max_score = cast(
                    float, max(top_suggestions, key=lambda x: cast(float, x["relevance_score"]))["relevance_score"]
                )
                if max_score < 0.5:
                    logger.warning(
                        "所有互补策略建议的相关性得分均低于阈值 (max_score=%.4f < 0.50)，建议审查现有策略组合",
                        max_score,
                    )

            log_call(
                "find_complementary",
                input={
                    "existing_count": len(existing),
                    "covered_directions": list(covered_directions),
                    "covered_mechanisms": list(covered_mechanisms),
                    "avg_sharpe": round(avg_sharpe, 4),
                },
                output={
                    "total_suggestions": len(suggestions),
                    "top_k_returned": len(top_suggestions),
                    "scores": [round(cast(float, s["relevance_score"]), 4) for s in top_suggestions],
                },
                extra={
                    "all_scores_details": [
                        {"reason": s["reason"], "score": s["relevance_score"], "breakdown": s["score_breakdown"]}
                        for s in top_suggestions
                    ]
                },
            )

            return top_suggestions

    def get_profiles_by_direction(self, direction: str) -> list[StrategyProfile]:
        """Return all profiles matching the given strategy direction.

        Useful for analysing how many alphas have been explored in a specific
        direction (e.g. "momentum" or "mean_reversion") and their characteristics.
        """
        return [p for p in self._profiles if p.direction == direction]

    def get_top_profiles(self, n: int = 5) -> list[StrategyProfile]:
        """Return the top *n* profiles sorted by sharpe (descending).

        Sharpe ratio serves as the success-rate proxy. Useful for quickly
        identifying the best-performing strategy profiles in the library.
        """
        return sorted(self._profiles, key=lambda p: p.sharpe, reverse=True)[:n]

    async def find_similar_by_embedding(self, query_expr: str, top_k: int = 3) -> list[dict]:
        if self._embed_fn is None:
            return []
        profiles_with_emb = [p for p in self._profiles if p.embedding is not None]
        if not profiles_with_emb:
            return []
        try:
            query_vec = await self._embed_fn(query_expr)
            if isinstance(query_vec, np.ndarray):
                query_vec = query_vec.tolist()
            if isinstance(query_vec, list) and len(query_vec) > 0 and isinstance(query_vec[0], list):
                query_vec = query_vec[0]
            if not query_vec:
                return []
            results = []
            for p in profiles_with_emb:
                if p.embedding is None:
                    continue
                dot = sum(a * b for a, b in zip(query_vec, p.embedding, strict=False))
                norm_a = math.sqrt(sum(a * a for a in query_vec))
                norm_b = math.sqrt(sum(b * b for b in p.embedding))
                if norm_a == 0 or norm_b == 0:
                    continue
                sim = dot / (norm_a * norm_b)
                results.append({"profile": p, "similarity": round(sim, 4)})
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
        except (OSError, ValueError, RuntimeError):
            logger.warning("StrategyClassifier: find_similar_by_embedding failed")
            return []
