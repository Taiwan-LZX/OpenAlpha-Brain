from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Awaitable, Optional

import numpy as np

from openalpha_brain.utils.algo_logger import algo_log
from openalpha_brain.utils.extract_json_from_llm import extract_json_from_llm
from openalpha_brain.monitoring.algorithm_telemetry import AlgorithmTelemetryCollector

if TYPE_CHECKING:
    from openalpha_brain.knowledge.evolution_db import EvolutionDatabase
    from openalpha_brain.evolution.quality_diversity import FeatureMap

logger = logging.getLogger(__name__)


@dataclass
class ReflectionResult:
    failure_stage: str = ""
    failure_reason: str = ""
    suggested_fix: str = ""
    confidence: float = 0.0
    avoid_patterns: list[str] = field(default_factory=list)
    llm_diagnosis_available: bool = False
    root_cause: str = ""
    composite_factors: list[str] = field(default_factory=list)


@dataclass
class NextPlan:
    direction: str = ""
    mechanism: str = ""
    time_horizon: str = ""
    strategy: str = ""
    reasoning: str = ""


@dataclass
class CritiqueResult:
    consistency_score: float | None = None
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    critique_available: bool = True


class ReflectionEngine:
    def __init__(self, path: str = "runtime/reflection_log.json", embed_fn: Callable[..., Awaitable] | None = None):
        self._path = Path(path)
        self._reflections: list[dict] = []
        self._llm_generate_fn: Callable[..., Awaitable[Any]] | None = None
        self._embed_fn = embed_fn
        self._tel = AlgorithmTelemetryCollector.get_instance()
        self._load()

    def set_embed_fn(self, fn: Callable[..., Awaitable]) -> None:
        self._embed_fn = fn

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._reflections = data.get("reflections", [])
            logger.info("ReflectionEngine: loaded %d reflections", len(self._reflections))
        except OSError:
            logger.warning("ReflectionEngine: failed to load", exc_info=True)
            self._reflections = []

    def _save(self) -> None:
        try:
            data = {"reflections": self._reflections[-500:]}
            self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError:
            logger.warning("ReflectionEngine: failed to save", exc_info=True)

    def set_llm_generate_fn(self, fn: Callable[..., Awaitable]) -> None:
        self._llm_generate_fn = fn

    @algo_log(log_args_to_skip=("self", "brain_result"))
    async def _llm_diagnose_failure(self, expression: str, brain_result: dict) -> dict:
        eid = None
        try:
            eid = await self._tel.record_enter("ReflectionEngine", cycle_id="unknown", expr_id=hash(expression) % 10000)
            t0 = time.perf_counter()
            import re as _re

            logger.info("[LLM-DIAG] Starting LLM-driven failure diagnosis for expression (len=%d)", len(expression))

            expr_truncated = expression[:300] + ("..." if len(expression) > 300 else "")
            sharpe = brain_result.get("sharpe", 0) or 0
            fitness = brain_result.get("fitness", 0) or 0
            turnover = brain_result.get("turnover", 0) or 0
            returns_val = brain_result.get("returns", 0) or 0
            checks = brain_result.get("checks", [])

            depth = expression.count("(")
            operators = sorted(set(_re.findall(r'\b(?:rank|ts_delta|ts_mean|ts_std|ts_max|ts_min|ts_decay_linear|ts_sum|ts_argmax|ts_argmin|ts_skewness|ts_kurtosis|ts_corr|ts_covariance|ts_product|ts_av_diff|group_neutralize|group_rank|group_zscore|log|abs|sign|pow|sigmoid|tanh|max|min|if_else|where)\b', expression)))
            fields = sorted(set(_re.findall(r'(?:close|open|high|low|volume|vwap|returns|sharesout|cap|sales|earnings|assets|market_cap|adv\d+|bid|ask|bidask|spread|trade_\w+|sale_\w+|fundamental_\w+)\b', expression)))
            neut_level = "subindustry" if "subindustry" in expression else ("industry" if "industry" in expression else ("market" if "market" in expression or "group_neutralize" in expression else "none"))
            has_decay = bool(_re.search(r'ts_decay_linear', expression))
            has_hump = bool(_re.search(r'trade_when|hump', expression, _re.IGNORECASE))

            prompt = (
                f"You are a WorldQuant BRAIN alpha factor diagnostic expert. Analyze why this factor failed WQ review.\n\n"
                f"Expression: {expr_truncated}\n"
                f"Sharpe: {sharpe} | Fitness: {fitness} | Turnover: {turnover}% | Returns: {returns_val}\n"
                f"Check Failures: {checks}\n"
                f"Structure: depth={depth}, ops={operators}, fields={fields}, neutralization={neut_level}, "
                f"decay={'yes' if has_decay else 'no'}, hump/trade_when={'yes' if has_hump else 'no'}\n\n"
                f"Diagnose:\n"
                f"1. Primary failure stage (one of: parameters/turnover/hypothesis/expression/originality/composite)\n"
                f"2. Secondary contributing factors\n"
                f"3. Root cause analysis (2-3 sentences)\n"
                f"4. Up to 3 specific fixes with priority order\n"
                f"5. Confidence score (0-1)\n"
                f"6. Patterns to avoid\n\n"
                f"Reply in JSON format only."
            )

            try:
                response = await asyncio.wait_for(
                    self._llm_generate_fn(system_prompt=prompt, history=[], user_msg="", session_id="reflection_diag", cycle=0),
                    timeout=20.0,
                )
                logger.info("[LLM-DIAG] LLM response received (%d chars)", len(response or ""))

                parsed = extract_json_from_llm(response or "")
                if not parsed or not isinstance(parsed, dict):
                    logger.warning("[LLM-DIAG] Failed to parse valid JSON from LLM response")
                    ms = (time.perf_counter() - t0) * 1000
                    try:
                        await self._tel.record_exit("ReflectionEngine", eid, metrics={"diagnosis_success": False, "error": "parse_failed"}, duration_ms=ms)
                    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, json.JSONDecodeError):
                        pass
                    return {}

                required_keys = {"primary_failure_stage", "root_cause_analysis", "confidence"}
                if not required_keys.issubset(parsed.keys()):
                    missing = required_keys - set(parsed.keys())
                    logger.warning("[LLM-DIAG] LLM JSON missing required keys: %s", missing)

                result = {
                    "primary_failure_stage": parsed.get("primary_failure_stage", ""),
                    "secondary_factors": parsed.get("secondary_factors", []),
                    "root_cause_analysis": parsed.get("root_cause_analysis", ""),
                    "suggested_fixes": parsed.get("suggested_fixes", []),
                    "confidence": float(parsed.get("confidence", 0.0)),
                    "avoid_patterns": parsed.get("avoid_patterns", []),
                }
                logger.info("[LLM-DIAG] Diagnosis complete: stage=%s, confidence=%.2f", result["primary_failure_stage"], result["confidence"])
                ms = (time.perf_counter() - t0) * 1000
                try:
                    await self._tel.record_exit("ReflectionEngine", eid, metrics={"diagnosis_success": True, "confidence": round(result["confidence"], 2)}, duration_ms=ms)
                except (OSError, ValueError, RuntimeError):
                    pass
                return result

            except asyncio.TimeoutError:
                logger.warning("[LLM-DIAG] LLM diagnosis timed out after 20s")
                ms = (time.perf_counter() - t0) * 1000
                try:
                    await self._tel.record_exit("ReflectionEngine", eid, metrics={"diagnosis_success": False, "error": "timeout"}, duration_ms=ms)
                except (OSError, ValueError, RuntimeError):
                    pass
                return {}
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, json.JSONDecodeError):
                logger.warning("[LLM-DIAG] LLM diagnosis failed with exception", exc_info=True)
                ms = (time.perf_counter() - t0) * 1000
                try:
                    await self._tel.record_exit("ReflectionEngine", eid, metrics={"diagnosis_success": False, "error": "exception"}, duration_ms=ms)
                except (OSError, ValueError, RuntimeError):
                    pass
                return {}
        except Exception as e:
            if eid:
                try:
                    await self._tel.record_error("ReflectionEngine", str(e), type(e).__name__)
                except (OSError, ValueError, RuntimeError):
                    pass
            raise

    @algo_log(log_args_to_skip=("self", "brain_result", "trajectory"))
    async def reflect_on_failure(
        self,
        expression: str,
        brain_result: dict | None = None,
        trajectory: dict | None = None,
    ) -> ReflectionResult:
        eid = None
        try:
            eid = await self._tel.record_enter("ReflectionEngine", cycle_id="unknown", expr_id=hash(expression) % 10000)
            t0 = time.perf_counter()
            
            result = ReflectionResult()
            if brain_result is None:
                result.failure_stage = "unknown"
                result.failure_reason = "No BRAIN result available"
                result.confidence = 0.1
                ms = (time.perf_counter() - t0) * 1000
                try:
                    await self._tel.record_exit("ReflectionEngine", eid, metrics={"failure_stage": result.failure_stage, "llm_available": self._llm_generate_fn is not None}, duration_ms=ms)
                except (OSError, ValueError, RuntimeError):
                    pass
                return result

            status = brain_result.get("status", "")
            sharpe = brain_result.get("sharpe", 0) or 0
            fitness = brain_result.get("fitness", 0) or 0
            turnover = brain_result.get("turnover", 0) or 0
            checks = brain_result.get("checks", [])

            if status == "ERROR":
                result.failure_stage = "expression"
                result.failure_reason = "Expression caused BRAIN simulation error"
                result.suggested_fix = "Simplify expression structure, verify FastExpr syntax"
                result.confidence = 0.8
            elif sharpe < 0:
                result.failure_stage = "hypothesis"
                result.failure_reason = f"Negative Sharpe ({sharpe:.2f}): signal direction likely wrong"
                result.suggested_fix = "Reverse signal direction (negate), or change from momentum to mean_reversion"
                result.confidence = 0.7
            elif sharpe > 0 and sharpe < 1.25:
                result.failure_stage = "parameters"
                result.failure_reason = f"Weak Sharpe ({sharpe:.2f}): signal exists but too weak"
                result.suggested_fix = "Try different lookback windows, add volatility normalization, or combine with complementary signal"
                result.confidence = 0.6
            elif sharpe >= 1.25 and fitness < 1.0:
                result.failure_stage = "turnover"
                result.failure_reason = f"Good Sharpe ({sharpe:.2f}) but low Fitness ({fitness:.2f}), Turnover={turnover:.1f}%"
                result.suggested_fix = "Apply ts_decay_linear to reduce turnover, target 15-30% range"
                result.confidence = 0.8
                result.avoid_patterns = ["high frequency signals without smoothing"]
            elif "SELF_CORRELATION" in checks:
                result.failure_stage = "originality"
                result.failure_reason = "Alpha too correlated with existing alphas"
                result.suggested_fix = "Change neutralization group, use different fields, or add interaction terms"
                result.confidence = 0.7
            elif "CONCENTRATED_WEIGHT" in checks:
                result.failure_stage = "weight_concentration"
                result.failure_reason = "Portfolio weight too concentrated"
                result.suggested_fix = "Add rank() transformation, use subindustry neutralization"
                result.confidence = 0.6
            else:
                result.failure_stage = "unknown"
                result.failure_reason = f"Unidentified failure: status={status}, sharpe={sharpe}, fitness={fitness}"
                result.suggested_fix = "Try a completely different approach"
                result.confidence = 0.3

            llm_diag = {}
            if self._llm_generate_fn is not None:
                try:
                    llm_diag = await self._llm_diagnose_failure(expression, brain_result)
                    if llm_diag:
                        result.llm_diagnosis_available = True
                        if llm_diag.get("primary_failure_stage"):
                            result.failure_stage = llm_diag["primary_failure_stage"]
                        if llm_diag.get("root_cause_analysis"):
                            result.root_cause = llm_diag["root_cause_analysis"]
                        if llm_diag.get("secondary_factors"):
                            result.composite_factors = llm_diag["secondary_factors"]
                        if llm_diag.get("suggested_fixes"):
                            fixes = llm_diag["suggested_fixes"]
                            if isinstance(fixes, list) and len(fixes) > 0 and isinstance(fixes[0], dict):
                                top_fix = fixes[0]
                                result.suggested_fix = top_fix.get("action", result.suggested_fix)
                            elif isinstance(fixes, list) and len(fixes) > 0:
                                result.suggested_fix = str(fixes[0])
                        if llm_diag.get("confidence") is not None:
                            result.confidence = max(result.confidence, float(llm_diag["confidence"]))
                        if llm_diag.get("avoid_patterns"):
                            merged_avoid = list(set(result.avoid_patterns + llm_diag["avoid_patterns"]))
                            result.avoid_patterns = merged_avoid
                        logger.info("[LLM-DIAG] Merged LLM diagnosis into reflection result")
                except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, json.JSONDecodeError):
                    logger.warning("[LLM-DIAG] LLM diagnosis integration failed, using rule-based fallback", exc_info=True)

            embedding = None
            if self._embed_fn is not None:
                try:
                    vec = await self._embed_fn(f"{result.failure_stage} {expression}")
                    if isinstance(vec, list):
                        if len(vec) > 0 and isinstance(vec[0], list):
                            embedding = vec[0]
                        else:
                            embedding = vec
                    elif isinstance(vec, np.ndarray):
                        embedding = vec.tolist()
                except (ValueError, TypeError, OSError):
                    logger.warning("ReflectionEngine: embed failed for reflection")

            self._reflections.append({
                "expr": expression[:80],
                "failure_stage": result.failure_stage,
                "failure_reason": result.failure_reason,
                "suggested_fix": result.suggested_fix,
                "confidence": result.confidence,
                "embedding": embedding,
            })
            self._save()
            
            ms = (time.perf_counter() - t0) * 1000
            try:
                await self._tel.record_exit("ReflectionEngine", eid, metrics={"failure_stage": result.failure_stage, "llm_available": self._llm_generate_fn is not None}, duration_ms=ms)
            except (OSError, ValueError, RuntimeError):
                pass
            return result
        except Exception as e:
            if eid:
                try:
                    await self._tel.record_error("ReflectionEngine", str(e), type(e).__name__)
                except (OSError, ValueError, RuntimeError):
                    pass
            raise

    def plan_next_iteration(
        self,
        reflection: ReflectionResult | None = None,
        evolution_db: EvolutionDatabase | None = None,
        feature_map: FeatureMap | None = None,
    ) -> NextPlan:
        plan = NextPlan()

        if feature_map is not None:
            unexplored = feature_map.get_unexplored_directions()
            if unexplored:
                plan.direction = unexplored[0]
                plan.reasoning = f"Direction '{unexplored[0]}' is unexplored in FeatureMap"

        if reflection is not None:
            if reflection.failure_stage == "hypothesis":
                plan.mechanism = "mean_reversion" if plan.mechanism != "mean_reversion" else "momentum"
                plan.strategy = "reverse_signal"
                if not plan.reasoning:
                    plan.reasoning = f"Reflecting on hypothesis failure: {reflection.failure_reason}"
            elif reflection.failure_stage == "turnover":
                plan.mechanism = "normalized"
                plan.strategy = "apply_decay"
                if not plan.reasoning:
                    plan.reasoning = f"Reflecting on turnover issue: {reflection.failure_reason}"
            elif reflection.failure_stage == "originality":
                plan.mechanism = "interaction"
                plan.strategy = "add_interaction"
                if not plan.reasoning:
                    plan.reasoning = f"Reflecting on originality: {reflection.failure_reason}"
            elif reflection.failure_stage == "parameters":
                plan.strategy = "tune_parameters"
                if not plan.reasoning:
                    plan.reasoning = f"Reflecting on weak signal: {reflection.failure_reason}"

        if evolution_db is not None:
            stats = evolution_db.get_stats()
            top_dir = max(stats.get("direction_distribution", {}).items(), key=lambda x: x[1], default=("momentum", 0))
            if not plan.direction:
                plan.direction = top_dir[0]

        if not plan.direction:
            plan.direction = "momentum"
        if not plan.strategy:
            plan.strategy = "explore"

        return plan

    @algo_log(log_args_to_skip=("self",))
    async def self_critique(self, expression: str, hypothesis: str) -> CritiqueResult:
        eid = None
        try:
            eid = await self._tel.record_enter("ReflectionEngine", cycle_id="unknown", expr_id=hash(expression) % 10000)
            t0 = time.perf_counter()

            result = CritiqueResult()
            if self._llm_generate_fn is None:
                result.consistency_score = None
                result.critique_available = False
                result.issues = ["LLM client not available for self-critique"]
                ms = (time.perf_counter() - t0) * 1000
                try:
                    await self._tel.record_exit("ReflectionEngine", eid, metrics={"critique_available": False}, duration_ms=ms)
                except (OSError, ValueError, RuntimeError):
                    pass
                return result

            max_expr_len = 300
            max_hyp_len = 200
            expr_short = expression[:max_expr_len] + ("..." if len(expression) > max_expr_len else "")
            hyp_short = hypothesis[:max_hyp_len] + ("..." if len(hypothesis) > max_hyp_len else "")

            prompt = (
                f"Evaluate consistency between hypothesis and alpha expression.\n"
                f"Hypothesis: {hyp_short}\n"
                f"Expression: {expr_short}\n\n"
                f"Rate 0.0-1.0. Reply ONLY: CONSISTENCY: <score>"
            )
            try:
                response = await asyncio.wait_for(
                    self._llm_generate_fn(system_prompt=prompt, history=[], user_msg="", session_id="reflection", cycle=0),
                    timeout=15.0,
                )
                text = response or ""
                import re as _re
                _consistency_match = _re.search(r'(?:\*\*|\b)CONSISTENCY(?:\*\*)?\s*[:：]\s*([\d.]+)', text)
                if _consistency_match:
                    result.consistency_score = min(1.0, max(0.0, float(_consistency_match.group(1))))
                    result.critique_available = True
                else:
                    for line in text.split("\n"):
                        if "CONSISTENCY" in line.upper():
                            try:
                                score_str = line.split(":")[-1].strip()
                                result.consistency_score = min(1.0, max(0.0, float(score_str)))
                                result.critique_available = True
                            except ValueError:
                                result.consistency_score = None
                                result.critique_available = False
                            break
                    else:
                        result.consistency_score = 0.5
                        result.critique_available = True

                _issues_match = _re.search(r'(?:\*\*|\b)ISSUES(?:\*\*)?\s*[:：]\s*(.+)', text)
                if _issues_match:
                    result.issues = [s.strip() for s in _issues_match.group(1).split(",") if s.strip()]
                _sugg_match = _re.search(r'(?:\*\*|\b)SUGGESTIONS(?:\*\*)?\s*[:：]\s*(.+)', text)
                if _sugg_match:
                    result.suggestions = [s.strip() for s in _sugg_match.group(1).split(",") if s.strip()]
            except asyncio.TimeoutError:
                logger.warning("ReflectionEngine: self_critique timed out after 15s")
                result.consistency_score = None
                result.critique_available = False
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, json.JSONDecodeError):
                logger.warning("ReflectionEngine: self_critique LLM call failed", exc_info=True)
                result.consistency_score = None
                result.critique_available = False

            ms = (time.perf_counter() - t0) * 1000
            try:
                await self._tel.record_exit("ReflectionEngine", eid, metrics={"critique_available": result.critique_available, "score": result.consistency_score}, duration_ms=ms)
            except (OSError, ValueError, RuntimeError):
                pass
            return result
        except Exception as e:
            if eid:
                try:
                    await self._tel.record_error("ReflectionEngine", str(e), type(e).__name__)
                except (OSError, ValueError, RuntimeError):
                    pass
            raise

    def get_recent_reflections(self, n: int = 5) -> list[dict]:
        return self._reflections[-n:]

    def get_failure_patterns(self) -> dict[str, int]:
        patterns: dict[str, int] = {}
        for r in self._reflections:
            stage = r.get("failure_stage", "unknown")
            patterns[stage] = patterns.get(stage, 0) + 1
        return patterns

    async def find_similar_failures(self, query: str, top_k: int = 3) -> list[dict]:
        if self._embed_fn is None:
            return []
        reflections_with_emb = [r for r in self._reflections if r.get("embedding") is not None]
        if not reflections_with_emb:
            return []
        try:
            query_vec = await self._embed_fn(query)
            if isinstance(query_vec, np.ndarray):
                query_vec = query_vec.tolist()
            if isinstance(query_vec, list) and len(query_vec) > 0 and isinstance(query_vec[0], list):
                query_vec = query_vec[0]
            if not query_vec:
                return []
            results = []
            for r in reflections_with_emb:
                emb = r.get("embedding")
                if emb is None:
                    continue
                dot = sum(a * b for a, b in zip(query_vec, emb))
                norm_a = math.sqrt(sum(a * a for a in query_vec))
                norm_b = math.sqrt(sum(b * b for b in emb))
                if norm_a == 0 or norm_b == 0:
                    continue
                sim = dot / (norm_a * norm_b)
                results.append({"reflection": r, "similarity": round(sim, 4)})
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
        except (ValueError, TypeError, OSError, RuntimeError):
            logger.warning("ReflectionEngine: find_similar_failures failed")
            return []
