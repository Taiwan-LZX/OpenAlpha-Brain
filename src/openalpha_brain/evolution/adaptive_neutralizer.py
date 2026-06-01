"""Adaptive Neutralization System — Intelligent neutralization level selection.

Research Foundation:
  1. Ehsani, Harvey & Li (FAJ 2021): "Is Sector-Neutrality in Factor Investing a Mistake?"
     Over-neutralization kills cross-sector signal strength.
  2. FactorMiner (ICLR 2026): Experience Memory + Ralph Loop for learning from past trials.
  3. AlphaBench (ICLR 2026): ICIR metrics for quality assessment.

Architecture:
  - NeutralizationExperienceTracker: records and retrieves historical trial outcomes
  - EhsaniConditionEvaluator: applies the SR_across/SR_within < ρ decision rule
  - AdaptiveNeutralizer: main facade that combines experience + theory + MAB sampling

Safety Constraints:
  - NEVER apply triple neutralization to momentum factors
  - Auto-downgrade when Sharpe < 0.8 after neutralization
  - Mark (category, level) as forbidden when success_rate < 0.3 after 5+ trials
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path

from openalpha_brain.monitoring.algorithm_telemetry import AlgorithmTelemetryCollector
from openalpha_brain.utils.algo_logger import algo_log

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "min_trials_for_decision": 3,
    "ehsani_correlation_estimate": 0.7,
    "success_rate_threshold": 0.5,
    "mab_exploration_weight": 0.3,
    "neutralization_levels": ["none", "industry", "subindustry", "double", "triple"],
    "category_defaults": {
        "momentum": {"default_level": "industry", "max_level": "subindustry"},
        "value": {"default_level": "subindustry", "max_level": "double"},
        "volatility": {"default_level": "industry", "max_level": "subindustry"},
        "quality": {"default_level": "industry", "max_level": "industry"},
        "size": {"default_level": "subindustry", "max_level": "double"},
        "liquidity": {"default_level": "industry", "max_level": "subindustry"},
    },
}

_LEVEL_ORDER = {"none": 0, "industry": 1, "subindustry": 2, "double": 3, "triple": 4}

_FORBIDDEN_PAIRS: set[tuple[str, str]] = set()


@dataclass
class NeutralizationTrial:
    category: str
    neutralization_level: str
    sharpe_before: float
    sharpe_after: float
    fitness_delta: float
    timestamp: float
    expression_id: str
    outcome: str


@dataclass
class NeutralizationExperience:
    category: str
    level: str
    total_trials: int = 0
    successes: int = 0
    avg_sharpe_delta: float = 0.0
    avg_fitness_delta: float = 0.0
    success_rate: float = 0.0
    mab_score: float = 0.5
    last_updated: float = 0.0
    is_forbidden: bool = False


@dataclass
class AdaptiveRecommendation:
    recommended_level: str
    confidence: float
    reasoning: str
    alternative_levels: list[tuple[str, float]] = dc_field(default_factory=list)
    is_forced: bool = False
    mab_adjusted: bool = False


class NeutralizationExperienceTracker:
    def __init__(self) -> None:
        self._experiences: dict[tuple[str, str], NeutralizationExperience] = {}

    @algo_log()
    def record_trial(self, trial: NeutralizationTrial) -> None:
        try:
            key = (trial.category, trial.neutralization_level)
            exp = self._experiences.get(key)
            if exp is None:
                exp = NeutralizationExperience(
                    category=trial.category,
                    level=trial.neutralization_level,
                )
                self._experiences[key] = exp

            exp.total_trials += 1
            is_success = trial.outcome in ("success", "partial")
            if is_success:
                exp.successes += 1

            n = exp.total_trials
            exp.avg_sharpe_delta += (trial.sharpe_after - trial.sharpe_before - exp.avg_sharpe_delta) / n
            exp.avg_fitness_delta += (trial.fitness_delta - exp.avg_fitness_delta) / n
            exp.success_rate = exp.successes / n
            exp.last_updated = trial.timestamp

            alpha = 1.0 + exp.successes
            beta_val = 1.0 + (exp.total_trials - exp.successes)
            exp.mab_score = alpha / (alpha + beta_val)

            if exp.total_trials >= 5 and exp.success_rate < 0.3:
                exp.is_forbidden = True
                _FORBIDDEN_PAIRS.add(key)
                logger.warning(
                    "[ADAPT-NEUT] [DEFENSIVE_LOG] Marking (%s, %s) as FORBIDDEN: success_rate=%.2f after %d trials",
                    trial.category,
                    trial.neutralization_level,
                    exp.success_rate,
                    exp.total_trials,
                )

            logger.info(
                "[ADAPT-NEUT] Recorded trial: cat=%s lvl=%s outcome=%s sharpe_delta=%.3f fitness_delta=%.3f sr=%.2f",
                trial.category,
                trial.neutralization_level,
                trial.outcome,
                trial.sharpe_after - trial.sharpe_before,
                trial.fitness_delta,
                exp.success_rate,
            )
        except (ValueError, TypeError) as e:
            logger.error("[ADAPT-NEUT] Failed to record trial: %s", e)

    @algo_log()
    def get_experience(self, category: str, level: str) -> NeutralizationExperience:
        key = (category, level)
        exp = self._experiences.get(key)
        if exp is None:
            exp = NeutralizationExperience(category=category, level=level)
            self._experiences[key] = exp
        return exp

    @algo_log()
    def get_best_level_for_category(self, category: str, min_trials: int = 3) -> str | None:
        try:
            levels = DEFAULT_CONFIG["neutralization_levels"]
            candidates: list[tuple[str, NeutralizationExperience]] = []
            for lvl in levels:
                key = (category, lvl)
                exp = self._experiences.get(key)
                if exp and exp.total_trials >= min_trials and not exp.is_forbidden:
                    candidates.append((lvl, exp))

            if not candidates:
                return None

            best = max(candidates, key=lambda x: x[1].avg_sharpe_delta)
            logger.info(
                "[ADAPT-NEUT] Best level for %s: %s (avg_sharpe_delta=%.3f)",
                category,
                best[0],
                best[1].avg_sharpe_delta,
            )
            return best[0]
        except (ValueError, KeyError) as e:
            logger.error("[ADAPT-NEUT] get_best_level failed: %s", e)
            return None

    @algo_log()
    def get_success_rate_matrix(self) -> dict[str, dict[str, float]]:
        try:
            matrix: dict[str, dict[str, float]] = {}
            for (cat, lvl), exp in self._experiences.items():
                matrix.setdefault(cat, {})[lvl] = exp.success_rate
            return matrix
        except (ValueError, KeyError) as e:
            logger.error("[ADAPT-NEUT] get_success_rate_matrix failed: %s", e)
            return {}

    @algo_log()
    def save_to_disk(self, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "experiences": [
                    {
                        "category": exp.category,
                        "level": exp.level,
                        "total_trials": exp.total_trials,
                        "successes": exp.successes,
                        "avg_sharpe_delta": exp.avg_sharpe_delta,
                        "avg_fitness_delta": exp.avg_fitness_delta,
                        "success_rate": exp.success_rate,
                        "mab_score": exp.mab_score,
                        "last_updated": exp.last_updated,
                        "is_forbidden": exp.is_forbidden,
                    }
                    for exp in self._experiences.values()
                ],
                "forbidden_pairs": [list(p) for p in _FORBIDDEN_PAIRS],
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info("[ADAPT-NEUT] Saved experience to %s (%d entries)", path, len(data["experiences"]))
        except OSError as e:
            logger.error("[ADAPT-NEUT] Failed to save to disk: %s", e)

    @algo_log()
    def load_from_disk(self, path: Path) -> None:
        try:
            if not path.exists():
                logger.info("[ADAPT-NEUT] No experience file at %s, starting fresh", path)
                return
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            self._experiences.clear()
            for entry in data.get("experiences", []):
                exp = NeutralizationExperience(
                    category=entry["category"],
                    level=entry["level"],
                    total_trials=entry.get("total_trials", 0),
                    successes=entry.get("successes", 0),
                    avg_sharpe_delta=entry.get("avg_sharpe_delta", 0.0),
                    avg_fitness_delta=entry.get("avg_fitness_delta", 0.0),
                    success_rate=entry.get("success_rate", 0.0),
                    mab_score=entry.get("mab_score", 0.5),
                    last_updated=entry.get("last_updated", 0.0),
                    is_forbidden=entry.get("is_forbidden", False),
                )
                self._experiences[(exp.category, exp.level)] = exp

            _FORBIDDEN_PAIRS.clear()
            for pair in data.get("forbidden_pairs", []):
                _FORBIDDEN_PAIRS.add((pair[0], pair[1]))

            logger.info(
                "[ADAPT-NEUT] Loaded experience from %s (%d entries, %d forbidden)",
                path,
                len(self._experiences),
                len(_FORBIDDEN_PAIRS),
            )
        except (OSError, json.JSONDecodeError) as e:
            logger.error("[ADAPT-NEUT] Failed to load from disk: %s", e)


class EhsaniConditionEvaluator:
    def __init__(self, config: dict | None = None) -> None:
        self._config = config or DEFAULT_CONFIG
        self._rho = self._config.get("ehsani_correlation_estimate", 0.7)

    @algo_log()
    def evaluate(self, sharpe_raw: float, sharpe_neutralized: float, correlation: float) -> bool:
        try:
            if sharpe_raw <= 0:
                return False
            sr_ratio = sharpe_neutralized / sharpe_raw if sharpe_raw > 0 else 0
            should_apply = sr_ratio < (correlation or self._rho)

            logger.info(
                "[ADAPT-NEUT] Ehsani eval: raw=%.3f neut=%.3f ratio=%.3f rho=%.3f → %s",
                sharpe_raw,
                sharpe_neutralized,
                sr_ratio,
                correlation or self._rho,
                "APPLY" if should_apply else "SKIP",
            )
            return should_apply
        except (ValueError, TypeError) as e:
            logger.error("[ADAPT-NEUT] Ehsani evaluate failed: %s", e)
            return False

    @algo_log()
    def estimate_sr_ratio(self, category: str, historical_data: list) -> float:
        try:
            if not historical_data:
                defaults = {
                    "momentum": 0.55,
                    "value": 0.75,
                    "volatility": 0.65,
                    "quality": 0.80,
                    "size": 0.70,
                    "liquidity": 0.68,
                }
                return defaults.get(category, 0.7)

            deltas = []
            for item in historical_data:
                sb = item.get("sharpe_before", 0)
                sa = item.get("sharpe_after", 0)
                if sb > 0:
                    deltas.append(sa / sb)

            if not deltas:
                return 0.7
            return sum(deltas) / len(deltas)
        except (ValueError, TypeError, ZeroDivisionError) as e:
            logger.error("[ADAPT-NEUT] estimate_sr_ratio failed: %s", e)
            return 0.7

    @algo_log()
    def recommend_level(
        self,
        category: str,
        current_sharpe: float,
        experience: NeutralizationExperienceTracker,
    ) -> tuple[str, float]:
        try:
            cat_defaults = self._config.get("category_defaults", {})
            default_info = cat_defaults.get(category, {"default_level": "industry", "max_level": "subindustry"})
            default_level = default_info.get("default_level", "industry")
            default_info.get("max_level", "subindustry")

            best_from_exp = experience.get_best_level_for_category(category)
            if best_from_exp:
                best_exp = experience.get_experience(category, best_from_exp)
                confidence = min(1.0, best_exp.total_trials / 10.0) * best_exp.success_rate
                return best_from_exp, confidence

            sr_ratio_est = self.estimate_sr_ratio(category, [])
            confidence = max(0.3, 1.0 - abs(sr_ratio_est - self._rho))

            if current_sharpe < 0.8:
                coarser = self._coarsen(default_level)
                logger.info(
                    "[ADAPT-NEUT] Low Sharpe %.2f → coarsening to %s",
                    current_sharpe,
                    coarser,
                )
                return coarser, confidence * 0.7

            return default_level, confidence
        except (ValueError, KeyError) as e:
            logger.error("[ADAPT-NEUT] recommend_level failed: %s", e)
            return "industry", 0.4

    @staticmethod
    def _coarsen(level: str) -> str:
        order = {
            "triple": "double",
            "double": "subindustry",
            "subindustry": "industry",
            "industry": "none",
            "none": "none",
        }
        return order.get(level, "none")


class AdaptiveNeutralizer:
    def __init__(self, experience_path: Path, config: dict | None = None) -> None:
        self._config = config or DEFAULT_CONFIG
        self._experience_path = Path(experience_path)
        self._tracker = NeutralizationExperienceTracker()
        self._evaluator = EhsaniConditionEvaluator(self._config)
        self._tel = AlgorithmTelemetryCollector.get_instance()
        self._tracker.load_from_disk(self._experience_path)

    @algo_log()
    def analyze_and_recommend(self, expression: str, category: str, wq_metrics: dict) -> AdaptiveRecommendation:
        eid = None
        try:
            eid = self._tel.record_enter_sync(
                "AdaptiveNeutralizer", cycle_id="unknown", expr_id=hash(expression) % 10000
            )
            t0 = time.perf_counter()
            expr_hash = hashlib.md5(expression.encode()).hexdigest()[:12]
            sharpe_raw = wq_metrics.get("sharpe_raw", wq_metrics.get("sharpe", 1.0))
            sharpe_neut = wq_metrics.get("sharpe_neutralized", sharpe_raw * 0.85)
            correlation = wq_metrics.get("correlation", self._config["ehsani_correlation_estimate"])

            cat_defaults = self._config.get("category_defaults", {})
            default_info = cat_defaults.get(category, {"default_level": "industry", "max_level": "subindustry"})
            default_info.get("default_level", "industry")
            max_level = default_info.get("max_level", "subindustry")

            apply_ehsani = self._evaluator.evaluate(sharpe_raw, sharpe_neut, correlation)
            rec_level, base_confidence = self._evaluator.recommend_level(
                category,
                sharpe_neut,
                self._tracker,
            )

            is_forced = False
            reasoning_parts: list[str] = []

            if category == "momentum" and rec_level in ("double", "triple"):
                rec_level = max_level
                is_forced = True
                reasoning_parts.append(
                    "[DEFENSIVE_LOG] Blocked triple/double for momentum (Ehsani: kills cross-sector signal)"
                )
                logger.warning(
                    "[ADAPT-NEUT] [DEFENSIVE_LOG] Overriding %s→%s for momentum factor",
                    "double" if rec_level == "double" else "triple",
                    rec_level,
                )

            if _LEVEL_ORDER.get(rec_level, 0) > _LEVEL_ORDER.get(max_level, 2):
                rec_level = max_level
                is_forced = True
                reasoning_parts.append(f"[DEFENSIVE_LOG] Capped at max_level={max_level}")

            if sharpe_neut < 0.8:
                rec_level = self._evaluator._coarsen(rec_level)
                is_forced = True
                reasoning_parts.append(f"[DEFENSIVE_LOG] Sharpe {sharpe_neut:.2f} < 0.8 → downgraded to {rec_level}")

            key = (category, rec_level)
            if key in _FORBIDDEN_PAIRS:
                fallback = self._find_safe_fallback(category)
                reasoning_parts.append(f"[DEFENSIVE_LOG] ({category}, {rec_level}) is FORBIDDEN → fallback {fallback}")
                rec_level = fallback
                is_forced = True

            weights = self.get_sampling_weights(category)
            mab_adjusted = False
            if weights and rec_level in weights:
                best_mab = max(weights, key=weights.get)
                if best_mab != rec_level and weights.get(best_mab, 0) > weights.get(rec_level, 0) * 1.5:
                    rec_level = best_mab
                    mab_adjusted = True
                    reasoning_parts.append(f"MAB override: {best_mab} (weight={weights[best_mab]:.3f}) beats baseline")

            if not reasoning_parts:
                if apply_ehsani:
                    reasoning_parts.append(
                        f"Ehsani: SR_ratio < ρ ({sharpe_neut / sharpe_raw:.3f} < {correlation:.2f}) → neutralize"
                    )
                else:
                    reasoning_parts.append(
                        f"Ehsani: SR_ratio ≥ ρ ({sharpe_neut / sharpe_raw:.3f} ≥ {correlation:.2f}) → minimal neutralization"  # noqa: E501
                    )
                reasoning_parts.append(f"Experience-based: {base_confidence:.1%} confidence")

            alternatives = sorted(
                [(l, w) for l, w in weights.items() if l != rec_level],
                key=lambda x: -x[1],
            )[:3]

            result = AdaptiveRecommendation(
                recommended_level=rec_level,
                confidence=min(1.0, base_confidence + (0.15 if mab_adjusted else 0)),
                reasoning="; ".join(reasoning_parts),
                alternative_levels=alternatives,
                is_forced=is_forced,
                mab_adjusted=mab_adjusted,
            )

            logger.info(
                "[ADAPT-NEUT] Recommend: expr=%s cat=%s level=%s conf=%.2f forced=%s mab=%s | %s",
                expr_hash[:8],
                category,
                rec_level,
                result.confidence,
                is_forced,
                mab_adjusted,
                result.reasoning,
            )
            ms = (time.perf_counter() - t0) * 1000
            with contextlib.suppress(OSError, ValueError, RuntimeError):
                self._tel.record_exit_sync(
                    "AdaptiveNeutralizer",
                    eid,
                    metrics={"recommended_level": rec_level, "confidence": result.confidence, "is_forced": is_forced},
                    duration_ms=ms,
                )
            return result
        except (ValueError, TypeError, OSError, RuntimeError, KeyError, AttributeError) as e:
            if eid:
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    self._tel.record_error_sync("AdaptiveNeutralizer", str(e), type(e).__name__)
            logger.warning("[ADAPT-NEUT] analyze_and_recommend failed, returning safe fallback: %s", e)
            return AdaptiveRecommendation(
                recommended_level="industry",
                confidence=0.3,
                reasoning=f"Error during analysis: {str(e)}",
                is_forced=True,
            )

    @algo_log()
    def record_outcome(
        self,
        expression: str,
        category: str,
        level: str,
        result_metrics: dict,
    ) -> None:
        eid = None
        try:
            eid = self._tel.record_enter_sync(
                "AdaptiveNeutralizer", cycle_id="unknown", expr_id=hash(expression) % 10000
            )
            t0 = time.perf_counter()
            expr_hash = hashlib.md5(expression.encode()).hexdigest()[:12]
            sharpe_before = result_metrics.get("sharpe_before", 0.0)
            sharpe_after = result_metrics.get("sharpe_after", 0.0)
            fitness_delta = result_metrics.get("fitness_delta", 0.0)

            if sharpe_after < 0.8:
                outcome = "failure"
            elif fitness_delta > 0:
                outcome = "success"
            else:
                outcome = "partial"

            trial = NeutralizationTrial(
                category=category,
                neutralization_level=level,
                sharpe_before=sharpe_before,
                sharpe_after=sharpe_after,
                fitness_delta=fitness_delta,
                timestamp=time.time(),
                expression_id=expr_hash,
                outcome=outcome,
            )
            self._tracker.record_trial(trial)
            self._tracker.save_to_disk(self._experience_path)

            logger.info(
                "[ADAPT-NEUT] Outcome recorded: expr=%s cat=%s lvl=%s → %s (Δsharpe=%.3f Δfitness=%.3f)",
                expr_hash[:8],
                category,
                level,
                outcome,
                sharpe_after - sharpe_before,
                fitness_delta,
            )
            ms = (time.perf_counter() - t0) * 1000
            with contextlib.suppress(OSError, ValueError, RuntimeError):
                self._tel.record_exit_sync("AdaptiveNeutralizer", eid, metrics={"outcome": outcome}, duration_ms=ms)
        except Exception as e:
            if eid:
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    self._tel.record_error_sync("AdaptiveNeutralizer", str(e), type(e).__name__)
            raise

    @algo_log()
    def get_sampling_weights(self, category: str) -> dict[str, float]:
        eid = None
        try:
            eid = self._tel.record_enter_sync("AdaptiveNeutralizer", cycle_id="unknown", expr_id=hash(category) % 10000)
            t0 = time.perf_counter()
            levels = self._config.get("neutralization_levels", DEFAULT_CONFIG["neutralization_levels"])
            cat_defaults = self._config.get("category_defaults", {})
            default_info = cat_defaults.get(category, {"default_level": "industry", "max_level": "subindustry"})
            max_level = default_info.get("max_level", "subindustry")
            exploration = self._config.get("mab_exploration_weight", 0.3)

            weights: dict[str, float] = {}
            max_order = _LEVEL_ORDER.get(max_level, 2)

            for lvl in levels:
                if _LEVEL_ORDER.get(lvl, 0) > max_order:
                    continue
                if (category, lvl) in _FORBIDDEN_PAIRS:
                    continue
                if category == "momentum" and lvl in ("double", "triple"):
                    continue

                exp = self._tracker.get_experience(category, lvl)
                base_weight = exploration / len(levels)
                exp_weight = (1.0 - exploration) * exp.mab_score
                weights[lvl] = base_weight + exp_weight

            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}

            logger.debug(
                "[ADAPT-NEUT] Sampling weights for %s: %s",
                category,
                {k: round(v, 3) for k, v in weights.items()},
            )
            ms = (time.perf_counter() - t0) * 1000
            with contextlib.suppress(OSError, ValueError, RuntimeError):
                self._tel.record_exit_sync(
                    "AdaptiveNeutralizer", eid, metrics={"weights_count": len(weights)}, duration_ms=ms
                )
            return weights
        except Exception as e:
            if eid:
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    self._tel.record_error_sync("AdaptiveNeutralizer", str(e), type(e).__name__)
            raise

    @algo_log()
    def should_upgrade_neutralization(self, category: str, current_level: str, sharpe: float) -> bool:
        try:
            if category == "momentum" and current_level in ("subindustry", "double"):
                logger.info("[ADAPT-NEUT] Upgrade blocked for momentum at %s", current_level)
                return False

            cat_defaults = self._config.get("category_defaults", {})
            max_level = cat_defaults.get(category, {}).get("max_level", "subindustry")

            if _LEVEL_ORDER.get(current_level, 0) >= _LEVEL_ORDER.get(max_level, 2):
                return False

            next_level = self._next_finer(current_level)
            if next_level is None:
                return False

            next_key = (category, next_level)
            if next_key in _FORBIDDEN_PAIRS:
                logger.info("[ADAPT-NEUT] Upgrade to %s blocked (FORBIDDEN)", next_level)
                return False

            exp = self._tracker.get_experience(category, current_level)
            min_trials = self._config.get("min_trials_for_decision", 3)

            if exp.total_trials >= min_trials and exp.success_rate > self._config.get("success_rate_threshold", 0.5):
                if sharpe >= 1.0:
                    logger.info(
                        "[ADAPT-NEUT] Upgrade recommended: %s %s→%s (sr=%.2f sharpe=%.2f)",
                        category,
                        current_level,
                        next_level,
                        exp.success_rate,
                        sharpe,
                    )
                    return True

            return False
        except (ValueError, TypeError) as e:
            logger.error("[ADAPT-NEUT] should_upgrade failed: %s", e)
            return False

    def _find_safe_fallback(self, category: str) -> str:
        cat_defaults = self._config.get("category_defaults", {})
        default_info = cat_defaults.get(category, {"default_level": "industry", "max_level": "subindustry"})
        preferred = ["industry", "none", "subindustry"]
        for lvl in preferred:
            if (category, lvl) not in _FORBIDDEN_PAIRS:
                return lvl
        return default_info.get("default_level", "industry")

    @staticmethod
    def _next_finer(current: str) -> str | None:
        finer_map = {
            "none": "industry",
            "industry": "subindustry",
            "subindustry": "double",
            "double": "triple",
            "triple": None,
        }
        return finer_map.get(current)


def create_adaptive_neutralizer(experience_path: Path, config: dict | None = None) -> AdaptiveNeutralizer:
    """Factory function for creating an AdaptiveNeutralizer instance."""
    return AdaptiveNeutralizer(experience_path=experience_path, config=config)
