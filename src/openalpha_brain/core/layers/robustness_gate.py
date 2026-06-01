"""
OpenAlpha-Brain RobustnessGate — Layer 5 of 6-Layer Architecture
=================================================================
Final quality gate before an alpha is accepted into the factor pool.
Inspired by Claude Code Quant's "False Discovery Gauntlet" — 5 statistical
robustness checks ensuring only statistically valid alphas survive.

职责:
  1. Anti-Overfit Gauntlet — multi-check overfit detection via AntiOverfitDetector
  2. Decay Pre-Screen — initial decay risk assessment via AlphaDecayDetector
  3. Correlation Gate — self-correlation with existing pool (redundancy prevention)
  4. Sharpe Threshold Gate — minimum quality floor on Sharpe ratio
  5. Fitness Threshold Gate — minimum quality floor on fitness score
  6. Composite Scoring — weighted multi-metric verdict synthesis

架构位置 (Layer 5):
  ┌──────────────────────┐     ┌──────────────────┐     ┌─────────────────┐
  │ImprovementOrchestra   │ →   │ RobustnessGate    │ →   │ Factor Pool     │
  │(Layer 4)             │     │(本模块 - Layer 5) │     │(Accept/Reject)  │
  └──────────────────────┘     └──────────────────┘     └─────────────────┘
                                      │
                              ┌───────┴──────────┐
                              ▼                  ▼
                   AntiOverfitDetector    AlphaDecayDetector
                   Correlation Engine     Threshold Gates

Gauntlet Tests (Claude Code Quant inspired):
  1. Permutation Test      — shuffle labels, verify signal is real (via AntiOverfitDetector)
  2. Deflated Sharpe Ratio — Bailey & Lopez de Prado 2014 (via AntiOverfitDetector placebo)
  3. Subsample Stability   — IC consistency across market regimes (via AntiOverfitDetector)
  4. Temporal Decay Test   — performance degradation over time (via AlphaDecayDetector)
  5. Cross-validation Consistency — multi-metric threshold agreement

Usage:
    gate = RobustnessGate(config={"min_sharpe": 0.5, "max_correlation": 0.95})
    result = await gate.evaluate(
        expression="ts_decay_linear(rank(close/volume), sector), 10)",
        brain_result=eval_result,
        session_id="sess_001",
        anti_overfit_detector=lightweight_detector,
        decay_detector=decay_detector,
        existing_alphas=current_pool_expressions,
    )
    if RobustnessGate.should_reject(result):
        logger.warning("Alpha rejected: %s", result.rejection_reasons)
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

_EPSILON = 1e-6


class RobustnessVerdict(str, Enum):  # noqa: UP042
    """Robustness gate verdict classification.

    ROBUST:    Passes all critical checks — safe to accept into factor pool
    MARGINAL:  Passes with warnings — accept but monitor closely
    UNSTABLE:  Fails some non-critical checks — conditional acceptance
    REJECTED:  Fails critical checks — must not enter factor pool
    """

    ROBUST = "robust"
    MARGINAL = "marginal"
    UNSTABLE = "unstable"
    REJECTED = "rejected"


@dataclass
class RobustnessCheckResult:
    """Complete result of the robustness gate evaluation.

    Attributes:
        verdict: Final verdict classification
        overall_score: Composite score 0.0–1.0 across all gauntlets
        anti_overfit_score: Anti-overfit detector score 0–100 (None if skipped)
        decay_score: Decay pre-screen score 0.0–1.0 (None if skipped)
        correlation_with_pool: Max absolute correlation with pool (None if no pool)
        sharpe_threshold_pass: Whether Sharpe meets minimum threshold
        fitness_threshold_pass: Whether fitness meets minimum threshold
        warnings: Non-fatal warning messages
        rejection_reasons: Fatal rejection reasons (non-empty => REJECTED)
        metadata: Extended diagnostic metadata dict
    """

    verdict: RobustnessVerdict
    overall_score: float
    anti_overfit_score: float | None = None
    decay_score: float | None = None
    correlation_with_pool: float | None = None
    sharpe_threshold_pass: bool = True
    fitness_threshold_pass: bool = True
    warnings: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "overall_score": round(self.overall_score, 4),
            "anti_overfit_score": self.anti_overfit_score,
            "decay_score": self.decay_score,
            "correlation_with_pool": round(self.correlation_with_pool, 4)
            if self.correlation_with_pool is not None
            else None,
            "sharpe_threshold_pass": self.sharpe_threshold_pass,
            "fitness_threshold_pass": self.fitness_threshold_pass,
            "warnings": self.warnings,
            "rejection_reasons": self.rejection_reasons,
            "metadata": self.metadata,
        }


class RobustnessGate:
    """Statistical robustness gate — Layer 5 of 6-Layer Architecture.

    Implements the "False Discovery Gauntlet" pattern: 5 independent statistical
    checks that must be passed before an alpha enters the factor pool. Inspired by
    Bailey & Lopez de Prado (2014) deflated Sharpe methodology and Claude Code
    Quant's multi-test validation framework.

    Design Principles:
    - Defense in depth: each check is independent; failure of one does not block others
    - Graceful degradation: missing dependencies produce warnings, not crashes
    - Weighted scoring: composite score enables fine-grained acceptance decisions
    - Audit trail: every decision carries full diagnostic context

    Config Parameters (passed via __init__ or evaluate()):
        min_sharpe:       Minimum Sharpe ratio to pass threshold gate (default 0.5)
        min_fitness:      Minimum fitness to pass threshold gate (default 0.3)
        max_correlation:  Maximum allowed correlation with existing pool (default 0.95)
        anti_overfit_weight: Weight for anti-overfit score in composite (default 0.30)
        decay_weight:     Weight for decay score in composite (default 0.20)
        correlation_weight: Weight for correlation penalty in composite (default 0.15)
        threshold_weight: Weight for Sharpe/Fitness thresholds in composite (default 0.35)
        marginal_score_floor:  Score below which verdict cannot be MARGINAL (default 0.50)
        unstable_score_floor: Score below which verdict becomes REJECTED (default 0.30)

    Gauntlet Weights (default):
        Anti-Overfit:  30%  — most important: catches data leakage and overfitting
        Decay Screen:   20%  — prevents accepting already-decaying factors
        Correlation:    15%  — ensures diversity in the factor pool
        Thresholds:     35%  — quality floor enforcement
    """

    DEFAULT_CONFIG: dict[str, Any] = {
        "min_sharpe": 0.5,
        "min_fitness": 0.3,
        "max_correlation": 0.95,
        "anti_overfit_weight": 0.30,
        "decay_weight": 0.20,
        "correlation_weight": 0.15,
        "threshold_weight": 0.35,
        "marginal_score_floor": 0.50,
        "unstable_score_floor": 0.30,
    }

    CRITICAL_REJECTION_TRIGGERS = {
        "anti_overfit_critical",
        "sharpe_below_minimum",
        "fitness_below_minimum",
        "correlation_too_high",
    }

    def __init__(self, config: dict | None = None):
        self.config: dict[str, Any] = {**self.DEFAULT_CONFIG, **(config or {})}

    async def evaluate(
        self,
        expression: str,
        brain_result: Any,
        improvement_result: Any = None,
        session_id: str = "",
        *,
        anti_overfit_detector: Any = None,
        decay_detector: Any = None,
        existing_alphas: list[Any] | None = None,
        min_sharpe: float = 0.5,
        min_fitness: float = 0.3,
        max_correlation: float = 0.95,
    ) -> RobustnessCheckResult:
        """Run the full 5-check robustness gauntlet on a candidate alpha.

        Pipeline:
          1. Extract metrics from brain_result (Sharpe, fitness, turnover, etc.)
          2. Run Anti-Overfit Detector (if provided)
          3. Run Decay Pre-Screen (if detector + brain_result.alpha_id available)
          4. Compute Correlation with Existing Pool (if pool provided)
          5. Apply Sharpe / Fitness Threshold Gates
          6. Synthesize composite score and verdict

        Args:
            expression: ThreeBlockTemplate alpha expression string
            brain_result: EvaluationResult from Layer 3 EvaluationGateway.
                         Must have attributes: sharpe, fitness, turnover, status,
                         and optionally alpha_id, real_sharpe, real_fitness.
            improvement_result: Optional ImprovementResult from Layer 4.
                               Used for contextual metadata only.
            session_id: Session identifier for logging traceability.
            anti_overfit_detector: LightweightAntiOverfitDetector or
                                  FullAntiOverfitDetector instance.
                                  If None, anti-overfit check is skipped with warning.
            decay_detector: AlphaDecayDetector instance.
                           If None, decay pre-screen is skipped.
            existing_alphas: List of existing alpha expressions/objects in the pool.
                            If None or empty, correlation check is skipped.
            min_sharpe: Minimum acceptable Sharpe ratio (overrides config).
            min_fitness: Minimum acceptable fitness score (overrides config).
            max_correlation: Maximum allowed abs(correlation) with pool (overrides config).

        Returns:
            RobustnessCheckResult with complete diagnostic information.
        """
        _sid = session_id or "anon"
        _expr_preview = expression[:60] + ("..." if len(expression) > 60 else "")

        logger.info(
            "[ROBUST-GATE] ◆ START | session=%s expr=%s",
            _sid,
            _expr_preview,
        )

        metrics = self._extract_metrics(brain_result)
        warnings: list[str] = []
        rejection_reasons: list[str] = []
        metadata: dict[str, Any] = {
            "session_id": _sid,
            "expression_preview": _expr_preview,
            "metrics_extracted": metrics is not None,
        }

        ao_score: float | None = None
        decay_score: float | None = None
        corr_value: float | None = None
        sharpe_pass = True
        fitness_pass = True

        _ao_w = self.config["anti_overfit_weight"]
        _decay_w = self.config["decay_weight"]
        _corr_w = self.config["correlation_weight"]
        _thresh_w = self.config["threshold_weight"]

        ao_score, ao_warnings, ao_rejections, ao_meta = await self._run_anti_overfit_gauntlet(
            anti_overfit_detector,
            metrics,
            _sid,
        )
        warnings.extend(ao_warnings)
        rejection_reasons.extend(ao_rejections)
        metadata["anti_overfit"] = ao_meta

        decay_score, decay_warnings, decay_meta = await self._run_decay_prescreen(
            decay_detector,
            brain_result,
            metrics,
            _sid,
        )
        warnings.extend(decay_warnings)
        metadata["decay"] = decay_meta

        corr_value, corr_warnings, corr_rejections = self._run_correlation_check(
            expression,
            existing_alphas,
            max_correlation,
            _sid,
        )
        warnings.extend(corr_warnings)
        rejection_reasons.extend(corr_rejections)
        metadata["correlation"] = {"max_correlation": corr_value}

        sharpe_pass, sharpe_warnings, sharpe_rejections = self._run_sharpe_gate(
            metrics,
            min_sharpe,
            _sid,
        )
        warnings.extend(sharpe_warnings)
        rejection_reasons.extend(sharpe_rejections)

        fitness_pass, fitness_warnings, fitness_rejections = self._run_fitness_gate(
            metrics,
            min_fitness,
            _sid,
        )
        warnings.extend(fitness_warnings)
        rejection_reasons.extend(fitness_rejections)

        overall_score = self._compute_composite_score(
            ao_score=ao_score,
            decay_score=decay_score,
            corr_value=corr_value,
            max_corr=max_correlation,
            sharpe_pass=sharpe_pass,
            fitness_pass=fitness_pass,
            ao_w=_ao_w,
            decay_w=_decay_w,
            corr_w=_corr_w,
            thresh_w=_thresh_w,
        )

        verdict = self._determine_verdict(
            overall_score=overall_score,
            rejection_reasons=rejection_reasons,
            sharpe_pass=sharpe_pass,
            fitness_pass=fitness_pass,
        )

        result = RobustnessCheckResult(
            verdict=verdict,
            overall_score=overall_score,
            anti_overfit_score=ao_score,
            decay_score=decay_score,
            correlation_with_pool=corr_value,
            sharpe_threshold_pass=sharpe_pass,
            fitness_threshold_pass=fitness_pass,
            warnings=warnings,
            rejection_reasons=rejection_reasons,
            metadata=metadata,
        )

        _emoji = {
            RobustnessVerdict.ROBUST: "✓",
            RobustnessVerdict.MARGINAL: "⚠",
            RobustnessVerdict.UNSTABLE: "△",
            RobustnessVerdict.REJECTED: "✗",
        }
        logger.info(
            "[ROBUST-GATE] %s COMPLETE | session=%s verdict=%s score=%.3f "
            "ao=%.0f decay=%.3f corr=%.3f sharpe_ok=%s fit_ok=%s warns=%d rejects=%d",
            _emoji.get(verdict, "?"),
            _sid,
            verdict.value,
            overall_score,
            ao_score or 0,
            decay_score or 0,
            corr_value or 0,
            sharpe_pass,
            fitness_pass,
            len(warnings),
            len(rejection_reasons),
        )

        if rejection_reasons:
            logger.warning(
                "[ROBUST-GATE] ✗ REJECTION REASONS | session=%s reasons=%s",
                _sid,
                rejection_reasons,
            )

        return result

    @staticmethod
    def should_reject(result: RobustnessCheckResult) -> bool:
        """Quick check: should this alpha be rejected from the factor pool?

        Returns True if the verdict is REJECTED or if there are any critical
        rejection reasons present.

        Args:
            result: RobustnessCheckResult from evaluate()

        Returns:
            bool: True if alpha should be rejected
        """
        return result.verdict == RobustnessVerdict.REJECTED or len(result.rejection_reasons) > 0

    def _extract_metrics(self, brain_result: Any) -> dict[str, Any] | None:
        """Extract standard metrics dict from brain_result for downstream checks.

        Supports both EvaluationResult (Layer 3) and raw BrainSubmissionResult objects.
        """
        if brain_result is None:
            return None

        metrics: dict[str, Any] = {}

        sharpe = getattr(brain_result, "sharpe", None)
        if sharpe is None:
            sharpe = getattr(brain_result, "real_sharpe", None)
        if sharpe is not None:
            metrics["sharpe"] = float(sharpe)

        fitness = getattr(brain_result, "fitness", None)
        if fitness is None:
            fitness = getattr(brain_result, "real_fitness", None)
        if fitness is not None:
            metrics["fitness"] = float(fitness)

        turnover = getattr(brain_result, "turnover", None)
        if turnover is None:
            turnover = getattr(brain_result, "real_turnover", None)
        if turnover is not None:
            metrics["turnover"] = float(turnover)

        drawdown = getattr(brain_result, "drawdown", None)
        if drawdown is not None:
            metrics["drawdown"] = float(drawdown)

        returns_val = getattr(brain_result, "returns", None)
        if returns_val is not None:
            metrics["returns"] = float(returns_val)

        parsed = getattr(brain_result, "parsed_result", None)
        if parsed is not None:
            checks = getattr(parsed, "checks", None)
            if checks is not None:
                metrics["checks"] = checks

        status = getattr(brain_result, "status", None)
        if status is not None:
            metrics["status"] = status.value if hasattr(status, "value") else str(status)

        return metrics if metrics else None

    async def _run_anti_overfit_gauntlet(
        self,
        detector: Any,
        metrics: dict[str, Any] | None,
        session_id: str,
    ) -> tuple[float | None, list[str], list[str], dict]:
        """Gauntlet 1+2+3: Anti-Overfit Detection (Permutation + Deflated SR + Subsample).

        Wraps both LightweightAntiOverfitDetector and FullAntiOverfitDetector.
        Returns normalized score 0–100, warnings, rejections, and metadata.
        """
        if detector is None:
            logger.debug("[ROBUST-GATE] No anti_overfit_detector provided, skipping gauntlet 1-3")
            return None, ["Anti-overfit detector not available"], [], {"skipped": True}

        if metrics is None:
            logger.warning("[ROBUST-GATE] Cannot extract metrics for anti-overfit check")
            return None, ["No metrics available for anti-overfit check"], [], {"skipped": True, "reason": "no_metrics"}

        try:
            if hasattr(detector, "evaluate"):
                ao_result = detector.evaluate(metrics)
            elif hasattr(detector, "run_all"):
                ao_result = detector.run_all()
            else:
                logger.warning("[ROBUST-GATE] Unknown detector type: %s", type(detector).__name__)
                return (
                    None,
                    [f"Unknown detector type: {type(detector).__name__}"],
                    [],
                    {"skipped": True, "reason": "unknown_type"},
                )

            score = getattr(ao_result, "score", 0.0)
            recommendation = getattr(ao_result, "recommendation", "")
            tests = getattr(ao_result, "tests", [])
            passed_count = getattr(ao_result, "passed_count", 0)
            total_count = getattr(ao_result, "total_count", 0)

            _warnings: list[str] = []
            _rejections: list[str] = []

            failed_tests = [t.name for t in tests if not t.passed]
            if failed_tests:
                _warnings.append(f"Anti-overfit failures: {', '.join(failed_tests)}")

            if score < 40.0:
                _rejections.append("anti_overfit_critical")
                logger.warning(
                    "[ROBUST-GATE] ANTI-OVERFIT FAIL | session=%s score=%.0f rec=%s tests=%d/%d",
                    session_id,
                    score,
                    recommendation,
                    passed_count,
                    total_count,
                )
            elif score < 60.0:
                _warnings.append(f"Anti-overfit score low ({score:.0f}/100): {recommendation}")
                logger.info(
                    "[ROBUST-GATE] ANTI-OVERFIT WARNING | session=%s score=%.0f rec=%s",
                    session_id,
                    score,
                    recommendation,
                )
            else:
                logger.info(
                    "[ROBUST-GATE] ANTI-OVERFIT PASS | session=%s score=%.0f rec=%s tests=%d/%d",
                    session_id,
                    score,
                    recommendation,
                    passed_count,
                    total_count,
                )

            _meta = {
                "score": score,
                "recommendation": recommendation,
                "passed_count": passed_count,
                "total_count": total_count,
                "failed_tests": failed_tests,
            }

            return score, _warnings, _rejections, _meta

        except (TypeError, AttributeError, ValueError, RuntimeError) as exc:
            logger.warning("[ROBUST-GATE] Anti-overfit gauntlet error: %s", exc)
            return None, [f"Anti-overfit check error: {exc}"], [], {"error": str(exc)}

    async def _run_decay_prescreen(
        self,
        detector: Any,
        brain_result: Any,
        metrics: dict[str, Any] | None,
        session_id: str,
    ) -> tuple[float | None, list[str], dict]:
        """Gauntlet 4: Temporal Decay Pre-Screen.

        Registers the alpha with AlphaDecayDetector and performs an initial
        decay assessment based on available metrics. Does NOT perform a full
        time-series decay check (that happens asynchronously post-acceptance).

        Returns score 0.0–1.0 where higher = less decay risk.
        """
        if detector is None:
            logger.debug("[ROBUST-GATE] No decay_detector provided, skipping gauntlet 4")
            return None, ["Decay detector not available"], {"skipped": True}

        _warnings: list[str] = []
        _meta: dict[str, Any] = {"skipped": False}

        alpha_id = None
        try:
            alpha_id = getattr(brain_result, "alpha_id", None)
            if alpha_id is None:
                br = getattr(brain_result, "brain_result", None)
                if br is not None:
                    alpha_id = getattr(br, "alpha_id", None)
        except (AttributeError, TypeError):
            alpha_id = None

        sharpe_val = 0.0
        if metrics and "sharpe" in metrics:
            sharpe_val = float(metrics["sharpe"])

        fingerprint = None
        with contextlib.suppress(AttributeError, TypeError):
            fingerprint = getattr(brain_result, "fingerprint", None)

        direction = ""
        with contextlib.suppress(AttributeError, TypeError):
            direction = getattr(brain_result, "direction", "") or ""

        if alpha_id and hasattr(detector, "register_alpha"):
            try:
                await detector.register_alpha(
                    alpha_id=alpha_id,
                    expression=getattr(brain_result, "expression", ""),
                    fingerprint=fingerprint,
                    direction=direction,
                    initial_sharpe=sharpe_val,
                )
                logger.debug("[ROBUST-GATE] DECAY: registered alpha %s for tracking", alpha_id)
                _meta["registered"] = True
                _meta["alpha_id"] = alpha_id
            except (TypeError, AttributeError, RuntimeError) as exc:
                _warnings.append(f"Decay detector registration failed: {exc}")
                logger.warning("[ROBUST-GATE] DECAY register failed: %s", exc)
                _meta["register_error"] = str(exc)

        if sharpe_val > 0 and metrics:
            drawdown = metrics.get("drawdown")
            dd_val = float(drawdown) if drawdown is not None else None

            is_os_ratio = metrics.get("is_os_ratio")
            ratio_val = float(is_os_ratio) if is_os_ratio is not None else 1.0

            decay_risk = 0.0
            risk_factors = 0

            if dd_val is not None and dd_val > 0.20:
                decay_risk += 0.25
                risk_factors += 1
                _warnings.append(f"High drawdown ({dd_val:.1%}) indicates potential decay risk")

            if ratio_val is not None and ratio_val < 0.5:
                decay_risk += 0.30
                risk_factors += 1
                _warnings.append(f"Low IS/OS ratio ({ratio_val:.2f}) suggests overfitting decay")

            if sharpe_val > 3.0:
                decay_risk += 0.15
                risk_factors += 1
                _warnings.append("Very high Sharpe may be unsustainable (mean-reversion risk)")

            turnover = metrics.get("turnover")
            if turnover is not None:
                to_val = float(turnover)
                to_pct = to_val if to_val > 1.0 else to_val * 100
                if to_pct > 60.0:
                    decay_risk += 0.15
                    risk_factors += 1
                    _warnings.append(f"High turnover ({to_pct:.1f}%) accelerates decay")

            score = max(0.0, 1.0 - decay_risk)
            _meta["decay_risk_score"] = round(decay_risk, 3)
            _meta["risk_factor_count"] = risk_factors
            _meta["initial_sharpe"] = round(sharpe_val, 4)

            logger.info(
                "[ROBUST-GATE] DECAY PRESCREEN | session=%s score=%.3f risk=%.3f factors=%d",
                session_id,
                score,
                decay_risk,
                risk_factors,
            )
            return score, _warnings, _meta

        logger.debug("[ROBUST-GATE] DECAY: no sharpe data for prescreen, skipping")
        return None, _warnings, _meta

    def _run_correlation_check(
        self,
        expression: str,
        existing_alphas: list[Any] | None,
        max_correlation: float,
        session_id: str,
    ) -> tuple[float | None, list[str], list[str]]:
        """Gauntlet 5: Correlation Check with Existing Factor Pool.

        Uses structural similarity heuristic when raw returns are unavailable:
          - Expression token overlap (Jaccard similarity on operator/field tokens)
          - Topology fingerprint comparison (if available)

        Returns max correlation found (or None), warnings, and rejections.
        """
        if not existing_alphas:
            logger.debug("[ROBUST-GATE] No existing alphas pool, skipping correlation check")
            return None, [], []

        _warnings: list[str] = []
        _rejections: list[str] = []
        max_corr_found = 0.0

        expr_tokens = self._tokenize_expression(expression)
        if not expr_tokens:
            logger.debug("[ROBUST-GATE] Cannot tokenize expression for correlation check")
            return None, ["Expression tokenization failed"], []

        for existing in existing_alphas:
            existing_str = ""
            if isinstance(existing, str):
                existing_str = existing
            elif hasattr(existing, "expression"):
                existing_str = existing.expression
            else:
                existing_str = str(existing)

            existing_tokens = self._tokenize_expression(existing_str)
            if not existing_tokens:
                continue

            corr = self._jaccard_similarity(expr_tokens, existing_tokens)
            if corr > max_corr_found:
                max_corr_found = corr

        if max_corr_found > max_correlation:
            _rejections.append("correlation_too_high")
            _warnings.append(f"Max pool correlation {max_corr_found:.3f} exceeds threshold {max_correlation:.3f}")
            logger.warning(
                "[ROBUST-GATE] CORRELATION REJECT | session=%s max_corr=%.3f > %.3f",
                session_id,
                max_corr_found,
                max_correlation,
            )
        elif max_corr_found > max_correlation * 0.85:
            _warnings.append(f"High pool correlation {max_corr_found:.3f} (threshold={max_correlation:.3f})")
            logger.info(
                "[ROBUST-GATE] CORRELATION WARNING | session=%s max_corr=%.3f",
                session_id,
                max_corr_found,
            )
        else:
            logger.debug(
                "[ROBUST-GATE] CORRELATION OK | session=%s max_corr=%.3f <= %.3f",
                session_id,
                max_corr_found,
                max_correlation,
            )

        return round(max_corr_found, 4), _warnings, _rejections

    def _run_sharpe_gate(
        self,
        metrics: dict[str, Any] | None,
        min_sharpe: float,
        session_id: str,
    ) -> tuple[bool, list[str], list[str]]:
        """Threshold Gate A: Sharpe ratio minimum quality check."""
        if metrics is None or "sharpe" not in metrics:
            logger.warning("[ROBUST-GATE] No Sharpe data for threshold check")
            return False, ["No Sharpe data available"], ["sharpe_below_minimum"]

        sharpe = float(metrics["sharpe"])
        passed = sharpe >= min_sharpe - _EPSILON
        _warnings: list[str] = []
        _rejections: list[str] = []

        if not passed:
            _rejections.append("sharpe_below_minimum")
            _warnings.append(f"Sharpe {sharpe:.4f} below minimum {min_sharpe}")
            logger.warning(
                "[ROBUST-GATE] SHARPE FAIL | session=%s sharpe=%.4f < %.4f",
                session_id,
                sharpe,
                min_sharpe,
            )
        else:
            logger.debug(
                "[ROBUST-GATE] SHARPE PASS | session=%s sharpe=%.4f >= %.4f",
                session_id,
                sharpe,
                min_sharpe,
            )

        return passed, _warnings, _rejections

    def _run_fitness_gate(
        self,
        metrics: dict[str, Any] | None,
        min_fitness: float,
        session_id: str,
    ) -> tuple[bool, list[str], list[str]]:
        """Threshold Gate B: Fitness score minimum quality check."""
        if metrics is None or "fitness" not in metrics:
            logger.debug("[ROBUST-GATE] No fitness data, assuming pass")
            return True, [], []

        fitness = float(metrics["fitness"])
        passed = fitness >= min_fitness - _EPSILON
        _warnings: list[str] = []
        _rejections: list[str] = []

        if not passed:
            _rejections.append("fitness_below_minimum")
            _warnings.append(f"Fitness {fitness:.4f} below minimum {min_fitness}")
            logger.warning(
                "[ROBUST-GATE] FITNESS FAIL | session=%s fitness=%.4f < %.4f",
                session_id,
                fitness,
                min_fitness,
            )
        else:
            logger.debug(
                "[ROBUST-GATE] FITNESS PASS | session=%s fitness=%.4f >= %.4f",
                session_id,
                fitness,
                min_fitness,
            )

        return passed, _warnings, _rejections

    def _compute_composite_score(
        self,
        ao_score: float | None,
        decay_score: float | None,
        corr_value: float | None,
        max_correlation: float,
        sharpe_pass: bool,
        fitness_pass: bool,
        ao_w: float,
        decay_w: float,
        corr_w: float,
        thresh_w: float,
    ) -> float:
        """Compute weighted composite robustness score 0.0–1.0.

        Each component contributes proportionally based on configured weights.
        Missing components (None) have their weight redistributed to others.
        """
        components: list[tuple[float, float]] = []
        total_allocated_w = 0.0

        if ao_score is not None:
            norm_ao = min(1.0, max(0.0, ao_score / 100.0))
            components.append((norm_ao, ao_w))
            total_allocated_w += ao_w

        if decay_score is not None:
            norm_decay = min(1.0, max(0.0, decay_score))
            components.append((norm_decay, decay_w))
            total_allocated_w += decay_w

        if corr_value is not None:
            corr_penalty = min(1.0, corr_value / max(max_correlation, _EPSILON))
            norm_corr = max(0.0, 1.0 - corr_penalty)
            components.append((norm_corr, corr_w))
            total_allocated_w += corr_w

        thresh_score = 1.0
        if not sharpe_pass:
            thresh_score -= 0.6
        if not fitness_pass:
            thresh_score -= 0.4
        thresh_score = max(0.0, thresh_score)
        components.append((thresh_score, thresh_w))
        total_allocated_w += thresh_w

        if total_allocated_w < _EPSILON:
            return 0.5

        normalized_components = [(score, weight / total_allocated_w) for score, weight in components]

        composite = sum(score * w for score, w in normalized_components)
        return round(max(0.0, min(1.0, composite)), 4)

    def _determine_verdict(
        self,
        overall_score: float,
        rejection_reasons: list[str],
        sharpe_pass: bool,
        fitness_pass: bool,
    ) -> RobustnessVerdict:
        """Map composite score and rejection reasons to final verdict.

        Rules:
          - Any critical rejection reason → REJECTED (regardless of score)
          - Both sharpe AND fitness fail → REJECTED
          - Score < unstable_floor → REJECTED
          - Score < marginal_floor → UNSTABLE
          - Score >= marginal_floor with warnings → MARGINAL
          - Score >= marginal_floor without warnings → ROBUST
        """
        has_critical = any(r in self.CRITICAL_REJECTION_TRIGGERS for r in rejection_reasons)

        if has_critical:
            return RobustnessVerdict.REJECTED

        if not sharpe_pass and not fitness_pass:
            return RobustnessVerdict.REJECTED

        unstable_floor = self.config.get("unstable_score_floor", 0.30)
        if overall_score < unstable_floor - _EPSILON:
            return RobustnessVerdict.REJECTED

        marginal_floor = self.config.get("marginal_score_floor", 0.50)
        if overall_score < marginal_floor - _EPSILON:
            return RobustnessVerdict.UNSTABLE

        if not sharpe_pass or not fitness_pass:
            return RobustnessVerdict.MARGINAL

        return RobustnessVerdict.ROBUST

    @staticmethod
    def _tokenize_expression(expression: str) -> set[str]:
        """Tokenize a WQ expression into meaningful structural tokens.

        Extracts operators (ts_, rank, group_neutralize, etc.) and field-like
        tokens for Jaccard similarity comparison. Ignores numeric literals.
        """
        import re

        operators = {
            "ts_",
            "rank",
            "group_neutralize",
            "ts_decay_linear",
            "ts_av_diff",
            "ts_mean",
            "ts_std_dev",
            "ts_sum",
            "ts_max",
            "ts_min",
            "ts_regression",
            "ts_corr",
            "ts_covariance",
            "ts_zscore",
            "ts_skewness",
            "ts_kurtosis",
            "ts_argmax",
            "ts_argmin",
            "ts_product",
            "ts_delta",
            "ts_moment",
            "ts_entropy",
            "signed_power",
            "abs",
            "log",
            "sign",
            "sqrt",
            "truncate",
            "normalize",
            "winsorize",
            "sigmoid",
            "step",
            "if_else",
            "cond",
            "min",
            "max",
            "avg",
            "sum",
            "count",
            "close",
            "open",
            "high",
            "low",
            "volume",
            "vwap",
            "returns",
            "sharesout",
            "cap",
            "sales",
            "earnings",
            "assets",
            "market_cap",
            "sector",
            "industry",
            "subindustry",
        }

        tokens: set[str] = set()
        expr_lower = expression.lower()
        for op in operators:
            if op.lower() in expr_lower:
                tokens.add(op.lower())

        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", expression)
        if numbers:
            tokens.add("_has_numeric")

        paren_depth = expression.count("(")
        if paren_depth > 3:
            tokens.add("_deep_nesting")

        return tokens

    @staticmethod
    def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
        """Compute Jaccard similarity between two token sets.

        |A ∩ B| / |A ∪ B| with epsilon guard on denominator.
        """
        if not set_a or not set_b:
            return 0.0

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        denom = union if union > 0 else _EPSILON
        return intersection / denom
