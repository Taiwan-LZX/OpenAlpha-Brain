from __future__ import annotations

import contextlib
import logging
import time

import aiohttp

from openalpha_brain.config.config import settings
from openalpha_brain.core.loop_state import (
    _algo_tick,
    _brain_cookies_lock,
    _evo_db,
    _experience_distiller,
    _failure_lib,
    _log,
    _logic_library,
    _mab,
    _pnl_analyzer,
    _rag_engine,
    _scheduler,
    _success_lib,
    _tool_factory,
    _whitelist_mgr,
    get_brain_cookies,
    set_brain_cookies,
)
from openalpha_brain.core.models import BrainSimStatus, BrainSubmissionResult, SessionStatus
from openalpha_brain.generation.alpha_generator import _summarise_rejected
from openalpha_brain.learning.mab import (
    PENALTY_BRAIN_ERROR,
    PENALTY_BRAIN_FAIL,
    REWARD_BRAIN_SUBMIT,
    REWARD_SHARPE_05,
    REWARD_SHARPE_10,
)
from openalpha_brain.learning.reward_updater import (
    _apply_mab_feedback,
    _extract_ops_and_fields,
    _sync_mab_bias_from_evidence,
)
from openalpha_brain.monitoring.algorithm_telemetry import AlgorithmTelemetryCollector
from openalpha_brain.services import brain_client
from openalpha_brain.services.brain_submitter import _build_brain_result_dict
from openalpha_brain.utils.paper_edge_enhancements import cluster_failure_patterns
from openalpha_brain.utils.resilience import async_timeout, get_circuit_breaker
from openalpha_brain.utils.volatility_detector import estimate_garch11
from openalpha_brain.validation import validator as val
from openalpha_brain.validation.overfit_detector import detect_overfit
from openalpha_brain.validation.validator import compute_hierarchical_reward, get_reward_level

logger = logging.getLogger(__name__)

DRAWDOWN_PENALTY_THRESHOLD = getattr(settings, "DRAWDOWN_PENALTY_THRESHOLD", 10.0)
DRAWDOWN_PENALTY = getattr(settings, "DRAWDOWN_PENALTY", 0.03)
OVERFITTING_WARNING_PENALTY = getattr(settings, "OVERFITTING_WARNING_PENALTY", 0.2)
MARGIN_EFFICIENCY_THRESHOLD = getattr(settings, "MARGIN_EFFICIENCY_THRESHOLD", 0.5)
HIERARCHICAL_REWARD_QUALITY_THRESHOLD = getattr(settings, "HIERARCHICAL_REWARD_QUALITY_THRESHOLD", 0.6)
HIERARCHICAL_REWARD_BASIC_THRESHOLD = getattr(settings, "HIERARCHICAL_REWARD_BASIC_THRESHOLD", 0.3)
GARCH_CLUSTERING_PENALTY = getattr(settings, "GARCH_CLUSTERING_PENALTY", 0.05)
GARCH_MIN_PNL_LENGTH = getattr(settings, "GARCH_MIN_PNL_LENGTH", 20)
OVERFIT_DETECTION_PENALTY = getattr(settings, "OVERFIT_DETECTION_PENALTY", 0.1)
OVERFIT_MAB_PENALTY = getattr(settings, "OVERFIT_MAB_PENALTY", 0.3)
HIGH_CORRELATION_THRESHOLD = getattr(settings, "HIGH_CORRELATION_THRESHOLD", 0.7)
CORRELATION_PENALTY_THRESHOLD = getattr(settings, "CORRELATION_PENALTY_THRESHOLD", 0.3)
CORRELATION_PENALTY_COEFFICIENT = getattr(settings, "CORRELATION_PENALTY_COEFFICIENT", 0.1)
PROD_CORRELATION_THRESHOLD = getattr(settings, "PROD_CORRELATION_THRESHOLD", 0.7)
PROD_CORRELATION_PENALTY_COEFFICIENT = getattr(settings, "PROD_CORRELATION_PENALTY_COEFFICIENT", 0.15)
DIVERSITY_BONUS_THRESHOLD = getattr(settings, "DIVERSITY_BONUS_THRESHOLD", 0.5)
DIVERSITY_BONUS = getattr(settings, "DIVERSITY_BONUS", 0.05)
DIVERSITY_PENALTY_THRESHOLD = getattr(settings, "DIVERSITY_PENALTY_THRESHOLD", 0.2)
DIVERSITY_PENALTY = getattr(settings, "DIVERSITY_PENALTY", 0.05)


def compute_hierarchical_reward_with_penalties(
    brain_result: BrainSubmissionResult,
    expression: str,
    session_id: str,
    log_prefix: str = "",
) -> tuple[float, str]:
    """[Brief description of function purpose.]

    Args:
        brain_result (BrainSubmissionResult): [Description]
        expression (str): [Description]
        session_id (str): [Description]
        log_prefix (str): [Description]

    Returns:
        tuple[float, str]: [Description]
    """
    hierarchical_reward = 0.0
    hierarchical_level_name = ""
    _dd_penalty = 0.0
    _of_penalty = 0.0
    if settings.HIERARCHICAL_REWARD_ENABLED:
        brain_result_dict = _build_brain_result_dict(brain_result)
        hierarchical_reward = compute_hierarchical_reward(expression, brain_result_dict)
        hierarchical_level_name, _ = get_reward_level(hierarchical_reward)
        if brain_result.real_drawdown is not None and brain_result.real_drawdown > DRAWDOWN_PENALTY_THRESHOLD:
            hierarchical_reward -= DRAWDOWN_PENALTY
            _dd_penalty = DRAWDOWN_PENALTY
            logger.info(
                "[%s] %sDrawdown penalty: %.1f%% > 10%%, reward -0.03",
                session_id,
                log_prefix,
                brain_result.real_drawdown,
            )
        if hasattr(brain_result, "overfitting_warning") and brain_result.overfitting_warning:
            hierarchical_reward -= OVERFITTING_WARNING_PENALTY
            _of_penalty = OVERFITTING_WARNING_PENALTY
            logger.info(
                "[%s] %sOverfitting penalty: IS/OS decay ratio %.2f < 0.5, reward -0.2",
                session_id,
                log_prefix,
                getattr(brain_result, "is_os_decay_ratio", 0),
            )
        logger.info(
            "[%s] %sHierarchical reward: %.2f (%s) — decision point",
            session_id,
            log_prefix,
            hierarchical_reward,
            hierarchical_level_name,
        )
        logger.info(
            "%sHIERARCHICAL_REWARD: level=%s reward=%.4f drawdown_penalty=%.4f overfit_penalty=%.4f",
            log_prefix,
            hierarchical_level_name,
            hierarchical_reward,
            _dd_penalty,
            _of_penalty,
        )
    return hierarchical_reward, hierarchical_level_name


def check_margin_efficiency(
    brain_result: BrainSubmissionResult,
    alpha,
    session_id: str,
    log_prefix: str = "",
) -> None:
    """[Brief description of function purpose.]

    Args:
        brain_result (BrainSubmissionResult): [Description]
        alpha: [Description]
        session_id (str): [Description]
        log_prefix (str): [Description]

    Returns:
        None: [Description]
    """
    if brain_result.real_returns is not None and brain_result.real_margin is not None and brain_result.real_margin > 0:
        margin_efficiency = brain_result.real_returns / brain_result.real_margin
        if margin_efficiency < MARGIN_EFFICIENCY_THRESHOLD:
            logger.info(
                "[%s] %sLow margin efficiency: %.2f (returns/margin)",
                session_id,
                log_prefix,
                margin_efficiency,
            )
        alpha.margin_efficiency = margin_efficiency


async def record_pass_feedback(
    brain_result: BrainSubmissionResult,
    alpha,
    expression: str,
    exploration_direction: str,
    parsed: dict | None,
    session_id: str,
    hierarchical_reward: float,
    exp_card_rule_ids: list[str] | None = None,
    log_prefix: str = "",
) -> None:
    eid = None
    tel = AlgorithmTelemetryCollector.get_instance()
    try:
        eid = await tel.record_enter("BrainResultProcessor", cycle_id=session_id, expr_id=hash(expression) % 10000)
        t0 = time.perf_counter()

        if _experience_distiller and exp_card_rule_ids:
            try:
                for _rid in exp_card_rule_ids:
                    _experience_distiller.record_card_usage(_rid, success=True)
            except (OSError, ValueError, RuntimeError):
                pass

        if settings.HIERARCHICAL_REWARD_ENABLED and hierarchical_reward >= HIERARCHICAL_REWARD_QUALITY_THRESHOLD:
            logger.info(
                "[%s] %sHierarchical reward %.2f >= 0.6 (Quality level) — BRAIN PASS with high quality",
                session_id,
                log_prefix,
                hierarchical_reward,
            )
            await _apply_mab_feedback(
                alpha.exploration_direction, expression, reward=REWARD_BRAIN_SUBMIT, _scheduler=_scheduler
            )
            logger.info(
                "%sPASS_FEEDBACK: direction=%s mab_reward=%.2f",
                log_prefix,
                alpha.exploration_direction or exploration_direction,
                REWARD_BRAIN_SUBMIT,
            )
        elif settings.HIERARCHICAL_REWARD_ENABLED and hierarchical_reward >= HIERARCHICAL_REWARD_BASIC_THRESHOLD:
            logger.info(
                "[%s] %sHierarchical reward %.2f >= 0.3 (Basic level) — BRAIN PASS with basic quality",
                session_id,
                log_prefix,
                hierarchical_reward,
            )
            await _apply_mab_feedback(
                alpha.exploration_direction, expression, reward=REWARD_SHARPE_05, _scheduler=_scheduler
            )
            logger.info(
                "%sPASS_FEEDBACK: direction=%s mab_reward=%.2f",
                log_prefix,
                alpha.exploration_direction or exploration_direction,
                REWARD_SHARPE_05,
            )
        else:
            await _apply_mab_feedback(
                alpha.exploration_direction, expression, reward=REWARD_BRAIN_SUBMIT, _scheduler=_scheduler
            )
            logger.info(
                "%sPASS_FEEDBACK: direction=%s mab_reward=%.2f",
                log_prefix,
                alpha.exploration_direction or exploration_direction,
                REWARD_BRAIN_SUBMIT,
            )

        if _rag_engine and hasattr(_rag_engine, "update_weights_from_feedback"):
            try:
                _rag_engine.update_weights_from_feedback(exploration_direction, [])
            except (OSError, ValueError, RuntimeError):
                logger.warning("RAG feedback update failed", exc_info=True)

        if settings.EVIDENCE_RECORDING_ENABLED and _logic_library:
            try:
                direction = alpha.exploration_direction or exploration_direction or "unknown"
                logics = _logic_library.get_logic_for_direction(direction)
                _algo_tick("evidence_recording")
                for logic in logics[:1]:
                    _logic_library.record_evidence(
                        logic.logic_id,
                        True,
                        expression=expression,
                        sharpe=brain_result.real_sharpe,
                        fitness=brain_result.real_fitness,
                        turnover=brain_result.real_turnover,
                        direction=alpha.exploration_direction or "",
                        fix_success=True,
                    )
            except (OSError, ValueError, RuntimeError):
                pass
            _sync_mab_bias_from_evidence()

        if _tool_factory:
            try:
                _original_failures = getattr(brain_result, "gate_failures", []) or []
                _tool_factory.record_fix_pattern(
                    failure_type=str(_original_failures[:2]),
                    fix_success=True,
                    direction=alpha.exploration_direction or exploration_direction,
                    fix_attempt=expression or "",
                )
            except (OSError, ValueError, RuntimeError):
                logger.warning("ToolFactory record_fix_pattern (pass) failed", exc_info=True)

        if brain_result.real_sharpe is not None:
            if brain_result.real_sharpe > 1.0:
                await _apply_mab_feedback(
                    alpha.exploration_direction, expression, reward=REWARD_SHARPE_10, _scheduler=_scheduler
                )
            elif brain_result.real_sharpe > 0.5:
                await _apply_mab_feedback(
                    alpha.exploration_direction, expression, reward=REWARD_SHARPE_05, _scheduler=_scheduler
                )

        if _whitelist_mgr:
            try:
                _, fields = _extract_ops_and_fields(expression)
                for field in fields:
                    _whitelist_mgr.solidify_field(field)
                    if brain_result.real_sharpe is not None:
                        _reward_val = 0.5 if brain_result.real_sharpe > 1.0 else 0.2
                        _whitelist_mgr.update_field_reward(field, reward=_reward_val)
            except (OSError, ValueError, RuntimeError):
                pass

        try:
            cc = val.get_complexity_controller()
            _algo_tick("complexity_control")
            cc_metrics = cc.compute_complexity(expression)
            cc.record_success(cc_metrics)
            cc.adapt_thresholds()
        except (OSError, ValueError, RuntimeError):
            logger.warning("Complexity recording failed", exc_info=True)

            ms = (time.perf_counter() - t0) * 1000
            with contextlib.suppress(OSError, ValueError, RuntimeError):
                await tel.record_exit("BrainResultProcessor", eid, metrics={"reward_applied": True}, duration_ms=ms)
    except Exception as e:
        if eid:
            with contextlib.suppress(OSError, ValueError, RuntimeError):
                await tel.record_error("BrainResultProcessor", str(e), type(e).__name__)
        raise


async def submit_for_review(
    brain_result: BrainSubmissionResult,
    alpha,
    session_id: str,
    state,
) -> None:
    """[Brief description of function purpose.]

    Args:
        brain_result (BrainSubmissionResult): [Description]
        alpha: [Description]
        session_id (str): [Description]
        state: [Description]

    Returns:
        None: [Description]
    """
    _review_cb = get_circuit_breaker("brain_review_api", failure_threshold=3, recovery_timeout=60.0)
    if _review_cb.is_open:
        logger.warning("[%s] REVIEW_CIRCUIT_OPEN: skipping review submission", session_id)
        return
    try:
        async with _brain_cookies_lock:
            if get_brain_cookies() is None:
                set_brain_cookies(
                    await brain_client.authenticate(
                        settings.BRAIN_EMAIL,
                        settings.BRAIN_PASSWORD,
                    )
                )
        review_ok = await async_timeout(
            brain_client.submit_alpha_for_review(
                brain_result.alpha_id,
                get_brain_cookies(),
            ),
            timeout_seconds=20.0,
            name="brain_review_submit",
        )
        _review_cb.record_success()
        brain_result.review_submitted = review_ok
        if review_ok:
            _log(
                state,
                "SUBMIT_REVIEW",
                f"Alpha {alpha.alpha_id} submitted for IQC review (brain_id={brain_result.alpha_id})",
                {"brain_id": brain_result.alpha_id},
            )
        else:
            _log(
                state,
                "SUBMIT_REVIEW_FAIL",
                f"Alpha {alpha.alpha_id} review submission failed (brain_id={brain_result.alpha_id})",
                {"brain_id": brain_result.alpha_id},
            )
    except TimeoutError:
        _review_cb.record_failure("Review timeout")
        logger.warning("[%s] Review submission timed out", session_id)
    except brain_client.BrainAuthError:
        _review_cb.record_failure("Auth expired")
        async with _brain_cookies_lock:
            set_brain_cookies(None)
        _log(state, "AUTH", "Cookie expired during review submission")
    except (TimeoutError, aiohttp.ClientError, ConnectionError) as exc:
        _review_cb.record_failure(str(exc))
        logger.error("[%s] Error submitting alpha for review: %s", session_id, exc)


async def run_stability_analysis(
    brain_result: BrainSubmissionResult,
    alpha,
    session_id: str,
    state,
    hierarchical_reward: float,
    log_prefix: str = "",
) -> tuple[float, list | None, list | None]:
    """[Brief description of function purpose.]

    Args:
        brain_result (BrainSubmissionResult): [Description]
        alpha: [Description]
        session_id (str): [Description]
        state: [Description]
        hierarchical_reward (float): [Description]
        log_prefix (str): [Description]

    Returns:
        tuple[float, list | None, list | None]: [Description]
    """
    pnl_curve = None
    yearly_data = None
    daily_pnl = None
    _pnl_report = None

    if _pnl_analyzer is not None:
        try:
            from openalpha_brain.services.brain_data_client import get_brain_data_client

            _bdc = get_brain_data_client()
            if _bdc and brain_result.alpha_id:
                yearly_data = await _bdc.get_yearly_performance(brain_result.alpha_id)
            if _bdc and brain_result.alpha_id:
                try:
                    pnl_curve = await _bdc.get_pnl_curve(brain_result.alpha_id)
                    logger.info(
                        "%scycle=0 PNL_FETCH: alpha_id=%s data_len=%d",
                        log_prefix,
                        brain_result.alpha_id,
                        len(pnl_curve) if pnl_curve else 0,
                    )
                except (ConnectionError, OSError, TimeoutError) as _pnl_exc:
                    logger.warning(
                        "%scycle=0 PNL_FETCH_FAILED: alpha_id=%s reason=%s fallback=%s",
                        log_prefix,
                        brain_result.alpha_id,
                        str(_pnl_exc),
                        "yearly_stats_only",
                    )
                    logger.warning("[%s] %sPnL curve fetch failed", session_id, log_prefix, exc_info=True)
            if _bdc and brain_result.alpha_id:
                try:
                    daily_pnl = await _bdc.get_daily_pnl(brain_result.alpha_id)
                    if daily_pnl:
                        logger.info(
                            "%scycle=0 DAILY_PNL_FETCH: alpha_id=%s data_len=%d",
                            log_prefix,
                            brain_result.alpha_id,
                            len(daily_pnl),
                        )
                except (TimeoutError, aiohttp.ClientError, ConnectionError):  # noqa: SIM105
                    logger.debug(
                        "[%s] %sDaily PnL fetch skipped for alpha %s",
                        session_id,
                        log_prefix,
                        brain_result.alpha_id,
                        exc_info=True,
                    )
            if yearly_data or pnl_curve:
                _algo_tick("pnl_stability_analysis")
                _pnl_report = _pnl_analyzer.generate_stability_report(
                    yearly_data=yearly_data,
                    pnl_curve=pnl_curve,
                )
                if _pnl_report.reward_adjustment != 0.0 and settings.HIERARCHICAL_REWARD_ENABLED:
                    hierarchical_reward += _pnl_report.reward_adjustment
                    logger.info(
                        "[%s] %sPnL stability adjustment: %.4f, hierarchical_reward now %.2f",
                        session_id,
                        log_prefix,
                        _pnl_report.reward_adjustment,
                        hierarchical_reward,
                    )
                if _pnl_report.warnings:
                    for _w in _pnl_report.warnings:
                        _log(state, "PNL_STABILITY_WARNING", _w)
        except (ValueError, TypeError, OSError, RuntimeError):
            logger.warning("[%s] %sPnL stability analysis failed", session_id, log_prefix, exc_info=True)

    if pnl_curve and len(pnl_curve) >= GARCH_MIN_PNL_LENGTH:
        try:
            _algo_tick(f"{log_prefix.lower()}garch_volatility" if log_prefix else "garch_volatility")
            _garch_result = estimate_garch11(pnl_curve)
            logger.info(
                "%scycle=0 GARCH: alpha_id=%s persistence=%.4f volatility=%.4f",
                log_prefix,
                brain_result.alpha_id,
                _garch_result.persistence,
                getattr(_garch_result, "volatility", 0.0),
            )
            if _garch_result.is_clustering:
                logger.warning(
                    "[%s] %sGARCH volatility clustering detected: persistence=%.4f, half_life=%.1f",
                    session_id,
                    log_prefix,
                    _garch_result.persistence,
                    _garch_result.half_life,
                )
                if settings.HIERARCHICAL_REWARD_ENABLED:
                    hierarchical_reward -= GARCH_CLUSTERING_PENALTY
                    logger.info(
                        "[%s] %sGARCH clustering penalty: -0.05, hierarchical_reward now %.2f",
                        session_id,
                        log_prefix,
                        hierarchical_reward,
                    )
            if _pnl_analyzer is not None and _garch_result.persistence > 0:
                _garch_score = _pnl_analyzer.compute_stability_score(
                    yearly_data=yearly_data,
                    pnl_curve=pnl_curve,
                    drawdown_pct=_pnl_report.drawdown_analysis.max_drawdown_pct
                    if _pnl_report and _pnl_report.drawdown_analysis
                    else None,
                    garch_persistence=_garch_result.persistence,
                )
                if _pnl_report:
                    _pnl_report.stability_score = _garch_score
        except (ValueError, TypeError, OSError, RuntimeError):
            logger.warning("[%s] %sGARCH analysis failed", session_id, log_prefix, exc_info=True)

    try:
        _algo_tick(f"{log_prefix.lower()}overfit_detection" if log_prefix else "overfit_detection")
        _yearly_sharpes = None
        if yearly_data:
            _yearly_sharpes = [
                yd.get("sharpe", 0.0) for yd in yearly_data if isinstance(yd.get("sharpe"), (int, float))
            ]
        _overfit_pnl_curve = daily_pnl if daily_pnl else pnl_curve
        _overfit_result = detect_overfit(
            is_sharpe=brain_result.real_sharpe,
            os_sharpe=getattr(brain_result, "os_sharpe", None),
            yearly_sharpes=_yearly_sharpes,
            pnl_curve=_overfit_pnl_curve,
        )
        logger.info(
            "%scycle=0 OVERFIT: alpha_id=%s is_os_ratio=%.4f warning=%s",
            log_prefix,
            brain_result.alpha_id,
            _overfit_result.is_os_decay_ratio,
            str(_overfit_result.warnings),
        )
        if _overfit_result.is_overfit:
            logger.warning(
                "[%s] %sOverfitting detected: %s", session_id, log_prefix, "; ".join(_overfit_result.warnings)
            )
            if settings.HIERARCHICAL_REWARD_ENABLED:
                hierarchical_reward -= OVERFIT_DETECTION_PENALTY
                logger.info(
                    "[%s] %sOverfit penalty: -0.1, hierarchical_reward now %.2f",
                    session_id,
                    log_prefix,
                    hierarchical_reward,
                )
            if _mab is not None:
                await _apply_mab_feedback(
                    alpha.exploration_direction, alpha.expression, reward=-OVERFIT_MAB_PENALTY, _scheduler=_scheduler
                )
                logger.info(
                    "[%s] MONITOR: overfit_penalty: direction=%s penalty=-0.3",
                    session_id,
                    alpha.exploration_direction,
                )
        elif _overfit_result.warnings:
            for _ow in _overfit_result.warnings:
                logger.info("[%s] %sOverfit warning: %s", session_id, log_prefix, _ow)
    except (ValueError, TypeError, OSError, RuntimeError):
        logger.warning("[%s] %sOverfit detection failed", session_id, log_prefix, exc_info=True)

    return hierarchical_reward, pnl_curve, yearly_data


async def fetch_correlation_analysis(
    brain_result: BrainSubmissionResult,
    alpha,
    session_id: str,
    hierarchical_reward: float,
    log_prefix: str = "",
) -> float:
    """[Brief description of function purpose.]

    Args:
        brain_result (BrainSubmissionResult): [Description]
        alpha: [Description]
        session_id (str): [Description]
        hierarchical_reward (float): [Description]
        log_prefix (str): [Description]

    Returns:
        float: [Description]
    """
    try:
        from openalpha_brain.services.brain_data_client import get_brain_data_client

        _bdc = get_brain_data_client()
    except (ImportError, AttributeError, RuntimeError):
        _bdc = None

    if _bdc and brain_result.alpha_id:
        try:
            _algo_tick("correlation_fetch")
            _corr_data = await _bdc.get_correlations(brain_result.alpha_id)
            if _corr_data:
                _corr_values = []
                if isinstance(_corr_data, dict):
                    for _ck, _cv in _corr_data.items():
                        if isinstance(_cv, (int, float)):
                            _corr_values.append(abs(_cv))
                        elif isinstance(_cv, dict):
                            for _ck2, _cv2 in _cv.items():
                                if isinstance(_cv2, (int, float)):
                                    _corr_values.append(abs(_cv2))
                elif isinstance(_corr_data, list):
                    for _item in _corr_data:
                        if isinstance(_item, (int, float)):
                            _corr_values.append(abs(_item))
                        elif isinstance(_item, dict):
                            for _v in _item.values():
                                if isinstance(_v, (int, float)):
                                    _corr_values.append(abs(_v))
                if not _corr_values and brain_result.brain_checks:
                    for _chk in brain_result.brain_checks:
                        if isinstance(_chk, dict) and "SELF_CORRELATION" in str(_chk.get("name", "")).upper():
                            _cv = _chk.get("value")
                            if isinstance(_cv, (int, float)):
                                _corr_values.append(abs(_cv))
                if _corr_values:
                    _max_corr = max(_corr_values)
                    _avg_corr = sum(_corr_values) / len(_corr_values)
                    if _max_corr > HIGH_CORRELATION_THRESHOLD:
                        logger.warning(
                            "[%s] %sHigh correlation %.2f for alpha %s",
                            session_id,
                            log_prefix,
                            _max_corr,
                            brain_result.alpha_id,
                        )
                    if settings.HIERARCHICAL_REWARD_ENABLED and _avg_corr > CORRELATION_PENALTY_THRESHOLD:
                        _corr_penalty = -CORRELATION_PENALTY_COEFFICIENT * (_avg_corr - CORRELATION_PENALTY_THRESHOLD)
                        hierarchical_reward += _corr_penalty
                        logger.info(
                            "[%s] %sCorrelation penalty: avg_corr=%.2f, penalty=%.4f, hierarchical_reward now %.2f",
                            session_id,
                            log_prefix,
                            _avg_corr,
                            _corr_penalty,
                            hierarchical_reward,
                        )
        except (ValueError, TypeError, OSError, RuntimeError):
            logger.warning(
                "[%s] %sCorrelation fetch failed for alpha %s",
                session_id,
                log_prefix,
                brain_result.alpha_id,
                exc_info=True,
            )

        if _bdc and brain_result.alpha_id:
            try:
                _algo_tick(f"{log_prefix.lower()}prod_correlation_check" if log_prefix else "prod_correlation_check")
                _prod_corr_data = await _bdc.get_prod_correlations(brain_result.alpha_id)
                if _prod_corr_data and isinstance(_prod_corr_data, dict):
                    _prod_max = _prod_corr_data.get("max")
                    _prod_records = _prod_corr_data.get("records", [])
                    _prod_corr_values = []
                    if isinstance(_prod_records, list):
                        for _pr in _prod_records:
                            if isinstance(_pr, (list, tuple)) and len(_pr) >= 3:
                                with contextlib.suppress(ValueError, TypeError):
                                    _prod_corr_values.append(abs(float(_pr[-1])))
                    if _prod_max is not None:
                        with contextlib.suppress(ValueError, TypeError):
                            _prod_corr_values.append(abs(float(_prod_max)))
                    if _prod_corr_values:
                        _max_prod_corr = max(_prod_corr_values)
                        if _max_prod_corr > PROD_CORRELATION_THRESHOLD:
                            logger.warning(
                                "[%s] %sHigh production correlation %.4f for alpha %s — likely duplicate",
                                session_id,
                                log_prefix,
                                _max_prod_corr,
                                brain_result.alpha_id,
                            )
                            if settings.HIERARCHICAL_REWARD_ENABLED:
                                _prod_penalty = -PROD_CORRELATION_PENALTY_COEFFICIENT * (
                                    _max_prod_corr - PROD_CORRELATION_THRESHOLD
                                )
                                hierarchical_reward += _prod_penalty
                                logger.info(
                                    "[%s] %sProd correlation penalty: max=%.4f, penalty=%.4f, hierarchical_reward now %.2f",  # noqa: E501
                                    session_id,
                                    log_prefix,
                                    _max_prod_corr,
                                    _prod_penalty,
                                    hierarchical_reward,
                                )
            except (TimeoutError, aiohttp.ClientError, ConnectionError):  # noqa: SIM105
                logger.debug(
                    "[%s] %sProd correlation check skipped for alpha %s (likely not submitted)",
                    session_id,
                    log_prefix,
                    brain_result.alpha_id,
                    exc_info=True,
                )

        if _bdc and brain_result.alpha_id:
            try:
                _algo_tick(f"{log_prefix.lower()}self_correlation_check" if log_prefix else "self_correlation_check")
                _self_corr_data = await _bdc.get_self_correlations(brain_result.alpha_id)
                if _self_corr_data and isinstance(_self_corr_data, dict):
                    _self_corr_values = []
                    for _sck, _scv in _self_corr_data.items():
                        if isinstance(_scv, (int, float)):
                            _self_corr_values.append(abs(_scv))
                        elif isinstance(_scv, dict):
                            for _scv2 in _scv.values():
                                if isinstance(_scv2, (int, float)):
                                    _self_corr_values.append(abs(_scv2))
                    if _self_corr_values:
                        _max_self_corr = max(_self_corr_values)
                        _avg_self_corr = sum(_self_corr_values) / len(_self_corr_values)
                        logger.info(
                            "[%s] %sSelf-correlation analysis: max=%.4f avg=%.4f for alpha %s",
                            session_id,
                            log_prefix,
                            _max_self_corr,
                            _avg_self_corr,
                            brain_result.alpha_id,
                        )
                        if _max_self_corr > HIGH_CORRELATION_THRESHOLD:
                            logger.warning(
                                "[%s] %sHigh self-correlation %.4f for alpha %s — may overlap with own submitted alphas",  # noqa: E501
                                session_id,
                                log_prefix,
                                _max_self_corr,
                                brain_result.alpha_id,
                            )
                            if settings.HIERARCHICAL_REWARD_ENABLED:
                                _self_corr_penalty = -CORRELATION_PENALTY_COEFFICIENT * (
                                    _max_self_corr - CORRELATION_PENALTY_THRESHOLD
                                )
                                hierarchical_reward += _self_corr_penalty
                                logger.info(
                                    "[%s] %sSelf-correlation penalty: max=%.4f, penalty=%.4f, hierarchical_reward now %.2f",  # noqa: E501
                                    session_id,
                                    log_prefix,
                                    _max_self_corr,
                                    _self_corr_penalty,
                                    hierarchical_reward,
                                )
            except (ValueError, TypeError, OSError, RuntimeError):
                logger.debug(
                    "[%s] %sSelf-correlation check skipped for alpha %s",
                    session_id,
                    log_prefix,
                    brain_result.alpha_id,
                    exc_info=True,
                )

    return hierarchical_reward


async def record_evo_and_success(
    brain_result: BrainSubmissionResult,
    alpha,
    expression: str,
    exploration_direction: str,
    session_id: str,
    hierarchical_reward: float,
    parsed: dict | None = None,
    log_prefix: str = "",
) -> float:
    """[Brief description of function purpose.]

    Args:
        brain_result (BrainSubmissionResult): [Description]
        alpha: [Description]
        expression (str): [Description]
        exploration_direction (str): [Description]
        session_id (str): [Description]
        hierarchical_reward (float): [Description]
        parsed (dict | None): [Description]
        log_prefix (str): [Description]

    Returns:
        float: [Description]
    """
    if _evo_db is not None:
        _algo_tick("add_record")
        await _evo_db.add_record(
            expression,
            sharpe=brain_result.real_sharpe,
            fitness=brain_result.real_fitness,
            turnover=brain_result.real_turnover,
            direction=alpha.exploration_direction or exploration_direction,
            session_id=session_id,
            status="PASS",
        )
        try:
            _portfolio_eval = await _evo_db.portfolio_level_evaluation(expression)
            _diversity_computed = _portfolio_eval.get("diversity_computed", False)
            _diversity_score = _portfolio_eval.get("diversity_score")
            if (
                _diversity_computed
                and _diversity_score is not None
                and _diversity_score > DIVERSITY_BONUS_THRESHOLD
                and settings.HIERARCHICAL_REWARD_ENABLED
            ):
                hierarchical_reward += DIVERSITY_BONUS
                logger.info(
                    "[%s] %sPortfolio diversity bonus: diversity=%.3f, hierarchical_reward adjusted to %.2f",
                    session_id,
                    log_prefix,
                    _diversity_score,
                    hierarchical_reward,
                )
            elif (
                _diversity_computed
                and _diversity_score is not None
                and _diversity_score < DIVERSITY_PENALTY_THRESHOLD
                and settings.HIERARCHICAL_REWARD_ENABLED
            ):
                hierarchical_reward -= DIVERSITY_PENALTY
                logger.info(
                    "[%s] %sPortfolio diversity penalty: diversity=%.3f, hierarchical_reward adjusted to %.2f",
                    session_id,
                    log_prefix,
                    _diversity_score,
                    hierarchical_reward,
                )
        except (ValueError, TypeError, OSError, RuntimeError):
            pass

    if _success_lib and settings.SUCCESS_CASE_LIBRARY_ENABLED:
        try:
            _algo_tick("success_case_add")
            await _success_lib.add_case(
                expr=expression,
                hypothesis=parsed.get("rationale", "") if parsed else alpha.rationale or "",
                sharpe=brain_result.real_sharpe or 0.0,
                fitness=brain_result.real_fitness or 0.0,
                turnover=brain_result.real_turnover or 0.0,
                direction=alpha.exploration_direction or exploration_direction,
                session_id=session_id,
            )
        except (OSError, ValueError, RuntimeError):
            logger.warning("SuccessCaseLibrary add_case failed", exc_info=True)

    return hierarchical_reward


async def record_fail_feedback(
    brain_result: BrainSubmissionResult,
    alpha,
    expression: str,
    exploration_direction: str,
    session_id: str,
    fingerprint_dict: dict,
    global_cycle: int,
    state,
    hierarchical_reward: float,
    exp_card_rule_ids: list[str] | None = None,
    log_prefix: str = "",
) -> None:
    eid = None
    tel = AlgorithmTelemetryCollector.get_instance()
    try:
        eid = await tel.record_enter("BrainResultProcessor", cycle_id=session_id, expr_id=hash(expression) % 10000)
        t0 = time.perf_counter()
        from openalpha_brain.generation.prompts import build_restart_trigger

        if _experience_distiller and exp_card_rule_ids:
            try:
                for _rid in exp_card_rule_ids:
                    _experience_distiller.record_card_usage(_rid, success=False)
            except (OSError, ValueError, RuntimeError):
                pass

        state.status = SessionStatus.ITERATING

        if brain_result.status == BrainSimStatus.ERROR:
            if settings.HIERARCHICAL_REWARD_ENABLED and hierarchical_reward >= HIERARCHICAL_REWARD_BASIC_THRESHOLD:
                logger.info(
                    "[%s] %sHierarchical reward %.2f >= 0.3 — BRAIN ERROR but soft penalty",
                    session_id,
                    log_prefix,
                    hierarchical_reward,
                )
                await _apply_mab_feedback(
                    alpha.exploration_direction, expression, penalty=PENALTY_BRAIN_FAIL, _scheduler=_scheduler
                )
                logger.info(
                    "%sFAIL_FEEDBACK: direction=%s mab_penalty=%.2f",
                    log_prefix,
                    alpha.exploration_direction or exploration_direction,
                    PENALTY_BRAIN_FAIL,
                )
            else:
                await _apply_mab_feedback(
                    alpha.exploration_direction, expression, penalty=PENALTY_BRAIN_ERROR, _scheduler=_scheduler
                )
                logger.info(
                    "%sFAIL_FEEDBACK: direction=%s mab_penalty=%.2f",
                    log_prefix,
                    alpha.exploration_direction or exploration_direction,
                    PENALTY_BRAIN_ERROR,
                )
            _log(
                state,
                "ABANDON",
                f"Alpha {alpha.alpha_id} abandoned due to API error: {brain_result.error_message}",
                {"failures": brain_result.gate_failures},
            )
        else:
            if settings.HIERARCHICAL_REWARD_ENABLED and hierarchical_reward >= HIERARCHICAL_REWARD_QUALITY_THRESHOLD:
                logger.info(
                    "[%s] %sHierarchical reward %.2f >= 0.6 (Quality level) — soft pass despite BRAIN strict check failure, recording as quality_pass",  # noqa: E501
                    session_id,
                    log_prefix,
                    hierarchical_reward,
                )
                if _success_lib and settings.SUCCESS_CASE_LIBRARY_ENABLED:
                    try:
                        _algo_tick("success_case_add")
                        await _success_lib.add_case(
                            expr=expression,
                            hypothesis=(alpha.rationale or "") + " [quality_pass]",
                            sharpe=brain_result.real_sharpe or 0.0,
                            fitness=brain_result.real_fitness or 0.0,
                            turnover=brain_result.real_turnover or 0.0,
                            direction=alpha.exploration_direction or exploration_direction,
                            session_id=session_id,
                        )
                    except (OSError, ValueError, RuntimeError):
                        logger.warning("SuccessCaseLibrary add_case (quality_pass) failed", exc_info=True)
                await _apply_mab_feedback(
                    alpha.exploration_direction, expression, reward=REWARD_SHARPE_05, _scheduler=_scheduler
                )
                logger.info(
                    "%sFAIL_FEEDBACK: direction=%s mab_penalty=%.2f",
                    log_prefix,
                    alpha.exploration_direction or exploration_direction,
                    -REWARD_SHARPE_05,
                )
                _log(
                    state,
                    "QUALITY_PASS",
                    f"Alpha {alpha.alpha_id} quality_pass (hierarchical_reward={hierarchical_reward:.2f}) — BRAIN strict checks failed but quality level reached",  # noqa: E501
                    {"hierarchical_reward": hierarchical_reward, "brain_id": brain_result.alpha_id},
                )
            elif settings.HIERARCHICAL_REWARD_ENABLED and hierarchical_reward >= HIERARCHICAL_REWARD_BASIC_THRESHOLD:
                logger.info(
                    "[%s] %sHierarchical reward %.2f >= 0.3 (Basic level) — BRAIN FAIL but no penalty",
                    session_id,
                    log_prefix,
                    hierarchical_reward,
                )
            else:
                logger.info(
                    "[%s] %sHierarchical reward %.2f < 0.3 — BRAIN FAIL with penalty",
                    session_id,
                    log_prefix,
                    hierarchical_reward,
                )
                await _apply_mab_feedback(
                    alpha.exploration_direction, expression, penalty=PENALTY_BRAIN_FAIL, _scheduler=_scheduler
                )
                logger.info(
                    "%sFAIL_FEEDBACK: direction=%s mab_penalty=%.2f",
                    log_prefix,
                    alpha.exploration_direction or exploration_direction,
                    PENALTY_BRAIN_FAIL,
                )
            _log(
                state,
                "ABANDON",
                f"Alpha {alpha.alpha_id} exhausted {state.brain_mutation_count} mutations — starting fresh ideation.",
                {"failures": brain_result.gate_failures},
            )

        if _evo_db is not None:
            _algo_tick("add_record")
            await _evo_db.add_record(
                expression,
                sharpe=brain_result.real_sharpe,
                fitness=brain_result.real_fitness,
                turnover=brain_result.real_turnover,
                direction=alpha.exploration_direction or exploration_direction,
                session_id=session_id,
                status="FAIL",
            )

        if _failure_lib and settings.FAILURE_FIX_LIBRARY_ENABLED:
            try:
                failure_type = "API_ERROR" if brain_result.status == BrainSimStatus.ERROR else "BRAIN_EXHAUSTED"
                _algo_tick("failure_fix_add")
                await _failure_lib.add_failure(
                    expr=expression,
                    failure_type=failure_type,
                    fix_attempt=None,
                    fix_success=False,
                    direction=alpha.exploration_direction or exploration_direction,
                    session_id=session_id,
                )
            except (OSError, ValueError, RuntimeError):
                logger.warning("FailureFixLibrary add_failure failed", exc_info=True)

        try:
            _algo_tick("failure_pattern_clustering")
            recent_failures = getattr(state, "failure_catalog", [])
            failure_history = [
                {
                    "expression": f.get("fingerprint", {}).get("expression", expression),
                    "failure_type": f.get("failure_type", "UNKNOWN"),
                    "sharpe": brain_result.real_sharpe,
                }
                for f in recent_failures[-10:]
                if isinstance(f, dict)
            ]
            if len(failure_history) >= 3:
                pattern_clusters = cluster_failure_patterns(failure_history, n_clusters=min(3, len(failure_history)))
                for pc in pattern_clusters:
                    if pc["size"] >= 2:
                        logger.info(
                            "[%s] FailurePatternCluster: id=%d size=%d dominant=%s fix='%s'",
                            session_id,
                            pc["cluster_id"],
                            pc["size"],
                            pc["dominant_failure_type"],
                            pc["suggested_fix"][:80],
                        )
        except (ValueError, TypeError, OSError, RuntimeError) as _fpc_exc:
            logger.debug("Failure pattern clustering skipped: %s", _fpc_exc)

        if _tool_factory:
            try:
                _gate_failures = brain_result.gate_failures if brain_result.gate_failures else []
                _tool_factory.record_fix_pattern(
                    failure_type=str(_gate_failures[:2]),
                    fix_attempt="",
                    fix_success=False,
                    direction=alpha.exploration_direction or exploration_direction,
                )
            except (OSError, ValueError, RuntimeError):
                logger.warning("ToolFactory record_fix_pattern (fail) failed", exc_info=True)

        if settings.EVIDENCE_RECORDING_ENABLED and _logic_library:
            try:
                direction = alpha.exploration_direction or exploration_direction or "unknown"
                logics = _logic_library.get_logic_for_direction(direction)
                _algo_tick("evidence_recording")
                for logic in logics[:1]:
                    _logic_library.record_evidence(
                        logic.logic_id,
                        False,
                        expression=expression,
                        sharpe=brain_result.real_sharpe,
                        fitness=brain_result.real_fitness,
                        turnover=brain_result.real_turnover,
                        direction=alpha.exploration_direction or "",
                        failure_type=str(brain_result.gate_failures[:3]),
                        fix_success=False,
                    )
            except (OSError, ValueError, RuntimeError):
                pass
            _sync_mab_bias_from_evidence()

        state.failure_catalog.append(
            {
                "fingerprint": fingerprint_dict,
                "failure_type": "API_ERROR" if brain_result.status == BrainSimStatus.ERROR else "BRAIN_EXHAUSTED",
                "mutations_tried": state.brain_mutation_count,
            }
        )
        state.conversation_history.append(
            {
                "role": "user",
                "content": build_restart_trigger(
                    global_cycle + 1,
                    _summarise_rejected([fingerprint_dict] + state.rejected_motifs[-3:]),
                ),
            }
        )

        ms = (time.perf_counter() - t0) * 1000
        with contextlib.suppress(OSError, ValueError, RuntimeError):
            await tel.record_exit("BrainResultProcessor", eid, metrics={"penalty_applied": True}, duration_ms=ms)
    except Exception as e:
        if eid:
            with contextlib.suppress(OSError, ValueError, RuntimeError):
                await tel.record_error("BrainResultProcessor", str(e), type(e).__name__)
        raise


async def run_post_brain_processing(
    alpha,
    brain_result: BrainSubmissionResult,
    expression: str,
    exploration_direction: str,
    session_id: str,
    state,
    log_prefix: str = "",
) -> None:
    """[Brief description of function purpose.]

    Args:
        alpha: [Description]
        brain_result (BrainSubmissionResult): [Description]
        expression (str): [Description]
        exploration_direction (str): [Description]
        session_id (str): [Description]
        state: [Description]
        log_prefix (str): [Description]

    Returns:
        None: [Description]
    """
    from openalpha_brain.core.loop_state import _reflection_engine, _strategy_classifier
    from openalpha_brain.learning.reward_updater import _refill_eliminated_fields

    if _strategy_classifier is not None:
        brain_dict = {
            "sharpe": brain_result.real_sharpe,
            "fitness": brain_result.real_fitness,
            "turnover": brain_result.real_turnover,
        }
        _algo_tick("classify")
        profile = await _strategy_classifier.classify(expression, brain_dict)
        logger.info(
            "[%s] Strategy classified: direction=%s horizon=%s mechanism=%s",
            session_id,
            profile.direction,
            profile.time_horizon,
            profile.mechanism,
        )

    await _refill_eliminated_fields(alpha.exploration_direction or exploration_direction)

    if _experience_distiller and _reflection_engine and _failure_lib:
        try:
            _algo_tick("experience_distillation")
            new_cards = await _experience_distiller.distill_from_failures(_reflection_engine, _failure_lib)
            if new_cards:
                logger.info(
                    "[%s] %sDistilled %d new experience cards",
                    session_id,
                    log_prefix,
                    len(new_cards),
                )
        except (OSError, ValueError, RuntimeError):
            pass

    if _experience_distiller and _logic_library:
        try:
            _algo_tick("evidence_distillation")
            new_evidence_cards = await _experience_distiller.distill_from_evidence(_logic_library, min_evidence=5)
            if new_evidence_cards:
                logger.info(
                    "[%s] %sDistilled %d cards from evidence",
                    session_id,
                    log_prefix,
                    len(new_evidence_cards),
                )
        except (OSError, ValueError, RuntimeError):
            pass

    if _tool_factory:
        try:
            _algo_tick("tool_conflict_detection")
            conflicts = _tool_factory.detect_conflicts()
            if conflicts:
                logger.info("[%s] Detected %d tool conflicts, resolved", session_id, len(conflicts))
        except (OSError, ValueError, RuntimeError):
            pass
