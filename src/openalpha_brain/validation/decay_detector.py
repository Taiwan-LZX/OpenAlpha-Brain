"""
Alpha Decay Sliding Window Detector (D3)
Monitors active alphas for performance decay using BRAIN PnL data and EWMA.

Decay Levels (4-tier response):
  L0 NONE         - No decay detected
  L1 WARNING      - EWMA sharpe declined <20% → UCB weight -20%, increase check freq
  L2 FACTOR_DECAY - EWMA sharpe declined 20-40% → unlist from success pool, log fingerprint
  L3 DIR_LIGHT    - EWMA sharpe declined 40-60% → UCB weight -50%, blacklist 30d, distill fingerprint, inject prompt
  L4 DIR_HEAVY    - EWMA sharpe declined >60% → permanent blacklist, remove from UCB, force other directions

Metrics:
  - OS Sharpe EWMA (primary)
  - IS/OS Decay Ratio EWMA
  - Composite score (weighted multi-metric)
  - GARCH(1,1) volatility anomaly flag
  - Year-over-year Sharpe trend
"""
from __future__ import annotations

import asyncio
import logging
import math
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import aiohttp

from openalpha_brain.core.models import AlphaFingerprint

logger = logging.getLogger(__name__)


class DecayLevel(str, Enum):
    NONE = "NONE"
    WARNING = "WARNING"
    FACTOR_DECAY = "FACTOR_DECAY"
    DIR_LIGHT = "DIR_LIGHT"
    DIR_HEAVY = "DIR_HEAVY"


@dataclass
class DecayFingerprint:
    direction: str
    topology: str
    temporal: str
    dataset: str
    normalization: str
    decay_level: str
    reason: str
    confirmation_checks: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    is_permanent: bool = False


@dataclass
class AlphaDecayRecord:
    alpha_id: str
    expression: str
    fingerprint: AlphaFingerprint | None = None
    direction: str = ""
    ewma_sharpe: float = 0.0
    ewma_is_os_ratio: float = 1.0
    composite_score: float = 1.0
    garch_anomaly: bool = False
    yearly_trend: float = 0.0
    decay_level: DecayLevel = DecayLevel.NONE
    consecutive_decay_checks: int = 0
    last_checked: datetime | None = None
    initial_sharpe: float = 0.0
    peak_sharpe: float = 0.0
    _sharpe_history: deque = field(default_factory=lambda: deque(maxlen=60))
    _ratio_history: deque = field(default_factory=lambda: deque(maxlen=60))


class AlphaDecayDetector:
    EWMA_HALF_LIFE = 20
    EWMA_DECAY_FACTOR = math.exp(math.log(0.5) / EWMA_HALF_LIFE)

    OS_SHARPE_WEIGHT = 0.40
    IS_OS_RATIO_WEIGHT = 0.25
    YEARLY_TREND_WEIGHT = 0.20
    GARCH_ANOMALY_WEIGHT = 0.15

    L1_WARNING_THRESHOLD = 0.80
    L2_FACTOR_DECAY_THRESHOLD = 0.60
    L3_DIR_LIGHT_THRESHOLD = 0.40
    L4_DIR_HEAVY_THRESHOLD = 0.20

    CONFIRMATION_CHECKS = 2
    BLACKLIST_DURATION_DAYS = 30

    def __init__(self) -> None:
        self._records: dict[str, AlphaDecayRecord] = {}
        self._lock = asyncio.Lock()
        self._blacklist: dict[str, datetime] = {}
        self._permanent_blacklist: set[str] = set()
        self._decay_fingerprints: list[DecayFingerprint] = []
        self._check_count: int = 0

        self._fetch_yearly: Callable[[str], Awaitable[list[dict] | None]] | None = None
        self._fetch_pnl: Callable[[str], Awaitable[list[float] | None]] | None = None
        self._estimate_garch: Callable[[list[float]], Awaitable[Any]] | None = None
        self._on_decay_detected: Callable[[str, DecayLevel, AlphaDecayRecord, str], Awaitable[None]] | None = None

    async def register_instruments(
        self,
        fetch_yearly: Callable[[str], Awaitable[list[dict] | None]],
        fetch_pnl: Callable[[str], Awaitable[list[float] | None]],
        estimate_garch: Callable[[list[float]], Awaitable[Any]] | None = None,
        on_decay_detected: Callable[[str, DecayLevel, AlphaDecayRecord, str], Awaitable[None]] | None = None,
    ) -> None:
        self._fetch_yearly = fetch_yearly
        self._fetch_pnl = fetch_pnl
        self._estimate_garch = estimate_garch
        self._on_decay_detected = on_decay_detected

    async def register_alpha(
        self,
        alpha_id: str,
        expression: str,
        fingerprint: AlphaFingerprint | None = None,
        direction: str = "",
        initial_sharpe: float = 0.0,
    ) -> None:
        async with self._lock:
            if alpha_id in self._records:
                return
            record = AlphaDecayRecord(
                alpha_id=alpha_id,
                expression=expression,
                fingerprint=fingerprint,
                direction=direction,
                initial_sharpe=initial_sharpe,
                peak_sharpe=initial_sharpe,
                ewma_sharpe=initial_sharpe,
            )
            if initial_sharpe > 0:
                record._sharpe_history.append(initial_sharpe)
            self._records[alpha_id] = record
            logger.info(
                "decay_detector: registered alpha %s dir=%s init_sharpe=%.3f",
                alpha_id, direction, initial_sharpe,
            )

    async def unregister_alpha(self, alpha_id: str) -> None:
        async with self._lock:
            self._records.pop(alpha_id, None)

    def is_blacklisted(self, direction: str) -> tuple[bool, str]:
        if direction in self._permanent_blacklist:
            return True, "permanent"
        if direction in self._blacklist:
            expiry = self._blacklist[direction]
            if datetime.now(UTC) < expiry:
                remaining = (expiry - datetime.now(UTC)).days
                return True, f"temporary ({remaining}d remaining)"
            del self._blacklist[direction]
        return False, ""

    def get_blacklisted_directions(self) -> list[str]:
        now = datetime.now(UTC)
        result: list[str] = []
        result.extend(self._permanent_blacklist)
        for d, expiry in list(self._blacklist.items()):
            if now < expiry:
                result.append(d)
            else:
                del self._blacklist[d]
        return result

    def get_decay_fingerprints(self, max_count: int = 50) -> list[DecayFingerprint]:
        return list(self._decay_fingerprints[-max_count:])

    def _compute_ewma(self, current: float, new_value: float) -> float:
        if current == 0.0:
            return new_value
        factor = self.EWMA_DECAY_FACTOR
        return factor * current + (1 - factor) * new_value

    def _compute_composite_score(self, record: AlphaDecayRecord, yearly_sharpes: list[float]) -> float:
        scores: list[tuple[float, float]] = []

        if record.initial_sharpe > 0 and record.ewma_sharpe > 0:
            os_score = min(1.0, record.ewma_sharpe / record.initial_sharpe)
            scores.append((os_score, self.OS_SHARPE_WEIGHT))

        ratio_score = 1.0
        if record.ewma_is_os_ratio < 0.3:
            ratio_score = 0.0
        elif record.ewma_is_os_ratio < 0.5:
            ratio_score = 0.3
        elif record.ewma_is_os_ratio < 0.7:
            ratio_score = 0.6
        elif record.ewma_is_os_ratio >= 0.7:
            ratio_score = 1.0
        scores.append((ratio_score, self.IS_OS_RATIO_WEIGHT))

        trend_score = 0.5
        if yearly_sharpes and len(yearly_sharpes) >= 2:
            recent_half = yearly_sharpes[len(yearly_sharpes) // 2:]
            older_half = yearly_sharpes[:len(yearly_sharpes) // 2]
            avg_recent = sum(recent_half) / len(recent_half) if recent_half else 0
            avg_older = sum(older_half) / len(older_half) if older_half else 0
            if avg_older > 0 and avg_recent >= avg_older:
                trend_score = 1.0
            elif avg_older > 0:
                trend_score = max(0.0, avg_recent / avg_older)
            record.yearly_trend = trend_score
        scores.append((trend_score, self.YEARLY_TREND_WEIGHT))

        garch_score = 0.3 if record.garch_anomaly else 1.0
        scores.append((garch_score, self.GARCH_ANOMALY_WEIGHT))

        total_weight = sum(w for _, w in scores)
        if total_weight == 0:
            return 1.0
        weighted = sum(s * w for s, w in scores) / total_weight
        return round(weighted, 4)

    def _classify_decay(self, record: AlphaDecayRecord) -> DecayLevel:
        if record.initial_sharpe <= 0:
            return DecayLevel.NONE

        ratio = record.ewma_sharpe / record.initial_sharpe if record.initial_sharpe > 0 else 1.0

        if ratio >= self.L1_WARNING_THRESHOLD:
            return DecayLevel.NONE
        if ratio >= self.L2_FACTOR_DECAY_THRESHOLD:
            return DecayLevel.WARNING
        if ratio >= self.L3_DIR_LIGHT_THRESHOLD:
            return DecayLevel.FACTOR_DECAY
        if ratio >= self.L4_DIR_HEAVY_THRESHOLD:
            return DecayLevel.DIR_LIGHT
        return DecayLevel.DIR_HEAVY

    async def check_single_alpha(self, alpha_id: str) -> AlphaDecayRecord | None:
        if self._fetch_yearly is None or self._fetch_pnl is None:
            logger.warning("decay_detector: fetch instruments not registered")
            return None

        async with self._lock:
            record = self._records.get(alpha_id)
            if record is None:
                return None

        try:
            yearly_data = await self._fetch_yearly(alpha_id)
        except (ConnectionError, OSError, TimeoutError) as exc:
            logger.warning("decay_detector: yearly fetch failed for %s: %s", alpha_id, exc)
            yearly_data = None

        yearly_sharpes: list[float] = []
        if yearly_data:
            for yd in yearly_data:
                s = yd.get("sharpe", 0.0)
                if isinstance(s, (int, float)) and math.isfinite(s):
                    yearly_sharpes.append(s)

        recent_sharpe = yearly_sharpes[-1] if yearly_sharpes else 0.0

        try:
            pnl_curve = await self._fetch_pnl(alpha_id)
        except (TimeoutError, aiohttp.ClientError, ConnectionError):  # noqa: SIM105
            pnl_curve = None

        garch_anomaly = False
        if pnl_curve and len(pnl_curve) >= 20 and self._estimate_garch is not None:
            try:
                garch_result = await self._estimate_garch(pnl_curve)
                if hasattr(garch_result, 'persistence'):
                    garch_anomaly = garch_result.persistence > 0.95
            except (ValueError, TypeError, RuntimeError) as exc:
                logger.debug("decay_detector: GARCH estimation failed for %s: %s", alpha_id, exc)

        async with self._lock:
            record = self._records.get(alpha_id)
            if record is None:
                return None

            record.ewma_sharpe = self._compute_ewma(record.ewma_sharpe, recent_sharpe)
            record._sharpe_history.append(recent_sharpe)

            if recent_sharpe > record.peak_sharpe:
                record.peak_sharpe = recent_sharpe

            record.garch_anomaly = garch_anomaly
            record.composite_score = self._compute_composite_score(record, yearly_sharpes)
            record.last_checked = datetime.now(UTC)

            new_level = self._classify_decay(record)

            if new_level == DecayLevel.NONE:
                record.consecutive_decay_checks = 0
            elif new_level == record.decay_level or new_level.value > record.decay_level.value:
                record.consecutive_decay_checks += 1
            elif new_level.value < record.decay_level.value:
                record.consecutive_decay_checks = max(0, record.consecutive_decay_checks - 1)

            confirmed_level = DecayLevel.NONE
            if new_level != DecayLevel.NONE and record.consecutive_decay_checks >= self.CONFIRMATION_CHECKS:
                confirmed_level = new_level
            elif record.decay_level != DecayLevel.NONE and new_level == DecayLevel.NONE:
                record.decay_level = DecayLevel.NONE
                record.consecutive_decay_checks = 0
                logger.info("decay_detector: alpha %s recovered, resetting decay level", alpha_id)

            if confirmed_level != DecayLevel.NONE and confirmed_level != record.decay_level:
                record.decay_level = confirmed_level
                reason = self._build_decay_reason(record)
                logger.warning(
                    "decay_detector: alpha %s decay=%s reason=%s composite=%.3f ewma_sharpe=%.3f",
                    alpha_id, confirmed_level.value, reason, record.composite_score, record.ewma_sharpe,
                )

                if self._on_decay_detected is not None:
                    try:
                        await self._on_decay_detected(alpha_id, confirmed_level, record, reason)
                    except (OSError, ValueError, RuntimeError) as exc:
                        logger.error("decay_detector: on_decay callback failed for %s: %s", alpha_id, exc)

                fp = self._record_decay_fingerprint(record, confirmed_level, reason)
                if fp:
                    self._decay_fingerprints.append(fp)

            return record

    def _build_decay_reason(self, record: AlphaDecayRecord) -> str:
        parts = [
            f"ewma_sharpe={record.ewma_sharpe:.2f}(init={record.initial_sharpe:.2f})",
            f"composite={record.composite_score:.3f}",
        ]
        if record.garch_anomaly:
            parts.append("garch_anomaly=True")
        if record.yearly_trend < 0.5:
            parts.append(f"yearly_trend={record.yearly_trend:.2f}")
        return "; ".join(parts)

    def _record_decay_fingerprint(
        self, record: AlphaDecayRecord, level: DecayLevel, reason: str,
    ) -> DecayFingerprint | None:
        fp = record.fingerprint
        if fp is None:
            return None
        is_perm = level == DecayLevel.DIR_HEAVY
        return DecayFingerprint(
            direction=getattr(fp, 'direction', record.direction) or record.direction,
            topology=getattr(fp, 'topology', ''),
            temporal=getattr(fp, 'temporal', ''),
            dataset=getattr(fp, 'dataset', ''),
            normalization=getattr(fp, 'normalization', ''),
            decay_level=level.value,
            reason=reason,
            confirmation_checks=record.consecutive_decay_checks,
            is_permanent=is_perm,
        )

    async def check_all_alphas(self) -> dict[str, AlphaDecayRecord]:
        self._check_count += 1
        check_id = f"decay_check_{self._check_count}"
        logger.info("%s: checking %d active alphas", check_id, len(self._records))

        alpha_ids: list[str] = []
        async with self._lock:
            alpha_ids = list(self._records.keys())

        semaphore = asyncio.Semaphore(3)
        results: dict[str, AlphaDecayRecord | None] = {}

        async def _check_one(aid: str) -> None:
            async with semaphore:
                results[aid] = await self.check_single_alpha(aid)

        tasks = [asyncio.create_task(_check_one(aid)) for aid in alpha_ids]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        valid_results: dict[str, AlphaDecayRecord] = {}
        for aid, record in results.items():
            if record is not None:
                valid_results[aid] = record

        decayed = sum(1 for r in valid_results.values() if r.decay_level != DecayLevel.NONE)
        logger.info("%s: checked %d alphas, %d decayed", check_id, len(valid_results), decayed)
        return valid_results

    async def get_decay_summary(self) -> dict[str, Any]:
        async with self._lock:
            total = len(self._records)
            levels: dict[str, int] = {}
            for record in self._records.values():
                lv = record.decay_level.value
                levels[lv] = levels.get(lv, 0) + 1
            blacklisted = self.get_blacklisted_directions()
            perm_blacklist = list(self._permanent_blacklist)
            return {
                "total_tracked": total,
                "by_level": levels,
                "blacklisted_directions": blacklisted,
                "permanent_blacklist": perm_blacklist,
                "decay_fingerprints_count": len(self._decay_fingerprints),
            }

    def build_decay_prompt_injection(self) -> str:
        all_dirs = self.get_blacklisted_directions()
        if not all_dirs:
            return ""

        recent_fps = self.get_decay_fingerprints(max_count=10)
        fp_lines = []
        for fp in recent_fps:
            fp_lines.append(
                f"  - direction={fp.direction}, topology={fp.topology}, "
                f"level={fp.decay_level}, reason={fp.reason}",
            )

        perm_dirs = [d for d in all_dirs if d in self._permanent_blacklist]
        temp_dirs = [d for d in all_dirs if d not in self._permanent_blacklist]

        parts: list[str] = []
        if perm_dirs:
            parts.append(f"\n[DECAY ALERT - PERMANENTLY BLACKLISTED DIRECTIONS: {', '.join(perm_dirs)}]")
            parts.append("DO NOT generate alphas in these directions. They have permanently decayed.")
        if temp_dirs:
            parts.append(f"\n[DECAY ALERT - TEMPORARILY BLACKLISTED DIRECTIONS: {', '.join(temp_dirs)}]")
            parts.append("Avoid these directions for now. Explore alternative strategies.")
        if fp_lines:
            parts.append("\nRecent decay fingerprints (learn from these failures):")
            parts.extend(fp_lines)

        return "\n".join(parts)


async def create_alpha_decay_handler(
    loop_state_module,
    multi_agent_orchestrator=None,
) -> Callable[[str, DecayLevel, AlphaDecayRecord, str], Awaitable[None]]:
    """Factory that creates a decay handler callback wired into the existing system.

    The returned callback implements the 4-tier decay response:
      L1 WARNING:     UCB weight -20%, increase check frequency
      L2 FACTOR_DECAY: Unlist from success pool, log fingerprint
      L3 DIR_LIGHT:   UCB weight -50%, blacklist 30d, distill fingerprint, inject prompt
      L4 DIR_HEAVY:   Permanent blacklist, remove from UCB, force other directions
    """
    async def _on_decay(
        alpha_id: str,
        level: DecayLevel,
        record: AlphaDecayRecord,
        reason: str,
    ) -> None:
        direction = record.direction
        if not direction and record.fingerprint:
            direction = getattr(record.fingerprint, 'direction', '')

        mab = getattr(loop_state_module, '_mab', None)
        evo_db = getattr(loop_state_module, '_evo_db', None)
        success_lib = getattr(loop_state_module, '_success_lib', None)
        experience_distiller = getattr(loop_state_module, '_experience_distiller', None)
        false_positive_patterns = getattr(loop_state_module, '_false_positive_patterns', set())

        if level == DecayLevel.WARNING:
            if mab is not None and direction:
                try:
                    mab.update(direction, penalty=0.2)
                    logger.info("decay_handler: L1 WARNING - UCB weight -20%% for direction=%s", direction)
                except (KeyError, AttributeError, OSError) as exc:
                    logger.warning("decay_handler: MAB penalty failed: %s", exc)

        elif level == DecayLevel.FACTOR_DECAY:
            if success_lib is not None:
                pass
            _add_to_false_positive_if_possible(false_positive_patterns, record)
            logger.info("decay_handler: L2 FACTOR_DECAY - alpha %s unlisted, fingerprint recorded", alpha_id)

        elif level == DecayLevel.DIR_LIGHT:
            if mab is not None and direction:
                try:
                    mab.update(direction, penalty=0.5)
                    logger.info("decay_handler: L3 DIR_LIGHT - UCB weight -50%% for direction=%s", direction)
                except (KeyError, AttributeError, OSError) as exc:
                    logger.warning("decay_handler: MAB penalty failed: %s", exc)

            _add_to_false_positive_if_possible(false_positive_patterns, record)

            if experience_distiller is not None:
                try:
                    fingerprint_dict = {
                        "direction": direction or "",
                        "topology": record.fingerprint.topology if record.fingerprint else "",
                        "temporal": record.fingerprint.temporal if record.fingerprint else "",
                        "dataset": record.fingerprint.dataset if record.fingerprint else "",
                        "decay_level": level.value,
                        "reason": reason,
                        "expression_preview": record.expression[:120],
                    }
                    await experience_distiller._store_pattern(
                        "alpha_decay",
                        {"fingerprint": fingerprint_dict, "decay_level": level.value},
                        strategy_id=direction,
                    )
                    logger.info("decay_handler: L3 DIR_LIGHT - decay fingerprint distilled for direction=%s", direction)
                except (OSError, ValueError, RuntimeError) as exc:
                    logger.warning("decay_handler: experience distillation failed: %s", exc)

            if multi_agent_orchestrator is not None:
                try:
                    blacklist_dirs = [direction] if direction else []
                    multi_agent_orchestrator._decay_blacklisted_dirs = blacklist_dirs
                    logger.info("decay_handler: L3 DIR_LIGHT - blacklist prompt injected for IdeaAgent")
                except (AttributeError, OSError, RuntimeError) as exc:
                    logger.warning("decay_handler: prompt injection failed: %s", exc)

            feature_map = getattr(loop_state_module, '_feature_map', None)
            if feature_map is not None and direction:
                try:
                    for th in ["short", "medium", "long"]:
                        for mech in ["signal", "normalized", "conditional", "interaction"]:
                            feature_map.mark_cell_decay(
                                direction=direction, time_horizon=th, mechanism=mech,
                                decay_level="L3_DIR_LIGHT",
                            )
                    logger.info("decay_handler: L3 DIR_LIGHT - FeatureMap cells observing for direction=%s", direction)
                except (OSError, ValueError, RuntimeError) as exc:
                    logger.warning("decay_handler: FeatureMap decay marking failed: %s", exc)

        elif level == DecayLevel.DIR_HEAVY:
            if mab is not None and direction and hasattr(mab, '_outer'):
                try:
                    outer = mab._outer
                    if direction in outer._arms:
                        del outer._arms[direction]
                        logger.info("decay_handler: L4 DIR_HEAVY - permanently removed direction=%s from UCB", direction)
                except (KeyError, AttributeError, OSError) as exc:
                    logger.warning("decay_handler: MAB arm removal failed: %s", exc)

            _add_to_false_positive_if_possible(false_positive_patterns, record)

            if experience_distiller is not None:
                try:
                    fingerprint_dict = {
                        "direction": direction or "",
                        "topology": record.fingerprint.topology if record.fingerprint else "",
                        "temporal": record.fingerprint.temporal if record.fingerprint else "",
                        "dataset": record.fingerprint.dataset if record.fingerprint else "",
                        "decay_level": level.value,
                        "reason": reason,
                        "is_permanent": True,
                    }
                    await experience_distiller._store_pattern(
                        "alpha_decay_permanent",
                        {"fingerprint": fingerprint_dict, "decay_level": "DIR_HEAVY"},
                        strategy_id=direction,
                    )
                    logger.info("decay_handler: L4 DIR_HEAVY - permanent decay fingerprint distilled")
                except (OSError, ValueError, RuntimeError) as exc:
                    logger.warning("decay_handler: permanent distillation failed: %s", exc)

            feature_map = getattr(loop_state_module, '_feature_map', None)
            if feature_map is not None and direction:
                try:
                    for th in ["short", "medium", "long"]:
                        for mech in ["signal", "normalized", "conditional", "interaction"]:
                            feature_map.mark_cell_decay(
                                direction=direction, time_horizon=th, mechanism=mech,
                                decay_level="L4_DIR_HEAVY",
                            )
                    logger.info("decay_handler: L4 DIR_HEAVY - FeatureMap cells blacklisted for direction=%s", direction)
                except (OSError, ValueError, RuntimeError) as exc:
                    logger.warning("decay_handler: FeatureMap blacklist failed: %s", exc)

    return _on_decay


def _add_to_false_positive_if_possible(container: set, record: AlphaDecayRecord) -> None:
    if container is not None and record.fingerprint and record.fingerprint.direction:
        marker = f"decay:{record.fingerprint.direction}:{record.fingerprint.topology}"
        container.add(marker)
