from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from openalpha_brain.config.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SignalSource:
    """[Brief description of class purpose.]"""

    source_name: str
    score: float
    weight: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArbitrationResult:
    """[Brief description of class purpose.]"""

    item_id: str
    final_score: float
    signal_breakdown: dict[str, float]
    confidence: float


DEFAULT_WEIGHTS = {
    "rag": 0.2,
    "mab": 0.3,
    "association": 0.2,
    "whitelist": 0.15,
    "market": 0.15,
}

WEIGHT_LOWER_BOUND = getattr(settings, "SIGNAL_ARBITER_WEIGHT_LOWER_BOUND", 0.05)
WEIGHT_UPPER_BOUND = getattr(settings, "SIGNAL_ARBITER_WEIGHT_UPPER_BOUND", 0.5)
WEIGHT_BOUNDS = (WEIGHT_LOWER_BOUND, WEIGHT_UPPER_BOUND)
ADJUSTMENT_STEP = getattr(settings, "SIGNAL_ARBITER_ADJUSTMENT_STEP", 0.05)
ADJUSTMENT_INTERVAL = getattr(settings, "SIGNAL_ARBITER_ADJUSTMENT_INTERVAL", 5)
SUCCESS_RATE_TRACKER_WINDOW = getattr(settings, "SIGNAL_ARBITER_TRACKER_WINDOW", 20)
DEFAULT_SCORE = getattr(settings, "SIGNAL_ARBITER_DEFAULT_SCORE", 0.5)
LOW_SUCCESS_THRESHOLD = getattr(settings, "SIGNAL_ARBITER_LOW_SUCCESS_THRESHOLD", 0.3)
HIGH_SUCCESS_THRESHOLD = getattr(settings, "SIGNAL_ARBITER_HIGH_SUCCESS_THRESHOLD", 0.6)


class SignalArbiter:
    """[Brief description of class purpose.]"""

    def __init__(self, weights: dict[str, float] | None = None, success_rate_tracker: deque | None = None):
        """[Brief description of function purpose.]

        Args:
            weights (Optional[dict[str, float]]): [Description]
            success_rate_tracker (Optional[deque]): [Description]
        """
        self._weights = dict(weights or DEFAULT_WEIGHTS)
        self._adjustment_counter = 0
        self._success_rate_tracker = success_rate_tracker

    @property
    def weights(self) -> dict[str, float]:
        """[Brief description of function purpose.]

        Returns:
            dict[str, float]: [Description]
        """
        return dict(self._weights)

    @property
    def adjustment_counter(self) -> int:
        return self._adjustment_counter

    @property
    def success_rate_tracker(self) -> deque | None:
        return self._success_rate_tracker

    def set_weight(self, source: str, value: float) -> None:
        """[Brief description of function purpose.]

        Args:
            source (str): [Description]
            value (float): [Description]

        Returns:
            None: [Description]
        """
        lo, hi = WEIGHT_BOUNDS
        self._weights[source] = max(lo, min(hi, value))

    def arbitrate(self, item_id: str, signals: list[SignalSource]) -> ArbitrationResult:
        """[Brief description of function purpose.]

        Args:
            item_id (str): [Description]
            signals (list[SignalSource]): [Description]

        Returns:
            ArbitrationResult: [Description]
        """
        if not signals:
            # When no signals are provided (or all signals have score 0.0),
            # final_score and confidence are both 0.0 because there is no
            # evidence to produce a meaningful score or confidence level.
            return ArbitrationResult(item_id=item_id, final_score=0.0, signal_breakdown={}, confidence=0.0)
        breakdown = {}
        weighted_sum = 0.0
        total_weight = 0.0
        for sig in signals:
            w = self._weights.get(sig.source_name, 0.0)
            contribution = w * sig.weight * sig.score
            breakdown[sig.source_name] = contribution
            weighted_sum += contribution
            total_weight += w
        if total_weight < 1e-9:
            logger.warning("total_weight near zero in arbitrate for item %s, returning default score 0.5", item_id)
            return ArbitrationResult(
                item_id=item_id, final_score=DEFAULT_SCORE, signal_breakdown=breakdown, confidence=0.0
            )
        final_score = weighted_sum / total_weight
        confidence = min(1.0, total_weight / max(sum(self._weights.values()), 1e-9)) if self._weights else 0.0
        return ArbitrationResult(
            item_id=item_id, final_score=final_score, signal_breakdown=breakdown, confidence=confidence
        )

    def rank_fields(self, field_signals: dict[str, list[SignalSource]], top_k: int = 0) -> list[ArbitrationResult]:
        """[Brief description of function purpose.]

        Args:
            field_signals (dict[str, list[SignalSource]]): [Description]
            top_k (int): [Description]

        Returns:
            list[ArbitrationResult]: [Description]
        """
        results = []
        for fid, signals in field_signals.items():
            results.append(self.arbitrate(fid, signals))
        max_score = max((r.final_score for r in results), default=0.0)
        if max_score > 0:
            results = [
                ArbitrationResult(
                    item_id=r.item_id,
                    final_score=r.final_score / max_score,
                    signal_breakdown=r.signal_breakdown,
                    confidence=r.confidence,
                )
                for r in results
            ]
        else:
            logger.warning("max_score is 0 in rank_fields, skipping normalization")
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:top_k] if top_k > 0 else results

    def rank_operators(self, op_signals: dict[str, list[SignalSource]], top_k: int = 0) -> list[ArbitrationResult]:
        """[Brief description of function purpose.]

        Args:
            op_signals (dict[str, list[SignalSource]]): [Description]
            top_k (int): [Description]

        Returns:
            list[ArbitrationResult]: [Description]
        """
        results = []
        for oid, signals in op_signals.items():
            results.append(self.arbitrate(oid, signals))
        max_score = max((r.final_score for r in results), default=0.0)
        if max_score > 0:
            results = [
                ArbitrationResult(
                    item_id=r.item_id,
                    final_score=r.final_score / max_score,
                    signal_breakdown=r.signal_breakdown,
                    confidence=r.confidence,
                )
                for r in results
            ]
        else:
            logger.warning("max_score is 0 in rank_operators, skipping normalization")
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:top_k] if top_k > 0 else results

    def adjust_weights(self, success_rate: float) -> dict[str, float]:
        """[Brief description of function purpose.]

        Args:
            success_rate (float): [Description]

        Returns:
            dict[str, float]: [Description]
        """
        self._adjustment_counter += 1
        if self._adjustment_counter % ADJUSTMENT_INTERVAL != 0:
            return self._weights
        adjusted = False
        if success_rate < LOW_SUCCESS_THRESHOLD:
            self._adjust("mab", ADJUSTMENT_STEP)
            self._adjust("market", -ADJUSTMENT_STEP)
            adjusted = True
        elif success_rate > HIGH_SUCCESS_THRESHOLD:
            self._adjust("market", ADJUSTMENT_STEP)
            self._adjust("mab", -ADJUSTMENT_STEP)
            adjusted = True
        if adjusted:
            self._normalize_weights()
        return self._weights

    def _adjust(self, source: str, delta: float) -> None:
        """[Brief description of function purpose.]

        Args:
            source (str): [Description]
            delta (float): [Description]

        Returns:
            None: [Description]
        """
        new_val = self._weights.get(source, 0.0) + delta
        self.set_weight(source, new_val)

    def _normalize_weights(self) -> None:
        """[Brief description of function purpose.]

        Returns:
            None: [Description]
        """
        total = sum(self._weights.values())
        if total > 0:
            for key in self._weights:
                self._weights[key] /= total

    def health_check(self) -> dict[str, Any]:
        """[Brief description of function purpose.]

        Returns:
            dict[str, Any]: [Description]
        """
        return {
            "module": "SignalArbiter",
            "status": "active",
            "weights": dict(self._weights),
            "weight_sum": sum(self._weights.values()),
            "adjustment_counter": self._adjustment_counter,
            "last_adjustment": self._adjustment_counter * ADJUSTMENT_INTERVAL,
        }

    def to_dict(self) -> dict[str, Any]:
        """[Brief description of function purpose.]

        Returns:
            dict[str, Any]: [Description]
        """
        return {
            "weights": self._weights,
            "adjustment_counter": self._adjustment_counter,
            "success_rate_tracker": list(self._success_rate_tracker) if self._success_rate_tracker else [],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], success_rate_tracker: deque | None = None) -> SignalArbiter:
        """[Brief description of function purpose.]

        Args:
            data (dict[str, Any]): [Description]
            success_rate_tracker (Optional[deque]): [Description]

        Returns:
            'SignalArbiter': [Description]
        """
        tracker = success_rate_tracker
        if tracker is None and data.get("success_rate_tracker"):
            tracker = deque(data.get("success_rate_tracker", []), maxlen=SUCCESS_RATE_TRACKER_WINDOW)
        arbiter = cls(weights=data.get("weights"), success_rate_tracker=tracker)
        arbiter._adjustment_counter = data.get("adjustment_counter", 0)
        return arbiter

    async def rank_with_adapters(
        self, field_adapters: list, op_adapters: list, top_k_fields: int = 0, top_k_ops: int = 0
    ) -> tuple[list[ArbitrationResult], list[ArbitrationResult]]:
        """[Brief description of function purpose.]

        Args:
            field_adapters (list): [Description]
            op_adapters (list): [Description]
            top_k_fields (int): [Description]
            top_k_ops (int): [Description]

        Returns:
            tuple[list[ArbitrationResult], list[ArbitrationResult]]: [Description]
        """
        t0 = time.monotonic()

        async def _fetch_adapter_fields(adapter):
            try:
                return await asyncio.to_thread(adapter.adapt_fields)
            except (OSError, ValueError, RuntimeError) as e:
                logger.warning("[SignalArbiter] adapt_fields 失敗 (%s): %s", type(adapter).__name__, e)
                return {}

        async def _fetch_adapter_ops(adapter):
            try:
                return await asyncio.to_thread(adapter.adapt_operators)
            except (OSError, ValueError, RuntimeError) as e:
                logger.warning("[SignalArbiter] adapt_operators 失敗 (%s): %s", type(adapter).__name__, e)
                return {}

        field_tasks = [_fetch_adapter_fields(a) for a in field_adapters]
        op_tasks = [_fetch_adapter_ops(a) for a in op_adapters]

        field_results = await asyncio.gather(*field_tasks, return_exceptions=True)
        op_results = await asyncio.gather(*op_tasks, return_exceptions=True)

        field_signals: dict[str, list[SignalSource]] = {}
        for result in field_results:
            if isinstance(result, Exception):
                continue
            if not isinstance(result, dict):
                continue
            for fid, signals in result.items():
                field_signals.setdefault(fid, []).extend(signals)

        op_signals: dict[str, list[SignalSource]] = {}
        for result in op_results:
            if isinstance(result, Exception):
                continue
            if not isinstance(result, dict):
                continue
            for oid, signals in result.items():
                op_signals.setdefault(oid, []).extend(signals)

        elapsed = time.monotonic() - t0
        logger.debug(
            "[SignalArbiter] 並行擷取完成 — %d 個 field adapter, %d 個 op adapter, 耗時 %.3fs",
            len(field_adapters),
            len(op_adapters),
            elapsed,
        )

        return self.rank_fields(field_signals, top_k_fields), self.rank_operators(op_signals, top_k_ops)


class RAGSignalAdapter:
    """[Brief description of class purpose.]"""

    def __init__(self, rag_result: dict[str, Any]) -> None:
        """[Brief description of function purpose.]

        Args:
            rag_result (dict[str, Any]): [Description]

        Returns:
            None: [Description]
        """
        self._rag_result = rag_result

    @property
    def rag_result(self) -> dict[str, Any]:
        return dict(self._rag_result)

    def adapt_fields(self) -> dict[str, list[SignalSource]]:
        """[Brief description of function purpose.]

        Returns:
            dict[str, list[SignalSource]]: [Description]
        """
        fields = self._rag_result.get("fields", [])
        if not fields:
            return {}
        max_score = max((f.get("score", 0.0) for f in fields), default=0.0)
        result: dict[str, list[SignalSource]] = {}
        for f in fields:
            fid = f.get("id", "")
            if not fid:
                continue
            score = f.get("score", 0.0) / max_score if max_score > 0 else 0.0
            result[fid] = [SignalSource(source_name="rag", score=score, weight=1.0)]
        return result

    def adapt_operators(self) -> dict[str, list[SignalSource]]:
        """[Brief description of function purpose.]

        Returns:
            dict[str, list[SignalSource]]: [Description]
        """
        ops = self._rag_result.get("operators", [])
        if not ops:
            return {}
        max_score = max((o.get("score", 0.0) for o in ops), default=0.0)
        result: dict[str, list[SignalSource]] = {}
        for o in ops:
            oid = o.get("id", "")
            if not oid:
                continue
            score = o.get("score", 0.0) / max_score if max_score > 0 else 0.0
            result[oid] = [SignalSource(source_name="rag", score=score, weight=1.0)]
        return result


class MABSignalAdapter:
    """[Brief description of class purpose.]"""

    def __init__(self, mab: Any) -> None:
        """[Brief description of function purpose.]

        Args:
            mab (Any): [Description]

        Returns:
            None: [Description]
        """
        self._mab = mab

    @property
    def mab(self) -> Any:
        return self._mab

    def adapt_fields(self) -> dict[str, list[SignalSource]]:
        """[Brief description of function purpose.]

        Returns:
            dict[str, list[SignalSource]]: [Description]
        """
        if self._mab is None:
            return {}
        result: dict[str, list[SignalSource]] = {}
        for arm_id, expectation in self._mab.get_field_expectations().items():
            result.setdefault(arm_id, []).append(
                SignalSource(source_name="mab", score=expectation, weight=1.0),
            )
        return result

    def adapt_operators(self) -> dict[str, list[SignalSource]]:
        """[Brief description of function purpose.]

        Returns:
            dict[str, list[SignalSource]]: [Description]
        """
        if self._mab is None:
            return {}
        result: dict[str, list[SignalSource]] = {}
        for arm_id, expectation in self._mab.get_operator_expectations().items():
            result.setdefault(arm_id, []).append(
                SignalSource(source_name="mab", score=expectation, weight=1.0),
            )
        return result


class AssociationSignalAdapter:
    """[Brief description of class purpose.]"""

    def __init__(self, association: Any, current_operator: str, current_field: str) -> None:
        """[Brief description of function purpose.]

        Args:
            association (Any): [Description]
            current_operator (str): [Description]
            current_field (str): [Description]

        Returns:
            None: [Description]
        """
        self._association = association
        self._current_operator = current_operator
        self._current_field = current_field

    @property
    def association(self) -> Any:
        return self._association

    @property
    def current_operator(self) -> str:
        return self._current_operator

    @property
    def current_field(self) -> str:
        return self._current_field

    def adapt_fields(self) -> dict[str, list[SignalSource]]:
        """[Brief description of function purpose.]

        Returns:
            dict[str, list[SignalSource]]: [Description]
        """
        if self._association is None:
            return {}
        top_fields = self._association.get_top_fields(self._current_operator)
        result: dict[str, list[SignalSource]] = {}
        for fid, score in top_fields:
            result[fid] = [SignalSource(source_name="association", score=score, weight=1.0)]
        return result

    def adapt_operators(self) -> dict[str, list[SignalSource]]:
        """[Brief description of function purpose.]

        Returns:
            dict[str, list[SignalSource]]: [Description]
        """
        if self._association is None:
            return {}
        top_ops = self._association.get_top_operators(self._current_field)
        result: dict[str, list[SignalSource]] = {}
        for oid, score in top_ops:
            result[oid] = [SignalSource(source_name="association", score=score, weight=1.0)]
        return result


class WhitelistSignalAdapter:
    """[Brief description of class purpose.]"""

    def __init__(self, whitelist_mgr: Any) -> None:
        """[Brief description of function purpose.]

        Args:
            whitelist_mgr (Any): [Description]

        Returns:
            None: [Description]
        """
        self._whitelist_mgr = whitelist_mgr

    @property
    def whitelist_mgr(self) -> Any:
        return self._whitelist_mgr

    def adapt_fields(self) -> dict[str, list[SignalSource]]:
        """[Brief description of function purpose.]

        Returns:
            dict[str, list[SignalSource]]: [Description]
        """
        if self._whitelist_mgr is None:
            return {}
        result: dict[str, list[SignalSource]] = {}
        solidified = self._whitelist_mgr.solidified_fields
        for fid, arm in solidified.items():
            result[fid] = [SignalSource(source_name="whitelist", score=arm.expectation, weight=1.0)]
        return result

    def adapt_operators(self) -> dict[str, list[SignalSource]]:
        """[Brief description of function purpose.]

        Returns:
            dict[str, list[SignalSource]]: [Description]
        """
        if self._whitelist_mgr is None:
            return {}
        return {}


class MarketSignalAdapter:
    """[Brief description of class purpose.]"""

    def __init__(
        self, market_state: Any, direction: str, field_ids: list[str] | None = None, op_ids: list[str] | None = None
    ) -> None:
        """[Brief description of function purpose.]

        Args:
            market_state (Any): [Description]
            direction (str): [Description]
            field_ids (list[str] | None): [Description]
            op_ids (list[str] | None): [Description]

        Returns:
            None: [Description]
        """
        self._market_state = market_state
        self._direction = direction
        self._field_ids = field_ids or []
        self._op_ids = op_ids or []

    @property
    def field_ids(self) -> list[str]:
        return list(self._field_ids)

    @property
    def op_ids(self) -> list[str]:
        return list(self._op_ids)

    def _get_direction_score(self) -> float:
        """[Brief description of function purpose.]

        Returns:
            float: [Description]
        """
        if self._market_state is None:
            return 0.5
        summary = self._market_state.get_market_state_summary()
        avg_sharpes = summary.get("avg_sharpes_by_direction", {})
        if not avg_sharpes:
            return 0.5
        max_sharpe = max(abs(v) for v in avg_sharpes.values()) if avg_sharpes else 0.0
        if max_sharpe <= 0:
            return 0.5
        direction_score = avg_sharpes.get(self._direction, 0.0)
        return max(0.0, min(1.0, (direction_score / max_sharpe + 1.0) / 2.0))

    def adapt_fields(self) -> dict[str, list[SignalSource]]:
        """[Brief description of function purpose.]

        Returns:
            dict[str, list[SignalSource]]: [Description]
        """
        score = self._get_direction_score()
        if not self._field_ids:
            return {}
        result: dict[str, list[SignalSource]] = {}
        for fid in self._field_ids:
            result[fid] = [SignalSource(source_name="market", score=score, weight=1.0)]
        return result

    def adapt_operators(self) -> dict[str, list[SignalSource]]:
        """[Brief description of function purpose.]

        Returns:
            dict[str, list[SignalSource]]: [Description]
        """
        score = self._get_direction_score()
        if not self._op_ids:
            return {}
        result: dict[str, list[SignalSource]] = {}
        for oid in self._op_ids:
            result[oid] = [SignalSource(source_name="market", score=score, weight=1.0)]
        return result
