from __future__ import annotations

import asyncio
import itertools
import logging
import math
from collections import deque

logger = logging.getLogger(__name__)

MAX_POOL_SIZE = 100
MAX_BRAIN_SLOTS = 3
GENERATION_THRESHOLD = 10
BACKLOG_IMPROVEMENT_THRESHOLD = 15

_slot_counter = itertools.count()


class AlphaCachePool:
    UCB_C_EXPLORE = 0.7
    UCB_C_STABLE = 0.5
    UCB_C_MATURE = 0.3
    UCB_STAGE_EXPLORE = 50
    UCB_STAGE_STABLE = 200

    def __init__(self, max_size: int = MAX_POOL_SIZE, max_slots: int = MAX_BRAIN_SLOTS):
        self._lock = asyncio.Lock()
        self._high_queue: deque[str] = deque()
        self._normal_queue: deque[str] = deque()
        self._active_slots: dict[str, str] = {}
        self._max_size = max_size
        self._max_slots = max_slots
        self._alpha_direction: dict[str, str] = {}
        self._alpha_source_queue: dict[str, str] = {}
        self._direction_stats: dict[str, dict] = {}
        self._slot_released = asyncio.Event()
        self._generation_green_light = asyncio.Event()
        self._generation_threshold = GENERATION_THRESHOLD
        self._update_green_light()

    def _get_ucb_c_value(self) -> float:
        total = sum(s.get("n", 0) for s in self._direction_stats.values())
        if total < self.UCB_STAGE_EXPLORE:
            return self.UCB_C_EXPLORE
        if total < self.UCB_STAGE_STABLE:
            return self.UCB_C_STABLE
        return self.UCB_C_MATURE

    def _ucb_score(self, direction: str, avg_sharpe: float, n: int) -> float:
        if n == 0:
            return float("inf")
        total_n = sum(s.get("n", 0) for s in self._direction_stats.values())
        total_n = max(total_n, 1)
        c = self._get_ucb_c_value()
        exploration_bonus = c * math.sqrt(math.log(total_n) / n)
        return avg_sharpe + exploration_bonus

    def _compute_direction_ucb_scores(self) -> dict[str, float]:
        scores: dict[str, float] = {}
        for direction, stats in self._direction_stats.items():
            n = stats.get("n", 0)
            avg_sharpe = stats.get("avg_sharpe", 0.0)
            scores[direction] = self._ucb_score(direction, avg_sharpe, n)
        return scores

    def record_submission_result(self, direction: str, sharpe: float) -> None:
        if not direction:
            return
        stats = self._direction_stats.setdefault(direction, {"n": 0, "sum_sharpe": 0.0, "avg_sharpe": 0.0})
        stats["n"] += 1
        stats["sum_sharpe"] += sharpe
        stats["avg_sharpe"] = stats["sum_sharpe"] / stats["n"]

    async def enqueue(self, alpha_id: str, priority: str = "normal", direction: str = "") -> None:
        async with self._lock:
            if self.is_full:
                logger.warning("pipeline: pool full (%d items), dropping alpha %s", len(self), alpha_id)
                return
            if direction:
                self._alpha_direction[alpha_id] = direction
            if priority == "high":
                self._high_queue.append(alpha_id)
                self._alpha_source_queue[alpha_id] = "high"
            else:
                self._normal_queue.append(alpha_id)
                self._alpha_source_queue[alpha_id] = "normal"
            pool_len = len(self)
            if pool_len >= 30:
                logger.warning(
                    "pipeline: BACKPRESSURE CRITICAL — pool_size=%d (queued=%d active=%d)",
                    pool_len,
                    len(self._high_queue) + len(self._normal_queue),
                    len(self._active_slots),
                )
            elif pool_len >= 15:
                logger.warning(
                    "pipeline: BACKPRESSURE HIGH — pool_size=%d (queued=%d active=%d)",
                    pool_len,
                    len(self._high_queue) + len(self._normal_queue),
                    len(self._active_slots),
                )
            else:
                logger.info(
                    "pipeline: enqueued alpha %s (priority=%s, dir=%s, pool_size=%d)",
                    alpha_id,
                    priority,
                    direction or "?",
                    pool_len,
                )
        self._update_green_light()

    @property
    def is_full(self) -> bool:
        return len(self) >= self._max_size

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    def __len__(self) -> int:
        return len(self._high_queue) + len(self._normal_queue) + len(self._active_slots)

    def available_slots(self) -> int:
        return self._max_slots - len(self._active_slots)

    async def release_slot(self, alpha_id: str) -> None:
        async with self._lock:
            to_remove = None
            for slot_id, aid in self._active_slots.items():
                if aid == alpha_id:
                    to_remove = slot_id
                    break
            if to_remove:
                del self._active_slots[to_remove]
                logger.info(
                    "pipeline: released %s for alpha %s (active=%d/%d)",
                    to_remove,
                    alpha_id,
                    len(self._active_slots),
                    self._max_slots,
                )
        self._slot_released.set()
        self._update_green_light()

    def _update_green_light(self) -> None:
        queued = len(self._high_queue) + len(self._normal_queue)
        active = len(self._active_slots)
        total = queued + active
        if total < self._generation_threshold and self.available_slots() > 0:
            self._generation_green_light.set()
        else:
            self._generation_green_light.clear()

    async def await_generation_slot(self, timeout: float = 10.0) -> bool:
        try:
            await asyncio.wait_for(self._generation_green_light.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False

    def generation_allowed(self) -> bool:
        total = len(self._high_queue) + len(self._normal_queue) + len(self._active_slots)
        return total < self._generation_threshold and self.available_slots() > 0

    def should_degrade_to_improvement(self) -> bool:
        queued = len(self._high_queue) + len(self._normal_queue)
        return self.available_slots() <= 0 and queued >= BACKLOG_IMPROVEMENT_THRESHOLD

    async def next_to_submit(self) -> str | None:
        async with self._lock:
            if self.available_slots() <= 0 or self.is_empty:
                return None

            if len(self._high_queue) + len(self._normal_queue) > 1 and self._direction_stats:
                ucb_scores = self._compute_direction_ucb_scores()
                if ucb_scores:
                    all_ids = list(self._high_queue) + list(self._normal_queue)
                    scored_ids = sorted(
                        all_ids,
                        key=lambda aid: ucb_scores.get(self._alpha_direction.get(aid, ""), 0.0),
                        reverse=True,
                    )
                    alpha_id = scored_ids[0]
                    for q in [self._high_queue, self._normal_queue]:
                        try:
                            q.remove(alpha_id)
                            break
                        except ValueError:
                            continue
                    logger.info(
                        "pipeline: UCB sorted selection: alpha %s dir=%s score=%.4f (queue_size=%d)",
                        alpha_id,
                        self._alpha_direction.get(alpha_id, "?"),
                        ucb_scores.get(self._alpha_direction.get(alpha_id, ""), 0.0),
                        len(all_ids),
                    )
            elif self._high_queue:
                alpha_id = self._high_queue.popleft()
            elif self._normal_queue:
                alpha_id = self._normal_queue.popleft()
            else:
                return None

            logger.info("pipeline: dequeued alpha %s (pool_size=%d)", alpha_id, len(self))
            if self.available_slots() <= 0:
                logger.warning("pipeline: no available slots for alpha %s", alpha_id)
                source = self._alpha_source_queue.get(alpha_id, "normal")
                if source == "high":
                    self._high_queue.append(alpha_id)
                else:
                    self._normal_queue.append(alpha_id)
                return None
            slot_id = f"slot_{next(_slot_counter)}"
            self._active_slots[slot_id] = alpha_id
            logger.info(
                "pipeline: alpha %s assigned to %s (active=%d/%d)",
                alpha_id,
                slot_id,
                len(self._active_slots),
                self._max_slots,
            )
            return alpha_id
