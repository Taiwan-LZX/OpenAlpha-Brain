"""
OpenAlpha-Brain — Async Pipeline System

四大組件 + 頂層編排器：
  1. SubmissionSlotManager — BRAIN API 提交槽管理（最多 3 並發）
  2. AlphaQueue — Alpha 排隊系統（優先級隊列）
  3. ImprovementWorkerPool — 自學習改進工作池
  4. ResourceDispatcher — LLM/向量模型動態分配器
  5. PipelineOrchestrator — 頂層流水線編排器

流程：
  Generator → AlphaQueue → [Slot Available?] → Submit to BRAIN
                                    ↓ No
                              ImprovementWorkerPool (改善高分待處理 alpha)
                                    ↓
                              Re-enqueue improved alphas
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from heapq import heapify, heappop, heappush
from typing import Any

from openalpha_brain.config.config import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 組件 1: SubmissionSlotManager — BRAIN API 提交槽管理器
# ═══════════════════════════════════════════════════════════════════════════════


class SlotState(Enum):
    """提交槽位的狀態機狀態。"""

    IDLE = auto()
    SUBMITTING = auto()
    POLLING = auto()
    PROCESSING = auto()


@dataclass
class SubmissionSlot:
    """單個 BRAIN 提交槽位，追蹤完整生命週期。"""

    id: int
    state: SlotState = SlotState.IDLE
    alpha_id: str = ""
    expression: str = ""
    submitted_at: float = 0.0
    result: dict | None = None
    task: asyncio.Task | None = None


@dataclass
class _PendingSubmission:
    """等待中的提交請求，按優先級排序。"""

    priority: float
    counter: int
    expression: str
    simulation_payload: dict | None = None
    metadata: dict = field(default_factory=dict)
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())


class SubmissionSlotManager:
    """BRAIN API 提交槽管理器。

    特性：
      - 最多 max_concurrent 個並發槽位（預設 3）
      - 一個槽完成後立即從 queue 取下一個補上
      - 支援優先級提交（高 Sharpe 預估值的優先）
      - 回調機制：完成/失敗/超時 都有對應回調
    """

    def __init__(
        self,
        max_concurrent: int = 3,
        submit_fn: Callable[..., Awaitable[dict]] | None = None,
        poll_fn: Callable[..., Awaitable[dict]] | None = None,
    ):
        self.max_concurrent = max_concurrent
        self._submit_fn = submit_fn
        self._poll_fn = poll_fn
        self._slots: list[SubmissionSlot] = [SubmissionSlot(id=i) for i in range(max_concurrent)]
        self._wait_queue: list[_PendingSubmission] = []
        self._counter = 0
        self._lock = asyncio.Lock()
        self._running = False
        self._on_complete_callback: Callable[[SubmissionSlot, dict], Awaitable[None]] | None = None
        self._on_fail_callback: Callable[[SubmissionSlot, Exception], Awaitable[None]] | None = None
        self._stats = {
            "total_submitted": 0,
            "total_completed": 0,
            "total_failed": 0,
            "total_timed_out": 0,
        }

    def set_callbacks(
        self,
        on_complete: Callable[[SubmissionSlot, dict], Awaitable[None]] | None = None,
        on_fail: Callable[[SubmissionSlot, Exception], Awaitable[None]] | None = None,
    ) -> None:
        """設置槽位完成/失敗的回調函數。"""
        self._on_complete_callback = on_complete
        self._on_fail_callback = on_fail

    async def submit(
        self,
        expression: str,
        priority: float = 0.0,
        simulation_payload: dict | None = None,
        metadata: dict | None = None,
        timeout: float = 600.0,
    ) -> str:
        submission_id = f"sub-{uuid.uuid4().hex[:8]}"

        async with self._lock:
            self._counter += 1
            pending = _PendingSubmission(
                priority=-priority,
                counter=self._counter,
                expression=expression,
                simulation_payload=simulation_payload,
                metadata=metadata or {},
            )
            pending.metadata["submission_id"] = submission_id
            heappush(self._wait_queue, pending)

        asyncio.create_task(self._try_dispatch())
        return submission_id

    async def _try_dispatch(self) -> None:
        """嘗試從等待佇列中取出任務分發到空閒槽位。"""
        async with self._lock:
            while self._wait_queue:
                idle_slot = next((s for s in self._slots if s.state == SlotState.IDLE), None)
                if idle_slot is None:
                    break
                pending = heappop(self._wait_queue)
                idle_slot.state = SlotState.SUBMITTING
                idle_slot.expression = pending.expression
                idle_slot.submitted_at = time.monotonic()
                if pending.simulation_payload:
                    idle_slot.alpha_id = pending.simulation_payload.get("alpha_id", "")
                slot_task = asyncio.create_task(self._run_slot(idle_slot, pending))
                idle_slot.task = slot_task

    async def _run_slot(self, slot: SubmissionSlot, pending: _PendingSubmission) -> None:
        """單個槽位的完整生命週期：SUBMITTING → POLLING → PROCESSING → IDLE。"""
        submission_id = pending.metadata.get("submission_id", "unknown")
        try:
            self._stats["total_submitted"] += 1
            logger.info("[SlotManager] 槽位 %d 開始提交: %s", slot.id, submission_id)

            if self._submit_fn is not None:
                slot.state = SlotState.SUBMITTING
                submit_result = await asyncio.wait_for(
                    self._submit_fn(slot.expression, pending.simulation_payload, pending.metadata),
                    timeout=120.0,
                )

            slot.state = SlotState.POLLING
            if self._poll_fn is not None:
                poll_timeout = float(settings.BRAIN_POLL_TIMEOUT)
                yield_interval = 10.0
                poll_start = time.monotonic()
                poll_result = None

                while True:
                    try:
                        remaining = poll_timeout - (time.monotonic() - poll_start)
                        if remaining <= 0:
                            raise TimeoutError(f"Poll 超時 ({poll_timeout}s)")

                        poll_result = await asyncio.wait_for(
                            self._poll_fn(submission_id, submit_result if self._submit_fn else None),
                            timeout=min(yield_interval, remaining),
                        )
                        break

                    except TimeoutError:
                        if time.monotonic() - poll_start >= poll_timeout:
                            raise TimeoutError(f"Poll 超時 ({poll_timeout}s)") from None
                        logger.debug(
                            "[SlotManager] 槽位 %d poll yield (%.0fs/%.0fs)",
                            slot.id,
                            time.monotonic() - poll_start,
                            poll_timeout,
                        )
                        await asyncio.sleep(0.1)
                        continue

                slot.result = poll_result
            elif self._submit_fn is not None:
                slot.result = submit_result

            slot.state = SlotState.PROCESSING
            self._stats["total_completed"] += 1

            if self._on_complete_callback is not None and slot.result:
                await self._on_complete_callback(slot, slot.result)

            logger.info(
                "[SlotManager] 槽位 %d 完成: %s sharpe=%s",
                slot.id,
                submission_id,
                slot.result.get("sharpe", "N/A") if slot.result else "N/A",
            )

        except TimeoutError:
            self._stats["total_timed_out"] += 1
            logger.warning("[SlotManager] 槽位 %d 超時: %s", slot.id, submission_id)
            if self._on_fail_callback is not None:
                await self._on_fail_callback(slot, TimeoutError(f"槽位 {slot.id} 操作超時"))
        except (OSError, ValueError, RuntimeError) as exc:
            self._stats["total_failed"] += 1
            logger.error("[SlotManager] 槽位 %d 異常: %s — %s", slot.id, submission_id, exc)
            if self._on_fail_callback is not None:
                await self._on_fail_callback(slot, exc)
        finally:
            async with self._lock:
                slot.state = SlotState.IDLE
                slot.alpha_id = ""
                slot.expression = ""
                slot.result = None
                slot.task = None
            asyncio.create_task(self._try_dispatch())

    @property
    def available_slots(self) -> int:
        """目前空閒槽位數量。"""
        return sum(1 for s in self._slots if s.state == SlotState.IDLE)

    @property
    def active_count(self) -> int:
        """目前正在使用的槽位數量。"""
        return sum(1 for s in self._slots if s.state != SlotState.IDLE)

    @property
    def queue_depth(self) -> int:
        """等待佇列深度。"""
        return len(self._wait_queue)

    def get_status(self) -> dict:
        """返回完整狀態供 dashboard 使用。"""
        return {
            "max_concurrent": self.max_concurrent,
            "available": self.available_slots,
            "active": self.active_count,
            "queue_depth": self.queue_depth,
            "slots": [
                {
                    "id": s.id,
                    "state": s.state.name,
                    "alpha_id": s.alpha_id,
                    "expression": s.expression[:80] if s.expression else "",
                    "submitted_at": s.submitted_at,
                }
                for s in self._slots
            ],
            "stats": dict(self._stats),
        }

    async def shutdown(self) -> None:
        """優雅關閉，等待所有槽完成。"""
        self._running = False
        tasks = [s.task for s in self._slots if s.task is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        async with self._lock:
            self._wait_queue.clear()
        logger.info("[SlotManager] 已關閉，所有槽位已釋放")


# ═══════════════════════════════════════════════════════════════════════════════
# 組件 2: AlphaQueue — Alpha 優先級排隊系統
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(order=True)
class AlphaQueueEntry:
    """Alpha 佇列條目，使用 heap 排序。"""

    priority: float
    counter: int
    expression: str = field(compare=False)
    direction: str = field(compare=False, default="")
    source: str = field(compare=False, default="generator")
    sharpe_estimate: float = field(compare=False, default=0.0)
    metadata: dict = field(default_factory=dict, compare=False)
    created_at: float = field(default_factory=time.monotonic, compare=False)
    queue_id: str = field(default="", compare=False)


class AlphaQueue:
    """Alpha 優先級排隊系統。

    排序策略：
      - 高 sharpe_estimate 的優先（priority 越小越優先）
      - 同分數下 FIFO（counter 保證）
      - improvement 來源的比新生成的優先（因為已經過一次驗證）

    與 SlotManager 整合：
      - SlotManager 有空位時自動從 queue pop
      - 改進後的 alpha 可 re-enqueue（新的 priority）
    """

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._heap: list[AlphaQueueEntry] = []
        self._entries_by_id: dict[str, AlphaQueueEntry] = {}
        self._counter = 0
        self._lock = asyncio.Lock()
        self._orchestrator = None
        self._stats = {
            "enqueued": 0,
            "dequeued": 0,
            "requeued": 0,
            "removed": 0,
            "dropped_full": 0,
        }

    def _source_priority_bonus(self, source: str) -> float:
        """根據來源給予額外優先權加成。improver 來源更優先。"""
        bonuses = {"improver": -0.5, "mcts": -0.3, "generator": 0.0}
        return bonuses.get(source, 0.0)

    async def enqueue(self, entry: AlphaQueueEntry) -> str:
        """將 alpha 加入佇列。

        Returns:
            queue_id: 此條目的唯一佇列 ID
        """
        async with self._lock:
            if len(self._heap) >= self.max_size:
                self._stats["dropped_full"] += 1
                logger.warning("[AlphaQueue] 佇列已滿 (%d)，丟棄新條目", self.max_size)
                raise QueueFullError(f"AlphaQueue 已達上限 {self.max_size}")

            self._counter += 1
            if not entry.queue_id:
                entry.queue_id = f"q-{uuid.uuid4().hex[:8]}"
            entry.counter = self._counter
            entry.priority = -entry.sharpe_estimate + self._source_priority_bonus(entry.source)
            if entry.created_at == 0.0:
                entry.created_at = time.monotonic()

            heappush(self._heap, entry)
            self._entries_by_id[entry.queue_id] = entry
            self._stats["enqueued"] += 1

            logger.debug(
                "[AlphaQueue] 入隊: %s src=%s sharpe_est=%.2f priority=%.2f",
                entry.queue_id,
                entry.source,
                entry.sharpe_estimate,
                entry.priority,
            )
            if self._orchestrator is not None:
                self._orchestrator._queue_event.set()
            return entry.queue_id

    async def dequeue(self) -> AlphaQueueEntry | None:
        """取出最高優先級的條目。"""
        async with self._lock:
            if not self._heap:
                return None
            entry = heappop(self._heap)
            self._entries_by_id.pop(entry.queue_id, None)
            self._stats["dequeued"] += 1
            return entry

    async def peek(self) -> AlphaQueueEntry | None:
        """偷看最高優先級條目但不取出。"""
        async with self._lock:
            return self._heap[0] if self._heap else None

    async def remove(self, queue_id: str) -> bool:
        """從佇列中移除指定條目（如果還在佇列中）。

        Returns:
            是否成功移除
        """
        async with self._lock:
            entry = self._entries_by_id.pop(queue_id, None)
            if entry is None:
                return False
            try:
                self._heap.remove(entry)
                heapify(self._heap)
            except ValueError:
                pass
            self._stats["removed"] += 1
            return True

    async def requeue_with_new_priority(self, queue_id: str, new_sharpe: float, **kwargs) -> bool:
        """改進後的 alpha 以新的 sharpe 估計值重新入隊。

        Returns:
            是否成功重新入隊
        """
        import heapq

        old_entry = self._entries_by_id.get(queue_id)
        if old_entry:
            try:
                self._heap.remove(old_entry)
                heapq.heapify(self._heap)
            except ValueError:
                pass
            self._entries_by_id.pop(queue_id, None)

        new_entry = AlphaQueueEntry(
            priority=0.0,
            counter=0,
            expression=old_entry.expression if old_entry else kwargs.get("expression", ""),
            direction=kwargs.get("direction", old_entry.direction if old_entry else ""),
            source="improver",
            sharpe_estimate=new_sharpe,
            metadata=kwargs.get("metadata", old_entry.metadata if old_entry else {}),
        )
        _enqueued_id = await self.enqueue(new_entry)
        return bool(_enqueued_id)

    @property
    def size(self) -> int:
        """當前佇列大小。"""
        return len(self._heap)

    @property
    def stats(self) -> dict:
        """佇列統計資訊。"""
        return {
            **dict(self._stats),
            "current_size": self.size,
            "max_size": self.max_size,
        }


class QueueFullError(Exception):
    """AlphaQueue 已滿時拋出。"""


# ═══════════════════════════════════════════════════════════════════════════════
# 組件 3: ImprovementWorkerPool — 自學習改進工作池
# ═══════════════════════════════════════════════════════════════════════════════


class ImprovementPriority(Enum):
    """改進任務的優先級分類。"""

    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class ImprovementJob:
    """單個自學習改進任務。"""

    expression: str
    current_sharpe: float
    failure_type: str | None = None
    improvement_hint: str = ""
    priority: ImprovementPriority = ImprovementPriority.MEDIUM
    source_queue_id: str = ""
    max_attempts: int = 3
    attempts: int = 0
    job_id: str = field(default_factory=lambda: f"job-{uuid.uuid4().hex[:8]}")
    metadata: dict = field(default_factory=dict)


@dataclass
class ImprovementResult:
    """改進任務的執行結果。"""

    success: bool
    expression: str
    new_sharpe_estimate: float
    job_id: str
    attempts_used: int
    mutation_description: str = ""


class ImprovementWorkerPool:
    """自學習改進工作池。

    當 submission queue 堆積或 slot 空閒時，
    工作池會從 pending improvements 中取 job 進行 LLM 改進。

    工作流程：
      1. 從 improvement_queue 取出一個 job
      2. 調用 LLM（透過 ResourceDispatcher）進行突變/改進
      3. 驗證改進後的表達式語法
      4. 計算新的 sharpe_estimate
      5. 如果改善了 → re-enqueue 到 AlphaQueue（更高優先級）
      6. 如果沒改善 → 降低優先級或丟棄
    """

    def __init__(
        self,
        max_workers: int = 2,
        llm_generate_fn: Callable[..., Awaitable[str]] | None = None,
        improve_fn: Callable[..., Awaitable[ImprovementResult]] | None = None,
        resource_dispatcher: ResourceDispatcher | None = None,
    ):
        self.max_workers = max_workers
        self._llm_generate_fn = llm_generate_fn
        self._improve_fn = improve_fn
        self._dispatcher = resource_dispatcher
        self._job_queue: list[ImprovementJob] = []
        self._lock = asyncio.Lock()
        self._job_available = asyncio.Condition(self._lock)
        self._running = False
        self._workers: list[asyncio.Task] = []
        self._stats = {
            "jobs_submitted": 0,
            "jobs_completed": 0,
            "jobs_improved": 0,
            "jobs_failed": 0,
            "jobs_abandoned": 0,
        }
        self._on_improvement_callback: Callable[[ImprovementResult], Awaitable[None]] | None = None

    def set_improvement_callback(
        self,
        callback: Callable[[ImprovementResult], Awaitable[None]],
    ) -> None:
        """設置改進成功時的回調。"""
        self._on_improvement_callback = callback

    def _classify_priority(self, sharpe: float, failure_type: str | None = None) -> ImprovementPriority:
        """根據 sharpe 和失敗類型自動分類優先級。"""
        if failure_type in ("gate_close", "margin_fail") and sharpe > 1.0:
            return ImprovementPriority.CRITICAL
        if sharpe > 1.5:
            return ImprovementPriority.HIGH
        if sharpe > 0.5:
            return ImprovementPriority.MEDIUM
        return ImprovementPriority.LOW

    async def submit_improvement_job(self, job: ImprovementJob) -> str:
        """提交一個改進任務到工作池。

        Returns:
            job_id: 任務 ID
        """
        if job.priority == ImprovementPriority.MEDIUM:
            job.priority = self._classify_priority(job.current_sharpe, job.failure_type)

        async with self._lock:
            self._job_queue.append(job)
            self._job_queue.sort(key=lambda j: j.priority.value)
            self._stats["jobs_submitted"] += 1
            self._job_available.notify(1)

        logger.info(
            "[ImprovementPool] 任務入隊: %s expr=%s sharpe=%.2f pri=%s",
            job.job_id,
            job.expression[:60],
            job.current_sharpe,
            job.priority.name,
        )
        return job.job_id

    async def _get_next_job(self) -> ImprovementJob | None:
        """從佇列取得下一個待處理任務。"""
        async with self._lock:
            for job in self._job_queue:
                if job.attempts < job.max_attempts:
                    return job
            return None

    async def _process_job(self, job: ImprovementJob) -> ImprovementResult | None:
        """處理單個改進任務的核心邏輯。"""
        job.attempts += 1
        try:
            if self._improve_fn is not None:
                result = await asyncio.wait_for(
                    self._improve_fn(job.expression, job.current_sharpe, job.failure_type, job.improvement_hint),
                    timeout=120.0,
                )
                return result

            if self._llm_generate_fn is not None and self._dispatcher is not None:
                prompt = self._build_improvement_prompt(job)
                improved_expr = await self._dispatcher.dispatch_llm_mutation(
                    prompt=prompt,
                    priority=2,
                )
                if improved_expr and improved_expr != job.expression:
                    new_sharpe = job.current_sharpe * 1.1
                    return ImprovementResult(
                        success=True,
                        expression=improved_expr,
                        new_sharpe_estimate=new_sharpe,
                        job_id=job.job_id,
                        attempts_used=job.attempts,
                        mutation_description="llm_mutation",
                    )

            return ImprovementResult(
                success=False,
                expression=job.expression,
                new_sharpe_estimate=job.current_sharpe,
                job_id=job.job_id,
                attempts_used=job.attempts,
            )

        except TimeoutError:
            logger.warning("[ImprovementPool] 任務 %s 超時 (attempt %d/%d)", job.job_id, job.attempts, job.max_attempts)
            return None
        except (ValueError, TypeError, RuntimeError) as exc:
            logger.error("[ImprovementPool] 任務 %s 異常: %s", job.job_id, exc)
            return None

        def _build_improvement_prompt(self, job: ImprovementJob) -> str:
            """構建送給 LLM 的改進提示詞。"""
            hint_part = f"\n改進方向提示: {job.improvement_hint}" if job.improvement_hint else ""
            failure_part = f"\n失敗類型: {job.failure_type}" if job.failure_type else ""
            return (
                f"你是一位量化研究專家。請對以下 alpha 表達式進行改進。\n"
                f"當前表達式: {job.expression}\n"
                f"當前 Sharpe 估計: {job.current_sharpe:.3f}{failure_part}{hint_part}\n"
                f"請輸出改進後的 FASTEXPR 表達式，不要解釋。"
            )

    async def _worker_loop(self, worker_id: int) -> None:
        """單個 worker 的主迴圈。"""
        logger.info("[ImprovementPool] Worker %d 啟動", worker_id)
        while self._running:
            job = await self._get_next_job()
            if job is None:
                async with self._job_available:
                    await self._job_available.wait()
                continue

            logger.info(
                "[ImprovementPool] Worker %d 處理任務 %s (attempt %d/%d)",
                worker_id,
                job.job_id,
                job.attempts,
                job.max_attempts,
            )

            result = await self._process_job(job)
            if result is None:
                continue

            if result.success and result.new_sharpe_estimate > job.current_sharpe:
                self._stats["jobs_improved"] += 1
                logger.info(
                    "[ImprovementPool] ✅ 任務 %s 改進成功: %.3f → %.3f",
                    job.job_id,
                    job.current_sharpe,
                    result.new_sharpe_estimate,
                )
                if self._on_improvement_callback is not None:
                    await self._on_improvement_callback(result)
                async with self._lock:
                    if job in self._job_queue:
                        self._job_queue.remove(job)
                self._stats["jobs_completed"] += 1
            else:
                if job.attempts >= job.max_attempts:
                    self._stats["jobs_abandoned"] += 1
                    logger.info("[ImprovementPool] ❌ 任務 %s 已達最大嘗試次數，放棄", job.job_id)
                    async with self._lock:
                        if job in self._job_queue:
                            self._job_queue.remove(job)
                    self._stats["jobs_completed"] += 1
                else:
                    self._stats["jobs_failed"] += 1
                    job.priority = ImprovementPriority(min(job.priority.value + 1, 3))

        logger.info("[ImprovementPool] Worker %d 已停止", worker_id)

    async def start(self) -> None:
        """啟動所有 workers。"""
        self._running = True
        self._workers = [asyncio.create_task(self._worker_loop(i)) for i in range(self.max_workers)]
        logger.info("[ImprovementPool] 已啟動 %d 個 workers", self.max_workers)

    async def drain(self) -> None:
        """等待所有 jobs 完成（不再接受新 job）。"""
        self._running = False
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        logger.info("[ImprovementPool] 所有 workers 已排空")

    def get_stats(self) -> dict:
        """返回工作池統計資訊。"""
        return {
            **dict(self._stats),
            "pending_jobs": len(self._job_queue),
            "active_workers": sum(1 for w in self._workers if not w.done()),
            "max_workers": self.max_workers,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 組件 4: ResourceDispatcher — LLM/向量模型動態分配器
# ═══════════════════════════════════════════════════════════════════════════════


class ResourceType(Enum):
    """資源類型枚舉。"""

    LLM_GENERATE = "llm_generate"
    LLM_MUTATE = "llm_mutate"
    EMBED = "embed"
    RAG_RETRIEVE = "rag_retrieve"
    BRAIN_SUBMIT = "brain_submit"


@dataclass
class ResourceRequest:
    """資源調度請求。"""

    resource_type: ResourceType
    priority: int = 5
    payload: dict = field(default_factory=dict)
    callback: Callable | None = None
    timeout: float = 30.0


@dataclass
class _ResourceStats:
    """單一資源類型的運行統計。"""

    total_requests: int = 0
    total_success: int = 0
    total_errors: int = 0
    total_timeouts: int = 0
    total_latency_ms: float = 0.0
    concurrent_now: int = 0


class ResourceDispatcher:
    """統一資源調度中心。

    所有需要 LLM / 向量模型 的調用都經過此 dispatcher：
      - multi_agent.py 的 FactorAgent/IdeaAgent/EvalAgent
      - brain_submitter.py 的 mutation loop
      - rag_engine.py 的 embedding/retrieval
      - semantic_mutator.py 的 MCTS simulate (if using LLM)
      - improvement_worker_pool 的改進任務

    功能：
      1. 並發控制：每種資源類型有獨立的 semaphore
      2. 優先級調度：mutation > generation > embed > rag
      3. 超時處理：長時間未返回的自動取消
      4. 統計追蹤：每種資源的使用率、平均延遲、佇列深度
      5. 自適應限流：當錯誤率升高時自動降低並發度
    """

    DEFAULT_LIMITS: dict[ResourceType, int] = {
        ResourceType.LLM_GENERATE: 3,
        ResourceType.LLM_MUTATE: 2,
        ResourceType.EMBED: 5,
        ResourceType.RAG_RETRIEVE: 3,
        ResourceType.BRAIN_SUBMIT: 3,
    }

    def __init__(self, custom_limits: dict[ResourceType, int] | None = None, owner=None):
        limits = {**self.DEFAULT_LIMITS, **(custom_limits or {})}
        self._semaphores: dict[ResourceType, asyncio.Semaphore] = {
            rt: asyncio.Semaphore(limit) for rt, limit in limits.items()
        }
        self._queues: dict[ResourceType, list[ResourceRequest]] = {rt: [] for rt in ResourceType}
        self._stats: dict[ResourceType, _ResourceStats] = {rt: _ResourceStats() for rt in ResourceType}
        self._handlers: dict[ResourceType, Callable[..., Awaitable[Any]]] = {}
        self._adaptive_factors: dict[ResourceType, float] = dict.fromkeys(ResourceType, 1.0)
        self._lock = asyncio.Lock()
        self._owner = owner

    def register_handler(self, resource_type: ResourceType, handler: Callable[..., Awaitable[Any]]) -> None:
        """註冊特定資源類型的處理函數。"""
        self._handlers[resource_type] = handler

    async def acquire(self, req: ResourceRequest) -> Any:
        """獲取資源，執行請求，釋放資源，返回結果。

        流程：
          1. 對應 semaphore acquire（考慮自適應因子）
          2. 執行實際調用
          3. 更新統計
          4. semaphore release
          5. 返回結果
        """
        stats = self._stats[req.resource_type]
        stats.total_requests += 1
        start = time.monotonic()

        sem = self._semaphores[req.resource_type]
        max(1, int(sem._value * self._adaptive_factors[req.resource_type]))

        try:
            if req.resource_type == ResourceType.LLM_MUTATE:
                owner = getattr(self, "_owner", None)
                if owner and hasattr(owner, "acquire_mutation_slot"):
                    acquired = await owner.acquire_mutation_slot(timeout=req.timeout * 0.5)
                    if not acquired:
                        raise TimeoutError("全域 mutation 預算已滿")

            async with sem:
                stats.concurrent_now += 1
                handler = self._handlers.get(req.resource_type)
                if handler is None:
                    raise RuntimeError(f"未註冊 {req.resource_type.value} 的處理函數")

                result = await asyncio.wait_for(
                    handler(**req.payload),
                    timeout=req.timeout,
                )

                latency_ms = (time.monotonic() - start) * 1000
                stats.total_success += 1
                stats.total_latency_ms += latency_ms
                return result

        except TimeoutError:
            stats.total_timeouts += 1
            self._adapt_down(req.resource_type)
            raise TimeoutError(f"資源 {req.resource_type.value} 請求超時 ({req.timeout}s)") from None
        except (OSError, ValueError, RuntimeError):
            stats.total_errors += 1
            self._adapt_down(req.resource_type)
            raise
        finally:
            stats.concurrent_now = max(0, stats.concurrent_now - 1)
            if req.resource_type == ResourceType.LLM_MUTATE:
                owner = getattr(self, "_owner", None)
                if owner and hasattr(owner, "release_mutation_slot"):
                    owner.release_mutation_slot()

    async def dispatch_llm_generate(self, **kwargs) -> str:
        """便捷方法：dispatch LLM 生成請求。"""
        return await self.acquire(
            ResourceRequest(
                resource_type=ResourceType.LLM_GENERATE,
                priority=5,
                payload=kwargs,
                timeout=kwargs.get("timeout", 60.0),
            )
        )

    async def dispatch_llm_mutation(self, **kwargs) -> str:
        """便捷方法：dispatch LLM 突變請求（優先級高於 generate）。"""
        return await self.acquire(
            ResourceRequest(
                resource_type=ResourceType.LLM_MUTATE,
                priority=2,
                payload=kwargs,
                timeout=kwargs.get("timeout", 90.0),
            )
        )

    async def dispatch_embed(self, text: str, **kwargs) -> list[float]:
        """便捷方法：dispatch embedding 請求。"""
        return await self.acquire(
            ResourceRequest(
                resource_type=ResourceType.EMBED,
                priority=7,
                payload={"text": text, **kwargs},
                timeout=kwargs.get("timeout", 15.0),
            )
        )

    async def dispatch_rag_retrieve(self, query: str, **kwargs) -> dict:
        """便捷方法：dispatch RAG 檢索請求。"""
        return await self.acquire(
            ResourceRequest(
                resource_type=ResourceType.RAG_RETRIEVE,
                priority=6,
                payload={"query": query, **kwargs},
                timeout=kwargs.get("timeout", 20.0),
            )
        )

    def _adapt_down(self, resource_type: ResourceType) -> None:
        """當錯誤/超時發生時降低並發度。"""
        self._adaptive_factors[resource_type] = max(0.3, self._adaptive_factors[resource_type] * 0.85)
        logger.debug(
            "[ResourceDispatcher] %s 自適應降級: %.2f", resource_type.value, self._adaptive_factors[resource_type]
        )

    def _adapt_up(self, resource_type: ResourceType) -> None:
        """當連續成功時恢復並發度。"""
        self._adaptive_factors[resource_type] = min(1.0, self._adaptive_factors[resource_type] + 0.05)

    def get_utilization(self) -> dict:
        """返回各資源的使用率統計。"""
        result = {}
        for rt, stats in self._stats.items():
            limit = self.DEFAULT_LIMITS.get(rt, 1)
            result[rt.value] = {
                "utilization": round(stats.concurrent_now / max(limit, 1), 3),
                "concurrent": stats.concurrent_now,
                "limit": limit,
                "adaptive_factor": round(self._adaptive_factors[rt], 3),
                "total_requests": stats.total_requests,
                "success_rate": round(stats.total_success / max(stats.total_requests, 1), 3),
                "avg_latency_ms": round(stats.total_latency_ms / max(stats.total_success, 1), 1),
                "error_count": stats.total_errors,
                "timeout_count": stats.total_timeouts,
            }
        return result

    def get_queue_depths(self) -> dict:
        """返回各資源的佇列深度。"""
        return {rt.value: len(q) for rt, q in self._queues.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# 組件 5: PipelineOrchestrator — 頂層流水線編排器
# ═══════════════════════════════════════════════════════════════════════════════


class PipelineOrchestrator:
    """頂層流水線編排器 — 協調以上四個組件。

    主迴圈：
    ┌─────────────────────────────────────────────┐
    │                                             │
    │  Generator ──→ AlphaQueue                   │
    │                     ↓                       │
    │          [SlotManager有空位?]               │
    │               ↙         ↘                  │
    │           是(提交)    否(等待)               │
    │                             ↓                │
    │                    ImprovementWorkerPool     │
    │                         (改進 pending alpha)│
    │                             ↓                │
    │                      AlphaQueue (re-enqueue)│
    │                             ↓                │
    │                    [檢查 SlotManager]       │
    │                                             │
    └─────────────────────────────────────────────┘
    """

    def __init__(
        self,
        max_slots: int = 3,
        max_improvement_workers: int = 2,
        queue_max_size: int = 100,
        submit_fn: Callable[..., Awaitable[dict]] | None = None,
        poll_fn: Callable[..., Awaitable[dict]] | None = None,
        improve_fn: Callable[..., Awaitable[ImprovementResult]] | None = None,
        custom_limits: dict[ResourceType, int] | None = None,
    ):
        self.queue = AlphaQueue(max_size=queue_max_size)
        self.queue._orchestrator = self
        self.slot_manager = SubmissionSlotManager(
            max_concurrent=max_slots,
            submit_fn=submit_fn,
            poll_fn=poll_fn,
        )
        self.dispatcher = ResourceDispatcher(custom_limits=custom_limits, owner=self)
        self.worker_pool = ImprovementWorkerPool(
            max_workers=max_improvement_workers,
            improve_fn=improve_fn,
            resource_dispatcher=self.dispatcher,
        )

        self._mutation_budget = asyncio.Semaphore(2)
        self._mutation_stats = {"active": 0, "total": 0, "rejected": 0, "waiting": 0}

        self._queue_event = asyncio.Event()

        self.slot_manager.set_callbacks(
            on_complete=self._on_slot_completed,
            on_fail=self._on_slot_failed,
        )
        self.worker_pool.set_improvement_callback(self._on_improvement_success)

        self._running = False
        self._dispatcher_task: asyncio.Task | None = None
        self._orchestrate_task: asyncio.Task | None = None
        self._pipeline_stats = {
            "alphas_accepted": 0,
            "alphas_submitted": 0,
            "alphas_passed": 0,
            "alphas_failed": 0,
            "alphas_improved": 0,
            "start_time": 0.0,
        }

    async def acquire_mutation_slot(self, timeout: float = 30.0) -> bool:
        """嘗試獲取一個全域 mutation 槽位。

        Returns:
            True 表示成功獲取，False 表示超時未獲得
        """
        self._mutation_stats["waiting"] += 1
        try:
            await asyncio.wait_for(self._mutation_budget.acquire(), timeout=timeout)
            self._mutation_stats["active"] += 1
            self._mutation_stats["total"] += 1
            return True
        except TimeoutError:
            self._mutation_stats["rejected"] += 1
            return False
        finally:
            self._mutation_stats["waiting"] = max(0, self._mutation_stats["waiting"] - 1)

    def release_mutation_slot(self) -> None:
        """釋放 mutation 槽位。"""
        self._mutation_budget.release()
        self._mutation_stats["active"] = max(0, self._mutation_stats["active"] - 1)

    def get_mutation_stats(self) -> dict:
        """返回 mutation budget 統計資訊。"""
        return dict(self._mutation_stats)

    async def start(self) -> None:
        """啟動整個流水線。"""
        self._running = True
        self._pipeline_stats["start_time"] = time.monotonic()
        await self.worker_pool.start()
        self._orchestrate_task = asyncio.create_task(self._orchestration_loop())
        logger.info(
            "[PipelineOrchestrator] 流水線已啟動 (slots=%d, workers=%d)",
            self.slot_manager.max_concurrent,
            self.worker_pool.max_workers,
        )

    async def _orchestration_loop(self) -> None:
        """主編排迴圈：持續檢查 queue 和 slot 的狀態，自動調度。"""
        while self._running:
            try:
                while self.slot_manager.available_slots > 0 and self.queue.size > 0:
                    entry = await self.queue.dequeue()
                    if entry is None:
                        break
                    try:
                        await self.slot_manager.submit(
                            expression=entry.expression,
                            priority=entry.sharpe_estimate,
                            simulation_payload=entry.metadata.get("simulation_payload"),
                            metadata=entry.metadata,
                        )
                        self._pipeline_stats["alphas_submitted"] += 1
                        logger.info(
                            "[PipelineOrchestrator] 分發到槽位: %s sharpe_est=%.2f",
                            entry.queue_id,
                            entry.sharpe_estimate,
                        )
                    except (OSError, ValueError, RuntimeError) as exc:
                        logger.error("[PipelineOrchestrator] 分發失敗: %s", exc)

                self._queue_event.clear()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._queue_event.wait(), timeout=1.0)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("[PipelineOrchestrator] 編排迴圈異常: %s", exc, exc_info=True)
                await asyncio.sleep(2.0)

    async def submit_alpha(
        self,
        expression: str,
        source: str = "generator",
        sharpe_estimate: float = 0.0,
        direction: str = "",
        simulation_payload: dict | None = None,
        metadata: dict | None = None,
    ) -> str:
        """外部 API：提交一個 alpha 到流水線。

        Args:
            expression: Alpha 表達式
            source: 來源 ("generator" | "improver" | "mcts")
            sharpe_estimate: Sharpe 比率預估值
            direction: 策略方向
            simulation_payload: BRAIN 模擬 payload
            metadata: 附加資料

        Returns:
            queue_id: 佇列 ID
        """
        self._pipeline_stats["alphas_accepted"] += 1
        entry = AlphaQueueEntry(
            priority=0.0,
            counter=0,
            expression=expression,
            direction=direction,
            source=source,
            sharpe_estimate=sharpe_estimate,
            metadata=metadata or {},
        )
        if simulation_payload:
            entry.metadata["simulation_payload"] = simulation_payload

        queue_id = await self.queue.enqueue(entry)
        logger.info("[PipelineOrchestrator] Alpha 入隊: %s src=%s sharpe_est=%.2f", queue_id, source, sharpe_estimate)
        return queue_id

    async def _on_slot_completed(self, slot: SubmissionSlot, result: dict) -> None:
        """Slot 完成回調：處理結果，決定是否送 improvement pool。"""
        sharpe = result.get("sharpe", result.get("real_sharpe", 0))
        status = result.get("status", "")

        if status in ("PASS", "pass"):
            self._pipeline_stats["alphas_passed"] += 1
        else:
            self._pipeline_stats["alphas_failed"] += 1
            if sharpe and sharpe > 0.5:
                job = ImprovementJob(
                    expression=slot.expression,
                    current_sharpe=float(sharpe),
                    failure_type=result.get("failure_type") or "; ".join(result.get("gate_failures", [])[:1]) or None,
                    improvement_hint=result.get("improvement_hint", ""),
                    source_queue_id="",
                    metadata=result,
                )
                await self.worker_pool.submit_improvement_job(job)
                logger.info("[PipelineOrchestrator] FAIL 但有潛力，送 improvement pool: sharpe=%.2f", sharpe)

    async def _on_slot_failed(self, slot: SubmissionSlot, exc: Exception) -> None:
        """Slot 失敗回調。"""
        logger.warning("[PipelineOrchestrator] 槽位 %d 失敗: %s", slot.id, exc)

    async def _on_improvement_success(self, result: ImprovementResult) -> None:
        """改進成功回調：將改進後的 alpha 重新入隊。"""
        self._pipeline_stats["alphas_improved"] += 1
        try:
            await self.submit_alpha(
                expression=result.expression,
                source="improver",
                sharpe_estimate=result.new_sharpe_estimate,
                metadata={"mutation_description": result.mutation_description, "source_job_id": result.job_id},
            )
            logger.info("[PipelineOrchestrator] 改進 alpha 重新入隊: %.3f", result.new_sharpe_estimate)
        except QueueFullError:
            logger.warning("[PipelineOrchestrator] 佇列滿，改進 alpha 無法入隊")

    async def shutdown(self) -> None:
        """優雅關閉：停止接受新 → drain queue → drain slots → drain workers。"""
        self._running = False
        logger.info("[PipelineOrchestrator] 開始關閉...")

        if self._orchestrate_task and not self._orchestrate_task.done():
            self._orchestrate_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._orchestrate_task

        await self.slot_manager.shutdown()
        await self.worker_pool.drain()

        uptime = time.monotonic() - self._pipeline_stats["start_time"]
        logger.info(
            "[PipelineOrchestrator] 已關閉 — 運行 %.0fs | 接受:%d 提交:%d 通過:%d 失敗:%d 改進:%d",
            uptime,
            self._pipeline_stats["alphas_accepted"],
            self._pipeline_stats["alphas_submitted"],
            self._pipeline_stats["alphas_passed"],
            self._pipeline_stats["alphas_failed"],
            self._pipeline_stats["alphas_improved"],
        )

    def get_pipeline_status(self) -> dict:
        """返回整個流水線的即時狀態（供 dashboard）。"""
        uptime = time.monotonic() - self._pipeline_stats["start_time"]
        return {
            "running": self._running,
            "uptime_seconds": round(uptime, 1),
            "queue": {
                "size": self.queue.size,
                "max_size": self.queue.max_size,
                "stats": self.queue.stats,
            },
            "slot_manager": self.slot_manager.get_status(),
            "worker_pool": self.worker_pool.get_stats(),
            "resource_dispatcher": {
                "utilization": self.dispatcher.get_utilization(),
                "queue_depths": self.dispatcher.get_queue_depths(),
            },
            "pipeline_stats": dict(self._pipeline_stats),
        }
