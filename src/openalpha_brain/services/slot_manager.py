"""
OpenAlpha-Brain Slot Manager
============================
WQ BRAIN 异步并发提交管理器 — 生产级实现。

特性:
  - 3 个 slot 并发运行（fire-and-forget 模式）
  - 自动队列填补（有空位立即取下一个任务）
  - 后台独立轮询（每个 slot 自己的 async task）
  - 完成回调驱动 MAB 更新 + 日志记录
  - 429 智能处理（指数退避重试）
  - 实时状态监控与指标统计

架构:
  ┌─────────────┐    submit()     ┌──────────────┐
  │  Producer   │ ──────────────→ │ Priority Queue│
  │ (LoopEngine) │                │ (max=100)     │
  └─────────────┘                └──────┬───────┘
                                        │ queue.get()
                         ┌──────────────┼──────────────┐
                         ↓              ↓              ↓
                    ┌─────────┐   ┌─────────┐   ┌─────────┐
                    │Slot 0   │   │Slot 1   │   │Slot 2   │
                    │Worker   │   │Worker   │   │Worker   │
                    └────┬────┘   └────┬────┘   └────┬────┘
                         │             │             │
                         ↓             ↓             ↓
                   [Submit→Poll→Callback→Repeat]
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from openalpha_brain.services.brain_client import (
    _SIM_DEFAULTS,
    _SIM_URL,
    BRAIN_BASE,
    BrainGateResult,
    BrainPollError,
    BrainSubmitError,
    _extract_gate_result,
    _safe_json,
)
from openalpha_brain.services.http_pool import get_client

logger = logging.getLogger(__name__)


class SlotState(Enum):
    """Slot 状态机"""

    IDLE = "idle"
    SUBMITTING = "submitting"
    RUNNING = "running"
    POLLING_RESULT = "polling_result"
    ERROR = "error"


class PriorityTier(Enum):
    """分层优先级队列 — 越小越优先"""

    TIER_0_EMERGENCY = 0
    TIER_1_HIGH_IMPROVED = 1
    TIER_2_IMPROVED = 2
    TIER_3_NORMAL = 3
    TIER_4_LOW = 4
    TIER_5_BACKGROUND = 5


@dataclass
class PriorityScore:
    """复合优先分 — 用于排序"""

    tier: PriorityTier
    predicted_pass_prob: float = 0.5
    improvement_generation: int = 0
    sharpe_history_avg: float = 0.0
    urgency_bonus: float = 0.0

    def to_sort_key(self) -> tuple:
        return (
            self.tier.value,
            -self.predicted_pass_prob,
            self.improvement_generation,
            -self.sharpe_history_avg,
            -self.urgency_bonus,
        )


@dataclass
class SlotInfo:
    """单个 slot 的实时状态"""

    slot_id: int
    state: SlotState = SlotState.IDLE
    expression: str = ""
    sim_id: str | None = None
    alpha_id: str | None = None
    submitted_at: datetime | None = None
    elapsed_sec: float = 0.0
    poll_count: int = 0
    result: BrainGateResult | None = None
    error: str | None = None
    task_name: str = ""
    metadata: dict = field(default_factory=dict)
    source: str = "generated"
    improvement_generation: int = 0


@dataclass
class SubmissionTask:
    """待提交任务"""

    expression: str
    name: str = ""
    strategy: str = ""
    priority: int = 10
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict = field(default_factory=dict)
    simulation_payload: dict | None = None
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    tier: PriorityTier = PriorityTier.TIER_3_NORMAL
    predicted_pass_prob: float = 0.5
    improvement_generation: int = 0
    parent_task_id: str | None = None
    wq_feedback: dict | None = None
    source: str = "generated"

    def build_priority_score(self) -> PriorityScore:
        return PriorityScore(
            tier=self.tier,
            predicted_pass_prob=self.predicted_pass_prob,
            improvement_generation=self.improvement_generation,
            sharpe_history_avg=self.wq_feedback.get("sharpe", 0.0) if self.wq_feedback else 0.0,
        )


@dataclass
class SlotMetrics:
    """全局统计指标"""

    total_submitted: int = 0
    total_completed: int = 0
    total_passed: int = 0
    total_failed: int = 0
    total_errors: int = 0
    avg_sharpe: float = 0.0
    best_sharpe: float = 0.0
    best_expression: str = ""
    best_alpha_id: str = ""
    slot_utilization: float = 0.0
    queue_depth: int = 0
    total_poll_count: int = 0
    uptime_seconds: float = 0.0
    _sharpe_sum: float = 0.0
    _sharpe_count: int = 0


CompletionCallback = Callable[[SlotInfo, BrainGateResult], Awaitable[None]]


class SlotManager:
    """
    WQ BRAIN 异步并发 Slot 管理器

    管理 3 个并发 simulation slot，实现：
    - 非阻塞提交（fire-and-forget）
    - 自动队列调度
    - 独立轮询循环
    - 智能错误恢复
    - 实时指标收集

    Usage:
        manager = SlotManager(cookies, max_slots=3)
        await manager.start()

        # 非阻塞提交
        task_id = await manager.submit(expression, name="alpha_001")

        # 查询状态
        slots = await manager.get_slot_status()
        metrics = manager.get_metrics()

        await manager.stop()
    """

    def __init__(
        self,
        cookies: Any,
        max_slots: int = 3,
        poll_interval: float = 5.0,
        max_queue_size: int = 100,
        max_poll_seconds: int = 300,
        submit_timeout: float = 60.0,
    ) -> None:
        """
        初始化 SlotManager

        Args:
            cookies: httpx.Cookies 认证凭证
            max_slots: 最大并发 slot 数（WQ 限制为 3）
            poll_interval: 轮询间隔秒数
            max_queue_size: 队列最大容量
            max_poll_seconds: 单次轮询超时秒数
            submit_timeout: 提交超时秒数
        """
        self._cookies = cookies
        self.max_slots = max_slots
        self.poll_interval = poll_interval
        self.max_queue_size = max_queue_size
        self.max_poll_seconds = max_poll_seconds
        self.submit_timeout = submit_timeout

        self._slots: list[SlotInfo] = [SlotInfo(slot_id=i) for i in range(max_slots)]
        self._queue: asyncio.PriorityQueue[tuple[tuple, SubmissionTask]] = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._metrics = SlotMetrics()
        self._callbacks: list[CompletionCallback] = []
        self._task_registry: dict[str, SubmissionTask] = {}

        self._running = False
        self._workers: list[asyncio.Task[None]] = []
        self._start_time: datetime | None = None
        self._queue_counter: int = 0
        self._stop_event: asyncio.Event = asyncio.Event()

        logger.info(
            "[SLOT-MGR] Initialized: %d slots, queue=%d, poll_interval=%.1fs",
            max_slots,
            max_queue_size,
            poll_interval,
        )

    async def start(self) -> None:
        """启动 slot manager，创建 worker tasks"""
        if self._running:
            logger.warning("[SLOT-MGR] Already running")
            return

        self._running = True
        self._start_time = datetime.now(UTC)
        self._stop_event.clear()

        logger.info("[SLOT-MGR] Starting %d slot workers...", self.max_slots)

        for i in range(self.max_slots):
            worker = asyncio.create_task(self._slot_worker(i), name=f"slot-worker-{i}")
            self._workers.append(worker)

        logger.info("[SLOT-MGR] ✓ All %d workers started", self.max_slots)

    async def stop(self) -> None:
        """优雅停止，等待所有运行中的任务完成"""
        if not self._running:
            return

        logger.info("[SLOT-MGR] Initiating graceful shutdown...")
        self._running = False
        self._stop_event.set()

        for worker in self._workers:
            worker.cancel()

        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)

        self._workers.clear()
        logger.info("[SLOT-MGR] Shutdown complete")

    async def submit(
        self,
        expression: str,
        name: str = "",
        strategy: str = "",
        priority: int = 10,
        metadata: dict | None = None,
        simulation_payload: dict | None = None,
        tier: PriorityTier = PriorityTier.TIER_3_NORMAL,
        predicted_pass_prob: float = 0.5,
        improvement_generation: int = 0,
        parent_task_id: str | None = None,
        wq_feedback: dict | None = None,
        source: str = "generated",
    ) -> str:
        """
        提交因子到队列（非阻塞！立即返回）

        Args:
            expression: FASTEXPR 因子表达式
            name: 因子名称
            strategy: 策略类型（momentum/reversal/volatility）
            priority: 优先级（越小越优先，0=最高）— 向后兼容，内部会转换为分层排序
            metadata: 自定义元数据
            simulation_payload: 完整的 simulation payload（可选）
            tier: 分层优先级层级（默认 TIER_3_NORMAL）
            predicted_pass_prob: 预测过审概率 (ProxyEvaluator 分数)
            improvement_generation: 改进代数 (0=原始, 1+=改进版)
            parent_task_id: 父任务ID (追踪改进链)
            wq_feedback: WQ返回的指标 {sharpe, turnover, ...}
            source: 来源类型 ("generated" | "improved" | "crossover" | "replay")

        Returns:
            task_id: 任务唯一标识符

        Raises:
            asyncio.QueueFull: 队列已满
        """
        task = SubmissionTask(
            expression=expression,
            name=name or f"alpha_{uuid.uuid4().hex[:6]}",
            strategy=strategy,
            priority=priority,
            metadata=metadata or {},
            simulation_payload=simulation_payload,
            tier=tier,
            predicted_pass_prob=predicted_pass_prob,
            improvement_generation=improvement_generation,
            parent_task_id=parent_task_id,
            wq_feedback=wq_feedback,
            source=source,
        )

        sort_key = task.build_priority_score().to_sort_key()

        try:
            self._queue.put_nowait((sort_key + (self._queue_counter,), task))
            self._queue_counter += 1
        except asyncio.QueueFull:
            raise asyncio.QueueFull(f"SlotManager queue full (max={self.max_queue_size})") from None

        self._task_registry[task.task_id] = task
        self._metrics.queue_depth = self._queue.qsize()

        logger.info(
            "[SLOT-MGR] Task queued: id=%s tier=%s prob=%.2f gen=%d src=%s queue_depth=%d",
            task.task_id,
            task.tier.name,
            task.predicted_pass_prob,
            task.improvement_generation,
            task.source,
            self._metrics.queue_depth,
        )

        return task.task_id

    async def get_slot_status(self) -> list[SlotInfo]:
        """获取所有 slot 的实时状态快照"""
        for slot in self._slots:
            if slot.state in (SlotState.RUNNING, SlotState.POLLING_RESULT) and slot.submitted_at:
                slot.elapsed_sec = (datetime.now(UTC) - slot.submitted_at).total_seconds()

        busy = sum(1 for s in self._slots if s.state not in (SlotState.IDLE, SlotState.ERROR))
        self._metrics.slot_utilization = busy / self.max_slots if self.max_slots > 0 else 0.0

        return list(self._slots)

    def get_metrics(self) -> SlotMetrics:
        """获取全局统计指标快照"""
        self._metrics.queue_depth = self._queue.qsize()

        if self._start_time:
            self._metrics.uptime_seconds = (datetime.now(UTC) - self._start_time).total_seconds()

        busy = sum(1 for s in self._slots if s.state not in (SlotState.IDLE, SlotState.ERROR))
        self._metrics.slot_utilization = busy / self.max_slots if self.max_slots > 0 else 0.0

        return self._metrics

    async def boost_priority(
        self,
        task_id: str,
        new_tier: PriorityTier,
        predicted_pass_prob: float | None = None,
        extra_metadata: dict | None = None,
    ) -> bool:
        """
        提升已有任务的优先级 (用于改进后重新入队)

        用途: 当一个因子从 WQ 收到反馈并被 LLM 改进后，
              调用此方法将其以更高优先级重新插入队列

        Args:
            task_id: 原任务 ID
            new_tier: 新的优先级层级
            predicted_pass_prob: 新的预测过审概率 (可选，不传则保留原值)
            extra_metadata: 额外元数据 (可选)

        Returns:
            True if successfully boosted, False if task not found
        """
        original_task = self._task_registry.get(task_id)
        if not original_task:
            logger.warning("[SLOT-MGR] boost_priority: task %s not found in registry", task_id)
            return False

        new_gen = original_task.improvement_generation + 1

        new_task_id = await self.submit(
            expression=original_task.expression,
            name=f"{original_task.name}_gen{new_gen}",
            strategy=original_task.strategy,
            priority=original_task.priority,
            metadata={**original_task.metadata, **(extra_metadata or {})},
            simulation_payload=original_task.simulation_payload,
            tier=new_tier,
            predicted_pass_prob=predicted_pass_prob
            if predicted_pass_prob is not None
            else original_task.predicted_pass_prob,
            improvement_generation=new_gen,
            parent_task_id=task_id,
            wq_feedback=original_task.wq_feedback,
            source="improved",
        )

        logger.info(
            "[SLOT-MGR] Task boosted: %s → %s (tier=%s gen=%d)",
            task_id,
            new_task_id,
            new_tier.name,
            new_gen,
        )

        return True

    async def submit_improved(
        self,
        expression: str,
        original_task_id: str,
        wq_feedback: dict,
        improvement_gen: int = 1,
        predicted_pass_prob: float = 0.6,
        priority_tier: PriorityTier | None = None,
    ) -> str:
        """
        便捷方法: 提交改进版因子 (自动设置 Tier 1 或 Tier 2)

        根据 wq_feedback 中的 sharpe 自动选择 tier:
          - sharpe >= 0.8 → TIER_1_HIGH_IMPROVED
          - sharpe >= 0.3 → TIER_2_IMPROVED
          - sharpe < 0.3 → TIER_3_NORMAL (提升有限)

        Args:
            expression: 改进后的因子表达式
            original_task_id: 原始任务 ID
            wq_feedback: WQ 返回的指标 {sharpe, turnover, ...}
            improvement_gen: 改进代数 (默认 1)
            predicted_pass_prob: 预测过审概率 (默认 0.6)
            priority_tier: 外部指定的优先级（如果提供则跳过自动检测）

        Returns:
            新任务的 task_id
        """
        if priority_tier is not None:
            tier = priority_tier
        else:
            sharpe = wq_feedback.get("sharpe", 0.0)
            if sharpe >= 0.8:
                tier = PriorityTier.TIER_1_HIGH_IMPROVED
            elif sharpe >= 0.3:
                tier = PriorityTier.TIER_2_IMPROVED
            else:
                tier = PriorityTier.TIER_3_NORMAL

        logger.info(
            "[SLOT-MGR] submit_improved: sharpe=%.3f → tier=%s",
            sharpe,
            tier.name,
        )

        return await self.submit(
            expression=expression,
            name=f"improved_{original_task_id}",
            strategy="",
            tier=tier,
            predicted_pass_prob=predicted_pass_prob,
            improvement_generation=improvement_gen,
            parent_task_id=original_task_id,
            wq_feedback=wq_feedback,
            source="improved",
        )

    async def submit_raw(self, expression: str, name: str, strategy: str = "") -> str:
        """
        便捷方法: 提交原始生成因子 (自动 TIER_3)

        Args:
            expression: 因子表达式
            name: 因子名称
            strategy: 策略类型 (可选)

        Returns:
            task_id
        """
        return await self.submit(
            expression=expression,
            name=name,
            strategy=strategy,
            tier=PriorityTier.TIER_3_NORMAL,
            predicted_pass_prob=0.5,
            improvement_generation=0,
            source="generated",
        )

    def register_callback(self, callback: CompletionCallback) -> None:
        """注册完成回调函数"""
        self._callbacks.append(callback)
        logger.debug("[SLOT-MGR] Callback registered: %s", callback.__name__)

    def unregister_callback(self, callback: CompletionCallback) -> None:
        """移除完成回调函数"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    async def _slot_worker(self, slot_id: int) -> None:
        """
        单个 slot 的 worker 循环

        循环流程:
          1. 从队列取任务（await queue.get()）
          2. 调用 brain_client 提交
          3. 后台轮询直到完成
          4. 触发完成回调
          5. 回到步骤 1 继续下一个
        """
        slot = self._slots[slot_id]
        logger.info("[SLOT-MGR] Worker %d started", slot_id)

        while self._running and not self._stop_event.is_set():
            try:
                sort_key, task = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0,
                )
            except TimeoutError:
                continue

            try:
                await self._execute_task(slot, task)
            except asyncio.CancelledError:
                logger.info("[SLOT-MGR] Slot %d worker cancelled", slot_id)
                break
            except Exception as exc:
                logger.error(
                    "[SLOT-MGR] Slot %d unhandled error: %s",
                    slot_id,
                    exc,
                    exc_info=True,
                )
                slot.state = SlotState.ERROR
                slot.error = str(exc)
                self._metrics.total_errors += 1
                await asyncio.sleep(2.0)

        logger.info("[SLOT-MGR] Worker %d stopped", slot_id)

    async def _execute_task(self, slot: SlotInfo, task: SubmissionTask) -> None:
        """执行单个任务：提交 → 轮询 → 回调"""
        slot.state = SlotState.SUBMITTING
        slot.expression = task.expression
        slot.task_name = task.name
        slot.result = None
        slot.error = None
        slot.alpha_id = None
        slot.sim_id = None
        slot.poll_count = 0

        logger.info(
            "[SLOT-MGR] Slot %d: %s → SUBMITTING expr=%.60s...",
            slot.slot_id,
            slot.state.value.upper(),
            task.expression,
        )

        try:
            sim_id, location = await self._submit_simulation(task)

            slot.state = SlotState.RUNNING
            slot.sim_id = sim_id
            slot.submitted_at = datetime.now(UTC)
            self._metrics.total_submitted += 1

            logger.info(
                "[SLOT-MGR] Slot %d: SUBMITTING → RUNNING sim_id=%s",
                slot.slot_id,
                sim_id[:12] if sim_id else "unknown",
            )

            result = await self._poll_simulation(slot, location)

            slot.state = SlotState.POLLING_RESULT
            slot.alpha_id = result.alpha_id
            slot.result = result

            logger.info(
                "[SLOT-MGR] Slot %d: RUNNING → COMPLETE sharpe=%.3f %s",
                slot.slot_id,
                result.sharpe or 0.0,
                "✅ PASS" if result.passed else "❌ FAIL",
            )

            await self._on_completion(slot, result, task)

        except BrainSubmitError as exc:
            logger.error("[SLOT-MGR] Slot %d Submit failed: %s", slot.slot_id, exc)
            slot.state = SlotState.ERROR
            slot.error = str(exc)
            self._metrics.total_failed += 1
            self._metrics.total_errors += 1

        except BrainPollError as exc:
            logger.error("[SLOT-MGR] Slot %d Poll failed: %s", slot.slot_id, exc)
            slot.state = SlotState.ERROR
            slot.error = str(exc)
            self._metrics.total_failed += 1
            self._metrics.total_errors += 1

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            logger.error(
                "[SLOT-MGR] Slot %d Unexpected error: %s",
                slot.slot_id,
                exc,
                exc_info=True,
            )
            slot.state = SlotState.ERROR
            slot.error = str(exc)
            self._metrics.total_errors += 1

        finally:
            self._queue.task_done()

    async def _submit_simulation(self, task: SubmissionTask) -> tuple[str | None, str | None]:
        """
        提交 simulation 到 BRAIN API

        Returns:
            (sim_id, poll_location) tuple
        """
        client = get_client()

        payload: dict[str, Any] = dict(_SIM_DEFAULTS)

        if task.simulation_payload:
            payload["settings"] = {
                **_SIM_DEFAULTS["settings"],
                **task.simulation_payload.get("settings", {}),
            }
            payload["regular"] = task.simulation_payload.get("regular", task.expression)
            if "type" in task.simulation_payload:
                payload["type"] = task.simulation["type"]
        else:
            payload["regular"] = task.expression

        logger.info(
            "[SLOT-MGR] Submitting to %s: %.80s...",
            _SIM_URL,
            task.expression,
        )

        sim_resp = await client.post(
            _SIM_URL,
            json=payload,
            cookies=self._cookies,
            timeout=self.submit_timeout,
        )

        if sim_resp.status_code not in (200, 201, 202):
            body = _safe_json(sim_resp)
            msg = (body or {}).get("message", sim_resp.text[:300]) if body else sim_resp.text[:300]
            raise BrainSubmitError(f"BRAIN simulation submit failed HTTP {sim_resp.status_code}: {msg}")

        location = sim_resp.headers.get("Location")
        if location and location.startswith("/"):
            location = BRAIN_BASE + location

        data = _safe_json(sim_resp) or {}
        sim_id = data.get("id")

        logger.info("[SLOT-MGR] Submitted successfully: sim_id=%s", sim_id)

        return sim_id, location

    async def _poll_simulation(self, slot: SlotInfo, location: str | None) -> BrainGateResult:
        """
        单个 simulation 的轮询逻辑

        复用 brain_client 的轮询模式：
        - GET /simulations/{id} 或 Location URL
        - 检查 Retry-After header
        - status != COMPLETE 就继续轮询
        - 429 智能处理（指数退避）
        """
        if not location:
            raise BrainPollError("No polling URL available")

        client = get_client()
        elapsed: float = 0.0
        consecutive_429 = 0

        while elapsed < self.max_poll_seconds and self._running:
            try:
                poll_resp = await client.get(
                    location,
                    cookies=self._cookies,
                    timeout=60.0,
                )
            except (ConnectionError, OSError, TimeoutError) as exc:
                logger.warning("[SLOT-MGR] Slot %d Poll request error: %s", slot.slot_id, exc)
                await asyncio.sleep(self.poll_interval)
                elapsed += self.poll_interval
                continue

            slot.poll_count += 1
            self._metrics.total_poll_count += 1

            if poll_resp.status_code in (200, 201, 202):
                data = _safe_json(poll_resp) or {}

                retry_after_raw = poll_resp.headers.get("Retry-After")
                if retry_after_raw is None:
                    retry_after = None
                else:
                    try:
                        retry_after = float(retry_after_raw)
                    except ValueError:
                        retry_after = None

                if retry_after is None:
                    logger.info(
                        "[SLOT-MGR] Slot %d Complete after %.1fs / %d polls",
                        slot.slot_id,
                        elapsed,
                        slot.poll_count,
                    )

                    prelim = _extract_gate_result(data)
                    alpha_id = prelim.alpha_id

                    if prelim.simulation_status == "ERROR":
                        logger.warning(
                            "[SLOT-MGR] Slot %d SIMULATION ERROR: %s",
                            slot.slot_id,
                            prelim.failures[0] if prelim.failures else "unknown",
                        )
                        return prelim

                    if alpha_id:
                        from openalpha_brain.services.brain_client import (
                            check_alpha,
                            fetch_alpha_details,
                        )

                        logger.info(
                            "[SLOT-MGR] Slot %d Fetching alpha %s details",
                            slot.slot_id,
                            alpha_id,
                        )
                        alpha_data = await fetch_alpha_details(alpha_id, self._cookies)
                        if alpha_data:
                            result = _extract_gate_result(alpha_data)
                            if result.sharpe is None and not result.brain_checks:
                                logger.info(
                                    "[SLOT-MGR] Slot %d No metrics — calling check API",
                                    slot.slot_id,
                                )
                                check_data = await check_alpha(alpha_id, self._cookies)
                                if check_data:
                                    result = _extract_gate_result(check_data)
                            return result

                    logger.info("[SLOT-MGR] Slot %d No alpha_id — using poll data", slot.slot_id)
                    return prelim

                wait = max(min(retry_after, 30.0), 1.0)
                logger.debug(
                    "[SLOT-MGR] Slot %d Running — Retry-After=%.0fs elapsed=%.1fs",
                    slot.slot_id,
                    retry_after,
                    elapsed,
                )
                await asyncio.sleep(wait)
                elapsed += wait
                consecutive_429 = 0

            elif poll_resp.status_code == 429:
                consecutive_429 += 1
                wait = float(poll_resp.headers.get("Retry-After", "10"))

                busy_slots = sum(1 for s in self._slots if s.state not in (SlotState.IDLE, SlotState.ERROR))

                if busy_slots >= self.max_slots:
                    logger.warning(
                        "[SLOT-MGR] Slot %d Rate limited (429 #%d) — all %d slots busy, waiting %.0fs",
                        slot.slot_id,
                        consecutive_429,
                        busy_slots,
                        wait,
                    )
                else:
                    logger.warning(
                        "[SLOT-MGR] Slot %d Unexpected 429 (#%d) with %d/%d slots free — waiting %.0fs",
                        slot.slot_id,
                        consecutive_429,
                        self.max_slots - busy_slots,
                        self.max_slots,
                        wait,
                    )

                backoff = min(wait * (1.5 ** (consecutive_429 - 1)), 120.0)
                await asyncio.sleep(backoff)
                elapsed += backoff

            else:
                body = _safe_json(poll_resp)
                msg = (body or {}).get("message", poll_resp.text[:200])
                raise BrainPollError(f"Unexpected poll response HTTP {poll_resp.status_code}: {msg}")

        raise BrainPollError(f"Simulation did not complete within {self.max_poll_seconds}s (elapsed={elapsed:.1f}s)")

    async def _on_completion(self, slot: SlotInfo, result: BrainGateResult, task: SubmissionTask) -> None:
        """
        完成回调 — 更新 metrics、触发用户回调、释放 slot
        """
        self._metrics.total_completed += 1

        if result.passed:
            self._metrics.total_passed += 1
        else:
            self._metrics.total_failed += 1

        if result.sharpe is not None:
            self._metrics._sharpe_sum += result.sharpe
            self._metrics._sharpe_count += 1
            self._metrics.avg_sharpe = round(self._metrics._sharpe_sum / max(self._metrics._sharpe_count, 1), 4)

            if result.sharpe > self._metrics.best_sharpe:
                self._metrics.best_sharpe = result.sharpe
                self._metrics.best_expression = slot.expression
                self._metrics.best_alpha_id = result.alpha_id or ""
                logger.info(
                    "[SLOT-MGR] 🏆 NEW BEST! sharpe=%.3f expr=%.60s...",
                    result.sharpe,
                    slot.expression,
                )

        for cb in self._callbacks:
            try:
                await cb(slot, result)
            except (OSError, ValueError, RuntimeError) as e:
                logger.error("[SLOT-MGR] Callback error: %s", e, exc_info=True)

        slot.state = SlotState.IDLE

        logger.info(
            "[SLOT-MGR] Slot %d recycled → IDLE (queue depth: %d)",
            slot.slot_id,
            self._queue.qsize(),
        )

    def status_summary(self) -> str:
        """生成人类可读的状态摘要"""
        metrics = self.get_metrics()
        lines = [
            "=" * 60,
            "  🎰 SLOT MANAGER STATUS",
            "=" * 60,
            f"  Uptime: {metrics.uptime_seconds:.0f}s",
            f"  Slots : {sum(1 for s in self._slots if s.state != SlotState.IDLE)}/{self.max_slots} busy",
            f"  Queue : {metrics.queue_depth}/{self.max_queue_size}",
            "-" * 60,
            f"  Submitted  : {metrics.total_submitted}",
            f"  Completed  : {metrics.total_completed}",
            f"  ✅ Passed  : {metrics.total_passed}",
            f"  ❌ Failed  : {metrics.total_failed}",
            f"  ⚠️ Errors  : {metrics.total_errors}",
            "-" * 60,
            f"  Avg Sharpe : {metrics.avg_sharpe:.3f}",
            f"  Best Sharpe: {metrics.best_sharpe:.3f}",
            f"  Utilization: {metrics.slot_utilization:.1%}",
            f"  Total Polls: {metrics.total_poll_count}",
            "=" * 60,
        ]

        for slot in self._slots:
            status_icon = {
                SlotState.IDLE: "⚪",
                SlotState.SUBMITTING: "🟡",
                SlotState.RUNNING: "🟢",
                SlotState.POLLING_RESULT: "🔵",
                SlotState.ERROR: "🔴",
            }.get(slot.state, "❓")

            expr_short = slot.expression[:40] + "..." if len(slot.expression) > 40 else slot.expression
            lines.append(
                f"  {status_icon} Slot {slot.slot_id}: {slot.state.value:<16} {expr_short:<42} {slot.elapsed_sec:.0f}s"
            )

        lines.append("=" * 60)
        return "\n".join(lines)


async def create_slot_manager(
    cookies: Any,
    max_slots: int = 3,
    **kwargs: Any,
) -> SlotManager:
    """
    工厂函数：创建并启动 SlotManager

    Args:
        cookies: httpx.Cookies 认证凭证
        max_slots: 并发 slot 数
        **kwargs: 传递给 SlotManager.__init__()

    Returns:
        已启动的 SlotManager 实例
    """
    manager = SlotManager(cookies, max_slots=max_slots, **kwargs)
    await manager.start()
    return manager
