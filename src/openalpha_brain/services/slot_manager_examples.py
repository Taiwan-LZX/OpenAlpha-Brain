"""
SlotManager 集成示例
===================

展示如何将 SlotManager 集成到 OpenAlpha-Brain 的主循环中。

本文件包含:
  1. 基础使用示例
  2. 与 LoopEngine 的集成
  3. 在 Launcher 中的部署方式
  4. 完整的回调函数示例（MAB 更新、日志记录等）
"""
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

from openalpha_brain.services.brain_client import BrainGateResult, authenticate
from openalpha_brain.services.slot_manager import (
    SlotInfo,
    SlotManager,
)

logger = logging.getLogger(__name__)


# ============================================================================
# 示例 1: 基础用法 — 独立使用 SlotManager
# ============================================================================

async def basic_usage_example():
    """基础用法：独立使用 SlotManager 提交因子"""

    # 1. 认证获取 cookies
    cookies = await authenticate(email="your@email.com", password="your_password")

    # 2. 创建并启动 SlotManager
    manager = SlotManager(
        cookies=cookies,
        max_slots=3,
        poll_interval=5.0,
        max_queue_size=100,
    )
    await manager.start()

    # 3. 注册完成回调
    async def on_alpha_complete(slot: SlotInfo, result: BrainGateResult):
        """当 alpha 完成时触发"""
        logger.info(
            "🎯 Alpha 完成! Slot=%d Sharpe=%.3f Passed=%s Expr=%.80s",
            slot.slot_id,
            result.sharpe,
            result.passed,
            slot.expression,
        )

        if result.passed and result.sharpe and result.sharpe >= 1.25:
            logger.info(
                "🏆 高质量 alpha! ID=%s Sharpe=%.3f",
                result.alpha_id,
                result.sharpe,
            )

    manager.register_callback(on_alpha_complete)

    # 4. 批量提交任务（非阻塞！）
    expressions = [
        "ts_decay_linear(volume, 10) / ts_std_dev(returns, 20)",
        "rank(ts_corr(close, volume, 15))",
        "-1 * delta(log(market_cap), 5)",
        "ts_zscore(earnings_yield, 60)",
        "signed_power(returns, 2) / ts_av_diff(returns, 10)",
    ]

    task_ids = []
    for i, expr in enumerate(expressions):
        task_id = await manager.submit(
            expression=expr,
            name=f"alpha_{i:03d}",
            strategy="momentum" if i % 2 == 0 else "reversal",
            priority=i,  # 越小越优先
        )
        task_ids.append(task_id)
        print(f"✓ Task {task_id} queued: {expr[:50]}...")

    # 5. 监控状态
    await asyncio.sleep(30)  # 等待一段时间让任务运行

    print("\n" + manager.status_summary())

    # 6. 查看指标
    metrics = manager.get_metrics()
    print("\n📊 Metrics:")
    print(f"   Submitted: {metrics.total_submitted}")
    print(f"   Completed: {metrics.total_completed}")
    print(f"   Passed:    {metrics.total_passed}")
    print(f"   Best Sharpe: {metrics.best_sharpe:.3f}")

    # 7. 优雅停止
    await manager.stop()


# ============================================================================
# 示例 2: 与 LoopEngine 集成 — 替换串行提交
# ============================================================================

class IntegratedSlotManager:
    """
    将 SlotManager 集成到 LoopEngine 中

    原来的流程 (串行):
      LoopEngine → submit_and_poll() → 等180s → 结果 → 下一个

    新流程 (并发):
      LoopEngine → SlotManager.submit() → 立即返回 → 继续生成下一个
                                              ↓
                                    [SlotManager 后台处理]
                                    Submit → Poll → Callback
    """

    def __init__(self, cookies):
        self.cookies = cookies
        self.manager: SlotManager | None = None
        self._results_cache: list[dict] = []

    async def start(self):
        """启动 SlotManager"""
        self.manager = SlotManager(
            cookies=self.cookies,
            max_slots=3,
            poll_interval=5.0,
        )

        # 注册 MAB 更新回调
        self.manager.register_callback(self._on_mab_update)

        # 注册日志回调
        self.manager.register_callback(self._log_result)

        await self.manager.start()
        logger.info("[Integrated] SlotManager started with 3 slots")

    async def stop(self):
        """停止 SlotManager"""
        if self.manager:
            await self.manager.stop()

    async def submit_alpha(self, expression: str, **kwargs) -> str:
        """
        提交 alpha 到队列（非阻塞）

        这是替换原来 submit_and_poll() 的接口
        """
        if not self.manager:
            raise RuntimeError("SlotManager not started")

        task_id = await self.manager.submit(
            expression=expression,
            **kwargs,
        )

        logger.info("[Integrated] Alpha submitted: %s (task_id=%s)", expression[:60], task_id)
        return task_id

    async def _on_mab_update(self, slot: SlotInfo, result: BrainGateResult):
        """
        MAB 回调 — 当 alpha 完成时更新 Multi-Armed Bandit

        这里可以:
        - 更新策略 arm 的 reward
        - 记录特征指纹
        - 触发进化算法
        """

        # 示例：更新 MAB reward
        if result.sharpe is not None:
            reward = result.sharpe if result.passed else -1.0

            # 这里应该调用实际的 MAB 更新逻辑
            # mab.update_arm(strategy=slot.task_name.split('_')[0], reward=reward)

            logger.info(
                "[MAB] Updated reward: strategy=%s sharpe=%.3f reward=%.3f",
                slot.task_name,
                result.sharpe,
                reward,
            )

    async def _log_result(self, slot: SlotInfo, result: BrainGateResult):
        """记录结果到缓存"""
        self._results_cache.append({
            "timestamp": datetime.now(UTC).isoformat(),
            "slot_id": slot.slot_id,
            "expression": slot.expression,
            "sharpe": result.sharpe,
            "passed": result.passed,
            "alpha_id": result.alpha_id,
            "elapsed_sec": slot.elapsed_sec,
        })

        # 保持缓存大小
        if len(self._results_cache) > 1000:
            self._results_cache = self._results_cache[-500:]

    def get_recent_results(self, n: int = 10) -> list[dict]:
        """获取最近 N 个结果"""
        return self._results_cache[-n:]


# ============================================================================
# 示例 3: 在 Launcher 中部署
# ============================================================================

async def launcher_integration_example():
    """
    展示如何在 BrainLauncher.run_main_loop() 中集成 SlotManager

    修改位置: launcher.py 的 run_main_loop() 方法
    """

    # ===== 在 run_main_loop() 开头添加 =====

    # from openalpha_brain.services.slot_manager import create_slot_manager

    # # 初始化 SlotManager（在认证成功后）
    # self._slot_manager = await create_slot_manager(
    #     cookies=cookies,  # 来自 authenticate_wq()
    #     max_slots=settings.PIPELINE_MAX_SLOTS,  # 默认 3
    # )

    # # 注册全局回调
    # self._slot_manager.register_callback(self._global_completion_callback)

    # ===== 修改 loop_engine 调用方式 =====

    # 原来:
    #   await loop_engine.run_loop(session_id)
    #
    # 改为 (需要在 loop_engine 内部调用 slot_manager.submit()):
    #   await loop_engine.run_loop_with_slots(session_id, self._slot_manager)

    # ===== 在 shutdown() 中清理 =====

    # if hasattr(self, '_slot_manager') and self._slot_manager:
    #     await self._slot_manager.stop()

    pass


# ============================================================================
# 示例 4: 完整的生产环境配置
# ============================================================================

async def production_setup():
    """生产环境的完整配置示例"""

    from openalpha_brain.config.config import settings

    # 1. 认证
    cookies = await authenticate(
        email=settings.BRAIN_EMAIL,
        password=settings.BRAIN_PASSWORD,
    )

    # 2. 创建带生产参数的 SlotManager
    manager = SlotManager(
        cookies=cookies,
        max_slots=settings.PIPELINE_MAX_SLOTS,  # 从配置读取，默认 3
        poll_interval=5.0,                       # BRAIN 建议 5s
        max_queue_size=200,                      # 允许更多排队
        max_poll_seconds=300,                    # 5 分钟超时
        submit_timeout=60.0,                     # 60s 提交超时
    )

    # 3. 注册多个回调（按顺序执行）
    async def callback_mab_update(slot: SlotInfo, result: BrainGateResult):
        """回调 1: 更新 MAB"""
        # ... MAB 更新逻辑
        pass

    async def callback_log_to_db(slot: SlotInfo, result: BrainGateResult):
        """回调 2: 记录到数据库"""
        # ... 数据库写入逻辑
        pass

    async def callback_notify_websocket(slot: SlotInfo, result: BrainGateResult):
        """回调 3: 通过 WebSocket 推送到前端"""
        # ... WebSocket 推送逻辑
        pass

    async def callback_auto_submit_review(slot: SlotInfo, result: BrainGateResult):
        """回调 4: 自动提交高质量 alpha 审核"""
        if result.passed and result.alpha_id and result.sharpe and result.sharpe >= 1.5:
            from openalpha_brain.services.brain_client import submit_alpha_for_review
            success = await submit_alpha_for_review(result.alpha_id, cookies)
            if success:
                logger.info("🚀 Auto-submitted alpha %s for review", result.alpha_id)

    manager.register_callback(callback_mab_update)
    manager.register_callback(callback_log_to_db)
    manager.register_callback(callback_notify_websocket)
    manager.register_callback(callback_auto_submit_review)

    # 4. 启动
    await manager.start()

    # 5. 返回给主程序使用
    return manager


# ============================================================================
# 示例 5: 错误处理与恢复
# ============================================================================

async def error_handling_example():
    """展示错误处理和恢复机制"""

    cookies = await authenticate(email="...", password="...")
    manager = SlotManager(cookies=cookies, max_slots=3)
    await manager.start()

    # 自定义错误处理回调
    async def monitor_errors(slot: SlotInfo, result: BrainGateResult):
        """监控错误并触发恢复"""
        if not result.passed:
            if any("CONCURRENT" in f for f in result.failures):
                logger.warning("⚠️ Slot %d hit concurrency limit", slot.slot_id)
                # 可以在这里实现自动降速逻辑

            if result.simulation_status == "ERROR":
                logger.error("❌ Slot %d simulation error: %s", slot.slot_id, result.failures)

    manager.register_callback(monitor_errors)

    # 定期检查健康状态
    async def health_check():
        while True:
            await asyncio.sleep(60)
            metrics = manager.get_metrics()

            # 错误率过高告警
            if metrics.total_completed > 0:
                error_rate = metrics.total_failed / metrics.total_completed
                if error_rate > 0.5:
                    logger.critical("🚨 High error rate: %.1f%%", error_rate * 100)

            # 打印状态
            print(manager.status_summary())

    # 启动健康检查
    health_task = asyncio.create_task(health_check())

    try:
        # ... 主逻辑 ...
        await asyncio.sleep(3600)  # 运行 1 小时
    finally:
        health_task.cancel()
        await manager.stop()


# ============================================================================
# 运行示例
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)-20s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # 运行基础示例
    # asyncio.run(basic_usage_example())

    print("=" * 70)
    print("  OpenAlpha-Brain SlotManager Integration Examples")
    print("=" * 70)
    print()
    print("可用示例:")
    print("  1. basic_usage_example()       — 基础用法")
    print("  2. IntegratedSlotManager         — 与 LoopEngine 集成")
    print("  3. launcher_integration_example() — Launcher 部署")
    print("  4. production_setup()            — 生产环境配置")
    print("  5. error_handling_example()      — 错误处理")
    print()
    print("取消注释 asyncio.run(...) 来运行对应示例")
