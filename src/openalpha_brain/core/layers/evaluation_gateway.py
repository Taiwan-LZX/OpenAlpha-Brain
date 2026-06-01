"""
OpenAlpha-Brain EvaluationGateway — Layer 3 of 6-Layer Architecture
===================================================================
WQ BRAIN 平台交互边界层，封装完整的 submit → poll → parse → decide 流水线。

职责:
  1. WQ BRAIN 提交 — 将 alpha 提交到 WQ 平台进行模拟
  2. 结果获取 — 轮询/等待异步 BRAIN 结果返回
  3. 结果路由 — 调用 ResultRouter.parse() 将原始结果转为 ParsedWQResult
  4. 决策制定 — 调用 DecisionEngine.decide_from_parsed_result() 生成决策
  5. 错误处理 — 重试逻辑、超时处理、HTTP 429 防护 (Semaphore(3))

架构位置 (Layer 3):
  ┌──────────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
  │ GenerationPipeline   │ →   │ EvaluationGateway   │ →   │ImprovementOrchestr│
  │ (Layer 2)            │     │ (本模块 - Layer 3)  │     │(Layer 4)         │
  └──────────────────────┘     └─────────────────────┘     └──────────────────┘
                                      │
                              ┌───────┴───────┐
                              ▼               ▼
                       WQ BRAIN API    ResultRouter + DecisionEngine

提取来源:
  - loop_engine.py L1795: brain_result = await _submit_to_brain(...)
  - loop_engine.py L1797-1832: 结果日志 + 事件发射
  - result_router.py: ResultRouter.route() → ParsedWQResult
  - decision_engine.py: DecisionEngine.decide_from_parsed_result() → DecisionOutcome
  - brain_submitter.py L916-1112: _submit_to_brain 完整实现
  - brain_submitter.py L1117: Semaphore(3) 模式

Usage:
    gateway = EvaluationGateway(config={"timeout_seconds": 300.0})
    result = await gateway.submit_and_evaluate(
        expression="ts_decay_linear(rank(close/volume), sector), 10)",
        session_id="sess_001",
        brain_submitter=_submit_to_brain,
        result_router=result_router,
        decision_engine=decision_engine,
        semaphore=asyncio.Semaphore(3),
    )
    print(result.status, result.sharpe, result.decision.action)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from openalpha_brain.core.models import BrainSimStatus
from openalpha_brain.evolution.near_pass_improver import NEAR_PASS_SHARPE_THRESHOLD
from openalpha_brain.monitoring.algorithm_telemetry import AlgorithmTelemetryCollector

logger = logging.getLogger(__name__)

_EPSILON = 1e-6


@dataclass
class EvaluationResult:
    """单次 BRAIN 评估的完整结果

    Attributes:
        brain_result: 原始 WQ Brain 返回对象 (BrainSubmissionResult)
        parsed_result: ResultRouter 解析后的结构化结果 (ParsedWQResult)
        decision: DecisionEngine 的决策输出 (DecisionOutcome)
        status: 状态分类 "PASS" | "FAIL" | "ERROR" | "TIMEOUT"
        sharpe: Sharpe 比率
        fitness: Fitness 值
        turnover: Turnover 值
        is_near_pass: 是否接近通过 (sharpe >= NEAR_PASS 但未 PASS)
        error_message: 错误信息
        duration_sec: 总耗时 (秒)
        metadata: 扩展元数据字典
    """

    brain_result: Any = None
    parsed_result: Any | None = None
    decision: Any | None = None
    status: str = ""
    sharpe: float = 0.0
    fitness: float = 0.0
    turnover: float | None = None
    is_near_pass: bool = False
    error_message: str = ""
    duration_sec: float = 0.0
    metadata: dict = field(default_factory=dict)
    failure_reason: str = ""
    failure_category: str = ""

    def to_log_dict(self) -> dict:
        """转换为可日志记录的字典"""
        return {
            "status": self.status,
            "sharpe": self.sharpe,
            "fitness": self.fitness,
            "turnover": self.turnover,
            "is_near_pass": self.is_near_pass,
            "decision_action": getattr(self.decision, "action", None),
            "error": self.error_message[:120] if self.error_message else "",
            "duration_sec": round(self.duration_sec, 3),
        }


class EvaluationGateway:
    """WQ BRAIN 平台交互网关 — Layer 3

    作为系统与 WorldQuant Brain API 之间的唯一边界层，
    封装提交、轮询、解析、决策的完整生命周期。

    Design Principles:
    - 边界隔离: 所有 WQ API 交互集中在此层，下游不接触 raw API
    - 依赖注入: brain_submitter / result_router / decision_engine / semaphore 全部外部传入
    - 容错分级: circuit_breaker → auth → submit → poll → parse → decide 逐级降级
    - 可观测性: 集成 Telemetry 记录每个阶段耗时
    - 速率控制: 强制 Semaphore(3) 防止 HTTP 429

    Config 参数:
        timeout_seconds: 单次评估总超时 (默认 300s)
        max_retries: 提交失败后的最大重试次数 (默认 2)
        retry_base_delay: 重试基础延迟秒数 (默认 2.0)
        near_pass_sharpe_threshold: 近通过阈值 (默认 0.8)
    """

    DEFAULT_CONFIG: dict[str, Any] = {
        "timeout_seconds": 300.0,
        "max_retries": 2,
        "retry_base_delay": 2.0,
        "near_pass_sharpe_threshold": NEAR_PASS_SHARPE_THRESHOLD,
    }

    def __init__(self, config: dict | None = None):
        self.config: dict[str, Any] = {**self.DEFAULT_CONFIG, **(config or {})}
        self._tel = AlgorithmTelemetryCollector.get_instance()
        self._stats: dict[str, int] = {"total": 0, "pass": 0, "fail": 0, "error": 0, "timeout": 0}

    @property
    def stats(self) -> dict[str, int]:
        """只读统计快照"""
        return dict(self._stats)

    async def submit_and_evaluate(
        self,
        expression: str,
        session_id: str,
        *,
        brain_submitter: Callable | None = None,
        result_router: Any | None = None,
        decision_engine: Any | None = None,
        semaphore: asyncio.Semaphore | None = None,
        timeout_seconds: float = 300.0,
        max_retries: int = 2,
    ) -> EvaluationResult:
        """提交 alpha 到 WQ BRAIN 并完成完整评估流水线

        流水线:
          1. Semaphore(3) 获取 → 防止 HTTP 429
          2. brain_submitter(alpha, session_id) → BrainSubmissionResult
          3. ResultRouter.route(slot_info, brain_result) → ParsedWQResult
          4. DecisionEngine.decide_from_parsed_result(parsed) → DecisionOutcome
          5. 分类 status + 组装 EvaluationResult

        Args:
            expression: ThreeBlockTemplate alpha 表达式
            session_id: 会话标识符
            brain_submitter: 提交函数 (签名: async fn(alpha, session_id, cycle) -> BrainSubmissionResult)
                           若为 None 则跳过提交，返回 ERROR 状态
            result_router: ResultRouter 实例 (用于解析原始结果)
            decision_engine: DecisionEngine 实例 (用于生成决策)
            semaphore: asyncio.Semaphore(3) 用于速率限制
            timeout_seconds: 单次评估超时 (秒)
            max_retries: 失败重试次数

        Returns:
            EvaluationResult: 包含 brain_result / parsed_result / decision / status 的完整结果
        """
        t0 = time.perf_counter()
        eid = None
        with contextlib.suppress(OSError, ValueError, RuntimeError):
            eid = await self._tel.record_enter(
                "EvaluationGateway",
                cycle_id=session_id,
                expr_id=hash(expression) % 10000,
            )

        result = EvaluationResult(metadata={"session_id": session_id, "expression_preview": expression[:80]})
        self._stats["total"] += 1

        try:
            if brain_submitter is None:
                logger.warning("[GATEWAY] ⚠ No brain_submitter provided, skipping submission")
                result.status = "ERROR"
                result.error_message = "No brain_submitter provided"
                return result

            _sem = semaphore or asyncio.Semaphore(3)
            _timeout = timeout_seconds or self.config["timeout_seconds"]
            _retries = max(0, min(max_retries, self.config["max_retries"]))
            _retry_delay = self.config["retry_base_delay"]

            brain_result = None
            last_error = ""

            for attempt in range(_retries + 1):
                try:
                    async with asyncio.timeout(_timeout):
                        async with _sem:
                            logger.info(
                                "[GATEWAY] ▶ SUBMITTING | session=%s expr=%.60s... attempt=%d/%d",
                                session_id,
                                expression,
                                attempt + 1,
                                _retries + 1,
                            )
                            brain_result = await brain_submitter(_build_alpha_proxy(expression), session_id, 0)

                    if brain_result is not None:
                        break

                    last_error = "brain_submitter returned None"
                    logger.warning("[GATEWAY] ⚠ NULL RESULT | session=%s attempt=%d", session_id, attempt + 1)

                except TimeoutError:
                    last_error = f"Timeout after {_timeout}s"
                    logger.error(
                        "[GATEWAY] ❌ TIMEOUT | session=%s attempt=%d/%d timeout=%.1fs",
                        session_id,
                        attempt + 1,
                        _retries + 1,
                        _timeout,
                    )
                    result.status = "TIMEOUT"
                    result.error_message = last_error
                    self._stats["timeout"] += 1
                    return result

                except (OSError, ValueError, RuntimeError) as submit_exc:
                    last_error = str(submit_exc)
                    logger.warning(
                        "[GATEWAY] ⚠ SUBMIT_ERROR | session=%s attempt=%d error=%s",
                        session_id,
                        attempt + 1,
                        last_error[:120],
                    )
                    if attempt < _retries:
                        _delay = _retry_delay * (2**attempt)
                        logger.info("[GATEWAY] ↻ RETRY in %.1fs | session=%s", _delay, session_id)
                        await asyncio.sleep(_delay)
                        continue

            if brain_result is None:
                result.status = "ERROR"
                result.error_message = last_error or "All retries exhausted with no result"
                self._stats["error"] += 1
                return result

            result.brain_result = brain_result
            result.sharpe = getattr(brain_result, "real_sharpe", None) or 0.0
            result.fitness = getattr(brain_result, "real_fitness", None) or 0.0
            result.turnover = getattr(brain_result, "real_turnover", None)

            _sim_status = getattr(brain_result, "status", None)
            logger.info(
                "[GATEWAY] ◆ BRAIN_RESULT | session=%s status=%s sharpe=%.4f fitness=%.4f turnover=%.2f",
                session_id,
                _sim_status.value if hasattr(_sim_status, "value") else str(_sim_status),
                result.sharpe,
                result.fitness,
                result.turnover or 0.0,
            )

            if result_router is not None:
                try:
                    slot_info = _build_slot_info(expression, brain_result)
                    result.parsed_result = await result_router.route(slot_info, brain_result)
                    logger.info(
                        "[GATEWAY] ◆ PARSED | session=%s parsed_sharpe=%.3f passed=%s grade=%s",
                        session_id,
                        getattr(result.parsed_result, "raw_sharpe", 0.0),
                        getattr(result.parsed_result, "passed", False),
                        getattr(result.parsed_result, "official_grade", "?"),
                    )
                except (OSError, ValueError, RuntimeError) as route_exc:
                    logger.warning("[GATEWAY] ⚠ ROUTE_FAILED | session=%s error=%s", session_id, route_exc)

            if decision_engine is not None and result.parsed_result is not None:
                try:
                    result.decision = decision_engine.decide_from_parsed_result(result.parsed_result)
                    logger.info(
                        "[GATEWAY] ◆ DECISION | session=%s action=%s priority=%d",
                        session_id,
                        getattr(result.decision, "action", None),
                        getattr(result.decision, "improvement_priority", 0),
                    )
                except (OSError, ValueError, RuntimeError) as dec_exc:
                    logger.warning("[GATEWAY] ⚠ DECIDE_FAILED | session=%s error=%s", session_id, dec_exc)

            result.status = self.classify_status(result)
            self._update_stats(result.status)

            result.failure_reason, result.failure_category = self._classify_failure(result)

        except Exception as gateway_exc:
            logger.error("[GATEWAY] ❌ GATEWAY_ERROR | session=%s error=%s", session_id, gateway_exc, exc_info=True)
            result.status = "ERROR"
            result.error_message = str(gateway_exc)
            self._stats["error"] += 1

        finally:
            result.duration_sec = time.perf_counter() - t0
            try:
                if eid:
                    await self._tel.record_exit(
                        "EvaluationGateway",
                        eid,
                        success=result.status in ("PASS", "NEAR_PASS"),
                        metrics={
                            "status": result.status,
                            "sharpe": result.sharpe,
                            "duration_sec": round(result.duration_sec, 3),
                        },
                    )
            except (OSError, ValueError, RuntimeError):
                pass

        logger.info(
            "[GATEWAY] ✓ COMPLETE | session=%s status=%s sharpe=%.3f fitness=%.3f duration=%.2fs",
            session_id,
            result.status,
            result.sharpe,
            result.fitness,
            result.duration_sec,
        )
        return result

    @staticmethod
    def classify_status(evaluation: EvaluationResult) -> str:
        """将 EvaluationResult 映射为标准状态分类

        规则:
          - PASS:      brain_result.status == BrainSimStatus.PASS
          - NEAR_PASS: sharpe >= NEAR_PASS_SHARPE_THRESHOLD 且非 PASS/ERROR/TIMEOUT
          - FAIL:      brain_result.status == BrainSimStatus.FAIL 或其他非预期状态
          - ERROR:     evaluation.status 已标记为 "ERROR" 或有错误消息
          - TIMEOUT:   evaluation.status 已标记为 "TIMEOUT"

        Args:
            evaluation: 已填充的 EvaluationResult

        Returns:
            状态字符串: "PASS" / "NEAR_PASS" / "FAIL" / "ERROR" / "TIMEOUT"
        """
        if evaluation.status in ("ERROR", "TIMEOUT"):
            return evaluation.status

        _br = evaluation.brain_result
        if _br is None:
            if evaluation.error_message:
                return "ERROR"
            return "FAIL"

        _status = getattr(_br, "status", None)
        if _status == BrainSimStatus.PASS:
            return "PASS"

        if _status == BrainSimStatus.ERROR:
            return "ERROR"

        _sharpe = evaluation.sharpe or 0.0
        _threshold = NEAR_PASS_SHARPE_THRESHOLD
        if _sharpe >= _threshold - _EPSILON and _status != BrainSimStatus.FAIL:
            return "NEAR_PASS"

        return "FAIL"

    def _update_stats(self, status: str) -> None:
        """更新内部统计计数器"""
        key = status.lower() if status.lower() in self._stats else "fail"
        self._stats[key] = self._stats.get(key, 0) + 1

    @staticmethod
    def _classify_failure(evaluation: EvaluationResult) -> tuple[str, str]:
        """将评估结果分类为结构化失败原因

        Returns:
            (failure_reason, failure_category) 元组
            failure_category 取值:
              "syntax_error" | "unknown_field" | "low_sharpe" | "high_turnover"
              | "correlation_high" | "timeout" | "accepted"
        """
        if evaluation.status == "TIMEOUT":
            return "Evaluation timeout exceeded", "timeout"

        if evaluation.status == "PASS":
            return "", "accepted"

        if evaluation.status == "ERROR":
            msg = evaluation.error_message or "Unknown error"
            if "unknown" in msg.lower() or "variable" in msg.lower():
                return msg[:200], "unknown_field"
            if "syntax" in msg.lower() or "parse" in msg.lower():
                return msg[:200], "syntax_error"
            return msg[:200], "syntax_error"

        sharpe = evaluation.sharpe or 0.0
        turnover = evaluation.turnover or 0.0

        if sharpe < 0.5:
            return f"Low Sharpe ratio ({sharpe:.3f})", "low_sharpe"

        if turnover > 0.7:
            return f"High turnover ({turnover:.2f})", "high_turnover"

        parsed = evaluation.parsed_result
        if parsed is not None:
            corr = getattr(parsed, "correlation", None)
            if corr is not None and corr > 0.95:
                return f"High correlation with existing alpha ({corr:.3f})", "correlation_high"

        return f"Evaluation failed with status={evaluation.status} sharpe={sharpe:.3f}", "low_sharpe"

    def reset_stats(self) -> None:
        """重置统计计数器"""
        self._stats = {"total": 0, "pass": 0, "fail": 0, "error": 0, "timeout": 0}


def _build_alpha_proxy(expression: str) -> Any:
    """构建轻量级 alpha 代理对象供 brain_submitter 使用

    brain_submitter 期望接收一个有 simulation_payload 和 alpha_id 属性的对象。
    此函数创建最小代理，避免依赖完整的 Alpha 数据模型。
    """

    class _AlphaProxy:
        def __init__(self, expr: str):
            self.alpha_id = f"eg_{hash(expr) % 100000:05d}"
            self.expression = expr
            self.simulation_payload: dict = {}

    return _AlphaProxy(expression)


def _build_slot_info(expression: str, brain_result: Any) -> Any:
    """构建轻量级 SlotInfo 代理对象供 ResultRouter.route() 使用"""

    class _SlotInfoProxy:
        def __init__(self, expr: str, br: Any):
            self.expression = expr
            self.slot_id = getattr(br, "alpha_id", None) or "unknown"
            self.task_name = f"eval_{getattr(br, 'alpha_id', 'anon')}"
            self.expression = expr

    return _SlotInfoProxy(expression, brain_result)
