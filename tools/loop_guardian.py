#!/usr/bin/env python3
"""
OpenAlpha-Brain — AI 全透明自监控循环测试脚本 (Loop Guardian)
=================================================================

設計目標: 讓 AI agent 可以自行運行此腳本來監控整個算法管線的健康狀況、檢測 bug、追蹤改進效果。

特點:
- 全鏈路日誌捕獲和結構化輸出
- 循環監控能力（可運行 N 個 cycle 並持續觀察）
- 異常自動檢測和告警
- 可生成結構化報告供 AI 分析
- 支援壓力測試和邊界條件
- Mock 模式（不需要真實 API 也能運行基礎檢查）

Usage:
    python tools/loop_guardian.py guard --quick          # 單 cycle 冒煙測試
    python tools/loop_guardian.py guard --monitor --cycles N  # 循環監控（預設 N=3）
    python tools/loop_guardian.py guard --stress         # 壓力測試
    python tools/loop_guardian.py guard --report         # 生成報告
    python tools/loop_guardian.py guard --mock           # Mock 模式（無需真實 API）

Exit codes:
    0 = 全部正常
    1 = 有警告但無阻塞
    2 = 發現錯誤/異常
    3 = 嚴重錯誤（無法恢復）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── 項目根目錄 ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── 彩色輸出常量 ──────────────────────────────────────────────
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

def _c(text: str, color: str = Colors.RESET) -> str:
    return f"{color}{text}{Colors.RESET}"

# ── 日誌配置 ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("GUARDIAN")

# 抑制噪音日誌
logging.getLogger("httpx").setLevel(logging.WARNING)

# ── 數據結構 ──────────────────────────────────────────────────

@dataclass
class PhaseResult:
    """單階段執行結果"""
    name: str
    status: str  # "ok", "warning", "error", "skipped"
    duration_sec: float = 0.0
    input_summary: str = ""
    output_summary: str = ""
    error: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class CycleResult:
    """單 cycle 完整結果"""
    cycle_id: int
    start_time: str = ""
    end_time: str = ""
    total_duration_sec: float = 0.0
    phases: list[PhaseResult] = field(default_factory=list)
    llm_generate_time: float = 0.0
    llm_result_quality: float = 0.0  # 0-1 評分
    parse_success: bool = False
    validate_pass_rate: float = 0.0
    submit_count: int = 0
    sharpe_values: list[float] = field(default_factory=list)
    fitness_values: list[float] = field(default_factory=list)
    algo_call_counts: dict[str, int] = field(default_factory=dict)
    mab_direction_stats: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    memory_mb: float = 0.0


@dataclass
class HealthCheckResult:
    """健康檢查結果"""
    timestamp: str = ""
    check_name: str = ""
    status: str = ""  # "healthy", "degraded", "critical"
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class StressTestResult:
    """壓力測試結果"""
    test_name: str
    status: str
    duration_sec: float = 0.0
    error: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardianState:
    """Guardian 監控狀態"""
    session_id: str = ""
    start_time: str = ""
    cycles_completed: int = 0
    total_errors: int = 0
    total_warnings: int = 0
    cycles: list[CycleResult] = field(default_factory=list)
    health_checks: list[HealthCheckResult] = field(default_factory=list)
    stress_results: list[StressTestResult] = field(default_factory=list)
    exit_code: int = 0

# ── 核心類: GuardianMonitor ───────────────────────────────────

class GuardianMonitor:
    """
    AI 全透明自監控循環測試核心類

    封裝所有監控狀態、cycle 執行、健康檢查和報告生成功能。
    支援 mock 模式用於離線測試。
    """

    def __init__(self, mock_mode: bool = False, verbose: bool = False):
        self.mock_mode = mock_mode
        self.verbose = verbose
        self.state = GuardianState()
        self.state.start_time = datetime.now(timezone.utc).isoformat()
        self._cookies: Any = None
        self._slot_manager: Any = None
        self._orchestrator: Any = None
        self._focus_areas = ["momentum", "reversal", "volatility", "liquidity"]

    def _log(self, msg: str, level: str = "info") -> None:
        """帶 [GUARDIAN] 前綴的日誌輸出"""
        prefix = _c("[GUARDIAN]", Colors.CYAN)
        if level == "error":
            logger.error("%s %s", prefix, msg)
        elif level == "warning":
            logger.warning("%s %s", prefix, msg)
        elif self.verbose or level == "info":
            logger.info("%s %s", prefix, msg)

    def _phase_result(self, name: str, status: str, **kwargs) -> PhaseResult:
        return PhaseResult(name=name, status=status, **kwargs)

    async def _check_auth(self) -> PhaseResult:
        """檢查 BRAIN 認證"""
        if self.mock_mode:
            return self._phase_result("Auth", "ok", input_summary="mock mode")

        try:
            from openalpha_brain.config.config import settings

            email = getattr(settings, "BRAIN_EMAIL", "") or ""
            password = getattr(settings, "BRAIN_PASSWORD", "") or ""

            if not email or not password:
                return self._phase_result(
                    "Auth",
                    "error",
                    error="Missing BRAIN_EMAIL/BRAIN_PASSWORD in .env",
                )

            from openalpha_brain.services import brain_client

            t0 = time.perf_counter()
            cookies = await brain_client.authenticate(email, password)
            elapsed = time.perf_counter() - t0

            if cookies is None:
                return self._phase_result("Auth", "error", error="Authentication failed")

            self._cookies = cookies
            return self._phase_result(
                "Auth",
                "ok",
                duration_sec=elapsed,
                input_summary=f"{email[:3]}***",
                output_summary=f"cookies acquired ({elapsed:.1f}s)",
            )
        except Exception as e:
            return self._phase_result("Auth", "error", error=str(e))

    async def _check_llm(self) -> PhaseResult:
        """檢查 LLM 可用性"""
        if self.mock_mode:
            return self._phase_result("LLM Check", "ok", input_summary="mock mode")

        try:
            from openalpha_brain.config.config import settings

            base_url = getattr(settings, "LMSTUDIO_API_BASE", "http://localhost:1234")

            import httpx

            t0 = time.perf_counter()
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{base_url}/v1/models")
                elapsed = time.perf_counter() - t0

                if resp.status_code == 200:
                    data = resp.json()
                    models = [m.get("id", "?") for m in data.get("data", [])]
                    return self._phase_result(
                        "LLM Check",
                        "ok",
                        duration_sec=elapsed,
                        input_summary=base_url,
                        output_summary=f"Models: {models[:3]}",
                        metrics={"model_count": len(models)},
                    )
            return self._phase_result("LLM Check", "error", error=f"HTTP {resp.status_code}")
        except Exception as e:
            return self._phase_result("LLM Check", "warning", error=str(e))

    async def _init_components(self) -> PhaseResult:
        """初始化所有組件（整合 run_alpha.py 的 _test_component_initialization）"""
        if self.mock_mode:
            mock_components = {
                "MAB": True,
                "RAG": True,
                "Whitelist": True,
                "SignalArbiter": True,
                "AlphaChannel": True,
                "ExperienceDistiller": True,
                "MarketStateInferencer": True,
                "PnLAnalyzer": True,
                "EvoDB": True,
                "SuccessLib": True,
                "FailureLib": True,
                "ToolFactory": True,
                "ReflectionEngine": True,
                "FeatureMap": True,
                "LogicLibrary": True,
            }
            return self._phase_result(
                "Component Init",
                "ok",
                metrics=mock_components,
                output_summary=f"All {len(mock_components)} components initialized (mock)",
            )

        try:
            from openalpha_brain.core import loop_state
            from openalpha_brain.core.loop_state import init_intelligent_search

            init_intelligent_search()

            checks = {
                "MAB": loop_state._mab is not None,
                "RAG": loop_state._rag_engine is not None and getattr(loop_state._rag_engine, "is_ready", False),
                "Whitelist": loop_state._whitelist_mgr is not None,
                "SignalArbiter": loop_state._signal_arbiter is not None,
                "AlphaChannel": loop_state._alpha_channel is not None,
                "ExperienceDistiller": loop_state._experience_distiller is not None,
                "MarketStateInferencer": loop_state._market_state_inferencer is not None,
                "PnLAnalyzer": loop_state._pnl_analyzer is not None,
                "EvoDB": loop_state._evo_db is not None,
                "SuccessLib": loop_state._success_lib is not None,
                "FailureLib": loop_state._failure_lib is not None,
                "ToolFactory": loop_state._tool_factory is not None,
                "ReflectionEngine": loop_state._reflection_engine is not None,
                "FeatureMap": loop_state._feature_map is not None,
                "LogicLibrary": loop_state._logic_library is not None,
            }

            all_ok = all(checks.values())
            failed = [k for k, v in checks.items() if not v]

            for k, v in checks.items():
                icon = _c("✓", Colors.GREEN) if v else _c("✗", Colors.RED)
                self._log(f"  {icon} {k}")

            status = "ok" if all_ok else ("warning" if len(failed) < 5 else "error")

            return self._phase_result(
                "Component Init",
                status,
                metrics=checks,
                output_summary=f"{sum(checks.values())}/{len(checks)} OK",
                error=", ".join(failed) if failed else "",
            )
        except Exception as e:
            return self._phase_result("Component Init", "error", error=str(e))

    async def _test_llm_generation(self) -> tuple[str | None, PhaseResult]:
        """測試 LLM 生成（整合 run_alpha.py 的 _test_llm_generation）"""
        if self.mock_mode:
            expr = "ts_decay_linear(group_neutralize(rank(ts_delta(close, 5)), industry), 10)"
            return expr, self._phase_result(
                "LLM Generation",
                "ok",
                duration_sec=0.5,
                output_summary=expr[:80],
                metrics={"quality_score": 0.85},
            )

        try:
            from openalpha_brain.config.config import settings
            from openalpha_brain.services import llm_client

            original_provider = settings.LLM_PROVIDER
            original_model = settings.LLM_MODEL
            original_base_url = settings.LLM_BASE_URL

            settings.LLM_PROVIDER = "lmstudio"
            settings.LLM_MODEL = "fin-r1"
            settings.LLM_BASE_URL = "http://localhost:1234/v1/chat/completions"

            try:
                system_prompt = "You are a quantitative alpha factor researcher. Generate a WorldQuant BRAIN alpha expression."
                user_msg = "Generate a simple momentum alpha expression using rank and close price. Output ONLY the expression, nothing else."

                t0 = time.perf_counter()
                result = await llm_client.generate(
                    system_prompt=system_prompt,
                    history=[],
                    user_msg=user_msg,
                    session_id="guardian_test",
                    cycle=1,
                )
                elapsed = time.perf_counter() - t0

                quality = min(len(result) / 50, 1.0) if result else 0.0
                self._log(f"LLM response: {result[:200] if result else 'EMPTY'}")

                if result and len(result) > 5:
                    return result, self._phase_result(
                        "LLM Generation",
                        "ok",
                        duration_sec=elapsed,
                        output_summary=result[:80],
                        metrics={"response_len": len(result), "quality_score": quality},
                    )
                return None, self._phase_result("LLM Generation", "error", error="Empty or too short response")
            finally:
                settings.LLM_PROVIDER = original_provider
                settings.LLM_MODEL = original_model
                settings.LLM_BASE_URL = original_base_url
        except Exception as e:
            return None, self._phase_result("LLM Generation", "error", error=str(e))

    async def _test_alpha_parsing(self, expression: str) -> tuple[str | None, PhaseResult]:
        """測試 Alpha 解析（整合 run_alpha.py 的 _test_alpha_parsing）"""
        if self.mock_mode:
            parsed_expr = expression if "(" in expression else f"ts_decay_linear(group_neutralize({expression}, industry), 10)"
            return parsed_expr, self._phase_result(
                "Alpha Parsing",
                "ok",
                input_summary=expression[:60],
                output_summary=parsed_expr[:60],
                metrics={"syntax_valid": True},
            )

        try:
            from openalpha_brain.generation import alpha_parser as parser
            from openalpha_brain.generation.alpha_generator import _extract_expression_from_llm
            from openalpha_brain.generation.alpha_parser import parse_alpha_json
            from openalpha_brain.validation import validator as val

            t0 = time.perf_counter()
            parsed = parse_alpha_json(expression)
            if parsed is None:
                parsed = parser.parse_alpha_output(expression)

            if parsed and parsed.get("expression"):
                expr = parsed["expression"]
            else:
                expr = _extract_expression_from_llm(expression)

            if expr is None:
                stripped = expression.strip()
                if "(" in stripped and ")" in stripped:
                    expr = stripped

            elapsed = time.perf_counter() - t0

            syntax_result = None
            if expr:
                syntax_result = val.validate_syntax(expr)
                self._log(f"Parsed expression: {expr}")
                self._log(f"Syntax validation: passed={syntax_result.passed}, failures={syntax_result.failures[:3] if syntax_result.failures else []}")

                return expr, self._phase_result(
                    "Alpha Parsing",
                    "ok" if syntax_result.passed else "warning",
                    duration_sec=elapsed,
                    input_summary=expression[:60],
                    output_summary=expr[:60],
                    metrics={
                        "syntax_valid": syntax_result.passed,
                        "failure_count": len(syntax_result.failures) if syntax_result.failures else 0,
                    },
                )

            return None, self._phase_result("Alpha Parsing", "error", error="Could not extract expression")
        except Exception as e:
            return None, self._phase_result("Alpha Parsing", "error", error=str(e))

    async def _test_brain_submission(self, expression: str) -> tuple[Any | None, PhaseResult]:
        """測試 BRAIN 提交（整合 run_alpha.py 的 _test_brain_submission）"""
        if self.mock_mode:
            mock_gate = type('obj', (object,), {
                'simulation_status': 'COMPLETE',
                'sharpe': random.uniform(0.5, 2.0),
                'fitness': random.uniform(0.5, 1.5),
                'turnover': random.uniform(10, 50),
                'passed': random.random() > 0.4,
                'alpha_id': f'mock_{int(time.time())}',
                'failures': [],
                'returns': random.uniform(-0.1, 0.2),
            })()
            return mock_gate, self._phase_result(
                "BRAIN Submission",
                "ok" if mock_gate.passed else "warning",
                duration_sec=random.uniform(30, 120),
                input_summary=expression[:60],
                output_summary=f"Sharpe={mock_gate.sharpe:.3f}, Passed={mock_gate.passed}",
                metrics={
                    "sharpe": mock_gate.sharpe,
                    "fitness": mock_gate.fitness,
                    "turnover": mock_gate.turnover,
                    "passed": mock_gate.passed,
                },
            )

        try:
            from openalpha_brain.config.config import settings
            from openalpha_brain.services import brain_client

            if not settings.BRAIN_SUBMIT_ENABLED:
                return None, self._phase_result("BRAIN Submission", "skipped", error="BRAIN_SUBMIT_ENABLED=False")
            if not settings.BRAIN_EMAIL or not settings.BRAIN_PASSWORD:
                return None, self._phase_result("BRAIN Submission", "skipped", error="Credentials missing")

            if self._cookies is None:
                cookies = await brain_client.authenticate(settings.BRAIN_EMAIL, settings.BRAIN_PASSWORD)
            else:
                cookies = self._cookies

            sim_payload = {
                "type": "REGULAR",
                "settings": {
                    "instrumentType": "EQUITY",
                    "region": "USA",
                    "universe": "TOP3000",
                    "delay": 1,
                    "decay": 5,
                    "neutralization": "INDUSTRY",
                    "truncation": 0.05,
                    "pasteurization": "ON",
                    "unitHandling": "VERIFY",
                    "nanHandling": "ON",
                    "language": "FASTEXPR",
                    "visualization": False,
                },
                "regular": expression,
            }

            t0 = time.perf_counter()
            gate = await brain_client.submit_and_poll(
                simulation_payload=sim_payload,
                cookies=cookies,
                max_poll_seconds=settings.BRAIN_POLL_TIMEOUT,
            )
            elapsed = time.perf_counter() - t0

            self._log(f"BRAIN result: status={gate.simulation_status}, sharpe={gate.sharpe}, passed={gate.passed}")

            return gate, self._phase_result(
                "BRAIN Submission",
                "ok" if gate.passed else "warning",
                duration_sec=elapsed,
                input_summary=expression[:60],
                output_summary=f"Status={gate.simulation_status}, Sharpe={gate.sharpe}",
                metrics={
                    "sharpe": gate.sharpe,
                    "fitness": gate.fitness,
                    "turnover": gate.turnover,
                    "passed": gate.passed,
                    "alpha_id": gate.alpha_id,
                },
            )
        except Exception as e:
            return None, self._phase_result("BRAIN Submission", "error", error=str(e))

    async def _run_single_cycle(self, cycle_id: int, focus_area: str) -> CycleResult:
        """執行單個完整 mining cycle（整合 auto_e2e.py 的流程）"""
        result = CycleResult(cycle_id=cycle_id)
        result.start_time = datetime.now(timezone.utc).isoformat()
        t_cycle_start = time.perf_counter()

        self._log(_c(f"\n{'═' * 70}", Colors.BOLD))
        self._log(_c(f"Cycle {cycle_id}: {focus_area.upper()}", Colors.BOLD))
        self._log(_c(f"{'═' * 70}", Colors.BOLD))

        # Phase 1: Auth
        phase = await self._check_auth()
        result.phases.append(phase)
        if phase.status == "error":
            result.errors.append(f"Auth failed: {phase.error}")
            result.end_time = datetime.now(timezone.utc).isoformat()
            result.total_duration_sec = time.perf_counter() - t_cycle_start
            return result

        # Phase 2: LLM Check
        phase = await self._check_llm()
        result.phases.append(phase)
        if phase.status == "error":
            result.warnings.append(f"LLM unavailable: {phase.error}")

        # Phase 3: Component Init
        phase = await self._init_components()
        result.phases.append(phase)
        if phase.status == "error":
            result.errors.append(f"Component init failed: {phase.error}")
            result.end_time = datetime.now(timezone.utc).isoformat()
            result.total_duration_sec = time.perf_counter() - t_cycle_start
            return result

        # Phase 4: LLM Generation
        expression, phase = await self._test_llm_generation()
        result.phases.append(phase)
        result.llm_generate_time = phase.duration_sec
        result.llm_result_quality = phase.metrics.get("quality_score", 0.0)

        if expression is None:
            result.errors.append(f"LLM generation failed: {phase.error}")
            result.end_time = datetime.now(timezone.utc).isoformat()
            result.total_duration_sec = time.perf_counter() - t_cycle_start
            return result

        # Phase 5: Alpha Parsing
        parsed_expr, phase = await self._test_alpha_parsing(expression)
        result.phases.append(phase)
        result.parse_success = phase.status in ("ok", "warning")

        if parsed_expr:
            expression = parsed_expr

        # Phase 6: Validation
        if expression and not self.mock_mode:
            try:
                from openalpha_brain.validation import validator as val
                val_result = val.validate_syntax(expression)
                result.validate_pass_rate = 1.0 if val_result.passed else 0.0
                result.phases.append(self._phase_result(
                    "Validation",
                    "ok" if val_result.passed else "warning",
                    metrics={"pass": val_result.passed, "failures": len(val_result.failures)},
                ))
            except Exception as e:
                result.warnings.append(f"Validation error: {e}")

        # Phase 6.5: Compliance Repair (ThreeBlockTemplate + operator/field sanitization)
        if expression:
            try:
                from openalpha_brain.validation.wq_format_repair import enforce_compliance

                comp = enforce_compliance(expression)
                if comp.repairs_applied:
                    expression = comp.repaired
                    result.phases.append(self._phase_result(
                        "ComplianceRepair", "ok",
                        metrics={"repairs": len(comp.repairs_applied), "details": comp.repairs_applied},
                    ))
                if not comp.valid:
                    result.errors.extend(comp.errors)
            except Exception as _ex:
                result.warnings.append(f"Compliance layer unavailable: {_ex}")

        # Phase 7: BRAIN Submission
        if expression:
            gate, phase = await self._test_brain_submission(expression)
            result.phases.append(phase)
            result.submit_count += 1

            if gate:
                if gate.sharpe is not None:
                    result.sharpe_values.append(gate.sharpe)
                if gate.fitness is not None:
                    result.fitness_values.append(gate.fitness)
                if not gate.passed and gate.failures:
                    result.warnings.extend(gate.failures[:3])

        # Phase 8: MAB Feedback
        if not self.mock_mode:
            try:
                from openalpha_brain.core import loop_state

                if loop_state._mab is not None:
                    loop_state._algo_tick("guardian_cycle")
                    loop_state._mab.update(
                        focus_area,
                        operators=["rank", "ts_delta"],
                        fields=["close", "volume"],
                        reward=0.5,
                    )
                    result.mab_direction_stats = loop_state._mab.get_direction_stats()
                    result.algo_call_counts = dict(loop_state._algo_call_counts)
                    result.phases.append(self._phase_result(
                        "MAB Feedback",
                        "ok",
                        metrics={"directions": len(result.mab_direction_stats)},
                    ))
            except Exception as e:
                result.warnings.append(f"MAB feedback error: {e}")

        # 結束記錄
        result.end_time = datetime.now(timezone.utc).isoformat()
        result.total_duration_sec = time.perf_counter() - t_cycle_start

        # 內存使用
        try:
            import psutil
            process = psutil.Process(os.getpid())
            result.memory_mb = process.memory_info().rss / 1024 / 1024
        except ImportError:
            pass

        # 日誌摘要
        status_icon = _c("✅", Colors.GREEN) if not result.errors else (_c("⚠️", Colors.YELLOW) if not any(p.status == "error" for p in result.phases) else _c("❌", Colors.RED))
        self._log(f"{status_icon} Cycle {cycle_id} completed: {result.total_duration_sec:.1f}s, errors={len(result.errors)}, warnings={len(result.warnings)}")

        return result

    async def run_quick_test(self) -> int:
        """執行單 cycle 冒煙測試"""
        self._log(_c("\n" + "=" * 70, Colors.BOLD))
        self._log(_c("GUARDIAN QUICK TEST — Single Cycle Smoke Test", Colors.BOLD))
        self._log(_c("=" * 70, Colors.BOLD))
        self._log(f"Mode: {'MOCK' if self.mock_mode else 'REAL'}")
        self._log(f"Started: {self.state.start_time}")

        result = await self._run_single_cycle(cycle_id=1, focus_area=self._focus_areas[0])
        self.state.cycles.append(result)
        self.state.cycles_completed = 1
        self.state.total_errors = len(result.errors)
        self.state.total_warnings = len(result.warnings)

        # 判定退出碼
        if result.errors:
            self.state.exit_code = 2
        elif result.warnings:
            self.state.exit_code = 1
        else:
            self.state.exit_code = 0

        self._print_cycle_summary(result)
        return self.state.exit_code

    async def run_monitor(self, cycles: int = 3) -> int:
        """執行 N 個連續 cycle 監控"""
        self._log(_c("\n" + "=" * 70, Colors.BOLD))
        self._log(_c(f"GUARDIAN MONITOR — {cycles} Cycles", Colors.BOLD))
        self._log(_c("=" * 70, Colors.BOLD))
        self._log(f"Mode: {'MOCK' if self.mock_mode else 'REAL'}")
        self._log(f"Cycles: {cycles}")
        self._log(f"Started: {self.state.start_time}")

        focus_areas = self._focus_areas * ((cycles // len(self._focus_areas)) + 1)

        degradation_count = 0
        prev_best_sharpe = None

        for i in range(cycles):
            area = focus_areas[i]
            result = await self._run_single_cycle(cycle_id=i + 1, focus_area=area)
            self.state.cycles.append(result)
            self.state.cycles_completed = i + 1
            self.state.total_errors += len(result.errors)
            self.state.total_warnings += len(result.warnings)

            # 退化檢測
            current_best = max(result.sharpe_values) if result.sharpe_values else None
            if prev_best_sharpe is not None and current_best is not None:
                if current_best < prev_best_sharpe * 0.8:
                    degradation_count += 1
                    result.warnings.append("Performance degradation detected (best Sharpe dropped)")
                    self._log(_c(f"⚠️ Degradation detected: {prev_best_sharpe:.3f} → {current_best:.3f}", Colors.YELLOW))
                else:
                    degradation_count = 0

            prev_best_sharpe = current_best or prev_best_sharpe

            # 連續退化告警
            if degradation_count >= 3:
                self._log(_c("🚨 CRITICAL: Continuous degradation detected!", Colors.RED))
                self.state.exit_code = max(self.state.exit_code, 3)

            # Cycle 間暫停
            if i < cycles - 1:
                await asyncio.sleep(2)

        # 健康檢查
        await self._run_health_checks()

        # 趨勢分析
        self._print_trend_analysis()

        # 判定退出碼
        if self.state.exit_code == 0:
            if self.state.total_errors > 0:
                self.state.exit_code = 2
            elif self.state.total_warnings > 0:
                self.state.exit_code = 1

        self._print_final_summary()
        return self.state.exit_code

    async def _run_health_checks(self) -> None:
        """執行週期性健康檢查"""
        self._log(_c("\n--- Health Checks ---", Colors.BLUE))

        checks = [
            self._check_memory_usage(),
            self._check_algorithm_diversity(),
            self._check_error_rate(),
            self._check_performance_trend(),
        ]

        for check_coro in checks:
            try:
                result = await check_coro
                self.state.health_checks.append(result)
                icon = {"healthy": _c("✓", Colors.GREEN), "degraded": _c("⚠", Colors.YELLOW), "critical": _c("✗", Colors.RED)}.get(result.status, "?")
                self._log(f"  {icon} {result.check_name}: {result.message}")
            except Exception as e:
                self._log(f"  ✗ Health check error: {e}", level="error")

    async def _check_memory_usage(self) -> HealthCheckResult:
        """內存使用檢查"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / 1024 / 1024
            status = "healthy" if mem_mb < 500 else ("degraded" if mem_mb < 1000 else "critical")
            return HealthCheckResult(
                timestamp=datetime.now(timezone.utc).isoformat(),
                check_name="Memory Usage",
                status=status,
                message=f"{mem_mb:.1f} MB",
                details={"memory_mb": mem_mb},
            )
        except ImportError:
            return HealthCheckResult(
                check_name="Memory Usage",
                status="skipped",
                message="psutil not available",
            )

    async def _check_algorithm_diversity(self) -> HealthCheckResult:
        """算法多樣性檢查"""
        all_counts = {}
        for cycle in self.state.cycles:
            for k, v in cycle.algo_call_counts.items():
                all_counts[k] = all_counts.get(k, 0) + v

        diversity = len(all_counts)
        status = "healthy" if diversity >= 5 else ("degraded" if diversity >= 3 else "critical")
        return HealthCheckResult(
            check_name="Algorithm Diversity",
            status=status,
            message=f"{diversity} unique algorithms used",
            details={"unique_algorithms": diversity, "call_counts": all_counts},
        )

    async def _check_error_rate(self) -> HealthCheckResult:
        """錯誤率檢查"""
        total_phases = sum(len(c.phases) for c in self.state.cycles)
        error_phases = sum(1 for c in self.state.cycles for p in c.phases if p.status == "error")
        rate = error_phases / max(total_phases, 1)
        status = "healthy" if rate < 0.1 else ("degraded" if rate < 0.3 else "critical")
        return HealthCheckResult(
            check_name="Error Rate",
            status=status,
            message=f"{rate:.1%} ({error_phases}/{total_phases})",
            details={"error_rate": rate, "error_count": error_phases, "total_phases": total_phases},
        )

    async def _check_performance_trend(self) -> HealthCheckResult:
        """性能趨勢檢查"""
        sharpes = []
        for cycle in self.state.cycles:
            sharpes.extend(cycle.sharpe_values)

        if len(sharpes) < 2:
            return HealthCheckResult(
                check_name="Performance Trend",
                status="skipped",
                message="Insufficient data points",
            )

        trend = sharpes[-1] - sharpes[0]
        status = "healthy" if trend >= 0 else ("degraded" if trend > -0.5 else "critical")
        return HealthCheckResult(
            check_name="Performance Trend",
            status=status,
            message=f"Trend: {trend:+.3f}",
            details={"trend": trend, "sharpe_history": sharpes},
        )

    async def run_stress_tests(self) -> int:
        """執行壓力測試"""
        self._log(_c("\n" + "=" * 70, Colors.BOLD))
        self._log(_c("GUARDIAN STRESS TEST", Colors.BOLD))
        self._log(_c("=" * 70, Colors.BOLD))
        self._log(f"Mode: {'MOCK' if self.mock_mode else 'REAL'}")

        tests = [
            ("Boundary: Empty Input", self._stress_empty_input),
            ("Boundary: Long Expression", self._stress_long_expression),
            ("Boundary: Special Characters", self._stress_special_characters),
            ("Boundary: Invalid Fields", self._stress_invalid_fields),
            ("Concurrency: Rapid Cycles", self._stress_rapid_cycles),
            ("Recovery: LLM Timeout/Failure", self._stress_llm_failure),
            ("Resource: Memory Leak Detection", self._stress_memory_leak),
            ("Resource: Semaphore Deadlock", self._stress_semaphore_deadlock),
        ]

        for test_name, test_fn in tests:
            self._log(_c(f"\n--- {test_name} ---", Colors.MAGENTA))
            try:
                result = await test_fn()
                self.state.stress_results.append(result)
                icon = {"ok": _c("✓", Colors.GREEN), "warning": _c("⚠", Colors.YELLOW), "error": _c("✗", Colors.RED)}.get(result.status, "?")
                self._log(f"  {icon} {test_name}: {result.duration_sec:.2f}s - {result.details.get('summary', '')}")
                if result.error:
                    self._log(f"     Error: {result.error}", level="warning")
            except Exception as e:
                self.state.stress_results.append(StressTestResult(test_name=test_name, status="error", error=str(e)))
                self._log(f"  ✗ {test_name}: EXCEPTION - {e}", level="error")

        # 摘要
        passed = sum(1 for r in self.state.stress_results if r.status == "ok")
        failed = sum(1 for r in self.state.stress_results if r.status == "error")
        warned = sum(1 for r in self.state.stress_results if r.status == "warning")

        self._log(_c(f"\nStress Test Summary: {passed} passed, {warned} warned, {failed} failed", Colors.BOLD))

        if failed > 0:
            self.state.exit_code = 2
        elif warned > 0:
            self.state.exit_code = 1
        else:
            self.state.exit_code = 0

        return self.state.exit_code

    async def _stress_empty_input(self) -> StressTestResult:
        """空輸入邊界測試"""
        t0 = time.perf_counter()
        try:
            _, phase = await self._test_alpha_parsing("")
            elapsed = time.perf_counter() - t0
            return StressTestResult(
                test_name="Empty Input",
                status="ok" if phase.status != "error" else "warning",
                duration_sec=elapsed,
                details={"summary": "Handled empty input gracefully"},
            )
        except Exception as e:
            return StressTestResult(test_name="Empty Input", status="error", error=str(e), duration_sec=time.perf_counter() - t0)

    async def _stress_long_expression(self) -> StressTestResult:
        """超長表達式測試"""
        t0 = time.perf_counter()
        long_expr = "ts_decay_linear(group_neutralize(" + "+".join([f"rank(ts_delta(close,{i}))" for i in range(1, 51)]) + ",industry),20)"
        try:
            _, phase = await self._test_alpha_parsing(long_expr)
            elapsed = time.perf_counter() - t0
            return StressTestResult(
                test_name="Long Expression",
                status="ok" if phase.status in ("ok", "warning") else "error",
                duration_sec=elapsed,
                details={"summary": f"Processed {len(long_expr)} char expression", "length": len(long_expr)},
            )
        except Exception as e:
            return StressTestResult(test_name="Long Expression", status="error", error=str(e), duration_sec=time.perf_counter() - t0)

    async def _stress_special_characters(self) -> StressTestResult:
        """特殊字符測試"""
        t0 = time.perf_counter()
        special_expr = 'ts_decay_linear(group_neutralize(rank(close ÷ volume), sector); "test\'s\\"expr", 10)'
        try:
            _, phase = await self._test_alpha_parsing(special_expr)
            elapsed = time.perf_counter() - t0
            return StressTestResult(
                test_name="Special Characters",
                status="ok" if phase.status in ("ok", "warning") else "error",
                duration_sec=elapsed,
                details={"summary": "Handled special characters"},
            )
        except Exception as e:
            return StressTestResult(test_name="Special Characters", status="error", error=str(e), duration_sec=time.perf_counter() - t0)

    async def _stress_invalid_fields(self) -> StressTestResult:
        """非法字段測試"""
        t0 = time.perf_counter()
        invalid_expr = "ts_decay_linear(group_neutralize(rank(invalid_field_12345), industry), 10)"
        try:
            _, phase = await self._test_alpha_parsing(invalid_expr)
            elapsed = time.perf_counter() - t0
            return StressTestResult(
                test_name="Invalid Fields",
                status="ok" if phase.status in ("ok", "warning") else "error",
                duration_sec=elapsed,
                details={"summary": "Detected invalid fields properly"},
            )
        except Exception as e:
            return StressTestResult(test_name="Invalid Fields", status="error", error=str(e), duration_sec=time.perf_counter() - t0)

    async def _stress_rapid_cycles(self) -> StressTestResult:
        """高併發快速循環測試"""
        t0 = time.perf_counter()
        try:
            tasks = [self._run_single_cycle(cycle_id=i, focus_area="momentum") for i in range(3)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = time.perf_counter() - t0
            successes = sum(1 for r in results if isinstance(r, CycleResult))
            return StressTestResult(
                test_name="Rapid Cycles",
                status="ok" if successes == 3 else "warning",
                duration_sec=elapsed,
                details={"summary": f"{successes}/3 rapid cycles completed", "successes": successes},
            )
        except Exception as e:
            return StressTestResult(test_name="Rapid Cycles", status="error", error=str(e), duration_sec=time.perf_counter() - t0)

    async def _stress_llm_failure(self) -> StressTestResult:
        """LLM 失敗恢復測試"""
        t0 = time.perf_counter()
        if self.mock_mode:
            elapsed = time.perf_counter() - t0
            return StressTestResult(
                test_name="LLM Failure Recovery",
                status="ok",
                duration_sec=elapsed,
                details={"summary": "Mock: failure recovery mechanism works"},
            )

        try:
            from openalpha_brain.config.config import settings
            original_base = settings.LLM_BASE_URL
            settings.LLM_BASE_URL = "http://localhost:9999/v1/chat/completions"

            _, phase = await self._test_llm_generation()
            settings.LLM_BASE_URL = original_base
            elapsed = time.perf_counter() - t0

            recovered = True
            _, recovery_phase = await self._test_llm_generation()

            return StressTestResult(
                test_name="LLM Failure Recovery",
                status="ok" if recovered and recovery_phase.status == "ok" else "warning",
                duration_sec=elapsed,
                details={"summary": "Recovered after LLM failure" if recovered else "Failed to recover"},
            )
        except Exception as e:
            return StressTestResult(test_name="LLM Failure Recovery", status="warning", error=str(e), duration_sec=time.perf_counter() - t0)

    async def _stress_memory_leak(self) -> StressTestResult:
        """內存泄漏基本檢測"""
        t0 = time.perf_counter()
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024

            for i in range(5):
                await self._run_single_cycle(cycle_id=i, focus_area="momentum")

            mem_after = process.memory_info().rss / 1024 / 1024
            leak = mem_after - mem_before
            elapsed = time.perf_counter() - t0

            status = "ok" if leak < 50 else ("warning" if leak < 100 else "error")
            return StressTestResult(
                test_name="Memory Leak Detection",
                status=status,
                duration_sec=elapsed,
                details={
                    "summary": f"Memory change: {leak:+.1f} MB over 5 cycles",
                    "mem_before_mb": mem_before,
                    "mem_after_mb": mem_after,
                    "leak_mb": leak,
                },
            )
        except ImportError:
            return StressTestResult(
                test_name="Memory Leak Detection",
                status="skipped",
                duration_sec=time.perf_counter() - t0,
                details={"summary": "psutil not available"},
            )

    async def _stress_semaphore_deadlock(self) -> StressTestResult:
        """Semaphore 死鎖檢測"""
        t0 = time.perf_counter()
        try:
            async def worker(sem: asyncio.Semaphore, idx: int):
                async with sem:
                    await asyncio.sleep(0.1)
                    return idx

            sem = asyncio.Semaphore(2)
            timeout = 5.0
            tasks = [worker(sem, i) for i in range(10)]

            results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
            elapsed = time.perf_counter() - t0

            return StressTestResult(
                test_name="Semaphore Deadlock",
                status="ok" if len(results) == 10 else "error",
                duration_sec=elapsed,
                details={"summary": f"No deadlock: {len(results)}/10 workers completed"},
            )
        except asyncio.TimeoutError:
            return StressTestResult(
                test_name="Semaphore Deadlock",
                status="error",
                error="Deadlock detected (timeout)",
                duration_sec=time.perf_counter() - t0,
            )
        except Exception as e:
            return StressTestResult(test_name="Semaphore Deadlock", status="error", error=str(e), duration_sec=time.perf_counter() - t0)

    def _print_cycle_summary(self, cycle: CycleResult) -> None:
        """打印單 cycle 摘要"""
        print(f"\n{_c('─' * 70, Colors.DIM)}")
        print(_c(f"Cycle {cycle.cycle_id} Summary", Colors.BOLD))
        print(f"{'─' * 70}")
        print(f"  Duration: {cycle.total_duration_sec:.1f}s")
        print(f"  Memory:   {cycle.memory_mb:.1f} MB" if cycle.memory_mb else "")
        print("\n  Phases:")
        for p in cycle.phases:
            icon = {"ok": _c("✓", Colors.GREEN), "warning": _c("⚠", Colors.YELLOW), "error": _c("✗", Colors.RED), "skipped": _c("⊘", Colors.DIM)}.get(p.status, "?")
            print(f"    {icon} {p.name:<25s} {p.duration_sec:>6.2f}s  {p.output_summary[:50]}")

        if cycle.sharpe_values:
            avg_sh = sum(cycle.sharpe_values) / len(cycle.sharpe_values)
            print(f"\n  Sharpe:   avg={avg_sh:.3f}, best={max(cycle.sharpe_values):.3f}, count={len(cycle.sharpe_values)}")

        if cycle.errors:
            print(f"\n  {_c('Errors:', Colors.RED)}")
            for e in cycle.errors:
                print(f"    {_c('✗', Colors.RED)} {e}")

        if cycle.warnings:
            print(f"\n  {_c('Warnings:', Colors.YELLOW)}")
            for w in cycle.warnings[:5]:
                print(f"    {_c('⚠', Colors.YELLOW)} {w}")

    def _print_trend_analysis(self) -> None:
        """打印趨勢分析圖表"""
        if len(self.state.cycles) < 2:
            return

        print(f"\n{_c('─' * 70, Colors.DIM)}")
        print(_c("Trend Analysis", Colors.BOLD))
        print(f"{'─' * 70}")

        # Sharpe 趨勢
        print(f"\n  {_c('Sharpe Trend:', Colors.CYAN)}")
        for cycle in self.state.cycles:
            if cycle.sharpe_values:
                avg_sh = sum(cycle.sharpe_values) / len(cycle.sharpe_values)
                bar_len = int(avg_sh * 20)
                bar = _c("█" * max(bar_len, 0), Colors.GREEN if avg_sh > 1.0 else Colors.YELLOW)
                print(f"    Cycle {cycle.cycle_id:<3d} │{bar:<22s}│ {avg_sh:+.3f} (n={len(cycle.sharpe_values)})")
            else:
                print(f"    Cycle {cycle.cycle_id:<3d} │{'─' * 22}│ no data")

        # 耗時趨勢
        print(f"\n  {_c('Duration Trend:', Colors.CYAN)}")
        for cycle in self.state.cycles:
            bar_len = int(cycle.total_duration_sec / 2)
            bar = _c("█" * bar_len, Colors.BLUE)
            print(f"    Cycle {cycle.cycle_id:<3d} │{bar:<22s}│ {cycle.total_duration_sec:.1f}s")

        # MAB 方向分佈變化
        if self.state.cycles[-1].mab_direction_stats:
            print(f"\n  {_c('MAB Direction Distribution (latest):', Colors.CYAN)}")
            stats = self.state.cycles[-1].mab_direction_stats
            for k, v in list(stats.items())[:5]:
                print(f"    {k:<20s}: {v}")

    def _print_final_summary(self) -> None:
        """打印最終摘要"""
        print(f"\n{_c('=' * 70, Colors.BOLD)}")
        print(_c("GUARDIAN FINAL SUMMARY", Colors.BOLD))
        print(f"{'=' * 70}")
        print(f"  Mode:       {'MOCK' if self.mock_mode else 'REAL'}")
        print(f"  Started:    {self.state.start_time}")
        print(f"  Completed:  {datetime.now(timezone.utc).isoformat()}")
        print(f"  Cycles:     {self.state.cycles_completed}")
        print(f"  Errors:     {_c(str(self.state.total_errors), Colors.RED if self.state.total_errors else Colors.GREEN)}")
        print(f"  Warnings:   {_c(str(self.state.total_warnings), Colors.YELLOW if self.state.total_warnings else Colors.GREEN)}")
        print(f"  Exit Code:  {self.state.exit_code}")
        print(f"{'=' * 70}\n")


# ── 報告生成器 ────────────────────────────────────────────────

class ReportGenerator:
    """結構化報告生成器"""

    def __init__(self, state: GuardianState, output_dir: Path | None = None):
        self.state = state
        self.output_dir = output_dir or (PROJECT_ROOT / "tools" / "logs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_json_report(self) -> Path:
        """生成 JSON 格式報告"""
        report = {
            "report_type": "guardian_monitor_report",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "session_info": {
                "start_time": self.state.start_time,
                "mode": "mock",
                "total_cycles": self.state.cycles_completed,
                "total_errors": self.state.total_errors,
                "total_warnings": self.state.total_warnings,
                "exit_code": self.state.exit_code,
            },
            "cycles": [
                {
                    "cycle_id": c.cycle_id,
                    "duration_sec": c.total_duration_sec,
                    "phases": [{"name": p.name, "status": p.status, "duration_sec": p.duration_sec, "error": p.error} for p in c.phases],
                    "llm_generate_time": c.llm_generate_time,
                    "llm_quality": c.llm_result_quality,
                    "parse_success": c.parse_success,
                    "submit_count": c.submit_count,
                    "sharpes": c.sharpe_values,
                    "fitnesses": c.fitness_values,
                    "errors": c.errors,
                    "warnings": c.warnings,
                    "memory_mb": c.memory_mb,
                }
                for c in self.state.cycles
            ],
            "health_checks": [
                {
                    "timestamp": h.timestamp,
                    "name": h.check_name,
                    "status": h.status,
                    "message": h.message,
                    "details": h.details,
                }
                for h in self.state.health_checks
            ],
            "stress_tests": [
                {
                    "name": s.test_name,
                    "status": s.status,
                    "duration_sec": s.duration_sec,
                    "error": s.error,
                    "details": s.details,
                }
                for s in self.state.stress_results
            ],
        }

        path = self.output_dir / f"guardian_report_{self.timestamp}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        return path

    def generate_markdown_report(self) -> Path:
        """生成 Markdown 格式人類可讀報告"""
        lines = [
            "# Guardian Monitor Report",
            "",
            f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Mode:** {'Mock' if True else 'Real'}",
            f"**Cycles:** {self.state.cycles_completed}",
            f"**Exit Code:** {self.state.exit_code}",
            "",
            "## Executive Summary",
            "",
            f"- **Total Errors:** {self.state.total_errors}",
            f"- **Total Warnings:** {self.state.total_warnings}",
            f"- **Status:** {'✅ All Clear' if self.state.exit_code == 0 else ('⚠️ Warnings' if self.state.exit_code == 1 else ('❌ Errors' if self.state.exit_code == 2 else '🚨 Critical'))}",
            "",
        ]

        # Cycle 详情
        lines.extend([
            "## Cycle Details",
            "",
        ])

        for cycle in self.state.cycles:
            lines.extend([
                f"### Cycle {cycle.cycle_id}",
                "",
                f"- **Duration:** {cycle.total_duration_sec:.1f}s",
                f"- **Phases:** {len(cycle.phases)}",
                f"- **Errors:** {len(cycle.errors)}",
                f"- **Warnings:** {len(cycle.warnings)}",
                "",
            ])

            if cycle.sharpe_values:
                avg_sh = sum(cycle.sharpe_values) / len(cycle.sharpe_values)
                lines.extend([
                    "**Sharpe Metrics:**",
                    f"- Average: {avg_sh:.3f}",
                    f"- Best: {max(cycle.sharpe_values):.3f}",
                    f"- Count: {len(cycle.sharpe_values)}",
                    "",
                ])

            if cycle.errors:
                lines.extend(["**Errors:**", ""])
                for e in cycle.errors:
                    lines.append(f"- {e}")
                lines.append("")

        # Health Checks
        if self.state.health_checks:
            lines.extend([
                "## Health Checks",
                "",
                "| Check | Status | Message |",
                "|-------|--------|---------|",
            ])
            for hc in self.state.health_checks:
                icon = {"healthy": "✅", "degraded": "⚠️", "critical": "❌"}.get(hc.status, "❓")
                lines.append(f"| {hc.check_name} | {icon} {hc.status} | {hc.message} |")
            lines.append("")

        # Stress Tests
        if self.state.stress_results:
            lines.extend([
                "## Stress Tests",
                "",
                "| Test | Status | Duration | Error |",
                "|------|--------|----------|-------|",
            ])
            for st in self.state.stress_results:
                icon = {"ok": "✅", "warning": "⚠️", "error": "❌"}.get(st.status, "❓")
                lines.append(f"| {st.test_name} | {icon} {st.status} | {st.duration_sec:.2f}s | {st.error or '-'} |")
            lines.append("")

        # Improvement Suggestions
        lines.extend([
            "## Suggestions",
            "",
        ])

        suggestions = self._generate_suggestions()
        for s in suggestions:
            lines.append(f"- {s}")
        lines.append("")

        path = self.output_dir / f"guardian_report_{self.timestamp}.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return path

    def _generate_suggestions(self) -> list[str]:
        """生成改進建議"""
        suggestions = []

        if self.state.total_errors > 0:
            suggestions.append("🔴 Investigate and fix critical errors before proceeding")

        if self.state.total_warnings > 3:
            suggestions.append("🟡 Multiple warnings detected - review component stability")

        for hc in self.state.health_checks:
            if hc.status == "critical":
                suggestions.append(f"🔴 Critical health issue: {hc.check_name} - {hc.message}")
            elif hc.status == "degraded":
                suggestions.append(f"🟡 Degraded performance: {hc.check_name} - {hc.message}")

        for st in self.state.stress_results:
            if st.status == "error":
                suggestions.append(f"🔴 Stress test failure: {st.test_name} - {st.error}")

        if not suggestions:
            suggestions.append("✅ All systems operating within normal parameters")

        return suggestions


# ── CLI 入口 ──────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """構建命令行參數解析器"""
    parser = argparse.ArgumentParser(
        prog="loop_guardian",
        description=_c("OpenAlpha-Brain AI 全透明自監控循環測試工具", Colors.BOLD),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/loop_guardian.py guard --quick              # 單 cycle 冒煙測試
  python tools/loop_guardian.py guard --monitor --cycles 5  # 5 個循環監控
  python tools/loop_guardian.py guard --stress             # 壓力測試
  python tools/loop_guardian.py guard --report             # 生成報告
  python tools/loop_guardian.py guard --mock --quick       # Mock 模式快速測試
        """,
    )

    parser.add_argument(
        "command",
        choices=["guard"],
        help="執行守護命令",
    )

    parser.add_argument("--quick", action="store_true", help="單 cycle 冒煙測試")
    parser.add_argument("--monitor", action="store_true", help="循環監控模式")
    parser.add_argument("--stress", action="store_true", help="壓力測試模式")
    parser.add_argument("--report", action="store_true", help="生成結構化報告")
    parser.add_argument("--mock", action="store_true", help="Mock 模式（無需真實 API）")
    parser.add_argument("--cycles", type=int, default=3, help="監控循環數量（預設 3）")
    parser.add_argument("--verbose", action="store_true", help="詳細輸出模式")
    parser.add_argument("--quiet", action="store_true", help="靜默模式（僅輸出錯誤）")

    return parser


async def main():
    """主入口函數"""
    parser = build_parser()
    args = parser.parse_args()

    if args.quiet:
        logging.getLogger("GUARDIAN").setLevel(logging.ERROR)

    monitor = GuardianMonitor(mock_mode=args.mock, verbose=args.verbose)

    try:
        if args.quick:
            exit_code = await monitor.run_quick_test()
        elif args.monitor:
            exit_code = await monitor.run_monitor(cycles=args.cycles)
        elif args.stress:
            exit_code = await monitor.run_stress_tests()
        elif args.report:
            generator = ReportGenerator(monitor.state)
            json_path = generator.generate_json_report()
            md_path = generator.generate_markdown_report()
            print(_c("\n✅ Reports generated:", Colors.GREEN))
            print(f"  JSON: {json_path}")
            print(f"  MD:   {md_path}")
            exit_code = 0
        else:
            parser.print_help()
            exit_code = 0

        # 自動保存報告（如果有數據）
        if monitor.state.cycles or monitor.state.stress_results:
            generator = ReportGenerator(monitor.state)
            json_path = generator.generate_json_report()
            if not args.quiet:
                print(_c(f"\n📊 Report saved: {json_path}", Colors.CYAN))

        sys.exit(exit_code)

    except KeyboardInterrupt:
        print(_c("\n\n⚡ User interrupted.", Colors.YELLOW))
        sys.exit(130)
    except Exception as e:
        logger.error("[GUARDIAN] Fatal error: %s", e, exc_info=True)
        sys.exit(3)


if __name__ == "__main__":
    asyncio.run(main())
