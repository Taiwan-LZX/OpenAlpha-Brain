#!/usr/bin/env python3
"""
Unified Health Check Diagnostic Script
=======================================
合并 BRAIN 连接验证、LLM 连接测试、模块冒烟测试、并发 Slot 测试于一体。

Usage:
    python tools/unified_health.py health              # 快速模式（默认）
    python tools/unified_health.py health --deep        # 深度模式（含 API 调用）
    python tools/unified_health.py health --dry-run     # 仅配置检查（CI/CD 友好）

Exit codes:
    0 = 全部通过
    1 = 配置问题 (.env 缺失或变量错误)
    2 = 认证失败
    3 = 提交失败
    4 = 网络问题
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── 彩色输出 ──────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def _ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


def _info(msg: str) -> None:
    print(f"  {BLUE}ℹ{RESET} {msg}")


def _header(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}")
    print("─" * len(title))


# ── 检查结果收集器 ────────────────────────────────────────────────

class CheckResult:
    """单条检查结果"""

    def __init__(self, category: str, name: str, status: str, detail: str = ""):
        self.category = category
        self.name = name
        self.status = status  # "PASS" / "FAIL" / "WARN" / "SKIP"
        self.detail = detail


class HealthReport:
    """健康检查报告"""

    def __init__(self):
        self.results: list[CheckResult] = []
        self.start_time: float = 0.0
        self._counts: dict[str, dict[str, int]] = {}

    def add(self, category: str, name: str, status: str, detail: str = "") -> None:
        self.results.append(CheckResult(category, name, status, detail))
        if category not in self._counts:
            self._counts[category] = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIP": 0}
        self._counts[category][status] += 1

    def get_category_status(self, category: str) -> str:
        c = self._counts.get(category, {})
        if c.get("FAIL", 0) > 0:
            return "FAIL"
        if c.get("WARN", 0) > 0:
            return "WARN"
        return "PASS"

    @property
    def total_pass(self) -> int:
        return sum(c.get("PASS", 0) for c in self._counts.values())

    @property
    def total_fail(self) -> int:
        return sum(c.get("FAIL", 0) for c in self._counts.values())

    @property
    def total_warn(self) -> int:
        return sum(c.get("WARN", 0) for c in self._counts.values())

    @property
    def total_checks(self) -> int:
        return len(self.results)


# ════════════════════════════════════════════════════════════════════
#  检查函数
# ════════════════════════════════════════════════════════════════════

async def check_env_config(report: HealthReport) -> dict[str, Any]:
    """Check 1: .env 配置文件存在性和必要变量"""
    _header("Check 1: 环境配置")

    env_file = _PROJECT_ROOT / ".env"
    results = {
        "env_exists": False,
        "brain_email": False,
        "brain_password": False,
        "brain_submit_enabled": False,
        "llm_api_key": False,
        "email": "",
        "password": "",
        "issues": [],
    }

    if not env_file.exists():
        _fail(".env 文件未找到")
        _info(f"期望路径: {env_file}")
        report.add("Configuration", ".env 文件存在性", "FAIL", "文件缺失")
        results["issues"].append("MISSING_ENV_FILE")
        return results

    _ok(f".env 文件已找到: {env_file}")
    report.add("Configuration", ".env 文件存在性", "PASS")
    results["env_exists"] = True

    env_vars = {}
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env_vars[key.strip()] = value.strip()

    if env_vars.get("BRAIN_EMAIL") and "@" in env_vars["BRAIN_EMAIL"]:
        email = env_vars["BRAIN_EMAIL"]
        masked = email[:3] + "***" + email.split("@")[1] if len(email) > 5 else "***@***"
        _ok(f"BRAIN_EMAIL: {masked}")
        report.add("Configuration", "BRAIN_EMAIL", "PASS")
        results["brain_email"] = True
        results["email"] = email
    else:
        _fail("BRAIN_EMAIL 缺失或格式无效")
        report.add("Configuration", "BRAIN_EMAIL", "FAIL", "缺失或格式无效")
        results["issues"].append("MISSING_BRAIN_EMAIL")

    if env_vars.get("BRAIN_PASSWORD") and len(env_vars["BRAIN_PASSWORD"]) > 3:
        pwd = env_vars["BRAIN_PASSWORD"]
        masked = pwd[:2] + "*" * (len(pwd) - 2) + pwd[-1:] if len(pwd) > 4 else "***"
        _ok(f"BRAIN_PASSWORD: {masked} ({len(pwd)} 字符)")
        report.add("Configuration", "BRAIN_PASSWORD", "PASS")
        results["brain_password"] = True
        results["password"] = pwd
    else:
        _fail("BRAIN_PASSWORD 缺失或过短")
        report.add("Configuration", "BRAIN_PASSWORD", "FAIL", "缺失或过短")
        results["issues"].append("MISSING_BRAIN_PASSWORD")

    submit_enabled = env_vars.get("BRAIN_SUBMIT_ENABLED", "false").lower() in ("true", "1", "yes")
    if submit_enabled:
        _ok("BRAIN_SUBMIT_ENABLED=true")
        report.add("Configuration", "BRAIN_SUBMIT_ENABLED", "PASS")
        results["brain_submit_enabled"] = True
    else:
        _warn("BRAIN_SUBMIT_ENABLED=false (提交已禁用)")
        report.add("Configuration", "BRAIN_SUBMIT_ENABLED", "WARN", "已禁用")
        results["issues"].append("SUBMISSION_DISABLED")

    if env_vars.get("LLM_API_KEY") and len(env_vars["LLM_API_KEY"]) > 10:
        key = env_vars["LLM_API_KEY"]
        masked = key[:6] + "..." + key[-4:]
        _ok(f"LLM_API_KEY: {masked}")
        report.add("Configuration", "LLM_API_KEY", "PASS")
        results["llm_api_key"] = True
    else:
        _warn("LLM_API_KEY 缺失或过短 (LLM 功能受限)")
        report.add("Configuration", "LLM_API_KEY", "WARN", "缺失或过短")
        results["issues"].append("MISSING_LLM_KEY")

    return results


async def check_brain_auth(report: HealthReport, email: str, password: str) -> dict[str, Any]:
    """Check 2: BRAIN 平台认证测试"""
    _header("Check 2: BRAIN 平台认证")

    try:
        from openalpha_brain.services.brain_client import authenticate

        cookies = await authenticate(email, password)

        cookie_count = 0
        has_session = False

        if cookies is None:
            _warn("认证返回 None (可能网络问题)")
            report.add("Authentication", "BRAIN 认证", "WARN", "返回 None")
            return {"success": False, "error": "NONE_RESPONSE"}

        if hasattr(cookies, "items"):
            cookie_count = len(cookies)
            has_session = any("session" in k.lower() for k in cookies.keys())
        elif hasattr(cookies, "__iter__") and not isinstance(cookies, str):
            cookie_list = list(cookies)
            cookie_count = len(cookie_list)
            has_session = any(
                (hasattr(c, "name") and "session" in c.name.lower())
                or (isinstance(c, tuple) and len(c) > 0 and "session" in str(c[0]).lower())
                for c in cookie_list
            )
        elif hasattr(cookies, "name"):
            cookie_count = 1
            has_session = (
                "session" in cookies.name.lower() if hasattr(cookies, "name") else False
            )
        else:
            cookie_count = 1 if cookies else 0
            has_session = bool(cookies)

        if has_session or cookie_count > 0:
            _ok(f"认证成功 ({cookie_count} 个 cookie)")
            report.add("Authentication", "BRAIN 认证", "PASS", f"{cookie_count} cookies")
            return {"success": True, "cookies": cookies}
        else:
            _fail(f"认证未返回有效会话 (type={type(cookies).__name__})")
            report.add("Authentication", "BRAIN 认证", "FAIL", "无有效 session")
            return {"success": False, "error": "NO_SESSION_COOKIE"}

    except ImportError as exc:
        _fail(f"无法导入 brain_client: {exc}")
        report.add("Authentication", "BRAIN 认证", "FAIL", f"ImportError: {exc}")
        return {"success": False, "error": "IMPORT_ERROR"}
    except Exception as exc:
        _fail(f"认证失败: {exc}")
        report.add("Authentication", "BRAIN 认证", "FAIL", str(exc))
        return {"success": False, "error": str(exc)}


async def check_llm_connection(report: HealthReport, full_test: bool = False) -> dict[str, Any]:
    """Check 3: LLM 连接与生成测试"""
    _header("Check 3: LLM 连接测试")

    try:
        from openalpha_brain.config.config import settings

        provider = getattr(settings, "LLM_PROVIDER", "unknown")
        model = getattr(settings, "LLM_MODEL", "unknown")
        base_url = getattr(settings, "LLM_BASE_URL", None)

        _info(f"Provider: {provider}")
        _info(f"Model: {model}")
        _info(f"Base URL: {base_url or 'provider default'}")
        report.add("LLM", "LLM 配置加载", "PASS", f"{provider}/{model}")

        if not full_test:
            _info("跳过实际 LLM 调用 (需 --deep 模式)")
            report.add("LLM", "LLM 实际调用", "SKIP", "非 deep 模式")
            return {"config_ok": True, "generation_skipped": True}

        from openalpha_brain.services import llm_client

        test_prompt = (
            "You are a quantitative finance expert. "
            "Generate a VERY SHORT alpha factor expression using WorldQuant syntax. "
            "Return ONLY the expression, nothing else. "
            "Use format: rank(some_operator(field, window))"
        )

        _info(f"发送测试 prompt 到 {provider}...")
        start = time.monotonic()

        try:
            response = await llm_client.generate(
                system_prompt="You are a quantitative finance expert. Generate alpha factors for WorldQuant BRAIN platform.",
                history=[],
                user_msg=test_prompt,
                session_id="health-check",
                cycle=0,
            )
            elapsed = time.monotonic() - start

            if response and len(response.strip()) > 0:
                _ok(f"LLM 响应成功 ({elapsed:.1f}s, {len(response)} 字符)")
                _info(f"响应预览: {response.strip()[:200]}...")

                has_alpha_kw = any(
                    kw in response.lower()
                    for kw in ["rank", "ts_", "delta", "corr", "mean", "stddev", "close", "volume"]
                )

                if has_alpha_kw:
                    _ok("响应包含 alpha 表达式关键词")
                    report.add("LLM", "LLM 实际生成", "PASS", f"{elapsed:.1f}s, 含 alpha 关键词")
                else:
                    _warn("响应不含 alpha 关键词 (但 LLM 可用)")
                    report.add("LLM", "LLM 实际生成", "WARN", f"{elapsed:.1f}s, 无 alpha 关键词")

                return {"config_ok": True, "generation_ok": True, "elapsed": elapsed}
            else:
                _fail(f"空响应 ({elapsed:.1f}s)")
                report.add("LLM", "LLM 实际生成", "FAIL", "空响应")
                return {"config_ok": True, "generation_ok": False}

        except Exception as exc:
            error_str = str(exc).lower()
            if any(kw in error_str for kw in ["connection", "refused", "timeout"]):
                _fail(f"连接错误: {exc}")
                report.add("LLM", "LLM 实际生成", "FAIL", f"网络错误: {exc}")
                return {"config_ok": True, "generation_ok": False, "error": "NETWORK"}
            else:
                _fail(f"LLM 错误: {exc}")
                report.add("LLM", "LLM 实际生成", "FAIL", str(exc))
                return {"config_ok": True, "generation_ok": False, "error": str(exc)}

    except ImportError as exc:
        _fail(f"导入错误: {exc}")
        report.add("LLM", "LLM 配置加载", "FAIL", f"ImportError: {exc}")
        return {"config_ok": False, "error": "IMPORT_ERROR"}
    except Exception as exc:
        _fail(f"配置加载错误: {exc}")
        report.add("LLM", "LLM 配置加载", "FAIL", str(exc))
        return {"config_ok": False, "error": str(exc)}


async def check_brain_submission(report: HealthReport, cookies: Any) -> dict[str, Any]:
    """Check 4: BRAIN 提交测试（安全表达式）"""
    _header("Check 4: BRAIN 提交测试")

    test_expression = "rank(ts_delta(close, 5))"
    test_payload = {
        "settings": {
            "decay": 5,
            "neutralization": "INDUSTRY",
            "truncation": 0.08,
            "pasteurization": "ON",
            "unitHandling": "VERIFY",
            "nanHandling": "ON",
            "language": "FASTEXPR",
            "visualization": False,
        },
        "type": "REGULAR",
        "regular": test_expression,
    }

    try:
        from openalpha_brain.services.brain_client import submit_and_poll

        _info(f"提交测试表达式: {test_expression}")

        result = await submit_and_poll(
            simulation_payload=test_payload,
            cookies=cookies,
            max_poll_seconds=120,
        )

        if result.passed:
            _ok(f"提交通过 | Sharpe={result.sharpe:.3f} Fitness={result.fitness:.3f}")
            report.add(
                "Submission",
                "BRAIN 安全提交",
                "PASS",
                f"Sharpe={result.sharpe:.3f}",
            )
        else:
            _warn(f"完成但未通过门控 | Sharpe={result.sharpe}")
            report.add(
                "Submission",
                "BRAIN 安全提交",
                "WARN",
                f"Sharpe={result.sharpe:.3f}, 未通过门控",
            )

        _info(f"Alpha ID: {result.alpha_id or 'N/A'}")
        _info(f"Status: {result.simulation_status}")

        return {
            "success": True,
            "sharpe": result.sharpe,
            "fitness": result.fitness,
            "passed": result.passed,
            "alpha_id": result.alpha_id,
        }

    except Exception as exc:
        error_str = str(exc)
        if "429" in error_str or "rate" in error_str.lower():
            _fail("限流 (HTTP 429) — 请稍后重试")
            report.add("Submission", "BRAIN 安全提交", "FAIL", "RATE_LIMITED")
            return {"success": False, "error": "RATE_LIMITED"}
        elif "401" in error_str or "403" in error_str or "auth" in error_str.lower():
            _fail("提交时认证错误")
            report.add("Submission", "BRAIN 安全提交", "FAIL", "AUTH_ERROR")
            return {"success": False, "error": "AUTH_ERROR"}
        elif "timeout" in error_str.lower():
            _fail("提交超时 (网络问题?)")
            report.add("Submission", "BRAIN 安全提交", "FAIL", "TIMEOUT")
            return {"success": False, "error": "TIMEOUT"}
        else:
            _fail(f"提交失败: {exc}")
            report.add("Submission", "BRAIN 安全提交", "FAIL", str(exc))
            return {"success": False, "error": str(exc)}


async def check_layer_modules(report: HealthReport) -> dict[str, Any]:
    """Check 5: 6 个 Layer 模块可导入性"""
    _header("Check 5: Layer 模块导入")

    layers = [
        ("L1 ExplorationDirector", "openalpha_brain.core.layers.exploration_director", "ExplorationDirector"),
        ("L2 GenerationPipeline", "openalpha_brain.core.layers.generation_pipeline", "GenerationPipeline"),
        ("L3 EvaluationGateway", "openalpha_brain.core.layers.evaluation_gateway", "EvaluationGateway"),
        ("L4 ImprovementOrchestra", "openalpha_brain.core.layers.improvement_orchestra", "ImprovementOrchestra"),
        ("L5 RobustnessGate", "openalpha_brain.core.layers.robustness_gate", "RobustnessGate"),
        ("L6 PersistenceLayer", "openalpha_brain.core.layers.persistence_layer", "PersistenceLayer"),
    ]

    layer_status: dict[str, Any] = {}
    ok_count = 0

    for name, module_path, class_name in layers:
        try:
            mod = __import__(module_path, fromlist=[class_name])
            cls = getattr(mod, class_name)
            _ = cls()
            _ok(f"{name}: {class_name}() 实例化成功")
            layer_status[name] = {"importable": True}
            report.add("Layers", name, "PASS")
            ok_count += 1
        except Exception as exc:
            _fail(f"{name}: {exc}")
            layer_status[name] = {"importable": False, "error": str(exc)}
            report.add("Layers", name, "FAIL", str(exc)[:80])

    all_ok = ok_count == len(layers)
    _info(f"Layer 就绪度: {ok_count}/{len(layers)}")
    return {"all_importable": all_ok, "ok": ok_count, "total": len(layers), "layers": layer_status}


async def check_algorithm_modules(report: HealthReport) -> dict[str, Any]:
    """Check 6: 18 个算法模块可导入性"""
    _header("Check 6: 算法模块导入")

    algorithms = [
        ("FeedbackOrchestrator", "openalpha_brain.core.feedback_orchestrator", "FeedbackLoopOrchestrator"),
        ("DecisionEngine", "openalpha_brain.core.decision_engine", "DecisionEngine"),
        ("ResultRouter", "openalpha_brain.core.result_router", "ResultRouter"),
        ("ReflectionEngine", "openalpha_brain.learning.reflection_engine", "ReflectionEngine"),
        ("AdaptiveNeutralizer", "openalpha_brain.evolution.adaptive_neutralizer", "AdaptiveNeutralizer"),
        ("NearPassImprover", "openalpha_brain.evolution.near_pass_improver", "NearPassImprover"),
        ("FitnessBoostEngine", "openalpha_brain.evolution.fitness_boost", "FitnessBoostEngine"),
        ("TurnoverOptimizer", "openalpha_brain.optimization.turnover_optimizer", "TurnoverOptimizer"),
        ("MutationEngine", "openalpha_brain.evolution.mutation_engine", "BrainAwareMutationEngine"),
        ("EASearchStrategy", "openalpha_brain.evolution.ea_search", "EASearchStrategy"),
        ("ExperienceDistiller", "openalpha_brain.learning.experience_distiller", "ExperienceDistiller"),
        ("SemanticMutator", "openalpha_brain.evolution.semantic_mutator", "SemanticMutator"),
        ("CrossoverMutationEngine", "openalpha_brain.evolution.crossover_mutation", "CrossoverMutationEngine"),
        ("TrajectoryMutation", "openalpha_brain.evolution.trajectory_mutation", "TrajectoryMutation"),
        ("NavigationFusion", "openalpha_brain.core.navigation_fusion", "NavigationFusion"),
        ("ParamOptimizer", "openalpha_brain.learning.param_optimizer", "ParamOptimizer"),
        ("ToTSearchStrategy", "openalpha_brain.evolution.tot_search", "ToTSearchStrategy"),
        ("StrategyClassifier", "openalpha_brain.evolution.strategy_classifier", "StrategyClassifier"),
    ]

    algo_status: dict[str, Any] = {}
    ok_count = 0

    for name, module_path, class_name in algorithms:
        try:
            mod = __import__(module_path, fromlist=[class_name])
            _ = getattr(mod, class_name)
            _ok(f"{name}: 可导入")
            algo_status[name] = {"importable": True}
            report.add("Algorithms", name, "PASS")
            ok_count += 1
        except Exception as exc:
            _warn(f"{name}: {type(exc).__name__}: {str(exc)[:60]}")
            algo_status[name] = {"importable": False, "error": str(exc)}
            report.add("Algorithms", name, "WARN", f"{type(exc).__name__}: {str(exc)[:60]}")

    pct = 100 * ok_count // len(algorithms) if algorithms else 0
    _info(f"算法就绪度: {ok_count}/{len(algorithms)} ({pct}%)")
    return {"total": len(algorithms), "ok": ok_count, "algorithms": algo_status}


async def check_core_components(report: HealthReport) -> dict[str, Any]:
    """Check 7: 关键组件初始化验证（15+ 组件）"""
    _header("Check 7: 核心组件初始化")

    components = [
        ("MAB Bandit", "_check_mab", "learning.mab_bandit"),
        ("RAG Engine", "_check_rag", "knowledge.rag_engine"),
        ("Whitelist", "_check_whitelist", "knowledge.whitelist_manager"),
        ("SignalArbiter", "_check_signal_arbiter", "core.signal_arbiter"),
        ("OperatorRegistry", "_check_operator_registry", "knowledge.operator_registry"),
        ("FieldProxyMap", "_check_field_proxy_map", "knowledge.field_proxy_map"),
        ("ExpressionValidator", "_check_expr_validator", "validation.wq_expression_validator"),
        ("OfficialScorer", "_check_scorer", "validation.official_scorer"),
        ("FormatRepair", "_check_format_repair", "validation.wq_format_repair"),
        ("AntiOverfitDetector", "_check_anti_overfit", "validation.anti_overfit_detector"),
        ("AlphaLogicLibrary", "_check_alpha_logic_lib", "generation.alpha_logics"),
        ("SlotManager", "_check_slot_manager", "services.slot_manager"),
        ("AdaptiveNeutralizer", "_check_adaptive_neutralizer_init", "evolution.adaptive_neutralizer"),
        ("TurnoverOptimizer", "_check_turnover_opt_init", "optimization.turnover_optimizer"),
        ("ExperienceDistiller", "_check_exp_distill_init", "learning.experience_distiller"),
    ]

    comp_status: dict[str, Any] = {}
    ok_count = 0

    for display_name, check_fn_name, _mod_hint in components:
        checker = getattr(self_module_for_components(), check_fn_name, None)
        if checker is None:
            _warn(f"{display_name}: 检查函数未注册")
            comp_status[display_name] = {"status": "WARN", "detail": "检查函数未注册"}
            report.add("Components", display_name, "WARN", "检查函数未注册")
            continue

        try:
            result = await checker() if asyncio.iscoroutinefunction(checker) else checker()
            if result.get("ok"):
                _ok(f"{display_name}: {result.get('detail', 'OK')}")
                comp_status[display_name] = {"status": "PASS", "detail": result.get("detail", "")}
                report.add("Components", display_name, "PASS", result.get("detail", ""))
                ok_count += 1
            else:
                _warn(f"{display_name}: {result.get('detail', 'FAIL')}")
                comp_status[display_name] = {"status": "WARN", "detail": result.get("detail", "")}
                report.add("Components", display_name, "WARN", result.get("detail", ""))
        except Exception as exc:
            _fail(f"{display_name}: {exc}")
            comp_status[display_name] = {"status": "FAIL", "detail": str(exc)[:80]}
            report.add("Components", display_name, "FAIL", str(exc)[:80])

    _info(f"组件就绪度: {ok_count}/{len(components)}")
    return {"total": len(components), "ok": ok_count, "components": comp_status}


async def check_slot_semaphore(report: HealthReport, cookies: Any) -> dict[str, Any]:
    """Check 8: 并发 Slot 测试（3-slot Semaphore 验证）"""
    _header("Check 8: 并发 Slot 测试")

    try:
        from openalpha_brain.services.slot_manager import SlotManager

        manager = SlotManager(
            cookies=cookies,
            max_slots=3,
            poll_interval=3.0,
            max_poll_seconds=30,
        )

        await manager.start()
        _ok("SlotManager 启动成功 (max_slots=3)")
        report.add("SlotTest", "SlotManager 启动", "PASS", "3 slots")

        metrics = manager.get_metrics()
        _info(f"初始指标: slots={metrics.max_slots}")

        test_expr = "rank(ts_delta(close, 5))"
        tid = await manager.submit(expression=test_expr, name="health-test", strategy="test")
        _ok(f"测试因子已提交 (task_id={tid})")
        report.add("SlotTest", "因子提交", "PASS", f"task_id={tid}")

        await asyncio.sleep(2)

        slots = await manager.get_slot_status()
        busy = sum(1 for s in slots if s.state.value not in ("idle", "error"))
        _info(f"当前忙碌 slot: {busy}/3")
        report.add("SlotTest", "Slot 状态查询", "PASS", f"{busy}/3 busy")

        await manager.stop()
        _ok("SlotManager 已停止")
        report.add("SlotTest", "SlotManager 停止", "PASS")

        return {"success": True, "slots_tested": 3, "submitted": 1}

    except ImportError as exc:
        _fail(f"无法导入 SlotManager: {exc}")
        report.add("SlotTest", "SlotManager 测试", "FAIL", f"ImportError: {exc}")
        return {"success": False, "error": "IMPORT_ERROR"}
    except Exception as exc:
        _fail(f"Slot 测试失败: {exc}")
        report.add("SlotTest", "SlotManager 测试", "FAIL", str(exc))
        return {"success": False, "error": str(exc)}


# ════════════════════════════════════════════════════════════════════
#  组件检查辅助函数（被 check_core_components 调用）
# ════════════════════════════════════════════════════════════════════

def self_module_for_components():
    """返回当前模块自身，供组件检查函数通过 getattr 获取"""
    import sys as _sys
    return _sys.modules[__name__]


async def _check_mab() -> dict:
    from openalpha_brain.learning.mab import HierarchicalMAB
    _ = HierarchicalMAB()
    return {"ok": True, "detail": "HierarchicalMAB initialized"}


async def _check_rag() -> dict:
    from openalpha_brain.knowledge.rag_engine import RAGEngine
    _ = RAGEngine()
    return {"ok": True, "detail": "initialized"}


async def _check_whitelist() -> dict:
    from openalpha_brain.utils.whitelist import WhitelistManager
    _ = WhitelistManager()
    return {"ok": True, "detail": "initialized"}


async def _check_signal_arbiter() -> dict:
    from openalpha_brain.validation.signal_arbiter import SignalArbiter
    _ = SignalArbiter()
    return {"ok": True, "detail": "initialized"}


def _check_operator_registry() -> dict:
    from openalpha_brain.knowledge.operator_registry import get_operator_registry
    reg = get_operator_registry()
    stats = reg.get_stats()
    return {"ok": True, "detail": f"{stats['total_operators']} operators"}


def _check_field_proxy_map() -> dict:
    from openalpha_brain.knowledge.field_proxy_map import get_field_proxy_map
    _ = get_field_proxy_map()
    return {"ok": True, "detail": "initialized"}


def _check_expr_validator() -> dict:
    from openalpha_brain.validation.wq_expression_validator import WQExpressionValidator
    v = WQExpressionValidator()
    r = v.validate_full("group_neutralize(rank(ts_decay_linear(close, 10)), industry)")
    return {"ok": r.passed, "detail": f"validate={'PASS' if r.passed else 'FAIL'}"}


def _check_scorer() -> dict:
    from openalpha_brain.validation.official_scorer import OfficialScoringAdapter
    s = OfficialScoringAdapter()
    r = s.compute_score({"sharpe": 1.5, "fitness": 1.2, "turnover": 0.25})
    return {"ok": r.overall_score > 0, "detail": f"score={r.overall_score:.0f}"}


def _check_format_repair() -> dict:
    from openalpha_brain.validation.wq_format_repair import WQFormatRepair
    r = WQFormatRepair()
    d = r.diagnose("lookback required", "rank(ts_mean(close))")
    return {"ok": d.error_type == "lookback", "detail": f"type={d.error_type}"}


def _check_anti_overfit() -> dict:
    from openalpha_brain.validation.anti_overfit_detector import LightweightAntiOverfitDetector
    d = LightweightAntiOverfitDetector()
    r = d.evaluate({"sharpe": 1.5, "fitness": 1.2, "turnover": 0.25})
    return {"ok": r.score >= 0, "detail": f"score={r.score}"}


def _check_alpha_logic_lib() -> dict:
    from openalpha_brain.generation.alpha_logics import AlphaLogicLibrary
    lib = AlphaLogicLibrary()
    logics = lib.get_top_logics(n=3)
    return {"ok": len(logics) >= 1, "detail": f"{len(logics)} logics"}


async def _check_slot_manager() -> dict:
    from openalpha_brain.services.slot_manager import SlotManager
    m = SlotManager(cookies={}, max_slots=3, poll_interval=5.0, max_poll_seconds=30)
    return {"ok": True, "detail": f"slots={m.max_slots}"}


async def _check_adaptive_neutralizer_init() -> dict:
    from openalpha_brain.evolution.adaptive_neutralizer import AdaptiveNeutralizer
    _ = AdaptiveNeutralizer(experience_path=Path(".data/neutralization_experiences.json"))
    return {"ok": True, "detail": "initialized"}


async def _check_turnover_opt_init() -> dict:
    from openalpha_brain.optimization.turnover_optimizer import TurnoverOptimizer
    _ = TurnoverOptimizer()
    return {"ok": True, "detail": "initialized"}


async def _check_exp_distill_init() -> dict:
    from openalpha_brain.learning.experience_distiller import ExperienceDistiller
    _ = ExperienceDistiller()
    return {"ok": True, "detail": "initialized"}


# ════════════════════════════════════════════════════════════════════
#  报告输出
# ════════════════════════════════════════════════════════════════════

def print_summary(report: HealthReport, elapsed: float, mode: str) -> int:
    """打印最终汇总报告并返回 exit code"""
    _header("汇总报告")

    categories = [
        ("Configuration", "配置"),
        ("Authentication", "认证"),
        ("LLM", "LLM 连接"),
        ("Submission", "提交"),
        ("Layers", "Layer 模块"),
        ("Algorithms", "算法模块"),
        ("Components", "核心组件"),
        ("SlotTest", "Slot 并发"),
    ]

    print(f"\n  总耗时: {elapsed:.1f}s  |  模式: {mode}")
    print(f"  总检查数: {report.total_checks}  |  通过: {GREEN}{report.total_pass}{RESET}  |  失败: {RED}{report.total_fail}{RESET}  |  警告: {YELLOW}{report.total_warn}{RESET}")
    print()

    header_fmt = "  {:<22s} {:<10s} {:<6s} {:<6s} {:<6s} {:<s}"
    print(header_fmt.format("类别", "状态", "PASS", "FAIL", "WARN", "备注"))
    print("  " + "-" * 72)

    for cat_key, cat_label in categories:
        cat_results = [r for r in report.results if r.category == cat_key]
        if not cat_results:
            continue
        status = report.get_category_status(cat_key)
        counts = report._counts.get(cat_key, {})
        p = counts.get("PASS", 0)
        f = counts.get("FAIL", 0)
        w = counts.get("WARN", 0)

        if status == "PASS":
            icon = f"{GREEN}OK{RESET}"
        elif status == "WARN":
            icon = f"{YELLOW}WARN{RESET}"
        else:
            icon = f"{RED}FAIL{RESET}"

        fails_in_cat = [r for r in cat_results if r.status == "FAIL"]
        note = ""
        if fails_in_cat:
            note = fails_in_cat[0].detail[:40]

        print(f"  {cat_label:<22s} {icon:<14s} {p:<6d} {f:<6d} {w:<6d} {note}")

    print()

    has_config_fail = report.get_category_status("Configuration") == "FAIL"
    has_auth_fail = report.get_category_status("Authentication") == "FAIL"
    has_submit_fail = report.get_category_status("Submission") == "FAIL"

    if report.total_fail == 0 and report.total_warn == 0:
        print(f"{GREEN}{BOLD}✓ 系统状态健康 — 所有检查通过{RESET}")
        return 0
    elif has_config_fail:
        print(f"{RED}{BOLD}✗ 配置问题 — 请修复 .env 后重试{RESET}")
        return 1
    elif has_auth_fail:
        print(f"{RED}{BOLD}✗ 认证失败 — 请检查凭据{RESET}")
        return 2
    elif has_submit_fail:
        print(f"{RED}{BOLD}✗ 提交失败 — 可能是 API 问题或限流{RESET}")
        return 3
    elif report.total_fail > 0:
        print(f"{RED}{BOLD}✗ 存在 {report.total_fail} 项失败 — 请检查上方详情{RESET}")
        return 1
    else:
        print(f"{YELLOW}{BOLD}⚠ 存在 {report.total_warn} 项警告 — 系统基本可用但有注意事项{RESET}")
        return 0


# ════════════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════════════

async def async_main(args: argparse.Namespace) -> int:
    """异步主函数"""
    mode = "deep" if args.deep else ("dry-run" if args.dry_run else "quick")

    print(f"\n{'='*65}")
    print(f"{BOLD}{CYAN}  OpenAlpha-Brain 统一健康检查{RESET}")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}  |  模式: {mode}")
    print(f"{'='*65}")

    report = HealthReport()
    report.start_time = time.monotonic()

    # ── Phase 1: 环境配置（所有模式都执行）────────────────────
    env_results = await check_env_config(report)

    if args.dry_run:
        _header("DRY RUN 模式")
        _info("仅检查配置，不调用任何 API")
        elapsed = time.monotonic() - report.start_time
        return print_summary(report, elapsed, mode)

    # ── Phase 2: 模块导入（快速模式也执行）────────────────────
    await check_layer_modules(report)
    await check_algorithm_modules(report)

    if not args.deep:
        elapsed = time.monotonic() - report.start_time
        return print_summary(report, elapsed, mode)

    # ── Phase 3+: 深度模式额外检查 ────────────────────────────

    # 认证前置条件检查
    if not env_results["brain_email"] or not env_results["brain_password"]:
        _header("中止")
        _fail("缺少有效 BRAIN 凭据，无法执行深度检查")
        elapsed = time.monotonic() - report.start_time
        return print_summary(report, elapsed, mode)

    # BRAIN 认证
    auth_results = await check_brain_auth(report, env_results["email"], env_results["password"])

    # LLM 连接（并行执行）
    llm_task = asyncio.create_task(check_llm_connection(report, full_test=True))

    # 如果认证成功，继续提交和 Slot 测试
    if auth_results.get("success"):
        _ = await check_brain_submission(report, auth_results["cookies"])
        _ = await check_slot_semaphore(report, auth_results["cookies"])
    else:
        report.add("Submission", "BRAIN 安全提交", "SKIP", "认证未通过")
        report.add("SlotTest", "SlotManager 测试", "SKIP", "认证未通过")
        _ = None
        _ = None

    # 等待 LLM 检查完成
    await llm_task

    # 核心组件初始化
    await check_core_components(report)

    elapsed = time.monotonic() - report.start_time
    return print_summary(report, elapsed, mode)


def main() -> int:
    """入口函数"""
    parser = argparse.ArgumentParser(
        description="OpenAlpha-Brain 统一健康检查诊断脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python tools/unified_health.py health              # 快速模式（默认）
  python tools/unified_health.py health --deep        # 深度模式（含 API 调用）
  python tools/unified_health.py health --dry-run     # 仅配置检查""",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="health",
        help="子命令（目前支持: health）",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="深度模式：包含 API 调用（认证/LLM/提交/组件/Slot）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅检查 .env 配置，不调用任何 API",
    )

    args = parser.parse_args()

    if args.command != "health":
        print(f"未知子命令: {args.command}（支持: health）")
        return 1

    if args.deep and args.dry_run:
        print("--deep 和 --dry-run 不能同时使用")
        return 1

    return asyncio.run(async_main(args))


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
