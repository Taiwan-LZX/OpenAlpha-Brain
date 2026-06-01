#!/usr/bin/env python3
"""
BRAIN Platform Connection Validator
==================================
Validates WQ BRAIN platform connectivity and configuration before E2E testing.

Usage:
    python tools/test_brain_connection.py          # Quick check (auth + submit test)
    python tools/test_brain_connection.py --full     # Full diagnostics (all checks)
    python tools/test_brain_connection.py --dry-run  # Config check only (no API calls)

Exit codes:
    0 = All checks passed (ready for E2E)
    1 = Configuration issues (.env missing/wrong)
    2 = Authentication failure (wrong credentials)
    3 = Submission failure (API error / rate limit)
    4 = Network/timeout issues

Environment:
    Requires .env file with:
      - BRAIN_EMAIL=your@email.com
      - BRAIN_PASSWORD=your_password
      - BRAIN_SUBMIT_ENABLED=true
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── Color output helpers ──────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
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


# ── Check functions ────────────────────────────────────────────────────


async def check_env_config() -> dict[str, Any]:
    """Check .env file existence and required variables."""
    _header("Check 1: Environment Configuration")
    
    env_file = _PROJECT_ROOT / ".env"
    results = {
        "env_exists": False,
        "brain_email": False,
        "brain_password": False,
        "brain_submit_enabled": False,
        "llm_api_key": False,
        "issues": [],
    }
    
    if not env_file.exists():
        _fail(".env file not found")
        _info(f"Expected at: {env_file}")
        _info("Copy .env.example to .env and fill in your credentials")
        results["issues"].append("MISSING_ENV_FILE")
        return results
    
    _ok(f".env file found: {env_file}")
    results["env_exists"] = True
    
    # Load .env manually (don't use pydantic to avoid import errors)
    env_vars = {}
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env_vars[key.strip()] = value.strip()
    
    # Check required variables
    if env_vars.get("BRAIN_EMAIL") and "@" in env_vars["BRAIN_EMAIL"]:
        email = env_vars["BRAIN_EMAIL"]
        masked = email[:3] + "***" + email.split("@")[1] if len(email) > 5 else "***@***"
        _ok(f"BRAIN_EMAIL: {masked}")
        results["brain_email"] = True
    else:
        _fail("BRAIN_EMAIL missing or invalid format")
        results["issues"].append("MISSING_BRAIN_EMAIL")
    
    if env_vars.get("BRAIN_PASSWORD") and len(env_vars["BRAIN_PASSWORD"]) > 3:
        pwd = env_vars["BRAIN_PASSWORD"]
        masked = pwd[:2] + "*" * (len(pwd) - 2) + pwd[-1:] if len(pwd) > 4 else "***"
        _ok(f"BRAIN_PASSWORD: {masked} ({len(pwd)} chars)")
        results["brain_password"] = True
    else:
        _fail("BRAIN_PASSWORD missing or too short")
        results["issues"].append("MISSING_BRAIN_PASSWORD")
    
    submit_enabled = env_vars.get("BRAIN_SUBMIT_ENABLED", "false").lower() in ("true", "1", "yes")
    if submit_enabled:
        _ok("BRAIN_SUBMIT_ENABLED=true")
        results["brain_submit_enabled"] = True
    else:
        _warn("BRAIN_SUBMIT_ENABLED=false (submission disabled)")
        results["issues"].append("SUBMISSION_DISABLED")
    
    if env_vars.get("LLM_API_KEY") and len(env_vars["LLM_API_KEY"]) > 10:
        key = env_vars["LLM_API_KEY"]
        masked = key[:6] + "..." + key[-4:]
        _ok(f"LLM_API_KEY: {masked}")
        results["llm_api_key"] = True
    else:
        _warn("LLM_API_KEY missing or too short (LLM features will be limited)")
        results["issues"].append("MISSING_LLM_KEY")
    
    return results


async def check_brain_auth(email: str, password: str) -> dict[str, Any]:
    """Test BRAIN platform authentication."""
    _header("Check 2: BRAIN Platform Authentication")
    
    try:
        from openalpha_brain.services.brain_client import authenticate
        
        cookies = await authenticate(email, password)
        
        # Handle different cookie return types (httpx.Cookies, dict, list, etc.)
        cookie_count = 0
        has_session = False
        
        if cookies is None:
            _warn("Authentication returned None (possible network issue)")
            return {"success": False, "error": "NONE_RESPONSE"}
        
        # Try to detect session cookie in various formats
        if hasattr(cookies, 'items'):  # dict-like
            cookie_count = len(cookies)
            has_session = any('session' in k.lower() for k in cookies.keys())
        elif hasattr(cookies, '__iter__') and not isinstance(cookies, str):  # iterable (list of Cookie objects)
            cookie_list = list(cookies)
            cookie_count = len(cookie_list)
            has_session = any(
                (hasattr(c, 'name') and 'session' in c.name.lower()) or
                (isinstance(c, tuple) and len(c) > 0 and 'session' in str(c[0]).lower())
                for c in cookie_list
            )
        elif hasattr(cookies, 'name'):  # Single Cookie object
            cookie_count = 1
            has_session = 'session' in cookies.name.lower() if hasattr(cookies, 'name') else False
        else:
            # Fallback: check if it's a non-empty response
            cookie_count = 1 if cookies else 0
            has_session = bool(cookies)
        
        if has_session or cookie_count > 0:
            _ok(f"Authentication successful ({cookie_count} cookies received)")
            if hasattr(cookies, 'items'):
                for k in list(cookies.keys())[:5]:
                    val = str(cookies[k])[:8] + "..." if len(str(cookies[k])) > 8 else str(cookies[k])
                    _info(f"  Cookie '{k}': {val}")
            return {"success": True, "cookies": cookies}
        else:
            _fail(f"Authentication returned no valid session (type={type(cookies).__name__})")
            return {"success": False, "error": "NO_SESSION_COOKIE"}
            
    except ImportError as exc:
        _fail(f"Cannot import brain_client: {exc}")
        return {"success": False, "error": "IMPORT_ERROR"}
    except Exception as exc:
        _fail(f"Authentication failed: {exc}")
        return {"success": False, "error": str(exc)}


async def check_brain_submission(cookies) -> dict[str, Any]:
    """Test a minimal BRAIN submission (safe test expression)."""
    _header("Check 3: BRAIN Submission Test")
    
    # Use a minimal, safe expression that won't cause issues
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
        from openalpha_brain.services.brain_client import submit_and_poll, get_client
        
        _info(f"Submitting test expression: {test_expression}")
        _info("(This is a safe, minimal expression for connectivity testing)")
        
        result = await submit_and_poll(
            simulation_payload=test_payload,
            cookies=cookies,
            max_poll_seconds=120,  # Shorter timeout for test
        )
        
        if result.passed:
            _ok(f"Submission PASSED | Sharpe={result.sharpe:.3f} Fitness={result.fitness:.3f}")
        else:
            _warn(f"Submission completed but gates not passed | Sharpe={result.sharpe}")
        
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
            _fail("Rate limited (HTTP 429) — wait a few minutes before retrying")
            return {"success": False, "error": "RATE_LIMITED"}
        elif "401" in error_str or "403" in error_str or "auth" in error_str.lower():
            _fail("Authentication error during submission")
            return {"success": False, "error": "AUTH_ERROR"}
        elif "timeout" in error_str.lower():
            _fail("Submission timed out (network issue?)")
            return {"success": False, "error": "TIMEOUT"}
        else:
            _fail(f"Submission failed: {exc}")
            return {"success": False, "error": str(exc)}


async def check_layer_modules() -> dict[str, Any]:
    """Verify all 6 Layer modules are importable."""
    _header("Check 4: Layer Module Importability")
    
    layer_status = {}
    layers = [
        ("L1 ExplorationDirector", "openalpha_brain.core.layers.exploration_director", "ExplorationDirector"),
        ("L2 GenerationPipeline", "openalpha_brain.core.layers.generation_pipeline", "GenerationPipeline"),
        ("L3 EvaluationGateway", "openalpha_brain.core.layers.evaluation_gateway", "EvaluationGateway"),
        ("L4 ImprovementOrchestra", "openalpha_brain.core.layers.improvement_orchestra", "ImprovementOrchestra"),
        ("L5 RobustnessGate", "openalpha_brain.core.layers.robustness_gate", "RobustnessGate"),
        ("L6 PersistenceLayer", "openalpha_brain.core.layers.persistence_layer", "PersistenceLayer"),
    ]
    
    for name, module_path, class_name in layers:
        try:
            mod = __import__(module_path, fromlist=[class_name])
            cls = getattr(mod, class_name)
            instance = cls()
            _ok(f"{name}: {class_name}() instantiated")
            layer_status[name] = {"importable": True, "instance": instance}
        except Exception as exc:
            _fail(f"{name}: {exc}")
            layer_status[name] = {"importable": False, "error": str(exc)}
    
    all_ok = all(s.get("importable") for s in layer_status.values())
    return {"all_importable": all_ok, "layers": layer_status}


async def check_algorithm_modules() -> dict[str, Any]:
    """Verify all 18 algorithm modules are importable."""
    _header("Check 5: Algorithm Module Importability")
    
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
    
    algo_status = {}
    ok_count = 0
    
    for name, module_path, class_name in algorithms:
        try:
            mod = __import__(module_path, fromlist=[class_name])
            cls = getattr(mod, class_name)
            _ok(f"{name}: importable")
            algo_status[name] = {"importable": True}
            ok_count += 1
        except Exception as exc:
            _warn(f"{name}: {type(exc).__name__}: {str(exc)[:60]}")
            algo_status[name] = {"importable": False, "error": str(exc)}
    
    _info(f"Algorithm readiness: {ok_count}/{len(algorithms)} ({100*ok_count//len(algorithms)}%)")
    return {"total": len(algorithms), "ok": ok_count, "algorithms": algo_status}


# ── Main execution ─────────────────────────────────────────────────────

async def main():
    """Run all connection checks."""
    args = sys.argv[1:]
    full_mode = "--full" in args
    dry_run = "--dry-run" in args
    
    print(f"\n{'='*60}")
    print(f"  OpenAlpha-Brain BRAIN Connection Validator")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    start_time = time.monotonic()
    
    # Check 1: Environment config
    env_results = await check_env_config()
    
    if dry_run:
        _header("DRY RUN MODE")
        _info("Skipping API calls (--dry-run flag set)")
        _print_summary(env_results, None, None, time.monotonic() - start_time)
        return 0 if not env_results["issues"] else 1
    
    # Stop if critical config missing
    if not env_results["brain_email"] or not env_results["brain_password"]:
        _header("ABORTED")
        _fail("Cannot proceed without valid BRAIN credentials")
        _print_summary(env_results, None, None, time.monotonic() - start_time)
        return 1
    
    # Check 2: Authentication
    email = os.environ.get("BRAIN_EMAIL", "")
    password = os.environ.get("BRAIN_PASSWORD", "")
    
    # Load from .env if not in environment
    if not email or not password:
        env_file = _PROJECT_ROOT / ".env"
        if env_file.exists():
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("BRAIN_EMAIL="):
                        email = line.split("=", 1)[1].strip()
                    elif line.startswith("BRAIN_PASSWORD="):
                        password = line.split("=", 1)[1].strip()
    
    auth_results = await check_brain_auth(email, password)
    
    if not auth_results.get("success"):
        _header("ABORTED")
        _fail("Authentication failed — cannot proceed with real data testing")
        _print_summary(env_results, auth_results, None, time.monotonic() - start_time)
        return 2
    
    # Check 3: Submission (only in full mode or if enabled)
    submit_results = None
    if full_mode or env_results.get("brain_submit_enabled"):
        submit_results = await check_brain_submission(auth_results["cookies"])
    
    # Check 4 & 5: Module imports (always run)
    layer_results = await check_layer_modules()
    algo_results = await check_algorithm_modules()
    
    elapsed = time.monotonic() - start_time
    _print_summary(env_results, auth_results, submit_results, elapsed, layer_results, algo_results)
    
    # Determine exit code
    if env_results["issues"]:
        return 1
    if not auth_results.get("success"):
        return 2
    if submit_results and not submit_results.get("success"):
        return 3
    return 0


def _print_summary(
    env_results: dict | None,
    auth_results: dict | None,
    submit_results: dict | None,
    elapsed: float,
    layer_results: dict | None = None,
    algo_results: dict | None = None,
) -> None:
    """Print final summary."""
    _header("Summary")
    
    print(f"\n  Total time: {elapsed:.1f}s")
    print()
    
    if env_results:
        config_ok = not env_results["issues"]
        status = GREEN + "READY" + RESET if config_ok else RED + "ISSUES" + RESET
        print(f"  Configuration:   {status}")
        if env_results["issues"]:
            for issue in env_results["issues"]:
                print(f"    - {issue}")
    
    if auth_results:
        status = GREEN + "OK" + RESET if auth_results.get("success") else RED + "FAIL" + RESET
        print(f"  Authentication:  {status}")
    
    if submit_results:
        status = GREEN + "OK" + RESET if submit_results.get("success") else RED + "FAIL" + RESET
        detail = ""
        if submit_results.get("sharpe") is not None:
            detail = f" (Sharpe={submit_results['sharpe']:.2f})"
        print(f"  Submission:      {status}{detail}")
    
    if layer_results:
        ready = sum(1 for l in layer_results["layers"].values() if l.get("importable"))
        total = len(layer_results["layers"])
        print(f"  Layers:          {ready}/{total} importable")
    
    if algo_results:
        print(f"  Algorithms:      {algo_results['ok']}/{algo_results['total']} importable ({100*algo_results['ok']//algo_results['total']}%)")
    
    print()
    
    if env_results and not env_results["issues"] and auth_results and auth_results.get("success"):
        print(f"{GREEN}{BOLD}✓ System is READY for E2E real-data testing{RESET}")
    else:
        print(f"{RED}{BOLD}✗ Fix the issues above before running E2E tests{RESET}")


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
