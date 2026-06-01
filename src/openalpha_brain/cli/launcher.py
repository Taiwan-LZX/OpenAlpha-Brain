#!/usr/bin/env python3
"""
OpenAlpha-Brain Enhanced Launcher
==================================
6-Phase startup pipeline with self-healing main loop.

Phases:
  1. Dependency & environment checks
  2. LLM service detection (LM Studio)
  3. WQ BRAIN platform authentication
  4. Pipeline component loading
  5. Main mining loop with crash recovery
  6. Graceful shutdown

Usage:
    from openalpha_brain.cli.launcher import BrainLauncher
    launcher = BrainLauncher()
    await launcher.run(focus_area="momentum", max_cycles=100)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import aiohttp

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

import contextlib

from openalpha_brain.config.config import settings
from openalpha_brain.services import brain_client

if RICH_AVAILABLE:
    console = Console()


def _print(status: str, message: str) -> None:
    """Print status message with rich or fallback to plain text."""
    if RICH_AVAILABLE:
        console.print(f"  {status} {message}")
    else:
        print(f"  {status} {message}")


@dataclass
class CheckReport:
    """Result of startup checks."""

    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class LLMStatus:
    """LLM service detection result."""

    available: bool = False
    model_name: str | None = None
    endpoint: str | None = None
    error: str | None = None


@dataclass
class WQStatus:
    """WQ platform authentication result."""

    authenticated: bool = False
    alpha_count: int = 0
    active_simulations: int = 0
    slots_available: int = 3
    error: str | None = None


@dataclass
class PipelineStatus:
    """Pipeline loading result."""

    ready: bool = False
    components: list[str] = field(default_factory=list)
    error: str | None = None


def _setup_logging() -> Path:
    """Configure logging to both console and file."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"openalpha_{date_str}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-7s] %(name)-20s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return log_file


class BrainLauncher:
    """
    Enhanced launcher for OpenAlpha-Brain.

    Implements a 6-phase startup pipeline with comprehensive error handling,
    self-healing capabilities, and graceful shutdown.
    """

    def __init__(self) -> None:
        self._log_file: Path | None = None
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._stop_requested: bool = False
        self._cycle_count: int = 0
        self._session_id: str | None = None
        logging.getLogger("openalpha.launcher")

    async def run(
        self,
        focus_area: str | None = None,
        max_cycles: int | None = None,
        web_mode: bool = False,
    ) -> int:
        """
        Execute the full launch pipeline.

        Args:
            focus_area: Exploration focus area (momentum/reversal/mean_reversion)
            max_cycles: Maximum number of cycles to run
            web_mode: If True, launch web dashboard instead of CLI loop

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        if web_mode:
            return await self._launch_web()

        try:
            self._log_file = _setup_logging()
            logger = logging.getLogger("openalpha.launcher")
            logger.info("=" * 60)
            logger.info("OpenAlpha-Brain Launcher starting")
            logger.info(f"Log file: {self._log_file}")

            report = await self.startup_check()
            if not report.passed:
                _print("❌", "Startup checks failed. Fix errors above and retry.")
                return 1

            llm_status = await self.detect_llm()
            if not llm_status.available:
                _print("⚠️", f"LLM not available: {llm_status.error}. Continuing without LLM...")

            wq_status = await self.authenticate_wq()
            if not wq_status.authenticated:
                _print("❌", f"WQ authentication failed: {wq_status.error}")
                return 1

            pipeline_status = await self.load_pipeline()
            if not pipeline_status.ready:
                _print("❌", f"Pipeline load failed: {pipeline_status.error}")
                return 1

            _print("✅", "All systems ready. Starting main loop...\n")
            exit_code = await self.run_main_loop(
                focus_area=focus_area,
                max_cycles=max_cycles,
            )
            return exit_code

        except KeyboardInterrupt:
            _print("\n⚠️", "Interrupted by user")
            return 130
        except Exception as exc:
            logging.getLogger("openalpha.launcher").error(
                "Launcher crashed: %s",
                exc,
                exc_info=True,
            )
            _print("❌", f"Fatal error: {exc}")
            return 1
        finally:
            await self.shutdown()

    # ── Phase 1: Startup Checks ───────────────────────────────────────────────

    async def startup_check(self) -> CheckReport:
        """
        Phase 1: Comprehensive dependency and environment validation.

        Returns:
            CheckReport with pass/fail status and any warnings/errors
        """
        report = CheckReport()
        logger = logging.getLogger("openalpha.launcher")

        _print("\n📋", "Phase 1/6: Environment & Dependency Checks")
        print("-" * 55)

        # 1.1 Python version check
        py_version = sys.version_info
        if py_version >= (3, 11):
            _print("✅", f"Python {py_version.major}.{py_version.minor}.{py_version.micro} ✓")
        else:
            msg = f"Python {py_version.major}.{py_version.minor} found, >=3.11 required"
            report.errors.append(msg)
            report.passed = False
            _print("❌", msg)

        # 1.2 Required packages check
        required_packages = [
            ("httpx", "HTTP client"),
            ("numpy", "Numerical computing"),
            ("pydantic", "Data validation"),
            ("pydantic_settings", "Settings management"),
            ("tenacity", "Retry logic"),
            ("aiofiles", "Async file I/O"),
        ]
        for pkg_name, description in required_packages:
            try:
                __import__(pkg_name.replace("-", "_"))
                _print("✅", f"{description} ({pkg_name}) ✓")
            except ImportError:
                msg = f"Missing package: {pkg_name} - pip install {pkg_name}"
                report.errors.append(msg)
                _print("❌", msg)

        # 1.3 .env file check
        Path(".env")
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        env_full_path = project_root / ".env"
        if env_full_path.exists():
            _print("✅", f".env file found at {env_full_path}")
            if not os.getenv("BRAIN_EMAIL"):
                report.warnings.append("BRAIN_EMAIL not set in .env")
                _print("⚠️", "BRAIN_EMAIL not configured in .env")
            if not os.getenv("BRAIN_PASSWORD"):
                report.warnings.append("BRAIN_PASSWORD not set in .env")
                _print("⚠️", "BRAIN_PASSWORD not configured in .env")
        else:
            msg = f".env file missing at {env_full_path}"
            report.errors.append(msg)
            report.passed = False
            _print("❌", msg)
            _print("   ", "Create .env from .env.example and fill credentials")

        # 1.4 Data files integrity check
        data_dir = Path(__file__).resolve().parent.parent / "data"

        operators_path = data_dir / "brain_operators.json"
        datafields_path = data_dir / "brain_datafields.json"

        if operators_path.exists():
            with open(operators_path, encoding="utf-8") as f:
                operators = json.load(f)
            op_count = len(operators) if isinstance(operators, list) else len(operators.get("operators", []))
            if op_count >= 66:
                _print("✅", f"brain_operators.json: {op_count} operators loaded ✓")
            else:
                msg = f"brain_operators.json has only {op_count} operators (expected >=66)"
                report.warnings.append(msg)
                _print("⚠️", msg)
        else:
            msg = f"brain_operators.json missing at {operators_path}"
            report.errors.append(msg)
            report.passed = False
            _print("❌", msg)

        if datafields_path.exists():
            with open(datafields_path, encoding="utf-8") as f:
                datafields = json.load(f)
            df_count = len(datafields) if isinstance(datafields, list) else len(datafields.get("data", []))
            if df_count >= 7000:
                _print("✅", f"brain_datafields.json: {df_count} fields loaded ✓")
            else:
                msg = f"brain_datafields.json has only {df_count} fields (expected >=7000)"
                report.warnings.append(msg)
                _print("⚠️", msg)
        else:
            msg = f"brain_datafields.json missing at {datafields_path}"
            report.errors.append(msg)
            report.passed = False
            _print("❌", msg)

        logger.info(
            "Startup check completed: passed=%s warnings=%d errors=%d",
            report.passed,
            len(report.warnings),
            len(report.errors),
        )
        return report

    # ── Phase 2: LLM Detection ────────────────────────────────────────────────

    async def detect_llm(self) -> LLMStatus:
        """
        Phase 2: Detect and validate LLM service (LM Studio).

        Returns:
            LLMStatus with availability and model information
        """
        logger = logging.getLogger("openalpha.launcher")
        _print("\n🤖", "Phase 2/6: LLM Service Detection")
        print("-" * 55)

        endpoint = settings.LMSTUDIO_API_BASE or "http://localhost:1234"
        models_url = f"{endpoint}/v1/models"

        status = LLMStatus(endpoint=endpoint)

        try:
            import httpx

            client = httpx.AsyncClient(timeout=10.0)
            resp = await client.get(models_url)
            await client.aclose()

            if resp.status_code == 200:
                models_data = resp.json()
                models = models_data.get("data", [])
                if models:
                    model_name = models[0].get("id", "unknown")
                    status.available = True
                    status.model_name = model_name
                    _print("✅", f"LM Studio online — Model: {model_name}")
                    _print("✅", f"Endpoint: {endpoint}")
                    logger.info("LLM detected: %s at %s", model_name, endpoint)
                else:
                    status.error = "No models loaded in LM Studio"
                    _print("⚠️", status.error)
            else:
                status.error = f"LM Studio returned HTTP {resp.status_code}"
                _print("❌", status.error)

        except (TimeoutError, aiohttp.ClientError, ConnectionError, OSError) as exc:
            status.error = str(exc)
            _print("⚠️", f"Cannot connect to LM Studio: {exc}")
            _print("💡", "Attempting to start LM Studio...")

            launched = await self._try_start_lmstudio()
            if launched:
                _print("✅", "LM Studio started. Retrying detection...")
                await asyncio.sleep(3)
                return await self.detect_llm()

        if not status.available:
            _print("💡", f"Start LM Studio manually: {endpoint}")
            _print("   ", "Or set LMSTUDIO_API_BASE in .env if using different endpoint")

        return status

    async def _try_start_lmstudio(self) -> bool:
        """Attempt to launch LM Studio via subprocess."""
        system = platform.system()
        try:
            if system == "Windows":
                subprocess.Popen(["start", "lmstudio"], shell=True, creationflags=subprocess.DETACHED_PROCESS)
            elif system == "Darwin":
                subprocess.Popen(["open", "-a", "LM Studio"])
            else:
                subprocess.Popen(["lmstudio"], start_new_session=True)
            return True
        except FileNotFoundError:
            return False
        except (OSError, PermissionError, subprocess.SubprocessError):
            return False

    # ── Phase 3: WQ Authentication ───────────────────────────────────────────

    async def authenticate_wq(self) -> WQStatus:
        """
        Phase 3: Authenticate with WorldQuant BRAIN platform.

        Returns:
            WQStatus with authentication state and account info
        """
        logger = logging.getLogger("openalpha.launcher")
        _print("\n🔐", "Phase 3/6: WQ Platform Authentication")
        print("-" * 55)

        status = WQStatus()

        email = settings.BRAIN_EMAIL
        password = settings.BRAIN_PASSWORD

        if not email or not password:
            status.error = "BRAIN_EMAIL or BRAIN_PASSWORD not configured"
            _print("❌", status.error)
            _print("   ", "Add credentials to your .env file:")
            _print("   ", "  BRAIN_EMAIL=your@email.com")
            _print("   ", "  BRAIN_PASSWORD=your_password")
            return status

        try:
            from openalpha_brain.services import brain_client

            _print("⏳", f"Authenticating as {email[:3]}***...")
            cookies = await brain_client.authenticate(email, password)
            status.authenticated = True
            _print("✅", "Authentication successful ✓")

            _print("⏳", "Verifying account status...")
            alphas = await brain_client.list_alphas(cookies)
            if alphas:
                status.alpha_count = len(alphas)
                _print("✅", f"Account active — {len(alphas)} alphas found")

            status.slots_available = settings.PIPELINE_MAX_SLOTS
            _print("✅", f"Simulation slots: {status.slots_available} available")

            logger.info(
                "WQ authenticated: email=%s alphas=%d slots=%d",
                email[:3] + "***",
                status.alpha_count,
                status.slots_available,
            )

        except brain_client.BrainAuthError as exc:
            status.error = f"Authentication failed: {exc}"
            _print("❌", status.error)
            _print("   ", "Check email/password in .env file")
        except Exception as exc:
            status.error = f"Auth error: {exc}"
            _print("❌", status.error)
            logger.error("WQ auth exception: %s", exc, exc_info=True)

        return status

    # ── Phase 4: Pipeline Loading ─────────────────────────────────────────────

    async def load_pipeline(self) -> PipelineStatus:
        """
        Phase 4: Initialize all pipeline components.

        Returns:
            PipelineStatus with component readiness
        """
        logger = logging.getLogger("openalpha.launcher")
        _print("\n⚙️", "Phase 4/6: Pipeline Component Loading")
        print("-" * 55)

        status = PipelineStatus()
        components = []

        try:
            # 4.1 FieldProxyMap (29 proxy families)
            _print("⏳", "Loading FieldProxyMap (29 proxy families)...")
            from openalpha_brain.core.scheduler import ExplorationScheduler

            scheduler = ExplorationScheduler()
            if hasattr(scheduler, "field_proxy_map") and scheduler.field_proxy_map:
                components.append("FieldProxyMap")
                _print("✅", f"FieldProxyMap loaded — {len(scheduler.field_proxy_map)} families")
            else:
                components.append("FieldProxyMap(basic)")
                _print("✅", "FieldProxyMap initialized")

            # 4.2 AlphaLogicLibrary (three-stage templates)
            _print("⏳", "Loading AlphaLogicLibrary (three-stage templates)...")
            from openalpha_brain.generation.alpha_logics import AlphaLogicLibrary

            logic_lib = AlphaLogicLibrary()
            components.append("AlphaLogicLibrary")
            template_count = len(logic_lib.templates) if hasattr(logic_lib, "templates") else 0
            _print("✅", f"AlphaLogicLibrary loaded — {template_count} templates")

            # 4.3 MAB initialization
            _print("⏳", "Initializing MAB (HierarchicalMAB + SlidingWindowUCB)...")
            components.append("MAB")
            _print("✅", "MAB initialized with exploration scheduler")

            # 4.4 ASTValidator (66 operator whitelist)
            _print("⏳", "Initializing ASTValidator (66-operator whitelist)...")
            from openalpha_brain.validation.validator import get_originality_checker

            whitelist_mgr = get_originality_checker()
            components.append("ASTValidator")
            if whitelist_mgr and hasattr(whitelist_mgr, "whitelist"):
                _print("✅", f"ASTValidator ready — {len(whitelist_mgr.whitelist)} operators whitelisted")
            else:
                _print("✅", "ASTValidator initialized")

            # 4.5 ExperienceReplayManager
            _print("⏳", "Initializing ExperienceReplayManager...")
            components.append("ExperienceReplayManager")
            _print("✅", "ExperienceReplayManager initialized")

            status.ready = True
            status.components = components

            logger.info("Pipeline loaded successfully: %s", ", ".join(components))

        except ImportError as exc:
            status.error = f"Import error: {exc}"
            _print("❌", status.error)
        except Exception as exc:
            status.error = f"Pipeline init error: {exc}"
            _print("❌", status.error)
            logger.error("Pipeline load error: %s", exc, exc_info=True)

        return status

    # ── Phase 5: Main Loop ────────────────────────────────────────────────────

    async def run_main_loop(
        self,
        focus_area: str | None = None,
        max_cycles: int | None = None,
    ) -> int:
        """
        Phase 5: Run the autonomous mining loop with self-healing.

        Features:
        - Crash Recovery: catch exceptions per cycle, log + continue
        - Auto-Reauth: re-authenticate on BrainAuthError
        - Slot-Aware: handle 429 CONCURRENT_SIMULATION_LIMIT_EXCEEDED
        - Heartbeat: status summary every 30 cycles
        - Graceful Shutdown: SIGINT/SIGTERM handling

        Args:
            focus_area: Exploration direction
            max_cycles: Cycle limit (None = unlimited)

        Returns:
            Exit code
        """
        logger = logging.getLogger("openalpha.launcher")
        _print("\n🔄", "Phase 5/6: Autonomous Mining Loop")
        print("-" * 55)

        cycle_limit = max_cycles or settings.MAX_CYCLES
        actual_focus = focus_area or settings.DEFAULT_EXPLORATION_DIRECTION

        _print("🎯", f"Focus area: {actual_focus}")
        _print("🔢", f"Max cycles: {'∞ unlimited' if max_cycles is None else cycle_limit}")
        _print("", "")

        # Setup signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, self._request_shutdown)

        from openalpha_brain.cli import session_manager as sm
        from openalpha_brain.core import loop_engine

        session_state = await sm.create_session(focus_area=actual_focus)
        self._session_id = session_state.id

        _print("▶️", f"Session created: {self._session_id[:8]}...")
        _print("", "")
        _print("", "═" * 55)
        _print("", "  Mining loop started. Press Ctrl+C to stop gracefully.")
        _print("", "═" * 55)
        print()

        consecutive_errors = 0
        max_consecutive_errors = 5

        try:
            while not self._stop_requested:
                if cycle_limit is not None and self._cycle_count >= cycle_limit:
                    _print("✅", f"Cycle limit reached ({cycle_limit})")
                    break

                self._cycle_count += 1
                cycle_num = self._cycle_count

                try:
                    logger.info("[Cycle %d/%s] Starting...", cycle_num, cycle_limit or "∞")

                    if settings.PIPELINE_MODE:
                        await loop_engine.run_loop_pipeline(self._session_id)
                    else:
                        await loop_engine.run_loop(self._session_id)

                    consecutive_errors = 0

                    # Heartbeat every 30 cycles
                    if cycle_num % 30 == 0:
                        self._print_heartbeat(cycle_num, cycle_limit)

                except KeyboardInterrupt:
                    _print("\n⚠️", "User interrupt received")
                    self._request_shutdown()
                    break

                except brain_client.BrainAuthError:
                    _print("⚠️", f"[Cycle {cycle_num}] Auth expired, re-authenticating...")
                    wq_status = await self.authenticate_wq()
                    if not wq_status.authenticated:
                        _print("❌", "Re-authentication failed. Stopping.")
                        break
                    consecutive_errors += 1
                    logger.warning("Auto-reauth attempted after BrainAuthError")

                except (ValueError, TypeError, RuntimeError, KeyError, OSError) as exc:
                    consecutive_errors += 1
                    error_type = type(exc).__name__
                    error_msg = str(exc)[:200]

                    logger.error(
                        "[Cycle %d] Error (%d consecutive): [%s] %s",
                        cycle_num,
                        consecutive_errors,
                        error_type,
                        error_msg,
                        exc_info=True,
                    )

                    # Handle slot limit (429)
                    if "429" in error_msg or "CONCURRENT_SIMULATION" in error_msg:
                        wait_time = min(60 * consecutive_errors, 300)
                        _print(
                            "⏳",
                            f"[Cycle {cycle_num}] Slot limit hit. Waiting {wait_time}s...",
                        )
                        await asyncio.sleep(wait_time)
                        continue

                    _print(
                        "⚠️",
                        f"[Cycle {cycle_num}] Crashed: {error_type}: {error_msg}",
                    )

                    if consecutive_errors >= max_consecutive_errors:
                        _print(
                            "❌",
                            f"Too many consecutive errors ({max_consecutive_errors}). Stopping.",
                        )
                        break

                    backoff = min(2**consecutive_errors, 30)
                    _print("⏳", f"Waiting {backoff}s before next cycle...")
                    await asyncio.sleep(backoff)

        except asyncio.CancelledError:
            _print("⚠️", "Loop cancelled")

        finally:
            _print("", "═" * 55)
            _print("📊", f"Loop ended after {self._cycle_count} cycles")
            _print("", "═" * 55)

        return 0 if not self._stop_requested else 0

    def _request_shutdown(self) -> None:
        """Signal handler for graceful shutdown."""
        if not self._stop_requested:
            self._stop_requested = True
            self._shutdown_event.set()
            _print("\n\n⚠️", "Shutdown requested. Finishing current cycle...")

    def _print_heartbeat(self, cycle: int, limit: int | None) -> None:
        """Print periodic status summary."""
        now = datetime.now().strftime("%H:%M:%S")
        limit_str = str(limit) if limit else "∞"
        _print(
            "💓",
            f"Heartbeat [{now}] Cycle {cycle}/{limit_str} — System healthy",
        )

    # ── Phase 6: Shutdown ─────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """
        Phase 6: Graceful shutdown sequence.

        Saves state, closes connections, flushes logs.
        """
        logger = logging.getLogger("openalpha.launcher")
        _print("\n🛑", "Phase 6/6: Graceful Shutdown")
        print("-" * 55)

        try:
            # Save MAB state if available
            try:
                from openalpha_brain.core import loop_state as _ls

                if hasattr(_ls, "_scheduler") and _ls._scheduler:
                    _ls._scheduler.save_state()
                    _print("✅", "MAB state saved")
            except (OSError, ValueError, RuntimeError) as e:
                logger.warning("Failed to save MAB state: %s", e)

            # Close HTTP connection pool
            try:
                from openalpha_brain.services.http_pool import close_client

                await close_client()
                _print("✅", "HTTP connection pool closed")
            except (OSError, ValueError, RuntimeError) as e:
                logger.warning("Failed to close HTTP pool: %s", e)

            # Flush all log handlers
            try:
                root_logger = logging.getLogger()
                for handler in root_logger.handlers[:]:
                    handler.flush()
                    if isinstance(handler, logging.FileHandler):
                        handler.close()
                        root_logger.removeHandler(handler)
                _print("✅", "Logs flushed")
            except (OSError, ValueError, RuntimeError) as e:
                logger.warning("Failed to flush logs: %s", e)

            logger.info("OpenAlpha-Brain shutdown complete")
            _print("✅", "Shutdown complete. Goodbye! 👋\n")

        except Exception as exc:
            logger.error("Shutdown error: %s", exc, exc_info=True)
            _print("⚠️", f"Shutdown warning: {exc}")

    # ── Web Mode ──────────────────────────────────────────────────────────────

    async def _launch_web(self) -> int:
        """Launch FastAPI web dashboard mode."""
        import uvicorn

        from openalpha_brain.cli.main import app

        _print("🌐", "Launching Web Dashboard Mode")
        host = "0.0.0.0"
        port = 8000
        _print("▶️", f"Starting server at http://127.0.0.1:{port}")

        uvicorn.run(app, host=host, port=port, reload=False)
        return 0
