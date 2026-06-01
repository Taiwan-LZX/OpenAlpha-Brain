#!/usr/bin/env python3
"""
OpenAlpha-Brain Unified Entry Point
===================================
Supports: python -m openalpha_brain [command] [options]

Commands:
  start      Start the autonomous alpha mining loop
  monitor    Launch web dashboard (FastAPI)
  status     Show system status and health check

Options:
  --focus-area AREA     Focus exploration area (default: momentum)
  --max-cycles N        Maximum cycles to run (default: from config)
  --web                 Start in web mode (alias for 'monitor')
"""
from __future__ import annotations

import asyncio
import sys
from typing import Optional

BANNER = """
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   ██████╗██╗   ██╗██████╗ ██╗██████╗  ██████╗ ███╗   ██╗███████╗  ║
║  ██╔════╝╚██╗ ██╔╝██╔══██╗██║██╔══██╗██╔═══██╗████╗  ██║██╔════╝  ║
║  ██║      ╚████╔╝ ██████╔╝██║██████╔╝██║   ██║██╔██╗ ██║█████╗    ║
║  ██║       ╚██╔╝  ██╔══██╗██║██╔══██╗██║   ██║██║╚██╗██║██╔══╝    ║
║  ╚██████╗   ██║   ██████╔╝██║██║  ██║╚██████╔╝██║ ╚████║███████╗  ║
║   ╚═════╝   ╚═╝   ╚═════╝ ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝  ║
║                                                                   ║
║              WorldQuant BRAIN Autonomous Alpha Mining              ║
║                       v3.0 — IQC 2026                              ║
╚═══════════════════════════════════════════════════════════════════╝
"""


def print_banner() -> None:
    print(BANNER)


def print_usage() -> None:
    print("""
Usage: python -m openalpha_brain <command> [options]

Commands:
  start       Start the autonomous alpha mining loop (CLI mode)
  monitor     Launch web dashboard with FastAPI server
  status      Run diagnostics and show system status

Options:
  --focus-area AREA     Set focus area for exploration (momentum/reversal/mean_reversion)
  --max-cycles N        Limit number of mining cycles
  --web                Start in web mode (same as 'monitor')
  --help, -h           Show this help message

Examples:
  python -m openalpha_brain start
  python -m openalpha_brain start --focus-area momentum --max-cycles 50
  python -m openalpha_brain monitor
  python -m openalpha_brain status
""")


async def cmd_start(focus_area: Optional[str] = None, max_cycles: Optional[int] = None) -> int:
    """Execute start command - launch the main loop."""
    from openalpha_brain.cli.launcher import BrainLauncher

    launcher = BrainLauncher()
    return await launcher.run(focus_area=focus_area, max_cycles=max_cycles)


async def cmd_monitor() -> int:
    """Execute monitor command - launch FastAPI dashboard."""
    import uvicorn
    from openalpha_brain.cli.main import app

    print_banner()
    host = "0.0.0.0"
    port = 8000
    print(f"  🌐  Starting Web Dashboard at http://127.0.0.1:{port}")
    print(f"  ▶  Press Ctrl+C to stop\n")

    uvicorn.run(app, host=host, port=port, reload=False)
    return 0


async def cmd_status() -> int:
    """Execute status command - run diagnostics."""
    from openalpha_brain.cli.launcher import BrainLauncher

    launcher = BrainLauncher()
    report = await launcher.startup_check()

    print("\n" + "=" * 60)
    print("  System Status Report")
    print("=" * 60)

    if report.passed:
        print(f"\n  ✅ All checks passed ({len(report.warnings)} warnings)")
    else:
        print(f"\n  ❌ {len(report.errors)} error(s) found")

    if report.warnings:
        print("\n  ⚠️  Warnings:")
        for w in report.warnings:
            print(f"     • {w}")

    if report.errors:
        print("\n  ❌ Errors:")
        for e in report.errors:
            print(f"     • {e}")

    print()

    return 0 if report.passed else 1


def parse_args(args: list[str]) -> tuple[Optional[str], dict]:
    """Parse command line arguments."""
    if not args or "--help" in args or "-h" in args:
        return None, {}

    command = args[0]
    options = {}

    i = 1
    while i < len(args):
        arg = args[i]
        if arg == "--focus-area" and i + 1 < len(args):
            options["focus_area"] = args[i + 1]
            i += 2
        elif arg == "--max-cycles" and i + 1 < len(args):
            options["max_cycles"] = int(args[i + 1])
            i += 2
        elif arg in ("--web", "-w"):
            options["web"] = True
            i += 1
        else:
            i += 1

    return command, options


async def main_async() -> int:
    """Main async entry point."""
    args = sys.argv[1:]

    command, options = parse_args(args)

    if command is None:
        print_banner()
        print_usage()
        return 0

    if command == "start":
        return await cmd_start(
            focus_area=options.get("focus_area"),
            max_cycles=options.get("max_cycles"),
        )
    elif command in ("monitor", "web") or options.get("web"):
        return await cmd_monitor()
    elif command == "status":
        return await cmd_status()
    else:
        print(f"Unknown command: {command}")
        print_usage()
        return 1


def main() -> None:
    """Synchronous entry point."""
    print_banner()

    exit_code = asyncio.run(main_async())
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
