#!/usr/bin/env python3
"""
OpenAlpha-Brain Unified Launcher
==================================
雙模式啟動入口：
  Web 模式  → FastAPI + Dashboard (瀏覽器操控)
  CLI 模式 → 終端 REPL (純命令行)

Usage:
    python start.py              # CLI 模式 (預設)
    python start.py --web        # Web 模式 (API + 瀏覽器)
    python start.py --run        # 直接開採 (非阻塞)
    python start.py --help       # 幫助
"""
from __future__ import annotations

import os
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _print_banner():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║           OpenAlpha-Brain  v3.0  Unified Launcher            ║")
    print("║     WorldQuant BRAIN Autonomous Alpha Mining System          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()


def launch_web(host: str = "0.0.0.0", port: int = 8000, open_browser: bool = True):
    """啟動 FastAPI 伺服器 + 自動打開瀏覽器"""
    import uvicorn
    _print_banner()
    url = f"http://127.0.0.1:{port}"
    print(f"  🌐  Web Mode  —  {url}")
    print("  ▶ Starting FastAPI server...")
    if open_browser:
        timer = threading.Timer(1.5, lambda: webbrowser.open(url))
        timer.daemon = True
        timer.start()
    os.chdir(ROOT)
    uvicorn.run("main:app", host=host, port=port, reload=False)


def launch_cli(args: list[str] | None = None):
    """啟動 CLI REPL 模式"""
    _print_banner()
    os.chdir(ROOT)
    cli_args = [str(ROOT / "alpha_cli.py")]
    if args:
        cli_args.extend(args)
    os.execv(sys.executable, [sys.executable] + cli_args)


def main():
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print(__doc__)
        return

    if "--web" in args or "-w" in args:
        port = 8000
        for i, a in enumerate(args):
            if a in ("--port", "-p") and i + 1 < len(args):
                port = int(args[i + 1])
        launch_web(port=port)
    elif "--run" in args or "-r" in args:
        rest = []
        i = 0
        while i < len(args):
            if args[i] in ("--run", "-r"):
                i += 1
                continue
            rest.append(args[i])
            i += 1
        launch_cli(["--run"] + rest)
    else:
        launch_cli()


if __name__ == "__main__":
    main()
