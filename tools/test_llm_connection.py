#!/usr/bin/env python3
"""
Real LLM Connection Test
=========================
Tests if LM Studio (or any LLM provider) is actually reachable and responding.
This is the CRITICAL test before running the full mining loop.

Usage:
    python tools/test_llm_connection.py              # Quick test
    python tools/test_llm_connection.py --full       # Full test with generation

Exit codes:
    0 = LLM working (real response received)
    1 = Config error
    2 = Connection failed (LM Studio not running?)
    3 = Response error (model loaded but failed)
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

def _ok(msg): print(f"  {GREEN}✓{RESET} {msg}")
def _fail(msg): print(f"  {RED}✗{RESET} {msg}")
def _warn(msg): print(f"  {YELLOW}⚠{RESET} {msg}")
def _info(msg): print(f"  {BLUE}ℹ{RESET} {msg}")

async def main():
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  Real LLM Connection Test{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}\n")
    
    full_mode = "--full" in sys.argv
    
    try:
        # Step 1: Load config
        from openalpha_brain.config.config import settings
        
        _info(f"Provider: {settings.LLM_PROVIDER}")
        _info(f"Model: {settings.LLM_MODEL}")
        _info(f"Base URL: {settings.LLM_BASE_URL or 'default for provider'}")
        
        if settings.LLM_PROVIDER == "lmstudio":
            from openalpha_brain.services.llm_client import LMSTUDIO_BASE_URL
            url = settings.LLM_BASE_URL or LMSTUDIO_BASE_URL
            _info(f"Effective URL: {url}")
        
        # Step 2: Import and test LLM client
        from openalpha_brain.services import llm_client
        
        if full_mode:
            # Step 3: Real generation test
            print(f"\n{BOLD}── Testing Real LLM Generation ──{RESET}")
            
            test_prompt = (
                "You are a quantitative finance expert. "
                "Generate a VERY SHORT alpha factor expression using WorldQuant syntax. "
                "Return ONLY the expression, nothing else. "
                "Use format: rank(some_operator(field, window))"
            )
            
            _info(f"Sending test prompt to {settings.LLM_PROVIDER}...")
            _info(f"Model: {settings.LLM_MODEL}")
            
            start = time.monotonic()
            try:
                response = await llm_client.generate(
                    system_prompt="You are a quantitative finance expert. Generate alpha factors for WorldQuant BRAIN platform.",
                    history=[],
                    user_msg=test_prompt,
                    session_id="llm-test",
                    cycle=0,
                )
                elapsed = time.monotonic() - start
                
                if response and len(response.strip()) > 0:
                    _ok(f"LLM responded in {elapsed:.1f}s")
                    _info(f"Response ({len(response)} chars):")
                    print(f"\n{CYAN}{response.strip()[:500]}{RESET}\n")
                    
                    # Check if it looks like an alpha expression
                    has_alpha_keywords = any(kw in response.lower() for kw in 
                        ["rank", "ts_", "delta", "corr", "mean", "stddev", "close", "volume"])
                    
                    if has_alpha_keywords:
                        _ok("Response contains alpha-like keywords ✅")
                    else:
                        _warn("Response doesn't look like alpha expression (but LLM works)")
                    
                    print(f"\n{BOLD}{GREEN}🎉 LLM IS WORKING! Real model calls will function.{RESET}")
                    return 0
                else:
                    _fail(f"Empty response after {elapsed:.1f}s")
                    return 3
                    
            except llm_client.LLMError as exc:
                _fail(f"LLMError: {exc}")
                return 3
            except Exception as exc:
                error_str = str(exc).lower()
                if "connection" in error_str or "refused" in error_str or "timeout" in error_str:
                    _fail(f"Connection error: {exc}")
                    _info("\nIs LM Studio running?")
                    _info("Start it: Open LM Studio → Select Model → Start Server")
                    return 2
                else:
                    _fail(f"Unexpected error: {exc}")
                    return 3
        else:
            # Quick mode: just check config
            _ok("Config loaded successfully")
            _info(f"Run with --full for actual LLM call test")
            return 0
            
    except ImportError as exc:
        _fail(f"Import error: {exc}")
        return 1
    except Exception as exc:
        _fail(f"Error: {exc}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
