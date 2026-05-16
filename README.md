# OpenAlpha-Quant

> 🧠 **Autonomous alpha research platform for WorldQuant BRAIN IQC 2026**  
> LLM-powered closed-loop pipeline — Generate → Validate → Submit → Mutate → Repeat until BRAIN approves

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![WorldQuant BRAIN](https://img.shields.io/badge/WorldQuant-BRAIN-orange)](https://worldquantbrain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What It Does

OpenAlpha-Quant runs as a fully autonomous agent that:

1. **Generates** alpha expressions using an LLM (Groq / Gemini / OpenAI) guided by a rigorous IQC research mandate
2. **Validates locally** — syntax, fingerprint anti-crowding, estimated metrics via exact IQC formulas
3. **Submits to WorldQuant BRAIN** — real simulation via the BRAIN REST API
4. **Reads real BRAIN results** — actual Sharpe, Fitness, Turnover + every gate check (`LOW_SHARPE`, `LOW_FITNESS`, `LOW_SUB_UNIVERSE_SHARPE`, `SELF_CORRELATION`, etc.)
5. **Mutates surgically** — each failing BRAIN check triggers a specific ELM mutation injected back into the LLM
6. **Loops** until all BRAIN checks pass — the alpha appears in your "My Alphas" account

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     OpenAlpha - Quant                           │
│                                                                 │
│   LLM (Groq/Gemini/OpenAI)                                      │
│        │                                                        │
│        ▼                                                        │
│   [GENERATION]  Alpha expression + metrics + fingerprint        │
│        │                                                        │
│        ▼                                                        │
│   [LOCAL VALIDATION]                                            │
│   • Syntax (operators, parens, variable whitelist)              │
│   • Fingerprint anti-crowding (5-dim POMDP memory)              │
│   • Estimated IQC gate check (Sharpe/Fitness/Turnover)          │
│        │ PASS                                                   │
│        ▼                                                        │
│   [BRAIN SUBMISSION]  POST /simulations → poll → COMPLETE       │
│        │                                                        │
│        ▼                                                        │
│   [REAL GATE CHECKS]  checks[] from BRAIN API                   │
│   • LOW_SHARPE → ts_zscore + volatility norm + subindustry      │
│   • LOW_FITNESS → ts_decay_linear(expr, 10)                     │
│   • LOW_SUB_UNIVERSE_SHARPE → subindustry + interaction factors │
│   • SELF_CORRELATION → full structural pivot (3+ dims)          │
│   • HIGH_TURNOVER → increase decay d=6→10→15→20                 │
│   • SIMULATION ERROR → enforce variable whitelist               │
│        │ FAIL → inject targeted mutation → back to LLM          │
│        │ PASS ──────────────────────────────────────────────►   │
│                    Alpha saved in BRAIN "My Alphas" ✓           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/hitendras510/OpenAlpha-Brain.git
cd OpenAlpha-Brain
pip install -r requirements.txt
```

### 2. Configure `.env`

```bash
cp .env.example .env
```

Edit `.env`:

| Variable | Required | Description |
|---|---|---|
| `LLM_PROVIDER` | ✅ | `groq` / `gemini` / `openai` / `anthropic` |
| `LLM_MODEL` | ✅ | e.g. `llama-3.3-70b-versatile` (Groq), `gemini-1.5-flash` |
| `LLM_API_KEY` | ✅ | Your LLM provider API key |
| `BRAIN_EMAIL` | ✅ | Your worldquantbrain.com account email |
| `BRAIN_PASSWORD` | ✅ | Your worldquantbrain.com account password |
| `BRAIN_SUBMIT_ENABLED` | ✅ | `true` to enable auto-submission to BRAIN |
| `BRAIN_POLL_TIMEOUT` | ❌ | Seconds to wait for simulation (default: 300) |
| `MAX_CYCLES` | ❌ | Max research cycles per session (default: 50) |

> **Free LLM Keys**  
> - Groq (30 RPM free): [console.groq.com](https://console.groq.com)  
> - Gemini (generous free tier, recommended): [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

### 3. Run

```bash
uvicorn main:app --port 8000
```

### 4. Start a Research Session

```bash
# Start a session
SESSION=$(curl -s -X POST http://localhost:8000/session/start \
  -H "Content-Type: application/json" \
  -d '{"focus_area":"price volume momentum"}' | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

# Watch until BRAIN submission passes
python3 watch_session.py $SESSION
```

Or open the dashboard: **http://localhost:8000/static/index.html**

---

## Closed-Loop Mutation System

When BRAIN returns a simulation result, every failing check is mapped to a **precise surgical mutation** that gets injected into the LLM's next generation cycle:

| BRAIN Check | What It Means | Mutation Applied |
|---|---|---|
| `LOW_SHARPE` | Sharpe < 1.25 | Switch `rank()` → `ts_zscore()`, add volatility norm `/ts_std_dev(close,20)`, tighten to `subindustry` |
| `LOW_FITNESS` | Fitness ≤ 1.0 | Wrap with `ts_decay_linear(expr, 10)` targeting 15–35% turnover |
| `LOW_SUB_UNIVERSE_SHARPE` | Fails within industry groups | Change neutralization to `subindustry`, add interaction `rank(A)*rank(B)` |
| `HIGH_TURNOVER` | Turnover > 70% | Increase decay `d=6→10→15→20` |
| `LOW_TURNOVER` | Turnover < 1% | Remove decay, shorter windows, use `ts_delta()` |
| `SELF_CORRELATION` | Correlated with existing alphas | Full 3-dim structural pivot to different factor family |
| `CONCENTRATED_WEIGHT` | Weights too concentrated | Add `scale()` + `signed_power(expr, 0.5)` |
| `SIMULATION ERROR` | Invalid variable name | Enforce variable whitelist, rewrite expression |

The LLM also receives the **exact real metric values** from BRAIN (Sharpe, Fitness, Turnover, Returns) alongside each mutation instruction.

---

## Confirmed BRAIN FastExpr Variables

Only these bare variable names work in BRAIN's FastExpr language:

```
Price/Volume : close, open, high, low, vwap, volume, adv20, returns, cap
Short Int    : short_ratio, days_to_cover
Analyst      : analyst_rating, price_target
Neutralize   : industry, subindustry, sector  (group_neutralize args only)
```

> ⚠️ Variables like `earnings`, `sales`, `short_interest`, `institutional_ownership` do **not** exist as bare names — BRAIN uses dataset-prefixed names accessed via Data Explorer.

---

## IQC Gate Thresholds

```
Sharpe   ≥ 1.25   (target > 2.0)
Fitness  > 1.0    Formula: Sharpe × sqrt(|Returns|) / max(Turnover, 0.125)
Turnover 1%–70%   Sweet spot: 15%–35%
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Server health check |
| `/session/start` | POST | Start a new research session |
| `/session/{id}` | GET | Get session state (status, passed alphas, BRAIN results) |
| `/session/{id}/stop` | POST | Stop a running session |
| `/static/index.html` | GET | Real-time research dashboard |

### Session Status Values

| Status | Meaning |
|---|---|
| `GENERATING` | LLM is writing an alpha |
| `VALIDATING` | Local gate checks running |
| `SUBMITTING` | Alpha sent to BRAIN, polling for result |
| `ITERATING` | BRAIN returned failures, LLM mutating |
| `PASS` | Alpha passed all BRAIN checks ✓ |
| `ERROR` | Unrecoverable error |

---

## Project Structure

```
OpenAlpha-Brain/
├── main.py              # FastAPI server + session routes
├── loop_engine.py       # Core autonomous research loop
├── brain_client.py      # WorldQuant BRAIN REST API client
├── prompts.py           # LLM system prompt + ELM mutation feedback builders
├── llm_client.py        # Multi-provider LLM client (Groq/Gemini/OpenAI)
├── validator.py         # Local syntax + IQC metric validator
├── alpha_parser.py      # Parses LLM output into structured AlphaResult
├── session_manager.py   # JSON-backed session persistence
├── models.py            # Pydantic models (AlphaResult, BrainSubmissionResult, etc.)
├── config.py            # Settings loaded from .env
├── watch_session.py     # CLI watcher — runs until BRAIN PASS
├── static/index.html    # Real-time dashboard
├── .env.example         # Environment template
└── requirements.txt
```

---

## POMDP Memory Architecture

Each research session maintains a **belief state** that prevents redundant exploration:

- **Topology Map** — tracks which AST topologies have been explored (PASSED/FAILED/CROWDED)
- **Dataset Usage** — counts attempts per factor family; forces pivot after 3 consecutive fails
- **Failure Catalog** — logs each failure with fingerprint, metric values, mutation attempted
- **Open Frontiers** — unexplored 5-dimensional fingerprint combinations
- **Rejected Motifs** — topologies permanently retired for this session

---

## Security

- `.env` is **gitignored** — credentials never leave your machine
- `/sessions` directory is **gitignored** — session state stays local
- BRAIN auth uses HTTP Basic Auth → session cookie (no API key storage)

---

## License

MIT — see [LICENSE](LICENSE)

---

*Built for WorldQuant IQC 2026 · Powered by Groq / Gemini · BRAIN API v1*
