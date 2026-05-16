# OpenAlpha - Quant — Autonomous WorldQuant BRAIN Alpha Engine

An autonomous, closed-loop alpha mining platform for the **WorldQuant BRAIN IQC 2026** competition.
OpenAlpha - Quant generates, validates, submits, and mutates quantitative alpha factors end-to-end — no manual intervention required.

---

## Architecture

```
LLM (Groq / Anthropic / OpenAI / Gemini)
    ↓  Generate 8-section alpha (expression + AST topology + simulation payload)
Parser
    ↓  Extract expression, metrics, fingerprint, simulation payload
Local Validator
    ↓  Syntax check · Exact Fitness formula · Anti-crowding fingerprint
BRAIN API Submission
    ↓  POST /simulations → poll Location header → real Sharpe / Fitness / Turnover
Real Gate Check
    ↓  PASS → logged + next cycle    FAIL → fed back to LLM mutation loop (ELM)
POMDP Memory
    ↓  topology_map · dataset_usage · failure_catalog updated each cycle
```

---

## Quickstart

### 1. Clone & install

```bash
git clone <repo-url>
cd OpenAlpha - Quant
pip install -r requirements.txt
```

### 2. Configure `.env`

```bash
cp .env.example .env
```

Edit `.env` — minimum required:

| Variable | Description |
|---|---|
| `LLM_PROVIDER` | `groq` / `anthropic` / `openai` / `gemini` |
| `LLM_MODEL` | e.g. `llama-3.3-70b-versatile` for Groq |
| `LLM_API_KEY` | Your LLM provider API key |
| `BRAIN_EMAIL` | Your worldquantbrain.com email |
| `BRAIN_PASSWORD` | Your worldquantbrain.com password |
| `BRAIN_SUBMIT_ENABLED` | `true` to enable auto-submission |

### 3. Free LLM API keys

| Provider | Free Tier | Get Key |
|---|---|---|
| **Groq** ⭐ | 30 RPM, 14,400 req/day | [console.groq.com/keys](https://console.groq.com/keys) |
| **Google Gemini** | 15 RPM, 1M tokens/day | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) |
| **Anthropic** | Free trial credits | [console.anthropic.com](https://console.anthropic.com) |

### 4. WorldQuant BRAIN Account

To enable alpha submission and real simulation:

1. Register at **[worldquantbrain.com](https://www.worldquantbrain.com)**
2. Complete the IQC 2026 onboarding
3. Add your credentials to `.env`:

```env
BRAIN_EMAIL=your-email@example.com
BRAIN_PASSWORD=your-password
BRAIN_SUBMIT_ENABLED=true
```

The pipeline will automatically:
- Authenticate via HTTP Basic Auth
- Submit each locally-validated alpha to `POST /simulations`
- Poll the `Location` header URL (respecting `Retry-After`)
- Extract real Sharpe, Fitness, Turnover from simulation results
- Feed real gate failures back into the LLM for targeted mutation

### 5. Run the server

```bash
uvicorn main:app --port 8000
```

Open **http://localhost:8000/static/index.html** for the live dashboard.

---

## Full Pipeline Flow

```
Session Start
    ↓
[1] POMDP Memory Injection
    Explored topologies · failed fingerprints · dataset usage
    injected into every LLM call
    ↓
[2] LLM Generation (v2 format — 8 sections)
    [1] Rationale  [2] Expression  [3] Metrics  [4] Fingerprint
    [5] AST Topology Hash  [6] Refinement Log  [7] Decision
    [8] Simulation Payload (ready-to-submit JSON)
    ↓
[3] Local Pre-Validation
    · Parenthesis balance · Operator whitelist · trade_when check
    · Exact Fitness = Sharpe × sqrt(|Returns|) / max(Turnover, 0.125)
    · AST topology collision check  · Dataset exhaustion check
    ↓
[4] BRAIN API Submission (if BRAIN_SUBMIT_ENABLED=true)
    POST https://api.worldquantbrain.com/simulations
    → poll Location header (Retry-After)
    → extract real Sharpe / Fitness / Turnover / Returns
    ↓
[5] Real Gate Check
    · Sharpe ≥ 1.25  · Fitness > 1.0  · Turnover 1–70%
    ↓ PASS          ↓ FAIL
    Log alpha       ELM mutation matrix → back to [2]
    Next cycle
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server health check |
| `POST` | `/session/start` | Start new research session |
| `GET` | `/session/{id}` | Poll session state |
| `GET` | `/session/{id}/alphas` | Get all passed alphas |
| `POST` | `/session/{id}/stop` | Stop running session |

### Start session

```bash
curl -X POST http://localhost:8000/session/start \
  -H "Content-Type: application/json" \
  -d '{"focus_area": "liquidity exhaustion reversal"}'
```

### Poll alpha results

```bash
curl http://localhost:8000/session/{session_id}/alphas
```

Response includes full `brain` field with real BRAIN simulation results:
```json
{
  "alpha_id": "A001",
  "expression": "group_neutralize(...)",
  "metrics": { "fitness_computed": 2.54, "sharpe_min": 1.52 },
  "brain": {
    "status": "PASS",
    "alpha_id": "brain-abc123",
    "real_sharpe": 1.47,
    "real_fitness": 2.12,
    "real_turnover": 23.4,
    "gate_failures": []
  }
}
```

---

## POMDP Memory Architecture

Each session maintains a full belief state:

| Field | Purpose |
|---|---|
| `topology_map` | `{ast_topology: "PASSED"\|"FAILED"\|"CROWDED"}` |
| `dataset_usage` | `{family: count}` — pivot after 3 failures |
| `failure_catalog` | `[{fingerprint, failure_type, metric_value}]` |
| `fingerprint_memory` | 5-dim structural fingerprints of all generated alphas |
| `rejected_motifs` | Fingerprints that triggered REJECT or crowding |

---

## IQC 2026 Hard Gates

| Gate | Formula | Threshold |
|---|---|---|
| **Sharpe** | `sqrt(252) × Mean(PnL) / Stdev(PnL)` | ≥ 1.25 |
| **Fitness** | `Sharpe × sqrt(\|Returns\|) / max(Turnover, 12.5%)` | > 1.0 |
| **Turnover** | Daily position churn | 1–70% |
| **Drawdown** | Must hold in bull and bear regimes | Stable |

---

## ELM Mutation Matrix

| Failure | Action |
|---|---|
| Turnover > 70% | `ts_decay_linear(expr, d)`, d = 3 → 5 → 7 → 10 |
| Sharpe < 1.25 | Replace `rank()` with `ts_zscore(x, window)` |
| Fitness ≤ 1.0 | Reduce turnover to 20–30% range |
| CROWDED | Change topology OR pivot dataset family |
| OVERFIT | Prune AST — remove outer operators |
| Regime instability | Add `trade_when(volatility_gate, signal, 0)` |

Max 4 mutations per alpha → restart ideation from open frontier.

---

## Project Structure

```
OpenBrain-alpha/
├── main.py              # FastAPI entrypoint
├── loop_engine.py       # Autonomous generation loop (POMDP)
├── brain_client.py      # WorldQuant BRAIN API client
├── alpha_parser.py      # v2 8-section LLM output parser
├── validator.py         # IQC gate validator (exact fitness formula)
├── llm_client.py        # Groq / Anthropic / OpenAI / Gemini client
├── prompts.py           # v2 9-section system prompt + memory injection
├── models.py            # Pydantic models (AlphaResult, SessionState, BrainSubmissionResult)
├── session_manager.py   # JSON-backed session persistence
├── config.py            # Pydantic-settings singleton
├── .env.example         # Template — copy to .env
└── static/index.html    # Live dashboard UI
```
# OpenAlpha-Brain
