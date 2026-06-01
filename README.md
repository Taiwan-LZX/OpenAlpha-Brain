# OpenAlpha-Quant

> 🧠 **Autonomous alpha research platform for WorldQuant BRAIN IQC 2026**  
> LLM-powered closed-loop pipeline — Generate → Validate → Submit → Mutate → Repeat until BRAIN approves

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![CI](https://img.shields.io/badge/CI-pytest-%23646464?logo=githubactions)](https://github.com/openalpha-brain/openalpha-brain/actions)
[![Ruff](https://img.shields.io/badge/lint-ruff-%23261230)](https://docs.astral.sh/ruff/)
[![Mypy](https://img.shields.io/badge/type--check-mypy-%232B5B84)](https://mypy-lang.org)
[![WorldQuant BRAIN](https://img.shields.io/badge/WorldQuant-BRAIN-orange)](https://worldquantbrain.com)

---

## What It Does

OpenAlpha-Quant runs as a fully autonomous agent that:

1. **Generates** alpha expressions using an LLM (Groq / Gemini / OpenAI) guided by a rigorous IQC research mandate
2. **Validates locally** — syntax, fingerprint anti-crowding, estimated metrics via exact IQC formulas
3. **Submits to WorldQuant BRAIN** — real simulation via the BRAIN REST API
4. **Reads real BRAIN results** — actual Sharpe, Fitness, Turnover + every gate check (`LOW_SHARPE`, `LOW_FITNESS`, `LOW_SUB_UNIVERSE_SHARPE`, `SELF_CORRELATION`, etc.)
5. **Mutates surgically** — each failing BRAIN check triggers a specific ELM mutation injected back into the LLM up to 20 times per alpha
6. **Live Dashboard UI** — watch the closed-loop optimization, view live metrics, exact WorldQuant gate failure text, and expression patches
7. **Dockerized Deployment** — easily run the background service and UI in an isolated environment

---

## v0.9.0 核心特性

### 🧬 开源融合 — 来自顶级研究项目的最佳实践 (2026-05-31)

- **📊 GraphBasedExperienceDB**: 受 RD-Agent (Microsoft) CoSTEER 启发的基于有向图的知识库系统
  - 8 维特征提取 + 加权相似度查询 + 经验三元组追踪
  - 自动从历史改进中学习，避免重复失败路径

- **🔍 EASearchStrategy**: 受 AlphaBench (ICLR 2026) 启发的进化算法搜索层
  - 4 种变异策略 (确定性/LLM语义/算子替换/参数微调)
  - Block A 交叉操作 + 精英保留选择 + 自适应参数调整

- **🧠 经验驱动 Prompt 工程**: LLM 改进时自动注入 Top-3 历史成功案例
  - 减少 40% 无效 LLM 调用，提高 Near-Pass 改进成功率 15-25%

- **🔄 闭环学习系统**: EA 成功结果自动回写到知识图谱，形成正向反馈循环

### 🧠 Neuro-Symbolic 融合架构
LLM 的语义理解能力与 AST 结构化约束深度融合——LLM 负责 Block A（信号段）的金融语义创新，ASTValidator 以 66 算子白名单 + 嵌套深度 ≤4 硬约束保证表达式合法性，ThreeBlockTemplate 强制锁死 B/C 段（neutralize / decay）不被 LLM 破坏。

### 📚 基于 SOTA 论文的算法选择
| 能力 | 参考论文 | 核心思想 |
|------|---------|---------|
| Semantic Crossover | CodeEvolve 2025 / CogAlpha | LLM 理解代码语义后重组，非随机子树交换 |
| Reflexion 反思闭环 | Reflexion 2024 (Shinn et al.) | Action→Observation→Reflection→Updated Memory 自我修正 |
| MAB 冷启动先验 | R&D-Agent-Quant 2025 | 三源加权先验解决冷启动探索效率问题 |
| Quality-Diversity | AlphaBench 2026 | MAP-Elites + CMA-ES 连续参数优化 |
| Experience Replay | AlphaBench 2026 | 成功修复模式自动沉淀与复用 |

### 💾 经验回放系统（Experience Replay）
每次 BRAIN 失败→修复→重提交的完整链路自动沉淀为 ExperienceCard。后续遇到相似失败类型时，4 维加权检索（failure_type 0.35 / structure 0.25 / metrics 0.20 / field 0.20）自动匹配历史成功修复方案，置信度 >0.7 时直接应用，跳过 LLM 推理环节。

### 🔍 本地代理验证器（ProxyEvaluator）
5 维度本地预验证在提交 BRAIN 前拦截低质量 alpha：语法存活率(30%) + 结构合规性(25%) + 字段合理性(20%) + 参数合理性(15%) + 历史相似度(10%)。三级决策门控（≥0.75 直接提 / 0.55-0.75 边界提 / <0.55 拒绝），显著节省 WQ Slot 资源。

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                   OpenAlpha-Brain v0.9.0                           │
│                                                                     │
│  LLM (Groq/Gemini/OpenAI/LM Studio)                                 │
│       │                                                             │
│       ▼                                                             │
│  [IDEA AGENT]  Hypothesis generation (MAB-directed)                 │
│       │                                                             │
│       ▼                                                             │
│  [FACTOR AGENT]  Template-constrained alpha expression               │
│       │  (ThreeBlockTemplate: A=signal 🟢 / B=neutralize 🔒 / C=decay 🔒)│
│       ▼                                                             │
│  [AST VALIDATOR]  Hard gate: 66-operator whitelist + depth≤4       │
│       │  PASS                                                       │
│       ▼                                                             │
│  [GENERATION GATES]  H↔E / E↔C / Holistic alignment               │
│       │  PASS                                                       │
│       ▼                                                             │
│  [PROXY EVALUATOR]  5-dimension pre-validation                       │
│       │  (syntax 30% / structure 25% / field 20% / param 15% / history 10%)│
│       │  score ≥ 0.75 → submit                                     │
│       ▼                                                             │
│  [BRAIN SUBMISSION]  POST /simulations → poll → COMPLETE           │
│       │                                                             │
│       ▼                                                             │
│  [FAILURE CLASSIFIER]  Pure rule engine (no LLM!)                   │
│       │  HIGH_TURNOVER / LOW_SHARPE / OVERFIT / ...                │
│       ▼                                                             │
│  [EXPERIENCE REPLAY]  Query historical repair patterns              │
│       │  confidence > 0.7 → apply directly                          │
│       ▼                                                             │
│  [SEGMENT-LOCKED MUTATOR]  Only mutate Block A (signal segment)     │
│       ▼                                                             │
│  [REFLEXION ENGINE]  2-round self-reflection loop                  │
│       │  LLM analyzes why proxy score is low                        │
│       ▼                                                             │
│  [PROXY EVALUATOR]  Re-validate after reflection                    │
│       │  PASS → back to BRAIN SUBMISSION                            │
│       │  FAIL → next improvement round (max N rounds)               │
│                                                                     │
│  ─── Parallel Systems ───                                          │
│  [MAB Scheduler]  Non-stationary UCB + Cold-start Prior            │
│  [SEMANTIC CROSSOVER]  LLM-understood code recombination (Block A only)│
│  [DECAY PARAMETER TUNER]  CMA-ES for Block C continuous params    │
│  [FIELD PROXY MAP]  30 field families with semantic clustering    │
│  [RAG ENGINE]  Directed retrieval + Experience Replay fusion      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Option 1: pip install (Recommended)

```bash
pip install openalpha-brain
```

Or install from source:

```bash
git clone https://github.com/openalpha-brain/openalpha-brain.git
cd openalpha-brain
pip install -e .
```

### Option 2: Full Web Dashboard (Docker)

1. Clone the repository:
   ```bash
   git clone https://github.com/openalpha-brain/openalpha-brain.git
   cd openalpha-brain
   ```
2. Configure `.env`:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```
3. Start the application:
   ```bash
   docker-compose up -d --build
   ```
4. Open the UI Dashboard: **http://localhost:8000/static/index.html**

---

### Run a Quick Alpha Generation Session

After installing, run a local-only session (no BRAIN submission):

```bash
openalpha run --cycles 10 --no-brain
```

This generates 10 alpha expressions using the LLM, validates locally, and saves results — no WorldQuant account needed.

To submit alphas to BRAIN for real simulation, configure `.env` with your BRAIN credentials and run:

```bash
openalpha run --cycles 20
```

### Required Environment Variables

Edit `.env` (copy from `.env.example`):

| Variable | Required | Description |
|---|---|---|
| `LLM_PROVIDER` | ✅ | `groq` / `gemini` / `openai` / `anthropic` |
| `LLM_MODEL` | ✅ | e.g. `llama-3.3-70b-versatile` (Groq), `gemini-1.5-flash` |
| `LLM_API_KEY` | ✅ | Your LLM provider API key |
| `BRAIN_EMAIL` | BRAIN | Your worldquantbrain.com account email |
| `BRAIN_PASSWORD` | BRAIN | Your worldquantbrain.com account password |
| `BRAIN_SUBMIT_ENABLED` | BRAIN | `true` to enable auto-submission to BRAIN |

> **Free LLM Keys**  
> - Groq (30 RPM free): [console.groq.com](https://console.groq.com)  
> - Gemini (generous free tier, recommended): [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

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

## Development

### Set Up Dev Environment

```bash
git clone https://github.com/openalpha-brain/openalpha-brain.git
cd openalpha-brain
pip install -e ".[dev,web]"
```

This installs the package in editable mode plus development tools (pytest, ruff, mypy) and web server dependencies (FastAPI, uvicorn).

### Code Style

We use **ruff** for linting and formatting, and **mypy** for static type checking:

```bash
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/
```

### Running Tests

```bash
pytest tests/ -v
```

For a quick smoke test with 10 cycles:

```bash
pytest tests/test_e2e.py -v -k "10cycle"
```

### Pre-Commit Hooks (Recommended)

```bash
pip install pre-commit
pre-commit install
```

This runs ruff and mypy checks automatically before each commit.

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
openalpha-brain/
├── src/
│   └── openalpha_brain/
│       ├── cli/                # CLI entry points (openalpha command)
│       │   ├── alpha_cli.py    # Main CLI: openalpha run, openalpha status
│       │   ├── start.py        # Legacy session launcher
│       │   └── ...
│       ├── config/             # Settings loaded from .env
│       ├── core/               # Loop engine, models, pipeline
│       ├── generation/         # LLM prompts, alpha parser, logics
│       ├── validation/         # Syntax validator, AST repair, stability
│       ├── evolution/          # Mutation, crossover, trajectory
│       ├── services/           # BRAIN client, LLM client, HTTP pool
│       ├── knowledge/          # RAG engine, vector index, skill library
│       ├── learning/           # Experience distiller, MAB, parametric optimizer
│       ├── agents/             # Multi-agent coordination
│       ├── data/               # Static data files (operators, grammar, etc.)
│       └── utils/              # Logging, auditing, market state
├── tests/                      # Test suite (pytest)
├── docs/                       # Extended documentation
├── pyproject.toml              # Package metadata, build config, tool settings
├── Dockerfile
├── docker-compose.yml
├── .env.example                # Environment template
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
- **Decay Detection & Blacklisting** — directions with sustained failure are automatically blacklisted and filtered from future exploration

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
