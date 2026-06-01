# CLAUDE.md — OpenAlpha-Brain Project Instructions

## Quick Context

This is an **alpha factor mining platform** that uses LLM-driven evolutionary algorithms to discover and improve quantitative trading factors for the WorldQuant Brain Platform.

**Architecture**: 6-Layer Hierarchical + Central Feedback Orchestrator + Event Bus
**Status**: Production-ready (ruff=0, pytest=1212, 18/18 algorithms active)

## Architecture Overview

```
src/openalpha_brain/
├── core/                    — Core engine & 6-Layer architecture
│   ├── loop_engine.py       — Thin orchestrator (main loop entry)
│   ├── feedback_orchestrator.py — Central hub (~3300 lines), coordinates all feedback
│   ├── layers/              — ★ 6-Layer Architecture (ACTIVE)
│   │   ├── exploration_director.py   — L1: Direction selection (MAB+NavFusion weighted)
│   │   ├── generation_pipeline.py    — L2: Alpha generation (LLM+RAG+MAB injection)
│   │   ├── evaluation_gateway.py     — L3: BRAIN submit + result collection
│   │   ├── improvement_orchestra.py  — L4: Improvement strategy dispatch
│   │   ├── robustness_gate.py        — L5: Anti-overfit + statistical gates
│   │   └── persistence_layer.py      — L6: State save/load + telemetry
│   ├── decision_engine.py    — Route classification (passed/failed/near-pass/reversal)
│   ├── result_router.py      — Result distribution to feedback consumers
│   ├── navigation_fusion.py  — Multi-source direction fusion
│   ├── pipeline.py           — Legacy pipeline (kept for compat)
│   ├── events.py             — Event bus (WebSocket/notification backbone)
│   ├── loop_state.py         — Global state hub (_mab, _rag_engine, _whitelist_mgr, etc.)
│   └── models.py             — Pydantic models (SessionState, AlphaTrajectory, etc.)
├── evolution/               — Evolutionary algorithms (95%+ utilization)
│   ├── crossover_mutation.py — SemanticCrossover + GradientMutation + CrossoverMutationEngine
│   ├── mutation_engine.py    — BrainAwareMutationEngine (18 mutation strategies)
│   ├── semantic_mutator.py   — LLM-driven semantic mutation
│   ├── ea_search.py          — EASearchStrategy (GA/GP population-based)
│   ├── tot_search.py         — ToTSearchStrategy (Tree-of-Thought exploration)
│   ├── adaptive_neutralizer.py — AdaptiveNeutralizer (sector/industry neutralization)
│   ├── near_pass_improver.py — NearPassImprover (gate-tweaking for near-pass alphas)
│   ├── fitness_boost.py      — FitnessBoostEngine (priority boost re-submission)
│   ├── strategy_classifier.py — StrategyClassifier (momentum/reversal/volatility/value)
│   └── trajectory_mutation.py — TrajectoryMutation (research-trajectory-level editing)
├── generation/              — Alpha expression generation
│   ├── alpha_generator.py    — Main generator (LLM orchestration)
│   ├── alpha_parser.py       — Expression parsing & JSON extraction
│   ├── alpha_logics.py       — Alpha logic templates library
│   ├── prompts.py            — LLM prompt templates
│   └── template_reasoning_generator.py — ThreeBlockTemplate reasoning generator
├── learning/                — Learning & self-improvement (with persistence)
│   ├── mab.py                — HierarchicalMAB (direction→operator→field, 3-layer)
│   │                          ★ save_state() / load_state() → .data/mab_state.json
│   ├── reflection_engine.py  — Success/failure reflection analysis
│   ├── experience_distiller.py — Experience card extraction from failures/evidence
│   │                          ★ save_cards() / load_cards() → .data/experience_cards.json
│   └── param_optimizer.py    — Hyperparameter optimization
├── knowledge/               — Knowledge base & retrieval
│   ├── rag_engine.py         — RAG vector retrieval (operators + fields + financial_logic)
│   ├── operator_registry.py  — 60+ temporal operators with risk classification
│   ├── field_proxy_map.py    — Field family proxy (close→price, volume→liquidity)
│   ├── skill_library.py      — Alpha skill templates
│   └── graph_experience_db.py — Graph-structured experience database
├── services/                — External service clients
│   ├── llm_client.py         — LLM API client (retry/fallback/multi-provider)
│   ├── brain_client.py       — BRAIN platform client (Semaphore(3) rate limiting)
│   ├── slot_manager.py       — 3-slot concurrent submission manager
│   ├── result_processor.py   — BRAIN result parsing (Sharpe/Fitness/Turnover)
│   ├── http_pool.py          — Async HTTP connection pool
│   └── algo_connectivity.py  — Service health checker
├── validation/              — Expression validation & quality gates
│   ├── wq_expression_validator.py — WQ syntax validator (ThreeBlockTemplate support)
│   ├── format_repair.py      — Expression format repair
│   ├── ast_repair.py         — AST-level expression repair
│   ├── signal_arbiter.py     — Multi-source signal ranking (RAG+MAB+Whitelist)
│   ├── anti_overfit_detector.py — GARCH + OS decay ratio overfit detection
│   ├── validator.py          — Unified validation entry point
│   ├── official_scorer.py    — WQ official scoring emulation
│   └── stability_guard.py    — Stability & reproducibility guard
├── optimization/            — Post-submission optimization
│   └── turnover_optimizer.py — Turnover reduction optimizer
├── cli/                     — User interface layer
│   ├── main.py               — FastAPI app (HTTP API + WebSocket server)
│   │                          ★ NEW: PATCH/GET /session/{id}/config, GET /mab/status
│   ├── run_alpha.py          — User operational script (start/check/run/mine/full)
│   ├── start.py              — Unified launcher (--web / --cli / --run)
│   ├── alpha_cli.py          — Interactive CLI REPL
│   ├── monitor_session.py    — Session monitoring dashboard
│   ├── algo_monitor.py       — Algorithm status monitor (WebSocket push)
│   └── ws_broadcaster.py     — WebSocket event broadcast
├── utils/                   — Shared utilities
│   ├── algo_logger.py        — @algo_log(), Timer, log_call (structured logging)
│   ├── resilience.py         — Circuit breaker pattern
│   └── market_state.py       — Market regime inference
├── monitoring/              — Runtime monitoring
│   └── algorithm_telemetry.py — Algorithm call counting & performance tracking
└── agents/                  — Multi-agent system
    └── adaptive_agent.py     — Adaptive strategy agent
```

## Operational Scripts (3-script architecture)

```
tools/
├── unified_health.py        — Unified health diagnostics
│   health                    Quick: .env + 6-Layer + 18-algo import check
│   health --deep            Deep: +BRAIN auth + LLM gen + submit + 15 components + Slot test
│   health --dry-run         CI-friendly config-only check
│
├── loop_guardian.py          — AI self-monitoring & testing (for agent use)
│   guard --mock --quick     Single-cycle smoke test (offline)
│   guard --mock --monitor --cycles N  Multi-cycle monitoring with trend analysis
│   guard --mock --stress    Stress test (boundary + concurrency + recovery)
│   guard --report           Generate JSON + Markdown reports
│
src/openalpha_brain/cli/
└── run_alpha.py             — User-facing operational interface
    start [--focus]          Start mining session via HTTP API
    check [session_id]       Check session status & alpha results
    run [--cycles N] [--focus]  Full E2E self-loop test
    mine [--focus] [--wait]  Start real mining & wait for results
    full                     Complete diagnostic pipeline
```

## Key Conventions

- **ruff**: 0 errors enforced (E501 globally ignored, line-length=120)
- **pytest**: 1212 tests must pass (1204 core + 8 persistence + 9 skipped)
- **Logging**: Use `@algo_log()`, `Timer(context)`, `log_call()` from [algo_logger.py](src/openalpha_brain/utils/algo_logger.py)
- **Defensive logging**: Use `[DEFENSIVE_LOG]` prefix for diagnostic logs (grep-friendly)
- **Async**: All async functions must be awaited
- **Factors**: ThreeBlockTemplate (A: expr + B: group_neutralize + C: ts_decay_linear)
- **Persistence**: MAB + ExperienceDistiller auto-save to `.data/` each cycle & on exit
- **Epsilon guards**: Always use `max(x, 1e-6)` on division by diversity/UCB metrics

## Critical Files (Updated)

| File | Why It Matters |
|------|---------------|
| [loop_engine.py](src/openalpha_brain/core/loop_engine.py) | Thin orchestrator — delegates to 6 Layers |
| [feedback_orchestrator.py](src/openalpha_brain/core/feedback_orchestrator.py) | Central coordination hub (~3300 lines) |
| [layers/](src/openalpha_brain/core/layers/) | 6-Layer architecture (L1-L6 all ACTIVE) |
| [llm_client.py](src/openalpha_brain/services/llm_client.py) | All LLM calls — retry/fallback/multi-provider |
| [crossover_mutation.py](src/openalpha_brain/evolution/crossover_mutation.py) | SemanticCrossover + GradientMutation |
| [mab.py](src/openalpha_brain/learning/mab.py) | HierarchicalMAB — now with get_operator_stats/get_field_stats + persistence |
| [generation_pipeline.py](src/openalpha_brain/core/layers/generation_pipeline.py) | L2 Layer — RAG integrity check + MAB prompt injection + RAG usage validation |
| [main.py](src/openalpha_brain/cli/main.py) | FastAPI app — 3 new runtime config endpoints |

## Data Flow Architecture (Post-fix)

```
                    ┌─────────────────────────────┐
                    │   FeedbackOrchestrator      │ ← Central Hub
                    │   (feedback coordination)    │
                    └───────────┬─────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ L1 Explor. Dir.  │  │ L2 Generate Pipe │  │ L3 Eval Gateway  │
│ MAB★ + NavFusion │  │ RAG★ + MAB★ inj │  │ BRAIN Semaphore(3)│
└────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
         │                      │                      │
         ▼                      ▼                      ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ L4 Improve Orch. │  │ L5 Robustness G. │  │ L6 Persist Layer │
│ Crossover+Mutate │  │ AntiOverfit★     │  │ .data/ save/load★ │
└──────────────────┘  └──────────────────┘  └──────────────────┘

★ = Enhanced in latest update (MAB data flow fix, RAG binding, persistence)
```

## HTTP API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/session/start` | Start new mining session |
| GET | `/session/{id}` | Full session state snapshot |
| GET | `/session/{id}/alphas` | Passed alphas list |
| PATCH | `/session/{id}/config` | **NEW** Runtime param adjustment (focus_area, temperature, max_cycles, mode) |
| GET | `/session/{id}/config` | **NEW** Config snapshot with MAB top-3 directions |
| GET | `/mab/status` | **NEW** Full MAB state (directions/operators/fields expectations) |
| POST | `/session/{id}/stop` | Request graceful stop |
| POST | `/session/{id}/pause` | Pause cycle execution |
| POST | `/session/{id}/resume` | Resume paused session |
| WS | `/ws/events` | Real-time event stream |
| WS | `/ws/monitor` | Algorithm monitoring channel |

## Common Pitfalls (Learned From Experience)

1. **Don't delete "unused" imports** — they may be test dependencies
2. **Don't use ruff --fix in sub-agents** — it auto-deletes needed code
3. **Don't revert to upstream** — this is an improved fork of hitendras510/OpenAlpha-Brain
4. **Always add epsilon guards** on division by diversity/UCB metrics (recommend 1e-6)
5. **LLM mutation needs multi-level fallback** — never single-point failure
6. **MAB stats must flow downstream** — get_direction_stats()/get_operator_stats() must be consumed by ExplorationDirector and GenerationPipeline
7. **RAG results must be enforced in Generator** — _check_rag_integrity() + _validate_rag_usage() guards the data binding
8. **Persistence is automatic** — MAB.save_state() + ExperienceDistiller.save_cards() called each cycle-end and on exit
9. **[DEFENSIVE_LOG] prefix** — use for all diagnostic/suspicion logs, grep-friendly in production

## Default Skill

When working on this project as the **main orchestrator agent**, invoke the grill-build-loop skill:
- 6-Phase lifecycle (Inquire → Assess → Plan → Execute → Verify → Loop)
- Behavioral invariant: every response MUST end with AskUserQuestion
- Project status assessment before planning; decision cards after every action cycle

**Sub-agent / Task agent behavior**: Do NOT invoke grill-build-loop. You are a focused task executor.
- Only read files that are directly relevant to your assigned task
- Do NOT re-read CLAUDE.md, 00-project-core.md, or SKILL.md unless explicitly asked
- Receive context via task parameters (task description, file paths, code snippets) — do not rebuild context from scratch
- If you need project conventions, request them from the parent agent rather than re-reading
