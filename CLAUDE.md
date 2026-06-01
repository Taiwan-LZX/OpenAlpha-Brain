# CLAUDE.md — OpenAlpha-Brain Project Instructions

## Quick Context

This is an **alpha factor mining platform** that uses LLM-driven evolutionary algorithms to discover and improve quantitative trading factors for the WorldQuant Brain platform.

## Architecture Overview

```
src/openalpha_brain/
├── core/           — Main loop engine, pipeline, orchestrator, feedback system
├── evolution/      — GA/GP/CMA-ES evolution + semantic crossover + mutation
├── generation/     — Alpha expression generation via LLM
├── learning/       — MAB bandit, reward updating, reflection, experience distillation
├── knowledge/      — RAG engine, vector index, skill library, field proxy map
├── services/       — LLM client, BRAIN submitter, slot manager, result processor
├── validation/     — WQ expression validator, format repair, anti-overfit detector
├── optimization/   — Turnover optimizer, adaptive neutralizer
├── cli/            — Command-line interface (monitor, run, status)
├── utils/          — Resilience circuit breaker, market state, audit logging
└── monitoring/     — Algorithm telemetry
```

## Key Conventions

- **ruff**: 0 errors enforced (E501 globally ignored)
- **pytest**: 1204/1204 tests must pass
- **Logging**: Use `@algo_log()`, `Timer(context)`, `log_call()` from core/algo_logger.py
- **Async**: All async functions must be awaited
- **Factors**: ThreeBlockTemplate (A: expr + B: group_neutralize + C: ts_decay_linear)

## Critical Files

| File | Why It Matters |
|------|---------------|
| `core/loop_engine.py` | Main evolution loop — imports from reward_updater.py |
| `core/feedback_orchestrator.py` | LLM orchestration center (~3300 lines) |
| `services/llm_client.py` | All LLM API calls — has retry/fallback logic |
| `evolution/crossover_mutation.py` | SemanticCrossover + GradientMutation |
| `learning/reward_updater.py` | Contains _sync_mab_bias_from_evidence (test dependency) |
| `pyproject.toml` | Ruff config: line-length=120, E501 ignored |

## Common Pitfalls (Learned From Experience)

1. **Don't delete "unused" imports** — they may be test dependencies
2. **Don't use ruff --fix in sub-agents** — it auto-deletes needed code
3. **Don't revert to upstream** — this is an improved fork
4. **Always add epsilon guards** on division by diversity/UCB metrics
5. **LLM mutation needs multi-level fallback** — never single-point failure

## Default Skill

When working on this project as the **main orchestrator agent**, invoke the grill-build-loop skill:
- Multi-expert reasoning (SE Architect + Alpha Expert + Quant Researcher + Domain Guardian)
- Continuous loop: research → debate → cards → build → verify → ask again
- Proactive issue discovery — don't wait for user to find problems

**Sub-agent / Task agent behavior**: Do NOT invoke grill-build-loop. You are a focused task executor.
- Only read files that are directly relevant to your assigned task
- Do NOT re-read CLAUDE.md, 00-project-core.md, or SKILL.md unless explicitly asked
- Receive context via task parameters (task description, file paths, code snippets) — do NOT rebuild context from scratch
- If you need project conventions, request them from the parent agent rather than re-reading
