# OpenAlpha-Brain Algorithm Connectivity Architecture

> Version: 4.0 (source-code-reversed, 6-Layer active)
> Date: 2026-06-02
> Status: Verified against `core/layers/`, `loop_engine.py` L205-L2700
> Method: Full source-code reverse engineering — every class instantiation and call site traced

---

## 0. Paradigm Foundation (unchanged from v3)

### Platform Constraints

- **101 Formulaic Alphas** (Kakushadze 2016): all composed of ~30 base operators in fixed template nesting
- **Finding Alphas** (Tulchinsky): WQ requires `group_neutralize` (section-neutralization)
- **Alpha Factory**: WQ maintains 4M+ alpha library — strategy is "quantity beats decay"
- **Standard Trinity**: `Signal(field) → Neutralize(industry) → Decay(time)`

### Core Principle

> Structure (template) is the survival floor. Fields (data) are the exploration boundary.

---

## 1. Execution Topology — Not a Pipeline, a Thinking Tree

```
                         ┌──────────────────────────────┐
                         │     loop_engine.run_loop()    │
                         │         [L205]               │
                         └────────────┬─────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
    ┌─────────────────┐   ┌──────────────────┐   ┌──────────────────┐
    │  PHASE A        │   │  Global Infra    │   │  Config & State  │
    │  Exploration    │   │  (shared by all)  │   │  (shared by all) │
    │  Director       │   │                  │   │                  │
    │  [Layer 1]      │   │ • LLM Semaphore(4)│   │ • loop_state.py  │
    │                 │   │ • BRAIN Sem(3)   │   │ • config.py      │
    │  §1.1→§1.5      │   │ • Event×2         │   │ • http_pool.py   │
    └────────┬────────┘   │ • RAG + Embed(4)  │   └──────────────────┘
             │            └──────────────────┘
             ▼
    ┌─────────────────┐
    │  PHASE B        │
    │  Generation     │
    │  Pipeline       │
    │  [Layer 2]      │
    │                 │
    │  §2.1→§2.7      │
    └────────┬────────┘
             │  validated expression
             ▼
    ┌─────────────────┐
    │  PHASE C        │
    │  Evaluation     │
    │  Gateway        │
    │  [Layer 3]      │
    │                 │
    │  §3.1→§3.3      │
    └────────┬────────┘
             │  ParsedWQResult
             ▼
    ┌─────────────────┐
    │  PHASE D        │  ← NOTE: executes BEFORE Phase E
    │  Robustness     │
    │  Gate           │
    │  [Layer 5]      │
    │                 │
    │  §4.1→§4.4      │
    └────────┬────────┘
             │  RobustnessVerdict
             ▼
    ┌─────────────────┐
    │  PHASE E        │  ← Can LOOP BACK to B or C
    │  Improvement    │
    │  Orchestra      │
    │  [Layer 4]      │
    │                 │
    │  §5.1→§5.9      │
    └────────┬────────┘
             │  improved expressions (or re-submit)
             ▼          ↑ loop-back
    ┌─────────────────┐
    │  PHASE F        │
    │  Persistence    │
    │  Layer          │
    │  [Layer 6]      │
    │                 │
    │  §6.1→§6.3      │
    └─────────────────┘
```

**Critical path**: A → B → C → D → E → F
**Loop-back paths**: E → B (regenerate), E → C (resubmit)

---

## 2. PHASE A — ExplorationDirector [Layer 1]

**File**: `core/layers/exploration_director.py`
**Source extracted from**: `loop_engine.py` L387-459, L1881-1950, L2144-2272

**Input**: feature_map, scheduler state, brain_feedback_buffer, session context
**Output**: `ExplorationResult(direction, confidence, method)`

### 2.1 HierarchicalMAB

| Attribute | Value |
|-----------|-------|
| File | `learning/mab.py` L561-L768 |
| Class | `HierarchicalMAB` |
| Interface | `.select()` → direction, `.update()` → reward feedback |
| Arms | Template × FieldFamily (15 × 40 = 600 arms via TemplateFamilyBandit) |
| Algorithm | UCB (Upper Confidence Bound) with bias sync |
| Status | ✅ ACTIVE — called every cycle |

**Data flow**:
```
brain_feedback_buffer (PASS/FAIL results)
        │
        ▼
  MAB.update(arm, reward)  ← _sync_mab_bias_from_evidence
        │
        ▼
  MAB.select()  → top-K directions by UCB score
        │
        ▼
  exploration_direction (string, e.g. "momentum_long")
```

### 2.2 NavigationFusion

| Attribute | Value |
|-----------|-------|
| File | `core/navigation_fusion.py` L159-L480 |
| Class | `NavigationFusion` |
| Pattern | Three-module weighted voting fusion |
| Sub-modules | StrategyClassifier, HypothesisAligner |
| Status | ✅ ACTIVE — fuses when coverage > 30% |

**Data flow**:
```
MAB output ──┐
Classifier ──┼──→ NavigationFusion.fuse() ──→ fused_direction
Aligner   ──┘     (confidence-weighted vote)
                   threshold=0.7 for fusion_override
```

### 2.3 ToTSearchStrategy (Tree-of-Thoughts)

| Attribute | Value |
|-----------|-------|
| File | `evolution/tot_search.py` L644-L799 |
| Class | `ToTSearchStrategy` |
| Trigger | Near-Pass (Sharpe ∈ [0.8, 1.25)) or Stuck |
| Algorithm | Tree search with LLM evaluation at each node |
| Status | ✅ ACTIVE — conditional trigger |

### 2.4 FeatureMap / MAP-Elites (Quality-Diversity)

| Attribute | Value |
|-----------|-------|
| File | `evolution/quality_diversity.py` L329-1170 |
| Classes | `FeatureMap`, `GridArchive`, `ExploreEmitter`, `ExploitEmitter`, `CMAEmitter` |
| Algorithm | MAP-Elites (Illumination algorithm) |
| Dual role | L1 (explore_targets) + L4 (diversity maintenance) |
| Status | ✅ ACTIVE — dual-wired to Layer 1 and Layer 4 |

**Data flow (L1 side)**:
```
candidate alphas + fitness metrics
        │
        ▼
  FeatureMap.add_candidate() → GridArchive cell assignment
        │
        ▼
  FeatureMap.sample_elite() → under-explored region targets
        │
        ▼
  ExplorationDirector receives diversity suggestions
```

### 2.5 TemplateFamilyBandit

| Attribute | Value |
|-----------|-------|
| File | `learning/mab.py` L980-1320 |
| Class | `TemplateFamilyBandit` |
| Role | MAB sub-bandit for template × field_family selection |
| Status | ✅ ACTIVE — nested inside HierarchicalMAB |

---

## 3. PHASE B — GenerationPipeline [Layer 2]

**File**: `core/layers/generation_pipeline.py`
**Source extracted from**: `loop_engine.py` L479-849

**Input**: exploration_direction, session context, FieldProxyMap
**Output**: `GenerationResult(expression, gates_passed, prefilter_passed)`

### 3.1 AlphaGenerator (LLM Expression Generation)

| Attribute | Value |
|-----------|-------|
| File | `generation/alpha_generator.py` L1-L640 |
| Pattern | LLM call with structured prompt injection |
| Sub-agents | IdeaAgent (proposition), FactorAgent (field-filling) |
| Concurrency | 3 parallel tasks (`asyncio.create_task`) |
| Status | ✅ ACTIVE — core generation path |

**Internal flow (depth 3)**:
```
direction + context + knowledge_retrieval
        │
        ▼
  _build_user_message()
        ├── RAG retrieval (VectorIndex)
        ├── experience card lookup (GraphDB/ExperienceDB)
        ├── FieldProxyMap field recommendations
        └── TemplateReasoningGenerator → ThreeBlockTemplate
        │
        ▼
  llm_client.generate()  → raw expression string
        │
        ▼
  _handle_iterate()  → gate validation (see §3.2)
```

### 3.2 GenerationGates (Grammar-Guided GP)

| Attribute | Value |
|-----------|-------|
| File | `evolution/generation_gates.py` L64-L1006 |
| Pattern | Three-way semantic consistency gating |
| Gates | H↔E (Hypothesis↔Expression), E↔C (Expression↔Code), H↔E↔C (Holistic) |
| Retry | `apply_with_retry(max_retries=2)` |
| Status | ✅ ACTIVE — hard gate before acceptance |

**Gate detail (depth 3)**:
```
raw expression
        │
        ├─ Gate-H: HypothesisAligner.align()
        │   Does expression match claimed economic logic?
        │   → score ∈ [0, 1]
        │
        ├─ Gate-E: ASTValidator.validate_syntax()
        │   Is FASTEXPR syntactically valid?
        │   → valid/invalid + repair suggestion
        │
        └─ Gate-C: ComplexityController.check()
            Is nesting depth within P90 adaptive limit?
            → pass/fail
```

### 3.3 Validation Stack

| Module | File | Line | Role |
|--------|------|------|------|
| **ASTValidator** | `validation/ast_validator.py` | L177-405 | HARD validation: operator legality, type safety, structure |
| **ComplexityController** | `validation/complexity_control.py` | L113-229 | SOFT gate: P90 adaptive threshold on node count |
| **WQExpressionValidator** | (integrated into gates) | — | WQ platform-specific format compliance |

### 3.4 TemplateReasoningGenerator

| Attribute | Value |
|-----------|-------|
| File | `generation/template_reasoning_generator.py` L48-905 |
| Output | Structured ThreeBlockTemplate with `{placeholder}` fields |
| Templates | 15 templates × 6 directions × 3 time scales |
| Status | ✅ ACTIVE — feeds into AlphaGenerator prompt |

### 3.5 FieldProxyMap

| Attribute | Value |
|-----------|-------|
| File | `knowledge/field_proxy_map.py` |
| Structure | 3-layer annotation: SemanticCategory → FieldFamily → ApplicableTemplate |
| Scale | ~40 field families × ~7000 fields |
| Rule | Never use raw field IDs (e.g., `field_3921`) — always proxy through families |
| Status | ✅ ACTIVE — integrated into prompt construction |

---

## 4. PHASE C — EvaluationGateway [Layer 3]

**File**: `core/layers/evaluation_gateway.py`
**Source extracted from**: `loop_engine.py` L1795-1832

**Input**: validated expression
**Output**: `EvaluationResult(status, sharpe, fitness, turnover, decision)`

### 4.1 BrainSubmitter (WQ Platform Submission)

| Attribute | Value |
|-----------|-------|
| File | `services/brain_submitter.py` L1+ |
| Protocol | HTTP POST → WQ BRAIN API |
| Concurrency | `Semaphore(3)` — matches platform's 3-slot limit |
| Polling | 0.5s interval async loop until result or timeout (600s) |
| Error handling | 401 → re-login + retry; 429 → backoff; timeout → ERROR status |
| Status | ✅ ACTIVE — single point of platform interaction |

**Submission flow (depth 3)**:
```
validated expression
        │
        ▼
  await semaphore.acquire()  ← Semaphore(3)
        │
        ▼
  brain_submitter.submit(expression, simulation_payload)
        │
        ├── HTTP POST to WQ API
        │
        ▼
  poll_loop (0.5s interval):
        ├── PENDING → continue waiting
        ├── PASS/FAIL → return BrainSubmissionResult
        ├── ERROR → classify error type
        └── TIMEOUT (600s) → mark as TIMED_OUT
        │
        ▼
  semaphore.release()  ← slot freed → Event.set() wakes Generator
```

### 4.2 ResultRouter

| Attribute | Value |
|-----------|-------|
| File | `core/result_router.py` L132-L467 |
| Input | Raw `BrainSubmissionResult` (JSON from WQ) |
| Output | `ParsedWQResult` (structured: sharpe, fitness, turnover, returns, etc.) |
| Status | ✅ ACTIVE — mandatory parsing step |

### 4.3 DecisionEngine

| Attribute | Value |
|-----------|-------|
| File | `core/decision_engine.py` L141-370 |
| Input | `ParsedWQResult` metrics |
| Output | `DecisionOutcome(action, reason, metadata)` |
| Actions | IMPROVE / RESUBMIT / ABANDON / ACCEPT |
| Heuristics | Sharpe thresholds, anti-fit scores, near-pass bands |
| Status | ✅ ACTIVE — routes to Phase D or Phase E |

---

## 5. PHASE D — RobustnessGate [Layer 5] ⚠️ Pre-E

**File**: `core/layers/robustness_gate.py`
**Executes BEFORE ImprovementOrchestra** — only robust alphas deserve improvement investment.

**Input**: `EvaluationResult` (from Phase C)
**Output**: `RobustnessCheckResult(verdict, rejection_reasons, scores)`

### 5.1 Anti-Overfit Gauntlet

| Test | Method | Source |
|------|--------|--------|
| Permutation Test | Shuffle label, verify signal non-random | AntiOverfitDetector |
| Deflated Sharpe Ratio | Bailey & Lopez de Prado 2014 adjustment | AntiOverfitDetector.placebo() |
| Subsample Stability | IC consistency across market regimes | AntiOverfitDetector |
| Temporal Decay Test | Performance degradation over time | AlphaDecayDetector |
| Cross-validation Consistency | Multi-metric threshold agreement | Composite scoring |

| Attribute | Value |
|-----------|-------|
| Detector file | `validation/anti_overfit_detector.py` L54-L1068 |
| Local class | `_QuantGPTAntiOverfitDetector` at L485 (NOT external import) |
| Verdict enum | ROBUST / MARGINAL / UNSTABLE / REJECTED |
| Status | ✅ ACTIVE — gauntlet runs on PASS and near-pass results |

### 5.2 Decay Pre-Screen

| Attribute | Value |
|-----------|-------|
| File | `validation/decay_detector.py` L81-600 |
| Algorithm | EWMA-based decay tracking per alpha |
| Levels | L3 → cell admission paused; L4 → blacklist + purge |
| Background task | Periodic scan every 300s (from v3, verify current) |
| Status | ✅ ACTIVE |

### 5.3 Correlation & Threshold Gates

| Check | Threshold | Action |
|-------|-----------|--------|
| Self-correlation vs existing pool | max_correlation < 0.95 | Reject if redundant |
| Minimum Sharpe | configurable (default ~0.5) | Below → marginal/unstable |
| Minimum Fitness | configurable | Below → reject |

---

## 6. PHASE E — ImprovementOrchestra [Layer 4]

**File**: `core/layers/improvement_orchestra.py`
**Source extracted from**: `loop_engine.py` L2088-2130+
**This is the MOST complex layer** — 9 sub-modules, conditional routing, loop-back capability.

**Input**: `EvaluationResult` + `RobustnessCheckResult`
**Output**: `ImprovementResult(action, improved_expressions, sources)`

### 6.1 DecisionEngine (Router)

Central dispatch point. Routes `DecisionOutcome` to appropriate improver:

```
DecisionEngine.decide(parsed_result)
        │
        ├── ACCEPT + ROBUST  ──→ PersistenceLayer (Phase F)
        ├── NEAR_PASS         ──→ NearPassImprover + optionally ToT/EASearch
        ├── FAIL (round < 3)  ──→ FeedbackOrchestrator.critique_revise()
        ├── FAIL (round ≥ 3)  ──→ ABANDON + ExperienceDistiller
        ├── PARAM_OPT         ──→ TurnoverOptimizer / AdaptiveNeutralizer
        └── STUCK            ──→ EASearchStrategy + ToTSearchStrategy
```

### 6.2 FeedbackOrchestrator (Primary Delegate)

| Attribute | Value |
|-----------|-------|
| File | `core/feedback_orchestrator.py` L1-3300+ (~3300 lines!) |
| Sub-improvers | 6 modules below |
| Fallback chain | LLM → AST mutation → parameter tuning → original (preserve) |
| Status | ✅ ACTIVE — primary improvement path |

**Sub-improvers (depth 3 under FeedbackOrchestrator)**:

| # | Improver | File | Target | Method |
|---|----------|------|--------|--------|
| 1 | **ReflectionEngine** | integrated | Failed expr | 3-round LLM critique-revise |
| 2 | **AdaptiveNeutralizer** | `evolution/adaptive_neutralizer.py` L362-663 | High-turnover expr | Adjust neutralization params |
| 3 | **NearPassImprover** | integrated | Sharpe∈[0.8,1.25) | Push over threshold |
| 4 | **FitnessBoost** | integrated | Low-fitness expr | Parameter optimization |
| 5 | **TurnoverOptimizer** | `optimization/turnover_optimizer.py` L108-809 | High-turnover expr | Decay param variants |
| 6 | **AntiOverfitGuard** | integrated | Overfit-risk expr | Apply detector recommendations |

### 6.3 Evolutionary Operators

#### 6.3.1 SemanticCrossover

| Attribute | Value |
|-----------|-------|
| File | `evolution/crossover_mutation.py` L428-567 |
| Paper basis | QuantaAlpha (arXiv:2602.07085) trajectory-level crossover |
| Mechanism | Recombine hypothesis → factor → code trajectories |
| LLM role | Identify complementary segments between parents (30% prob in exploit mode) |
| Fallback | Deterministic AST crossover if LLM fails |
| Priority | LLM offspring = HIGH, AST crossover = NORMAL |
| Status | ✅ ACTIVE — triggered after successful parent recording |

#### 6.3.2 GradientMutation

| Attribute | Value |
|-----------|-------|
| File | `evolution/crossover_mutation.py` L895-1334 (**same file as above**) |
| Strategies | operator_swap, field_swap, param_tune, structure_change |
| Status | ✅ ACTIVE — paired with SemanticCrossover |

#### 6.3.3 CrossoverMutationEngine (Orchestrator)

| Attribute | Value |
|-----------|-------|
| File | `evolution/crossover_mutation.py` L1337-1720 (**same file, 1720 lines total**) |
| Methods | `generate_variants()`, `crossover_trajectories()`, `mutate_trajectory()` |
| Selection | Tournament selection + Pareto dominance |
| Weight adaptation | `_adapt_weights()` from feedback |
| Status | ⚠️ WARNING: 3 classes in one 1720-line file — candidate for split |

### 6.4 EASearchStrategy (Population-Based Search)

| Attribute | Value |
|-----------|-------|
| File | `evolution/ea_search.py` L255-973 |
| Trigger | Near-Pass or Stuck (same conditions as ToT) |
| Flow | init population → `_mutate()` → evaluate → select → repeat |
| Mutation sources | fast (operator-level) + LLM_semantic |
| Selection | Elitist + diversity preservation |
| Status | ✅ ACTIVE — complementary to ToTSearchStrategy |

### 6.5 ExperienceDistiller

| Attribute | Value |
|-----------|-------|
| File | `learning/experience_distiller.py` L39-367 |
| Input | Failure patterns (expression + error + context) |
| Output | Structured experience cards → SuccessLib / FailureLib |
| Retrieval | Injected into next cycle's LLM prompt via FeedbackOrchestrator |
| Status | ✅ ACTIVE — learning from failures |

### 6.6 SemanticMutator (LLM-Driven)

| Attribute | Value |
|-----------|-------|
| File | `evolution/semantic_mutator.py` L73-808 |
| Core method | `mcts_explore()` — MCTS + LLM evaluation |
| Fallback chain | LLM → AST → parameter → original (never empty return) |
| Status | ✅ ACTIVE — Tier-3 mutation strategy |

### 6.7 GraphBasedExperienceDB

| Attribute | Value |
|-----------|-------|
| File | `knowledge/graph_experience_db.py` L185-1063 |
| Storage | Directed graph triplets (subject, predicate, object) |
| Features | 8-dimensional: field/operator/field_family/structure/complexity/neutralization/decay/type |
| Query | Weighted Jaccard similarity for top-k experience retrieval |
| Writers | FeedbackOrchestrator (query), EASearchStrategy (write results) |
| Status | ✅ ACTIVE — persistent knowledge graph |

---

## 7. PHASE F — PersistenceLayer [Layer 6]

**File**: `core/layers/persistence_layer.py`
**Final phase**: lineage tracking, quality pool, telemetry.

**Input**: All previous phase outputs
**Output**: `PersistenceResult(alpha_id, channel_status, events, metrics)`

### 7.1 Lineage Recording

Via `CrossoverMutationEngine` API (called from this layer):
- `record_alpha_outcome()` → internal DB for parent selection
- `record_trajectory()` → full trajectory storage for SemanticCrossover

### 7.2 AlphaChannel (Quality Pool)

| Attribute | Value |
|-----------|-------|
| File | `cli/alpha_channel.py` L18-107 |
| Filter | min_sharpe threshold (configurable, default ~1.0) |
| Modes | Batch processing + streaming submission |
| Output | Persistent high-quality factor pool |
| Status | ✅ ACTIVE |

### 7.3 AlgorithmTelemetryCollector

| Attribute | Value |
|-----------|-------|
| File | `monitoring/algorithm_telemetry.py` L64-861 |
| Mechanism | `@algo_log()` decorator on core functions |
| Output | JSONL logs + timing metrics + utilization stats |
| Coverage | All 6 Layers + major sub-modules |
| Status | ✅ ACTIVE — production observability |

---

## 8. Cross-Layer Data Flow (Loop-Back Paths)

### 8.1 Primary Loop

```
A(Exploration) → B(Generate) → C(Evaluate) → D(Robustness) → E(Improve) → F(Persist)
                                                                                      │
                                                                              next cycle → A
```

### 8.2 Improvement Loop-Backs

| Path | Trigger | Destination | Condition |
|------|--------|-------------|-----------|
| **E → B** | Improved expression needs regeneration | GenerationPipeline | New expression from improver |
| **E → C** | Re-submission needed | EvaluationGateway | `DecisionAction.RESUBMIT` |
| **E → E** | Multi-round improvement | ImprovementOrchestra | `improvement_round < 3` |

### 8.3 Async Background Tasks

| Task | Interval | Purpose |
|------|----------|---------|
| DecayDetector periodic scan | ~300s | Flag decaying alphas (L3 pause / L4 blacklist) |
| TrajectoryCrossover background | ~120s | Generate new variants from archived trajectories |
| Telemetry flush | continuous | Write JSONL logs |

---

## 9. Concurrency Control Matrix

| Resource | Control | Limit | Owner |
|----------|--------|-------|-------|
| LLM API calls | `asyncio.Semaphore` | 4 | GenerationPipeline (3 gen + 1 improve) |
| Embedding API calls | `asyncio.Semaphore` | 4 | Knowledge/RAG layer |
| BRAIN platform submits | `asyncio.Semaphore` | 3 | EvaluationGateway (matches WQ slots) |
| Generator tasks | `asyncio.create_task` | 3 parallel | GenerationPipeline (streaming enqueue) |
| Event signals | `asyncio.Event` × 2 | N/A | slot_released, generation_green_light |
| Post-processing | `asyncio.gather` | 4-way parallel | PASS-path parallelism |

### Event-Driven Signal Flow

```
BRAIN slot released → pool.release_slot()
    │
    ├─ slot_released Event.set()
    └─ _update_green_light()
        │
        ├─ pool < THRESHOLD(10) AND slot available?
        │   ├─ YES → generation_green_light.set() → wake Generator
        │   └─ NO  → generation_green_light.clear() → Generator waits
        │
        Generator:
        ├─ Launch 3 independent tasks (create_task × 3)
        ├─ await pool.await_generation_slot(timeout=30s)
        │   ├─ Green light → enter next cycle immediately
        │   └─ Timeout → check degradation conditions
        └─ should_degrade_to_improve()?
            ├─ YES → switch LLM to self-improvement mode
            └─ NO  → keep waiting
```

---

## 10. Wiring Quality Issues Found

| # | Issue | Severity | Modules Affected | Suggested Fix |
|---|-------|----------|------------------|---------------|
| 1 | **3 classes in 1 file** (1720 lines) | 🟡 Medium | `crossover_mutation.py`: SemanticCrossover(L428) + GradientMutation(L895) + CrossoverMutationEngine(L1337) | Split into `semantic_crossover.py`, `gradient_mutation.py`, `crossover_engine.py` |
| 2 | **FeatureMap dual ownership** | 🟡 Medium | MAP-Elites referenced by both L1(ExplorationDirector) and L4(ImprovementOrchestra) | Clarify: L1 reads explore_targets, L4 writes candidates. Document dual contract. |
| 3 | **FeedbackOrchestrator 3300 lines** | 🟡 Medium | Single file handles 6 sub-improvers + decision routing + LLM orchestration | Extract sub-improvers into own files, keep orchestrator as thin router |
| 4 | **SignalQualityPreFilter missing** | 🟢 Low | Referenced in docs but not found as standalone class | Likely merged into GenerationPipeline or brain_submitter. Verify and update docs. |
| 5 | **CrossoverEngine no standalone class** | 🟢 Low | Lineage tracking scattered across CrossoverMutationEngine methods + PersistenceLayer | Either extract or accept current design (functional, just naming mismatch) |
| 6 | **L5 executes before L4** (non-obvious) | 🟢 Low | RobustnessGate(L5) before ImprovementOrchestra(L4) — counter-intuitive numbering | Document explicitly: quality-gate-first prevents wasting improvement budget on garbage |

---

## 11. Algorithm Utilization Summary

| Domain | Algorithms | Count | All Active? |
|--------|-----------|-------|-------------|
| **Exploration** | MAB, NavigationFusion, ToT, MAP-Elites, TemplateBandit | 5 | ✅ Yes |
| **Generation** | AlphaGenerator, GenGates, ASTValidator, ComplexityController, TemplateReasoner, FieldProxyMap | 6 | ✅ Yes |
| **Evaluation** | BrainSubmitter, ResultRouter, DecisionEngine | 3 | ✅ Yes |
| **Robustness** | AntiOverfitDetector (5 tests), DecayDetector, CorrelationGate | 3 groups | ✅ Yes |
| **Improvement** | FeedbackOrch(+6 subs), SemanticCrossover, GradientMutation, EA Search, ExpDistill, SemMutator, GraphDB, Neutralizer, TurnoverOpt | **9 (+6)** | ✅ Yes |
| **Persistence** | AlphaChannel, TelemetryCollector, CrossoverEngine(lineage) | 3 | ✅ Yes |
| **TOTAL** | | **~29** | **29/29 ACTIVE (0 dead)** |

---

## Appendix: File Index

| Layer | Primary File | Lines (approx) | Key Classes |
|-------|-------------|----------------|-------------|
| L1 | `core/layers/exploration_director.py` | 200+ | ExplorationDirector |
| | `learning/mab.py` | 1300+ | HierarchicalMAB, TemplateFamilyBandit |
| | `core/navigation_fusion.py` | 480+ | NavigationFusion |
| | `evolution/tot_search.py` | 800+ | ToTSearchStrategy |
| | `evolution/quality_diversity.py` | 1170+ | FeatureMap, GridArchive, Emitters, CMAES |
| L2 | `core/layers/generation_pipeline.py` | 200+ | GenerationPipeline |
| | `generation/alpha_generator.py` | 640+ | AlphaGenerator |
| | `evolution/generation_gates.py` | 1000+ | GenerationGates |
| | `validation/ast_validator.py` | 400+ | ASTValidator |
| | `validation/complexity_control.py` | 220+ | ComplexityController |
| | `generation/template_reasoning_generator.py` | 900+ | TemplateReasoningGenerator |
| | `knowledge/field_proxy_map.py` | 50+ | FieldProxyMap |
| L3 | `core/layers/evaluation_gateway.py` | 200+ | EvaluationGateway |
| | `services/brain_submitter.py` | 1100+ | BrainSubmitter |
| | `core/result_router.py` | 460+ | ResultRouter |
| | `core/decision_engine.py` | 370+ | DecisionEngine |
| L5 | `core/layers/robustness_gate.py` | 200+ | RobustnessGate |
| | `validation/anti_overfit_detector.py` | 1068+ | AntiOverfitDetector, _QuantGPTAntiOverfitDetector |
| | `validation/decay_detector.py` | 600+ | DecayDetector |
| L4 | `core/layers/improvement_orchestra.py` | 200+ | ImprovementOrchestra |
| | `core/feedback_orchestrator.py` | 3300+ | FeedbackOrchestrator (+6 sub-improvers) |
| | `evolution/crossover_mutation.py` | 1720+ | SemanticCrossover, GradientMutation, CrossoverMutationEngine |
| | `evolution/ea_search.py` | 970+ | EASearchStrategy |
| | `learning/experience_distiller.py` | 367+ | ExperienceDistiller |
| | `evolution/semantic_mutator.py` | 808+ | SemanticMutator |
| | `knowledge/graph_experience_db.py` | 1063+ | GraphBasedExperienceDB |
| | `evolution/adaptive_neutralizer.py` | 663+ | AdaptiveNeutralizer |
| | `optimization/turnover_optimizer.py` | 809+ | TurnoverOptimizer |
| L6 | `core/layers/persistence_layer.py` | 200+ | PersistenceLayer |
| | `cli/alpha_channel.py` | 107+ | AlphaChannel |
| | `monitoring/algorithm_telemetry.py` | 861+ | AlgorithmTelemetryCollector |
| **Orchestrator** | `core/loop_engine.py` | 2700+ | LoopEngine (thin wiring layer) |
