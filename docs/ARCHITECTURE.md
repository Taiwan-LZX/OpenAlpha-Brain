# OpenAlpha-Brain Architecture: Self-Looping Evolutionary System

> Version: 2.0 (from Pipeline to Living System)
> Date: 2026-06-01
> Status: Design Document (pending implementation)

---

## 1. Core Paradigm Shift

### What We Are NOT Building

```
❌ Linear Pipeline:
   [Explore] → [Generate] → [Validate] → [Submit] → [Improve] → (loop)
   
   Problems:
   - Fixed sequence — cannot adapt flow based on real-time findings
   - Stage boundaries — knowledge lost between stages
   - Human-in-the-loop — requires external orchestration
   - No emergence — output is deterministic function of input
```

### What We ARE Building

```
✅ Self-Looping Evolutionary Ecosystem:

   ┌─────────────────────────────────────────────────────┐
   │                  Knowledge Graph                      │
   │            (shared memory, all algorithms)            │
   │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐       │
   │  │Factor│ │Strategy│ │Pattern│ │Failure│ │Success│       │
   │  │ Space │ │ Library│ │  DB   │ │ Modes │ │ Cases │       │
   │  └──┬───┘ └───┬──┘ └───┬──┘ └───┬──┘ └───┬──┘       │
   └──────┼─────────┼─────────┼─────────┼─────────┼─────────┘
          │         │         │         │         │
   ═══════╧═════════╧═════════╧═════════╧═════════╧══════════
          │    Signal Bus (any-to-any communication)     │
          │                                             │
   ┌──────┴─────┐  ┌──────┴─────┐  ┌──────┴─────┐
   │  Algorithm │  │  Algorithm │  │  Algorithm │  ... (32 nodes)
   │    Node A  │←→│    Node B  │←→│    Node C  │
   │            │  │            │  │            │
   │ reads/writes│  │reads/writes│  │reads/writes│
   │  Knowledge │  │  Knowledge │  │  Knowledge │
   └────────────┘  └────────────┘  └────────────┘
          │               │               │
          └───────────────┼───────────────┘
                          ▼
              ┌───────────────────────┐
              │  Self-Awareness Layer  │
              │  (system knows itself) │
              │  · Where am I weak?   │
              │  · What should I try? │
              │  · Am I improving?    │
              └───────────────────────┘
                          │
                          ▼
                   (perpetual cycle)
```

### Key Principles

| # | Principle | Description |
|---|-----------|-------------|
| P1 | **No Fixed Sequence** | Any algorithm can trigger any other. Flow emerges from signals, not hardcoded order. |
| P2 | **Shared Knowledge** | All algorithms read/write to one Knowledge Graph. Discoveries are immediately available system-wide. |
| P3 | **Signal-Based Communication** | Algorithms emit typed signals (`WeaknessDetected`, `OpportunityFound`, `StagnationAlert`). Others subscribe and react. |
| P4 | **Self-Awareness** | The system monitors its own performance metrics (Sharpe trajectory, diversity, coverage). Auto-redirects effort to weak areas. |
| P5 | **Emergent Strategy** | New approaches emerge from algorithm interactions. Not pre-programmed — discovered through exploration. |
| P6 | **7×24 Autonomous** | No human needed for operation. System runs perpetually: explore → learn → improve → repeat. |
| P7 | **Knowledge Compounds** | Each cycle produces new knowledge (what works, what doesn't, patterns). Knowledge is never lost, only refined. |

---

## 2. Architecture Components

### 2.1 Knowledge Graph (The Central Nervous System)

```python
class KnowledgeGraph:
    """All algorithms share this. Single source of truth."""
    
    # Factor Space — what we've discovered
    factor_space: FactorDatabase        # All generated alphas + their metrics
    
    # Strategy Library — what works in which regime
    strategy_library: StrategyDatabase   # "When Sharpe < 0.5, use X; when turnover > Y, use Z"
    
    # Pattern Database — emergent patterns
    pattern_db: PatternDatabase          # "ts_decay(10) + group_neutralize → stable but low return"
    
    # Failure Modes — what doesn't work
    failure_modes: FailureDatabase       # "operator X with field Y always fails validation"
    
    # Success Cases — what works exceptionally well  
    success_cases: SuccessDatabase       # "this expression pattern achieved Sharpe > 2.0"
    
    # Evolution State — where are we now?
    state: EvolutionState                # Current cycle number, best Sharpe, diversity index
    
    def emit_signal(self, signal: Signal) -> None:
        """Broadcast signal to all subscribers."""
        
    def query(self, query: KnowledgeQuery) -> KnowledgeResult:
        """Any algorithm can ask questions."""
        
    def record(self, record: KnowledgeRecord) -> None:
        """Any algorithm can contribute knowledge."""
```

**Signal Types (the language of inter-algorithm communication):**

| Signal | Emitted By | Consumed By | Meaning |
|--------|-----------|-------------|---------|
| `FactorGenerated` | Generator | Validator, Scorer | New alpha ready |
| `ValidationFailed` | Validator | Repairer, Generator | Alpha has issues |
| `SharpeLow` | Scorer | Improver, Explorer | Need better alphas |
| `TurnoverHigh` | Scorer | TurnoverOptimizer | Trading cost too high |
| `StagnationDetected` | Telemetry | Explorer, StrategySwitcher | Not improving |
| `NewPatternFound` | PatternMiner | All | Discovered useful pattern |
| `RegimeChange` | MarketState | All | Market conditions shifted |
| `KnowledgeMatured` | ExperienceDistiller | All | New reusable knowledge |
| `ResourceAvailable` | SlotManager | Submitter | Can submit to BRAIN |
| `WeaknessIdentified` | SelfAwareness | Relevant algorithm | System knows where it's weak |

### 2.2 Algorithm Node (The Fundamental Unit)

Every algorithm in the system is an **AlgorithmNode** — not a function, not a class method, but an **autonomous agent** with:

```python
class AlgorithmNode(ABC):
    """An autonomous unit in the evolutionary ecosystem."""
    
    node_id: str                           # Unique: "ea_search", "semantic_mut", ...
    category: NodeCategory                 # GENERATOR / VALIDATOR / IMPROVER / EXPLORER / LEARNER / SCORER
    
    # === Input/Output (data contract) ===
    input_signals: list[SignalType]        # What signals trigger this node
    output_signals: list[SignalType]       # What signals this node emits
    
    # === Self-regulation ===
    activation_condition: Callable[[KnowledgeGraph], bool]  # When should I run?
    resource_cost: float                   # How expensive am I? (for scheduling)
    cooldown_ticks: int                    # Min ticks between runs (prevent spam)
    last_run_tick: int = 0                 # When did I last run?
    
    # === Knowledge interaction ===
    knowledge_reads: list[str]             # What knowledge does this node need?
    knowledge_writes: list[str]           # What knowledge does this node produce?
    
    @abstractmethod
    async def activate(self, ctx: RunContext) -> list[Signal]:
        """
        Main execution. 
        Reads from KnowledgeGraph, does work, emits Signals.
        Returns signals that will be broadcast.
        """
        ...
    
    @abstractmethod
    def should_activate(self, kg: KnowledgeGraph) -> bool:
        """
        Self-decision: should I run this tick?
        Based on current system state, not external command.
        """
        ...
```

**Critical Design Decision**: Each node decides **for itself** whether to run. There is no central orchestrator saying "now run EA Search". Instead:
- EA Search checks KnowledgeGraph: "Are there enough parent candidates? Is diversity low?"
- If yes → activates → produces new factors → broadcasts `FactorGenerated` signal
- If no → stays idle

This is the opposite of a pipeline. It's an **economy of autonomous agents**.

### 2.3 The Loop (Tick-Based, Not Stage-Based)

```python
class EvolutionLoop:
    """
    Not a pipeline. A tick-based simulation.
    Each tick = one round of signal processing.
    Runs forever (7×24).
    """
    
    tick: int = 0                         # Monotonically increasing
    knowledge: KnowledgeGraph             # Shared state
    nodes: dict[str, AlgorithmNode]      # All registered algorithms
    signal_bus: SignalBus                 # Inter-node communication
    
    async def run_forever(self) -> None:
        while True:                       # ← 7×24, no exit condition
            self.tick += 1
            
            # Phase 1: Each node self-decides whether to activate
            activations = [
                node.activate(ctx) 
                for node in self.nodes.values() 
                if node.should_activate(self.knowledge)
            ]
            
            # Phase 2: Collect all emitted signals
            all_signals = chain.from_iterable(activations)
            
            # Phase 3: Broadcast signals → update KnowledgeGraph
            for signal in all_signals:
                self.knowledge.emit_signal(signal)
                self.signal_bus.broadcast(signal)
            
            # Phase 4: Self-awareness check
            # (Is the system stuck? Should it redirect?)
            await self.self_assess()
            
            # Phase 5: Persistence (save state for recovery)
            if self.tick % PERSIST_INTERVAL == 0:
                await self.knowledge.persist()
            
            # Phase 6: Adaptive sleep (faster when active, slower when stagnant)
            await self.adaptive_sleep()
```

### 2.4 Self-Awareness Layer (The Meta-Cognition)

```python
class SelfAwarenessLayer:
    """
    The system's ability to know about itself.
    Runs every N ticks to assess overall health and redirect effort.
    """
    
    async def assess(self, kg: KnowledgeGraph) -> SystemDiagnosis:
        """
        Analyze the entire system state and produce actionable insights.
        This is the "conscience" of the ecosystem.
        """
        
        # 1. Performance Trajectory Analysis
        sharpe_history = kg.factor_space.best_sharpe_history(last_n=100)
        trajectory = analyze_trend(sharpe_history)
        
        # 2. Coverage Analysis (are we exploring all directions?)
        coverage = kg.factor_space.coverage_by_field_family()
        under_explored = [f for f, c in coverage.items() if c < THRESHOLD]
        
        # 3. Diversity Health
        diversity = kg.factor_space.diversity_index()
        
        # 4. Algorithm Utilization (which nodes are contributing?)
        utilization = {nid: node.contribution_score for nid, node in ...}
        dormant = [nid for nid, score in utilization.items() if score == 0]
        
        # 5. Knowledge Growth Rate (are we learning?)
        knowledge_growth = kg.knowledge_growth_rate(last_n=50)
        
        # 6. Produce Diagnosis → Emit Signals
        diagnosis = SystemDiagnosis(
            trajectory=trajectory,
            weak_areas=under_explored,
            diversity=diversity,
            dormant_algorithms=dormant,
            learning_rate=knowledge_growth,
        )
        
        # Convert diagnosis into actionable signals
        if trajectory == Trend.STAGNANT:
            kg.emit_signal(StagnationDetected(suggestion="try_new_direction"))
        if len(under_explored) > 0:
            kg.emit_signal(W weaknessIdentified(areas=under_explored))
        if diversity < DIVERSITY_FLOOR:
            kg.emit_signal(DiversityAlert(current=diversity))
        
        return diagnosis
```

---

## 3. How Algorithms Naturally Connect (Real Call Graph)

Based on code audit of 32 modules. These connections emerge from **signal subscriptions**, not hardcoded calls:

### Connection Map (Signal Flow)

```
                    ┌────────────────────────────────────┐
                    │        Knowledge Graph              │
                    │  (all read/write here)              │
                    └──────────────┬─────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
           ▼                       ▼                       ▼
    ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
    │  GENERATORS  │       │  VALIDATORS  │       │  IMPROVERS   │
    │ (produce αs) │       │ (check αs)   │       │ (fix αs)     │
    ├──────────────┤       ├──────────────┤       ├──────────────┤
    │AlphaGenerator│──────→│AST_Validator │       │SegmentLocked  │
    │EA_Search     │       │WQ_Validator  │──────→│DecayTweaker  │
    │SemanticMut   │       │OverfitDetect│       │StrategySwitch │
    │NearPassImpr  │       │DecayDetector │       │Re-explore    │
    │TOT_Search    │       │ComplexityCtrl│       │              │
    │RAG_Generator │       │StabilityGuard│       │              │
    └──────┬───────┘       │SignalArbiter│       └──────┬───────┘
           │               │OfficialScorer│              │
           │               └──────┬───────┘              │
           │                      │                       │
           │               ┌──────┴───────┐              │
           │               │   SCORERS    │              │
           │               │ (rate αs)    │              │
           │               ├──────────────┤              │
           │               │BRAIN_Submit  │              │
           │               │OfflineScorer │              │
           │               │TurnoverOpt   │──────────────┘
           │               └──────┬───────┘
           │                      │
           │               ┌──────┴───────┐
           │               │   LEARNERS   │
           │               │ (extract ∴)  │
           │               ├──────────────┤
           │               │MAB           │
           │               │ReflectEngine │
           │               │ExpDistiller  │
           │               │RewardUpdater │
           │               └──────┬───────┘
           │                      │
           │               ┌──────┴───────┐
           │               │  EXPLORERS   │
           │               │ (where to go?)│
           │               ├──────────────┤
           │               │ExplorationDir│
           │               │NavFusion     │
           │               │StrategyClass │
           │               └──────────────┘
           │
           └──────────────→ (signals flow back via KnowledgeGraph)
```

### Example: One Complete Cycle (What Actually Happens)

```
Tick #18427
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[SelfAwareness] → emitting WeaknessIdentified(areas=["volume", "momentum"])
                              ↑
                              │ system detected low coverage in these field families

[MAB] → should_activate? YES (new weakness signal received)
       → select arm: "momentum_exploration" (highest UCB for this weakness)
       → emit: DirectionSelected(direction="momentum", confidence=0.82)

[AlphaGenerator] → should_activate? YES (DirectionSelected signal)
                 → prompt LLM with momentum-focused template + RAG context
                 → generate 12 candidate expressions
                 → emit: FactorsGenerated(count=12, source="llm_momentum")

[AST_Validator] → should_activate? YES (FactorsGenerated signal)
               → validate 12 expressions → 8 pass, 4 fail
               → emit: ValidationResults(passed=8, failed=4)
               → for each failure: emit ValidationFailed(factor=...)

[AST_Repair] → should_activate? YES (ValidationFailed × 4)
             → repair 4 failed expressions → 3 recovered, 1 abandoned
             → emit: RepairedFactors(count=3)

[WQ_Validator] → should_activate? YES (new valid ASTs)
              → check WQ format compliance → 10 pass, 1 fail
              → emit: WQValidated(count=10)

[Overfit_Detector] → should_activate? YES (WQValidated signal)
                  → analyze 10 factors → 2 show overfit signals
                  → emit: OverfitRisk(factors=[...], severity="medium")

[Anti_Overfit] → should_activate? YES (OverfitRisk signal)
              → apply regularization to 2 factors
              → emit: TreatedFactors(count=2, treatment="regularization")

[BRAIN_Submitter] → should_activate? YES (SlotAvailable signal)
                → submit 10 factors → 4 accepted by platform
                → receive results: Sharpe=[0.3, 0.7, 1.2, 0.9]
                → emit: SubmissionResults(sharpes=[...])

[Turnover_Optimizer] → should_activate? YES (SubmissionResults)
                  → factor with Sharpe=1.2 has turnover=0.45 (too high)
                  → adjust decay window: 10 → 18
                  → emit: OptimizedFactors(count=1)

[ExperienceDistiller] → should_activate? YES (cycle end pattern)
                     → analyze this cycle: momentum factors avg Sharpe=0.78
                     → record pattern: "ts_delta(volume, 5) + group_neutralize → good for trending"
                     → emit: KnowledgeMatured(pattern="momentum_ts_delta_v5")

[RewardUpdater] → should_activate? YES (KnowledgeMatured)
               → MAB update: "momentum_exploration" arm reward += 0.15
               → next cycle will favor this direction more

[PersistenceLayer] → save all new knowledge to disk
                  → checkpoint state at tick #18427

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Next tick: #18428 → loop continues forever...
```

---

## 4. Non-Linear Behaviors (How It Differs From Pipeline)

### Scenario A: Stagnation Recovery (Self-Directed)

```
Pipeline way:    [Explore] → [Gen] → [Val] → [Submit] → (same thing again)
                 ↓ stagnates, keeps doing same thing

Our way:        
  Tick N:    Normal operation...
  Tick N+50:  [SelfAwareness] detects Sharpe flat for 50 ticks
             → emit StagnationDetected(severity="high")
             
  Tick N+51:  [StrategyClassifier] receives signal
             → classifies problem: "exploitation too heavy, exploration too light"
             → emit Suggestion(pivot="exploration_weight += 0.3")
             
  Tick N+52:  [MAB] receives suggestion
             → temporarily boosts exploration arm UCB bonus
             
  Tick N+53:  [ExplorationDirector] gets higher UCB for random/exploratory arms
             → selects unexplored direction: "cross_sectional"
             
  Tick N+54-100: System naturally explores new area WITHOUT any human intervention
             → discovers new factor family with Sharpe > 1.5
             → [ExperienceDistiller] records this as a success pattern
             → Future cycles will auto-favor similar explorations
```

### Scenario B: Emergent Strategy Discovery

```
Tick M:    [SemanticMutator] tries combining ts_rank with ts_decay
           → produces factor that passes validation but gets low Sharpe (0.3)
           
Tick M+1:  [ReflectionEngine] analyzes why it failed
           → hypothesis: "ts_rank needs longer lookback when combined with decay"
           → records hypothesis in KnowledgeGraph
           
Tick M+200: [AlphaGenerator] picks up this hypothesis via RAG
           → generates variant: ts_rank(volume, 20) + ts_decay(close, 15)
           → Sharpe = 1.8 🎉
           
Tick M+201:  [ExperienceDistiller] extracts pattern:
           "ts_rank(X, ≥15) + ts_decay(Y, ≥10) → high Sharpe for trending"
           
Result: A NEW STRATEGY emerged from the interaction of 3 algorithms,
        none of which were programmed to discover this together.
        This is EMERGENT BEHAVIOR.
```

### Scenario C: Regime Adaptation

```
Tick K:    [MarketState] detects regime change (volatility spike)
           → emit RegimeChange(old="low_vol", new="high_vol")
           
Tick K+1:  Multiple algorithms receive signal simultaneously:
           - [AdaptiveNeutralizer] switches to stronger neutralization
           - [DecayDetector] shortens decay windows (faster adaptation)
           - [TurnoverOptimizer] reduces target turnover (higher cost in high vol)
           - [GenerationPipeline] favors simpler expressions (more robust)
           
Result: System-wide autonomous adaptation in ONE tick.
        No central controller needed. Each algorithm knows how to react.
```

---

## 5. Migration Path (From Current Codebase)

### Phase 0: Infrastructure (New Files)

```
src/openalpha_brain/core/nodes/
├── __init__.py           # Re-exports
├── protocol.py           # AlgorithmNode ABC, Signal types, RunContext
├── knowledge_graph.py    # KnowledgeGraph implementation
├── signal_bus.py        # Pub/sub signal routing
├── registry.py          # Node registration & discovery
├── context.py           # RunContext data object
├── self_awareness.py    # System meta-cognition
└── loop_engine.py       # New tick-based loop (replaces old one)
```

### Phase 1: Wrap Existing Modules (Minimal Change)

Each existing module gets wrapped as an AlgorithmNode without changing internal logic:

```python
# Before: standalone function/class
class EASearch:
    def search(self, population, fitness_fn) -> list[Alpha]: ...

# After: wrapped as autonomous node
@node_executor(
    id="ea_search",
    category=NodeCategory.GENERATOR,
    subscribes_to=[DirectionSelected, StagnationDetected],
    emits=[FactorsGenerated],
    reads=["factor_space.population"],
    writes=["factor_space.candidates"],
)
class EANode(AlgorithmNode):
    def __init__(self):
        self._engine = EASearch()  # reuse existing logic!
    
    async def activate(self, ctx: RunContext) -> list[Signal]:
        result = self._engine.search(ctx.kg.factor_space.population, ...)
        ctx.kg.factor_space.add_candidates(result)
        return [FactorsGenerated(source="ea_search", count=len(result))]
```

**Key**: Internal logic 100% preserved. Only add the "when/why to run" wrapper.

### Phase 2: Extract Signal Logic

Move implicit if/else decisions into explicit signal emissions:

```python
# Before (implicit decision hidden inside function):
def generate_alphas(self, direction):
    if direction == "mab":
        result = self.mab_generate()
    elif direction == "nav":
        result = self.nav_generate()
    ...

# After (decision becomes signal-driven):
class GenerationPipelineNode(AlgorithmNode):
    async def activate(self, ctx):
        direction = ctx.kg.query(CurrentDirection())
        
        # Emit request, let specialized nodes respond
        ctx.emit(GenerationRequest(direction=direction))
        # Don't generate here! Let MAB_Node / NavFusion_Node / etc. respond
        
        # This node becomes a pure coordinator, not a monolith
```

### Phase 3: Enable Self-Awareness

Wire up the meta-cognitive layer:

```python
# In the main loop:
if tick % SELF_AWARENESS_INTERVAL == 0:
    diagnosis = await self_awareness.assess(knowledge_graph)
    # Diagnosis automatically emits signals
    # Nodes react on next tick
```

### Phase 4: Remove Old Orchestration

Delete/simplify:
- Old `loop_engine.py` (500+ lines) → replace with ~100 line tick loop
- Old `layers/*.py` → nodes manage themselves
- Hardcoded sequences → signal subscriptions

---

## 6. Constraints & Guarantees

### Must Preserve (Domain Rules)

| Rule | How It's Enforced |
|------|------------------|
| ThreeBlockTemplate | Block B/C enforced in Validator nodes, not in generator |
| FieldProxyMap | Generator nodes query KG for allowed fields |
| Semaphore(3) | BRAIN_Submitter node manages its own semaphore |
| GlobalRateLimiter | Submitter node includes rate limiting logic |
| Anti-overfit | Detector + Treatment nodes form automatic chain |

### Must Achieve (Operational Goals)

| Goal | Metric | How Measured |
|------|--------|-------------|
| 7×24 uptime | Mean time between failures > 72h | Telemetry |
| Continuous improvement | Best Sharpe monotonic increase (with noise) | KnowledgeGraph trend |
| Full algorithm utilization | No dormant nodes for > 100 ticks | SelfAwareness report |
| Knowledge compounding | KnowledgeGraph size grows, quality improves | Storage + scoring |
| Autonomous recovery | Recovers from stagnation without human intervention | Log analysis |

---

## 7. File Structure (Target State)

```
src/openalpha_brain/
├── core/
│   ├── nodes/                    # ★ NEW: Autonomous node infrastructure
│   │   ├── protocol.py           # ABC, Signal types, Categories
│   │   ├── knowledge_graph.py    # Shared memory
│   │   ├── signal_bus.py         # Pub/sub
│   │   ├── registry.py           # Node discovery
│   │   ├── context.py            # RunContext
│   │   ├── self_awareness.py     # Meta-cognition
│   │   └── loop_engine.py        # Tick-based eternal loop
│   │
│   ├── layers/                   # DEPRECATED: Will be removed in Phase 4
│   ├── loop_engine.py            # DEPRECATED: Replaced by nodes/loop_engine.py
│   ├── feedback_orchestrator.py  # DEPRECATED: Split into multiple nodes
│   │
│   ├── evolution/                # Each file becomes a Node
│   │   ├── ea_search.py          # → EANode
│   │   ├── semantic_mutator.py   # → SemanticMutatorNode
│   │   ├── crossover_mutation.py # → CrossoverNode
│   │   └── ... (each → Node)
│   │
│   ├── validation/               # Each validator → Node
│   ├── learning/                 # Each learner → Node
│   ├── generation/               # Each generator → Node
│   ├── optimization/             # Each optimizer → Node
│   ├── services/                # Each service → Node
│   │
│   └── ... (rest unchanged)
```

---

## 8. Open Questions (For Discussion)

1. **Tick duration**: Real-time (each tick = seconds) or simulated (batch mode)? Affects how we handle BRAIN API rate limits.
2. **Concurrency**: Should multiple nodes run in parallel within one tick? Requires asyncio/gather design.
3. **Knowledge persistence**: SQLite (current) vs vector DB for semantic search of patterns?
4. **Human override**: Can human operator inject signals or pause specific nodes?
5. **Multi-instance**: Can we run multiple instances exploring different directions concurrently?

---

*Document End*
