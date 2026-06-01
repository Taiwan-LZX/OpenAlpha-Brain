|     | BrainAlpha: |     |     | An  | Autonomous |     |     | Multi-Agent |
| --- | ----------- | --- | --- | --- | ---------- | --- | --- | ----------- |
System
|     |     |     | for Quantitative |             |            | Alpha       |            | Discovery |
| --- | --- | --- | ---------------- | ----------- | ---------- | ----------- | ---------- | --------- |
|     |     |     |                  |             | Nuthdanai  | Wangpratham |            |           |
|     |     |     |                  | QuantCorner | Laboratory | ·           | February   | 27, 2026  |
|     |     |     | Pioneering       | Innovation  | in         | Automated   | Investment | Research  |
Abstract
Alpha decay is not a risk to be managed—it is an operational rate that must be outrun.
Empirical estimates place the average decay rate of a quantitative signal in U.S. equities at
5.6% per year (McLean and Pontiff, 2016), implying that a portfolio of 1,000 live factors must
replace 56–100 signals annually merely to remain static. No human research organization
can sustain this throughput indefinitely. This paper presents BrainAlpha, an end-to-end
autonomous multi-agent system that replicates the complete alpha discovery workflow of
a professional quantitative researcher: hypothesis ideation, signal specification, expression
generation, simulation-based evaluation, error recovery, and iterative repair—all without
humanintervention. ThesystemcomprisessixspecializedLangGraphmodules(M0–M4,M3.5),
coordinated through a shared Signal Alpha Object (SAO) that carries every candidate from
raw investment thesis to Brain-validated alpha. The architecture introduces four innovations
absentfrompriorsystems: (i)a660-cellstructuredexplorationgridthattransformshypothesis
search into an exhaustible coverage problem; (ii) a RAG-enhanced specification encoder that
eliminates field hallucination at zero simulation cost; (iii) a failure-mode taxonomy (FM-1
through FM-6) that converts binary evaluation feedback into typed diagnostic signals; and
(iv)aconvergence-governedrepairenginewithfiveformalstopconditionsthatpreventswasted
simulation budget on structurally unrecoverable expressions. All evaluation is grounded
| exclusively |     | in  | WorldQuant | Brain’s | production | WebSim | infrastructure. |     |
| ----------- | --- | --- | ---------- | ------- | ---------- | ------ | --------------- | --- |
Keywords: quantitative finance · alpha discovery · large language models · multi-agent
systems · retrieval-augmented generation · WorldQuant Brain · automated research
1 Introduction
| 1.1 | The | Alpha | Throughput |     | Problem |     |     |     |
| --- | --- | ----- | ---------- | --- | ------- | --- | --- | --- |
Consider the following arithmetic. A systematic investment firm maintains a live
portfolio of 1,000 uncorrelated alpha signals. The expected lifespan of each signal, given
an empirically documented decay rate of 5.6% per year (McLean and Pontiff, 2016), is
approximately18years. Thissoundscomfortable—untiloneaccountsforpost-publication
arbitrage, factor crowding, and microstructure adaptation. In practice, signals with
high academic visibility decay at double the baseline rate; those based on widely traded

BrainAlpha — Working Paper February 2026
factors may lose statistical significance within five years. The net result: the firm must
retire and replace 56–100 signals per year simply to hold its portfolio steady.
At a productivity rate of 2–5 viable ideas per researcher-week with a 15–30% quality
screening pass rate (Tang et al., 2025), sustaining a 1,000-signal portfolio requires the
continuous effort of dozens of dedicated quantitative researchers—a talent pool that is
expensive, scarce, and increasingly competitive to recruit.
Central claim. Alphadiscoveryisnotanirreduciblyhumanactivity. Itisastructuredcognitive
task that is decomposable into well-defined sub-tasks, each within the capability of current LLM
technology—and amenable to automation at a scale and cost structure no human organization
can match.
1.2 Large Language Models as Quantitative Researchers
Forthreedecades,automatedalphadiscoveryreliedongeneticprogramming—evolutionary
searchthatscalescomputationallybutlackseconomicintuition. Withoutpriorknowledge
of which regions of the expression space contain meaningful signals, evolutionary methods
spend the majority of their budget on formulae no human researcher would consider.
Large language models change this calculus. Pre-trained on decades of financial literature
and academic asset pricing research, frontier LLMs have internalized the causal structure
of financial markets to a measurable degree. A recent large-scale study evaluating
161,000 stock-day observations across the Russell 1000 found that an agentic LLM—with
no curated data inputs, no fine-tuning, and no look-ahead bias—delivered 0.185%
daily alpha (t = 2.51) after Fama-French six-factor risk adjustment, corresponding to
approximately 46% annualized excess return at a Sharpe ratio of 2.43 (Chen, 2026).
System-levelevidencereinforcesthisfoundation(Table1). Alpha-GPT(Wangetal.,2025)
achieved a top-10 global ranking at WorldQuant’s International Quant Championship
2024among41,000+teamswitha284%ICimprovementoverunaidedhumanresearchers.
AlphaAgent (Tang et al., 2025) achieved an 81% improvement in pass rate via AST-
similarity regularization. QuantaAlpha (Han et al., 2026), current state of the art as of
February 2026, achieves IC=0.1501, ICIR=0.9110, and 27.75% annualized return on
CSI300.
1.3 WorldQuant Brain as Evaluation Oracle
Acriticalprerequisiteforautonomousalpharesearchisarigorous,standardizedevaluation
environment. In-housebacktestingrequiresdatainfrastructure,survivorshipbiascontrols,
and simulation expertise—barriers that impede the rapid iteration LLM-based generation
demands.
WorldQuantBrainremovesthisbarrier. Brainprovidesthesamesimulationinfrastructure
underlyingWorldQuant’slivestrategies, evaluating expressionsagainstup to3,000stocks
with a standardized battery of machine-readable quality checks: Sharpe ratio, fitness
score, turnover bounds, drawdown constraints, and self-correlation limits. These checks
are deterministic, interpretable, and constitute precisely the structured feedback that an
autonomous optimization loop requires.
2

BrainAlpha — Working Paper February 2026
This creates an architectural separation of concerns central to BrainAlpha: the LLM
subsystem is responsible exclusively for creativity; Brain is responsible exclusively for
truth. The agent generates; Brain evaluates. Neither subsystem contaminates the other.
1.4 Contributions
C1. Structured exploration grid (M0): 660-cell Cartesian grid transforming hypoth-
esis search into an exhaustible coverage problem with priority-weighted selection.
C2. RAG-enhanced specification (M1): grounds every hypothesis in Brain’s live
field catalogue, eliminating field hallucination at zero simulation cost.
C3. Failure-mode taxonomy (FM-1 through FM-6): converts binary simulation
feedback into typed, actionable diagnostic signals for targeted repair.
C4. Convergence-governed repair engine (M4): five formal stop conditions prevent
wasted budget on structurally unrecoverable expressions.
C5. Error recovery layer (M3.5): escalating three-strategy repair sequence intercepts
API failures before repair budget is consumed.
3

| BrainAlpha | — Working | Paper |     |     | February | 2026 |
| ---------- | --------- | ----- | --- | --- | -------- | ---- |
| 2 Related  | Work      |       |     |     |          |      |
✓=addressed;
Table1: ComparisonofLLM-basedalphaminingsystemsonfourkeycapabilities.
| ◦=partial; | ×=not | addressed. |           |            |        |            |
| ---------- | ----- | ---------- | --------- | ---------- | ------ | ---------- |
| System     |       |            | Key Fully | Structured | Field- | Production |
Con-
tri-
bu-
tion
|           |       |               | Autonomous | Feedback | Grounded | Evaluation |
| --------- | ----- | ------------- | ---------- | -------- | -------- | ---------- |
| Alpha-GPT | (Wang | et al., 2025) | Human- ×   | ×        | ◦        | ◦          |
AI
in-
ter-
ac-
tive
min-
ing
✓
| AlphaAgent | (Tang | et al., 2025) | AST | ×   | ◦   | ×   |
| ---------- | ----- | ------------- | --- | --- | --- | --- |
di-
ver-
sity
reg-
u-
lar-
iza-
tion
| RD-Agent | (Microsoft | Research, 2025) | Full ✓ | ×   | ×   | ×   |
| -------- | ---------- | --------------- | ------ | --- | --- | --- |
re-
search
loop
agent
| QuantaAlpha | (Han | et al., 2026) | Trajectory-✓ | ×   | ◦   | ×   |
| ----------- | ---- | ------------- | ------------ | --- | --- | --- |
level
evo-
lu-
tion
| BrainAlpha | (ours) |     | FM- ✓ | ✓   | ✓   | ✓   |
| ---------- | ------ | --- | ----- | --- | --- | --- |
aware
re-
pair
pipeline
Theunresolvedgap. Nopriorsystemaddressesstructured,failure-mode-awareiterative
repair within a production evaluation environment. Prior systems treat evaluation as a
black box returning pass or fail. BrainAlpha treats evaluation as a diagnostic instrument
returning typed, actionable failure signals—and builds a repair engine architected around
| that signal | structure.   |     |     |     |     |     |
| ----------- | ------------ | --- | --- | --- | --- | --- |
| 3 System    | Architecture |     |     |     |     |     |
4

| BrainAlpha |        | — Working  | Paper      |           |     |     |     |     | February 2026 |
| ---------- | ------ | ---------- | ---------- | --------- | --- | --- | --- | --- | ------------- |
| 3.1        | Design |            | Principles |           |     |     |     |     |               |
| P1         | —      | Separation | of         | Concerns. |     |     |     |     |               |
LLM modules (M0, M1, M2, M4) handle creativity; the evaluation oracle (M3,
|     | Brain) | handles  | truth.         | Neither | subsystem | contaminates |     | the other. |     |
| --- | ------ | -------- | -------------- | ------- | --------- | ------------ | --- | ---------- | --- |
| P2  | —      | Stateful | Observability. |         |           |              |     |            |     |
All pipeline state is persisted at every LangGraph node transition, enabling deter-
|     | ministic |            | replay from | any checkpoint |      | without re-execution. |     |     |     |
| --- | -------- | ---------- | ----------- | -------------- | ---- | --------------------- | --- | --- | --- |
| P3  | —        | Structured | Feedback    |                | over | Binary Retry.         |     |     |     |
Failures are classified into typed failure modes before repair. Targeted strategies
|     | are | selected      | from | a structured | catalogue | indexed | by  | failure class. |     |
| --- | --- | ------------- | ---- | ------------ | --------- | ------- | --- | -------------- | --- |
| P4  | —   | Data-Grounded |      | Generation.  |           |         |     |                |     |
M1 retrieves verified Brain field names before expression generation. M2 enforces
Brain’s documented operator vocabulary via a deterministic validator. Field
hallucination is eliminated before any simulation budget is consumed.
| 3.2 | Pipeline |     | Topology |     |     |     |     |     |     |
| --- | -------- | --- | -------- | --- | --- | --- | --- | --- | --- |
BrainAlphaisadirectedgraphwithconditionalrouting. Figure1illustratesthecomplete
topology.
M0
|     |     |     |     |     | Thesis | Generation |     |     |     |
| --- | --- | --- | --- | --- | ------ | ---------- | --- | --- | --- |
M1
|     |     |     |     |     | RAG | Specification |     |     |     |
| --- | --- | --- | --- | --- | --- | ------------- | --- | --- | --- |
STOP
M2
|     |     |     |     |     | Expression | Gen. |     |     |     |
| --- | --- | --- | --- | --- | ---------- | ---- | --- | --- | --- |
RECOVERED
|       | M3.5 |          | ERROR |           |       | M3     |     | PASS | Output Log       |
| ----- | ---- | -------- | ----- | --------- | ----- | ------ | --- | ---- | ---------------- |
| Error |      | Recovery |       |           | Brain | Oracle |     |      | (Passing Alphas) |
|       |      |          |       | ABANDONED |       | FAIL   |     |      |                  |
PASS
M4
Repair Engine
Figure 1: BrainAlpha pipeline topology. Every branch terminates: passing alphas accumulate
in the output log; unrecoverable candidates return to M0 for fresh generation.
| 3.3 | The | Signal | Alpha | Object |     |     |     |     |     |
| --- | --- | ------ | ----- | ------ | --- | --- | --- | --- | --- |
All inter-module data flows through a single shared state object: the Signal Alpha
Object (SAO). The SAO is a typed dictionary that evolves as a candidate progresses
through the pipeline, encoding the complete provenance of every candidate signal
5

| BrainAlpha | —   | Working | Paper |     |     |     |     |     |     | February 2026 |
| ---------- | --- | ------- | ----- | --- | --- | --- | --- | --- | --- | ------------- |
(Table 2).
|            |     | Table      | 2: SAO        | field     | groups, | representative | fields, | and owning | module. |            |
| ---------- | --- | ---------- | ------------- | --------- | ------- | -------------- | ------- | ---------- | ------- | ---------- |
| Group      |     | Key        | Fields        |           |         |                |         |            | Owner   | Stage      |
| Identity   |     | id,        | source_agent, |           | cell    |                |         |            | M0      | Generation |
| Hypothesis |     | statement, |               | category, |         | horizon        |         |            | M0      | Generation |
Specification primary_fields, neutralization, M1 Specification
lookback_range
Expression expression, complexity, diversity_score M2 Generation
| Simulation |         | alpha_id,          |          | sharpe,       | fitness, |             | checks[] |     | M3  | Evaluation |
| ---------- | ------- | ------------------ | -------- | ------------- | -------- | ----------- | -------- | --- | --- | ---------- |
| Evaluation |         | status,            |          | failure_mode, |          | m4_eligible |          |     | M3  | Evaluation |
| Repair     |         | repair_iterations, |          |               |          | stop_reason |          |     | M4  | Repair     |
| 3.4        | Failure | Mode               | Taxonomy |               |          |             |          |     |     |            |
Table 3: BrainAlpha Failure Mode Taxonomy (FM-1 through FM-6). Each FM code maps to a
| structured | repair | action |     | catalogue | in         | M4.    |     |         |        |     |
| ---------- | ------ | ------ | --- | --------- | ---------- | ------ | --- | ------- | ------ | --- |
| Code       | Name   |        |     |           | Diagnostic | Signal |     | Primary | Repair |     |
FM-1 Low Signal Quality Sharpe < threshold, flat PnL Neutralization, lookback ex-
tension
FM-2 Excessive Turnover Turnover > upper bound Decaysmoothing,lookbackin-
crease
FM-3 High Drawdown Max drawdown > bound Winsorization, volatility con-
ditioning
FM-4 Signal Reversal Negative annualized returns Negation (-1*), inversion
FM-5 Low Coverage Long/short count below mini- pasteurize(), filter relax-
|     |     |     |     |     | mum |     |     | ation |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- |
FM-6 Low Turnover Turnover < lower bound Lookback reduction, cross-
|     |     |     |     |     |     |     |     | sectional | variation |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | --------- | --- |
4 Methodology
| 4.1   | M0: | Priority-Weighted |     |     | Structured |     | Exploration |     |     |     |
| ----- | --- | ----------------- | --- | --- | ---------- | --- | ----------- | --- | --- | --- |
| 4.1.1 | The | Exploration       |     |     | Grid       |     |             |     |     |     |
The hypothesis space is discretized into a 660-cell exploration grid:
|     | Grid | =   | 11           |     | ×               | 5    | × 3 ×   | 4            | =   | 660 cells |
| --- | ---- | --- | ------------ | --- | --------------- | ---- | ------- | ------------ | --- | --------- |
|     |      |     | |{z}         |     |                 | |{z} | |{z}    | |{z}         |     |           |
|     |      |     | SignalFamily |     | OperatorPattern |      | Horizon | Conditioning |     |           |
Each cell maintains a lifecycle status from {empty,assigned,explored,exhausted}, enforc-
ing breadth-first coverage. The consequence: alpha discovery is no longer a search over
an infinite space. It is an exhaustible coverage problem over 660 well-defined cells.
| 4.1.2 | Cell | Priority |     | Scoring |     |     |     |     |     |     |
| ----- | ---- | -------- | --- | ------- | --- | --- | --- | --- | --- | --- |
Priority(c, a) = 0.5·Novelty(c) + 0.3·ExpectedYield(c) + 0.2·AgentAlignment(c, a)
(1)
Novelty(c) ∈ {1.00,0.50,0.25,0.00} decreases monotonically with exploration status.
ExpectedYield(c) encodes a family-level prior on pass rates updated online after each
confirmed passing alpha (+0.10 per pass in the same cell). AgentAlignment(c,a) ∈
6

| BrainAlpha | —   | Working Paper |     |     |     | February | 2026 |
| ---------- | --- | ------------- | --- | --- | --- | -------- | ---- |
{0.3,0.5,0.7,1.0} captures semantic fit between cell taxonomy and the requesting agent’s
domain expertise.
| 4.2 M1: | RAG-Enhanced |     |     | Specification |     |     |     |
| ------- | ------------ | --- | --- | ------------- | --- | --- | --- |
The translation from thesis to formal specification is where naive LLM generation most
frequently fails: models generate expressions referencing plausible-sounding but non-
existent data fields. This field hallucination problem is structurally unavoidable—Brain’s
fieldcatalogueevolvescontinuously,andnomodelcanremaincurrentthroughpretraining
alone.
M1 embeds Brain’s live field catalogue (≈20,000 verified field names) into a vector index
and retrieves the top-k most relevant fields for each incoming thesis. Retrieved fields are
injected into the specification prompt as the exclusive permissible data vocabulary.
The output is then validated by a six-rule deterministic checker enforcing: non-empty
primary fields; lookback range validity (1 ≤ min ≤ max ≤ 120); recognized neutralization
strategy; non-empty operator family; non-empty direction; hypothesis length (≥ 20
characters). Validation is zero-cost: no LLM invocation, no simulation budget consumed.
| 4.3 M2: | Diversity-Enforced |     |     | Expression | Generation |     |     |
| ------- | ------------------ | --- | --- | ---------- | ---------- | --- | --- |
M2 generates candidate expressions using Chain-of-Thought prompting at temperature
T = 0.7 and applies a Jaccard diversity gate. Expression e is accepted only if:
new
|     |     |     |     | |ops(e )∩ops(e)| |     |     |     |
| --- | --- | --- | --- | ---------------- | --- | --- | --- |
new
|     |     |     | max |                  | < 0.70 |     | (2) |
| --- | --- | --- | --- | ---------------- | ------ | --- | --- |
|     |     |     | e∈A | |ops(e )∪ops(e)| |        |     |     |
new
where ops(e) is the multi-set of operator tokens in expression e and A is the set of
accepted expressions. This enforces structural diversity—not merely lexical variation.
| 4.4 M4: | Convergence-Governed |                   |     | Repair     |     |     |     |
| ------- | -------------------- | ----------------- | --- | ---------- | --- | --- | --- |
| 4.4.1   | Phase                | 1: Recoverability |     | Assessment |     |     |     |
Before any repair iteration, three hard-stop conditions trigger immediate termination
| without | LLM | invocation: |     |     |     |     |     |
| ------- | --- | ----------- | --- | --- | --- | --- | --- |
I1. SELF_CORRELATION=FAIL:structuralduplicationofalivesignal. Expression-
| level | repairs | cannot | resolve | idea-level duplication. |     |     |     |
| ----- | ------- | ------ | ------- | ----------------------- | --- | --- | --- |
I2. Reversed complex signal: returns ≤ 0 and complexity > 20 AST nodes.
Negation cannot propagate through a deeply nested expression without operator-
| semantics |     | conflicts. |     |     |     |     |     |
| --------- | --- | ---------- | --- | --- | --- | --- | --- |
I3. Excessive failures: n ≥ 4. No single-action repair can address four indepen-
failing
| dent | deficiencies. |     |     |     |     |     |     |
| ---- | ------------- | --- | --- | --- | --- | --- | --- |
A soft stop applies when: max m gap_pct(m) > 0.50, where gap_pct(m) = |v m −
| τ |/|τ | |.  |     |     |     |     |     |     |
| ------ | --- | --- | --- | --- | --- | --- | --- |
m m
7

BrainAlpha — Working Paper February 2026
4.4.2 Phase 2: Strategy Selection and Repair Catalogue
The Strategy Selector maps the primary failing check to an ordered list of repair actions.
Selection is fully deterministic: no LLM. Table 4 presents the catalogue for the two most
frequent failing checks.
Table 4: RepairactioncatalogueforLOW_SHARPEandLOW_FITNESS.Actionsaretriedinpriority
order; the first untried action is selected each iteration.
Check Priority Action & Instruction
1 Wrap with group_neutralize(x, sector) to remove sector
beta
LOW_SHARPE 2 Apply group_neutralize(x, subindustry) for finer neutral-
ization
3 Apply zscore(x) cross-sectionally to reduce noise
4 Clip outliers with winsorize(x, std=3)
5 Increase all lookback parameters by 2×
1 Smooth via decay_linear(x, 5) to reduce turnover
2 Normalize as ts_mean(x,d)/ts_std_dev(x,d)
LOW_FITNESS
3 Add pasteurize(x) to improve coverage
4 Restructure using rank() cross-sectionally
4.4.3 Phase 3: LLM Rewrite and Convergence Governance
The Rewriter generates a revised expression at T = 0.3—the conservative temperature
enforces targeted modification rather than creative exploration. The prompt includes:
the current expression, all failing check values with gap percentages, the complete repair
history, and the originating investment thesis.
The loop terminates when any of five conditions is met: (i) the expression passes all
checks; (ii) maximum iteration count reached; (iii) per-iteration Sharpe improvement
< δ = 0.02 for three consecutive iterations; (iv) gap percentage grows rather than shrinks
(divergence); or (v) all catalogue actions exhausted.
5 Discussion
5.1 Structure Matters More Than Scale
ThecentralmethodologicallessonofBrainAlphaisthatstructure matters more than scale
in LLM-based generation tasks with formal correctness requirements. The exploration
grid, the RAG-grounded specification encoder, the deterministic validator, the failure-
mode classifier, and the repair-strategy catalogue each independently reduce failure rates;
together, they produce a system whose output quality is substantially higher than naive
LLM prompting—even with the same underlying model.
This principle generalizes: any domain where LLM generation must satisfy formal
constraints benefits from the same pattern: constrain the generation space first, then use
the LLM to explore within that constrained space with domain-informed intelligence.
8

| BrainAlpha | — Working | Paper    |                   | February 2026 |
| ---------- | --------- | -------- | ----------------- | ------------- |
| 5.2 The    | Economics | of Alpha | Industrialization |               |
At the throughputs demonstrated in initial deployment cycles, a BrainAlpha instance
running continuously against Brain’s WebSim API can generate validated alpha signals
at a rate rivaling a small human research team—at a marginal cost approaching zero per
| additional | signal. |     |     |     |
| ---------- | ------- | --- | --- | --- |
The implication is significant. Alpha discovery is not an irreducibly human activity. The
question for any quantitative organization is not whether to automate alpha discovery,
but when the cost of building the automation infrastructure will be recovered by the
throughput gains it delivers. Given current performance estimates and the documented
cost of quantitative research talent, the crossover point is substantially shorter than most
| practitioners | assume. |                 |     |     |
| ------------- | ------- | --------------- | --- | --- |
| 5.3 Human-AI  |         | Complementarity |     |     |
BrainAlpha does not propose to replace human quantitative researchers. The system
reflects an explicit division of labor: the LLM operates within a structured space defined
by human domain expertise; human researchers define the space. Researchers can focus
on activities requiring genuine judgment—evaluating economic coherence, assessing
regime sensitivity, managing portfolio-level risk—while BrainAlpha handles high-volume
expression generation, evaluation, and repair at throughputs no human can match.
6 Conclusion
This paper has presented BrainAlpha—an autonomous multi-agent pipeline for quan-
titative alpha discovery operating end-to-end against WorldQuant Brain’s production
infrastructure.
The system’s four principal innovations—the structured exploration grid, the RAG-
enhancedspecificationencoder,thefailure-modetaxonomy,andtheconvergence-governed
repair engine—together constitute a system that is not merely technically interesting but
economicallyconsequential. Alphadecayisaquantifiablethroughputdeficit. BrainAlpha
is a quantifiable throughput generator. The gap between the two is the business case
for automated alpha discovery—and it is a gap that only grows larger as systematic
| investment | management | scales. |     |     |
| ---------- | ---------- | ------- | --- | --- |
“Alpha decay waits for no one. Neither should the research pipeline.”
References
Zefeng Chen (2026). Agentic AI as autonomous equity analyst: Generating actionable investment
| signals | at scale. Under | review. |     |     |
| ------- | --------------- | ------- | --- | --- |
Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. Journal of
| Finance, | 25(2):383–417. |     |     |     |
| -------- | -------------- | --- | --- | --- |
Han, Y. et al. (2026). QuantaAlpha: Trajectory-level evolutionary optimization for quantitative
| alpha | mining. arXiv | preprint. |     |     |
| ----- | ------------- | --------- | --- | --- |
9

| BrainAlpha | — Working | Paper |     | February 2026 |
| ---------- | --------- | ----- | --- | ------------- |
Harvey, C. R., Liu, Y., and Zhu, H. (2016). ...and the cross-section of expected returns. Review
| of Financial | Studies, | 29(1):5–68. |     |     |
| ------------ | -------- | ----------- | --- | --- |
Jegadeesh, N. and Titman, S. (1993). Returns to buying winners and selling losers. Journal of
| Finance,    | 48(1):65–91. |                       |                          |     |
| ----------- | ------------ | --------------------- | ------------------------ | --- |
| Kakushadze, | Z. (2016).   | 101 formulaic alphas. | Wilmott, 2016(84):72–80. |     |
McLean,R.D.andPontiff,J.(2016). Doespublishingresearchdestroystockreturnpredictability?
| Journal | of Finance, | 71(1):5–32. |     |     |
| ------- | ----------- | ----------- | --- | --- |
Microsoft Research (2025). RD-Agent: Towards data-centric automatic R&D. NeurIPS 2025.
Tang, Y. et al. (2025). AlphaAgent: Anti-decay alpha mining with LLM agents. KDD 2025.
Wang, X. et al. (2025). Alpha-GPT 2.0: Human-AI interactive alpha mining. AAAI 2025.
Working Paper — Not for citation without permission. Draft: February 27, 2026
10
