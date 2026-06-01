"""
OpenAlpha - Quant — All LLM Prompt Strings
SYSTEM_PROMPT is the full v2 IQC Alpha Researcher prompt (9 sections).
Builder functions construct context-aware user messages including POMDP memory injection.
"""

import json
import logging

from openalpha_brain.data import get_data_path
from openalpha_brain.utils import extract_json_from_llm

logger = logging.getLogger(__name__)


def _load_operator_list() -> str:
    schema_path = get_data_path("brain_operators.json")
    fallback = "rank  ts_rank  ts_mean  ts_std_dev  ts_delta  ts_zscore  ts_decay_linear  group_neutralize  abs  log  signed_power  max  min  scale  ts_delay  ts_sum  ts_corr  ts_regression  ts_arg_max  ts_arg_min  ts_backfill  trade_when  zscore  normalize  winsorize  hump  group_rank  group_zscore  ts_av_diff  ts_quantile  quantile  vec_sum  vec_avg"
    if not schema_path.exists():
        return fallback
    try:
        with open(schema_path, encoding="utf-8") as f:
            data = json.load(f)
        names = sorted({op.get("name", "") for op in data if op.get("name")})
        if names:
            line1 = "  ".join(names[:22])
            line2 = "  ".join(names[22:44])
            line3 = "  ".join(names[44:]) if len(names) > 44 else ""
            result = f"  {line1}"
            if line2:
                result += f"\n  {line2}"
            if line3:
                result += f"\n  {line3}"
            return result
    except (OSError, ValueError, RuntimeError):
        pass
    return fallback

ALPHA_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string", "description": "Economic rationale for the alpha"},
        "expression": {"type": "string", "description": "Pure FastExpr expression, no natural language"},
        "metrics": {
            "type": "object",
            "properties": {
                "sharpe_min": {"type": "number"},
                "sharpe_max": {"type": "number"},
                "fitness_min": {"type": "number"},
                "turnover_min": {"type": "number"},
                "turnover_max": {"type": "number"},
                "returns_pct": {"type": "number"},
                "corr_risk": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]}
            },
            "required": ["sharpe_min", "sharpe_max", "fitness_min"]
        },
        "fingerprint": {
            "type": "object",
            "properties": {
                "dataset": {"type": "string"},
                "topology": {"type": "string"},
                "temporal": {"type": "string"},
                "normalization": {"type": "string"},
                "neutral": {"type": "string"}
            }
        },
        "ast_topology": {"type": "string", "description": "AST pattern like Rank(TSDelta(Divide, Int))"},
        "refinement_log": {"type": "string"},
        "decision": {"type": "string", "enum": ["SUBMIT CANDIDATE", "ADVANCE TO TEST", "ITERATE", "REJECT"]},
        "simulation_payload": {
            "type": "object",
            "properties": {
                "settings": {"type": "object"},
                "regular": {"type": "string"}
            }
        }
    },
    "required": ["rationale", "expression", "metrics", "fingerprint", "ast_topology", "refinement_log", "decision"]
}

_SYSTEM_PROMPT_TEMPLATE = """\
You MUST output valid JSON matching the schema in SECTION 9. Do NOT use markdown headers or section numbers.
══════════════════════════════════════════════════════════════════════
SYSTEM PROMPT — OPENALPHA v2
AUTONOMOUS WORLDQUANT BRAIN ALPHA RESEARCHER
Research basis: AlphaAgent, Chain-of-Alpha, RD-Agent, ELM
══════════════════════════════════════════════════════════════════════

You are an elite, autonomous Quantitative AI Researcher operating as
the core intelligence of OpenAlpha - Quant — a closed-loop alpha mining
platform for WorldQuant BRAIN IQC 2026.

You do not engage in dialogue.
You do not generate boilerplate.
You do not explain what you are about to do.
You do not ask clarifying questions.
You conduct mathematically rigorous, self-optimizing research.
You output structured alpha payloads and nothing else.

You operate as a multi-agent system within a single context:
  — IDEATION AGENT   : formulates economic hypotheses grounded in
                       real market microstructure inefficiencies
  — GENERATION AGENT : constructs valid Fast Expression syntax
  — EVALUATION AGENT : estimates all metrics using exact fitness mathematics
  — DIAGNOSTIC AGENT : classifies failure modes precisely
  — MUTATION AGENT   : applies targeted ELM keyed to the failure type

Your memory is a POMDP belief state. Every generated alpha updates it.
Every failure narrows the search space. Every pass expands the frontier.
You do not repeat. You do not regress. You advance.

══════════════════════════════════════════════════════════════════════
SECTION 1 — IQC HARD GATES (MATHEMATICAL SPECIFICATION)
══════════════════════════════════════════════════════════════════════

▌ Gate 1 — Sharpe Ratio
  Formula  : Sharpe = sqrt(252) × Mean(PnL) / Stdev(PnL)
  Gate     : Sharpe ≥ 1.25   Target: Sharpe > 2.0
  Failure  : High variance → normalize with ts_zscore, tighten to subindustry.

▌ Gate 2 — Fitness (CRITICAL — exact formula)
  Formula  : Fitness = Sharpe × sqrt(|Returns|) / max(Turnover, 0.125)
  Gate     : Fitness > 1.0   Target: Fitness > 2.0
  Dynamic 1: sqrt() dampens returns — do NOT chase raw returns.
  Dynamic 2: Turnover in denominator — halving TO doubles Fitness.
  Dynamic 3: Floor at 12.5% — do not over-smooth below this.
  Sweet spot: Sharpe~1.4, Returns~20%, Turnover~20% → Fitness~2.0
  Worked example:
    Sharpe=1.3, Returns=20%, Turnover=60%:
      Fitness = 1.3 × sqrt(0.20) / 0.60 = 0.97  FAIL
    Same alpha, Turnover=25%:
      Fitness = 1.3 × 0.447 / 0.25 = 2.32  PASS

▌ Gate 3 — Turnover
  Gate: 1% ≤ Turnover ≤ 70%
  Primary lever : ts_decay_linear(alpha, 5) — use 3, 5, 7, or 10
  Secondary     : trade_when(volume > adv20, alpha, -1)
  API lever     : decay=5 applies on top of expression-level smoothing

▌ Gate 4 — Drawdown Stability
  Must hold in both bull AND bear regimes.
  Add volatility conditioning if regime-dependent:
    trade_when(ts_std_dev(close,20) < ts_mean(ts_std_dev(close,20),60), signal, 0)

══════════════════════════════════════════════════════════════════════
SECTION 2 — FAST EXPRESSION LANGUAGE (SYNTAX LAWS)
══════════════════════════════════════════════════════════════════════

▌ PERMITTED OPERATORS
  {OPERATOR_LIST_PLACEHOLDER}

  trade_when(condition, value_if_true, value_if_false)
    — value_if_false = -1 holds inverted position; = 0 holds cash.

  TIME-SERIES OPERATORS REQUIRE LOOKBACK WINDOW (integer as 2nd arg):
    ts_mean(x, N)  ts_std_dev(x, N)  ts_zscore(x, N)  ts_rank(x, N)
    ts_decay_linear(x, N)  ts_delay(x, N)  ts_sum(x, N)  ts_corr(x, y, N)
    ts_delta(x, N)  ts_arg_max(x, N)  ts_arg_min(x, N)  ts_av_diff(x, N)
    ts_quantile(x, N)  ts_regression(y, x, N, lag, rettype)
    ts_regression(y, x, N, lag, rettype) — rettype integer values:
      0=beta  1=se_beta  2=alpha  3=se_alpha  4=corr  5=se_corr  6=residual  7=se_residual  8=r_squared
      ALWAYS use integer rettype (0-8), NEVER use string like "eps" or "beta"
    NEVER omit the lookback window — "Required attribute lookback" = instant BRAIN ERROR.

▌ MANDATORY SYNTAX RULES
  1. INTEGER LOOKBACK WINDOWS ONLY — ts_mean(close, 20) not 20.0
  2. POSITIONAL ARGUMENTS ONLY — ts_decay_linear(close, 5) NOT ts_decay_linear(close, d=5)
  3. group_neutralize(..., industry|subindustry|sector) REQUIRED on every alpha
  4. BALANCED PARENTHESES — stack-verify before output
  5. ONLY USE CONFIRMED BRAIN VARIABLES — the system will provide a curated list each cycle.
     Core price/volume: open, high, low, close, vwap, volume, adv20, returns, cap
     Additional fields are provided dynamically per cycle based on exploration direction.

  6. NO STRING LITERALS — ts_regression(close, vwap, 20, 3, 0) NOT ts_regression(close, vwap, 20, 3, "eps")
  7. NO TRIVIAL PATTERNS — no moving-average crossovers, no raw price ratios

  8. MINIMUM COMPLEXITY — every expression must contain ≥5 operators total
     Count: ts_delta=1, rank=1, ts_decay_linear=1, group_neutralize=1, etc.
     Minimum viable: group_neutralize(rank(ts_decay_linear(field_a / field_b, 10)), industry) = 5 ops
     Expressions with <5 operators will be rejected by PreFilter.

  9. CROSS-FAMILY FIELD REQUIREMENT — use fields from ≥2 different families
     Forbidden: expressions using ONLY {close, open, high, low, vwap, volume, returns}
     Required: at least one fundamental/sentiment/microstructure field mixed with price data
     Exception: if no non-price fields available this cycle, use complex operator composition instead

  10. ANTI-TRIVIAL PATTERNS — these patterns are PROHIBITED:
      ✗ rank(ts_delta(close, N)) — too crowded, near-zero Sharpe
      ✗ rank(ts_mean(volume, N)) — no signal content
      ✗ group_neutralize(rank(A), g) where A has <3 operators inside
      ✗ Pure additive: rank(A) + rank(B) — high crowding risk
      ✗ Any expression without ts_decay_linear OR ts_av_diff OR trade_when

══════════════════════════════════════════════════════════════════════
SECTION 3 — STRUCTURAL FINGERPRINTING & ANTI-CROWDING
══════════════════════════════════════════════════════════════════════

BRAIN filters operate on AST topology, NOT text.
Expression A: -1 * rank(ts_delta(close, 5))
Expression B: -1 * rank(ts_delta(vwap, 20))
Both are Multiply(Const, Rank(TimeDelta(PriceData, Int))) — IDENTICAL topology.
Changing the field name or integer window is cosmetic, NOT structural.

▌ THE 5-DIMENSIONAL STRUCTURAL FINGERPRINT
  DIM 1 — DATASET FAMILY
    Price/Vol    : HIGH RISK — most crowded
    Fundamental  : MEDIUM RISK
    Analyst      : LOW RISK
    ShortInt     : LOW RISK
    Ownership    : LOW RISK
    Constraint: Price/Vol must not exceed 40% of session attempts.

  DIM 2 — OPERATOR TOPOLOGY
    Additive       : rank(A) + rank(B)  — HIGHEST crowding risk. Avoid.
    Multiplicative : rank(fundamental) * rank(volume) — LOW risk. Preferred.
    NestedNonlinear: ts_delta(ts_mean(x,d1),d2) — LOWEST risk. Prioritize.
    Conditional    : trade_when(condition, signal, fallback) — NOVEL.

  DIM 3 — TEMPORAL STRUCTURE
    Short  <10d | Medium 10-60d | Long >60d | Mixed (SHORT×LONG) — most novel

  DIM 4 — NORMALIZATION GEOMETRY
    Rank | ZScore | SignedPower | Scale

  DIM 5 — NEUTRALIZATION SCOPE
    Sector | Industry | SubIndustry

▌ ANTI-CROWDING RULE
  If new alpha shares ≥ 2 dims with any prior alpha: REJECT internally.
  Never change only the lookback window (trivial). Never add abs() cosmetically.

▌ FORBIDDEN TOPOLOGIES
  ✗ Multiply(Const, Rank(TimeDelta(PriceData, Int)))
  ✗ Add(Rank(DataA), Rank(DataB))
  ✗ Rank(Divide(Fund1, Fund2))
  ✗ Any topology differing only in lookback integer

══════════════════════════════════════════════════════════════════════
SECTION 4 — ALPHA IDEATION (ECONOMIC LOGIC FIRST)
══════════════════════════════════════════════════════════════════════

State the inefficiency BEFORE writing any expression.
Must be: (a) plausible, (b) non-obvious, (c) persistent.

ECONOMIC EXPLANATION REQUIREMENT: For every alpha expression you generate, you MUST also provide a brief economic explanation (1-2 sentences) describing the market inefficiency or financial logic the expression captures. Format: ECONOMIC_RATIONALE: <your explanation>

▌ INNOVATION PALETTE (adapted from real BRAIN-passed patterns, ALL >=5 operators)
  Cross-Family Value-Momentum: ts_decay_linear(group_neutralize(-rank(ts_av_diff(close, 10)) + rank(debt / enterprise_value), sector), 10)
    → Combines price reversal with fundamental value, sector-neutralized, decay-weighted (Sharpe 1.77 proven, 6 ops)
  Liquidity-Adjusted Reversal: ts_decay_linear(group_neutralize(-rank(signed_power(ts_zscore(close / vwap, 20), 2)), sector), 10)
    → Price-to-VWAP z-score with nonlinear transform, sector-neutralized (Sharpe 1.69 proven, 5 ops)
  Volume-Quality Momentum: ts_decay_linear(group_neutralize(-rank(ts_delta(ts_rank(returns * volume / adv20, 5), 5)), sector), 10)
    → Returns scaled by relative volume, double-smoothed by delta+rank, neutralized (Sharpe 1.60 proven, 6 ops)
  Volatility-Conditioned Signal: ts_decay_linear(group_neutralize(rank(signed_power(ts_decay_linear(signal / ts_std_dev(close, 20), 10), 2)), industry), 10)
    → Signal normalized by volatility, nonlinear transform, industry-neutralized (5+ ops)
  Fundamental-Price Interaction: ts_decay_linear(group_neutralize(rank(ts_delta(fund_field, 5)) * rank(ts_delta(volume / adv20, 5)), sector), 10)
    → Fundamental change rate multiplied by volume anomaly, cross-confirmed (6 ops)
  Correlation Lead-Lag: ts_decay_linear(group_neutralize(rank(ts_corr(ts_delta(price_field, 5), ts_delta(vol_field, 5), 5)), sector), 10)
    → Price-volume correlation dynamics with full three-block structure (5 ops)

══════════════════════════════════════════════════════════════════════
SECTION 4.5 — PROVEN SUCCESS PATTERNS (from real WQ submissions)
══════════════════════════════════════════════════════════════════════

These patterns have PASSED BRAIN validation with Sharpe>1.5.
Study their structure and ADAPT (not copy) for your generations:

PATTERN A — Cross-Family Interaction (Sharpe 1.77, Fitness 1.26):
  Real expression: ts_decay_linear(group_neutralize(-rank(ts_av_diff(close, 10)) + rank(debt / enterprise_value), sector), 10)
  Structure: decay(neutralize(negation * rank(temporal) ± rank(fundamental_ratio)))
  Key insight: Mix price data with fundamental data for lower crowding risk
  Template: ts_decay_linear(group_neutralize(-rank(ts_av_diff/ts_decay_linear(PRICE_FIELD, W)) + rank(FUND_FIELD / FUND_FIELD2), sector), W)
  Operators: ts_decay_linear + group_neutralize + rank + ts_av_diff + rank = 5+ ✅
  Why it works: price momentum captures short-term mispricing, fundamental ratio captures long-term value signal

PATTERN B — Ratio Decay Reversal (Sharpe 1.69, Fitness 1.07):
  Real expression: ts_decay_linear(group_neutralize(-rank(signed_power(ts_zscore(close / vwap, 20), 2)), sector), 10)
  Structure: decay(neutralize(negation * rank(nonlinear(zscore(ratio)))))
  Key insight: Price-to-volume ratio captures mean-reversion in liquidity
  Template: ts_decay_linear(group_neutralize(-rank(signed_power(ts_zscore(FIELD_A / FIELD_B, ZSCORE_LB), POWER)), sector), DECAY_LB)
  Operators: ts_decay_linear + group_neutralize + rank + signed_power + ts_zscore = 5 ✅
  Why it works: close/vwap deviation from fair value reverts; z-score normalizes; signed_power adds nonlinearity

PATTERN C — Volume-Weighted Momentum (Sharpe 1.60, Fitness 1.03):
  Real expression: ts_decay_linear(group_neutralize(-rank(ts_delta(ts_rank(returns * volume / adv20, 5), 5)), sector), 10)
  Structure: decay(neutralize(negation * rank(delta(rank(volume_scaled_returns)))))
  Key insight: Scale returns by relative volume for quality-weighted signal
  Template: ts_decay_linear(group_neutralize(-rank(ts_delta(ts_rank(returns * VOLUME_PROXY, SHORT_LB), MED_LB)), sector), DECAY_LB)
  Operators: ts_decay_linear + group_neutralize + rank + ts_delta + ts_rank = 5 ✅
  Why it works: high-volume returns are more informative; double-smoothing reduces noise

RULES FOR ADAPTING THESE PATTERNS:
1. ALWAYS wrap with ts_decay_linear(..., W) + group_neutralize(..., sector) — this is MANDATORY
2. Minimum 5 operators per expression (count: ts_* + group_* + rank + zscore + etc.)
3. Change INNER fields to your cycle's allowed fields from FieldProxyMap
4. NEVER use only close/open/high/low — always include at least one non-price field
5. Window sizes: short=3-7, medium=10-20, long=30-60
6. These patterns achieved Sharpe>1.5 because they combine MULTIPLE signal families
7. Pure price-only expressions rarely exceed Sharpe=0.5 in current BRAIN environment
8. If your expression has <5 operators: ADD ts_zscore/signed_power/ts_delta wrapper immediately

══════════════════════════════════════════════════════════════════════
SECTION 5 — MEMORY ARCHITECTURE (POMDP BELIEF STATE)
══════════════════════════════════════════════════════════════════════

The Python loop engine injects current memory state into every user message.
Read it before generating each new alpha.

▌ MEMORY COMPONENTS
  1. EXPLORED TOPOLOGY MAP: fingerprints with PASSED|FAILED|CROWDED status
  2. FAILURE CATALOG: {fingerprint, failure_type, metric_value, mutation_tried}
  3. DATASET EXHAUSTION: track usage per family; force pivot after 3 failures
  4. FAMILY ROTATION: if last 3 same family → force switch
  5. OPEN FRONTIERS: unexplored 5-dim fingerprint combinations

══════════════════════════════════════════════════════════════════════
SECTION 6 — ELM: EVOLUTION THROUGH LARGE MODELS (DETERMINISTIC)
══════════════════════════════════════════════════════════════════════

Every failure maps to ONE primary mutation. Apply it exactly.

▌ FAILURE → MUTATION MATRIX
  TURNOVER > 70%  : ts_decay_linear(expr, 5) — try window 3, 5, 7, 10
  SHARPE < 1.25   : ts_zscore(x, window) replaces rank(); tighten neutralization
  FITNESS ≤ 1.0   : ts_decay_linear to target 20-30% turnover (Turnover in denominator)
  CROWDED         : Change topology (Additive→Multiplicative) OR pivot dataset family
  OVERFIT         : AST pruning — remove outer operators, revert to core signal
  REGIME_INSTAB.  : trade_when(ts_std_dev(close,20) < ts_mean(ts_std_dev(close,20),60), s, 0)

  Max 4 mutations per alpha. Same failure after 2 mutations → try secondary fix.
  After 4 failures → log FAILED, restart ideation from open frontier.

══════════════════════════════════════════════════════════════════════
SECTION 7 — SIMULATION SETTINGS
══════════════════════════════════════════════════════════════════════

  delay=1 (required, eliminates look-ahead), decay=5 (turnover lever),
  truncation=0.05, neutralization=INDUSTRY, universe=TOP3000

══════════════════════════════════════════════════════════════════════
SECTION 8 — AUTONOMOUS VALIDATION PIPELINE
══════════════════════════════════════════════════════════════════════

Run internally before every output. Only output when Step 6 clears.

  STEP 1 — IDEATION: state inefficiency, pick unexplored fingerprint, check <2 overlap dims
  STEP 2 — EXPRESSION: write, verify all 5 syntax rules
  STEP 3 — FITNESS SIMULATION: compute Fitness = Sharpe × sqrt(|Returns|) / max(TO,0.125)
  STEP 4 — FAILURE DIAGNOSIS: name exact root cause + metric value
  STEP 5 — ELM MUTATION: apply matrix entry for diagnosed failure only
  STEP 6 — FINAL OUTPUT: all 4 gates pass → output below format

══════════════════════════════════════════════════════════════════════
SECTION 9: MANDATORY OUTPUT FORMAT — JSON
You MUST output a single valid JSON object matching this schema:
{
  "rationale": "Economic rationale text",
  "expression": "group_neutralize(rank(ts_delta(close,5)), industry)",
  "metrics": {
    "sharpe_min": 1.45, "sharpe_max": 2.0,
    "fitness_min": 1.0, "turnover_min": 10.0, "turnover_max": 30.0,
    "returns_pct": 22.0, "corr_risk": "LOW"
  },
  "fingerprint": {
    "dataset": "Price/Vol", "topology": "NestedNonlinear",
    "temporal": "medium", "normalization": "Rank", "neutral": "industry"
  },
  "ast_topology": "Rank(TSDelta(Close, Int))",
  "refinement_log": "Original idea: raw delta. Fixed by adding rank normalization.",
  "decision": "SUBMIT CANDIDATE",
  "simulation_payload": {"settings": {"delay": 1, "decay": 5}, "regular": "group_neutralize(rank(ts_delta(close, 5)), industry)"}
}
CRITICAL: Output ONLY the JSON object. No markdown fences, no section headers, no extra text.
The "expression" field MUST contain ONLY the FastExpr expression — no natural language.
NEVER use "..." as a placeholder — every function call MUST have complete, valid arguments.
══════════════════════════════════════════════════════════════════════
"""

_OPERATOR_LIST_CACHE: str | None = None


def get_system_prompt() -> str:
    global _OPERATOR_LIST_CACHE
    if _OPERATOR_LIST_CACHE is None:
        _OPERATOR_LIST_CACHE = _load_operator_list()
    return _SYSTEM_PROMPT_TEMPLATE.replace("{OPERATOR_LIST_PLACEHOLDER}", _OPERATOR_LIST_CACHE)


SYSTEM_PROMPT = get_system_prompt()


def build_memory_injection(state) -> str:
    """
    v2: Build POMDP memory string injected into every user message.
    Gives the LLM a structured view of the explored search space.
    """
    explored = []
    for i, fp in enumerate(state.fingerprint_memory):
        topo = state.topology_map.get(fp.get("topology", ""), "PASSED")
        parts = ", ".join(f"{k}={v}" for k, v in fp.items() if v)
        explored.append(f"  FP-{i+1} [{topo}]: {parts}")

    failed_topos = [
        t for t, status in state.topology_map.items() if status == "FAILED"
    ]

    dataset_usage = dict(state.dataset_usage) if state.dataset_usage else {}

    last_3_families = []
    for alpha in state.passed_alphas[-3:]:
        ds = alpha.fingerprint.dataset if alpha.fingerprint else None
        if ds:
            last_3_families.append(ds)

    failure_lines = []
    for fc in state.failure_catalog[-5:]:
        failure_lines.append(
            f"  {fc.get('failure_type','?')}: fp={fc.get('fingerprint',{})} "
            f"metric={fc.get('metric_value','?')} mutation={fc.get('mutation_tried','?')}"
        )

    frontiers = state.open_frontiers[:5] if state.open_frontiers else []

    # Full topology map summary (PASSED + FAILED + CROWDED)
    topo_summary = dict(state.topology_map) if state.topology_map else {}

    lines = [
        "\n\nSESSION MEMORY STATE:",
        "Explored fingerprints:",
    ]
    lines.extend(explored or ["  None yet"])
    lines.append(f"Topology map: {topo_summary or 'empty'}")
    lines.append(f"Failed topologies: {failed_topos or 'None'}")
    lines.append(f"Dataset usage: {dataset_usage}")
    lines.append(f"Last 3 families: {last_3_families or 'None'}")
    lines.append(f"Open frontiers: {frontiers or 'Not yet initialized'}")
    lines.append("Failure catalog (last 5):")
    lines.extend(failure_lines or ["  None yet"])
    lines.append(f"Rejected motifs: {[m.get('topology') for m in state.rejected_motifs if m.get('topology')]}")

    if state.hallucination_log:
        banned_vars = set()
        banned_ops = set()
        for entry in state.hallucination_log:
            if entry.get("error_type") in ("BRAIN_UNKNOWN_VAR", "INVALID_VAR"):
                banned_vars.add(entry.get("variable", ""))
            elif entry.get("error_type") in ("UNKNOWN_OPERATOR",):
                banned_ops.add(entry.get("variable", ""))
        if banned_vars:
            lines.append("\n▌ BANNED VARIABLES (DO NOT USE — caused BRAIN errors):")
            for v in sorted(banned_vars):
                lines.append(f"  ✗ {v}")
        if banned_ops:
            lines.append("\n▌ BANNED OPERATORS (DO NOT USE — not in BRAIN schema):")
            for o in sorted(banned_ops):
                lines.append(f"  ✗ {o}")

    return "\n".join(lines) + "\n"


def build_start_trigger(cycle: int, focus_area: str) -> str:
    focus_str = f"Focus area: {focus_area}." if focus_area and focus_area != "auto" else ""
    return (
        f"Begin Alpha research session. Generate Alpha {cycle}. "
        f"{focus_str} "
        "Follow all 9 sections of your research mandate. "
        "Output the full 8-field v2 format exactly as specified in Section 9. "
        "Do not skip any field."
    )


def build_failure_feedback(
    failures: list,
    expression: str,
    cycle: int,
    values: dict | None = None,
) -> str:
    values_str = ""
    if values:
        values_str = "\n     Specific values: " + ", ".join(
            f"{k}={v}" for k, v in values.items()
        )
    failures_list = "\n     - ".join(failures)
    return (
        f"Alpha {cycle} failed local validation.\n"
        f"     Failed gates:\n     - {failures_list}\n"
        f"     Expression attempted:\n     {expression}\n"
        f"{values_str}\n"
        "     ELM Mutation Matrix — apply the EXACT fix for your failure type:\n"
        "       TURNOVER > 70%  → ts_decay_linear(expr, 5) — try window 3, 5, 7, 10\n"
        "       SHARPE < 1.25   → replace rank() with ts_zscore(x, window)\n"
        "       FITNESS ≤ 1.0   → reduce turnover to 20-30% range\n"
        "       CROWDED         → change topology OR pivot dataset family\n"
        "       INVALID_VAR     → use ONLY fields listed in ALLOWED DATA FIELDS THIS CYCLE\n"
        "     Mutate ONLY the failing component. Output the full 8-field v2 format."
    )


# ── BRAIN check name → targeted ELM mutation instructions ────────────────────
_BRAIN_CHECK_MUTATIONS: dict[str, str] = {
    "LOW_SHARPE": (
        "LOW_SHARPE (real Sharpe < 1.25):\n"
        "  Root cause: signal-to-noise too low. The alpha is directionally wrong or too noisy.\n"
        "  MANDATORY fixes (apply ALL):\n"
        "    1. Replace rank() with ts_zscore(x, 20) — zscore is more stable cross-sectionally\n"
        "    2. Add volatility normalization: divide signal by ts_std_dev(close, 20)\n"
        "    3. Tighten neutralization: change industry → subindustry\n"
        "    4. Use a different temporal structure — if you used short window (<10d), try medium (20-60d)\n"
        "    5. Add regime conditioning: trade_when(volume > ts_mean(volume, 20), signal, 0)\n"
        "  AVOID: simple ts_delta(close, N) — too crowded, near-zero Sharpe"
    ),
    "LOW_FITNESS": (
        "LOW_FITNESS (real Fitness ≤ 1.0):\n"
        "  Formula: Fitness = Sharpe × sqrt(|Returns|) / max(Turnover, 0.125)\n"
        "  Root cause: Turnover too high OR Sharpe too low relative to returns.\n"
        "  MANDATORY fix — apply ts_decay_linear with calculated window:\n"
        "    Window calculation: TO>50% → window=15, TO>35% → window=10, TO>25% → window=6\n"
        "    Wrap inner signal: group_neutralize(ts_decay_linear(YOUR_SIGNAL, WINDOW), industry)\n"
        "    DO NOT double-wrap if ts_decay_linear already present\n"
        "  Expected result: Turnover drops from ~35-50% → ~15-25%, Fitness doubles\n"
        "  Example: group_neutralize(ts_decay_linear(rank(ts_delta(close, 10) / ts_std_dev(close, 20)), 10), industry)"
    ),
    "HIGH_TURNOVER": (
        "HIGH_TURNOVER (real Turnover > 70%):\n"
        "  MANDATORY fix — apply ts_decay_linear with increasing window until TO < 70%:\n"
        "    window=6 → 10 → 15 → 20\n"
        "  Outer wrap: group_neutralize(ts_decay_linear(YOUR_SIGNAL, 10), industry)\n"
        "  Also consider longer lookback windows (20d → 60d)"
    ),
    "LOW_TURNOVER": (
        "LOW_TURNOVER (real Turnover < 1%):\n"
        "  Signal is too static — position barely changes day-to-day.\n"
        "  MANDATORY fix:\n"
        "    - Remove ts_decay_linear if present (or reduce window from 20 → 5)\n"
        "    - Use shorter lookback windows (60d → 10d)\n"
        "    - Use ts_delta() instead of ts_mean() as the outer operator"
    ),
    "LOW_SUB_UNIVERSE_SHARPE": (
        "LOW_SUB_UNIVERSE_SHARPE (alpha underperforms within industry groups):\n"
        "  Root cause: signal works market-wide but reverses within industry peers.\n"
        "  This means the signal is capturing industry-level effects, not stock-specific alpha.\n"
        "  MANDATORY fixes:\n"
        "    1. Change neutralization from 'industry' → 'subindustry' in group_neutralize()\n"
        "    2. Add a second layer: group_neutralize(ts_zscore(signal, 20), subindustry)\n"
        "    3. Use INTERACTION effects: rank(signal_A) * rank(signal_B) where B is volume/returns\n"
        "       This creates cross-sectional dispersion within industries\n"
        "    4. Avoid pure price momentum — it's industry-correlated by nature"
    ),
    "CONCENTRATED_WEIGHT": (
        "CONCENTRATED_WEIGHT (position weights too concentrated):\n"
        "  MANDATORY fix:\n"
        "    - Wrap with scale(): scale(group_neutralize(expr, industry), 1)\n"
        "    - Add truncation in payload: truncation=0.05 (already set)\n"
        "    - Add signed_power(expr, 0.5) to compress outliers before neutralize"
    ),
    "SELF_CORRELATION": (
        "SELF_CORRELATION (too correlated with your existing alphas):\n"
        "  Your existing alphas in BRAIN already capture this signal.\n"
        "  MANDATORY fix — complete structural pivot:\n"
        "    1. Change factor family entirely (if Price/Vol → use Fundamental or Ownership)\n"
        "    2. Change topology (if Additive → Multiplicative, if Rank → ZScore)\n"
        "    3. Change temporal structure (if Short <10d → Long >60d)\n"
        "    4. Change neutralization scope (sector → subindustry)\n"
        "  Must differ in at least 3 of 5 fingerprint dimensions from prior alphas"
    ),
    "BRAIN_SIMULATION_ERROR": (
        "BRAIN SIMULATION ERROR — expression used an INVALID variable:\n"
        "  ONLY these bare variable names exist in BRAIN FastExpr:\n"
        "    close, open, high, low, vwap, volume, adv20, returns, cap\n"
        "  Additional fields are provided dynamically per cycle — ONLY use fields listed in ALLOWED DATA FIELDS.\n"
        "  NEVER USE any variable not explicitly listed in the current cycle's allowed fields.\n"
        "  Rewrite the expression using ONLY confirmed valid variables."
    ),
    # ═══════════════════════════════════════════════════════════
    # Tier 1: Core Learning Mechanisms (from AlphaAgent KDD'25, CRANE ICML'25)
    # ═══════════════════════════════════════════════════════════
    "AST_NON_ORIGINAL": (
        "AST_NON_ORIGINAL (expression structurally similar to existing alphas — AlphaAgent R1):\n"
        "  Your expression's AST sub-tree structure matches previously submitted alphas.\n"
        "  This means you're generating TOPologically REDUNDANT expressions.\n"
        "  MANDATORY structural pivots (apply at least 2):\n"
        "    1. Change OPERATOR family: rank→ts_zscore→ts_rank→group_rank (cross-sectional transform)\n"
        "    2. Change TEMPORAL pattern: ts_delta(N)→ts_regression(N,M)→ts_av_diff(N,M)→ts_corr(N,M)\n"
        "    3. Change COMPOSITION: Additive(a+b) → Multiplicative(a*b) → Ratio(a/b)\n"
        "    4. Change NESTING depth: flatten nested group_neutralize or add new inner layer\n"
        "    5. Change NEUTRALIZATION scope: industry→sector→subindustry→market\n"
        "  Reference: AlphaAgent (KDD'25) Formula 5 — Sub-tree Isomorphism Detection\n"
        "  Goal: Achieve ≤0.7 cosine similarity in AST embedding space vs all prior alphas"
    ),
    "HYPOTHESIS_MISALIGN": (
        "HYPOTHESIS_MISALIGN (expression does NOT reflect the stated economic hypothesis — AlphaAgent R2):\n"
        "  The generated expression's mathematical form contradicts its claimed mechanism.\n"
        "  Example mismatch: hypothesis='mean reversion' but expression uses ts_delta(momentum-like).\n"
        "  MANDATORY realignment:\n"
        "    1. Identify the hypothesis core direction (momentum/value/mean_reversion/liquidity/quality)\n"
        "    2. Map to canonical operator set for that direction:\n"
        "       momentum → ts_delta, ts_regression, ts_rank (trend-following)\n"
        "       value → rank, zscore, group_rank (contrarian/discounting)\n"
        "       mean_reversion → ts_mean, ts_decay_linear, ts_av_diff (reversal)\n"
        "       liquidity → scale, normalize, volume-weighted (microstructure)\n"
        "       quality → group_zscore, group_rank (fundamental ranking)\n"
        "    3. Verify: the outermost operator MUST belong to the mapped set\n"
        "    4. If using hybrid: dominant operator (by nesting depth) determines direction\n"
        "  Reference: AlphaAgent (KDD'25) Formula 7 — Semantic Alignment Score ≥0.6 required"
    ),
    "OVER_COMPLEXITY": (
        "OVER_COMPLEXITY (expression exceeds adaptive complexity threshold — AlphaAgent R3):\n"
        "  Your expression has too many nested operators / too deep AST / too many nodes.\n"
        "  BRAIN penalizes complex expressions (higher overfit risk, lower capacity).\n"
        "  Complexity reduction strategy (3-stage regularization per AlphaAgent Formula 8):\n"
        "    Stage 1 — Operator consolidation:\n"
        "      - Replace ts_std_dev(ts_delta(x,N),M) with a single volatility operator\n"
        "      - Merge adjacent normalization: zscore(rank(x)) → just rank(x) or zscore(x)\n"
        "    Stage 2 — Nesting flattening:\n"
        "      - Reduce depth from >5 layers to ≤3: pull inner ops outward\n"
        "      - Replace group_neutralize(group_neutralize(x,a),b) with group_neutralize(x,b)\n"
        "    Stage 3 — Node count pruning:\n"
        "      - Target: ≤15 AST nodes for optimal BRAIN fitness\n"
        "      - Remove redundant wrappers that don't change semantic meaning\n"
        "  Current threshold: P90 of historical successful alphas complexity distribution\n"
        "  Reference: complexity_control.py adaptive threshold based on success history"
    ),
    # ═══════════════════════════════════════════════════════════
    # Tier 2: Logic Improvement (from Alpha-GPT EMNLP'25, CogAlpha arXiv)
    # ═══════════════════════════════════════════════════════════
    "HUMAN_INSPIRED_PIVOT": (
        "HUMAN_INSPIRED_PIVOT (standard mutations failing — try human-discovered patterns — Alpha-GPT):\n"
        "  Pure LLM-generated mutations keep hitting the same failure mode.\n"
        "  Switch to HUMAN-INSPIRED structural templates that historically passed BRAIN:\n"
        "    Template A (value reversal): rank(-ts_delta(close,5) / ts_std_dev(volume,20))\n"
        "    Template B (quality signal): group_zscore(returns / adv20, subindustry)\n"
        "    Template C (liquidity-adjusted momentum): scale(ts_rank(returns/vwap,10) * (1-adv20/cap))\n"
        "    Template D (interaction effect): rank(ts_corr(close,volume,10)) * rank(ts_delta(close,5))\n"
        "  Task: Pick the template CLOSEST to your current direction, then ADAPT one operator\n"
        "  to use your cycle's specific allowed fields. Do NOT copy verbatim.\n"
        "  Reference: Alpha-GPT (EMNLP'25) — RAG-based human-AI collaborative mining"
    ),
    "COGALPHA_DIVERSITY_INJECTION": (
        "COGALPHA_DIVERSITY_INJECTION (search space collapsing — need novel structures — CogAlpha):\n"
        "  Your recent expressions occupy a narrow region of the alpha search space.\n"
        "  Diversity injection by exploration level (CogAlpha 3-tier diversity):\n"
        "    Level 1 (Light): Swap ONE leaf node — change field only (close→open→vwap→returns)\n"
        "    Level 2 (Medium): Change operator family + field — ts_delta→ts_regression+new_field\n"
        "    Level 3 (Creative): Complete topology change — additive→multiplicative→ratio→conditional\n"
        "  Current level selection based on consecutive failure count:\n"
        "    1-2 failures → Level 1, 3-5 failures → Level 2, 6+ failures → Level 3\n"
        "  Goal: Maintain MAP-Elites feature cell occupancy across (sharpe_range, turnover_range, direction)\n"
        "  Reference: CogAlpha (arXiv 2511.18850) — Multi-level diversity injection"
    ),
    "COGALPHA_ELITE_CROSSOVER": (
        "COGALPHA_ELITE_CROSSOVER (leverage top-performing alphas — CogAlpha elite assembly):\n"
        "  Cross-breed your expression with ELITE parent expressions (historical sharpe>1.5):\n"
        "  Crossover operations (pick ONE):\n"
        "    OP1 — Semantic swap: replace your core operator subtree with elite's core subtree\n"
        "    OP2 — Field graft: keep operator structure, swap in elite's high-performing fields\n"
        "    OP3 — Wrapper exchange: swap outer neutralization/normalization layer\n"
        "    OP4 — Hybrid fusion: linear_comb = 0.6 * your_expr + 0.4 * elite_expr (normalized)\n"
        "  Available elite parents are provided in INSPIRATION section above.\n"
        "  Constraint: resulting expression must pass syntax validation and whitelist check\n"
        "  Reference: CogAlpha — Elite-assisted crossover with fitness inheritance"
    ),
    # ═══════════════════════════════════════════════════════════
    # Tier 3: Search/Learning Mechanisms (from MCTS, AlphaBench ICLR'26)
    # ═══════════════════════════════════════════════════════════
    "MCTS_FREQUENT_SUBTREE_AVOID": (
        "MCTS_FREQUENT_SUBTREE_AVOID (over-explored pattern detected — MCTS paper):\n"
        "  The sub-expression pattern you're using has been tried TOO MANY TIMES recently\n"
        "  without producing improved results. MCTS UCT score for this branch is depleted.\n"
        "  Forced exploration moves:\n"
        "    1. Identify the REPEATED subtree (check your last 5 expressions' common patterns)\n"
        "    2. BAN that subtree structure for this mutation attempt\n"
        "    3. Select an UNDER-EXPLORED region from:\n"
        "       - Unexplored operator combinations in your direction\n"
        "       - Fields you haven't used in the last 10 cycles\n"
        "       - Temporal windows outside your usual range\n"
        "    4. Apply bonus exploration weight: try something with <10% historical usage\n"
        "  Reference: LLM-Powered MCTS (arXiv 2505.11122) — UCT exploration with frequent-subtree penalty"
    ),
    "ALPHABENCH_WEAK_EVALUATION_GUARD": (
        "ALPHABENCH_WEAK_EVALUATION_GUARD (alpha too weak to be meaningful — AlphaBench ICLR'26):\n"
        "  Your alpha's metrics fall below the MINIMUM VIABILITY THRESHOLD:\n"
        "    Sharpe < 0.5 OR Fitness < 0.3 OR |Returns| < 1%\n"
        "  Such weak alphas waste BRAIN compute quota and pollute the learning signal.\n"
        "  Instead of mutating this weak alpha, consider:\n"
        "    1. ABANDON this lineage entirely — ideate a fresh hypothesis\n"
        "    2. Switch to a completely different direction (if stuck in momentum → try value/quality)\n"
        "    3. Use the Inspiration expressions as starting points instead of mutating failures\n"
        "  Threshold rationale: AlphaBench shows alphas below these levels have <5% chance\n"
        "  of ever reaching sharpe≥1.25 after optimization. Cut losses early.\n"
        "  Reference: AlphaBench (ICLR'26) — Zero-shot weak alpha detection"
    ),
    "ALPHABENCH_STABILITY_ENFORCEMENT": (
        "ALPHABENCH_STABILITY_ENFORCEMENT (output instability between attempts — AlphaBench):\n"
        "  Your LLM is generating INCONSISTENT expressions across mutation attempts.\n"
        "  Expression structure drifts too much between attempts — no convergence.\n"
        "  Stability enforcement protocol:\n"
        "    1. LOCK the outer structure: always output group_neutralize(..., industry)\n"
        "    2. LOCK the core field set: use the same 2-3 fields across next 3 attempts\n"
        "    3. ONLY vary: operator choice, window sizes, and inner transformations\n"
        "    4. If after 3 locked attempts still FAIL → then unlock and pivot direction\n"
        "  Convergence metric: structural similarity between consecutive attempts should be >0.5\n"
        "  Reference: AlphaBench (ICLR'26) — Output stability as predictor of final quality"
    ),
}


def _elm_fuzzy_match(check_name: str) -> str | None:
    """Fallback fuzzy matcher for unrecognized BRAIN check names."""
    _ALIASES: dict[str, list[str]] = {
        "LOW_SHARPE": ["SHARPE", "POOR_SHARPE", "SHARPE_FAIL", "MIN_SHARPE"],
        "LOW_FITNESS": ["FITNESS", "POOR_FITNESS", "FITNESS_FAIL"],
        "HIGH_TURNOVER": ["TURNOVER_HIGH", "EXCESSIVE_TURNOVER", "TOO_MUCH_TRADING"],
        "LOW_TURNOVER": ["TURNOVER_LOW", "STATIC_SIGNAL", "NO_TRADING"],
        "SELF_CORRELATION": ["CORRELATION", "HIGH_CORRELATION", "SIMILAR_ALPHA", "DUPLICATE"],
        "CONCENTRATED_WEIGHT": ["CONCENTRATION", "WEIGHT_CONCENTRATION", "CONCENTRATED"],
        "BRAIN_SIMULATION_ERROR": ["SIMULATION_ERROR", "UNKNOWN_VAR", "INVALID_VARIABLE",
                              "SYNTAX_ERROR", "PARSE_ERROR", "RUNTIME_ERROR"],
        "GRAPHER": ["GRAPHER_FAIL", "GRAPH_ERROR"],
        "MARGINAL": ["MARGINAL_CONTRIB", "LOW_MARGINAL", "MARGINAL_FAIL"],
        "AST_NON_ORIGINAL": ["NON_ORIGINAL", "NOT_UNIQUE", "TOPOLOGY_CLASH", "SIMILAR_TO_EXISTING"],
        "HYPOTHESIS_MISALIGN": ["MISALIGN", "DIRECTION_MISMATCH", "LOGIC_INCONSISTENCY"],
        "OVER_COMPLEXITY": ["TOO_COMPLEX", "COMPLEX", "DEEP_NESTING", "TOO_MANY_OPS"],
    }
    check_upper = check_name.upper().strip()
    for canonical, aliases in _ALIASES.items():
        if check_upper == canonical or check_upper in [a.upper() for a in aliases]:
            return canonical
    for canonical, aliases in _ALIASES.items():
        for alias in aliases:
            if alias.upper() in check_upper or check_upper in alias.upper():
                return canonical
    return None


def build_brain_failure_feedback(
    brain_checks: list[dict],
    expression: str,
    cycle: int,
    real_sharpe: float | None,
    real_fitness: float | None,
    real_turnover: float | None,
    real_returns: float | None,
    brain_alpha_id: str | None,
    mutation_attempt: int = 1,
    error_message: str = "",
    inspiration_exprs: list[str] | None = None,
    real_drawdown: float | None = None,
) -> str:
    """
    Build a highly targeted LLM feedback message from BRAIN's real gate check results.

    Maps each failing BRAIN check name to a precise ELM mutation instruction.
    The LLM receives exact real metrics + surgical fix instructions for each failure.
    """
    # Identify failed and pending checks
    failed_checks  = [c for c in brain_checks if c.get("result") == "FAIL"]
    pending_checks = [c for c in brain_checks if c.get("result") == "PENDING"]

    # Build mutation instructions per failure
    mutation_blocks = []

    # Handle simulation error (bad variable name)
    if error_message and "unknown variable" in error_message.lower():
        mutation_blocks.append(_BRAIN_CHECK_MUTATIONS["BRAIN_SIMULATION_ERROR"])
    elif error_message:
        mutation_blocks.append(
            f"BRAIN SIMULATION ERROR: {error_message[:200]}\n"
            "  Fix the expression syntax and resubmit."
        )

    for chk in failed_checks:
        name  = chk.get("name", "")
        value = chk.get("value")
        limit = chk.get("limit")
        mutation = _BRAIN_CHECK_MUTATIONS.get(name)
        if mutation:
            # Inject real values into the instruction
            block = f"▌ BRAIN CHECK FAILED: {name}\n"
            if value is not None and limit is not None:
                block += f"  Real value={value}  |  Required limit={limit}\n"
            block += mutation
            mutation_blocks.append(block)
        else:
            matched = _elm_fuzzy_match(name)
            if matched and _BRAIN_CHECK_MUTATIONS.get(matched):
                block = f"▌ BRAIN CHECK FAILED: {name} → mapped to [{matched}]\n"
                if value is not None and limit is not None:
                    block += f"  Real value={value}  |  Required limit={limit}\n"
                block += _BRAIN_CHECK_MUTATIONS[matched]
                mutation_blocks.append(block)
            else:
                mutation_blocks.append(
                    f"▌ BRAIN CHECK FAILED: {name} (value={value}, limit={limit})\n"
                    "  No specific ELM mapping available. Apply general improvement:\n"
                    "    1. If Sharpe-related: normalize volatility, tighten neutralization\n"
                    "    2. If Turnover-related: add/remove ts_decay_linear smoothing\n"
                    "    3. If correlation-related: change operator family or field set entirely\n"
                    "    4. Reduce complexity: flatten nesting, remove redundant wrappers"
                )

    for chk in pending_checks:
        mutation_blocks.append(
            f"▌ BRAIN CHECK PENDING: {chk.get('name')} — will be evaluated after Sharpe/Fitness pass"
        )

    if real_drawdown is not None and real_drawdown > 10.0:
        mutation_blocks.append(
            f"WARNING: High drawdown ({real_drawdown:.1f}%) indicates unstable returns. Consider adding ts_decay_linear or reducing position concentration."
        )

    # Determine primary failure for headline
    primary_failures = [c.get("name", "?") for c in failed_checks]

    # Format real metrics
    def fmt(v, pct=False):
        if v is None: return "N/A"
        return f"{v:.3f}" + ("%" if pct else "")

    # Build inspiration prefix if available
    inspiration_prefix = ""
    if inspiration_exprs:
        inspiration_lines = [
            "━━━ INSPIRATION — Reference Expressions ━━━",
            "These structurally similar expressions have passed BRAIN validation before:",
        ]
        for i, expr in enumerate(inspiration_exprs[:3]):
            inspiration_lines.append(f"  {i+1}. {expr}")
        inspiration_lines.append("")
        inspiration_prefix = "\n".join(inspiration_lines) + "\n"

    lines = [
        f"━━━ BRAIN REAL SIMULATION RESULTS — Alpha {cycle} (mutation attempt {mutation_attempt}) ━━━",
        f"Expression submitted: {expression[:100]}",
        f"BRAIN Alpha ID      : {brain_alpha_id or 'n/a'}",
        "",
        "── REAL METRICS FROM WORLDQUANT BRAIN ──────────────────────────────",
        f"  Sharpe   : {fmt(real_sharpe)}    (gate: ≥ 1.25)",
        f"  Fitness  : {fmt(real_fitness)}    (gate: > 1.0)",
        f"  Turnover : {fmt(real_turnover, pct=True)}  (gate: 1%-70%)",
        f"  Returns  : {fmt(real_returns, pct=True)}",
        "",
        f"── FAILED BRAIN CHECKS: {', '.join(primary_failures) or 'SIMULATION ERROR'} ──────────────",
        "",
    ]

    lines.extend(mutation_blocks)

    lines += [
        "",
        "━━━ YOUR TASK ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "Apply EXACTLY the mutations described above. Follow these rules:",
        "  1. Fix the PRIMARY failure first (listed above)",
        "  2. Use ONLY confirmed BRAIN variables listed in ALLOWED DATA FIELDS THIS CYCLE",
        "  3. Keep group_neutralize() as the outer wrapper",
        "  4. The fix must change the fundamental structure, not just the window size",
        f"  5. This is mutation attempt {mutation_attempt} — if same failure persists after 2 attempts,",
        "     abandon this alpha and ideate a completely different one",
        "",
        "╔════════════════════════════════════════════════════════════════════╗",
        "║  CRITICAL OUTPUT FORMAT — READ CAREFULLY                          ║",
        "║                                                                    ║",
        "║  You MUST output ONE LINE containing ONLY a raw FastExpr expression ║",
        "║  NO JSON object. NO rationale field. NO explanation.               ║",
        "║  NO markdown fences. NO section headers.                           ║",
        "║  DO NOT output: { \"expression\": \"...\", \"rationale\": \"...\" }      ║",
        "║  DO output:   group_neutralize(rank(ts_delta(close, 10)), industry) ║",
        "║                                                                    ║",
        "║  If you output ANYTHING other than a single raw expression line,    ║",
        "║  BRAIN will reject it with 'Unexpected character {'                ║",
        "╚════════════════════════════════════════════════════════════════════╝",
        "",
        "Output ONLY the mutated FastExpr expression. No JSON, no markdown, no explanation.",
        "Example output: group_neutralize(rank(ts_delta(close, 10) / ts_std_dev(close, 20)), industry)",
    ]

    return inspiration_prefix + "\n".join(lines)


_BRAIN_CHECK_DIAGNOSTICS: dict[str, str] = {
    "LOW_SHARPE": (
        "The market anomaly your hypothesis targets has too weak a signal. Possible causes:\n"
        "  1. The anomaly has already been arbitraged away in the current market\n"
        "  2. The causal chain of the hypothesis is not direct enough; need a more precise mechanism description\n"
        "  3. The time window selection may be wrong — short windows have high noise, long windows have signal decay"
    ),
    "HIGH_TURNOVER": (
        "The indicators your hypothesis depends on fluctuate too much, causing excessive daily position changes. Possible causes:\n"
        "  1. The indicator itself is too volatile (e.g., daily returns)\n"
        "  2. Lack of smoothing mechanism; need to incorporate trend persistence or mean reversion in the hypothesis"
    ),
    "LOW_TURNOVER": (
        "The signal produced by your hypothesis is too static; positions barely change. Possible causes:\n"
        "  1. The mechanism described changes too slowly\n"
        "  2. Need a more sensitive market anomaly or a shorter time window"
    ),
    "LOW_SUB_UNIVERSE_SHARPE": (
        "LOW_SUB_UNIVERSE_SHARPE (alpha underperforms within industry groups):\n"
        "-> Underperformance within industries; may need industry neutralization\n"
        "-> Suggestion: add group_neutralize(expr, industry) or group_zscore(expr, industry)\n"
        "-> Check if using signals that are not applicable across industries"
    ),
    "CONCENTRATED_WEIGHT": (
        "CONCENTRATED_WEIGHT (position weights too concentrated):\n"
        "-> Positions too concentrated; need diversification\n"
        "-> Suggestion: add rank() or normalize() to spread weights\n"
        "-> Check if using scale-related fields like cap/size causing concentration\n"
        "-> Consider using winsorize() to clip extreme values"
    ),
    "SELF_CORRELATION": (
        "The market anomaly your hypothesis targets has already been captured by existing alphas. Possible causes:\n"
        "  1. This logic has been extensively researched (e.g., simple momentum/reversal)\n"
        "  2. Need to find more subtle market frictions or behavioral biases"
    ),
    "LOW_FITNESS": (
        "Your hypothesis direction has some signal, but risk-adjusted returns are insufficient. Possible causes:\n"
        "  1. Signal strength is insufficient; high turnover causes trading costs to erode returns\n"
        "  2. Need a stronger signal source or lower turnover"
    ),
    "LOW_IC": (
        "LOW_IC (information coefficient too low):\n"
        "-> Information coefficient too low; signal predictive power is weak\n"
        "-> Suggestion: enhance signal strength, use ts_zscore normalization\n"
        "-> Check if signal direction is correct (may need negation)\n"
        "-> Consider using a longer lookback window"
    ),
    "HIGH_DRAWDOWN": (
        "HIGH_DRAWDOWN (maximum drawdown too large):\n"
        "-> Maximum drawdown too large; signal is unstable\n"
        "-> Suggestion: add ts_decay_linear smoothing to reduce volatility\n"
        "-> Check if using high-volatility fields (e.g., volume)\n"
        "-> Consider using group_neutralize to eliminate industry concentration risk"
    ),
    "LOW_TURNOVER_EXTREME": (
        "LOW_TURNOVER_EXTREME (turnover near zero):\n"
        "-> Turnover extremely low; signal almost never changes\n"
        "-> Suggestion: use shorter lookback windows to increase signal variation\n"
        "-> Check if using overly stable fields (e.g., cap)\n"
        "-> Consider using ts_delta or ts_av_diff to add time-variation"
    ),
    "HIGH_BETA": (
        "HIGH_BETA (beta to market too high):\n"
        "-> Market beta too high; signal too correlated with the market\n"
        "-> Suggestion: add market-neutral treatment\n"
        "-> Use group_neutralize(expr, market) or subtract market returns\n"
        "-> Check if using market-related fields like returns"
    ),
    "LOW_FITNESS_DETAILED": (
        "LOW_FITNESS_DETAILED (comprehensive fitness score too low):\n"
        "-> Comprehensive score insufficient; poor multi-dimensional performance\n"
        "-> Suggestion: improve Sharpe, Turnover, and Drawdown simultaneously\n"
        "-> Check overall signal quality rather than a single dimension\n"
        "-> Consider using rank() + group_neutralize() combo to improve stability"
    ),
    "REGIME_DEPENDENCY": (
        "REGIME_DEPENDENCY (alpha only works in specific market regimes):\n"
        "-> Signal depends on specific market conditions; poor generalization\n"
        "-> Suggestion: add cross-regime robustness treatment\n"
        "-> Use ts_zscore or zscore normalization to adapt to different environments\n"
        "-> Check if overly relying on patterns from specific time periods"
    ),
    "CONCENTRATED_WEIGHT_ORIGINAL": (
        "Your hypothesis leads to overly concentrated positions. Possible causes:\n"
        "  1. The signal only works on very few stocks\n"
        "  2. Need a more broadly applicable market anomaly"
    ),
    "BRAIN_SIMULATION_ERROR": (
        "BRAIN SIMULATION ERROR — expression uses invalid variables:\n"
        "-> Only these bare variable names exist in BRAIN FastExpr:\n"
        "  close, open, high, low, vwap, volume, adv20, returns, cap\n"
        "-> Other fields are dynamically provided by the system; only use fields listed in the current cycle's ALLOWED DATA FIELDS\n"
        "-> Never use unlisted variables; rewrite the expression"
    ),
}


def build_idea_diagnostic_feedback(
    brain_checks: list[dict],
    sharpe: float | None,
    fitness: float | None,
    turnover: float | None,
    direction: str,
) -> str:
    failed_checks = [c for c in brain_checks if c.get("result") == "FAIL"]
    if not failed_checks:
        return ""
    lines = [
        f"Hypothesis for direction '{direction}' failed BRAIN backtest.",
        f"Sharpe={sharpe}, Fitness={fitness}, Turnover={turnover}",
        "",
        "Diagnosis:",
    ]
    for chk in failed_checks:
        name = chk.get("name", "")
        value = chk.get("value")
        limit = chk.get("limit")
        diagnostic = _BRAIN_CHECK_DIAGNOSTICS.get(name)
        if diagnostic:
            lines.append(f"  [{name}] (value={value}, limit={limit})")
            lines.append(f"  {diagnostic}")
        else:
            lines.append(f"  [{name}] (value={value}, limit={limit}) — Unknown check type; adjust hypothesis direction")
    return "\n".join(lines)


def build_success_feedback(
    alpha_id: str,
    cycle: int,
    next_cycle: int,
    fingerprint: dict,
    all_fingerprints: list,
) -> str:
    fp_summary = ", ".join(f"{k}={v}" for k, v in fingerprint.items() if v)
    all_fp_str = "\n".join(
        f"  FP-{i+1}: " + ", ".join(f"{k}={v}" for k, v in fp.items() if v)
        for i, fp in enumerate(all_fingerprints)
    )
    return (
        f"Alpha {cycle} ({alpha_id}) PASSED all IQC gates.\n"
        f"Fingerprint logged: {fp_summary}\n\n"
        f"All accepted fingerprints this session:\n{all_fp_str}\n\n"
        f"Now generate Alpha {next_cycle}.\n"
        "MUST use a different factor family and structural geometry.\n"
        "Reject any idea sharing ≥ 2 fingerprint dims with any logged entry.\n"
        "Output the full 8-field v2 format."
    )


def build_restart_trigger(cycle: int, memory_summary: str) -> str:
    return (
        f"Starting fresh ideation for Alpha {cycle}.\n"
        f"Rejected motifs this session:\n{memory_summary}\n\n"
        "Generate a structurally novel alpha from a completely new family.\n"
        "Do NOT revisit any fingerprint pattern listed above.\n"
        "Full 8-field v2 output required."
    )


def build_family_switch_warning(family: str, cycle: int) -> str:
    return (
        f"\n⚠ FAMILY LOCK: '{family}' used 3 consecutive times. "
        f"For Alpha {cycle} you MUST choose a different factor family. "
        "Failure to switch = automatic REJECT."
    )


def _get_decay_context_injection() -> str:
    try:
        from openalpha_brain.core import loop_state as _ls
        detector = getattr(_ls, '_decay_detector', None)
        if detector is None:
            return ""
        return detector.build_decay_prompt_injection()
    except (OSError, ValueError, AttributeError, RuntimeError):
        return ""


def _get_crossover_context_injection() -> str:
    try:
        from openalpha_brain.core import loop_state as _ls
        parts: list[str] = []
        insights = getattr(_ls, '_trajectory_crossover_insights', None)
        if insights:
            parts.append("\n▶ TRAJECTORY CROSSOVER INSIGHTS (recently discovered strategy combinations):")
            for i, ins in enumerate(insights[:5]):
                direction = ins.get("direction", "?")
                strategy = ins.get("strategy", "?")
                mechanism = ins.get("mechanism", "")
                complementary = ins.get("complementary", [])
                analysis = ins.get("analysis", "")
                line = f"  {i+1}. direction={direction}, strategy={strategy}"
                if mechanism:
                    line += f", mechanism={mechanism[:120]}"
                if complementary:
                    line += f", complementary_segments={complementary}"
                if analysis:
                    line += f", analysis={analysis[:150]}"
                parts.append(line)
            parts.append("  Consider these crossover-derived directions for your next alpha — they combine complementary strengths from multiple trajectories.")
        alerts = getattr(_ls, '_weak_segment_alerts', None)
        if alerts:
            parts.append("\n▶ WEAK SEGMENT ALERTS (recently diagnosed failure patterns from trajectory mutation):")
            for i, alert in enumerate(alerts[:5]):
                weak_seg = alert.get("weak_segment", "?")
                diagnosis = alert.get("diagnosis", "")
                direction = alert.get("direction", "")
                line = f"  {i+1}. weak_segment={weak_seg}"
                if diagnosis:
                    line += f", diagnosis={diagnosis[:150]}"
                if direction:
                    line += f", mutated_direction={direction}"
                parts.append(line)
            parts.append("  AVOID repeating these weak patterns — they have been diagnosed as failure-prone in recent trajectory mutations.")
        return "\n".join(parts) if parts else ""
    except (OSError, ValueError, AttributeError, RuntimeError):
        return ""


def _build_field_family_map(field_ids: list[str]) -> dict[str, tuple[str, str]] | None:
    """
    Build a mapping from field names to (family_id, crowd_level) tuples.
    Used to annotate fields in the dynamic context prompt with crowding risk information.

    Family definitions based on WorldQuant BRAIN data taxonomy:
      - price_trend: core OHLCV fields — HIGH crowd risk
      - volume_liquidity: volume-related — HIGH crowd risk
      - valuation: fundamental value ratios — LOW crowd risk (preferred)
      - growth: earnings/sales growth — LOW crowd risk
      - quality: profitability/efficiency metrics — MEDIUM crowd risk
      - sentiment: analyst estimates/revisions — LOW crowd risk
      - microstructure: trade-level features — MEDIUM crowd risk
      - ownership: institutional holdings — LOW crowd risk
      - risk: volatility/drawdown measures — MEDIUM crowd risk
    """
    if not field_ids:
        return None

    family_definitions: dict[str, tuple[str, str]] = {
        # price_trend family (HIGH CROWDING)
        "close": ("price_trend", "high"),
        "open": ("price_trend", "high"),
        "high": ("price_trend", "high"),
        "low": ("price_trend", "high"),
        "vwap": ("price_trend", "high"),
        "returns": ("price_trend", "high"),

        # volume_liquidity family (HIGH CROWDING)
        "volume": ("volume_liquidity", "high"),
        "adv20": ("volume_liquidity", "high"),
        "adv60": ("volume_liquidity", "high"),
        "cap": ("volume_liquidity", "medium"),

        # valuation family (LOW CROWDING — PREFERRED)
        "debt": ("valuation", "low"),
        "enterprise_value": ("valuation", "low"),
        "market_cap": ("valuation", "low"),
        "book_value": ("valuation", "low"),
        "sales": ("valuation", "low"),
        "earnings": ("valuation", "low"),
        "ebitda": ("valuation", "low"),
        "assets": ("valuation", "low"),
        "equity": ("valuation", "low"),
        "cash_flow": ("valuation", "low"),
        "revenue": ("valuation", "low"),
        "net_income": ("valuation", "low"),
        "total_debt": ("valuation", "low"),
        "working_capital": ("valuation", "low"),
        "inventory": ("valuation", "low"),
        "receivables": ("valuation", "low"),
        "payables": ("valuation", "low"),

        # growth family (LOW CROWDING)
        "earnings_growth": ("growth", "low"),
        "revenue_growth": ("growth", "low"),
        "sales_growth": ("growth", "low"),
        "eps_growth": ("growth", "low"),
        "ebitda_growth": ("growth", "low"),

        # quality family (MEDIUM CROWDING)
        "roe": ("quality", "medium"),
        "roa": ("quality", "medium"),
        "profit_margin": ("quality", "medium"),
        "gross_margin": ("quality", "medium"),
        "operating_margin": ("quality", "medium"),
        "asset_turnover": ("quality", "medium"),
        "current_ratio": ("quality", "medium"),
        "quick_ratio": ("quality", "medium"),
        "debt_to_equity": ("quality", "medium"),
        "interest_coverage": ("quality", "medium"),

        # sentiment family (LOW CROWDING)
        "analyst_revision": ("sentiment", "low"),
        "eps_estimate": ("sentiment", "low"),
        "target_price": ("sentiment", "low"),
        "recommendation": ("sentiment", "low"),
        "consensus": ("sentiment", "low"),

        # microstructure family (MEDIUM CROWDING)
        "bid_ask_spread": ("microstructure", "medium"),
        "trade_count": ("microstructure", "medium"),
        "trade_size": ("microstructure", "medium"),
        "volatility": ("microstructure", "medium"),

        # ownership family (LOW CROWDING)
        "institutional_holdings": ("ownership", "low"),
        "insider_holdings": ("ownership", "low"),
        "short_interest": ("ownership", "low"),
        "float": ("ownership", "low"),

        # risk family (MEDIUM CROWDING)
        "beta": ("risk", "medium"),
        "variance": ("risk", "medium"),
        "downside_risk": ("risk", "medium"),
        "max_drawdown": ("risk", "medium"),
        "value_at_risk": ("risk", "medium"),
    }

    result = {}
    for field in field_ids:
        field_lower = field.lower().strip()
        if field_lower in family_definitions:
            result[field] = family_definitions[field_lower]
        else:
            result[field] = ("unknown", "medium")

    return result if result else None


def build_dynamic_context(rag_context: dict | None = None, global_blacklist: list[dict] | None = None, experience_cards: list[dict] | None = None) -> str:
    """
    Build dynamic context injection from RAG + MAB retrieval results
    and global hallucination blacklist.
    Returns a string to be appended to SYSTEM_PROMPT.
    If rag_context, global_blacklist, and experience_cards are all None, returns empty string (backward compatible).
    """
    if rag_context is None and global_blacklist is None and experience_cards is None:
        return ""

    lines = ["\n\n══════════════════════════════════════════════════════════════════════",
             "DYNAMIC RESEARCH CONTEXT (RAG-retrieved for this cycle)",
             "══════════════════════════════════════════════════════════════════════"]

    if rag_context:
        direction = rag_context.get("exploration_direction", "")
        if direction:
            lines.append(f"\n▶ EXPLORATION DIRECTION: {direction}")

        top_ops = rag_context.get("top_ops_detailed", [])
        if top_ops:
            lines.append("\n▶ RECOMMENDED OPERATORS (detailed):")
            for op in top_ops:
                lines.append(f"  • {op['name']}")
                if op.get("category"):
                    lines.append(f"    Category: {op['category']}")
                if op.get("definition"):
                    lines.append(f"    Signature: {op['definition']}")
                if op.get("description"):
                    desc = op["description"][:200]
                    lines.append(f"    Description: {desc}")

        remaining_ops = rag_context.get("remaining_op_names", [])
        if remaining_ops:
            lines.append(f"\n▶ OTHER AVAILABLE OPERATORS (name only): {', '.join(remaining_ops)}")

        field_ids = rag_context.get("field_ids", [])
        if field_ids:
            lines.append(f"\n▶ ALLOWED DATA FIELDS THIS CYCLE: {', '.join(field_ids)}")
            lines.append("  You may ONLY use these data fields in your expression. Using any other field will cause BRAIN ERROR.")

            field_family_map = _build_field_family_map(field_ids)
            if field_family_map:
                lines.append("\n▶ FIELD FAMILY ASSIGNMENT:")
                for field, (family, crowd_level) in field_family_map.items():
                    crowd_marker = "HIGH CROWDING — minimize usage" if crowd_level == "high" else \
                                   "MEDIUM CROWDING" if crowd_level == "medium" else \
                                   "LOW CROWDING — PREFERRED"
                    lines.append(f"  {field} → {family} ({crowd_marker})")

                low_crowd_fields = [f for f, (_, cl) in field_family_map.items() if cl == "low"]
                high_crowd_fields = [f for f, (_, cl) in field_family_map.items() if cl == "high"]
                if low_crowd_fields:
                    lines.append(f"\n  RECOMMENDATION: Use at least 1 field from LOW CROWDING family: {', '.join(low_crowd_fields)}")
                if high_crowd_fields:
                    lines.append(f"  AVOID overusing HIGH CROWDING fields: {', '.join(high_crowd_fields)}")

        finlogic_ids = rag_context.get("financial_logic_ids", [])
        if finlogic_ids:
            lines.append(f"\n▶ RELEVANT FINANCIAL LOGIC: {', '.join(finlogic_ids)}")

    if global_blacklist:
        lines.append("\n▶ GLOBAL BLACKLIST (variables/operators confirmed invalid across sessions):")
        lines.append("  DO NOT USE any of the following — they have been repeatedly rejected by BRAIN:")
        for entry in global_blacklist:
            count = entry.get("count", 1)
            last_seen = entry.get("last_seen", "unknown")
            lines.append(f"  ✗ {entry['content']} (seen {count}x, last: {last_seen})")

    if experience_cards:
        lines.append("\n▶ LEARNED EXPERIENCE RULES:")
        for card in experience_cards:
            pattern = card.get("failure_pattern", "")
            fix = card.get("fix_strategy", "")
            cond = card.get("applicable_conditions", "")
            conf = card.get("confidence", 0.0)
            lines.append(f"  • [{pattern}] Fix: {fix}")
            if cond:
                lines.append(f"    Condition: {cond}")
            lines.append(f"    Confidence: {conf:.2f}")

    decay_injection = _get_decay_context_injection()
    if decay_injection:
        lines.append(decay_injection)

    crossover_injection = _get_crossover_context_injection()
    if crossover_injection:
        lines.append(crossover_injection)

    lines.append("\n══════════════════════════════════════════════════════════════════════")
    return "\n".join(lines)


def build_fine_tune_guidance(direction: str, brain_checks: list[dict] | None = None) -> str:
    try:
        hints = []
        direction_lower = direction.lower()
        if "momentum" in direction_lower:
            hints.append("For momentum signals: prefer ts_delta(close, N) or ts_zscore(close, N) with N=5-20")
        elif "value" in direction_lower:
            hints.append("For value signals: prefer rank(...) or group_zscore(..., industry)")
        elif "volatility" in direction_lower:
            hints.append("For volatility signals: prefer ts_std_dev(..., N) or ts_av_diff(..., N)")
        elif "mean_reversion" in direction_lower:
            hints.append("For mean reversion: prefer ts_mean(..., N) - ... or ts_decay_linear(..., N) - ...")

        if brain_checks:
            for chk in brain_checks:
                name = chk.get("name", "")
                if name == "LOW_SHARPE":
                    hints.append("LOW_SHARPE detected: add normalization (ts_zscore, rank) or extend lookback window")
                elif name == "HIGH_TURNOVER":
                    hints.append("HIGH_TURNOVER detected: add smoothing (ts_decay_linear) or increase decay parameter")
                elif name == "SELF_CORRELATION":
                    hints.append("SELF_CORRELATION detected: replace core operator with alternative (e.g., ts_delta→ts_zscore)")

        return "\n".join(f"  - {h}" for h in hints) if hints else ""
    except (OSError, ValueError, RuntimeError):
        return ""


async def llm_diagnose_failure(
    brain_checks: list[dict],
    expression: str,
    hypothesis_text: str,
    llm_call_fn=None,
) -> dict | None:
    if not llm_call_fn:
        return None

    check_summary = "\n".join(
        f"- {c.get('name', '?')}: value={c.get('value', 'N/A')}, limit={c.get('limit', 'N/A')}"
        for c in brain_checks
    )

    prompt = (
        f"You are a quantitative finance diagnostic expert.\n"
        f"An alpha factor EXPRESSION failed BRAIN validation.\n\n"
        f"EXPRESSION: {expression}\n"
        f"HYPOTHESIS: {hypothesis_text}\n"
        f"FAILED CHECKS:\n{check_summary}\n\n"
        f"Provide a structured diagnosis:\n"
        f"1. failure_type: (e.g., 'signal_too_weak', 'excessive_trading', 'concentration_risk')\n"
        f"2. root_cause: (1-2 sentence explanation)\n"
        f"3. suggested_fix: (specific modification to the expression)\n"
        f"4. confidence: (0.0-1.0)\n\n"
        f"Respond in JSON format only."
    )

    try:
        result = await llm_call_fn(prompt)
        if result:
            text = result if isinstance(result, str) else str(result)
            parsed = extract_json_from_llm(text)
            if isinstance(parsed, dict):
                return {
                    "failure_type": parsed.get("failure_type", "unknown"),
                    "root_cause": parsed.get("root_cause", ""),
                    "suggested_fix": parsed.get("suggested_fix", ""),
                    "confidence": float(parsed.get("confidence", 0.5)),
                }
    except (ValueError, TypeError) as exc:
        logger.warning("llm_diagnose_failure: failed to parse LLM diagnosis: %s", exc)

    return None
