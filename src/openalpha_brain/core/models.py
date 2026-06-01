"""
OpenAlpha - Quant — Pydantic Data Models
Single source of truth for all data shapes used across the system.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class SessionStatus(StrEnum):
    IDLE = "IDLE"
    GENERATING = "GENERATING"
    PARSING = "PARSING"
    VALIDATING = "VALIDATING"
    ITERATING = "ITERATING"
    SUBMITTING = "SUBMITTING"
    PASS = "PASS"
    FAIL = "FAIL"
    STOPPED = "STOPPED"
    ERROR = "ERROR"
    CRASHED = "CRASHED"


class BrainSimStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"  # no BRAIN credentials configured


class PipelineStatus(StrEnum):
    GENERATED = "GENERATED"
    VALIDATED = "VALIDATED"
    QUEUED = "QUEUED"
    BRAIN_SUBMITTED = "BRAIN_SUBMITTED"
    BRAIN_POLLING = "BRAIN_POLLING"
    BRAIN_RESULT = "BRAIN_RESULT"
    IMPROVING = "IMPROVING"
    REVIEW_SUBMITTED = "REVIEW_SUBMITTED"
    COMPLETED = "COMPLETED"
    ABANDONED = "ABANDONED"


class BrainSubmissionResult(BaseModel):
    """Result from submitting an alpha to WorldQuant BRAIN API."""

    status: BrainSimStatus = BrainSimStatus.PENDING
    alpha_id: str | None = None  # BRAIN-assigned alpha ID
    real_sharpe: float | None = None
    real_fitness: float | None = None
    real_turnover: float | None = None
    real_returns: float | None = None
    real_drawdown: float | None = None
    real_margin: float | None = None
    gate_failures: list[str] = []
    gate_warnings: list[str] = Field(default_factory=list)
    brain_checks: list[dict] = Field(default_factory=list)
    error_message: str | None = None
    review_submitted: bool = False
    submitted_at: datetime | None = None
    completed_at: datetime | None = None
    os_sharpe: float | None = None
    os_fitness: float | None = None
    os_returns: float | None = None
    is_os_decay_ratio: float | None = None
    overfitting_warning: str | None = None


class AlphaMetrics(BaseModel):
    """Estimated metrics extracted from the LLM's [3] ESTIMATED METRICS section."""

    sharpe_min: float | None = None
    sharpe_max: float | None = None
    fitness_min: float | None = None
    fitness_max: float | None = None
    fitness_computed: float | None = None  # v2: computed via exact formula
    fitness_breakdown: str | None = None  # v2: shows arithmetic
    turnover_min: float | None = None
    turnover_max: float | None = None
    returns_pct: float | None = None  # v2: annualized returns estimate
    corr_risk: str | None = None  # LOW | MEDIUM | HIGH


class AlphaFingerprint(BaseModel):
    """6-field structural fingerprint for anti-crowding memory."""

    dataset: str | None = None  # fundamental | price_volume | analyst | ...
    topology: str | None = None  # additive | multiplicative | nonlinear
    temporal: str | None = None  # short | medium | long
    normalization: str | None = None  # rank | zscore | scale | signed_power
    direction: str | None = None  # mean-reverting | trending | regime-switching
    neutral: str | None = None  # sector | industry


class AlphaResult(BaseModel):
    """A single fully-parsed alpha that survived the validation pipeline."""

    alpha_id: str
    family: str | None = None
    expression: str
    rationale: str
    metrics: AlphaMetrics
    fingerprint: AlphaFingerprint
    decision: str
    refinement_log: str | None = None
    mutation_paths: list[str] = []
    # v2 fields
    ast_topology: str | None = None
    ast_collision: list[str] = []
    simulation_payload: dict | None = None
    # BRAIN submission result
    brain: BrainSubmissionResult | None = None
    pipeline_status: PipelineStatus = PipelineStatus.GENERATED
    exploration_direction: str | None = None
    template_id: str | None = None
    family_id: str | None = None
    hierarchical_reward: float | None = None
    hierarchical_level: str | None = None
    semantic_alignment_score: float | None = None
    economic_rationale: str | None = None
    margin_efficiency: float | None = None
    cycle_num: int
    passed: bool = False
    exp_card_rule_ids: list[str] = []
    exp_card_rule_ids2: list[str] = []
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SessionState(BaseModel):
    """Full session state — serialised to / deserialised from sessions/{id}.json."""

    id: str
    status: SessionStatus = SessionStatus.IDLE
    cycle: int = 0
    focus_area: str = ""
    passed_alphas: list[AlphaResult] = []
    fingerprint_memory: list[dict] = []  # raw dicts for JSON round-trip simplicity
    family_run_tracker: list[str] = []  # last N families generated
    rejected_motifs: list[dict] = []  # fingerprints that caused REJECT
    conversation_history: list[dict] = []  # [{role, content}] for LLM context
    mutation_count: int = 0  # mutations on the current alpha
    brain_mutation_count: int = 0  # BRAIN improvement attempts on current alpha
    stop_requested: bool = False
    last_decision: str | None = None
    consecutive_same_decision: int = 0
    error_message: str | None = None
    # v2 POMDP belief-state components
    topology_map: dict = {}  # {topology_hash: "PASSED"|"FAILED"|"CROWDED"}
    dataset_usage: dict = {}  # {family_name: count}
    failure_catalog: list[dict] = []  # [{fingerprint, failure_type, mutation_tried}]
    open_frontiers: list[dict] = []  # unexplored 5-dim fingerprint combos
    # v3: live activity log for UI display
    activity_log: list[dict] = []  # [{time, type, message, detail}]
    current_brain_alpha_id: str | None = None  # BRAIN alpha being improved
    hallucination_log: list[dict] = Field(default_factory=list)
    pipeline_queue: list[str] = []
    brain_slots: dict = {}
    trajectories: list[dict] = []  # AlphaTrajectory snapshots from MultiAgent
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class StartSessionRequest(BaseModel):
    focus_area: str = "auto"


class ValidationResult(BaseModel):
    """Returned by every validator function."""

    passed: bool
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    # v2: exact fitness formula result
    fitness_computed: float | None = None
    fitness_breakdown: str | None = None
