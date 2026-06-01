from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import alpha_research_pipeline as pipeline


@dataclass
class SignalAlphaObject:
    """Signal Alpha Object (論文 Table 2)

    Single shared state object flowing through the entire M0→M1→M2→M3→M4 pipeline.
    Carries complete provenance from thesis to Brain-validated alpha.
    """

    # ── Identity (M0) ──
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    source_cell_id: str = ""
    source_paper: str = ""

    # ── Hypothesis (M0) ──
    statement: str = ""
    category: str = ""
    horizon: str = ""

    # ── Specification (M1) ──
    primary_fields: List[str] = field(default_factory=list)
    neutralization: str = "SECTOR"
    lookback_min: int = 1
    lookback_max: int = 20
    operator_family: str = ""
    direction: str = "long_short"

    # ── Expression (M2) ──
    expression: str = ""
    complexity: int = 0
    diversity_score: float = 0.0
    idea_name: str = ""
    family: str = ""

    # ── Simulation (M3) — filled after Brain API call ──
    alpha_id: Optional[str] = None
    sharpe: Optional[float] = None
    fitness: Optional[float] = None
    turnover: Optional[float] = None
    returns: Optional[float] = None
    drawdown: Optional[float] = None
    margin: Optional[float] = None
    checks: List[Dict[str, Any]] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)

    # ── Evaluation (M3) ──
    status: str = "pending"           # pending / ok / error
    failure_mode: Optional[str] = None  # FM-1 .. FM-6
    failed_checks: List[str] = field(default_factory=list)
    failed_blocking: List[str] = field(default_factory=list)
    failed_correlation: List[str] = field(default_factory=list)
    m4_eligible: bool = False

    # ── Repair (M4) ──
    repair_iterations: int = 0
    stop_reason: Optional[str] = None
    expression_history: List[str] = field(default_factory=list)
    sharpe_history: List[float] = field(default_factory=list)
    tried_strategies: List[str] = field(default_factory=list)

    # ── Internal ──
    parent_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ────────────────────────────────
    # Backward-compatible conversions
    # ────────────────────────────────

    def to_candidate(self) -> pipeline.Candidate:
        return pipeline.Candidate(
            expression=self.expression,
            settings=self.settings or {},
            family=self.family or self.category,
            idea_name=self.idea_name or self._default_idea_name(),
            stage=self._stage_label(),
            priority=self._compute_priority(),
            parent_key=self.parent_key,
            metadata={
                "sao_id": self.id,
                "source_cell_id": self.source_cell_id,
                "source_paper": self.source_paper,
                "statement": self.statement,
                "category": self.category,
                "horizon": self.horizon,
                "primary_fields": self.primary_fields,
                "neutralization": self.neutralization,
                "lookback_min": self.lookback_min,
                "lookback_max": self.lookback_max,
                "operator_family": self.operator_family,
                "direction": self.direction,
                "complexity": self.complexity,
                "diversity_score": self.diversity_score,
                "failure_mode": self.failure_mode,
                "m4_eligible": self.m4_eligible,
                "repair_iterations": self.repair_iterations,
                "stop_reason": self.stop_reason,
                **self.metadata,
            },
        )

    @classmethod
    def from_candidate(
        cls,
        candidate: pipeline.Candidate,
        *,
        source_cell_id: str = "",
        source_paper: str = "",
        statement: str = "",
    ) -> SignalAlphaObject:
        meta = candidate.metadata or {}
        return cls(
            source_cell_id=source_cell_id or str(meta.get("source_cell_id", "")),
            source_paper=source_paper or str(meta.get("source_paper", "")),
            statement=statement or str(meta.get("statement", "")),
            category=candidate.family,
            horizon=str(meta.get("horizon", "")),
            primary_fields=meta.get("primary_fields", []),
            neutralization=str(candidate.settings.get("neutralization", "SECTOR")),
            lookback_min=int(meta.get("lookback_min", 1)),
            lookback_max=int(meta.get("lookback_max", 20)),
            operator_family=str(meta.get("operator_family", "")),
            direction=str(meta.get("direction", "long_short")),
            expression=candidate.expression,
            complexity=int(meta.get("complexity", 0)),
            diversity_score=float(meta.get("diversity_score", 0.0)),
            idea_name=candidate.idea_name,
            family=candidate.family,
            settings=candidate.normalized_settings(),
            failure_mode=str(meta.get("failure_mode")) if meta.get("failure_mode") else None,
            m4_eligible=bool(meta.get("m4_eligible", False)),
            repair_iterations=int(meta.get("repair_iterations", 0)),
            stop_reason=str(meta.get("stop_reason")) if meta.get("stop_reason") else None,
            parent_key=candidate.parent_key,
            metadata=meta,
        )

    @classmethod
    def from_record(
        cls,
        record: Dict[str, Any],
        *,
        source_cell_id: str = "",
        source_paper: str = "",
        statement: str = "",
    ) -> SignalAlphaObject:
        meta = record.get("metadata") or {}
        settings = record.get("settings") or {}
        metrics = record.get("metrics") or {}
        failed = record.get("failed_checks") or []
        return cls(
            alpha_id=str(record.get("alpha_id")) if record.get("alpha_id") else None,
            source_cell_id=source_cell_id or str(meta.get("source_cell_id", "")),
            source_paper=source_paper or str(meta.get("source_paper", "")),
            statement=statement or str(meta.get("statement", "")),
            category=str(record.get("family", meta.get("category", ""))),
            horizon=str(meta.get("horizon", "")),
            primary_fields=meta.get("primary_fields", []),
            neutralization=str(settings.get("neutralization", meta.get("neutralization", "SECTOR"))),
            lookback_min=int(meta.get("lookback_min", 1)),
            lookback_max=int(meta.get("lookback_max", 20)),
            operator_family=str(meta.get("operator_family", "")),
            direction=str(meta.get("direction", "long_short")),
            expression=str(record.get("expression", meta.get("expression", ""))),
            complexity=int(meta.get("complexity", 0)),
            diversity_score=float(meta.get("diversity_score", 0.0)),
            idea_name=str(record.get("idea_name", "")),
            family=str(record.get("family", "")),
            sharpe=_safe_float(metrics.get("sharpe")),
            fitness=_safe_float(metrics.get("fitness")),
            turnover=_safe_float(metrics.get("turnover")),
            returns=_safe_float(metrics.get("returns")),
            drawdown=_safe_float(metrics.get("drawdown")),
            margin=_safe_float(metrics.get("margin")),
            checks=record.get("summary", record.get("checks", [])),
            settings=settings,
            status=str(record.get("status", "pending")),
            failure_mode=str(meta.get("failure_mode")) if meta.get("failure_mode") else None,
            failed_checks=[str(n) for n in failed if isinstance(n, str)],
            failed_blocking=[str(n) for n in (record.get("failed_blocking_checks") or [])],
            failed_correlation=[str(n) for n in (record.get("failed_correlation_checks") or [])],
            m4_eligible=bool(meta.get("m4_eligible", False)),
            repair_iterations=int(meta.get("repair_iterations", 0)),
            stop_reason=str(meta.get("stop_reason")) if meta.get("stop_reason") else None,
            expression_history=meta.get("expression_history", []),
            sharpe_history=meta.get("sharpe_history", []),
            tried_strategies=meta.get("tried_strategies", []),
            parent_key=record.get("parent_key", record.get("candidate_key")),
            metadata=meta,
        )

    def to_record(self) -> Dict[str, Any]:
        candidate = self.to_candidate()
        base = candidate.to_record()
        base["sao_id"] = self.id
        base["source_cell_id"] = self.source_cell_id
        base["source_paper"] = self.source_paper
        base["statement"] = self.statement
        base["alpha_id"] = self.alpha_id
        base["sharpe"] = self.sharpe
        base["fitness"] = self.fitness
        base["turnover"] = self.turnover
        base["returns"] = self.returns
        base["drawdown"] = self.drawdown
        base["margin"] = self.margin
        base["failed_checks"] = self.failed_checks
        base["failed_blocking_checks"] = self.failed_blocking
        base["failed_correlation_checks"] = self.failed_correlation
        base["failure_mode"] = self.failure_mode
        base["m4_eligible"] = self.m4_eligible
        base["repair_iterations"] = self.repair_iterations
        base["stop_reason"] = self.stop_reason
        base["expression_history"] = self.expression_history
        base["sharpe_history"] = self.sharpe_history
        base["tried_strategies"] = self.tried_strategies
        base["status"] = self.status
        base["score"] = self.sharpe
        return base

    def compact(self) -> Dict[str, Any]:
        return {
            "sao_id": self.id,
            "alpha_id": self.alpha_id,
            "family": self.family or self.category,
            "idea_name": self.idea_name or self._default_idea_name(),
            "source_cell_id": self.source_cell_id,
            "source_paper": self.source_paper,
            "statement": self.statement,
            "expression": self.expression,
            "sharpe": self.sharpe,
            "fitness": self.fitness,
            "status": self.status,
            "failure_mode": self.failure_mode,
            "m4_eligible": self.m4_eligible,
            "repair_iterations": self.repair_iterations,
            "stop_reason": self.stop_reason,
            "failed_checks": self.failed_checks,
        }

    # ────────────────────────────────
    # Helpers
    # ────────────────────────────────

    def _default_idea_name(self) -> str:
        parts = [self.family or self.category, self.horizon, self.operator_family, self.direction]
        return ".".join(p for p in parts if p)

    def _stage_label(self) -> str:
        parts = []
        if self.repair_iterations > 0:
            parts.append(f"repair_v{self.repair_iterations}")
        if self.source_cell_id:
            parts.append("grid")
        return "_".join(parts) if parts else "grid_seed"

    def _compute_priority(self) -> float:
        score = 10.0
        if self.sharpe is not None:
            score += self.sharpe * 100.0
        if self.diversity_score:
            score += self.diversity_score * 50.0
        if self.fitness is not None:
            score += self.fitness * 80.0
        return score


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
