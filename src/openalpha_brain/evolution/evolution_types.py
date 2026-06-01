from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field


@dataclass
class AlphaTrajectory:
    hypothesis_direction: str
    hypothesis_mechanism: str
    expression_versions: list[str] = dataclass_field(default_factory=list)
    brain_feedbacks: list[dict] = dataclass_field(default_factory=list)
    decision_points: list[dict] = dataclass_field(default_factory=list)
    final_status: str = "PENDING"
    final_sharpe: float | None = None

    def add_decision(self, decision_type: str, chosen: str, alternatives: list[str]) -> None:
        self.decision_points.append(
            {
                "type": decision_type,
                "chosen": chosen,
                "alternatives": alternatives,
            }
        )

    def add_expression_version(self, expression: str) -> None:
        self.expression_versions.append(expression)

    def add_brain_feedback(self, feedback: dict) -> None:
        self.brain_feedbacks.append(feedback)
