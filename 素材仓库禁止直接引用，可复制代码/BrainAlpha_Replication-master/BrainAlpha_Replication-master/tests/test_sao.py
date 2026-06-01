import unittest
from typing import Any, Dict, List, Optional

from alpha_agent.sao import SignalAlphaObject, _safe_float


def _make_minimal_candidate() -> Any:
    class FakeCandidate:
        def __init__(self) -> None:
            self.expression = "rank(close)"
            self.settings: Dict[str, Any] = {"neutralization": "SECTOR"}
            self.family = "value"
            self.idea_name = "value_short_cot"
            self.stage = "grid_seed"
            self.priority = 50.0
            self.parent_key = None
            self.metadata: Dict[str, Any] = {}

        def signature(self) -> str:
            return "test_sig"

        def normalized_settings(self) -> Dict[str, Any]:
            return self.settings

        def to_record(self) -> Dict[str, Any]:
            return {
                "expression": self.expression,
                "settings": self.settings,
                "family": self.family,
                "idea_name": self.idea_name,
                "stage": self.stage,
                "priority": self.priority,
                "parent_key": self.parent_key,
                "metadata": self.metadata,
            }

    return FakeCandidate()


class TestSignalAlphaObject(unittest.TestCase):
    def test_default_creation(self) -> None:
        sao = SignalAlphaObject()
        self.assertEqual(sao.status, "pending")
        self.assertEqual(sao.neutralization, "SECTOR")
        self.assertEqual(sao.direction, "long_short")
        self.assertEqual(len(sao.id), 10)

    def test_full_initialization(self) -> None:
        sao = SignalAlphaObject(
            source_cell_id="value_cross_sectional_short_raw",
            source_paper="Fama-French 1992",
            statement="High book-to-market outperforms",
            category="value",
            horizon="short",
            primary_fields=["bookvalue_ps", "close"],
            neutralization="INDUSTRY",
            lookback_min=20,
            lookback_max=60,
            operator_family="cross_sectional",
            direction="long",
            expression="rank(bookvalue_ps)",
            idea_name="value_short_rank",
            family="value",
            diversity_score=0.85,
        )
        self.assertEqual(sao.source_cell_id, "value_cross_sectional_short_raw")
        self.assertEqual(sao.statement, "High book-to-market outperforms")
        self.assertEqual(sao.expression, "rank(bookvalue_ps)")
        self.assertEqual(sao.diversity_score, 0.85)

    def test_to_candidate(self) -> None:
        sao = SignalAlphaObject(
            source_cell_id="momentum_ts_medium_raw",
            source_paper="Jegadeesh-Titman 1993",
            statement="Momentum effect",
            category="momentum",
            horizon="medium",
            primary_fields=["close", "returns"],
            expression="ts_mean(rank(close), 20)",
            idea_name="momentum_medium_ts",
            family="momentum",
            diversity_score=0.92,
            sharpe=1.34,
            fitness=1.02,
            m4_eligible=True,
        )
        candidate = sao.to_candidate()
        self.assertEqual(candidate.expression, "ts_mean(rank(close), 20)")
        self.assertEqual(candidate.family, "momentum")
        self.assertEqual(candidate.idea_name, "momentum_medium_ts")
        self.assertIsNotNone(candidate.metadata)
        self.assertEqual(candidate.metadata.get("sao_id"), sao.id)
        self.assertEqual(candidate.metadata.get("source_paper"), "Jegadeesh-Titman 1993")
        self.assertEqual(candidate.metadata.get("m4_eligible"), True)

    def test_from_candidate(self) -> None:
        candidate = _make_minimal_candidate()
        candidate.metadata = {
            "source_cell_id": "value_cross_sectional_short_raw",
            "source_paper": "Fama-French 1992",
            "statement": "Value premium",
            "horizon": "short",
            "complexity": 5,
            "diversity_score": 0.75,
            "failure_mode": "FM-1",
            "m4_eligible": True,
            "repair_iterations": 2,
            "stop_reason": "sharpe threshold met",
        }
        sao = SignalAlphaObject.from_candidate(
            candidate,
            source_cell_id="override_cell",
            source_paper="override_paper",
            statement="override_statement",
        )
        self.assertEqual(sao.source_cell_id, "override_cell")
        self.assertEqual(sao.source_paper, "override_paper")
        self.assertEqual(sao.statement, "override_statement")
        self.assertEqual(sao.expression, "rank(close)")
        self.assertEqual(sao.family, "value")
        self.assertEqual(sao.failure_mode, "FM-1")
        self.assertEqual(sao.m4_eligible, True)
        self.assertEqual(sao.repair_iterations, 2)
        self.assertEqual(sao.stop_reason, "sharpe threshold met")

    def test_from_candidate_without_overrides(self) -> None:
        candidate = _make_minimal_candidate()
        candidate.metadata = {
            "source_cell_id": "mom_ts_short_raw",
            "source_paper": "Jegadeesh-Titman 1993",
            "statement": "Momentum",
            "horizon": "short",
            "complexity": 3,
        }
        sao = SignalAlphaObject.from_candidate(candidate)
        self.assertEqual(sao.source_cell_id, "mom_ts_short_raw")
        self.assertEqual(sao.source_paper, "Jegadeesh-Titman 1993")
        self.assertEqual(sao.statement, "Momentum")
        self.assertEqual(sao.horizon, "short")

    def test_to_record(self) -> None:
        sao = SignalAlphaObject(
            expression="rank(close)",
            family="momentum",
            idea_name="mom_rank",
            sharpe=1.5,
            fitness=1.1,
            failed_checks=["LOW_SHARPE"],
            status="ok",
        )
        record = sao.to_record()
        self.assertEqual(record["expression"], "rank(close)")
        self.assertEqual(record["sao_id"], sao.id)
        self.assertEqual(record["sharpe"], 1.5)
        self.assertEqual(record["fitness"], 1.1)
        self.assertEqual(record["failed_checks"], ["LOW_SHARPE"])
        self.assertEqual(record["status"], "ok")

    def test_from_record(self) -> None:
        record: Dict[str, Any] = {
            "alpha_id": "A-12345",
            "expression": "zscore(rank(close))",
            "family": "value",
            "idea_name": "value_zscore",
            "settings": {"neutralization": "INDUSTRY"},
            "metrics": {"sharpe": 2.1, "fitness": 1.5, "turnover": 0.3},
            "status": "ok",
            "failed_checks": [],
            "failed_blocking_checks": [],
            "failed_correlation_checks": [],
            "metadata": {
                "source_cell_id": "value_cs_long_raw",
                "source_paper": "Fama 1992",
                "statement": "Value premium in CS",
                "horizon": "long",
                "complexity": 4,
                "diversity_score": 0.6,
                "failure_mode": None,
                "m4_eligible": False,
                "repair_iterations": 0,
            },
        }
        sao = SignalAlphaObject.from_record(record)
        self.assertEqual(sao.alpha_id, "A-12345")
        self.assertEqual(sao.expression, "zscore(rank(close))")
        self.assertEqual(sao.family, "value")
        self.assertEqual(sao.sharpe, 2.1)
        self.assertEqual(sao.fitness, 1.5)
        self.assertEqual(sao.turnover, 0.3)
        self.assertEqual(sao.source_cell_id, "value_cs_long_raw")
        self.assertEqual(sao.horizon, "long")

    def test_from_record_with_overrides(self) -> None:
        record: Dict[str, Any] = {
            "alpha_id": "A-999",
            "expression": "rank(volume)",
            "family": "liquidity",
            "idea_name": "liq_rank",
            "settings": {},
            "metrics": {"sharpe": 0.8},
            "status": "pending",
            "failed_checks": ["HIGH_TURNOVER"],
            "failed_blocking_checks": [],
            "failed_correlation_checks": [],
        }
        sao = SignalAlphaObject.from_record(
            record,
            source_cell_id="liq_cs_short_raw",
            source_paper="Amihud 2002",
            statement="Liquidity premium",
        )
        self.assertEqual(sao.source_cell_id, "liq_cs_short_raw")
        self.assertEqual(sao.source_paper, "Amihud 2002")
        self.assertEqual(sao.statement, "Liquidity premium")
        self.assertEqual(sao.failed_checks, ["HIGH_TURNOVER"])

    def test_compact(self) -> None:
        sao = SignalAlphaObject(
            expression="rank(close)",
            family="value",
            sharpe=1.2,
            status="ok",
            m4_eligible=True,
        )
        compact = sao.compact()
        self.assertEqual(compact["sao_id"], sao.id)
        self.assertEqual(compact["family"], "value")
        self.assertEqual(compact["sharpe"], 1.2)
        self.assertEqual(compact["status"], "ok")
        self.assertEqual(compact["m4_eligible"], True)

    def test_stage_label_repair(self) -> None:
        sao = SignalAlphaObject(
            repair_iterations=3,
            source_cell_id="test_cell",
        )
        self.assertIn("repair_v3", sao._stage_label())
        self.assertIn("grid", sao._stage_label())

    def test_stage_label_seed(self) -> None:
        sao = SignalAlphaObject()
        self.assertEqual(sao._stage_label(), "grid_seed")

    def test_default_idea_name(self) -> None:
        sao = SignalAlphaObject(
            family="value",
            horizon="short",
            operator_family="cross_sectional",
            direction="long",
        )
        name = sao._default_idea_name()
        self.assertEqual(name, "value.short.cross_sectional.long")

    def test_default_idea_name_partial(self) -> None:
        sao = SignalAlphaObject(family="momentum", direction="long_short")
        name = sao._default_idea_name()
        self.assertEqual(name, "momentum.long_short")

    def test_compute_priority_with_sharpe(self) -> None:
        sao = SignalAlphaObject(sharpe=1.5, diversity_score=0.8, fitness=1.2)
        priority = sao._compute_priority()
        expected = 10.0 + (1.5 * 100.0) + (0.8 * 50.0) + (1.2 * 80.0)
        self.assertAlmostEqual(priority, expected)

    def test_compute_priority_default(self) -> None:
        sao = SignalAlphaObject()
        self.assertAlmostEqual(sao._compute_priority(), 10.0)

    def test_safe_float_valid(self) -> None:
        self.assertAlmostEqual(_safe_float(3.14), 3.14)
        self.assertAlmostEqual(_safe_float("2.5"), 2.5)
        self.assertAlmostEqual(_safe_float(0), 0.0)

    def test_safe_float_none(self) -> None:
        self.assertIsNone(_safe_float(None))

    def test_safe_float_invalid(self) -> None:
        self.assertIsNone(_safe_float("not_a_number"))

    def test_round_trip_to_candidate_and_back(self) -> None:
        original = SignalAlphaObject(
            source_cell_id="round_trip_cell",
            source_paper="Test Paper 2026",
            statement="Round trip test hypothesis",
            category="test",
            horizon="medium",
            primary_fields=["field_a", "field_b"],
            expression="rank(field_a) + rank(field_b)",
            idea_name="test_composite",
            family="test",
            diversity_score=0.5,
            m4_eligible=True,
            repair_iterations=1,
            stop_reason="converged",
        )
        candidate = original.to_candidate()
        restored = SignalAlphaObject.from_candidate(candidate)
        self.assertEqual(restored.expression, original.expression)
        self.assertEqual(restored.family, original.family)
        self.assertEqual(restored.diversity_score, original.diversity_score)
        self.assertEqual(restored.m4_eligible, original.m4_eligible)
        self.assertEqual(restored.repair_iterations, original.repair_iterations)
        self.assertEqual(restored.stop_reason, original.stop_reason)

    def test_to_record_round_trip(self) -> None:
        original = SignalAlphaObject(
            source_cell_id="rt_cell",
            source_paper="RT Paper",
            statement="RT hypothesis",
            category="test",
            horizon="long",
            expression="ts_mean(rank(close), 20)",
            family="test",
            sharpe=2.0,
            fitness=1.8,
            turnover=0.15,
            failed_checks=["LOW_TURNOVER"],
            status="ok",
            m4_eligible=False,
        )
        record = original.to_record()
        restored = SignalAlphaObject.from_record(record)
        self.assertEqual(restored.expression, original.expression)
        self.assertEqual(restored.family, original.family)
        self.assertEqual(restored.failed_checks, original.failed_checks)
        self.assertEqual(restored.status, original.status)


if __name__ == "__main__":
    unittest.main()
