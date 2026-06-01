import pytest

from openalpha_brain.learning.experience_distiller import ExperienceCard, ExperienceDistiller


class _StubReflectionEngine:
    def __init__(self, patterns):
        self._patterns = patterns

    def get_failure_patterns(self):
        return self._patterns


class _StubFailureLib:
    def __init__(self, fixes):
        self._fixes = fixes
        self.search_calls = []

    async def search_fix(self, pattern, top_k=5):
        self.search_calls.append((pattern, top_k))
        return self._fixes


class _StubLogic:
    def __init__(self, logic_id, category, evidence_count, evidence_records):
        self.logic_id = logic_id
        self.category = category
        self.evidence_count = evidence_count
        self.evidence_records = evidence_records


class _StubAlphaLogicLib:
    def __init__(self, logics):
        self._logics = logics

    def all_logics(self):
        return self._logics


class TestDistillFromFailures:
    @pytest.mark.asyncio
    async def test_creates_cards_from_recurring_patterns(self, tmp_path):
        reflection_engine = _StubReflectionEngine(
            {
                "signal_too_weak": 5,
                "high_turnover": 2,
            }
        )

        failure_lib = _StubFailureLib(
            [
                {"fix_attempt": "use ts_zscore instead of ts_delta", "fix_success": True, "direction": "momentum"},
                {"fix_attempt": "add ts_decay_linear", "fix_success": True, "direction": "momentum"},
            ]
        )

        distiller = ExperienceDistiller(path=str(tmp_path / "cards.json"))
        cards = await distiller.distill_from_failures(reflection_engine, failure_lib, min_occurrences=3)

        assert len(cards) >= 1
        assert cards[0].failure_pattern == "signal_too_weak"
        assert cards[0].confidence > 0.0
        assert len(failure_lib.search_calls) == 1
        assert failure_lib.search_calls[0] == ("signal_too_weak", 5)

    @pytest.mark.asyncio
    async def test_skips_patterns_below_min_occurrences(self, tmp_path):
        reflection_engine = _StubReflectionEngine({"rare_pattern": 1})

        failure_lib = _StubFailureLib(
            [
                {"fix_attempt": "some fix", "fix_success": True, "direction": ""},
            ]
        )

        distiller = ExperienceDistiller(path=str(tmp_path / "cards.json"))
        cards = await distiller.distill_from_failures(reflection_engine, failure_lib, min_occurrences=3)

        assert len(cards) == 0
        assert len(failure_lib.search_calls) == 0

    @pytest.mark.asyncio
    async def test_skips_when_no_successful_fixes(self, tmp_path):
        reflection_engine = _StubReflectionEngine({"bad_pattern": 4})

        failure_lib = _StubFailureLib(
            [
                {"fix_attempt": "failed fix", "fix_success": False, "direction": ""},
            ]
        )

        distiller = ExperienceDistiller(path=str(tmp_path / "cards.json"))
        cards = await distiller.distill_from_failures(reflection_engine, failure_lib, min_occurrences=3)

        assert len(cards) == 0


class TestDistillFromEvidence:
    @pytest.mark.asyncio
    async def test_creates_cards_from_logics_with_sufficient_evidence(self, tmp_path):
        logic = _StubLogic(
            logic_id="mom_001",
            category="momentum",
            evidence_count=8,
            evidence_records=[
                {"expression": "ts_zscore(close, 10)", "fix_success": True},
                {"expression": "ts_zscore(close, 20)", "fix_success": True},
                {"expression": "ts_delta(close, 5)", "fix_success": False},
            ],
        )

        alpha_logic_lib = _StubAlphaLogicLib([logic])

        distiller = ExperienceDistiller(path=str(tmp_path / "cards.json"))
        cards = await distiller.distill_from_evidence(alpha_logic_lib, min_evidence=5)

        assert len(cards) >= 1
        assert "mom_001" in cards[0].failure_pattern
        assert cards[0].confidence > 0.0

    @pytest.mark.asyncio
    async def test_skips_logics_below_min_evidence(self, tmp_path):
        logic = _StubLogic(
            logic_id="low_001",
            category="value",
            evidence_count=2,
            evidence_records=[],
        )

        alpha_logic_lib = _StubAlphaLogicLib([logic])

        distiller = ExperienceDistiller(path=str(tmp_path / "cards.json"))
        cards = await distiller.distill_from_evidence(alpha_logic_lib, min_evidence=5)

        assert len(cards) == 0


class TestGetApplicableCards:
    @pytest.mark.asyncio
    async def test_returns_matching_cards(self, tmp_path):
        distiller = ExperienceDistiller(path=str(tmp_path / "cards.json"))
        distiller._cards = [
            ExperienceCard(
                failure_pattern="LOW_SHARPE+momentum", fix_strategy="use ts_zscore", confidence=0.8, usage_count=2
            ),
            ExperienceCard(failure_pattern="HIGH_TURNOVER", fix_strategy="add decay", confidence=0.6, usage_count=0),
        ]

        results = await distiller.get_applicable_cards("low_sharpe failure in momentum")
        assert len(results) >= 1
        assert results[0].failure_pattern == "LOW_SHARPE+momentum"

    @pytest.mark.asyncio
    async def test_returns_empty_for_empty_context(self, tmp_path):
        distiller = ExperienceDistiller(path=str(tmp_path / "cards.json"))
        distiller._cards = [
            ExperienceCard(failure_pattern="LOW_SHARPE", fix_strategy="use ts_zscore", confidence=0.8),
        ]

        assert await distiller.get_applicable_cards("") == []

    @pytest.mark.asyncio
    async def test_respects_top_k(self, tmp_path):
        distiller = ExperienceDistiller(path=str(tmp_path / "cards.json"))
        distiller._cards = [
            ExperienceCard(failure_pattern="LOW_SHARPE+momentum", fix_strategy="fix1", confidence=0.9, usage_count=5),
            ExperienceCard(failure_pattern="LOW_SHARPE+value", fix_strategy="fix2", confidence=0.7, usage_count=1),
            ExperienceCard(failure_pattern="HIGH_TURNOVER", fix_strategy="fix3", confidence=0.5, usage_count=0),
        ]

        results = await distiller.get_applicable_cards("low_sharpe", top_k=2)
        assert len(results) == 2


class TestRecordCardUsage:
    def test_updates_confidence_and_usage_on_success(self, tmp_path):
        distiller = ExperienceDistiller(path=str(tmp_path / "cards.json"))
        card = ExperienceCard(
            failure_pattern="test", fix_strategy="fix", confidence=0.5, usage_count=0, success_count=0
        )
        distiller._cards = [card]

        distiller.record_card_usage(card.rule_id, success=True)

        assert card.usage_count == 1
        assert card.success_count == 1
        assert card.confidence == 1.0

    def test_increments_usage_on_failure_without_success(self, tmp_path):
        distiller = ExperienceDistiller(path=str(tmp_path / "cards.json"))
        card = ExperienceCard(
            failure_pattern="test", fix_strategy="fix", confidence=0.5, usage_count=0, success_count=0
        )
        distiller._cards = [card]

        distiller.record_card_usage(card.rule_id, success=False)

        assert card.usage_count == 1
        assert card.success_count == 0
        assert card.confidence == 0.5

    def test_noop_for_unknown_rule_id(self, tmp_path):
        distiller = ExperienceDistiller(path=str(tmp_path / "cards.json"))
        distiller.record_card_usage("nonexistent", success=True)
        assert len(distiller._cards) == 0


class TestExperienceDistillerPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "cards.json")
        distiller = ExperienceDistiller(path=path)
        distiller._cards = [
            ExperienceCard(failure_pattern="p1", fix_strategy="fix1", confidence=0.7, usage_count=3, success_count=2),
        ]
        distiller._save()

        distiller2 = ExperienceDistiller(path=path)
        assert len(distiller2._cards) == 1
        assert distiller2._cards[0].failure_pattern == "p1"
        assert distiller2._cards[0].confidence == 0.7
        assert distiller2._cards[0].usage_count == 3
