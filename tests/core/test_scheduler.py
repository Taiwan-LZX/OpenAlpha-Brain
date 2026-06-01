from openalpha_brain.core.scheduler import ExplorationScheduler
from openalpha_brain.learning.mab import _TEMPLATE_DIRECTION_MAP, TemplateFamilyBandit


def _make_scheduler_with_arms() -> ExplorationScheduler:
    bandit = TemplateFamilyBandit()
    template_ids = list(_TEMPLATE_DIRECTION_MAP.keys())[:3]
    family_ids = ["price_trend", "volume_liquidity"]
    bandit.initialize_arms_from_library(template_ids, family_ids)
    sched = ExplorationScheduler(template_bandit=bandit)
    sched._initialized = True
    return sched


class TestSchedulerInitialize:
    def test_scheduler_initialize(self):
        sched = ExplorationScheduler()
        sched.initialize(
            template_ids=list(_TEMPLATE_DIRECTION_MAP.keys())[:3],
            family_ids=["price_trend", "volume_liquidity"],
        )
        assert sched._initialized is True
        assert sched.bandit.arm_count > 0


class TestSchedulerSelectExplorationArm:
    def test_select_exploration_arm(self):
        sched = _make_scheduler_with_arms()
        result = sched.select_exploration_arm()
        assert result is not None
        assert "direction" in result
        assert "template_id" in result
        assert "family_id" in result


class TestSchedulerSelectDiverseDirections:
    def test_select_diverse_directions(self):
        sched = _make_scheduler_with_arms()
        results = sched.select_diverse_directions(num=3)
        assert isinstance(results, list)
        assert len(results) <= 3
        if len(results) >= 2:
            directions = [r.get("direction", "") for r in results]
            unique = set(directions)
            assert len(unique) >= 1


class TestSchedulerRecordResult:
    def test_record_result(self):
        sched = _make_scheduler_with_arms()
        stats_before = sched.bandit.get_stats()
        sched.record_result(
            template_id=list(_TEMPLATE_DIRECTION_MAP.keys())[0],
            family_id="price_trend",
            direction="momentum",
            reward=1.0,
        )
        stats_after = sched.bandit.get_stats()
        first_key = f"{list(_TEMPLATE_DIRECTION_MAP.keys())[0]}::price_trend"
        assert stats_after[first_key]["visits"] > stats_before[first_key]["visits"]
        assert "ucb_score" in stats_after[first_key]


class TestSchedulerRecordDirectionResult:
    def test_record_direction_result(self):
        sched = _make_scheduler_with_arms()
        dir_stats_before = sched.get_direction_stats()
        sched.record_direction_result("momentum", reward=1.0)
        dir_stats_after = sched.get_direction_stats()
        if "momentum" in dir_stats_after:
            assert dir_stats_after["momentum"]["total_visits"] >= dir_stats_before.get("momentum", {}).get(
                "total_visits", 0
            )


class TestSchedulerGetDirectionStats:
    def test_get_direction_stats(self):
        sched = _make_scheduler_with_arms()
        stats = sched.get_direction_stats()
        assert isinstance(stats, dict)
        if stats:
            first_dir = next(iter(stats.values()))
            assert "avg_expectation" in first_dir
            assert "total_visits" in first_dir
            assert "arm_count" in first_dir


class TestSchedulerHealthCheck:
    def test_health_check(self):
        sched = _make_scheduler_with_arms()
        hc = sched.health_check()
        assert hc["scheduler"] == "active"
        assert hc["initialized"] is True
        assert "bandit" in hc
        assert "feature_map" in hc
        assert "field_proxy_map_loaded" in hc


class TestSchedulerSerialization:
    def test_to_dict_from_dict_roundtrip(self):
        sched = _make_scheduler_with_arms()
        sched.record_result(
            template_id=list(_TEMPLATE_DIRECTION_MAP.keys())[0],
            family_id="price_trend",
            direction="momentum",
            reward=0.5,
        )
        d = sched.to_dict()
        assert "bandit" in d
        assert "initialized" in d

        restored = ExplorationScheduler.from_dict(d)
        assert restored._initialized == sched._initialized
        assert restored.bandit.arm_count == sched.bandit.arm_count
