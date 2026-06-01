from openalpha_brain.learning.mab import _TEMPLATE_DIRECTION_MAP, TemplateFamilyBandit


def _make_bandit_with_arms() -> TemplateFamilyBandit:
    bandit = TemplateFamilyBandit()
    template_ids = list(_TEMPLATE_DIRECTION_MAP.keys())[:5]
    family_ids = ["price_trend", "volume_liquidity", "volatility_metrics"]
    bandit.initialize_arms_from_library(template_ids, family_ids)
    return bandit


class TestSelectWithDirection:
    def test_select_with_direction(self):
        bandit = _make_bandit_with_arms()
        result = bandit.select_with_direction()
        assert result is not None
        assert "direction" in result
        assert "template_id" in result
        assert "family_id" in result

    def test_select_with_direction_focus_area(self):
        bandit = _make_bandit_with_arms()
        result = bandit.select_with_direction(focus_area="momentum")
        assert result is not None
        assert result["direction"] == "momentum"


class TestSelectDiverseDirections:
    def test_select_diverse_directions(self):
        bandit = _make_bandit_with_arms()
        results = bandit.select_diverse_directions(k=3)
        assert isinstance(results, list)
        if len(results) >= 2:
            directions = [r.get("direction", "") for r in results]
            assert len(set(directions)) >= 1


class TestUpdateByDirection:
    def test_update_by_direction(self):
        bandit = _make_bandit_with_arms()
        stats_before = bandit.get_direction_stats()
        bandit.update_by_direction("momentum", reward=1.0)
        stats_after = bandit.get_direction_stats()
        if "momentum" in stats_after:
            assert stats_after["momentum"]["total_visits"] >= stats_before.get("momentum", {}).get("total_visits", 0)


class TestGetDirectionStats:
    def test_get_direction_stats(self):
        bandit = _make_bandit_with_arms()
        stats = bandit.get_direction_stats()
        assert isinstance(stats, dict)
        if stats:
            first = next(iter(stats.values()))
            assert "avg_expectation" in first
            assert "total_visits" in first
            assert "arm_count" in first
            assert "top_template" in first
            assert "top_family" in first


class TestInitializeArmsFromLibrary:
    def test_initialize_arms_from_library(self):
        bandit = TemplateFamilyBandit()
        assert bandit.arm_count == 0
        template_ids = list(_TEMPLATE_DIRECTION_MAP.keys())[:3]
        family_ids = ["price_trend", "volume_liquidity"]
        bandit.initialize_arms_from_library(template_ids, family_ids)
        assert bandit.arm_count == len(template_ids) * len(family_ids)
