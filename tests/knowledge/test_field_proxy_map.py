import json
import pytest

from openalpha_brain.knowledge.field_proxy_map import (
    FieldProxyMap,
    FieldFamily,
    FIELD_FAMILIES,
)


def _make_loaded_fpm() -> FieldProxyMap:
    fpm = FieldProxyMap()
    fpm._fields = [
        {"id": "close", "description": "closing price", "family_id": "price_trend", "l1_category": "price", "coverage": 0.95, "user_count": 100, "alpha_count": 50, "is_cold": False},
        {"id": "volume", "description": "trading volume", "family_id": "volume_liquidity", "l1_category": "price", "coverage": 0.90, "user_count": 80, "alpha_count": 30, "is_cold": False},
        {"id": "cold_field_x", "description": "low coverage field", "family_id": "price_trend", "l1_category": "price", "coverage": 0.1, "user_count": 1, "alpha_count": 0, "is_cold": True},
        {"id": "roe", "description": "return on equity", "family_id": "profitability", "l1_category": "fundamental", "coverage": 0.85, "user_count": 60, "alpha_count": 20, "is_cold": False},
    ]
    fpm._field_map = {f["id"]: f for f in fpm._fields}
    fpm._family_map = {
        "price_trend": ["close", "cold_field_x"],
        "volume_liquidity": ["volume"],
        "profitability": ["roe"],
    }
    fpm._cold_fields = {"cold_field_x"}
    fpm._loaded = True
    return fpm


class TestFieldProxyMapLoad:
    def test_field_proxy_map_load(self):
        fpm = _make_loaded_fpm()
        assert fpm.field_count > 0
        assert fpm.is_ready is True


class TestGetFieldFamily:
    def test_get_field_family(self):
        fpm = _make_loaded_fpm()
        family = fpm.get_field_family("close")
        assert family is not None
        assert family.family_id == "price_trend"

    def test_get_field_family_unknown(self):
        fpm = _make_loaded_fpm()
        family = fpm.get_field_family("nonexistent_field")
        assert family is None


class TestGetFamilies:
    def test_get_families(self):
        fpm = _make_loaded_fpm()
        families = fpm.get_families()
        assert isinstance(families, list)
        assert len(families) > 0
        assert all(isinstance(f, FieldFamily) for f in families)


class TestRecommendFieldsForTemplate:
    def test_recommend_fields_for_template(self):
        fpm = _make_loaded_fpm()
        recommended = fpm.recommend_fields_for_template(
            template_id="momentum_short_term_reversal",
            top_k=10,
            exclude_cold=True,
        )
        assert isinstance(recommended, list)
        assert "close" in recommended
        assert "cold_field_x" not in recommended

    def test_recommend_fields_with_family(self):
        fpm = _make_loaded_fpm()
        recommended = fpm.recommend_fields_for_template(
            template_id="momentum_short_term_reversal",
            family_id="price_trend",
            top_k=10,
            exclude_cold=True,
        )
        assert isinstance(recommended, list)
        assert "close" in recommended


class TestSearchFieldsSync:
    def test_search_fields_sync(self):
        fpm = _make_loaded_fpm()
        results = fpm.search_fields_sync("close", top_k=5)
        assert isinstance(results, list)
        assert len(results) > 0
        field_ids = [r[0] for r in results]
        assert "close" in field_ids


class TestKeywordSearchFallback:
    def test_keyword_search_fallback(self):
        fpm = _make_loaded_fpm()
        results = fpm._keyword_search_fallback("volume", top_k=5)
        assert isinstance(results, list)
        if results:
            field_ids = [r[0] for r in results]
            assert "volume" in field_ids

    def test_keyword_search_fallback_description_match(self):
        fpm = _make_loaded_fpm()
        results = fpm._keyword_search_fallback("equity", top_k=5)
        assert isinstance(results, list)
        field_ids = [r[0] for r in results]
        assert "roe" in field_ids
