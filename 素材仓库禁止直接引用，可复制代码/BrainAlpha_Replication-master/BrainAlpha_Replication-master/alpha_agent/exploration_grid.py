from __future__ import annotations

import logging
import random
import zlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from alpha_agent.datasets_loader import EMPTY_META, FieldMetadata

logger = logging.getLogger(__name__)


@dataclass
class Thesis:
    authors: str
    year: int
    title: str
    journal: str
    key_finding_en: str
    key_finding_zh: str

    def short_str(self) -> str:
        return f"{self.authors} ({self.year}), \"{self.title}\", {self.journal}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "authors": self.authors,
            "year": self.year,
            "title": self.title,
            "journal": self.journal,
        }


class CellStatus(Enum):
    EMPTY = "empty"
    ASSIGNED = "assigned"
    EXPLORED = "explored"
    EXHAUSTED = "exhausted"


class DatasetCategory(Enum):
    ANALYST       = ("analyst",       "Analyst estimates, ratings, revisions")
    FUNDAMENTAL   = ("fundamental",   "Financial statement data, ratios, ownership")
    MODEL         = ("model",         "Pre-built alpha factors, ML model outputs")
    NEWS          = ("news",          "News article data, sentiment, headline analysis")
    OPTION        = ("option",        "Options OI, implied volatility, greeks")
    PRICE_VOLUME  = ("price_volume",  "Price, volume, market cap, technical indicators")
    SENTIMENT     = ("sentiment",     "Sentiment scores, NLP, social media sentiment")
    SOCIAL_MEDIA  = ("social_media",  "Social media activity, web search trends")

    @property
    def label(self) -> str:
        return self.value[0]


class OperatorCategory(Enum):
    CROSS_SECTIONAL = ("cross_sectional", "Cross-sectional rank, scale, normalize")
    TIME_SERIES     = ("time_series",     "Time-series mean, delta, rank, std_dev")
    GROUP           = ("group",           "Group rank, zscore, neutralize")

    @property
    def label(self) -> str:
        return self.value[0]


class Horizon(Enum):
    SHORT  = ("short",  1,  10)
    MEDIUM = ("medium", 10, 60)
    LONG   = ("long",   60, 252)

    @property
    def label(self) -> str:
        return self.value[0]

    @property
    def min_days(self) -> int:
        return self.value[1]

    @property
    def max_days(self) -> int:
        return self.value[2]


_THESIS_MAP_CACHE: Dict[str, Any] = {}


def _ensure_thesis_map() -> Dict[str, Any]:
    global _THESIS_MAP_CACHE
    if _THESIS_MAP_CACHE:
        return _THESIS_MAP_CACHE
    _path = Path(__file__).resolve().parent.parent / "data" / "thesis_map.json"
    if not _path.exists():
        logger.warning("thesis_map.json not found at %s", _path)
        _THESIS_MAP_CACHE = {}
        return _THESIS_MAP_CACHE
    import json
    with open(_path, "r", encoding="utf-8") as f:
        _THESIS_MAP_CACHE = json.load(f)
    return _THESIS_MAP_CACHE


def get_thesis(dc: DatasetCategory, oc: OperatorCategory) -> List[Thesis]:
    key = f"{dc.label}/{oc.label}"
    raw = _ensure_thesis_map().get(key, [])
    return [
        Thesis(
            authors=t["authors"],
            year=t["year"],
            title=t["title"],
            journal=t["journal"],
            key_finding_en=t["key_finding_en"],
            key_finding_zh=t["key_finding_zh"],
        )
        for t in raw
    ]


@dataclass
class GridCell:
    dataset_category: DatasetCategory
    operator_category: OperatorCategory
    horizon: Horizon
    status: CellStatus = CellStatus.EMPTY
    explored_count: int = 0
    pass_count: int = 0
    total_count: int = 0
    last_sharpe: Optional[float] = None
    candidate_fields: List[str] = field(default_factory=list)
    thesis: List[Thesis] = field(default_factory=list)

    def cell_id(self) -> str:
        return f"{self.dataset_category.label}_{self.operator_category.label}_{self.horizon.label}"

    def expected_yield(self) -> float:
        if self.total_count == 0:
            return 0.1
        return self.pass_count / max(1, self.total_count)

    def novelty(self) -> float:
        return {
            CellStatus.EMPTY: 1.00,
            CellStatus.ASSIGNED: 0.50,
            CellStatus.EXPLORED: 0.25,
            CellStatus.EXHAUSTED: 0.00,
        }[self.status]

    def priority(self, domain_dc: Optional[DatasetCategory] = None) -> float:
        return cell_priority(self, domain_dc=domain_dc)

    def build_hypothesis(self) -> str:
        return (
            f"Explore {self.operator_category.label} signals using "
            f"{self.dataset_category.label} data "
            f"at {self.horizon.label} horizon "
            f"({self.horizon.min_days}-{self.horizon.max_days} days)."
        )

    def cell_key(self) -> str:
        return self.cell_id()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell_id": self.cell_id(),
            "dataset_category": self.dataset_category.label,
            "operator_category": self.operator_category.label,
            "horizon": self.horizon.label,
            "status": self.status.value,
            "explored_count": self.explored_count,
            "pass_count": self.pass_count,
            "total_count": self.total_count,
            "expected_yield": round(self.expected_yield(), 4),
            "novelty": self.novelty(),
            "priority": round(self.priority(), 4),
            "last_sharpe": self.last_sharpe,
            "candidate_fields": self.candidate_fields,
            "thesis": [t.to_dict() for t in self.thesis],
            "hypothesis": self.build_hypothesis(),
        }

def cell_priority(cell: GridCell, domain_dc: Optional[DatasetCategory] = None) -> float:
    novelty = cell.novelty()
    yield_score = cell.expected_yield()

    alignment = 0.3
    if domain_dc is not None and cell.dataset_category == domain_dc:
        alignment = 1.0

    category_order = {
        DatasetCategory.PRICE_VOLUME: 0, DatasetCategory.FUNDAMENTAL: 1,
        DatasetCategory.MODEL: 2, DatasetCategory.ANALYST: 3,
        DatasetCategory.NEWS: 4, DatasetCategory.OPTION: 5,
        DatasetCategory.SENTIMENT: 6, DatasetCategory.SOCIAL_MEDIA: 7,
    }
    operator_order = {
        OperatorCategory.CROSS_SECTIONAL: 0, OperatorCategory.TIME_SERIES: 1,
        OperatorCategory.GROUP: 2,
    }
    horizon_order = {Horizon.SHORT: 0, Horizon.MEDIUM: 1, Horizon.LONG: 2}
    structural = (
        category_order.get(cell.dataset_category, 4) * 0.01
        + operator_order.get(cell.operator_category, 1) * 0.005
        + horizon_order.get(cell.horizon, 1) * 0.003
    )
    tiebreak = (zlib.adler32(cell.cell_id().encode()) % 1000) / 100000.0
    return 0.5 * novelty + 0.3 * yield_score + 0.2 * alignment + structural + tiebreak


IDEA_LIBRARY_TO_GRID: Dict[Tuple[str, str], Tuple[DatasetCategory, OperatorCategory, Horizon]] = {
    ("social_buzz", "raw"):                 (DatasetCategory.SENTIMENT, OperatorCategory.CROSS_SECTIONAL, Horizon.SHORT),
    ("social_buzz", "mean"):                (DatasetCategory.SENTIMENT, OperatorCategory.TIME_SERIES, Horizon.MEDIUM),
    ("social_buzz", "decay_linear"):        (DatasetCategory.SENTIMENT, OperatorCategory.TIME_SERIES, Horizon.MEDIUM),
    ("social_buzz", "liquidity_gate"):      (DatasetCategory.SENTIMENT, OperatorCategory.TIME_SERIES, Horizon.SHORT),
    ("social_sentiment", "raw"):            (DatasetCategory.SENTIMENT, OperatorCategory.CROSS_SECTIONAL, Horizon.SHORT),
    ("social_sentiment", "mean"):           (DatasetCategory.SENTIMENT, OperatorCategory.TIME_SERIES, Horizon.MEDIUM),
    ("social_sentiment", "decay_linear"):   (DatasetCategory.SENTIMENT, OperatorCategory.TIME_SERIES, Horizon.MEDIUM),
    ("news_attention", "raw"):              (DatasetCategory.NEWS, OperatorCategory.CROSS_SECTIONAL, Horizon.SHORT),
    ("news_attention", "mean"):             (DatasetCategory.NEWS, OperatorCategory.TIME_SERIES, Horizon.SHORT),
    ("news_attention", "decay_linear"):     (DatasetCategory.NEWS, OperatorCategory.TIME_SERIES, Horizon.SHORT),
    ("price_reversion", "close_vs_mean"):   (DatasetCategory.PRICE_VOLUME, OperatorCategory.CROSS_SECTIONAL, Horizon.SHORT),
    ("price_reversion", "vwap_gap"):        (DatasetCategory.PRICE_VOLUME, OperatorCategory.CROSS_SECTIONAL, Horizon.SHORT),
    ("price_reversion", "return_sum"):      (DatasetCategory.PRICE_VOLUME, OperatorCategory.TIME_SERIES, Horizon.SHORT),
    ("fundamental_quality", "rank"):        (DatasetCategory.FUNDAMENTAL, OperatorCategory.CROSS_SECTIONAL, Horizon.LONG),
    ("fundamental_quality", "zscore"):      (DatasetCategory.FUNDAMENTAL, OperatorCategory.CROSS_SECTIONAL, Horizon.LONG),
    ("fundamental_quality", "ts_rank"):     (DatasetCategory.FUNDAMENTAL, OperatorCategory.TIME_SERIES, Horizon.LONG),
    ("social_price_combo", "news_plus_price"):  (DatasetCategory.SENTIMENT, OperatorCategory.CROSS_SECTIONAL, Horizon.MEDIUM),
    ("social_price_combo", "news_plus_vwap"):   (DatasetCategory.SENTIMENT, OperatorCategory.CROSS_SECTIONAL, Horizon.MEDIUM),
    ("social_price_repair", "group_rank_reversion"):      (DatasetCategory.SENTIMENT, OperatorCategory.GROUP, Horizon.MEDIUM),
    ("social_price_repair", "group_neutralized_reversion"): (DatasetCategory.SENTIMENT, OperatorCategory.GROUP, Horizon.MEDIUM),
    ("social_price_repair", "group_zscore_reversion"):    (DatasetCategory.SENTIMENT, OperatorCategory.GROUP, Horizon.MEDIUM),
    ("news_price_combo", "news_plus_price"): (DatasetCategory.NEWS, OperatorCategory.CROSS_SECTIONAL, Horizon.MEDIUM),
    ("news_price_combo", "news_plus_vwap"):  (DatasetCategory.NEWS, OperatorCategory.CROSS_SECTIONAL, Horizon.MEDIUM),
}


class ExplorationGrid:
    GRID_SIZE = 8 * 3 * 3

    @classmethod
    def load_dataset_field_map(cls, path: str | Path) -> Dict[str, List[str]]:
        import json
        with open(path, "r") as f:
            raw = json.load(f)
        for key, fields in raw.items():
            try:
                dc = DatasetCategory[key.upper()]
                cls.DATASET_FIELD_MAP[dc] = fields
            except (KeyError, ValueError):
                logger.warning("Unknown DatasetCategory '%s' in external map, skipping", key)
        return {k.label: v for k, v in cls.DATASET_FIELD_MAP.items()}

    DATASET_FIELD_MAP: Dict[DatasetCategory, List[str]] = {
        DatasetCategory.ANALYST: [
            "actual_eps_value_quarterly", "actual_cashflow_per_share_value_quarterly",
            "actual_sales_value_annual", "actual_eps_value_annual",
            "analyst_recommendation_value", "analyst_price_target_value",
            "analyst_rating_value", "eps_surprise_value",
            "analyst_consensus_value", "analyst_coverage_value",
            "mean_estimate_value", "high_estimate_value", "low_estimate_value",
            "number_of_estimates_value", "standard_deviation_value",
            "price_target_high_value", "price_target_mean_value",
            "price_target_low_value", "revision_value",
            "img_cnn_feature1_us_ibes_1_b2_d1", "img_avg_actual_ebit_us",
            "avg_first_biasfree_fundamental_estimate", "avg_first_biasfree_price_target_estimate",
            "aggregate_large_target_long_horizon_return", "aggregate_prediction_accuracy_score",
            "beat_precision_rate", "consensus_mape_percent",
        ],
        DatasetCategory.FUNDAMENTAL: [
            "bookvalue_ps", "return_equity", "return_assets", "ebit", "ebitda",
            "debt_lt", "debt", "cashflow_op", "cashflow", "fnd6_fopo",
            "pretax_income", "sales", "enterprise_value", "cogs",
            "fnd6_mfma1_capx", "fnd6_mfma1_at", "fnd6_mfma1_csho", "fnd6_mfma1_dp",
            "fnd6_mfma2_oancf", "fnd6_mfma2_revt", "operating_income",
            "assets", "equity", "cash", "depre_amort", "capex", "sales_growth",
            "eps", "income", "revenue", "rd_expense",
            "eps_change_absolute_value", "ern6_1q",
            "aggregate_equity_value_all_owners", "aggregate_equity_value_institutions",
            "aggregate_share_count_all_owners", "aggregate_share_count_institutions",
            "count_institutional_buyers_security", "count_institutional_sellers_security",
            "board_member_count", "count_connected_companies", "count_external_board_connections",
            "mean_connected_company_market_value",
            "insd3_10k_freq_flag", "insd3_10k_freq_latest_filed",
            "insd3_10k_freq_latest_release", "insd3_10q_freq_flag",
            "top1000", "top200", "top2000", "top3000",
            "mcr63_ldi_membership", "mcr63_mdi_membership", "mcr63_sdi_membership",
            "oth83_company_sentiment", "oth83_insider_holdings", "oth83_market_cap",
            "oth250_item_display_price", "oth250_item_rating",
            "oth250_item_review_count", "oth250_rank",
            "oth432_aacr_trkdpitdeltapredict_funda_predict",
            "oth432_acae_a_profitability_profitability1",
            "oth432_aacr_trkdpitdeltapredict_funda_mae",
        ],
        DatasetCategory.MODEL: [
            "analyst_revision_rank_derivative", "cashflow_efficiency_rank_derivative",
            "composite_factor_score_derivative", "earnings_certainty_rank_derivative",
            "fscore_bfl_growth", "analyst_sentiment_rank_derivative",
            "value_score_bfl", "quality_score_bfl", "growth_score_bfl",
            "momentum_score_bfl", "size_score_bfl", "low_volatility_score_bfl",
            "aerospace_defense_score", "airline_industry_score",
            "automobile_manufacturing_score", "apparel_accessories_score",
            "principal_component_features", "principal_component_features_2",
            "commercial_realty_exposure_beta", "residential_realty_exposure_beta",
            "beta_lower_confidence_band", "beta_prediction_dispersion",
            "beta_prediction_uncertainty",
            "earnings_bin_label1", "prob_rank_bin1_10d_img_news",
            "predicted_daily_volume_from_news_image", "predicted_daily_volume_from_ohlcv_image",
            "close_return_quantile_1_1day", "high_return_quantile_1_1day",
            "confidence_five_quantile_prediction", "confidence_hundred_quantile_prediction",
            "imb5_score", "imb5_mktcap",
            "oth432_aacr_trkdpitdeltapredict_funda_mada",
            "oth432_aacr_trkdpitdeltapredict_funda_madp",
        ],
        DatasetCategory.NEWS: [
            "nws18_relevance", "nws18_event_relevance", "nws18_nip", "nws18_ssc",
            "nws18_sse", "news_ratio_vol", "news_session_range_pct",
            "news_pct_30min", "news_pct_60min",
            "nws5_01l", "earnings_news_mention_count",
            "headline_negative_score_value", "headline_positive_score_value",
            "negative_sentiment_average", "negative_sentiment_average_10",
            "negative_sentiment_average_11",
            "negative_sentiment_confidence_lower", "negative_sentiment_confidence_upper",
            "count_negative_phrases", "count_positive_phrases",
            "mean_sentiment_score_article",
            "aggregated_sentiment_value_1",
        ],
        DatasetCategory.OPTION: [
            "opt3_openintecallatm", "opt3_openinteputatm",
            "opt3_impliedvolatilitycallatm", "opt3_impliedvolatilityputatm",
            "atm_call_option_delta_value", "atm_call_option_estimated_forward",
            "atm_call_volatility_1080d_long_term_2", "atm_call_volatility_10d_long_term_2",
            "atm_call_put_iv_spread", "earnings_volatility_risk_premium",
            "implied_earnings_volatility_estimate",
            "daily_option_contracts_traded", "daily_option_contracts_traded_365",
            "broker_dealer_bearish_contracts_noto", "broker_dealer_bearish_trade_count",
            "option_contracts_traded_value", "option_open_interest_value",
            "implied_volatility_value", "put_call_ratio_value",
        ],
        DatasetCategory.PRICE_VOLUME: [
            "close", "open", "high", "low", "volume", "adv20", "cap", "sharesout",
            "returns", "vwap", "sector", "country", "currency",
            "adjfactor", "return_1d", "return_5d", "return_10d", "return_20d",
            "demo_close", "demo_volume", "demo_return_5d", "demo_beta",
            "momentum_shift_indicator", "momentum_strength_index",
            "price_acceleration_factor", "price_acceleration_index",
            "absolute_price_oscillator_10d", "absolute_price_oscillator_50d",
            "accumulation_distribution_10d", "accumulation_distribution_50d",
            "fifth_ask_price_int60", "fifth_bid_price_int60",
            "fifth_high_price_int60", "fifth_low_price_int60",
            "ask_order_deletion_count", "ask_order_execution_count",
            "ask_order_insertion_count_2", "ask_order_modification_count",
            "bid_order_deletion_count", "bid_order_execution_count",
            "shrt2_t12m_volatility_rank", "shrt2_t3m_volatility_rank",
            "star_si_cap_rank", "star_si_country_rank",
            "rsk60_crowding", "rsk60_last", "rsk60_offer", "rsk60_datatime",
        ],
        DatasetCategory.SENTIMENT: [
            "scl12_buzz", "scl12_sentiment", "snt_value", "snt_buzz",
            "snt_buzz_bfl", "snt_buzz_ret", "snt_social_value", "snt_social_volume",
            "scl12_buzz_fast_d1", "scl12_sentiment_fast_d1",
            "snt_buzz_fast_d1", "snt_buzz_bfl_fast_d1",
            "snt_buzz_ret_fast_d1", "snt_value_fast_d1",
            "scl12_buzzvec", "scl12_sentvec", "scl12_typevec",
            "scl12_alltype_buzzvec", "scl12_alltype_sentvec", "scl12_alltype_typevec",
            "snt1_cored1_score", "snt1_d1_analystcoverage", "snt1_d1_buyrecpercent",
            "answer_chunk_count", "assetutilization_negative_score",
            "aggregate_sentiment_score_2", "financial_note_quantity",
            "opinion_score_delta", "opinion_score_numeric", "standardized_opinion_score",
            "aggregated_sentiment_value_1", "aggregated_sentiment_value_10",
            "relative_interest_score", "relative_interest_score_2",
            "search_interest_14d_corporate_name", "search_interest_14d_equity_symbol",
            "search_interest_1y_corporate_name", "search_interest_1y_equity_symbol",
        ],
        DatasetCategory.SOCIAL_MEDIA: [
            "snt_social_value", "snt_social_volume",
            "snt18_followers", "snt18_following_stocks",
            "social_media_sentiment_score", "social_media_volume_score",
            "tweet_count_value", "retweet_count_value",
            "social_buzz_index", "social_engagement_score",
        ],
    }

    _EXTERNAL_MAP_LOADED = False

    def __init__(self, field_metadata: Optional[Dict[str, FieldMetadata]] = None) -> None:
        self.cells: Dict[str, GridCell] = {}
        self.field_metadata: Dict[str, FieldMetadata] = field_metadata or {}
        self._initialize_grid()
        if self.field_metadata:
            self._validate_fields()

    def _initialize_grid(self) -> None:
        for dc in DatasetCategory:
            for oc in OperatorCategory:
                for hz in Horizon:
                    cell = GridCell(
                        dataset_category=dc,
                        operator_category=oc,
                        horizon=hz,
                        candidate_fields=list(self.DATASET_FIELD_MAP.get(dc, [])),
                        thesis=get_thesis(dc, oc),
                    )
                    self.cells[cell.cell_id()] = cell

    def _validate_fields(self) -> None:
        for dc, fields in self.DATASET_FIELD_MAP.items():
            for f in fields:
                meta = self.field_metadata.get(f)
                if meta is None:
                    logger.warning("[%s] field '%s' not found in any CSV", dc.label, f)
                elif meta.coverage == 0:
                    logger.warning("[%s] field '%s' has 0%% coverage", dc.label, f)

    def get_candidate_fields(
        self, dc: DatasetCategory, min_coverage: float = 0
    ) -> List[str]:
        fields = list(self.DATASET_FIELD_MAP.get(dc, []))
        if min_coverage <= 0 or not self.field_metadata:
            return fields
        return [
            f for f in fields
            if self.field_metadata.get(f, EMPTY_META).coverage >= min_coverage
        ]

    def get_cell(self, cell_id: str) -> Optional[GridCell]:
        return self.cells.get(cell_id)

    def select_cells(
        self,
        budget: int,
        domain_dc: Optional[DatasetCategory] = None,
        exclude_exhausted: bool = True,
        shuffle: bool = False,
    ) -> List[GridCell]:
        eligible = [
            cell for cell in self.cells.values()
            if not exclude_exhausted or cell.status != CellStatus.EXHAUSTED
        ]
        if not eligible:
            return []
        eligible.sort(key=lambda cell: cell.priority(domain_dc=domain_dc), reverse=True)
        selected = eligible[:max(1, budget)]
        for cell in selected:
            if cell.status == CellStatus.EMPTY:
                cell.status = CellStatus.ASSIGNED
        if shuffle:
            rng = random.Random(42)
            rng.shuffle(selected)
        return selected

    def mark_result(self, cell_id: str, passed: bool, sharpe: Optional[float] = None) -> None:
        cell = self.cells.get(cell_id)
        if cell is None:
            return
        cell.total_count += 1
        if passed:
            cell.pass_count += 1
        if sharpe is not None:
            cell.last_sharpe = sharpe
        if cell.status == CellStatus.ASSIGNED:
            cell.status = CellStatus.EXPLORED

    def mark_exhausted(self, cell_id: str) -> None:
        cell = self.cells.get(cell_id)
        if cell is not None:
            cell.status = CellStatus.EXHAUSTED

    def populate_from_library(self, library: Dict[str, Any]) -> None:
        for item in library.get("manual_seeds", []):
            if not isinstance(item, dict):
                continue
            family_name = str(item.get("family", ""))
            expression = str(item.get("expression", ""))
            for (lib_family, lib_template), (dc, oc, hz) in IDEA_LIBRARY_TO_GRID.items():
                if lib_family == family_name and lib_template in expression:
                    cell_id = GridCell(
                        dataset_category=dc, operator_category=oc, horizon=hz
                    ).cell_id()
                    cell = self.cells.get(cell_id)
                    if cell and cell.status == CellStatus.EMPTY:
                        cell.status = CellStatus.EXPLORED
                        cell.total_count = 1
                        cell.pass_count = 1

    def unused_cells(self) -> List[GridCell]:
        return [cell for cell in self.cells.values() if cell.status in {
            CellStatus.EMPTY, CellStatus.ASSIGNED,
        }]

    def cell_count_by_status(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for cell in self.cells.values():
            s = cell.status.value
            counts[s] = counts.get(s, 0) + 1
        return counts

    def summary(self) -> Dict[str, Any]:
        by_status = self.cell_count_by_status()
        return {
            "total_cells": len(self.cells),
            "by_status": by_status,
            "exhausted_pct": round(by_status.get("exhausted", 0) / max(1, len(self.cells)) * 100, 1),
            "unused_count": len(self.unused_cells()),
            "top_cells": [
                cell.to_dict()
                for cell in sorted(
                    self.cells.values(),
                    key=lambda c: c.priority(),
                    reverse=True,
                )[:8]
            ],
        }
