"""
OpenAlpha-Brain — FieldProxyMap (字段代理图谱)

v3 模板约束范式转移的核心基础设施。三层标注: L1语义类别 → L2字段族 → L3适用模板。

策略: BRAIN 原生 category/subcategory/dataset 直接映射 + 关键词语义拆分 +
冷字段(coverage<0.3)标记，不需要 LLM 逐个标注 7000+ 字段。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from openalpha_brain.cli.algo_monitor import AlgoMonitor
from openalpha_brain.data import get_data_path
from openalpha_brain.knowledge.vector_index import VectorStore
from openalpha_brain.services.llm_client import embed
from openalpha_brain.utils.algo_logger import Timer, algo_log

logger = logging.getLogger(__name__)
_monitor = AlgoMonitor.get_instance()

FIELDS_PATH = get_data_path("brain_datafields.json")
VEC_STORE_DIR = get_data_path("vec_store")
PROXY_MAP_PATH = get_data_path("field_proxy_map.json")

COLD_COVERAGE_THRESHOLD = 0.3
COLD_USERCOUNT_THRESHOLD = 5


# ── L1 语义类别 (8) ─────────────────────────────────────────────────────────
L1_SEMANTIC_CATEGORIES: dict[str, dict[str, Any]] = {
    "price": {
        "name": "量价",
        "description": "价格、成交量、波动率等市场微观数据",
        "template_directions": ["momentum", "mean_reversion", "volatility", "lead_lag"],
    },
    "fundamental": {
        "name": "基本面",
        "description": "财务报表数据、估值比率、盈利质量",
        "template_directions": ["value", "quality", "momentum"],
    },
    "sentiment": {
        "name": "情绪/预期",
        "description": "分析师预期、新闻情绪、社交媒体",
        "template_directions": ["momentum", "mean_reversion", "lead_lag"],
    },
    "derived_factor": {
        "name": "衍生因子模型",
        "description": "Analysts' Factor Model 的技术因子",
        "template_directions": ["momentum", "value", "quality", "volatility", "liquidity"],
    },
    "microstructure": {
        "name": "微观结构",
        "description": "期权隐含信息、做空数据、信用风险",
        "template_directions": ["lead_lag", "volatility", "mean_reversion"],
    },
    "alternative": {
        "name": "另类数据",
        "description": "供应链、ESG、卫星、网络流量等非传统数据",
        "template_directions": ["lead_lag", "momentum", "value"],
    },
    "macro": {
        "name": "宏观/系统",
        "description": "宏观经济指标、利率、汇率暴露",
        "template_directions": ["value", "momentum", "lead_lag"],
    },
    "industry_sector": {
        "name": "行业/板块",
        "description": "行业特定字段",
        "template_directions": ["momentum", "value"],
    },
}


# ── L2 字段族 (~30) ─────────────────────────────────────────────────────────
@dataclass
class FieldFamily:
    family_id: str
    family_name: str
    l1_category: str
    description: str
    typical_keywords: list[str] = field(default_factory=list)
    applicable_templates: list[str] = field(default_factory=list)
    data_freshness: str = "daily"
    coverage_risk: str = "medium"
    field_ids: list[str] = field(default_factory=list)
    field_count: int = 0
    avg_coverage: float = 0.0
    cold_ratio: float = 0.0


FIELD_FAMILIES: dict[str, FieldFamily] = {
    # ── price ──
    "price_trend": FieldFamily(
        family_id="price_trend",
        family_name="价格趋势",
        l1_category="price",
        description="价格趋势、动量、回报率信号：close/open/high/low/vwap/returns",
        typical_keywords=["close", "open", "high", "low", "vwap", "return", "price", "trend", "momentum_factor"],
        applicable_templates=["momentum_short_term_reversal", "momentum_medium_term_continuation", "momentum_long_term_reversal", "mean_reversion_zscore", "mean_reversion_bollinger"],
        data_freshness="daily",
    ),
    "volume_liquidity": FieldFamily(
        family_id="volume_liquidity",
        family_name="成交量/流动性",
        l1_category="price",
        description="成交量、换手率、流动性指标",
        typical_keywords=["volume", "turnover", "liquidity", "adv", "trading"],
        applicable_templates=["momentum_volume_confirmed", "liquidity_premium", "liquidity_improvement_signal"],
    ),
    "volatility_metrics": FieldFamily(
        family_id="volatility_metrics",
        family_name="波动率/离散度",
        l1_category="price",
        description="波动率、标准差、价格范围等离散度指标",
        typical_keywords=["volatility", "std_dev", "variance", "range", "beta", "dispersion", "cv"],
        applicable_templates=["volatility_low_vol_anomaly", "volatility_change_signal", "volatility_clustering", "mean_reversion_bollinger"],
        data_freshness="daily",
    ),
    # ── fundamental ──
    "profitability": FieldFamily(
        family_id="profitability",
        family_name="盈利能力",
        l1_category="fundamental",
        description="ROE/ROA/利润率/毛利率等盈利能力指标",
        typical_keywords=["roe", "roa", "margin", "profit", "eps", "income", "net_income", "operating_income", "npm"],
        applicable_templates=["quality_earnings_stability", "value_earnings_quality"],
        data_freshness="quarterly",
    ),
    "valuation": FieldFamily(
        family_id="valuation",
        family_name="估值比率",
        l1_category="fundamental",
        description="PE/PB/PS/EV/EBITDA 等估值倍数",
        typical_keywords=["pe", "pb", "ps", "ev", "ebitda", "dividend_yield", "yield", "earning_yield", "valuefactor"],
        applicable_templates=["value_regression", "size_small_cap_premium", "mean_reversion_valuation"],
        data_freshness="quarterly",
    ),
    "growth_rates": FieldFamily(
        family_id="growth_rates",
        family_name="增长率",
        l1_category="fundamental",
        description="营收增长/EPS增长/历史增长率/预期增长率",
        typical_keywords=["growth", "chg", "yoy", "qoq", "historicalgrowthfactor", "sfyg", "lfyg", "gr"],
        applicable_templates=["momentum_medium_term_continuation", "value_earnings_quality"],
        data_freshness="quarterly",
    ),
    "balance_sheet": FieldFamily(
        family_id="balance_sheet",
        family_name="资产负债",
        l1_category="fundamental",
        description="资产/负债/权益/杠杆比率",
        typical_keywords=["asset", "liability", "equity", "debt", "leverage", "tangible", "current_ratio", "quick_ratio"],
        applicable_templates=["quality_asset_turnover", "size_small_cap_premium"],
        data_freshness="quarterly",
    ),
    "cash_flow": FieldFamily(
        family_id="cash_flow",
        family_name="现金流",
        l1_category="fundamental",
        description="经营现金流/自由现金流/资本支出",
        typical_keywords=["cash_flow", "free_cash", "capex", "ocf", "fcf", "operating_cash"],
        applicable_templates=["quality_earnings_stability", "value_regression"],
        data_freshness="quarterly",
    ),
    "footnotes_detail": FieldFamily(
        family_id="footnotes_detail",
        family_name="报表附注明细",
        l1_category="fundamental",
        description="Report Footnotes: 应计项/摊销/减值/养老金等细节数据",
        typical_keywords=["accrued", "amortization", "impairment", "pension", "deferred", "restructuring"],
        applicable_templates=["value_earnings_quality"],
        data_freshness="quarterly",
        coverage_risk="high",
    ),
    # ── sentiment ──
    "analyst_estimates": FieldFamily(
        family_id="analyst_estimates",
        family_name="分析师预期",
        l1_category="sentiment",
        description="分析师EPS预测/评级/目标价/修正方向",
        typical_keywords=["analyst", "estimate", "consensus", "fy1", "fy2", "revision", "recommendation", "target"],
        applicable_templates=["lead_lag_price_volume", "lead_lag_cross_field", "lead_lag_industry_rotation", "momentum_short_term_reversal"],
        data_freshness="daily",
    ),
    "earnings_surprise": FieldFamily(
        family_id="earnings_surprise",
        family_name="盈利惊喜",
        l1_category="sentiment",
        description="实际EPS vs 预期EPS的偏离、盈利公告效应",
        typical_keywords=["surprise", "surstd", "sue", "earnings_release", "abnormal_return", "post_event"],
        applicable_templates=["lead_lag_cross_field", "lead_lag_industry_rotation", "momentum_short_term_reversal"],
        data_freshness="quarterly",
    ),
    "news_headline": FieldFamily(
        family_id="news_headline",
        family_name="新闻流/头条",
        l1_category="sentiment",
        description="新闻数量/头条情绪/事件频率",
        typical_keywords=["news", "headline", "article", "ravenpack", "rpa", "event", "media"],
        applicable_templates=["lead_lag_cross_field", "lead_lag_industry_rotation", "momentum_volume_confirmed"],
        data_freshness="daily",
    ),
    "social_media": FieldFamily(
        family_id="social_media",
        family_name="社交媒体",
        l1_category="sentiment",
        description="Twitter/X/Reddit 等社交媒体情绪/热度",
        typical_keywords=["social", "twitter", "reddit", "buzz", "post", "mention"],
        applicable_templates=["lead_lag_cross_field", "momentum_short_term_reversal"],
        data_freshness="daily",
        coverage_risk="high",
    ),
    "insider_corporate": FieldFamily(
        family_id="insider_corporate",
        family_name="内部人/公司行为",
        l1_category="sentiment",
        description="内部人交易/回购/增发/并购相关信号",
        typical_keywords=["insider", "buyback", "share_repurchase", "ipo", "secondary", "merger", "acquisition"],
        applicable_templates=["lead_lag_cross_field", "momentum_short_term_reversal"],
        data_freshness="daily",
        coverage_risk="high",
    ),
    # ── derived_factor ──
    "momentum_factor_model": FieldFamily(
        family_id="momentum_factor_model",
        family_name="动量模型因子",
        l1_category="derived_factor",
        description="Technical Models中的价格动量/反转因子: pricemomentumfactor, return/trend",
        typical_keywords=["pricemomentumfactor", "return", "trend", "w", "week_return", "month_return", "active_return"],
        applicable_templates=["momentum_short_term_reversal", "momentum_medium_term_continuation", "momentum_long_term_reversal", "mean_reversion_zscore"],
    ),
    "earnings_momentum_model": FieldFamily(
        family_id="earnings_momentum_model",
        family_name="盈利动量模型因子",
        l1_category="derived_factor",
        description="Analysts' Factor Model 盈利动量: earningmomentumfactor, revision, estimate change",
        typical_keywords=["earningmomentumfactor", "revision", "estimate", "epsrm", "spe1yfvc", "earnings_yield"],
        applicable_templates=["value_earnings_quality", "momentum_medium_term_continuation"],
    ),
    "deep_value_model": FieldFamily(
        family_id="deep_value_model",
        family_name="深度价值模型因子",
        l1_category="derived_factor",
        description="Analysts' Factor Model 价值: deepvaluefactor, PE, PB, dividend, sales/price",
        typical_keywords=["deepvaluefactor", "divyield", "ep", "bp", "sp", "cfp", "ttm"],
        applicable_templates=["value_regression", "size_small_cap_premium", "mean_reversion_valuation"],
    ),
    "historical_growth_model": FieldFamily(
        family_id="historical_growth_model",
        family_name="历史增长模型因子",
        l1_category="derived_factor",
        description="Analysts' Factor Model 历史增长: historicalgrowthfactor, chg, growth trajectory",
        typical_keywords=["historicalgrowthfactor", "chg", "growth", "pfcfghc", "sustainable", "ocfp"],
        applicable_templates=["value_earnings_quality", "momentum_medium_term_continuation"],
    ),
    "risk_factor_model": FieldFamily(
        family_id="risk_factor_model",
        family_name="风险模型因子",
        l1_category="derived_factor",
        description="Analysts' Factor Model 风险/质量: liquidityriskfactor, creditrisk, volatility, leverage",
        typical_keywords=["liquidityriskfactor", "creditrisk", "volatility", "leverage", "beta", "mad", "skewness"],
        applicable_templates=["volatility_low_vol_anomaly", "volatility_change_signal", "volatility_clustering"],
    ),
    "global_region_model": FieldFamily(
        family_id="global_region_model",
        family_name="全球/区域模型因子",
        l1_category="derived_factor",
        description="Analysts' Factor Model 全球/区域: globaldev, region-specific factors",
        typical_keywords=["globaldev", "northamerica", "region", "country", "sector"],
        applicable_templates=["momentum_medium_term_continuation", "value_regression", "lead_lag_industry_rotation"],
    ),
    "quality_factor_model": FieldFamily(
        family_id="quality_factor_model",
        family_name="质量因子模型",
        l1_category="derived_factor",
        description="Analysts' Factor Model 质量: profitability, stability, composite quality signals",
        typical_keywords=["quality", "stability", "composite", "score", "fundamental_score"],
        applicable_templates=["quality_earnings_stability", "quality_asset_turnover"],
    ),
    # ── microstructure ──
    "option_implied": FieldFamily(
        family_id="option_implied",
        family_name="期权隐含信息",
        l1_category="microstructure",
        description="期权隐含波动率/看跌看涨比/Greeks",
        typical_keywords=["option", "implied", "volatility", "put", "call", "skew", "delta", "gamma", "vega", "open_interest"],
        applicable_templates=["volatility_low_vol_anomaly", "lead_lag_price_volume"],
        data_freshness="daily",
    ),
    "credit_risk": FieldFamily(
        family_id="credit_risk",
        family_name="信用风险",
        l1_category="microstructure",
        description="信用风险度量/CDS/违约概率/信用评级",
        typical_keywords=["credit", "default", "cds", "rating", "creditworthiness", "distance_to_default"],
        applicable_templates=["value_regression", "quality_earnings_stability"],
        data_freshness="daily",
    ),
    "short_interest": FieldFamily(
        family_id="short_interest",
        family_name="做空/持仓",
        l1_category="microstructure",
        description="做空比例/持仓集中度/机构持仓",
        typical_keywords=["short", "borrow", "lending", "institutional", "holding", "concentration"],
        applicable_templates=["mean_reversion_zscore", "lead_lag_cross_field"],
        data_freshness="daily",
        coverage_risk="high",
    ),
    "relationship_network": FieldFamily(
        family_id="relationship_network",
        family_name="关系网络",
        l1_category="microstructure",
        description="供应链关系/客户依赖/供应商集中度",
        typical_keywords=["relationship", "supply", "chain", "customer", "supplier", "network", "revenue_share"],
        applicable_templates=["lead_lag_cross_field", "value_regression"],
        data_freshness="quarterly",
    ),
    # ── alternative ──
    "alternative_geospatial": FieldFamily(
        family_id="alternative_geospatial",
        family_name="卫星/地理空间",
        l1_category="alternative",
        description="卫星图像/停车场/航运/地理空间数据",
        typical_keywords=["satellite", "geo", "parking", "shipping", "vessel", "cargo"],
        applicable_templates=["lead_lag_cross_field", "momentum_short_term_reversal"],
        data_freshness="daily",
        coverage_risk="high",
    ),
    "alternative_esg": FieldFamily(
        family_id="alternative_esg",
        family_name="ESG/可持续",
        l1_category="alternative",
        description="ESG评分/碳排放/可持续发展指标",
        typical_keywords=["esg", "carbon", "emission", "sustainability", "environment", "governance", "social"],
        applicable_templates=["value_regression", "quality_earnings_stability"],
        data_freshness="quarterly",
        coverage_risk="high",
    ),
    "alternative_web_traffic": FieldFamily(
        family_id="alternative_web_traffic",
        family_name="网络/APP流量",
        l1_category="alternative",
        description="网站流量/APP使用/信用卡收据等消费者行为数据",
        typical_keywords=["web", "traffic", "app", "credit_card", "receipt", "consumer", "app_usage"],
        applicable_templates=["lead_lag_cross_field", "momentum_short_term_reversal"],
        data_freshness="daily",
        coverage_risk="high",
    ),
    "alternative_supply_chain": FieldFamily(
        family_id="alternative_supply_chain",
        family_name="另类供应链",
        l1_category="alternative",
        description="非传统供应链数据: 航运/物流/订单积压等",
        typical_keywords=["supply_chain", "shipment", "order", "backlog", "inventory", "logistics"],
        applicable_templates=["lead_lag_cross_field", "momentum_short_term_reversal"],
        data_freshness="daily",
        coverage_risk="high",
    ),
}


# ── BRAIN subcategory/category → field family 映射规则 ────────────────────────

def _subcategory_to_family(subcat_name: str) -> str | None:
    return {
        "Price Volume": "price_trend",
        "Option Volatility": "volatility_metrics",
        "Option Analytics": "option_implied",
        "Fundamental Data": None,  # 按描述关键词拆分
        "Footnotes": "footnotes_detail",
        "Valuation Models": "valuation",
        "Analyst Estimates": "analyst_estimates",
        "News": "news_headline",
        "News Sentiment": "news_headline",
        "Social Media": "social_media",
        "Sentiment": "social_media",
        "Relationship": "relationship_network",
        "Risk Based Models": "credit_risk",
        "Risk Models": "risk_factor_model",
    }.get(subcat_name)


def _dataset_to_family(dataset_name: str) -> str | None:
    return {
        "Price Volume Data for Equity": "price_trend",
        "Volatility Data": "volatility_metrics",
        "Options Analytics": "option_implied",
        "Company Fundamental Data for Equity": None,  # 按描述拆分
        "Report Footnotes": "footnotes_detail",
        "Fundamental Scores": "quality_factor_model",
        "Analyst Estimate Data for Equity": "analyst_estimates",
        "US News Data": "news_headline",
        "Ravenpack News Data": "news_headline",
        "Research Sentiment Data": "social_media",
        "Sentiment Data for Equity": "social_media",
        "Social Media Data for Equity": "social_media",
        "Relationship Data for Equity": "relationship_network",
        "Creditworthiness Risk Measure Model": "credit_risk",
        "Systematic Risk Metrics": "risk_factor_model",
    }.get(dataset_name)


# ── Technical Models (Analysts' Factor Model) 关键词→字段族规则 ──────────────
_TECH_MODEL_KEYWORD_RULES: list[tuple[str, str]] = [
    ("pricemomentumfactor", "momentum_factor_model"),
    ("earningmomentumfactor", "earnings_momentum_model"),
    ("deepvaluefactor", "deep_value_model"),
    ("historicalgrowthfactor", "historical_growth_model"),
    ("liquidityriskfactor", "risk_factor_model"),
    ("creditrisk", "risk_factor_model"),
    ("globaldev", "global_region_model"),
    ("earning_yield", "deep_value_model"),
    ("earnings_yield", "deep_value_model"),
    ("dividend", "deep_value_model"),
    ("valuefactor", "deep_value_model"),
    ("growthfactor", "historical_growth_model"),
    ("qualityfactor", "quality_factor_model"),
    ("profitability", "quality_factor_model"),
    ("momentum", "momentum_factor_model"),
    ("return", "momentum_factor_model"),
    ("trend", "momentum_factor_model"),
    ("revision", "earnings_momentum_model"),
    ("estimate", "earnings_momentum_model"),
    ("volatility", "risk_factor_model"),
    ("leverage", "risk_factor_model"),
    ("beta", "risk_factor_model"),
    ("skewness", "risk_factor_model"),
    ("kurtosis", "risk_factor_model"),
    ("liquidity", "risk_factor_model"),
    ("sustainable_growth", "historical_growth_model"),
    ("margin", "quality_factor_model"),
    ("pe", "deep_value_model"),
    ("pb", "deep_value_model"),
    ("ps", "deep_value_model"),
    ("active_return", "momentum_factor_model"),
    ("rsi", "momentum_factor_model"),
    ("macd", "momentum_factor_model"),
    ("surstd", "earnings_momentum_model"),
    ("eps", "earnings_momentum_model"),
    ("region", "global_region_model"),
    ("northamerica", "global_region_model"),
    ("sector", "global_region_model"),
]


def _classify_tech_model(field_id: str, description: str) -> str:
    combined = (field_id + " " + (description or "")).lower()
    for keyword, family in _TECH_MODEL_KEYWORD_RULES:
        if keyword in combined:
            return family
    return "momentum_factor_model"


# ── Fundamental Data 关键词→字段族规则 ──────────────────────────────────────
_FUND_KEYWORD_RULES: list[tuple[str, str]] = [
    ("roe", "profitability"),
    ("roa", "profitability"),
    ("margin", "profitability"),
    ("profit", "profitability"),
    ("eps", "profitability"),
    ("net_income", "profitability"),
    ("operating_income", "profitability"),
    ("ebitda", "profitability"),
    ("pe_ratio", "valuation"),
    ("pb_ratio", "valuation"),
    ("ps_ratio", "valuation"),
    ("ev_", "valuation"),
    ("dividend", "valuation"),
    ("yield", "valuation"),
    ("price_to", "valuation"),
    ("value", "valuation"),
    ("growth", "growth_rates"),
    ("chg", "growth_rates"),
    ("yoy", "growth_rates"),
    ("qoq", "growth_rates"),
    ("asset", "balance_sheet"),
    ("liability", "balance_sheet"),
    ("equity", "balance_sheet"),
    ("debt", "balance_sheet"),
    ("leverage", "balance_sheet"),
    ("tangible", "balance_sheet"),
    ("current_ratio", "balance_sheet"),
    ("quick_ratio", "balance_sheet"),
    ("cash", "cash_flow"),
    ("cash_flow", "cash_flow"),
    ("free_cash", "cash_flow"),
    ("capex", "cash_flow"),
    ("ocf", "cash_flow"),
    ("fcf", "cash_flow"),
    ("operating_cash", "cash_flow"),
    ("accrued", "footnotes_detail"),
    ("amortization", "footnotes_detail"),
    ("impairment", "footnotes_detail"),
    ("pension", "footnotes_detail"),
    ("deferred", "footnotes_detail"),
    ("restructuring", "footnotes_detail"),
    ("goodwill", "footnotes_detail"),
    ("intangible", "footnotes_detail"),
]


def _classify_fundamental(field_id: str, description: str) -> str:
    combined = (field_id + " " + (description or "")).lower()
    for keyword, family in _FUND_KEYWORD_RULES:
        if keyword in combined:
            return family
    return "balance_sheet"


class FieldProxyMap:
    def __init__(self) -> None:
        self._fields: list[dict[str, Any]] = []
        self._field_map: dict[str, dict[str, Any]] = {}
        self._family_map: dict[str, list[str]] = {}  # family_id → [field_id]
        self._cold_fields: set[str] = set()
        self._vector_store: VectorStore | None = None
        self._loaded = False

    @algo_log()
    def load(self, force_rebuild: bool = False) -> None:
        if self._loaded and not force_rebuild:
            return

        # 尝试加载缓存的代理图谱
        if PROXY_MAP_PATH.exists() and not force_rebuild:
            try:
                self._load_from_cache()
                logger.info("FieldProxyMap loaded from cache: %d fields, %d families, %d cold",
                            len(self._fields), len(self._family_map), len(self._cold_fields))
                self._loaded = True
                return
            except OSError:
                logger.warning("Cache load failed, rebuilding...")

        with Timer("field_proxy_map_build"):
            self._build_from_raw()
        self._loaded = True

    def _load_from_cache(self) -> None:
        data = json.loads(PROXY_MAP_PATH.read_text(encoding="utf-8"))
        self._fields = data.get("fields", [])
        self._field_map = {f["id"]: f for f in self._fields}
        self._family_map = data.get("families", {})
        self._cold_fields = set(data.get("cold_fields", []))
        self._try_load_vector_store()

    def _build_from_raw(self) -> None:
        if not FIELDS_PATH.exists():
            logger.error("brain_datafields.json not found at %s", FIELDS_PATH)
            return

        raw_fields: list[dict] = json.loads(FIELDS_PATH.read_text(encoding="utf-8"))
        logger.info("Building FieldProxyMap from %d raw fields...", len(raw_fields))

        self._family_map = {fid: [] for fid in FIELD_FAMILIES}

        for f in raw_fields:
            fid = f.get("id", "")
            description = f.get("description", "") or ""
            category_name = f.get("category", {}).get("name", "")
            subcat_name = f.get("subcategory", {}).get("name", "")
            dataset_name = f.get("dataset", {}).get("name", "")
            coverage = f.get("coverage", 0.0) or 0.0
            user_count = f.get("userCount", 0) or 0
            alpha_count = f.get("alphaCount", 0) or 0

            family_id = self._resolve_family(fid, description, subcat_name, dataset_name, category_name)
            if family_id is None:
                continue

            l1 = FIELD_FAMILIES[family_id].l1_category

            is_cold = coverage < COLD_COVERAGE_THRESHOLD or user_count < COLD_USERCOUNT_THRESHOLD
            if is_cold:
                self._cold_fields.add(fid)

            field_entry = {
                "id": fid,
                "description": description,
                "family_id": family_id,
                "l1_category": l1,
                "coverage": coverage,
                "user_count": user_count,
                "alpha_count": alpha_count,
                "is_cold": is_cold,
            }
            self._fields.append(field_entry)
            self._field_map[fid] = field_entry
            self._family_map[family_id].append(fid)

        for fam in FIELD_FAMILIES.values():
            fam.field_ids = self._family_map.get(fam.family_id, [])
            fam.field_count = len(fam.field_ids)
            if fam.field_count > 0:
                covs = [self._field_map[fid]["coverage"] for fid in fam.field_ids if fid in self._field_map]
                fam.avg_coverage = sum(covs) / len(covs) if covs else 0.0
                cold_count = sum(1 for fid in fam.field_ids if fid in self._cold_fields)
                fam.cold_ratio = cold_count / fam.field_count

        self._save_cache()
        self._try_load_vector_store()

        total = len(self._fields)
        cold_count = len(self._cold_fields)
        logger.info("FieldProxyMap built: %d fields, %d families, %d cold (%.1f%%)",
                    total, len(self._family_map), cold_count, cold_count / max(total, 1) * 100)

    def _resolve_family(
        self,
        field_id: str,
        description: str,
        subcat_name: str,
        dataset_name: str,
        category_name: str,
    ) -> str | None:
        # 1. 直接 subcategory 映射
        family = _subcategory_to_family(subcat_name)
        if family:
            return family

        # 2. 直接 dataset 映射
        family = _dataset_to_family(dataset_name)
        if family:
            return family

        # 3. Technical Models (Analysts' Factor Model) 关键词分类
        if subcat_name == "Technical Models" or dataset_name == "Analysts' Factor Model":
            return _classify_tech_model(field_id, description)

        # 4. Fundamental Data 关键词分类
        if category_name == "Fundamental" or dataset_name == "Company Fundamental Data for Equity":
            return _classify_fundamental(field_id, description)

        # 5. 默认到 price_trend / 未知归类到 price
        if category_name == "Price Volume":
            return "price_trend"

        return "momentum_factor_model"

    def _save_cache(self) -> None:
        data = {
            "fields": self._fields,
            "families": self._family_map,
            "cold_fields": list(self._cold_fields),
        }
        PROXY_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROXY_MAP_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("FieldProxyMap cache saved → %s", PROXY_MAP_PATH)

    def _try_load_vector_store(self) -> None:
        vec_path = VEC_STORE_DIR / "vec_fields.json"
        if vec_path.exists():
            try:
                self._vector_store = VectorStore.load_index(vec_path)
            except OSError:
                logger.warning("Failed to load vec_fields.json")

    # ── 查询 API ──────────────────────────────────────────────────────────────

    def get_field_family(self, field_id: str) -> FieldFamily | None:
        entry = self._field_map.get(field_id)
        if entry is None:
            return None
        return FIELD_FAMILIES.get(entry["family_id"])

    def get_field_info(self, field_id: str) -> dict[str, Any] | None:
        return self._field_map.get(field_id)

    def get_families(self) -> list[FieldFamily]:
        return list(FIELD_FAMILIES.values())

    def get_family(self, family_id: str) -> FieldFamily | None:
        return FIELD_FAMILIES.get(family_id)

    def get_fields_in_family(self, family_id: str, exclude_cold: bool = False) -> list[str]:
        ids = self._family_map.get(family_id, [])
        if exclude_cold:
            ids = [fid for fid in ids if fid not in self._cold_fields]
        return ids

    def get_cold_fields(self) -> set[str]:
        return self._cold_fields

    def get_family_stats(self) -> dict[str, dict[str, Any]]:
        stats = {}
        for fid, fam in FIELD_FAMILIES.items():
            field_ids = self._family_map.get(fid, [])
            stats[fid] = {
                "name": fam.family_name,
                "l1": fam.l1_category,
                "field_count": len(field_ids),
                "avg_coverage": fam.avg_coverage,
                "cold_ratio": fam.cold_ratio,
                "applicable_templates": fam.applicable_templates,
            }
        return stats

    def recommend_fields_for_template(
        self,
        template_id: str,
        family_id: str | None = None,
        top_k: int = 20,
        exclude_cold: bool = True,
    ) -> list[str]:
        if family_id:
            field_ids = self.get_fields_in_family(family_id)
        else:
            field_ids = []
            for fid, fam in FIELD_FAMILIES.items():
                if template_id in fam.applicable_templates:
                    field_ids.extend(self._family_map.get(fid, []))

        if exclude_cold:
            field_ids = [f for f in field_ids if f not in self._cold_fields]

        scored: list[tuple[str, float]] = []
        for fid in field_ids:
            entry = self._field_map.get(fid)
            if entry is None:
                continue
            score = entry.get("alpha_count", 0) * 0.01 + entry.get("coverage", 0)
            scored.append((fid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [fid for fid, _ in scored[:top_k]]

    async def search_fields(self, query: str, top_k: int = 20) -> list[tuple[str, float, dict]]:
        if self._vector_store is None:
            return self._keyword_search_fallback(query, top_k)
        try:
            query_vec = np.array(await embed(query), dtype=np.float32)
            results = self._vector_store.query(query_vec, top_k=top_k)
            enriched = []
            for did, sim, meta in results:
                entry = self._field_map.get(did)
                if entry:
                    enriched.append((did, sim, {**meta, "family_id": entry["family_id"], "is_cold": entry["is_cold"]}))
                else:
                    enriched.append((did, sim, meta))
            return enriched
        except (ValueError, TypeError, OSError):
            logger.warning("embed() failed, falling back to keyword search")
            return self._keyword_search_fallback(query, top_k)

    def _keyword_search_fallback(self, query: str, top_k: int = 20) -> list[tuple[str, float, dict]]:
        query_lower = query.lower()
        tokens = query_lower.split()
        scored: list[tuple[str, float, dict]] = []
        for fid, entry in self._field_map.items():
            desc = (entry.get("description") or "").lower()
            fid_lower = fid.lower()
            score = 0.0
            for token in tokens:
                if token in fid_lower:
                    score += 2.0
                if token in desc:
                    score += 1.0
            if score > 0:
                scored.append((fid, score, {"family_id": entry["family_id"], "is_cold": entry["is_cold"]}))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search_fields_sync(self, query: str, top_k: int = 20) -> list[tuple[str, float, dict]]:
        return self._keyword_search_fallback(query, top_k)

    @property
    def field_count(self) -> int:
        return len(self._fields)

    @property
    def family_count(self) -> int:
        return len(self._family_map)

    @property
    def is_ready(self) -> bool:
        return self._loaded


_global_proxy_map: FieldProxyMap | None = None


def get_field_proxy_map() -> FieldProxyMap:
    global _global_proxy_map
    if _global_proxy_map is None:
        _global_proxy_map = FieldProxyMap()
        _global_proxy_map.load()
    return _global_proxy_map
