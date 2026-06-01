"""
OpenAlpha-Brain — AlphaLogics Market Logic Library

Implements the AlphaLogics (2026) concept: reverse-extract market logics from
public factor libraries (Alpha101/191/158/360), build a structured, iteratively
optimizable logic library that guides alpha factor generation.

Core concepts:
  - Market Logic = verifiable market hypothesis + factor templates
  - Logic Library = structured collection of all known market logics
  - Logic-guided factor generation = new factors must derive from a logic
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from openalpha_brain.cli.algo_monitor import AlgoMonitor
from openalpha_brain.data import get_data_path
from openalpha_brain.utils.algo_logger import algo_log, Timer, log_call
from openalpha_brain.validation.ast_validator import ASTValidator

_monitor = AlgoMonitor.get_instance()

_LIBRARY_PATH = get_data_path("market_logics.json")

_DIRECTION_CATEGORY_MAP: dict[str, list[str]] = {
    "momentum": ["momentum"],
    "mean_reversion": ["momentum", "mean_reversion"],
    "volatility": ["volatility"],
    "value": ["value"],
    "quality": ["quality"],
    "liquidity": ["liquidity"],
    "size": ["size"],
    "lead_lag": ["lead_lag", "momentum"],
    "industry_rotation": ["lead_lag", "momentum", "value"],
    "temporal": ["momentum", "volatility", "mean_reversion"],
    "cross_sectional": ["momentum", "value", "quality"],
    "interaction": ["momentum", "value", "quality", "volatility", "liquidity", "size", "lead_lag"],
}


class BlockType(Enum):
    SIGNAL = "A"
    NEUTRALIZE = "B"
    DECAY = "C"


@dataclass
class TemplateBlock:
    block_type: BlockType
    template_str: str
    locked: bool = False
    editable_params: list[str] = field(default_factory=list)


@dataclass
class ThreeBlockTemplate:
    template_id: str
    name: str
    category: str
    block_a: TemplateBlock
    block_b: TemplateBlock
    block_c: TemplateBlock

    @algo_log()
    def assemble(self, **kwargs) -> str:
        signal_kwargs = {k: v for k, v in kwargs.items() if k in self.block_a.editable_params}
        signal_expr = self.block_a.template_str
        for key, value in signal_kwargs.items():
            signal_expr = signal_expr.replace("{" + key + "}", str(value))

        neutralize_kwargs = {k: v for k, v in kwargs.items() if k in self.block_b.editable_params}
        neutralize_expr = self.block_b.template_str.replace("{signal}", signal_expr)
        for key, value in neutralize_kwargs.items():
            neutralize_expr = neutralize_expr.replace("{" + key + "}", str(value))

        decay_kwargs = {k: v for k, v in kwargs.items() if k in self.block_c.editable_params}
        decay_expr = self.block_c.template_str.replace("{neutralized}", neutralize_expr)
        for key, value in decay_kwargs.items():
            decay_expr = decay_expr.replace("{" + key + "}", str(value))

        return decay_expr

    @algo_log()
    def validate_assembly(self, expr: str) -> bool:
        validator = ASTValidator()
        result = validator.validate(expr)

        if result.errors:
            logger.warning(
                "ThreeBlockTemplate.validate_assembly: [DEFENSIVE_LOG] AST 校验发现致命错误: %s | expr=%s",
                result.errors, expr[:100],
            )
            return False

        has_neutralize = result.structure_info.get("has_neutralize", False)
        has_decay = result.structure_info.get("has_decay", False)
        is_valid_three_block = result.structure_info.get("is_valid_three_block", False)

        if not (has_neutralize and has_decay):
            logger.info(
                "ThreeBlockTemplate.validate_assembly: 三段式结构不完整 | "
                "has_neutralize=%s | has_decay=%s | is_valid_three_block=%s | warnings=%s",
                has_neutralize, has_decay, is_valid_three_block,
                result.warnings[:3] if result.warnings else [],
            )
            return False

        if result.warnings:
            for w in result.warnings:
                logger.debug(
                    "ThreeBlockTemplate.validate_assembly: [DEFENSIVE_LOG] AST 警告: %s", w
                )

        log_call(
            "ThreeBlockTemplate.AST_validate",
            input={"expr": expr[:80]},
            output={
                "passed": True,
                "has_neutralize": has_neutralize,
                "has_decay": has_decay,
                "is_valid_three_block": is_valid_three_block,
                "nesting_depth": result.structure_info.get("nesting_depth"),
                "ops_used": result.structure_info.get("operators_used"),
                "warnings_count": len(result.warnings),
            },
            level=logging.DEBUG,
        )

        return True


@dataclass
class EvidenceRecord:
    expression: str = ""
    sharpe: float | None = None
    fitness: float | None = None
    turnover: float | None = None
    direction: str = ""
    failure_type: str = ""
    fix_attempt: str = ""
    fix_success: bool = False
    timestamp: float = 0.0


@dataclass
class MarketLogic:
    logic_id: str
    category: str
    hypothesis: str
    mechanism: str
    factor_templates: list[str]
    time_horizon: str
    evidence_count: int = 0
    last_updated: float = 0.0
    evidence_records: list[dict] = field(default_factory=list)
    accumulated_diagnoses: list[dict] = field(default_factory=list)

    def instantiate(self, fields: dict[str, str]) -> str:
        result = self.factor_templates[0]
        for key, value in fields.items():
            result = result.replace("{" + key + "}", value)
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MarketLogic:
        return cls(
            logic_id=d["logic_id"],
            category=d["category"],
            hypothesis=d["hypothesis"],
            mechanism=d["mechanism"],
            factor_templates=d["factor_templates"],
            time_horizon=d["time_horizon"],
            evidence_count=d.get("evidence_count", 0),
            last_updated=d.get("last_updated", 0.0),
            evidence_records=d.get("evidence_records", []),
            accumulated_diagnoses=d.get("accumulated_diagnoses", []),
        )


def _build_default_logics() -> dict[str, MarketLogic]:
    logics: dict[str, MarketLogic] = {}

    # === Momentum ===
    logics["momentum_short_term_reversal"] = MarketLogic(
        logic_id="momentum_short_term_reversal",
        category="momentum",
        hypothesis="Short-term winners reverse: stocks with high recent returns underperform going forward",
        mechanism="Overreaction and profit-taking by short-term traders create mean-reversion pressure",
        factor_templates=[
            "-rank(ts_delta({price_field}, {short_lb}))",
            "-rank(ts_returns({price_field}, {short_lb}))",
            "-ts_zscore(ts_delta({price_field}, {short_lb}), {zscore_lb})",
        ],
        time_horizon="short",
    )
    logics["momentum_medium_term_continuation"] = MarketLogic(
        logic_id="momentum_medium_term_continuation",
        category="momentum",
        hypothesis="Medium-term trends persist: stocks with sustained momentum continue to outperform",
        mechanism="Gradual information diffusion and herding behavior sustain price trends",
        factor_templates=[
            "rank(ts_delta({price_field}, {medium_lb}))",
            "rank(ts_decay_linear({price_field}, {medium_lb}))",
            "rank(ts_mean(ts_delta({price_field}, 1), {medium_lb}))",
        ],
        time_horizon="medium",
    )
    logics["momentum_volume_confirmed"] = MarketLogic(
        logic_id="momentum_volume_confirmed",
        category="momentum",
        hypothesis="Volume change confirms price trend: price moves backed by volume are more reliable",
        mechanism="Volume reflects conviction; price-volume divergence signals weak trends",
        factor_templates=[
            "rank(ts_corr(ts_delta({price_field}, {lb}), ts_delta({volume_field}, {lb}), {corr_lb}))",
            "rank(ts_delta({price_field}, {lb}) * ts_delta({volume_field}, {lb}))",
            "rank(sign(ts_delta({price_field}, {lb})) * ts_delta({volume_field}, {lb}))",
        ],
        time_horizon="short",
    )

    # === Value ===
    logics["value_regression"] = MarketLogic(
        logic_id="value_regression",
        category="value",
        hypothesis="Low valuation stocks earn higher future returns due to mean reversion in valuation multiples",
        mechanism="Value premium from risk compensation and mispricing correction over time",
        factor_templates=[
            "-rank({price_field} / {fundamental_field})",
            "-rank(group_zscore({price_field} / {fundamental_field}, sector))",
            "-ts_zscore({price_field} / {fundamental_field}, {zscore_lb})",
        ],
        time_horizon="long",
    )
    logics["value_earnings_quality"] = MarketLogic(
        logic_id="value_earnings_quality",
        category="value",
        hypothesis="High earnings-to-price ratio indicates undervaluation and predicts higher returns",
        mechanism="Earnings yield captures profitability relative to price; high yield signals underpricing",
        factor_templates=[
            "rank(ts_sum({earnings_field}, {sum_lb}) / {price_field})",
            "rank(ts_decay_linear({earnings_field}, {sum_lb}) / {price_field})",
            "rank(ts_mean({earnings_field}, {sum_lb}) / {price_field})",
        ],
        time_horizon="medium",
    )

    # === Quality ===
    logics["quality_earnings_stability"] = MarketLogic(
        logic_id="quality_earnings_stability",
        category="quality",
        hypothesis="Companies with stable earnings outperform due to lower uncertainty and risk premium",
        mechanism="Earnings stability reduces discount rates and attracts risk-averse institutional capital",
        factor_templates=[
            "-rank(ts_std_dev({earnings_field}, {std_lb}))",
            "-rank(ts_std_dev(ts_delta({earnings_field}, 1), {std_lb}))",
            "rank(-ts_std_dev({earnings_field}, {std_lb}) / ts_mean(abs({earnings_field}), {std_lb}))",
        ],
        time_horizon="medium",
    )
    logics["quality_asset_turnover"] = MarketLogic(
        logic_id="quality_asset_turnover",
        category="quality",
        hypothesis="High asset turnover indicates operational efficiency and predicts superior returns",
        mechanism="Efficient resource utilization signals competitive advantage and management quality",
        factor_templates=[
            "rank({revenue_field} / {asset_field})",
            "rank(ts_delta({revenue_field} / {asset_field}, {delta_lb}))",
            "rank(group_zscore({revenue_field} / {asset_field}, sector))",
        ],
        time_horizon="medium",
    )

    # === Size ===
    logics["size_small_cap_premium"] = MarketLogic(
        logic_id="size_small_cap_premium",
        category="size",
        hypothesis="Small-cap stocks earn higher returns due to higher risk and lower analyst coverage",
        mechanism="Size premium from liquidity risk, information asymmetry, and neglected-firm effect",
        factor_templates=[
            "-rank({cap_field})",
            "-rank(log({cap_field}))",
            "-rank(group_zscore({cap_field}, sector))",
        ],
        time_horizon="long",
    )

    # === Volatility ===
    logics["volatility_low_vol_anomaly"] = MarketLogic(
        logic_id="volatility_low_vol_anomaly",
        category="volatility",
        hypothesis="Low-volatility stocks deliver superior risk-adjusted returns due to leverage constraints and lottery preferences",
        mechanism="Investors overpay for lottery-like payoffs in high-vol stocks, depressing their risk-adjusted returns",
        factor_templates=[
            "-rank(ts_std_dev({price_field}, {vol_lb}))",
            "-rank(ts_std_dev(ts_returns({price_field}, 1), {vol_lb}))",
            "-rank(ts_mean(abs(ts_delta({price_field}, 1)), {vol_lb}))",
        ],
        time_horizon="medium",
    )
    logics["volatility_change_signal"] = MarketLogic(
        logic_id="volatility_change_signal",
        category="volatility",
        hypothesis="Declining volatility predicts positive returns as uncertainty resolves favorably",
        mechanism="Volatility compression signals market consensus and reduced risk premium",
        factor_templates=[
            "-rank(ts_delta(ts_std_dev({price_field}, {vol_lb}), {delta_lb}))",
            "-rank(ts_delta(ts_std_dev(ts_returns({price_field}, 1), {vol_lb}), {delta_lb}))",
            "rank(-ts_delta(ts_std_dev({price_field}, {vol_lb}), {delta_lb}) / ts_std_dev({price_field}, {vol_lb}))",
        ],
        time_horizon="short",
    )

    # === Liquidity ===
    logics["liquidity_premium"] = MarketLogic(
        logic_id="liquidity_premium",
        category="liquidity",
        hypothesis="Illiquid stocks earn higher returns as compensation for higher transaction costs and difficulty of exit",
        mechanism="Liquidity premium compensates for the risk of being unable to sell quickly at fair value",
        factor_templates=[
            "-rank({volume_field} / {cap_field})",
            "-rank(ts_mean({volume_field}, {liq_lb}) / {cap_field})",
            "-rank(adv20 / {cap_field})",
        ],
        time_horizon="long",
    )
    logics["liquidity_improvement_signal"] = MarketLogic(
        logic_id="liquidity_improvement_signal",
        category="liquidity",
        hypothesis="Improving liquidity signals growing institutional interest and predicts positive returns",
        mechanism="Increasing volume attracts more market participants, reducing information asymmetry",
        factor_templates=[
            "rank(ts_delta({volume_field}, {liq_lb}))",
            "rank(ts_delta({volume_field} / {cap_field}, {liq_lb}))",
            "rank(ts_delta(adv20, {liq_lb}))",
        ],
        time_horizon="short",
    )

    # === Lead-Lag (v3 新增) ===
    logics["lead_lag_price_volume"] = MarketLogic(
        logic_id="lead_lag_price_volume",
        category="lead_lag",
        hypothesis="Price changes lead volume changes: price moves predict subsequent volume reaction",
        mechanism="Informed traders move prices first; uninformed traders react with volume lag",
        factor_templates=[
            "rank(ts_delta({price_field}, {short_lb}) - ts_delta({volume_field}, {short_lb}))",
            "rank(ts_corr(ts_delta({price_field}, 1), ts_delta({volume_field}, 1), {corr_lb}))",
            "rank(group_neutralize(ts_delta({price_field}, {short_lb}) - ts_delta({volume_field}, {short_lb}), industry))",
        ],
        time_horizon="short",
    )
    logics["lead_lag_cross_field"] = MarketLogic(
        logic_id="lead_lag_cross_field",
        category="lead_lag",
        hypothesis="One data field's movement precedes another: cross-field lead-lag captures information diffusion delay",
        mechanism="Different data sources update at different speeds, creating exploitable temporal gaps",
        factor_templates=[
            "rank(ts_delta({lead_field}, {short_lb}) - ts_delta({lag_field}, {short_lb}))",
            "rank(ts_corr(ts_delta({lead_field}, 1), ts_delta({lag_field}, 1), {corr_lb}))",
            "rank(group_neutralize(ts_delta({lead_field}, {short_lb}) / ts_std_dev({lag_field}, {vol_lb}), industry))",
        ],
        time_horizon="medium",
    )

    # === Momentum Long-Term Reversal (v3 新增) ===
    logics["momentum_long_term_reversal"] = MarketLogic(
        logic_id="momentum_long_term_reversal",
        category="momentum",
        hypothesis="Long-term winners underperform long-term losers over multi-year horizons due to overreaction correction",
        mechanism="De Bondt & Thaler overreaction: extreme past returns predict reversal over 3-5 year horizons",
        factor_templates=[
            "-rank(ts_delta({price_field}, {long_lb}))",
            "-rank(ts_decay_linear(ts_delta({price_field}, 1), {long_lb}))",
            "-rank(group_zscore(ts_delta({price_field}, {long_lb}), sector))",
        ],
        time_horizon="long",
    )

    # === Lead-Lag Industry Rotation (v3 新增) ===
    logics["lead_lag_industry_rotation"] = MarketLogic(
        logic_id="lead_lag_industry_rotation",
        category="lead_lag",
        hypothesis="Industry rotation follows predictable patterns: leading industries' returns predict lagging industries",
        mechanism="Sector fund flows and economic cycle create temporal lead-lag across industries",
        factor_templates=[
            "rank(ts_corr(ts_delta({price_field}, {short_lb}), ts_delta(group_mean({price_field}, sector), {short_lb}), {corr_lb}))",
            "rank(ts_delta({price_field}, {medium_lb}) - ts_delta(group_mean({price_field}, sector), {medium_lb}))",
            "rank(group_neutralize(ts_delta({price_field}, {long_lb}) - ts_delta({cap_field}, {long_lb}), sector))",
        ],
        time_horizon="long",
    )

    # === Mean Reversion ===
    logics["mean_reversion_zscore"] = MarketLogic(
        logic_id="mean_reversion_zscore",
        category="mean_reversion",
        hypothesis="Extreme Z-score deviations from short-term mean predict reversion to mean",
        mechanism="Prices deviate from statistical mean due to noise traders; rational arbitrageurs push back",
        factor_templates=[
            "-rank(ts_zscore(ts_delta({price_field}, {short_lb}), {zscore_lb}))",
            "-rank(ts_zscore(ts_returns({price_field}, {short_lb}), {zscore_lb}))",
            "-rank(ts_zscore({price_field}, {zscore_lb}))",
        ],
        time_horizon="short",
    )
    logics["mean_reversion_bollinger"] = MarketLogic(
        logic_id="mean_reversion_bollinger",
        category="mean_reversion",
        hypothesis="Prices revert to moving average after breaching Bollinger-style bands",
        mechanism="Overbought/oversold signals from statistical boundaries create predictable reversion",
        factor_templates=[
            "-rank(({price_field} - ts_mean({price_field}, {medium_lb})) / ts_std_dev({price_field}, {medium_lb}))",
            "rank((ts_mean({price_field}, {short_lb}) - {price_field}) / ts_std_dev({price_field}, {medium_lb}))",
            "-rank(ts_zscore({price_field} - ts_mean({price_field}, {medium_lb}), {zscore_lb}))",
        ],
        time_horizon="medium",
    )
    logics["mean_reversion_valuation"] = MarketLogic(
        logic_id="mean_reversion_valuation",
        category="mean_reversion",
        hypothesis="Extreme valuation multiples revert toward long-term sector median",
        mechanism="Sector-relative valuation extremes reflect temporary mispricing that corrects over time",
        factor_templates=[
            "-rank(group_zscore({price_field} / {fundamental_field}, sector))",
            "-rank(ts_zscore(group_zscore({price_field} / {fundamental_field}, sector), {long_lb}))",
            "rank(-abs(group_zscore({price_field} / {fundamental_field}, sector)))",
        ],
        time_horizon="long",
    )

    # === Volatility Clustering (v3 新增) ===
    logics["volatility_clustering"] = MarketLogic(
        logic_id="volatility_clustering",
        category="volatility",
        hypothesis="Volatility clusters persist: periods of high (low) volatility tend to be followed by high (low) volatility",
        mechanism="Volatility regime persistence creates predictable risk premium variation",
        factor_templates=[
            "rank(ts_mean(ts_std_dev({price_field}, {short_lb}), {medium_lb}) - ts_std_dev({price_field}, {medium_lb}))",
            "rank(ts_corr(ts_std_dev({price_field}, {short_lb}), ts_std_dev({price_field}, {medium_lb}), {corr_lb}))",
            "-rank(ts_delta(ts_std_dev({price_field}, {short_lb}), {medium_lb}))",
        ],
        time_horizon="medium",
    )

    now = time.time()
    for logic in logics.values():
        logic.last_updated = now

    return logics


def _build_three_block_templates() -> dict[str, ThreeBlockTemplate]:
    templates: dict[str, ThreeBlockTemplate] = {}

    def _std_bc(category: str) -> dict[str, TemplateBlock]:
        if category in ("lead_lag",):
            b = TemplateBlock(BlockType.NEUTRALIZE, "group_neutralize(group_mean({signal}, sector), sector)", True, [])
            c = TemplateBlock(BlockType.DECAY, "ts_decay_linear({neutralized}, {decay_lb})", True, ["decay_lb"])
        elif category in ("mean_reversion",):
            b = TemplateBlock(BlockType.NEUTRALIZE, "group_zscore({signal}, sector)", True, [])
            c = TemplateBlock(BlockType.DECAY, "ts_decay_linear({neutralized}, {decay_lb})", True, ["decay_lb"])
        elif category in ("volatility",):
            b = TemplateBlock(BlockType.NEUTRALIZE, "group_neutralize({signal}, sector)", True, [])
            c = TemplateBlock(BlockType.DECAY, "ts_decay_linear({neutralized}, {decay_lb})", True, ["decay_lb"])
        else:
            b = TemplateBlock(BlockType.NEUTRALIZE, "group_neutralize({signal}, sector)", True, [])
            c = TemplateBlock(BlockType.DECAY, "ts_decay_linear({neutralized}, {decay_lb})", True, ["decay_lb"])
        return {"block_b": b, "block_c": c}

    def _multi_neutralize_bc(category: str) -> dict[str, TemplateBlock]:
        multi_neutralize_strategies = {
            "momentum": "B1",
            "value": "B3",
            "volatility": "B4",
            "quality": "B1",
        }
        strategy = multi_neutralize_strategies.get(category, "B1")

        strategy_templates = {
            "B1": "group_neutralize(group_neutralize({signal}, sector), industry)",
            "B2": "group_neutralize(group_neutralize(group_neutralize({signal}, market), sector), industry)",
            "B3": "group_neutralize(group_zscore({signal}, market), industry)",
            "B4": "group_neutralize({signal}, subindustry)",
        }

        neutralize_expr = strategy_templates.get(strategy, strategy_templates["B1"])
        b = TemplateBlock(BlockType.NEUTRALIZE, neutralize_expr, True, [])
        c = TemplateBlock(BlockType.DECAY, "ts_decay_linear({neutralized}, {decay_lb})", True, ["decay_lb"])
        return {"block_b": b, "block_c": c}

    templates["momentum_short_term"] = ThreeBlockTemplate(
        template_id="momentum_short_term", name="短期动量反转", category="momentum",
        block_a=TemplateBlock(BlockType.SIGNAL, "ts_rank({price_field}, {short_lb}) - ts_rank({price_field}, {long_lb})", False, ["price_field", "short_lb", "long_lb"]),
        **_std_bc("momentum"),
    )
    templates["momentum_medium_term"] = ThreeBlockTemplate(
        template_id="momentum_medium_term", name="中期动量延续", category="momentum",
        block_a=TemplateBlock(BlockType.SIGNAL, "rank(ts_delta({price_field}, {medium_lb}))", False, ["price_field", "medium_lb"]),
        **_std_bc("momentum"),
    )
    templates["momentum_volume_confirmed"] = ThreeBlockTemplate(
        template_id="momentum_volume_confirmed", name="量价确认动量", category="momentum",
        block_a=TemplateBlock(BlockType.SIGNAL, "rank(ts_corr(ts_delta({price_field}, {lb}), ts_delta({volume_field}, {lb}), {corr_lb}))", False, ["price_field", "volume_field", "lb", "corr_lb"]),
        **_std_bc("momentum"),
    )

    templates["value_regression"] = ThreeBlockTemplate(
        template_id="value_regression", name="价值回归", category="value",
        block_a=TemplateBlock(BlockType.SIGNAL, "-rank({price_field} / {fundamental_field})", False, ["price_field", "fundamental_field"]),
        **_std_bc("value"),
    )
    templates["value_earnings_quality"] = ThreeBlockTemplate(
        template_id="value_earnings_quality", name="盈利质量价值", category="value",
        block_a=TemplateBlock(BlockType.SIGNAL, "rank(ts_sum({earnings_field}, {sum_lb}) / {price_field})", False, ["earnings_field", "sum_lb", "price_field"]),
        **_std_bc("value"),
    )

    templates["quality_earnings_stability"] = ThreeBlockTemplate(
        template_id="quality_earnings_stability", name="盈利稳定性", category="quality",
        block_a=TemplateBlock(BlockType.SIGNAL, "-rank(ts_std_dev({earnings_field}, {std_lb}))", False, ["earnings_field", "std_lb"]),
        **_std_bc("quality"),
    )
    templates["quality_asset_turnover"] = ThreeBlockTemplate(
        template_id="quality_asset_turnover", name="资产周转率", category="quality",
        block_a=TemplateBlock(BlockType.SIGNAL, "rank({revenue_field} / {asset_field})", False, ["revenue_field", "asset_field"]),
        **_std_bc("quality"),
    )

    templates["size_small_cap_premium"] = ThreeBlockTemplate(
        template_id="size_small_cap_premium", name="小市值溢价", category="size",
        block_a=TemplateBlock(BlockType.SIGNAL, "-rank({cap_field})", False, ["cap_field"]),
        **_std_bc("size"),
    )

    templates["volatility_low_vol_anomaly"] = ThreeBlockTemplate(
        template_id="volatility_low_vol_anomaly", name="低波动异象", category="volatility",
        block_a=TemplateBlock(BlockType.SIGNAL, "-rank(ts_std_dev({price_field}, {vol_lb}))", False, ["price_field", "vol_lb"]),
        **_std_bc("volatility"),
    )
    templates["volatility_change_signal"] = ThreeBlockTemplate(
        template_id="volatility_change_signal", name="波动率变化信号", category="volatility",
        block_a=TemplateBlock(BlockType.SIGNAL, "-rank(ts_delta(ts_std_dev({price_field}, {vol_lb}), {delta_lb}))", False, ["price_field", "vol_lb", "delta_lb"]),
        **_std_bc("volatility"),
    )
    templates["volatility_clustering"] = ThreeBlockTemplate(
        template_id="volatility_clustering", name="波动率聚类", category="volatility",
        block_a=TemplateBlock(BlockType.SIGNAL, "rank(ts_mean(ts_std_dev({price_field}, {short_lb}), {medium_lb}) - ts_std_dev({price_field}, {medium_lb}))", False, ["price_field", "short_lb", "medium_lb"]),
        **_std_bc("volatility"),
    )

    templates["liquidity_premium"] = ThreeBlockTemplate(
        template_id="liquidity_premium", name="流动性溢价", category="liquidity",
        block_a=TemplateBlock(BlockType.SIGNAL, "-rank({volume_field} / {cap_field})", False, ["volume_field", "cap_field"]),
        **_std_bc("liquidity"),
    )
    templates["liquidity_improvement_signal"] = ThreeBlockTemplate(
        template_id="liquidity_improvement_signal", name="流动性改善信号", category="liquidity",
        block_a=TemplateBlock(BlockType.SIGNAL, "rank(ts_delta({volume_field}, {liq_lb}))", False, ["volume_field", "liq_lb"]),
        **_std_bc("liquidity"),
    )

    templates["lead_lag_price_volume"] = ThreeBlockTemplate(
        template_id="lead_lag_price_volume", name="价格领先量", category="lead_lag",
        block_a=TemplateBlock(BlockType.SIGNAL, "rank(ts_delta({price_field}, {short_lb}) - ts_delta({volume_field}, {short_lb}))", False, ["price_field", "volume_field", "short_lb"]),
        **_std_bc("lead_lag"),
    )
    templates["lead_lag_cross_field"] = ThreeBlockTemplate(
        template_id="lead_lag_cross_field", name="跨字段领先滞后", category="lead_lag",
        block_a=TemplateBlock(BlockType.SIGNAL, "rank(ts_delta({lead_field}, {short_lb}) - ts_delta({lag_field}, {short_lb}))", False, ["lead_field", "lag_field", "short_lb"]),
        **_std_bc("lead_lag"),
    )
    templates["lead_lag_industry_rotation"] = ThreeBlockTemplate(
        template_id="lead_lag_industry_rotation", name="行业轮动", category="lead_lag",
        block_a=TemplateBlock(BlockType.SIGNAL, "rank(ts_delta({price_field}, {medium_lb}) - ts_delta(group_mean({price_field}, sector), {medium_lb}))", False, ["price_field", "medium_lb"]),
        **_std_bc("lead_lag"),
    )

    templates["momentum_long_term_reversal"] = ThreeBlockTemplate(
        template_id="momentum_long_term_reversal", name="长期动量反转", category="momentum",
        block_a=TemplateBlock(BlockType.SIGNAL, "-rank(ts_delta({price_field}, {long_lb}))", False, ["price_field", "long_lb"]),
        **_std_bc("momentum"),
    )

    templates["mean_reversion_zscore"] = ThreeBlockTemplate(
        template_id="mean_reversion_zscore", name="均值回归Z分", category="mean_reversion",
        block_a=TemplateBlock(BlockType.SIGNAL, "-rank(ts_zscore(ts_delta({price_field}, {short_lb}), {zscore_lb}))", False, ["price_field", "short_lb", "zscore_lb"]),
        **_std_bc("mean_reversion"),
    )
    templates["mean_reversion_bollinger"] = ThreeBlockTemplate(
        template_id="mean_reversion_bollinger", name="布林带回归", category="mean_reversion",
        block_a=TemplateBlock(BlockType.SIGNAL, "-rank(({price_field} - ts_mean({price_field}, {medium_lb})) / ts_std_dev({price_field}, {medium_lb}))", False, ["price_field", "medium_lb"]),
        **_std_bc("mean_reversion"),
    )
    templates["mean_reversion_valuation"] = ThreeBlockTemplate(
        template_id="mean_reversion_valuation", name="估值均值回归", category="mean_reversion",
        block_a=TemplateBlock(BlockType.SIGNAL, "-rank(group_zscore({price_field} / {fundamental_field}, sector))", False, ["price_field", "fundamental_field"]),
        **_std_bc("mean_reversion"),
    )

    core_template_ids = [
        "momentum_short_term", "momentum_medium_term", "momentum_volume_confirmed",
        "value_regression", "value_earnings_quality",
        "quality_earnings_stability", "quality_asset_turnover",
        "size_small_cap_premium",
        "volatility_low_vol_anomaly", "volatility_change_signal", "volatility_clustering",
        "liquidity_premium", "liquidity_improvement_signal",
        "lead_lag_price_volume", "lead_lag_cross_field", "lead_lag_industry_rotation",
        "momentum_long_term_reversal",
        "mean_reversion_zscore", "mean_reversion_bollinger", "mean_reversion_valuation",
    ]
    for tid in core_template_ids:
        orig = templates.get(tid)
        if orig is not None:
            mn_id = f"{tid}_mn"
            mn_blocks = _multi_neutralize_bc(orig.category)
            templates[mn_id] = ThreeBlockTemplate(
                template_id=mn_id,
                name=f"{orig.name} (多层中性化)",
                category=orig.category,
                block_a=orig.block_a,
                **mn_blocks,
            )

    return templates


class AlphaLogicLibrary:
    def __init__(self, library_path: str | Path | None = None) -> None:
        self._path = Path(library_path) if library_path else _LIBRARY_PATH
        self._logics: dict[str, MarketLogic] = {}
        self._three_block_templates: dict[str, ThreeBlockTemplate] = {}
        self._load()
        if not self._logics:
            self._init_default_logics()
        if not self._three_block_templates:
            self._three_block_templates = _build_three_block_templates()

    def _init_default_logics(self) -> None:
        self._logics = _build_default_logics()
        self._three_block_templates = _build_three_block_templates()
        self._save()
        logger.info(
            "AlphaLogicLibrary: initialized with %d default logics, %d three-block templates",
            len(self._logics),
            len(self._three_block_templates),
        )

    def get_logics_by_category(self, category: str) -> list[MarketLogic]:
        return [
            logic
            for logic in self._logics.values()
            if logic.category == category
        ]

    def get_logic_for_direction(self, direction: str) -> list[MarketLogic]:
        categories = _DIRECTION_CATEGORY_MAP.get(direction)
        _monitor.record("STEP", "alpha_logics", "select_logic", f"direction={direction}")
        if not categories:
            return list(self._logics.values())
        result: list[MarketLogic] = []
        seen: set[str] = set()
        for cat in categories:
            for logic in self.get_logics_by_category(cat):
                if logic.logic_id not in seen:
                    seen.add(logic.logic_id)
                    result.append(logic)
        return result

    def record_evidence(self, logic_id: str, passed: bool,
                        expression: str = "", sharpe: float | None = None,
                        fitness: float | None = None, turnover: float | None = None,
                        direction: str = "", failure_type: str = "",
                        fix_attempt: str = "", fix_success: bool = False) -> None:
        logic = self._logics.get(logic_id)
        if logic is None:
            logger.warning("AlphaLogicLibrary: unknown logic_id %s", logic_id)
            return
        if passed:
            logic.evidence_count += 1
        else:
            logic.evidence_count = max(0, logic.evidence_count - 1)
        logic.last_updated = time.time()
        if expression:
            logic.evidence_records.append({
                "expression": expression[:200],
                "sharpe": sharpe,
                "fitness": fitness,
                "turnover": turnover,
                "direction": direction,
                "failure_type": failure_type,
                "fix_attempt": fix_attempt[:200] if fix_attempt else "",
                "fix_success": fix_success,
                "timestamp": time.time(),
            })
            if len(logic.evidence_records) > 100:
                logic.evidence_records = logic.evidence_records[-100:]
        _monitor.record("STEP", "alpha_logics", "record_evidence", f"logic={logic_id} passed={passed}")
        self._save()

    def get_direction_weights(self) -> dict[str, float]:
        category_to_directions: dict[str, list[str]] = {}
        for direction, categories in _DIRECTION_CATEGORY_MAP.items():
            for cat in categories:
                category_to_directions.setdefault(cat, []).append(direction)
        weights: dict[str, float] = {}
        for logic in self._logics.values():
            directions = category_to_directions.get(logic.category, [logic.category])
            for direction in directions:
                ec = logic.evidence_count
                if ec > 0:
                    weights[direction] = weights.get(direction, 0.0) + ec
                else:
                    weights[direction] = weights.get(direction, 0.0) + 0.5
        return weights

    def get_top_logics(self, n: int = 5) -> list[MarketLogic]:
        """Return the top *n* logics sorted by evidence_count (descending).

        Useful for inspecting which market hypotheses have accumulated the most
        empirical support across sessions.
        """
        _monitor.record("STEP", "alpha_logics", "get_top_logics", f"n={n}")
        sorted_logics = sorted(
            self._logics.values(),
            key=lambda l: l.evidence_count,
            reverse=True,
        )
        return sorted_logics[:n]

    @algo_log()
    def instantiate_template(
        self,
        template_id: str,
        fields: dict[str, str],
        params: dict | None = None,
        template_idx: int = 0,
    ) -> str | None:
        three_block = self._three_block_templates.get(template_id)
        if three_block is not None:
            _monitor.record("STEP", "alpha_logics", "instantiate_template", f"template={template_id} mode=three_block")
            all_kwargs = {**fields}
            if params:
                all_kwargs.update({k: v for k, v in params.items()})
            expr = three_block.assemble(**all_kwargs)

            # ── AST 硬校验 + auto-fix（AlphaBench ICLR'26）────────────────────
            validator = ASTValidator()
            ast_result = validator.validate(expr)
            if not ast_result.passed or not three_block.validate_assembly(expr):
                logger.warning(
                    "P2_TEMPLATE_LOCK: [DEFENSIVE_LOG] AST 校验失败，强制注入默认 B/C 段 | "
                    "template=%s | errors=%s | warnings=%s",
                    template_id, ast_result.errors, ast_result.warnings[:3],
                )
                log_call(
                    "AlphaLogicLibrary.AST_auto_fix",
                    input={"template": template_id, "expr_before_fix": expr[:100]},
                    output={
                        "ast_passed": ast_result.passed,
                        "errors": ast_result.errors,
                        "warnings": ast_result.warnings[:3],
                        "structure": ast_result.structure_info,
                        "fix_suggestions": ast_result.fix_suggestions[:3],
                    },
                    level=logging.WARNING,
                )
                default_decay_lb = params.get("decay_lb", 10) if params else 10
                signal_expr = three_block.block_a.template_str
                for key, value in fields.items():
                    signal_expr = signal_expr.replace("{" + key + "}", str(value))
                neutralized = three_block.block_b.template_str.replace("{signal}", signal_expr)
                expr = three_block.block_c.template_str.replace("{neutralized}", neutralized).replace("{decay_lb}", str(default_decay_lb))

            _monitor.record("PASS", "alpha_logics", "instantiate_template", f"template={template_id} mode=three_block validated=True")
            return expr

        logic = self._logics.get(template_id)
        if logic is None:
            return None
        if template_idx < 0 or template_idx >= len(logic.factor_templates):
            return None
        template = logic.factor_templates[template_idx]
        _monitor.record("STEP", "alpha_logics", "instantiate_template", f"logic={template_id} mode=legacy")
        for key, value in fields.items():
            template = template.replace("{" + key + "}", value)
        _monitor.record("PASS", "alpha_logics", "instantiate_template", f"logic={template_id} mode=legacy")
        return template

    def get_templates_for_direction(self, direction: str, adaptive_weights: dict[str, float] | None = None) -> list[str]:
        try:
            categories = _DIRECTION_CATEGORY_MAP.get(direction)
            if not categories:
                categories = [direction]
            templates: list[str] = []
            for logic in self._logics.values():
                if logic.category in categories:
                    if logic.factor_templates:
                        templates.extend(logic.factor_templates)
            
            import logging
            _logger = logging.getLogger(__name__)
            
            if adaptive_weights and len(templates) > 1:
                _logger.info(
                    "[ADAPT-NEUT-WEIGHT] Templates BEFORE sorting: %s",
                    [t[:60] for t in templates[:10]],
                )
                import random
                weighted = [(t, adaptive_weights.get(t, 1.0)) for t in templates]
                total_weight = sum(w for _, w in weighted)
                if total_weight > 0:
                    normalized = [(t, w / total_weight) for t, w in weighted]
                    templates = [t for t, _ in sorted(normalized, key=lambda x: -x[1])]
                    _logger.info(
                        "[ADAPT-NEUT-WEIGHT] Templates AFTER weighted sorting: %s",
                        [t[:60] for t in templates[:10]],
                    )
            return templates[:5]
        except (OSError, ValueError, RuntimeError):
            return []

    def get_logic(self, logic_id: str) -> MarketLogic | None:
        return self._logics.get(logic_id)

    @algo_log()
    def get_three_block_template(self, template_id: str) -> ThreeBlockTemplate | None:
        return self._three_block_templates.get(template_id)

    def all_logics(self) -> list[MarketLogic]:
        return list(self._logics.values())

    @property
    def logics(self) -> list[MarketLogic]:
        """Return a defensive copy of all market logics in the library.

        The returned list is independent of internal storage so callers can
        iterate or sort without affecting the library state.
        """
        return list(self._logics.values())

    def accumulate_diagnosis(self, direction: str, diagnosis_result: dict) -> None:
        try:
            for logic in self._logics.values():
                logic_dir = logic.category.lower() if hasattr(logic, 'category') else ""
                if direction.lower() in logic_dir or logic_dir in direction.lower():
                    logic.accumulated_diagnoses.append({
                        "failure_type": diagnosis_result.get("failure_type", "unknown"),
                        "root_cause": diagnosis_result.get("root_cause", ""),
                        "suggested_fix": diagnosis_result.get("suggested_fix", ""),
                        "confidence": diagnosis_result.get("confidence", 0.5),
                    })
                    self._save()
                    break
        except (OSError, ValueError, RuntimeError):
            pass

    @property
    def logic_count(self) -> int:
        """Return the total number of market logics currently in the library."""
        return len(self._logics)

    def evolve_logics(self) -> dict:
        split_count = 0
        merged_count = 0
        gaps: list[str] = []

        to_split: list[str] = []
        for lid, logic in list(self._logics.items()):
            if logic.evidence_count <= 20:
                continue
            _scan_sub_patterns: dict[str, list[dict]] = {}
            for rec in logic.evidence_records:
                ft = rec.get("failure_type", "unknown") or "unknown"
                d = rec.get("direction", "unknown") or "unknown"
                key = f"{ft}|{d}"
                _scan_sub_patterns.setdefault(key, []).append(rec)
            if len(_scan_sub_patterns) < 2:
                continue
            to_split.append(lid)

        for lid in to_split:
            logic = self._logics.get(lid)
            if logic is None:
                continue
            sub_patterns: dict[str, list[dict]] = {}
            for rec in logic.evidence_records:
                ft = rec.get("failure_type", "unknown") or "unknown"
                d = rec.get("direction", "unknown") or "unknown"
                key = f"{ft}|{d}"
                sub_patterns.setdefault(key, []).append(rec)
            for key, records in sub_patterns.items():
                ft, d = key.split("|", 1)
                child_id = f"{lid}_{ft}_{d}".replace(" ", "_").lower()
                if child_id in self._logics:
                    continue
                child = MarketLogic(
                    logic_id=child_id,
                    category=logic.category,
                    hypothesis=f"{logic.hypothesis} (sub-pattern: {ft} in {d})",
                    mechanism=logic.mechanism,
                    factor_templates=logic.factor_templates[:],
                    time_horizon=logic.time_horizon,
                    evidence_count=len(records),
                    last_updated=time.time(),
                    evidence_records=records[:],
                )
                self._logics[child_id] = child
                split_count += 1
            logic.evidence_records = [{"split_into": [f"{lid}_{k.split('|')[0]}_{k.split('|')[1]}".replace(" ", "_").lower() for k in sub_patterns.keys()]}]

        all_logics = list(self._logics.values())
        to_merge: list[tuple[str, str]] = []
        for i in range(len(all_logics)):
            for j in range(i + 1, len(all_logics)):
                la, lb = all_logics[i], all_logics[j]
                if la.evidence_count <= 5 or lb.evidence_count <= 5:
                    continue
                sim = self._simple_string_similarity(la.hypothesis, lb.hypothesis)
                if sim > 0.7:
                    to_merge.append((la.logic_id, lb.logic_id))

        merged_ids: set[str] = set()
        for aid, bid in to_merge:
            if aid in merged_ids or bid in merged_ids:
                continue
            la = self._logics.get(aid)
            lb = self._logics.get(bid)
            if la is None or lb is None:
                continue
            merged_id = f"{aid}_merged_{bid}".replace(" ", "_").lower()
            if merged_id in self._logics:
                continue
            merged = MarketLogic(
                logic_id=merged_id,
                category=la.category,
                hypothesis=la.hypothesis,
                mechanism=f"{la.mechanism}; {lb.mechanism}",
                factor_templates=list(dict.fromkeys(la.factor_templates + lb.factor_templates))[:5],
                time_horizon=la.time_horizon,
                evidence_count=la.evidence_count + lb.evidence_count,
                last_updated=time.time(),
                evidence_records=(la.evidence_records + lb.evidence_records)[-100:],
            )
            self._logics[merged_id] = merged
            del self._logics[aid]
            del self._logics[bid]
            merged_ids.add(aid)
            merged_ids.add(bid)
            merged_count += 1

        covered_directions: set[str] = set()
        covered_mechanisms: set[str] = set()
        for logic in self._logics.values():
            for d, cats in _DIRECTION_CATEGORY_MAP.items():
                if logic.category in cats:
                    covered_directions.add(d)
            if logic.mechanism:
                covered_mechanisms.add(logic.mechanism.split()[0].lower() if logic.mechanism else "")
        for d in _DIRECTION_CATEGORY_MAP:
            if d not in covered_directions:
                gaps.append(f"direction:{d}")
        _known_mechanism_types = {"overreaction", "gradual", "volume", "value", "earnings", "volatility", "liquidity", "size", "momentum", "mean_reversion"}
        for mech in _known_mechanism_types:
            if mech not in covered_mechanisms:
                gaps.append(f"mechanism:{mech}")

        if split_count > 0 or merged_count > 0:
            self._save()

        return {"split": split_count, "merged": merged_count, "gaps": gaps}

    @staticmethod
    def _simple_string_similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        a_words = set(a.lower().split())
        b_words = set(b.lower().split())
        if not a_words and not b_words:
            return 1.0
        intersection = a_words & b_words
        union = a_words | b_words
        return len(intersection) / len(union) if union else 0.0

    async def propose_new_logic(self, gap_description: str, llm_generate_fn=None) -> dict | None:
        if llm_generate_fn is None:
            return None
        prompt = (
            "You are a quantitative finance researcher. A gap has been identified in our market logic library.\n"
            f"Gap: {gap_description}\n\n"
            "Propose a new market logic to fill this gap. Output ONLY a JSON object with:\n"
            '{"hypothesis": "...", "mechanism": "...", "factor_templates": ["..."], "category": "...", "time_horizon": "short|medium|long"}\n\n'
            "Requirements:\n"
            "- hypothesis: a specific, testable market hypothesis\n"
            "- mechanism: the economic rationale\n"
            "- factor_templates: at least one FASTEXPR template with {placeholder} variables\n"
            "- category: one of momentum, value, quality, volatility, liquidity, size\n"
            "- time_horizon: short, medium, or long"
        )
        try:
            raw = await llm_generate_fn(
                system_prompt=prompt,
                history=[],
                user_msg=f"Propose a new market logic for gap: {gap_description}",
                session_id="logic_evolution",
                cycle=0,
            )
            text = raw.strip()
            if text.startswith("```"):
                first_nl = text.find("\n")
                if first_nl >= 0:
                    text = text[first_nl + 1:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            data = json.loads(text)
            if not data.get("hypothesis") or not data.get("mechanism") or not data.get("factor_templates"):
                return None
            if len(data["factor_templates"]) < 1:
                return None
            logic_id = f"evolved_{gap_description.replace(':', '_').replace(' ', '_')}_{int(time.time())}"
            new_logic = MarketLogic(
                logic_id=logic_id,
                category=data.get("category", "momentum"),
                hypothesis=data["hypothesis"],
                mechanism=data["mechanism"],
                factor_templates=data["factor_templates"][:5],
                time_horizon=data.get("time_horizon", "medium"),
                evidence_count=0,
                last_updated=time.time(),
            )
            self._logics[logic_id] = new_logic
            self._save()
            logger.info("AlphaLogicLibrary: proposed new logic '%s' for gap '%s'", logic_id, gap_description)
            return new_logic.to_dict()
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            logger.warning("AlphaLogicLibrary: propose_new_logic failed: %s", exc)
            return None

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                lid: logic.to_dict()
                for lid, logic in self._logics.items()
            }
            import tempfile
            import shutil
            fd, tmp_path = tempfile.mkstemp(
                suffix=".json.tmp",
                dir=str(self._path.parent),
                prefix=".market_logics_",
            )
            try:
                with open(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                shutil.move(tmp_path, str(self._path))
            except Exception:
                try:
                    import os
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as exc:
            logger.error("AlphaLogicLibrary: failed to save (atomic): %s", exc)

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            self._logics = {
                lid: MarketLogic.from_dict(d) for lid, d in data.items()
            }
            logger.info(
                "AlphaLogicLibrary: loaded %d logics from %s",
                len(self._logics),
                self._path,
            )
        except (ValueError, TypeError, OSError) as exc:
            logger.error("AlphaLogicLibrary: failed to load: %s", exc)
            self._logics = {}
