"""
OpenAlpha-Brain — Build Vector Indexes for RAG
Reads brain_operators.json and brain_datafields.json, generates embeddings
via LM Studio embedding API, and saves three separate vector stores:
  - vec_operators.json  (66 operators)
  - vec_fields.json     (7000+ data fields)
  - vec_finlogic.json   (financial logic knowledge base)

Usage:
    python build_vectors.py [--incremental]
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

from openalpha_brain.config.config import settings
from openalpha_brain.data import VEC_STORE_DIR, get_data_path
from openalpha_brain.knowledge.vector_index import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 64
CONCURRENCY = 1
MAX_BATCH_RETRIES = 3

FINANCIAL_LOGIC_ENTRIES: list[dict] = [
    {
        "id": "momentum_classic",
        "text": (
            "Momentum strategy: assets that have performed well recently "
            "tend to continue performing well. Use ts_delta or ts_returns "
            "over 5-20 day windows, rank cross-sectionally. Classic "
            "Jegadeesh-Titman momentum."
        ),
        "category": "momentum",
    },
    {
        "id": "momentum_reversal",
        "text": (
            "Short-term reversal: assets with extreme returns over past "
            "1-5 days tend to reverse. Use negative ts_delta(close, 1-5) "
            "with group_neutralize. Contrarian signal."
        ),
        "category": "momentum",
    },
    {
        "id": "momentum_cross",
        "text": (
            "Cross-sectional momentum: rank assets by past returns relative to peers. Use"
            "rank(ts_delta(close, 20)) with group_neutralize by industry or sector."
        ),
        "category": "momentum",
    },
    {
        "id": "mean_reversion_price",
        "text": (
            "Mean reversion: prices revert to their historical mean. Use ts_zscore(close, 20-60) to"
            "identify overbought/oversold conditions. Negative signal when z-score is extreme."
        ),
        "category": "mean_reversion",
    },
    {
        "id": "mean_reversion_volume",
        "text": (
            "Volume mean reversion: abnormal volume tends to revert. Use ts_zscore(volume, 20) or"
            "ts_delta(volume, 5) with negative sign. High volume often precedes reversal."
        ),
        "category": "mean_reversion",
    },
    {
        "id": "mean_reversion_spread",
        "text": (
            "Spread mean reversion: price spreads between related assets revert. Use subtract(close,"
            "vwap) or ts_zscore of spread measures."
        ),
        "category": "mean_reversion",
    },
    {
        "id": "volatility_breakout",
        "text": (
            "Volatility breakout: low volatility periods precede large moves. Use ts_std_dev(returns,"
            "20) as regime indicator. Rank by inverse volatility for breakout signals."
        ),
        "category": "volatility",
    },
    {
        "id": "volatility_risk_premium",
        "text": (
            "Volatility risk premium: implied volatility typically exceeds realized. Use"
            "ts_std_dev(returns, 20) vs ts_mean(returns, 20) spread. Low vol assets offer risk"
            "premium."
        ),
        "category": "volatility",
    },
    {
        "id": "volatility_clustering",
        "text": (
            "Volatility clustering: high vol follows high vol. Use"
            "ts_decay_linear(ts_std_dev(returns, 5), 10) to capture persistent volatility regimes."
        ),
        "category": "volatility",
    },
    {
        "id": "value_fundamental",
        "text": (
            "Value investing: cheap assets outperform. Use fundamental ratios like sales/market_cap,"
            "equity/debt, revenue/assets. Rank by value metrics with group_neutralize."
        ),
        "category": "value",
    },
    {
        "id": "value_earnings_yield",
        "text": (
            "Earnings yield: inverse P/E ratio as value signal. Use revenue/cap or"
            "earnings/market_cap measures. Higher earnings yield suggests undervaluation."
        ),
        "category": "value",
    },
    {
        "id": "value_book_value",
        "text": (
            "Book value: assets with low price-to-book outperform. Use equity or assets relative to"
            "market cap. Fundamental value signal."
        ),
        "category": "value",
    },
    {
        "id": "quality_profitability",
        "text": (
            "Quality factor: profitable, stable companies outperform. Use revenue growth stability,"
            "low debt/equity, high return on assets. Combine profitability and leverage metrics."
        ),
        "category": "quality",
    },
    {
        "id": "quality_leverage",
        "text": (
            "Leverage risk: high debt companies underperform. Use debt/equity or liabilities/assets"
            "ratio. Low leverage indicates quality. Inverse signal."
        ),
        "category": "quality",
    },
    {
        "id": "quality_earnings_stability",
        "text": (
            "Earnings stability: consistent earnings indicate quality. Use ts_std_dev of revenue or"
            "earnings over long windows. Low volatility of fundamentals is positive."
        ),
        "category": "quality",
    },
    {
        "id": "liquidity_amihud",
        "text": (
            "Amihud illiquidity: |return|/volume measures price impact. High illiquidity predicts"
            "higher returns. Use abs(returns)/volume with ts_mean."
        ),
        "category": "liquidity",
    },
    {
        "id": "liquidity_turnover",
        "text": (
            "Turnover as liquidity proxy: high turnover indicates liquid assets. Use volume/adv20 or"
            "volume/cap ratio. Low turnover assets may have liquidity premium."
        ),
        "category": "liquidity",
    },
    {
        "id": "liquidity_volume_price",
        "text": (
            "Volume-price interaction: unusual volume at price extremes signals reversal or"
            "continuation. Combine ts_delta(volume, 5) with ts_zscore(close, 20)."
        ),
        "category": "liquidity",
    },
    {
        "id": "size_market_cap",
        "text": (
            "Size effect: small-cap stocks outperform large-cap. Use cap or log(cap) as size proxy."
            "Rank by market cap with group_neutralize."
        ),
        "category": "size",
    },
    {
        "id": "size_adv",
        "text": (
            "Average daily volume as size proxy: adv20 captures trading size. Use log(adv20) or"
            "rank(adv20) for size-based signals."
        ),
        "category": "size",
    },
    {
        "id": "industry_rotation",
        "text": (
            "Industry rotation: sector-relative momentum. Use ts_delta(close, 20) with"
            "group_neutralize by industry. Capture sector-specific trends."
        ),
        "category": "industry",
    },
    {
        "id": "industry_relative",
        "text": (
            "Relative strength within industry: compare asset to industry average. Use"
            "subtract(close, group_mean) or rank within industry group."
        ),
        "category": "industry",
    },
    {
        "id": "ts_decay_signal",
        "text": (
            "Time-decay weighting: recent data more important. Use ts_decay_linear with windows 5-20."
            "Exponential decay captures recency bias in signals."
        ),
        "category": "temporal",
    },
    {
        "id": "ts_rank_signal",
        "text": (
            "Temporal ranking: rank current value within historical distribution. Use ts_rank(close,"
            "20) to measure where current price sits in recent range. Values near 1 = near high."
        ),
        "category": "temporal",
    },
    {
        "id": "ts_corr_signal",
        "text": (
            "Rolling correlation: correlation between two series over time. Use ts_corr(close,"
            "volume, 20) to detect volume-price relationship changes."
        ),
        "category": "temporal",
    },
    {
        "id": "ts_regression_signal",
        "text": (
            "Time-series regression: factor exposure over time. Use ts_regression(returns,"
            "market_returns, 20, rettype) for beta, residual, or alpha extraction."
        ),
        "category": "temporal",
    },
    {
        "id": "cross_sectional_rank",
        "text": (
            "Cross-sectional ranking: rank assets against each other at each point in time. Use"
            "rank() to normalize distributions. Essential for market-neutral alphas."
        ),
        "category": "cross_sectional",
    },
    {
        "id": "cross_sectional_zscore",
        "text": (
            "Cross-sectional z-score: standardize across assets. Use zscore() or group_zscore() for"
            "industry-relative normalization."
        ),
        "category": "cross_sectional",
    },
    {
        "id": "cross_sectional_neutralize",
        "text": (
            "Group neutralization: remove group effects. Use group_neutralize(expr, industry) to"
            "create market-neutral signals. Critical for IQC submission."
        ),
        "category": "cross_sectional",
    },
    {
        "id": "interaction_vol_momentum",
        "text": (
            "Volatility-conditioned momentum: momentum stronger in low-vol regimes. Combine"
            "rank(ts_delta(close, 20)) with inverse ts_std_dev(returns, 20)."
        ),
        "category": "interaction",
    },
    {
        "id": "interaction_liquidity_value",
        "text": (
            "Liquidity-conditioned value: value premium stronger in illiquid stocks. Combine"
            "fundamental ratios with volume/adv20 inverse."
        ),
        "category": "interaction",
    },
    {
        "id": "interaction_size_quality",
        "text": (
            "Size-quality interaction: quality premium concentrated in small caps. Combine quality"
            "metrics with log(cap) inverse."
        ),
        "category": "interaction",
    },
]


async def ensure_embedding_model_loaded() -> None:
    from openalpha_brain.services import llm_client

    await llm_client._ensure_embed_model_loaded()


async def _embed_batch(texts: list[str]) -> np.ndarray:
    import requests as _sync_requests

    last_exc = None
    for retry in range(MAX_BATCH_RETRIES):
        try:
            payload = {
                "model": settings.EMBED_MODEL,
                "input": texts,
            }
            resp = await asyncio.to_thread(
                _sync_requests.post,
                settings.EMBED_BASE_URL,
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = []
            for item in sorted(data["data"], key=lambda x: x["index"]):
                embeddings.append(item["embedding"])
            return np.array(embeddings, dtype=np.float32)
        except (ConnectionError, OSError, TimeoutError) as exc:
            last_exc = exc
            if retry < MAX_BATCH_RETRIES - 1:
                wait = 2**retry
                logger.warning(
                    "Embed batch failed (retry %d/%d after %ds): %s", retry + 1, MAX_BATCH_RETRIES, wait, exc
                )
                await asyncio.sleep(wait)
    raise last_exc  # type: ignore[misc]


async def embed_texts(texts: list[str]) -> np.ndarray:
    all_vecs: list[np.ndarray] = []
    total = len(texts)
    for i in range(0, total, BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        logger.info(
            "Embedding batch %d/%d (%d texts)...", i // BATCH_SIZE + 1, math.ceil(total / BATCH_SIZE), len(batch)
        )
        vecs = await _embed_batch(batch)
        all_vecs.append(vecs)
        if i + BATCH_SIZE < total:
            await asyncio.sleep(0.1)
    return np.concatenate(all_vecs, axis=0)


def _build_operator_text(op: dict) -> str:
    parts = [f"Operator: {op['name']}"]
    if op.get("category"):
        parts.append(f"Category: {op['category']}")
    if op.get("definition"):
        parts.append(f"Signature: {op['definition']}")
    if op.get("description"):
        parts.append(f"Description: {op['description']}")
    return " | ".join(parts)


def _build_field_text(field: dict) -> str:
    parts = [f"Field: {field['id']}"]
    if field.get("description"):
        parts.append(f"Meaning: {field['description']}")
    cat = field.get("category", {})
    if cat.get("name"):
        parts.append(f"Category: {cat['name']}")
    subcat = field.get("subcategory", {})
    if subcat.get("name"):
        parts.append(f"Subcategory: {subcat['name']}")
    ds = field.get("dataset", {})
    if ds.get("name"):
        parts.append(f"Dataset: {ds['name']}")
    if field.get("region"):
        parts.append(f"Region: {field['region']}")
    if field.get("coverage") is not None:
        parts.append(f"Coverage: {field['coverage']:.2%}")
    return " | ".join(parts)


def _build_finlogic_text(entry: dict) -> str:
    return entry["text"]


async def build_operator_index(ops_path: Path, out_path: Path, incremental: bool = False) -> None:
    existing_ids: set[str] = set()
    existing_store: VectorStore | None = None
    if incremental and out_path.exists():
        existing_store = VectorStore.load_index(out_path)
        existing_ids = set(existing_store.get_all_ids())
        logger.info("Incremental: %d existing operators", len(existing_ids))

    ops = json.loads(ops_path.read_text(encoding="utf-8"))
    new_ops = [op for op in ops if op["name"] not in existing_ids]
    if not new_ops:
        logger.info("No new operators to index.")
        if existing_store:
            existing_store.save_index(out_path)
        return

    texts = [_build_operator_text(op) for op in new_ops]
    ids = [op["name"] for op in new_ops]
    metas = [
        {
            "category": op.get("category", ""),
            "definition": op.get("definition", ""),
            "description": op.get("description", ""),
        }
        for op in new_ops
    ]

    vecs = await embed_texts(texts)
    store = existing_store or VectorStore(dim=vecs.shape[1])
    store.add_documents(ids, [vecs[i] for i in range(len(ids))], metas)
    store.save_index(out_path)
    logger.info("Operator index: %d docs → %s", store.count, out_path)


async def build_field_index(fields_path: Path, out_path: Path, incremental: bool = False) -> None:
    existing_ids: set[str] = set()
    existing_store: VectorStore | None = None
    if incremental and out_path.exists():
        existing_store = VectorStore.load_index(out_path)
        existing_ids = set(existing_store.get_all_ids())
        logger.info("Incremental: %d existing fields", len(existing_ids))

    fields = json.loads(fields_path.read_text(encoding="utf-8"))
    new_fields = [f for f in fields if f["id"] not in existing_ids]
    if not new_fields:
        logger.info("No new fields to index.")
        if existing_store:
            existing_store.save_index(out_path)
        return

    texts = [_build_field_text(f) for f in new_fields]
    ids = [f["id"] for f in new_fields]
    metas = [
        {
            "category": f.get("category", {}).get("name", ""),
            "subcategory": f.get("subcategory", {}).get("name", ""),
            "dataset": f.get("dataset", {}).get("name", ""),
            "region": f.get("region", ""),
            "coverage": f.get("coverage", 0),
        }
        for f in new_fields
    ]

    vecs = await embed_texts(texts)
    store = existing_store or VectorStore(dim=vecs.shape[1])
    store.add_documents(ids, [vecs[i] for i in range(len(ids))], metas)
    store.save_index(out_path)
    logger.info("Field index: %d docs → %s", store.count, out_path)


async def build_finlogic_index(out_path: Path) -> None:
    texts = [_build_finlogic_text(e) for e in FINANCIAL_LOGIC_ENTRIES]
    ids = [e["id"] for e in FINANCIAL_LOGIC_ENTRIES]
    metas = [{"category": e["category"]} for e in FINANCIAL_LOGIC_ENTRIES]

    vecs = await embed_texts(texts)
    store = VectorStore(dim=vecs.shape[1])
    store.add_documents(ids, [vecs[i] for i in range(len(ids))], metas)
    store.save_index(out_path)
    logger.info("Financial logic index: %d docs → %s", store.count, out_path)


async def main(incremental: bool = False) -> None:
    await ensure_embedding_model_loaded()
    ops_path = get_data_path("brain_operators.json")
    fields_path = get_data_path("brain_datafields.json")
    vec_dir = VEC_STORE_DIR
    vec_dir.mkdir(exist_ok=True)

    logger.info("=== Building Operator Vector Index ===")
    await build_operator_index(ops_path, vec_dir / "vec_operators.json", incremental)

    logger.info("=== Building Field Vector Index ===")
    await build_field_index(fields_path, vec_dir / "vec_fields.json", incremental)

    logger.info("=== Building Financial Logic Vector Index ===")
    await build_finlogic_index(vec_dir / "vec_finlogic.json")

    logger.info("=== All indexes built successfully ===")


if __name__ == "__main__":
    incr = "--incremental" in sys.argv
    asyncio.run(main(incremental=incr))
