"""DEPRECATED: This module classifies fields into the old 11-SignalFamily grid.

After the 2026-05-22 Grid Redefinition (660 → 72 cells), SignalFamily was replaced
by DatasetCategory (8 categories: ANALYST, FUNDAMENTAL, MODEL, NEWS, OPTION,
PRICE_VOLUME, SENTIMENT, SOCIAL_MEDIA). This module is no longer compatible with
the current ExplorationGrid and should not be used for new field classification.

Field mapping is now derived from the prefix-based analysis in
`docs/grid-redefinition-discovery.md` and stored in `ExplorationGrid.DATASET_FIELD_MAP`.
"""
from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from alpha_agent.datasets_loader import DatasetsLoader, FieldMetadata
from alpha_agent.llm_client import LLMClient

logger = logging.getLogger(__name__)

BATCH_SIZE = 25

FAMILY_DEFINITIONS = {
    "VALUE": {
        "description": "Fama-French 1992 HML — 價值因子，基於淨值、現金流、盈餘等基本面指標",
        "examples": ["bookvalue_ps", "return_equity", "pretax_income", "ebitda"],
    },
    "MOMENTUM": {
        "description": "Jegadeesh-Titman 1993 — 動能因子，基於價格與報酬趨勢",
        "examples": ["close", "returns", "high", "low", "open"],
    },
    "QUALITY": {
        "description": "Novy-Marx 2013, Fama-French 2015 RMW/CMA — 品質因子，獲利能力、資產效率、償債能力",
        "examples": ["return_equity", "operating_income", "cashflow_op", "assets", "debt", "depre_amort"],
    },
    "LOW_VOLATILITY": {
        "description": "Frazzini-Pedersen 2014 BAB — 低波動因子，價格波動與成交量相關",
        "examples": ["returns", "close", "high", "low", "volume", "adv20"],
    },
    "SIZE": {
        "description": "Banz 1981, Fama-French 1993 SMB — 規模因子，市值與流通股數",
        "examples": ["cap", "sharesout", "close", "volume"],
    },
    "SENTIMENT": {
        "description": "Baker-Wurgler 2006 — 情緒因子，社群媒體情緒與分析師情緒",
        "examples": ["scl12_buzz", "scl12_sentiment", "snt_value", "snt_buzz"],
    },
    "NEWS_ATTENTION": {
        "description": "Tetlock 2007 — 新聞關注度因子，新聞影響機率與關注度",
        "examples": ["nws18_relevance", "nws18_nip", "news_ratio_vol", "news_pct_30min"],
    },
    "SHORT_REVERSAL": {
        "description": "Jegadeesh 1990, Lehmann 1990 — 短反轉因子，短期價格反轉",
        "examples": ["close", "vwap", "returns", "open", "high", "low"],
    },
    "LIQUIDITY": {
        "description": "Amihud 2002, Pastor-Stambaugh 2003 — 流動性因子，成交量與市值",
        "examples": ["volume", "adv20", "cap", "close", "returns", "sharesout"],
    },
    "GROWTH": {
        "description": "Lakonishok-Shleifer-Vishny 1994 — 成長因子，營收與盈餘成長",
        "examples": ["sales_growth", "eps", "income", "revenue", "cashflow_op", "capex"],
    },
    "TECHNICAL_TREND": {
        "description": "Brock-Lakonishok-LeBaron 1992 — 技術趨勢因子，價格與成交量技術指標",
        "examples": ["close", "volume", "high", "low", "open", "returns", "vwap"],
    },
    "NONE": {
        "description": "不屬於上述任何因子家族的 data field（如識別碼、分類標籤、symbol type）",
        "examples": ["cusip", "isin", "sedol", "ticker", "sector", "industry"],
    },
}

ALL_FAMILY_NAMES = [k for k in FAMILY_DEFINITIONS if k != "NONE"]


@dataclass
class FieldProposal:
    field: str
    proposed_family: str
    confidence: float
    reasoning: str
    status: str = "pending"
    reviewer_notes: Optional[str] = None


_SYSTEM_PROMPT = """你是量化因子分類專家。你的任務是將 financial data field 歸類到適當的 SignalFamily。

以下是各 SignalFamily 的定義與範例：

{family_definitions}

回傳 JSON 格式：
{{
  "classifications": [
    {{
      "field": "field_id",
      "family": "VALUE",
      "confidence": 0.95,
      "reason": "簡短理由（中文或英文皆可）"
    }}
  ]
}}

注意：
- family 必須是上述 SignalFamily 之一，若無法歸類則設為 "NONE"
- confidence 介於 0.0 到 1.0 之間
- 每個 field 只歸類到一個 family
- 基於 field 名稱、Description 與 Type 做判斷
"""


def _build_system_prompt() -> str:
    lines = []
    for name, info in FAMILY_DEFINITIONS.items():
        examples = ", ".join(info["examples"])
        lines.append(f"  {name}: {info['description']} (例: {examples})")
    return _SYSTEM_PROMPT.format(family_definitions="\n".join(lines))


def _build_user_prompt(fields: List[FieldMetadata]) -> str:
    items = []
    for f in fields:
        items.append({
            "id": f.field_id,
            "description": f.description,
            "type": f.type,
            "dataset": f.dataset_id,
        })
    return json.dumps({"fields": items}, ensure_ascii=False, indent=2)


class FieldClassifier:
    def __init__(self, llm_client: LLMClient) -> None:
        warnings.warn(
            "FieldClassifier is deprecated. It classifies fields into the old "
            "11-SignalFamily grid, which was replaced by 8 DatasetCategory on 2026-05-22. "
            "Use ExplorationGrid.DATASET_FIELD_MAP for field mapping.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._llm = llm_client
        self._system_prompt = _build_system_prompt()

    def classify_batch(self, fields: List[FieldMetadata]) -> List[FieldProposal]:
        user_prompt = _build_user_prompt(fields)
        response = self._llm.request_json(
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
        )
        raw_classifications = response.get("classifications", [])
        proposals: List[FieldProposal] = []
        for item in raw_classifications:
            proposals.append(FieldProposal(
                field=str(item.get("field", "")),
                proposed_family=str(item.get("family", "NONE")),
                confidence=float(item.get("confidence", 0.0)),
                reasoning=str(item.get("reason", "")),
            ))
        return proposals

    def classify_all(
        self, loader: DatasetsLoader, output_path: str | Path,
    ) -> List[FieldProposal]:
        loader.load()
        metadata = loader.all_metadata()
        all_fields = list(metadata.values())

        all_proposals: List[FieldProposal] = []
        total = len(all_fields)
        logger.info("Classifying %d fields in batches of %d...", total, BATCH_SIZE)

        for start in range(0, total, BATCH_SIZE):
            batch = all_fields[start:start + BATCH_SIZE]
            logger.info("  Batch %d/%d (%d-%d)", start // BATCH_SIZE + 1, (total + BATCH_SIZE - 1) // BATCH_SIZE, start + 1, min(start + BATCH_SIZE, total))
            try:
                proposals = self.classify_batch(batch)
                all_proposals.extend(proposals)
            except Exception as exc:
                logger.warning("  Batch failed: %s", exc)
                for f in batch:
                    all_proposals.append(FieldProposal(
                        field=f.field_id,
                        proposed_family="NONE",
                        confidence=0.0,
                        reasoning=f"Classification failed: {exc}",
                    ))

        output = [asdict(p) for p in all_proposals]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info("Saved %d proposals to %s", len(all_proposals), output_path)
        return all_proposals


def load_proposals(path: str | Path) -> List[FieldProposal]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [
        FieldProposal(**item) if isinstance(item, dict) else item
        for item in raw
    ]


def approved_proposals_to_map(
    proposals: List[FieldProposal],
) -> Dict[str, List[str]]:
    family_map: Dict[str, List[str]] = {name: [] for name in ALL_FAMILY_NAMES}
    for p in proposals:
        if p.status == "approved" and p.proposed_family in family_map:
            family_map[p.proposed_family].append(p.field)
    return {k: v for k, v in family_map.items() if v}
