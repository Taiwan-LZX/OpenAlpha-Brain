# Plan: CSV Field Metadata Integration

## Overview

將 `datasets/` 下 238 個 CSV 的 field metadata（Description, Type, Coverage, Users, Alphas）整合進 pipeline，取代目前僅依賴 `wqb_data_fields_summary.json`（純 ID）和硬編碼 `FAMILY_FIELD_MAP`（~70 field）的現狀。

---

## Current State vs Target State

| 面向 | 現況 | 改進後 |
|------|------|--------|
| **field 來源** | 硬編碼 ~70 個 field ID | CSV 238 個 dataset 全量載入 |
| **field 驗證** | 無（`pre_tax_income` 不存在但無警告） | 啟動時驗證，不存在 / coverage=0% 的 field 會警示 |
| **中繼資料** | JSON 只有 ID，無 Description / Type | 完整 Description、Type、Coverage、Users、Alphas |
| **RAG 推薦** | 純 field ID 字元比對（TF-IDF 無語義） | 用 Description 做語義比對，推薦更精準 |
| **表達式生成** | LLM 只看 field ID，不知道欄位意義 | LLM 可讀 Description，產生更貼合的表達式 |
| **MAP 擴充** | 手動維護，需開發者自己查 CSV | LLM 讀 Description 初判，開發者審核 |
| **coverage 過濾** | 無（可能用到 coverage=0% 的 field） | 可設定最低 coverage 門檻 |

### 具體數字

| 指標 | 現況 | 改進後 |
|------|------|--------|
| FAMILY_FIELD_MAP 欄位數 | ~70 | 預估 200-500（經 LLM + 審核） |
| 可探索表達式組合 | ~70 × 5 × 3 × 4 = **4,200** | 預估 ~200 × 5 × 3 × 4 = **12,000** |
| 錯誤 field 數 | 至少 1 個（`pre_tax_income`） | 0（啟動時驗證） |
| RAG 推薦品質 | 字元層級比對 | 語義層級比對 |

---

## Phase 0: 命名修正（1 行）

**目標：** 修正 `pre_tax_income` → `pretax_income`

**檔案：** `alpha_agent/exploration_grid.py:222`

**理由：**
- `pre_tax_income` 不存在於任何 CSV 或 `wqb_data_fields_summary.json`
- CSV `fundamental6_fields_formatted.csv` 中實際欄位為 `pretax_income`
- JSON 中也是 `pretax_income`

**變更：** 1 行字串替換。

**影響：** 無下游破壞，`SignalAlphaObject.primary_fields` 會正確傳遞。

---

## Phase 1: Build DatasetsLoader

**新檔案：** `alpha_agent/datasets_loader.py`

### FieldMetadata dataclass

```python
@dataclass
class FieldMetadata:
    field_id: str
    description: str
    type: str           # MATRIX | VECTOR | GROUP | SYMBOL | UNIVERSE
    coverage: float     # 0.0 - 100.0
    users: int
    alphas: int
    dataset_id: str     # 來源 dataset（從檔名推導）
```

### DatasetsLoader class

```python
class DatasetsLoader:
    def __init__(self, datasets_dir: str | Path): ...
    def load(self) -> None: ...           # 掃描、解析、建立索引
    def get_metadata(self, field_id: str) -> Optional[FieldMetadata]: ...
    def get_fields_by_dataset(self, dataset_id: str) -> List[FieldMetadata]: ...
    def search_fields(self, keyword: str) -> List[FieldMetadata]: ...
    def all_field_ids(self) -> Set[str]: ...
    def all_metadata(self) -> Dict[str, FieldMetadata]: ...
```

### 實作要點

1. **CSV 解析：** 用 `csv.DictReader` 解析所有 `*_fields_formatted.csv`
2. **索引：** `Dict[str, FieldMetadata]`，key 為 field_id
3. **記憶體：** 238 檔 × 平均 ~40 fields = ~9,500 筆 metadata，dict 約 3-5 MB
4. **防禦性程式：** 檢查 Field 是否為合法 snake_case，不合規者 skip + log warning

### 測試（`tests/test_datasets_loader.py`）

- 測試 CSV 解析正確性
- 測試 field ID 唯一性（無跨 dataset 重複）
- 測試 search / 過濾功能
- 測試 coverage 轉換（"39%" → 39.0）

---

## Phase 2: Integrate into ExplorationGrid

### exploration_grid.py 變更

1. **ExplorationGrid.__init__** 新增可選參數：
```python
def __init__(self, field_metadata: dict | None = None) -> None:
    self.field_metadata = field_metadata or {}
    ...
```

2. **FAMILY_FIELD_MAP 驗證**（`_initialize_grid` 之後呼叫 `_validate_fields`）：
```python
def _validate_fields(self) -> List[str]:
    """檢查 FAMILY_FIELD_MAP 中的 field 是否存在於 CSV，以及 coverage 是否 > 0%"""
    warnings = []
    for family, fields in self.FAMILY_FIELD_MAP.items():
        for f in fields:
            meta = self.field_metadata.get(f)
            if meta is None:
                warnings.append(f"[{family.label}] field '{f}' not found in any CSV")
            elif meta.coverage == 0:
                warnings.append(f"[{family.label}] field '{f}' has 0% coverage")
    return warnings
```

3. **Coverage 過濾**（可選門檻）：
```python
def get_candidate_fields(self, family: SignalFamily, min_coverage: float = 0) -> List[str]:
    fields = self.FAMILY_FIELD_MAP.get(family, [])
    if not self.field_metadata or min_coverage <= 0:
        return fields
    return [f for f in fields if self.field_metadata.get(f, EMPTY_META).coverage >= min_coverage]
```

### engine.py 變更

在 `ResearchToolbox.ensure_grid()` 中傳入 DatasetsLoader：

```python
def ensure_grid(self) -> Any:
    ...
    if self._grid is None:
        loader = DatasetsLoader(self.agent_cfg.datasets_dir)
        loader.load()
        self._grid = ExplorationGrid(field_metadata=loader.all_metadata())
    ...
```

### 影響範圍

- `exploration_grid.py` API 向後相容（新參數為 optional）
- 現有 `engine.py` 呼叫點增加兩行
- `tests/test_expression_generator.py` 不受影響（未傳 metadata 時行為不變）

---

## Phase 3: Enrich RAGSpecEncoder with CSV Descriptions

### rag_spec.py 變更

1. **新增 factory method：**
```python
@classmethod
def from_csv_metadata(cls, metadata: Dict[str, FieldMetadata]) -> RAGSpecEncoder:
    """用 CSV Description 建立 TF-IDF corpus，取代純 field ID 字元比對。"""
    field_ids = list(metadata.keys())
    corpus = [metadata[fid].description for fid in field_ids]
    field_index = {fid: i for i, fid in enumerate(field_ids)}
    ...
```

2. **保留舊 method** `from_fields_summary` 作為 fallback。

3. **DeterministicValidator 擴充：**
```python
@classmethod
def from_csv_metadata(cls, metadata: Dict[str, FieldMetadata]) -> DeterministicValidator:
    known_fields = set(metadata.keys())
    return cls(known_fields=known_fields)
```

### 效果

| 查詢 | 現況（純 ID TF-IDF） | 改進後（Description TF-IDF） |
|------|---------------------|---------------------------|
| "value factor" | 匹配 `value_*` pattern | 匹配含 "book value", "earnings", "cash flow" 描述的 field |
| "sentiment buzz" | 匹配 `*buzz*` pattern | 匹配含 "social media sentiment", "buzz score" 描述的 field |
| "short interest" | 匹配 `*short*` pattern | 匹配含 "short interest ratio", "days to cover" 描述的 field |

---

## Phase 4: LLM-Assisted Field Classification

### 新檔案：`alpha_agent/field_classifier.py`

```python
class FieldClassifier:
    def __init__(self, llm_client: LLMClient, model_config: ModelConfig): ...
    def classify_batch(self, fields: List[FieldMetadata]) -> List[FieldProposal]: ...
    def classify_all(self, loader: DatasetsLoader) -> List[FieldProposal]: ...
```

### 流程

```
CSV metadata (Field, Description, Type, dataset)
        ↓
   LLM 批量分類 (每批 25 field)
        ↓
   field_family_proposals.json
        ↓
   人為審核 (pending → approved / rejected)
        ↓
   approved 寫入 field_family_map.json (外部 config)
        ↓
   ExplorationGrid 讀取外部 config 更新 FAMILY_FIELD_MAP
```

### LLM Prompt 設計

**System prompt：**
```
你是量化因子分類專家。將 data field 歸類到以下 SignalFamily：
  VALUE (價值): Fama-French HML — 淨值、現金流、盈餘 (例: bookvalue_ps, ebitda, cashflow_op)
  MOMENTUM (動能): Jegadeesh-Titman — 價格報酬趨勢 (例: close, returns, high, low)
  QUALITY (品質): Novy-Marx — 獲利能力、資產效率、償債 (例: return_equity, return_assets, debt)
  LOW_VOLATILITY (低波動): Frazzini-Pedersen BAB (例: returns, volume, adv20)
  SIZE (規模): Banz, Fama-French SMB — 市值、流通股數 (例: cap, sharesout)
  SENTIMENT (情緒): Baker-Wurgler — 社群情緒、分析師情緒 (例: scl12_buzz, snt_value)
  NEWS_ATTENTION (新聞關注): Tetlock — 新聞影響、事件關注 (例: nws18_relevance, news_ratio_vol)
  SHORT_REVERSAL (短反轉): Jegadeesh, Lehmann — 短期價格反轉 (例: close, vwap, returns)
  LIQUIDITY (流動性): Amihud, Pastor-Stambaugh (例: volume, adv20, cap)
  GROWTH (成長): LSLV — 營收/盈餘成長 (例: sales_growth, eps, revenue)
  TECHNICAL_TREND (技術趨勢): 價格與成交量技術指標 (例: close, volume, vwap, adv20)
  NONE: 不屬於上述任何家族（如 identifier、symbol type）
```

**User prompt（每批）：**
```json
{
  "fields": [
    {"id": "fnd6_mfma1_at", "description": "Total Assets (Quarterly)", "type": "MATRIX", "dataset": "fundamental6"},
    {"id": "scl12_buzz", "description": "Social media buzz score", "type": "MATRIX", "dataset": "socialmedia12"}
  ]
}
```

**LLM 輸出：**
```json
{
  "classifications": [
    {"field": "fnd6_mfma1_at", "family": "QUALITY", "confidence": 0.92, "reason": "Total Assets 為品質/規模基礎指標"},
    {"field": "scl12_buzz", "family": "SENTIMENT", "confidence": 0.97, "reason": "社群媒體熱度為情緒因子"}
  ]
}
```

### 輸出檔案：`field_family_proposals.json`

```json
[
  {
    "field": "scl12_buzzvec",
    "proposed_family": "SENTIMENT",
    "confidence": 0.97,
    "reasoning": "社群情緒向量為情緒因子",
    "status": "pending",
    "reviewer_notes": null
  }
]
```

### field_family_map.json（審核後匯入）

```json
{
  "VALUE": ["bookvalue_ps", "return_equity", "pretax_income", "..."],
  "MOMENTUM": ["close", "returns", "high", "low", "open", "..."],
  ...
}
```

### ExplorationGrid 整合

```python
# exploration_grid.py
import json
from pathlib import Path

class ExplorationGrid:
    FAMILY_FIELD_MAP: Dict[SignalFamily, List[str]] = {
        # ... 保持為 fallback
    }

    @classmethod
    def load_family_field_map(cls, path: str | Path) -> None:
        """從外部 JSON 載入 field-to-family 對應，覆蓋硬編碼 MAP。"""
        with open(path) as f:
            raw = json.load(f)
        for key, fields in raw.items():
            family = SignalFamily[key]
            cls.FAMILY_FIELD_MAP[family] = fields
```

---

## Execution Order

| # | Phase | 依賴 | 產出 |
|---|-------|------|------|
| 1 | Phase 0 | 無 | 1 行修正 |
| 2 | Phase 1 | 無 | `datasets_loader.py` + `test_datasets_loader.py` |
| 3 | Phase 2 | Phase 1 | `exploration_grid.py` + `engine.py` 整合 |
| 4 | Phase 3 | Phase 1 | `rag_spec.py` 加強 |
| 5 | Phase 4 | Phase 1 | `field_classifier.py` + proposals + config |

Phase 0-3 可合併執行（總計 ~200 行新 code + ~100 行測試）。
Phase 4 獨立執行（需 LLM API，耗時取決於 field 數量）。

---

## Risks & Mitigations

| 風險 | 緩解 |
|------|------|
| DatasetsLoader 增加啟動時間 | 238 檔約 0.5-1s，一次性快取 |
| LLM 分類錯誤 | confidence < 0.7 標示 low，人為優先審 |
| MAP 擴充後組合爆炸 | coverage + users 門檻過濾 |
| CSV 資料過時與線上資料不一致 | 目前以 CSV 為主，後續可加入 SQLite fallback |

---

## Rollback

- `FAMILY_FIELD_MAP` 硬編碼保持不變，外部 config 為 optional
- 不傳 metadata 時 pipeline 行為與現況完全一致
- `rag_spec.py` 保留 `from_fields_summary` 舊方法

---

## Review & Debug 紀錄

### 執行後發現並修復的 Bug

| # | 檔案 | 問題 | 嚴重度 | 修復方式 |
|---|------|------|--------|----------|
| 1 | `datasets_loader.py:42` | `load()` 非冪等，重複呼叫會累加資料 | **高** | 開頭加 `self._metadata.clear()` 和 `self._dataset_fields.clear()` |
| 2 | `datasets_loader.py:49-67` | `row.get("Coverage", "0%")` 在 DictReader 欄位缺失時返回 `None` 而非預設值（`dict.get` 的 default 只在 key 不存在時生效） | **高** | 全部改為 `(row.get("X") or "default").strip()` |
| 3 | `engine.py:323` | `_ensure_metadata()` 被 `ensure_grid`、`ensure_rag_encoder`、`ensure_validator` 各自呼叫，重複建立 DatasetsLoader 讀取 238 個 CSV | **高** | 加 `self._field_metadata` 快取，只讀一次 |

### 程式碼清理

| # | 檔案 | 問題 | 修復 |
|---|------|------|------|
| 4 | `field_classifier.py:5,10` | 未使用的 import `field`（dataclasses）、`SignalFamily` | 移除 |
| 5 | `exploration_grid.py:225` | 死旗標 `_EXTERNAL_MAP_LOADED`（寫入但從未讀取） | 移除 |

### 驗證結果

- **169 項測試全部通過**（含 15 項新測試 `test_datasets_loader.py`）
- 冪等性驗證：`load()` 兩次呼叫結果一致（59,453 筆）
- 所有模組 import 正常：`engine.py`、`field_classifier.py`、`exploration_grid.py`、`rag_spec.py`
- 整合驗證：DatasetsLoader → ExplorationGrid → RAGSpecEncoder → DeterministicValidator 全鏈路正常

### 變更檔案清單

| 檔案 | 變更類型 |
|------|----------|
| `alpha_agent/datasets_loader.py` | 新建（119 行） |
| `alpha_agent/field_classifier.py` | 新建（200 行） |
| `alpha_agent/exploration_grid.py` | 修改（+45 行） |
| `alpha_agent/engine.py` | 修改（+40 行） |
| `alpha_agent/rag_spec.py` | 修改（+19 行） |
| `alpha_agent/config.py` | 修改（+2 行） |
| `alpha_research_pipeline.py` | 修改（+1 行） |
| `tests/test_datasets_loader.py` | 新建（176 行） |
| `docs/field-metadata-csv-integration.md` | 新建（本文件） |
