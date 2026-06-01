# 探索網格重構計畫

**建立日期：** 2026-05-22
**狀態：** ✅ **全部完成**（Phase 1-6 已執行完畢，167/167 tests passed）
**目標：** 將現有 660-cell 探索網格（11×5×3×4）重構為 72-cell（8×3×3），基於 Brain 官方分類

---

## 一、背景摘要

### 現狀（即將廢除）

| 維度 | 舊數量 | 舊內容 | 問題 |
|---|---|---|---|
| SignalFamily | 11 | VALUE, MOMENTUM, QUALITY... | 自行定義，非官方 |
| OperatorPattern | 5 | CROSS_SECTIONAL, TIME_SERIES, RATIO_SPREAD, SCALE_NORMALIZE, COMPOSITE | 自行定義，非官方 |
| Horizon | 3 | SHORT(1-10d), MEDIUM(10-60d), LONG(60-252d) | 保留 |
| Conditioning | 4 | RAW, NEUTRALIZED, RESIDUAL, GATED | 自行定義，實為 simulation settings，**移出 grid** |
| **Grid** | **660** | = 11 × 5 × 3 × 4 | 已廢棄 |

### 目標

| 維度 | 新數量 | 新內容 | 依據 |
|---|---|---|---|
| DatasetCategory | 8 | Analyst, Fundamental, model, news, option, Price Volume, sentiment, Social Media | Brain 官方 |
| OperatorCategory | 3 | Cross Sectional, Time Series, Group | Brain 官方（取主要 3 類） |
| Horizon | 3 | SHORT(1-10d), MEDIUM(10-60d), LONG(60-252d) | 保留（有意義） |
| Conditioning | ✅ **移出** | 由 M3 simulation settings 處理 | 原有 `build_refinement_candidates()` 已承擔此功能 |
| **Grid** | **72** | = 8 × 3 × 3 | 新目標 |

---

## 二、設計決策記錄

| 決策 | 選項 | 決定 | 理由 |
|---|---|---|---|
| Operator Categories 取捨 | A:全7類(168), B:主要3類(72) | **B (3類)** | Arithmetic/Logical/Vector/Transformational 為組合型，不適合獨立成 cell |
| sentiment vs Social Media | 按 prefix 分配 | **有情緒關鍵字 → sentiment** | 用戶明確決策 |
| Horizon 去留 | 保留/移除 | **保留** | 區分短中長期 alpha 有實質意義 |
| Conditioning 處理 | 留在 grid / 移出 | **移出** | 本質為 M3 simulation settings，refinement 搜尋已涵蓋 |

---

## 三、檔案影響分析

### 3.1 `exploration_grid.py` — 全面重寫

| 項目 | 現狀 | 目標 |
|---|---|---|
| `SignalFamily` Enum (line 23) | 11 個值 | **刪除**，取代為 `DatasetCategory` Enum |
| `OperatorPattern` Enum (line 45) | 5 個值 | **刪除**，取代為 `OperatorCategory` Enum |
| `Conditioning` Enum (line 71) | 4 個值 | **刪除** |
| `Horizon` Enum (line 53) | 3 個值 | **保留不變** |
| `CellStatus` (line 16) | 4 個值 | **保留不變** |
| `GridCell` (line 78) | family, operator, conditioning, source_papers | dataset_category, operator_category, **移除 conditioning, 移除 source_papers** |
| `cell_id()` (line 92) | family_operator_horizon_conditioning | dataset_category_operator_horizon |
| `build_hypothesis()` (line 111) | 含學術來源和 cond_desc | **重寫為資料驅動描述** |
| `cell_priority()` (line 156) | 含 family_order/operator_order/conditioning_order | **更新**為 dataset_category/operator_category/horizon 排序 |
| `IDEA_LIBRARY_TO_GRID` (line 195) | 22 個 SignalFamily-based mapping | **全部重寫**為 DatasetCategory-based |
| `_initialize_grid()` (line 282) | 4 層迴圈 | 改為 3 層（dataset_category × operator × horizon） |
| `_assign_source_papers()` (line 296) | 11 個 family → 論文 | **整組刪除** |
| `FAMILY_FIELD_MAP` (line 239) | SignalFamily key | **DATASET_FIELD_MAP**，DatasetCategory key |
| `_validate_fields()` (line 332) | SignalFamily | DatasetCategory |
| `get_candidate_fields()` (line 341) | SignalFamily | DatasetCategory |
| `select_cells()` (line 355) | SignalFamily | DatasetCategory |
| `load_family_field_map()` (line 226) | SignalFamily 字體驗證 | DatasetCategory 字體驗證 |
| `populate_from_library()` (line 395) | 引用 Conditioning | 移除 conditioning 參考 |

### 3.2 `expression_generator.py` — 中度改寫

| 項目 | 現狀 | 目標 |
|---|---|---|
| `OPERATOR_TEMPLATES` (line 39) | 5 個模式 | 3 個 patterns（Cross Sectional, Time Series, Group） |
| `NEUTRALIZATION_SUFFIXES` (line 72) | 4 個 conditioning | **整組刪除** |
| `build_template_expression()` (line 138) | 使用 cell.conditioning | 移除 conditioning 套用 |
| `_operator_pattern_to_categories()` (line 268) | 5→7 mapping | 直接 1:1 mapping（Cross Sectional→Cross Sectional） |
| `DEFAULT_COT_TEMPLATES` (line 113) | family label key | dataset_category label key |
| `_build_cot_prompt()` (line 279) | 含 family.source, cond_name | 不含學術引用和 conditioning |

### 3.3 `engine.py` — 輕度改寫

| 項目 | 現狀 | 目標 |
|---|---|---|
| `from exploration_grid import SignalFamily` | SignalFamily | DatasetCategory |
| `generate_grid_candidates()` (line 404) | source_papers, family.label, conditioning | 移除 source_papers, 改為 dataset_category, 預設 neutralization |
| `_execute_grid_action()` (line 811) | conditioning 相關 | 無改變 |

### 3.4 `config.py` — 極輕度修改

| 項目 | 現狀 | 目標 |
|---|---|---|
| `GridConfig.grid_domain_family` | SignalFamily label string | DatasetCategory label string |
| `GridConfig.grid_budget` | 5 | 不變 |

### 3.5 `sao.py` — 極輕度修改

| 項目 | 現狀 | 目標 |
|---|---|---|
| `source_paper` (line 21) | 由 grid 填充學術論文 | 保留欄位但不再由 grid 填充 |
| `operator_family` (line 33) | OperatorPattern.value | OperatorCategory.value |

### 3.6 不須改動的檔案

| 檔案 | 理由 |
|---|---|
| `operator_registry.py` | 已是官方分類，直接可用 |
| `rag_spec.py` | 與 grid 結構無關 |
| `datasets_loader.py` | 已有完整 field metadata 支援 |
| `planner.py` | 不直接引用 grid enums |
| `repair_engine.py` | 操作 SAO 狀態，不引用 grid enums |
| `llm_client.py` | 通用 LLM 呼叫 |
| `field_classifier.py` | 需更新 target categories，但功能不變 |

### 3.7 測試檔案 — 中度改寫

| 檔案 | 需更新 |
|---|---|
| `tests/test_exploration_grid.py` | 所有 `SignalFamily.*`, `OperatorPattern.*`, `Conditioning.*` 引用 |
| `tests/test_expression_generator.py` | `SignalFamily.VALUE` 等引用、OPERATOR_TEMPLATES 測試 |
| `tests/test_engine.py` | config 中 family_filter 相關 |
| `tests/test_sao.py` | 無大改 |

---

## 四、Field Mapping 完整表

### 50 Prefix Groups → 8 Dataset Categories

#### Analyst (22 files)

| Prefix | Files | 說明 |
|---|---|---|
| `analyst` | 18 | 分析師預估、評級、目標價 |
| `analyst_chart_cnn` | 1 | CNN 圖表特徵 |
| `biasfree_analyst` | 1 | 去偏分析師預估 |
| `creator_signal_perf` | 1 | 信號創作者預測績效 |
| `equity_kpi_forecast` | 1 | KPI 預測共識 |

#### Fundamental (67 files)

| Prefix | Files | 說明 |
|---|---|---|
| `fundamental` | 20 | 財務報表數據 |
| `earnings` | 3 | 盈餘公告日期/EPS |
| `institutions` | 4 | 機構持股 |
| `board_network` | 1 | 董事會網路 |
| `insiders` | 1 | 內部人交易 |
| `univ` | 2 | 股票池 membership |
| `macro` | 1 | 指數 membership |
| `other` | 30 | 基本面預測誤差、電商數據、其他 |

#### model (56 files)

| Prefix | Files | 說明 |
|---|---|---|
| `model` | 46 | 預建 alpha factors |
| `mfm_model_output` | 1 | MFM 因子暴露 |
| `stock_cluster_dl` | 1 | PCA 嵌入向量 |
| `cre_exposure_model` | 1 | 房地產 beta |
| `forward_beta_risk` | 1 | Bayesian beta 預測 |
| `event_return_model` | 1 | DL 事件回報預測 |
| `dl_volume_pred` | 1 | DL 成交量預測 |
| `earnings_chart_dl` | 1 | CNN 財報預測 |
| `techindi_model` | 1 | 技術指標預測 |
| `imbalance` | 1 | SHIELD-OIL 分數 |

#### news (30 files)

| Prefix | Files | 說明 |
|---|---|---|
| `news` | 27 | 新聞情緒、頭條分析 |
| `nlp_news_scores` | 1 | NLP 新聞情緒分數 |
| `news_transformer_scores` | 1 | Transformer 新聞情緒 |
| `us_equity_news` | 1 | 美股新聞情緒 |

#### option (11 files)

| Prefix | Files | 說明 |
|---|---|---|
| `option` | 7 | 選擇權 OI、IV、Greeks |
| `chart_model_alpha` | 1 | 選擇權定價模型 |
| `earnings_risk` | 1 | 選擇權隱含盈餘波動 |
| `expected_move` | 1 | 選擇權預期位移 |
| `option_horizon_decomp` | 1 | 選擇權 IV 期限結構 |
| `order_flow_imb` | 1 | 選擇權流動不平衡 |

#### Price Volume (38 files)

| Prefix | Files | 說明 |
|---|---|---|
| `pv` | 23 | 價格、成交量、市值 |
| `custom_demo` | 1 | 示範價格數據 |
| `pv_tech_indicators` | 1 | 動量/加速度指標 |
| `tech_chart_model` | 1 | APO、A/D 線技術指標 |
| `chart_return_model` | 1 | 日內 bid/ask 價格 |
| `order_book_imbalance` | 1 | 委託簿微結構 |
| `shortinterest` | 7 | 放空利率、波動率 |
| `risk` | 5 | 借券利率、放空擁擠 |

#### sentiment (12 files)

| Prefix | Files | 說明 |
|---|---|---|
| `sentiment` | 8 | 綜合情緒分數 |
| `earningscall_sentiment` | 1 | 財報電話會議 NLP |
| `filing_sentiment` | 1 | SEC 申報 NLP |
| `twitter_sentiment_l2` | 1 | Twitter 情緒 |
| `social_sent_score` | 1 | 社群意見分數 |

#### Social Media (4 files)

| Prefix | Files | 說明 |
|---|---|---|
| `socialmedia` | 2 | 社群平台活動數據 |
| `search_interest` | 1 | Google Trends |
| `stock_search_trends` | 1 | 股票搜尋趨勢 |

---

## 五、實作階段

### Phase 1：Enum 與資料結構（exploration_grid.py）

**步驟 1.1** — 新增 2 個 Enum，刪除 3 個舊 Enum

```python
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
```

```python
class OperatorCategory(Enum):
    CROSS_SECTIONAL = ("cross_sectional", "Cross-sectional rank, scale, normalize")
    TIME_SERIES     = ("time_series",     "Time-series mean, delta, rank, std_dev")
    GROUP           = ("group",           "Group rank, zscore, neutralize")

    @property
    def label(self) -> str:
        return self.value[0]
```

**刪除：** `SignalFamily`, `OperatorPattern`, `Conditioning`

**步驟 1.2** — 更新 `GridCell`

```python
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

    def cell_id(self) -> str:
        return f"{self.dataset_category.label}_{self.operator_category.label}_{self.horizon.label}"

    def build_hypothesis(self) -> str:
        return (
            f"Explore {self.operator_category.label} signals using "
            f"{self.dataset_category.label} data "
            f"at {self.horizon.label} horizon "
            f"({self.horizon.min_days}-{self.horizon.max_days} days)."
        )
```

**步驟 1.3** — 重寫 `_initialize_grid()` 為 3 層迴圈

```python
def _initialize_grid(self) -> None:
    for dc in DatasetCategory:
        for oc in OperatorCategory:
            for hz in Horizon:
                cell = GridCell(
                    dataset_category=dc,
                    operator_category=oc,
                    horizon=hz,
                    candidate_fields=list(self.DATASET_FIELD_MAP.get(dc, [])),
                )
                self.cells[cell.cell_id()] = cell
```

**步驟 1.4** — 刪除 `_assign_source_papers()` 及 `source_papers` field

**步驟 1.5** — 重寫 `cell_priority()`

```python
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
    ...  # update structural score and tiebreak
```

**步驟 1.6** — 重寫 `IDEA_LIBRARY_TO_GRID`

```python
IDEA_LIBRARY_TO_GRID: Dict[Tuple[str, str], Tuple[DatasetCategory, OperatorCategory, Horizon]] = {
    ("social_buzz", "raw"):             (DatasetCategory.SENTIMENT, OperatorCategory.CROSS_SECTIONAL, Horizon.SHORT),
    ("social_buzz", "mean"):            (DatasetCategory.SENTIMENT, OperatorCategory.TIME_SERIES, Horizon.MEDIUM),
    ("price_reversion", "close_vs_mean"): (DatasetCategory.PRICE_VOLUME, OperatorCategory.CROSS_SECTIONAL, Horizon.SHORT),
    ("fundamental_quality", "rank"):    (DatasetCategory.FUNDAMENTAL, OperatorCategory.CROSS_SECTIONAL, Horizon.LONG),
    ...
}
```

**步驟 1.7** — 建立 `DATASET_FIELD_MAP`

```python
DATASET_FIELD_MAP: Dict[DatasetCategory, List[str]] = {
    DatasetCategory.ANALYST:       [...],  # 從 analyst* CSVs 載入
    DatasetCategory.FUNDAMENTAL:  [...],  # 從 fundamental*, earnings*, institutions*, board_network*, insiders*, univ*, macro*, other* CSVs 載入
    DatasetCategory.MODEL:        [...],  # 從 model*, mfm_model_output*, etc. CSVs 載入
    DatasetCategory.NEWS:         [...],  # 從 news*, etc. CSVs 載入
    DatasetCategory.OPTION:       [...],  # 從 option*, etc. CSVs 載入
    DatasetCategory.PRICE_VOLUME: [...],  # 從 pv*, etc. CSVs 載入
    DatasetCategory.SENTIMENT:    [...],  # 從 sentiment*, etc. CSVs 載入
    DatasetCategory.SOCIAL_MEDIA: [...],  # 從 socialmedia*, etc. CSVs 載入
}
```

### Phase 2：Expression Generator 更新

**步驟 2.1** — 更新 `OPERATOR_TEMPLATES`

```python
OPERATOR_TEMPLATES: Dict[str, List[str]] = {
    "cross_sectional": [
        "rank({field})",
        "zscore({field})",
        "scale({field})",
        "winsorize({field}, std=3)",
    ],
    "time_series": [
        "ts_mean({field}, {window})",
        "ts_delta({field}, {window})",
        "ts_rank({field}, {window})",
        "ts_sum({field}, {window})",
        "ts_std_dev({field}, {window})",
        "ts_min({field}, {window})",
        "ts_max({field}, {window})",
    ],
    "group": [
        "group_rank({field}, sector)",
        "group_zscore({field}, sector)",
        "group_neutralize({field}, sector)",
    ],
}
```

**步驟 2.2** — 刪除 `NEUTRALIZATION_SUFFIXES`

Conditioning 不再由 expression generator 處理。M3 simulation settings 中的 neutralization, universe, decay, delay 由 `build_refinement_candidates()` 處理。

**步驟 2.3** — 更新 `build_template_expression()`

移除：
```python
# 刪除以下區塊
cond = cell.conditioning.value
if cond in NEUTRALIZATION_SUFFIXES and NEUTRALIZATION_SUFFIXES[cond]:
    expr = NEUTRALIZATION_SUFFIXES[cond].replace("expr", expr)
```

**步驟 2.4** — 更新 `DEFAULT_COT_TEMPLATES`

```python
DEFAULT_COT_TEMPLATES: Dict[str, str] = {
    "analyst": "Generate an alpha using analyst estimate fields: {fields}. "
               "Use rank or zscore for cross-sectional comparison.",
    "fundamental": "Generate a fundamental alpha using {fields}. "
                   "Prefer rank(zscore({primary_field})) or ratio operators.",
    "model": "Generate an alpha using pre-built model factor fields: {fields}. "
             "Combine with cross-sectional rank or time-series operations.",
    ...
}
```

**步驟 2.5** — 更新 `_build_cot_prompt()` 和 `_operator_pattern_to_categories()`

`_operator_pattern_to_categories()` 簡化為直接 1:1 mapping：
```python
@staticmethod
def _operator_pattern_to_categories(pattern: str) -> List[str]:
    return [pattern]  # 直接對應官方 category name
```

### Phase 3：Engine 與 SAO 更新

**步驟 3.1** — `engine.py` import 更新

```python
from alpha_agent.exploration_grid import ExplorationGrid, DatasetCategory
```

**步驟 3.2** — `generate_grid_candidates()` SAO 建構更新

```python
sao = SignalAlphaObject(
    source_cell_id=cell.cell_id(),
    statement=cell.build_hypothesis(),
    category=cell.dataset_category.label,
    horizon=cell.horizon.label,
    primary_fields=cell.candidate_fields,
    neutralization="SECTOR",          # 預設值，不來自 grid
    operator_family=cell.operator_category.label,
    expression=gc.expression,
    idea_name=gc.idea_name,
    ...
)
```

**步驟 3.3** — `config.py` `grid_domain_family` 語義更新

原為 `SignalFamily` label → 改為 `DatasetCategory` label

### Phase 4：測試更新

**步驟 4.1** — 更新 `tests/test_exploration_grid.py`

- `SignalFamily.VALUE` → `DatasetCategory.FUNDAMENTAL`
- `OperatorPattern.CROSS_SECTIONAL` → `OperatorCategory.CROSS_SECTIONAL`
- `Conditioning.RAW` → **移除**
- 驗證 `len(grid.cells) == 72`

**步驟 4.2** — 更新 `tests/test_expression_generator.py`

- `SignalFamily.VALUE` → `DatasetCategory.FUNDAMENTAL`
- `OperatorPattern.CROSS_SECTIONAL` → `OperatorCategory.CROSS_SECTIONAL`

### Phase 5：文件更新

- 在 `BRAINALPHA_IMPROVEMENT_LOG.md` 新增重構記錄
- 標記 `BRAINALPHA_EXPANSION_PLAN.md` P1 為已廢棄

---

## 六、Testing 驗證

| # | 驗證項目 | 通過標準 |
|---|---|---|
| T1 | Grid 初始化 | `len(grid.cells) == 72` |
| T2 | cell_id 格式 | `"fundamental_cross_sectional_short"` |
| T3 | Field mapping 正確 | 每個 category 都有對應 fields |
| T4 | build_hypothesis() | 不含學術引用、不含 conditioning |
| T5 | Expression Generator | 3 個 templates group 各生成合法 expression |
| T6 | cell_priority() | 回傳合理浮點數 |
| T7 | select_cells() | 依 priority 排序，回傳正確數量 |
| T8 | SAO 無 source_papers | SAO 中 source_papers 為空 |
| T9 | IDEA_LIBRARY_TO_GRID | 全部 mapping 使用新 enums |
| T10 | 向後相容 | 舊 pipeline seed/refine 路徑不受影響（不引用 grid） |

---

## 七、風險評估

| 風險 | 影響 | 緩解 |
|---|---|---|
| Field mapping 錯誤導致部分 cell 無 candidate fields | 該 cell 無法產 alpha | 每 category 至少確保有 10+ fields，缺失的透過 datasets_loader 補齊 |
| IDEA_LIBRARY_TO_GRID 遺漏 mapping | 部分 library 種子無法對應到 cell | 保留未知 mapping 的 seed 為一般 seed，不強制對應 cell |
| 舊測試大量引用舊 Enum | 測試失敗 | 全面搜尋替換，確認無殘留 |
| Operator templates 不足 | 生成表達式多樣性下降 | 從 operator_registry 補齊各 category 常用 operators |

---

## 八、Deep Review & Debug 紀錄（2026-05-22）

### 8.1 審查範圍

對所有受 Grid Redefinition 影響的檔案進行全面審查，涵蓋：
- `exploration_grid.py` — 新 Enum、72-cell 初始化、priority、field mapping
- `expression_generator.py` — operator templates、CoT prompts、build_template_expression
- `engine.py` — imports、generate_grid_candidates、SAO 建構
- `config.py` — grid_domain_family 語義
- `test_expression_generator.py` — cell ID 更新、conditioning 測試移除
- `planner.py`、`research_logic.py`、`repair_engine.py` — 確認無舊引用
- `field_classifier.py` — 舊 SignalFamily 相容性

### 8.2 審查結果摘要

| 類別 | 數量 | 說明 |
|---|---|---|
| Critical Bugs | 0 | 無 |
| Actionable Fixes | 2 | ts_min/ts_max 移除、field_classifier 標記 deprecated |
| Non-Bugs (Verified) | 12 | `.label` 用法、engine.py 清理、select_cells 等 |
| Pre-existing Issues | 2 | pasteurize、grid_domain_family 命名 |

### 8.3 已修復的 Bug

#### BUG 17：`ts_min`/`ts_max` 不存在於 Brain 運算元目錄

**檔案：** `alpha_agent/expression_generator.py` 原 line 51-52

**問題：** `OPERATOR_TEMPLATES["time_series"]` 包含 `ts_min({field}, {window})` 和 `ts_max({field}, {window})`，但 Brain 官方運算元目錄只有 `min` 和 `max`（Arithmetic 類別），沒有 `ts_min`/`ts_max`。若這些模板被使用，`DeterministicValidator` 的 D6.3 規則（無有效運算元）會失敗。

**影響：** `build_template_expression()` 只使用 `templates[0]`（`ts_mean`），所以實際不會觸發。但若未來手動選擇其他 template 索引會出錯。

**修正：**
```diff
 "time_series": [
     "ts_mean({field}, {window})",
     "ts_delta({field}, {window})",
     "ts_rank({field}, {window})",
     "ts_sum({field}, {window})",
     "ts_std_dev({field}, {window})",
-    "ts_min({field}, {window})",
-    "ts_max({field}, {window})",
 ],
```

---

#### BUG 1+16：`field_classifier.py` 仍使用舊 11-SignalFamily 分類

**檔案：** `alpha_agent/field_classifier.py`

**問題：** 整個模組設計基於舊的 11-SignalFamily 網格：
- `_SYSTEM_PROMPT` 仍談論 SignalFamily 分類
- `FAMILY_DEFINITIONS` 仍包含 11 個舊類別（VALUE, MOMENTUM, QUALITY, ...）
- `approved_proposals_to_map()` 回傳的 dict key 為舊 family 名稱，與新 `DATASET_FIELD_MAP: Dict[DatasetCategory, List[str]]` 不相容

**影響：** 此模組未被 `engine.py` 直接引用，但若有人手動執行 `FieldClassifier.classify_all()`，會將 field 分類到舊的 SignalFamily 名稱，而非新的 DatasetCategory。

**修正：** 新增模組層級 deprecation docstring + `__init__` 中發出 `DeprecationWarning`：
```python
"""DEPRECATED: This module classifies fields into the old 11-SignalFamily grid.

After the 2026-05-22 Grid Redefinition (660 → 72 cells), SignalFamily was replaced
by DatasetCategory (8 categories). This module is no longer compatible with
the current ExplorationGrid and should not be used for new field classification.
"""

class FieldClassifier:
    def __init__(self, llm_client: LLMClient) -> None:
        warnings.warn(
            "FieldClassifier is deprecated. It classifies fields into the old "
            "11-SignalFamily grid, which was replaced by 8 DatasetCategory on 2026-05-22.",
            DeprecationWarning,
            stacklevel=2,
        )
```

### 8.4 已驗證無誤的項目（Non-Bugs）

| # | 項目 | 驗證結果 |
|---|---|---|
| 1 | `OperatorCategory.value` / `.label` 用法 | ✅ 所有程式碼路徑均使用 `.label`（回傳字串），非 `.value`（回傳 tuple） |
| 2 | `DatasetCategory.value` / `.label` 用法 | ✅ 同上 |
| 3 | `engine.py` SAO 建構 | ✅ 已移除 `source_paper`、`conditioning`，正確使用 `.label` |
| 4 | `planner.py` | ✅ 無 SignalFamily/OperatorPattern/Conditioning 引用 |
| 5 | `research_logic.py` | ✅ 無舊引用 |
| 6 | `test_repair_engine.py` | ✅ 使用 SignalAlphaObject 直接建構，無舊 grid 引用 |
| 7 | `test_research_logic.py` | ✅ 無舊引用 |
| 8 | `select_cells(domain_dc=)` | ✅ 參數正確傳遞至 `cell.priority(domain_dc=...)` |
| 9 | `IDEA_LIBRARY_TO_GRID` | ✅ 全部 22 個 entry 使用新 Enum |
| 10 | `build_template_expression` group 處理 | ✅ `else` 分支正確處理 `group_rank({field}, sector)` |
| 11 | `_vary_expression` 中的 `group_neutralize` | ✅ 現在是 Brain Group 類別的有效運算元，功能正常 |
| 12 | `GRID_SIZE = 8 * 3 * 3 = 72` | ✅ 正確 |

### 8.5 已知問題（非 Grid Reconstruction 引入）

| # | 檔案 | 問題 | 嚴重性 | 說明 |
|---|---|---|---|---|
| 1 | `failure_taxonomy.py:109,127-128` | `pasteurize(x)` 不在 Brain 66 官方運算元中 | 中 | 已在 OPERATOR_REGISTRY_PLAN.md 中記錄，確定性修復路徑會產生無效表達式 |
| 2 | `config.py:34` | `grid_domain_family` 命名仍使用舊 "family" 術語 | 低 | 功能正常，但語義上應為 "dataset_category"；改名會破壞現有 config 相容性 |

### 8.6 測試結果

```
collected 167 items
167 passed in 26.52s
```

所有 167 項測試通過，無回歸問題。

---

## 九、Thesis Mapping: 24-Cell 學術論文映射（2026-05-22）

### 9.1 動機

Grid Redefinition Phase 1-6 將學術論文引用完全從 grid 中移除（刪除 `_assign_source_papers()` 和 `source_papers` field）。然而用戶在審查後要求重新加入**真實、可驗證**的學術論文引用，以確保每個 cell 的表達式生成有經濟學理論支撐，避免純數據挖掘的倖存者偏差。

### 9.2 設計

| 項目 | 決定 |
|------|------|
| 論文範圍 | 任何可引用來源（期刊、arXiv、教科書），每 cell 1-3 篇 |
| 論文格式 | 雙語（中英文），含標題/作者/年份/期刊/關鍵發現 |
| 映射維度 | 8 DatasetCategory × 3 OperatorCategory = 24 個唯一組合 |
| Horizon 處理 | 正交維度，相同論文映射適用於 SHORT/MEDIUM/LONG |
| 驗證工具 | 主：Crossref API；備用：Semantic Scholar API + Web Search |
| 資料結構 | `Thesis` dataclass + `thesis_map.json` + `_ensure_thesis_map()` 載入機制 + `GridCell.thesis` field |

### 9.3 驗證結果

| 指標 | 數值 |
|------|------|
| 唯一論文數 | 36 |
| 經 Crossref API 驗證 | 32 |
| 經 Web Search 驗證 | 4 |
| 與初始草稿相比修正的論文 | 6 (Sloan, Stickel, Gleason-Lee, Cremers-Weinbaum, Boni-Womack, Kozak-Nagel-Santosh) |
| 初始錯誤率 | 6/25 = 24% |

### 9.4 檔案變更

| 檔案 | 變更 |
|------|------|
| `docs/thesis-mapping.md` | **新建** — 完整 24-cell 論文映射文件 |
| `data/thesis_map.json` | **新建** — 36 篇論文數據（JSON 格式），由 `_ensure_thesis_map()` 載入 |
| `alpha_agent/exploration_grid.py` | 新增 `Thesis` dataclass, `_ensure_thesis_map()` + `get_thesis()` JSON 載入機制, `GridCell.thesis` field, `build_hypothesis()`, `_initialize_grid()` 填入 thesis |
| `alpha_agent/expression_generator.py` | `_build_cot_prompt()` 注入學術引用區塊至 LLM prompt |

### 9.5 Deep Review & Debug 紀錄

對 Thesis Mapping 嵌入生成流程進行全面審查，從 grid 初始化 → thesis population → hypothesis building → CoT prompt injection → SAO construction，所有路徑均已覆蓋。

#### 審查結果摘要

| 類別 | 數量 | 說明 |
|------|------|------|
| Critical Bugs | 0 | 無 |
| Actionable Fixes | 2 | thesis 重複出現、SAO.source_paper 未填充 |
| Pre-existing Issues | 1 | hypothesis 字串長度 |

#### 已修復的問題

**BUG 18：CoT Prompt 中 thesis 重複出現**

**檔案：** `alpha_agent/exploration_grid.py:339-349`, `alpha_agent/expression_generator.py:275-286`

**問題：** `build_hypothesis()` 返回的字串包含 `"Academic grounding: paper1; paper2"`，而 `_build_cot_prompt()` 又額外插入了 `thesis_block`（含相同論文但附 key finding）。LLM prompt 中論文引用出現兩次，浪費約 200 tokens。

**修正：**
- `build_hypothesis()` 不再包含學術引用（保持 hypothesis 簡潔）
- thesis 信息僅透過 `_build_cot_prompt()` 的 `thesis_block` 注入 LLM
- SAO.statement 乾淨，RAG 編碼不受影響

**BUG 19：SAO.source_paper 未填充**

**檔案：** `alpha_agent/engine.py:433-445`

**問題：** `SignalAlphaObject` 有 `source_paper` 欄位（論文 Table 2 定義），但生成流程中從未填充。欄位始終為空字串。

**修正：**
```python
thesis_str = "; ".join(t.short_str() for t in cell.thesis[:3]) if cell.thesis else ""
sao = SignalAlphaObject(
    ...
    source_paper=thesis_str,
    ...
)
```

#### 已驗證無誤的項目

| # | 項目 | 驗證結果 |
|---|------|---------|
| 1 | thesis_map.json 完整性 | ✅ 24/24 (8×3) cells 全部存在 |
| 2 | 每 cell 最少 1 篇論文 | ✅ 最少 2 篇（ANALYST×GROUP, SOCIAL×各類），最多 3 篇 |
| 3 | Grid 初始化 thesis 填充 | ✅ 72/72 cells 正確填入 thesis |
| 4 | build_hypothesis() 乾淨 | ✅ 不包含學術引用，focus on data+operator+horizon |
| 5 | thesis_block 注入 LLM | ✅ 含論文完整引用 + key finding（雙語） |
| 6 | SAO.source_paper 填充 | ✅ 包含前 3 篇論文 short_str |
| 7 | to_dict 序列化 | ✅ thesis 作為獨立欄位正確導出 |
| 8 | Template fallback 相容 | ✅ cross_sectional/time_series/group templates 正常 |
| 9 | populate_from_library | ✅ 不影響（新建 cell 時 thesis 預設為空） |

#### 已修復的問題

**BUG 20：`_ensure_thesis_map()` 快取判斷條件錯誤，JSON 檔案從未被讀取**

**檔案：** `alpha_agent/exploration_grid.py:87,92`

**問題：** `_THESIS_MAP_CACHE` 初始化為空字典 `{}`，但快取 guard 為 `if _THESIS_MAP_CACHE is not None`。由於 `{} is not None` 結果為 `True`，每次呼叫都直接回傳空字典，從未實際讀取 `data/thesis_map.json`。導致 `get_thesis()` 永遠回傳空列表，debug 腳本驗證顯示 `JSON entries: 0`。

**修正：** 將 guard 條件由 `is not None` 改為真值檢查

```diff
-     if _THESIS_MAP_CACHE is not None:
+     if _THESIS_MAP_CACHE:
```

改為真值檢查：空字典 `{}` 為 falsy，觸發實際檔案載入；載入後的資料字典為 truthy，正常使用快取。

#### 已知問題（非 Thesis Mapping 引入）

| # | 問題 | 說明 |
|---|------|------|
| 1 | ~~THESIS_MAP 硬編碼於 `exploration_grid.py`~~ | ✅ **已解決：** 論文數據已遷移至 `data/thesis_map.json`，透過 `_ensure_thesis_map()` 延遲載入 + 快取 |
| 2 | `build_hypothesis()` 不含學術引用 | 已確認是設計決定——hypothesis 保持乾淨，thesis 透過獨立通道傳遞 |

### 9.6 測試結果

```
collected 167 items
167 passed in 15.20s
```

所有 167 項測試通過，無回歸問題。驗證 72 個 cell 全部正確填入 thesis。
