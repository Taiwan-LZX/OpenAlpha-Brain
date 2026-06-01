# 探索網格重新定義 — 發現過程與始末

## 緣起

原本的探索網格 (Exploration Grid) 定義為：

```
Grid = 11 SignalFamily × 5 OperatorPattern × 3 Horizon × 4 Conditioning = 660 cells
```

所有的維度與內容都是基於「論文 BrainAlpha 說 660 = 11 × 5 × 3 × 4」這個公式，但具體內容是我們自行根據學術文獻定義的，論文明確列出的只有維度數量，沒有各維度的實際內容。

---

## 一、OperatorPattern 探索

### 發現歷程

1. 從 `components/operators/operators.json` 讀取 Brain 官方 operators
2. 發現官方有 **66 個 operators，分屬 7 大類別**

### 官方 7 大 Operator Categories

| 類別 | 數量 | 範例 |
|---|---|---|
| Time Series | 24 | `ts_mean`, `ts_corr`, `ts_decay_linear`, `ts_rank`, `ts_std_dev` |
| Arithmetic | 15 | `add`, `subtract`, `multiply`, `divide`, `log`, `power`, `abs` |
| Logical | 11 | `if_else`, `and`, `or`, `not`, `greater`, `less` |
| Cross Sectional | 6 | `rank`, `zscore`, `scale`, `winsorize`, `normalize`, `quantile` |
| Group | 6 | `group_rank`, `group_zscore`, `group_neutralize`, `group_scale` |
| Vector | 2 | `vec_avg`, `vec_sum` |
| Transformational | 2 | `bucket`, `trade_when` |

### 關鍵發現

原本的 OperatorPattern（5 個：CROSS_SECTIONAL, TIME_SERIES, RATIO_SPREAD, SCALE_NORMALIZE, COMPOSITE）**是我們自行定義的抽象組合模式**，而非官方分類。官方分類是**原子函數**層級。

原先的 5 個 OperatorPattern 與官方 7 類的對應存在本質差異：
- CROSS_SECTIONAL 對應 Cross Sectional → 合理
- TIME_SERIES 對應 Time Series → 合理
- RATIO_SPREAD 對應 Arithmetic（組合型）→ mapping 較弱
- SCALE_NORMALIZE 對應 Group + Cross Sectional → mapping 較弱
- COMPOSITE 對應多類別組合 → mapping 弱

---

## 二、Dataset 探索

### 發現歷程

1. 使用 explore agent 掃描 `data/datasets/` 目錄
2. 共找到 **238 個 CSV 檔案**
3. 按 filename prefix 分組，共 **50 個 prefix groups**
4. 嘗試將 50 個 prefix groups mapping 到我們自定義的 11 個 SignalFamily

### 原始探索結果（50 prefix groups → 11 SignalFamily）

| SignalFamily | 對應 Prefix Groups | 檔案數 |
|---|---|---|
| FUNDAMENTAL | fundamental, earnings | 23 |
| ANALYST | analyst, biasfree_analyst, creator_signal_perf, equity_kpi_forecast | 22 |
| TECHNICAL | model(46), chart_return_model, pv_tech_indicators, tech_chart_model, order_book_imbalance | 51 |
| SENTIMENT | sentiment, socialmedia, twitter_sentiment_l2, earningscall_sentiment, filing_sentiment | 14 |
| NEWS | news(27), nlp_news_scores, news_transformer_scores, us_equity_news | 30 |
| OPTION | option(7), chart_model_alpha, earnings_risk, expected_move, order_flow_imb | 11 |
| PV | pv(23), custom_demo | 24 |
| MACRO | macro | 1 |
| META | univ | 2 |
| ALTERNATIVE | other(30), institutions(4), shortinterest(7), risk(5), board_network, insiders... | 48 |
| ETF | *(無)* | 0 |

### 用戶指正：官方 Dataset 分類

用戶指出 Brain 官方將 datasets 分為 **8 類**，非我們定義的 11 類：

| 官方 Dataset Category | 對應的 Prefix Groups |
|---|---|
| **Analyst** | analyst, biasfree_analyst, creator_signal_perf, equity_kpi_forecast, analyst_chart_cnn |
| **Fundamental** | fundamental, earnings |
| **model** | model (46 files — 最大單一群組) |
| **news** | news, nlp_news_scores, news_transformer_scores, us_equity_news |
| **option** | option, chart_model_alpha, earnings_risk, expected_move, order_flow_imb, option_horizon_decomp |
| **Price Volume** | pv, custom_demo |
| **sentiment** | sentiment, socialmedia, twitter_sentiment_l2, earningscall_sentiment, filing_sentiment, social_sent_score |
| **Social Media** | (部分 sentiment 相關) |

### 關鍵發現

SignalFamily（11 個）是**我們自行定義**的，基於經典學術因子文獻（Fama-French 1992, Jegadeesh-Titman 1993 等），Brain 官方 dataset 分類僅有 8 類。

---

## 三、Conditioning 探索

### 原始定義

原本的 4 個 Conditioning 是我們自行定義的，基於學術文獻：

| Conditioning | 含義 | 學術對應 |
|---|---|---|
| RAW | 無條件化 | 原始因子暴露 |
| NEUTRALIZED | 行業/子行業中性化 | Moskowitz-Grinblatt 1999 |
| RESIDUAL | 市場/規模回歸殘差 | Carhart 1997 |
| GATED | 流動性/波動率條件 | Amihud 2002 |

### 用戶指正：官方 Conditioning

用戶指出 Brain 官方的 Conditioning 應為：

| 官方 Conditioning | Brain 對應 |
|---|---|
| **Neutralization** | `neutralize()` 函數，參數: SECTOR, INDUSTRY, SUBINDUSTRY |
| **Universe** | 股票池過濾: TOP3000, TOP2000, TOP1000, TOP500, TOP200, TOPSP500 |
| **Decay** | 衰減: `ts_decay_linear()`, `ts_decay_exp()`，參數: 0, 2, 4, 6, 8 |
| **Delay** | 延遲避免 look-ahead bias: `delay()` |

---

## 四、關鍵轉折點

### 4.1 自定義 vs 官方

逐維度確認各項是否為論文明確列出後，發現：

| 維度 | 數量 | 內容 | 來源 |
|---|---|---|---|
| SignalFamily | 11 | VALUE, MOMENTUM, QUALITY... | **我們自定義**（學術文獻），論文未列出 |
| OperatorPattern | 5 | CROSS_SECTIONAL, TIME_SERIES, RATIO_SPREAD... | **我們自定義**，論文未列出 |
| Horizon | 3 | SHORT, MEDIUM, LONG | **我們自定義**，論文未列出 |
| Conditioning | 4 | RAW, NEUTRALIZED, RESIDUAL, GATED | **我們自定義**（學術文獻），論文未列出 |

**論文只在 Section 4.1.1 說明了「660 = 11 × 5 × 3 × 4」公式，但從未列出每個維度的具體內容。** 所有內容都是我們基於對學術文獻的理解自行定義的。

### 4.2 從「Cell 數量」到「Pipeline 流程」

用戶引導思考轉向：

> 「Cell 主要用途是啥？總數目一點都不重要」
> 「不應該僅看 cells，而是看整個流程節點，各節點 input 是什麼？output 又是什麼？」

這引出了 key insight：**cell 的本質是結構化地定義 alpha 假說類型，數字多少不重要，重要的是每個 cell 代表的假說類型是否有意義。**

### 4.3 Pipeline 完整分析

```
M0: thesis generation (cell selection)
    Input:  exploration grid, paper library, field catalogue
    Output: selected cell + investment thesis statement

M1: RAG specification
    Input:  thesis, field catalogue (vector index)
    Output: specification (primary_fields, neutralization, lookback, operator_family)

M2: expression generation
    Input:  specification, operator vocabulary, Jaccard gate
    Output: Brain expression (e.g. rank(ts_mean(returns,20)))

M3: Brain simulation
    Input:  expression + settings (neutralization, decay, universe, delay, ...)
    Output: sharpe, fitness, checks, status, failure_mode

M4: repair engine
    Input:  SAO with failure_mode
    Output: modified expression/settings → back to M3
```

### 4.4 Conditioning 的真正定位

分析現有程式碼後發現關鍵：**Conditioning（Neutralization, Universe, Decay, Delay）不是 grid dimension，而是 M3 的 Brain simulation settings。**

現有程式碼 `alpha_research_pipeline.py:1029` 的 `build_refinement_candidates()` 已經是對這些參數做 grid search：

```python
decay_choices = (0, 2, 4, 6, 8)
neutralizations = ("SECTOR", "INDUSTRY", "SUBINDUSTRY")
universes = ("TOP3000", "TOP1000", "TOP500")
truncations = (0.05, 0.08, 0.1)
```

**兩者是不同層級的概念：**
- **Grid Cell** → 決定 alpha 假說的本質（用什麼資料 + 什麼運算 + 什麼時間尺度）
- **Settings** → 決定 Brain simulation 的參數配置（neutralization, universe, decay, delay）

---

## 五、最終提案：重新定義的 Cell 結構

### Cell 三維度

| 維度 | 數量 | 內容 | 依據 |
|---|---|---|---|
| **Dataset Category** | 8 | Analyst, Fundamental, model, news, option, Price Volume, sentiment, Social Media | Brain 官方 |
| **Operator Category** | 7 | Cross Sectional, Time Series, Arithmetic, Logical, Group, Vector, Transformational | Brain 官方 |
| **Horizon** | 3 | Short(1-10d), Medium(10-60d), Long(60-252d) | alpha 假說的本質差異 |

**Total: 8 × 7 × 3 = 168 cells**

### 移出 Grid 的項目

| 原項目 | 替代/處理方式 |
|---|---|
| **Conditioning** (RAW/NEUTRALIZED/RESIDUAL/GATED) | 全部移出。Neutralization, Universe, Decay, Delay 等列為 M3 Brain simulation settings，由 `build_refinement_candidates()` 處理 |
| **Fields** (~59K) | 不作為維度。分配到對應的 Dataset Category 作為候選欄位 |
| **SignalFamily** (VALUE, MOMENTUM, QUALITY...) | 被官方 Dataset Category 取代 |

### Cell 實例

```
Cell: (Fundamental, Cross Sectional, Long)
→ 假說類型: 「用基本面數據做橫截面排序的長期 alpha」
→ 候選 fields: [bookvalue_ps, return_equity, ebit, ...]
→ 可用 operators: [rank, zscore, scale, ...]
→ Settings (M3): neutralization=SECTOR, universe=TOP3000, decay=4, delay=1
→ 輸出: rank(bookvalue_ps), rank(return_equity), zscore(ebit) ...
```

```
Cell: (news, Logical, Short)
→ 假說類型: 「用新聞數據做邏輯判斷的短期 alpha」
→ 候選 fields: [nws18_relevance, nws18_nip, ...]
→ 可用 operators: [if_else, and, or, greater, less, ...]
→ Settings (M3): neutralization=INDUSTRY, universe=TOP1000, decay=0, delay=0
→ 輸出: if_else(greater(nws18_relevance, 0.5), rank(close), 0)
```

---

## 六、影響分析：現有程式碼中「固定論文」的全面崩塌

### 6.1 「固定論文」在現有程式碼中的三個層次

#### 層次一：`_assign_source_papers()` (exploration_grid.py:296-330)

每個 cell 根據 SignalFamily 對應 2-3 篇學術論文：

```python
paper_map = {
    SignalFamily.VALUE: ["Fama & French (1992) — Cross-section of expected returns", ...],
    SignalFamily.MOMENTUM: ["Jegadeesh & Titman (1993) — Returns to buying winners", ...],
    SignalFamily.QUALITY: ["Novy-Marx (2013) — Quality investing", ...],
    ...
}
```

→ 換成 Dataset Category (Fundamental, news, option...) 後，**無法將學術論文 assign 到數據類別**。你不能說「Fundamental 這個數據類別的論文是 Fama-French」。

#### 層次二：`build_hypothesis()` (exploration_grid.py:111-130)

每個 cell 的假說字串包含學術引用：

```python
f"Explore cross-sectional rank signals in the {self.family.label} family "
f"at {self.horizon.label} horizon ({self.horizon.min_days}-{self.horizon.max_days} days) "
f"{cond_desc[self.conditioning]}. "
f"Academic source: {self.family.source}."
```

→ `self.family.source`（如 "Fama-French 1992 HML"）、`cond_desc`（如 "with industry neutralization"）、`operator_desc`（如 "cross-sectional rank"）**全部失效**。新 cell 的假說應該是「用 fundamental data + cross sectional operator + long horizon」而非「根據 Fama-French 1992 HML 文獻」。

#### 層次三：`IDEA_LIBRARY_TO_GRID` (exploration_grid.py:195-219)

把具體 alpha idea template 對應到 SignalFamily + OperatorPattern + Horizon：

```python
("fundamental_quality", "rank"): (SignalFamily.QUALITY, OperatorPattern.CROSS_SECTIONAL, Horizon.LONG)
("social_buzz", "raw"):          (SignalFamily.SENTIMENT, OperatorPattern.CROSS_SECTIONAL, Horizon.SHORT)
...
```

→ **所有 22 個 mapping 全部失效**，需重新 mapping 到 Dataset Category + Operator Category + Horizon。

### 6.2 受影響的完整清單

| 位置 | 依賴的舊概念 | 影響程度 |
|---|---|---|
| `SignalFamily` Enum (line 23-34) | — | **整組刪除** |
| `OperatorPattern` Enum (line 45-50) | — | **整組刪除** |
| `Conditioning` Enum (line 71-75) | — | **整組刪除** |
| `GridCell.family` (line 80) | SignalFamily | 型別改為 DatasetCategory |
| `GridCell.operator` (line 81) | OperatorPattern | 型別改為 OperatorCategory |
| `GridCell.conditioning` (line 83) | Conditioning | **移除** |
| `GridCell.build_hypothesis()` (line 111) | family.source, operator_desc, cond_desc | **整段重寫** |
| `cell_priority()` domain_family (line 156) | SignalFamily 排序字典 | 改為 Dataset Category 排序 |
| `IDEA_LIBRARY_TO_GRID` (line 195) | SignalFamily, OperatorPattern | **整組重寫** |
| `_initialize_grid()` (line 282) | SignalFamily, OperatorPattern, Conditioning for 迴圈 | 改為 Dataset Category, Operator Category |
| `_assign_source_papers()` (line 296) | SignalFamily → 學術論文 | **整組刪除**（論文與 grid 脫鉤） |
| `_validate_fields()` (line 332) | FAMILY_FIELD_MAP 的 key | 改為 Dataset Category |
| `get_candidate_fields()` (line 341) | SignalFamily 參數 | 改為 Dataset Category |
| `select_cells()` (line 355) | SignalFamily domain_family | 改為 Dataset Category |
| `load_family_field_map()` (line 223) | SignalFamily 字體驗證 | 改為 Dataset Category |
| `FAMILY_FIELD_MAP` (line 239) | SignalFamily key | 整份 map 需要重新 mapping 到 Dataset Categories |
| `expression_generator.py` CoT templates | SignalFamily 對應 CoT prompt | 改為 Dataset Category 對應 prompt |
| `engine.py` SignalFamily import | SignalFamily | 改為 Dataset Category |
| `tests/` 中所有 `SignalFamily.VALUE` 等 | SignalFamily 參照 | 全面更新 |

### 6.3 P1 擴充計畫的根本問題

原本的 `BRAINALPHA_EXPANSION_PLAN.md` P1 計畫建立在一個錯誤前提上：

| P1 的假設 | 實際情況 |
|---|---|
| 「固定論文」是正確的框架，只是需要可擴充 | 「固定論文」**從一開始就是錯的**，因為內容是我們自定義的 |
| `papers_library.json` 可加新論文擴充 grid | SignalFamily 本不存在，擴充失去意義 |
| `N × 5 × 3 × 4` 是擴充後的公式 | Conditioning 不該在 grid 裡，公式應該是 `8 × 7 × 3` |
| 驗證 P1-1: `len(grid.cells) == 660` | 基準應改為 `len(grid.cells) == 168` |

**P1 擴充計畫的前提錯誤，導致整個計畫無效。** 不是「擴充固定論文」，而是「廢除固定論文，改用官方分類」。

### 6.4 教訓

**所有維度的內容都是我們自行定義的，論文只說了「660 = 11 × 5 × 3 × 4」。** 我們花了大量時間去「復現」一個不存在的論文框架：
- 找 11 篇學術文獻對應到 SignalFamily
- 定義 5 個 OperatorPattern 的抽象模式
- 定義 4 個 Conditioning 類型
- 把所有內容硬編碼進程式碼

這些全部建立在「我們覺得論文應該是這樣」的假設上，而不是實際的官方資料上。與其胡亂腦補、毫無根據地復現，不如直接去查 Brain 官方的實際分類，從真實資料出發。

---

## 七、結論

### 最終設計原則

1. **Cell 定義「什麼類型的 alpha」**，不包含「如何配置 Brain simulation」
2. **使用官方分類**取代自定義分類：
   - 8 個 Dataset Categories → 取代 11 個 SignalFamily
   - 7 個 Operator Categories → 取代 5 個 OperatorPattern
3. **Horizon 保留**為獨立維度，因為它決定 alpha 的核心行為特徵
4. **Conditioning 降級為 simulation settings**，不屬於 grid，由 refinement 階段處理
5. **固定論文全面廢除**：學術 paper reference 不再與 grid cell 綁定，source_papers 從 cell 中移除

### 待辦事項

- [ ] 刪除 `SignalFamily`, `OperatorPattern`, `Conditioning` 三個 Enum
- [ ] 新增 `DatasetCategory`, `OperatorCategory` Enum
- [ ] 修改 `GridCell`：family→dataset_category, operator→operator_category, 移除 conditioning
- [ ] 重寫 `_initialize_grid()`：只對 8 × 7 × 3 做迴圈
- [ ] 重寫 `build_hypothesis()`：使用 Dataset + Operator + Horizon 描述
- [ ] 刪除 `_assign_source_papers()`：論文與 cell 脫鉤
- [ ] 重寫 `IDEA_LIBRARY_TO_GRID`：改用 Dataset Category + Operator Category
- [ ] 重建 `FAMILY_FIELD_MAP` → `DATASET_FIELD_MAP`：將 ~59K fields 分配到 8 個 Dataset Categories
- [ ] 重建 operator mapping：將 ~66 個 Brain operators 分配到 7 個 Operator Categories
- [ ] 修改 `select_cells()`, `get_candidate_fields()`, `load_family_field_map()` 等 method signatures
- [ ] 更新 `expression_generator.py` CoT templates 為 Dataset Category 對應
- [ ] 更新 `engine.py` import 和 type references
- [ ] 更新所有 test files
- [ ] 拋棄舊版 `BRAINALPHA_EXPANSION_PLAN.md` P1 計畫，重新規劃
