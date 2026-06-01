# BrainAlpha 擴充改善計畫 [DEPRECATED]

> **⚠️ 本計劃已廢止。** 基於 2026-05-22 Grid Redefinition 重構，SignalFamily（11 families）已被 8-category DatasetCategory（ANALYST, FUNDAMENTAL, MODEL, NEWS, OPTION, PRICE_VOLUME, SENTIMENT, SOCIAL_MEDIA）取代。OperatorPattern（5 patterns）已被 3-category OperatorCategory（CROSS_SECTIONAL, TIME_SERIES, GROUP）取代。Conditioning 維度已移除。新網格為 8×3×3 = 72 cells。詳見 [`BRAINALPHA_IMPROVEMENT_LOG.md`](BRAINALPHA_IMPROVEMENT_LOG.md) 第九節。

**建立日期：** 2026-05-21  
**最後更新：** 2026-05-22（Grid Redefinition 後標記為 DEPRECATED）  
**基準專案：** [worldquant-alpha-research-agent](https://github.com/zeron-G/worldquant-alpha-research-agent)  
**目標論文：** BrainAlpha: An Autonomous Multi-Agent System for Quantitative Alpha Discovery (SSRN, Feb 2026)

---

## 一、總覽

本計畫記錄 BrainAlpha 復現專案的兩階段改善路線：

| 階段 | 內容 | 預估工時 | 優先級 | 狀態 |
|---|---|---|---|---|
| **P0** | 完成兩個核心 LLM 環節（M2 表達式生成 + M4 Phase 3 修復重寫） | ~5.5 小時 | 高 | ✅ 已完成 |
| **P1** | 將固定論文改為可擴充文獻庫（Registry + JSON Config） | ~6 小時 | 中 | ⏳ 待執行 |

**執行順序：先 P0 後 P1。** 理由：LLM 是論文核心價值，可擴充性是次要。先讓 LLM 能跑，再讓論文能擴充。否則擴充了 JSON 但沒有 LLM 推理，無法體現不同論文之間的差異。

---

## 二、P0：完成兩個核心 LLM 環節 ✅ 已完成

### 2.1 現狀分析

| 模組 | 論文設計 | 實際實作 | 差距 |
|---|---|---|---|
| **M2** 表達式生成 | LLM CoT (T=0.7) + Jaccard 多樣性閘 | 模板生成 + `random.choice()` 固定變體 | ✅ 已修復 |
| **M4 Phase 3** 修復重寫 | LLM Rewrite (T=0.3) | `wrap_expression()` 確定性字串包裹 | ✅ 已修復 |

### 2.2 現有基礎設施（可直接復用）

| 已有基礎 | 位置 | 狀態 |
|---|---|---|
| `LLMGenConfig`（含 `enabled`, `cot_enabled`, `rewrite_enabled`, `provider`）| `config.py:50-55` | ✅ 已定義，已串接 |
| `ModelConfig`（含 `base_url`, `temperature`, `api_key_env`）| `config.py:20-27` | ✅ 已使用於 Planner + LLMClient |
| 共享 `LLMClient` | `alpha_agent/llm_client.py` | ✅ 已建立，供 M2/M4/Planner 共用 |
| `DEFAULT_COT_TEMPLATES`（11 個家族的 CoT prompt 模板）| `expression_generator.py:112-134` | ✅ 已使用於 `_build_cot_prompt` |
| `JaccardDiversityGate` + `DeterministicValidator` | `expression_generator.py` | ✅ 已使用 |
| `ConvergenceGovernance` + `REPAIR_CATALOGUE` | `repair_engine.py` | ✅ 已使用 |
| `GeneratedCandidate.cot_reasoning` 欄位（已預留）| `expression_generator.py:24` | ✅ 已使用 |
| `RepairEngine.execute()` Phase 1-2 已完整 | `repair_engine.py:128-213` | ✅ 已實作 |
| fallback 模式（OpenAIJsonPlanner → HeuristicPlanner）| `planner.py:270-277` | ✅ 已實作 |

### 2.3 P0-A：M2 LLM 表達式生成 ✅ 已完成

#### 改動點：`ExpressionGenerator.generate_from_cell()`

**現狀流程：**
```
Cell → build_template_expression() → 固定模板 → _vary_expression(random.choice()) → 候選列表
```

**目標流程：**
```
Cell → [LLM 已啟用?] ──是──→ CoT Prompt → LLM (T=0.7) → JSON 表達式 → 驗證器 → Jaccard 閘 → 候選列表
           │                                                        ↓ 失敗
           └──否──→ build_template_expression() → 固定模板 ──────────┘
```

#### 實作步驟

| # | 項目 | 檔案 | 說明 | 狀態 |
|---|---|---|---|---|
| 1 | 抽取共享 LLM 底層 | 新建 `alpha_agent/llm_client.py` | 將 `OpenAIJsonPlanner._chat_completion()` 的 HTTP 邏輯抽成獨立 `LLMClient` 類別，供 M2 和 M4 共用 | ✅ |
| 2 | 擴充 `ExpressionGenerator.__init__` | `expression_generator.py:205` | 接受 `LLMClient` + `LLMGenConfig` 參數 | ✅ |
| 3 | 新增 `_llm_generate_expression()` 方法 | `expression_generator.py:295` | CoT prompt + LLM (T=0.7) + JSON 解析 + 驗證 | ✅ |
| 4 | 改寫 `generate_from_cell()` | `expression_generator.py:222` | LLM 優先 → fallback 模板 | ✅ |
| 5 | 串接 `LLMGenConfig` | `engine.py:335` | `ensure_expression_generator()` 從 `AgentConfig.llm_gen` 讀取設定並傳入 | ✅ |
| 6 | 新增測試 | `tests/test_expression_generator.py` | 6 項測試（LLM 路徑、fallback、disabled、no client、cot_enabled=False、empty response） | ✅ |

#### 驗證條件

| # | 驗證項目 | 通過標準 | 狀態 |
|---|---|---|---|
| P0A-1 | LLM 生成路徑正常執行 | 傳回 1+ 個有效表達式，每個通過 6-rule 驗證 | ✅ |
| P0A-2 | LLM 輸出格式正確 | 回傳 JSON 含 `expression` 和 `reasoning` 欄位 | ✅ |
| P0A-3 | Fallback 路徑正常 | LLM 失敗時自動退回模板生成，不中斷管線 | ✅ |
| P0A-4 | CoT prompt 格式完整 | prompt 包含 cell、source、hypothesis、fields、operators、constraints | ✅ |
| P0A-5 | Jaccard 多樣性閘正常 | LLM 生成的多個候選通過多樣性過濾 | ✅ |
| P0A-6 | 溫度設定正確 | LLM 呼叫使用 T=0.7 | ✅ |
| P0A-7 | 向後相容 | 預設 `llm_gen.enabled=False` 時行為與現狀一致 | ✅ |
| P0A-8 | 測試套件通過 | 115/115 測試通過 | ✅ |

---

### 2.4 P0-B：M4 Phase 3 LLM Rewrite ✅ 已完成

#### 改動點：`RepairEngine.execute()` 的 Phase 3 段

**現狀流程：**
```
Phase 2 選擇策略 → wrap_expression(strategy_expr) → 確定性字串包裹 → RepairResult
```

**目標流程：**
```
Phase 2 選擇策略 → [LLM 已啟用?] ──是──→ LLM (T=0.3) 根據完整上下文重寫 → 驗證 → RepairResult
                       │                                                          ↓ 失敗
                       └──否──→ wrap_expression(strategy_expr) ─────────────────────┘
```

#### 實作步驟

| # | 項目 | 檔案 | 說明 | 狀態 |
|---|---|---|---|---|
| 1 | 復用共享 LLM 底層 | （同 P0-A 步驟 1） | 已在 P0-A 中建立 `LLMClient` | ✅ |
| 2 | 擴充 `RepairEngine.__init__` | `repair_engine.py:117` | 接受 `LLMClient` 參數 | ✅ |
| 3 | 新增 `_llm_rewrite_expression()` 方法 | `repair_engine.py:240` | Phase 3 prompt + LLM (T=0.3) + JSON 解析 | ✅ |
| 4 | 改寫 `execute()` 中的 Phase 3 呼叫點 | `repair_engine.py:188` | LLM rewrite → fallback `wrap_expression()` | ✅ |
| 5 | 串接到 engine | `engine.py:354` | `ensure_repair_engine()` 將 LLM config 傳入 | ✅ |
| 6 | 新增測試 | `tests/test_repair_engine.py` | 5 項測試（rewrite、fallback、no client、empty response、missing key） | ✅ |

#### 驗證條件

| # | 驗證項目 | 通過標準 | 狀態 |
|---|---|---|---|
| P0B-1 | LLM rewrite 正常執行 | 回傳重寫後的表達式，與原始表達式不同 | ✅ |
| P0B-2 | LLM 輸出格式正確 | 回傳 JSON 含 `expression` 欄位 | ✅ |
| P0B-3 | Fallback 路徑正常 | LLM 失敗時自動退回 `wrap_expression()` | ✅ |
| P0B-4 | Phase 3 prompt 完整 | prompt 包含 expression、strategy、failed_checks、repair_history、hypothesis | ✅ |
| P0B-5 | 溫度設定正確 | LLM 呼叫使用 T=0.3 | ✅ |
| P0B-6 | 收斂治理正常 | 重寫後的表達式仍通過 5 個終止條件檢查 | ✅ |
| P0B-7 | 向後相容 | 預設 `llm_gen.enabled=False` 時行為與現狀一致 | ✅ |
| P0B-8 | 測試套件通過 | 115 項測試通過（含新增測試） | ✅ |

---

### 2.5 P0 總體驗證 ✅ 已完成

| # | 整合驗證項目 | 通過標準 | 狀態 |
|---|---|---|---|
| P0-1 | 完整管線（M0→M1→M2→M3→M4） | 設定 `llm_gen.enabled=True`，執行完整 agent run，確認 M2 LLM 生成 + M4 LLM 修復均正常 | ✅ |
| P0-2 | 無 LLM 模式 | 設定 `llm_gen.enabled=False`，行為與現狀完全一致 | ✅ |
| P0-3 | LLM 失敗 graceful fallback | LLM API 不可用時，管線不中斷，自動退回模板/包裹 | ✅ |
| P0-4 | 額外 API 成本可控 | 每次運行 LLM 呼叫數 < 200 次（取決於 budget） | ✅ |
| P0-5 | 表達式複雜度提升 | LLM 生成的表達式平均嵌套深度 > 模板生成（4-6 vs 2-3） | ✅ |

---

### 2.6 P0 Deep Review 發現與修復

Deep Review 共發現 **14 項問題**，全部修復。詳見 [P0_DEEP_REVIEW_LOG.md](./P0_DEEP_REVIEW_LOG.md)。

#### 關鍵修復摘要

| 類別 | 數量 | 說明 |
|---|---|---|
| **Critical** | 5 | URLError 未捕獲、JSONDecodeError 未捕獲、failed_checks 未傳遞、cot_enabled 未檢查、Mock `or` fallthrough |
| **Design** | 3 | 新增 `rewrite_enabled` config 欄位、engine.py 串接 rewrite_enabled、_build_cot_prompt 空欄位保護 |
| **Cleanup** | 3 | 移除 dead `api_key` 參數、移除 dead PHASE 常數、移除重複 diversity filter |
| **Tests** | 3 | 新增 test_llm_client.py (11 tests)、expression_generator edge cases (+2)、repair_engine edge cases (+1) |

**最終測試結果：115/115 通過**

---

## 三、P1：可擴充文獻庫（Registry + JSON Config）

### 3.1 現狀分析

| 維度 | 位置 | 數量 | 擴充方式 |
|---|---|---|---|
| **SignalFamily** | `exploration_grid.py:17-28` | 11 個 | 改程式碼 |
| **OperatorPattern** | `exploration_grid.py:39-44` | 5 個 | 改程式碼 |
| **Horizon** | `exploration_grid.py:47-50` | 3 個 | 改程式碼 |
| **Conditioning** | `exploration_grid.py:65-69` | 4 個 | 改程式碼 |

### 3.2 受影響範圍統計

| 檔案 | 硬編碼點數量 | 改動類型 |
|---|---|---|
| `exploration_grid.py` | **73 處** | Enum 定義、初始化迴圈、排序字典、欄位映射、論文映射、假說構建、IDEA_LIBRARY_TO_GRID |
| `expression_generator.py` | **23 處** | OPERATOR_TEMPLATES、DEFAULT_COT_TEMPLATES |
| `engine.py` | **1 處** | `SignalFamily(self.grid_domain_family)` |
| `tests/test_expression_generator.py` | **16 處** | 所有 `SignalFamily.VALUE` 等參照 |

### 3.3 目標架構

```
papers_library.json          ← 使用者編輯這個即可擴充
     │
     ▼
exploration_grid.py
  ├─ SignalFamilyRegistry    ← 從 JSON 動態載入
  ├─ OperatorRegistry
  ├─ HorizonRegistry
  ├─ ConditioningRegistry
  │
  ▼
N × 5 × 3 × 4 cells          ← 自動計算：N_families × N_operators × N_horizons × N_conditionings
```

#### `papers_library.json` 結構

```json
{
  "signal_families": {
    "value": {
      "label": "value",
      "source": "Fama-French 1992 HML",
      "fields": ["bookvalue_ps", "return_equity", "ebit", "debt_lt", "cashflow_op"],
      "cot_prompt": "Generate a value-oriented alpha using {fields}. Prefer rank(zscore({primary_field})) or ratio of enterprise_value to cashflow operators.",
      "source_papers": [
        "Fama & French (1992) — Cross-section of expected returns",
        "Fama & French (1993) — Common risk factors"
      ]
    },
    "momentum": { ... },
    "your_new_theory": {
      "label": "earnings_surprise",
      "source": "Bernard-Thomas 1989 PEAD",
      "fields": ["eps", "nws18_ssc", "close"],
      "cot_prompt": "Generate a post-earnings announcement drift alpha...",
      "source_papers": ["Bernard & Thomas (1989) — Post-earnings-announcement drift"]
    }
  },
  "operators": [
    {"name": "cross_sectional", "label": "cross-sectional rank", "templates": ["rank({field})", "zscore({field})", "group_rank({field}, sector)"]},
    {"name": "time_series", "label": "time-series smoothed", "templates": ["ts_mean({field}, {window})", ...]},
    {"name": "ratio_spread", ...},
    {"name": "scale_normalize", ...},
    {"name": "composite", ...}
  ],
  "horizons": [
    {"name": "short", "label": "short", "min_days": 1, "max_days": 10},
    {"name": "medium", "label": "medium", "min_days": 10, "max_days": 60},
    {"name": "long", "label": "long", "min_days": 60, "max_days": 252}
  ],
  "conditionings": [
    {"name": "raw", "label": "without conditioning", "suffix": ""},
    {"name": "neutralized", "label": "with industry neutralization", "suffix": "group_neutralize(expr, sector)"},
    {"name": "residual", "label": "with residual-based adjustment", "suffix": "ts_regression(expr, group_mean(expr, sector), 60)"},
    {"name": "gated", "label": "with liquidity or volatility gating", "suffix": "if_else(adv20 > ts_mean(adv20, 60), expr, 0)"}
  ]
}
```

### 3.4 實作步驟

#### 階段一：建立 Registry 基礎設施（~2 小時）

| # | 項目 | 檔案 | 說明 |
|---|---|---|---|
| 1 | `SignalFamily` Enum → Registry | `exploration_grid.py` | 建立 `SignalFamilySpec` dataclass + `SignalFamily` Registry 類別，含 `load()`, `all()`, `get()` 方法 |
| 2 | `OperatorPattern` Enum → Registry | `exploration_grid.py` | 同上 |
| 3 | `Horizon` Enum → Registry | `exploration_grid.py` | 同上 |
| 4 | `Conditioning` Enum → Registry | `exploration_grid.py` | 同上 |

#### 階段二：遷移硬編碼資料到 JSON（~1.5 小時）

| # | 項目 | 檔案 | 說明 |
|---|---|---|---|
| 5 | 新建 `papers_library.json` | 新建 | 從現有 Enum/dict 提取所有資料 |
| 6 | `FAMILY_FIELD_MAP` → config | `exploration_grid.py` | 從 `SignalFamilySpec.fields` 讀取 |
| 7 | `_assign_source_papers()` → config | `exploration_grid.py` | 從 `SignalFamilySpec.source_papers` 讀取 |
| 8 | `build_hypothesis()` → config | `exploration_grid.py` | 從 config 的 `label` 和 `source` 讀取 |
| 9 | `cell_priority()` 排序 → 動態 | `exploration_grid.py` | 根據 config 陣列索引動態生成 |
| 10 | `_initialize_grid()` → 動態迴圈 | `exploration_grid.py` | `for f in SignalFamily.all()` |
| 11 | `GRID_SIZE` → 自動計算 | `exploration_grid.py` | `len(families) × len(operators) × len(horizons) × len(conditionings)` |
| 12 | `IDEA_LIBRARY_TO_GRID` → config | `exploration_grid.py` | 移至 `papers_library.json` |

#### 階段三：遷移表達式生成相關（~1 小時）

| # | 項目 | 檔案 | 說明 |
|---|---|---|---|
| 13 | `OPERATOR_TEMPLATES` → config | `expression_generator.py` | 從 `OperatorPattern` config 讀取 |
| 14 | `DEFAULT_COT_TEMPLATES` → config | `expression_generator.py` | 從 `SignalFamilySpec.cot_prompt` 讀取 |
| 15 | `NEUTRALIZATION_SUFFIXES` → config | `expression_generator.py` | 從 `Conditioning` config 讀取 |

#### 階段四：串接與測試（~1.5 小時）

| # | 項目 | 檔案 | 說明 |
|---|---|---|---|
| 16 | 新增 `PapersLibraryConfig` | `config.py` | 指向 `papers_library.json` 路徑 |
| 17 | 串接 config 到 engine | `engine.py` | `ExplorationGrid.__init__` 接受 config 參數 |
| 18 | 更新測試 | `tests/` | 所有 Enum 參照改為 Registry 參照 |
| 19 | 新增擴充測試 | `tests/` | 測試載入自訂 JSON、新增家庭、動態網格大小 |

### 3.5 驗證條件

| # | 驗證項目 | 通過標準 | 測試方法 |
|---|---|---|---|
| P1-1 | 預設 JSON 載入正常 | 使用內建 `papers_library.json`，網格大小 = 660 cells | 執行 `ExplorationGrid()`，檢查 `len(grid.cells) == 660` |
| P1-2 | 新增訊號家族 | 在 JSON 加入新家族，網格自動擴充 | 加入第 12 個家族，檢查 `len(grid.cells) == 720` |
| P1-3 | 新增運算子 | 在 JSON 加入新運算子，模板自動生效 | 加入新運算子，檢查 `build_template_expression()` 使用新模板 |
| P1-4 | 向後相容 | 無 `papers_library.json` 時使用硬編碼預設值 | 刪除 JSON 檔案，檢查行為與現狀一致 |
| P1-5 | 優先級評分正常 | 新家族的 `cell_priority()` 計算正確 | 檢查新家族 cell 的 priority 值在合理範圍 |
| P1-6 | 假說構建正常 | 新家族的 `build_hypothesis()` 包含正確的 source 和 label | 檢查新家族 cell 的 hypothesis 字串 |
| P1-7 | LLM 整合正常 | P0 完成後，新家族的 CoT prompt 被 LLM 使用 | 設定 `llm_gen.enabled=True`，檢查 LLM 收到新 prompt |
| P1-8 | 測試套件通過 | 所有測試通過 | `python -m pytest tests/ -v` |

---

## 四、風險評估

| 風險 | 影響 | 機率 | 緩解措施 |
|---|---|---|---|
| LLM 輸出格式不穩定 | 解析失敗 → fallback | 中 | prompt 中明確要求 `response_format: json_object`，parser 做防禦性處理 |
| Enum → Registry 改寫量大 | 測試遺漏 → 回歸 | 高 | 嚴格測試覆蓋，保留舊 Enum 為 default fallback |
| LLM API 成本 | 每次運行 $1-5 | 低 | 可選功能，預設關閉 |
| JSON config 格式錯誤 | 載入失敗 | 低 | 提供 schema 驗證 + 詳細錯誤訊息 |
| 動態網格大小影響效能 | 初始化變慢 | 低 | 網格大小從 660 到 720 差異不大 |

---

## 五、向後相容策略

| 模組 | 策略 |
|---|---|
| `papers_library.json` 不存在 | 使用硬編碼預設值（現有 Enum） |
| `llm_gen.enabled=False` | 完全使用模板/包裹，行為與現狀一致 |
| LLM API 不可用 | 自動 fallback 到模板/包裹，不中斷管線 |
| 舊測試 | 保留所有現有測試，新增測試覆蓋新路徑 |

---

## 六、完成後體驗

### P0 完成後

```
Cell → LLM CoT 推理 → 結構性新穎表達式 → BRAIN 評估 → LLM 修復重寫 → 提交
```

- 表達式從模板填充升級為學術推理生成
- 修復從字串包裹升級為上下文感知重寫
- 每次運行額外成本 ~$1-5
- 表達式平均嵌套深度從 2-3 提升到 4-6

### P1 完成後

```
papers_library.json 加入新論文 → 網格自動擴充 → LLM 根據新 CoT prompt 生成 → 評估 → 修復
```

- 加新論文只需編輯 JSON，無需改程式碼
- 網格大小自動計算
- 每個論文自帶欄位、CoT prompt、來源文獻

---

*本計畫基於 BrainAlpha 論文 (SSRN, February 2026) 與實際程式碼審查結果，於 2026 年 5 月 21 日制定。*
