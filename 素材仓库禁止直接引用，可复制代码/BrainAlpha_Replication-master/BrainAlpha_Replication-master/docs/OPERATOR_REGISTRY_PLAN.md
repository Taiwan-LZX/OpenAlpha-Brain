# Operators 官方資源整合計畫

**建立日期：** 2026-05-21  
**資源來源：** WorldQuant Brain 官方  
**目標：** 將 66 個官方運算元 + 53 份文件整合至 M1/M2/M4 管線

---

## 一、資源分析

### 1.1 operators.json

| 屬性 | 說明 |
|---|---|
| 總數 | **66 個運算元** |
| 欄位 | `name`, `category`, `scope`, `definition`, `description`, `documentation`, `level` |
| 格式 | JSON array |

### 1.2 docs/*.json

| 屬性 | 說明 |
|---|---|
| 總數 | **53 份文件** |
| 格式 | JSON，含 `id`, `title`, `content`（TEXT/TABLE block 陣列） |
| 內容 | HTML 格式說明、範例計算、使用範例、Tip |
| 無文件運算元 | 13 個：`and`, `or`, `not_equal`, `divide`, `multiply`, `subtract`, `equal`, `greater`, `greater_equal`, `less`, `less_equal`, `xor`, `not`（邏輯/算術類，定義簡單） |

### 1.3 類別分佈

| 類別 | 數量 | 代表運算元 |
|---|---|---|
| **Arithmetic** | ~12 | `abs`, `add`, `log`, `power`, `sign`, `sqrt`, `densify` |
| **Time Series** | ~22 | `ts_mean`, `ts_delta`, `ts_rank`, `ts_decay_linear`, `ts_regression`, `ts_std_dev` |
| **Group** | ~10 | `group_neutralize`, `group_rank`, `group_mean`, `group_zscore`, `group_backfill` |
| **Logical** | ~8 | `and`, `or`, `if_else`, `is_nan`, `equal`, `greater` |
| **Transformational** | ~14 | `rank`, `zscore`, `winsorize`, `scale`, `bucket`, `hump`, `normalize` |

### 1.4 現狀差距

| 項目 | 現有 | 官方 | 差距 |
|---|---|---|---|
| 運算元白名單 | 43 個硬編碼 | 66 個官方 | 可能包含無效運算元（如 `pasteurize`, `top_k`, `ts_skewness`）且遺漏真實運算元 |
| CoT Prompt | 泛稱模板 | 真實定義+文件 | LLM 可能生成不存在或錯誤簽章的運算元 |
| Rewrite Prompt | 字串包裹 | 真實運算元目錄 | 修復建議不受限於真實運算元能力 |

---

## 二、實作計畫

### 2.1 新增 `alpha_agent/operator_registry.py`（~120 行）

**職責：** 單一來源載入、解析、查詢所有官方運算元資訊。

```
OperatorRegistry
├── _load_operators()     → 解析 operators.json → dict[name] → { definition, description, category, scope, level }
├── _load_docs()          → 遍歷 docs/*.json → dict[name] → 純文字（剝離 HTML）
├── get_operator(name)    → { definition, description, doc_text, category }
├── get_valid_names()     → set[str]（66 個運算元名稱）
├── get_by_category(cat)  → list[dict]（按類別篩選）
├── get_relevant(pattern) → list[dict]（按運算元模式篩選相關運算元）
└── graceful fallback     → operators/ 目錄不存在時返回空註冊表
```

**HTML 剝離策略：**
- 移除 `<p>`, `<ul>`, `<li>`, `<b>`, `<i>` 等標籤
- 保留表格內容（轉為文字格式）
- 提取 "Examples" 和 "Tip" 區塊

**Fallback 行為：**
- 若 `operators/` 目錄不存在 → 返回空註冊表
- 若 `docs/` 目錄不存在 → 僅載入 operators.json，doc_text 為空
- 不影響現有管線 — 驗證器降級至空集（不攔截任何運算元）

### 2.2 修改 `alpha_agent/rag_spec.py`（~5 行）

```diff
# 現有：硬編碼 43 個運算元
- VALID_OPERATORS: Set[str] = {
-     "rank", "zscore", "scale", "pasteurize", "winsorize", ...
- }

# 改為：從 operator_registry 動態載入
+ from alpha_agent.operator_registry import get_operator_registry
+ def _get_valid_operators() -> Set[str]:
+     reg = get_operator_registry()
+     return reg.get_valid_names() if reg else set()
```

**影響：** D6.3（運算子家族非空）和 `detect_operators()` 現在使用真實 66 個運算元白名單。

### 2.3 修改 `alpha_agent/expression_generator.py`（~20 行）

**`_build_cot_prompt()` 增強：**

```
現有 user_prompt：
  Operators: Generate a value-oriented alpha using {fields}.
             Prefer rank(zscore({primary_field}))...

增強後 user_prompt：
  Operators: Generate a value-oriented alpha using {fields}.
             Prefer rank(zscore({primary_field}))...

  Available operators (Brain official catalog, category: {category}):
  - rank(x, rate=2): Maps values to [0,1] across universe...
  - group_neutralize(x, group): Neutralize within group...
  - ts_mean(x, d): Simple average over d days...
  ...

  Use ONLY the operators listed above. Follow their exact signatures.
```

**過濾策略：**
- 按 cell 的 `operator_pattern` 對應到官方 category
- 每類別最多取 8 個運算元，避免 prompt 過長
- 優先包含該類別中最常用的運算元（按文件 sequence 排序）

### 2.4 修改 `alpha_agent/repair_engine.py`（~15 行）

**`_build_rewrite_prompt()` 增強：**

```
現有 user_prompt：
  Repair strategy: neutralize_sector
  Current expression: rank(close)
  Failed checks: LOW_SHARPE
  ...

增強後 user_prompt：
  Repair strategy: neutralize_sector
  Current expression: rank(close)
  Failed checks: LOW_SHARPE
  ...

  Available repair operators:
  - group_neutralize(x, group): Neutralizes alpha values within each group...
  - winsorize(x, std=4): Clamps outliers beyond ±4σ...
  - zscore(x): Cross-sectional z-score normalization...

  Apply the repair strategy using ONLY the operators listed above.
```

**過濾策略：**
- 按 `strategy_name` 對應修復類型 → 篩選相關運算元
- 例如 `neutralize_*` 策略 → 只包含 Group 類別運算元
- 例如 `winsorize_*` 策略 → 只包含 Transformational 類別運算元

### 2.5 新增 `tests/test_operator_registry.py`（~80 行）

| # | 測試名稱 | 驗證點 |
|---|---|---|
| 1 | `test_load_operators` | 66 個運算元正確載入，每個有 name/category/definition/description |
| 2 | `test_load_docs` | 53 份文件載入，HTML 已剝離為純文字 |
| 3 | `test_get_valid_names` | 返回集合大小 = 66，包含 rank/ts_mean/group_neutralize |
| 4 | `test_get_operator_with_doc` | 有文件的運算元返回 definition + doc_text |
| 5 | `test_get_operator_no_doc` | 無文件運算元（如 `and`）只返回 definition + description |
| 6 | `test_get_by_category` | 按類別篩選返回正確子集 |
| 7 | `test_fallback_no_operators_dir` | 目錄不存在時返回空註冊表，不崩潰 |
| 8 | `test_doc_text_no_html` | doc_text 不包含 `<p>`, `<ul>`, `<li>` 等 HTML 標籤 |
| 9 | `test_existing_operators_subset` | 現有 43 個硬編碼中，每個都出現在 66 個官方清單中（或標記差異） |

---

## 三、驗證標準

### 3.1 單元測試驗證

```bash
python -m pytest tests/test_operator_registry.py -v
```

- 所有 9 項測試通過
- 測試覆蓋：載入、查詢、過濾、fallback、HTML 剝離

### 3.2 整合測試驗證

```bash
python -m pytest tests/ -v
```

- 全部測試通過（現有 144 + 新增 9 = **153 項**）
- 無回歸：現有 test_rag_spec.py、test_expression_generator.py、test_repair_engine.py 不受影響

### 3.3 功能驗證

#### 3.3.1 VALID_OPERATORS 比對

```python
from alpha_agent.operator_registry import get_operator_registry
from alpha_agent.rag_spec import VALID_OPERATORS

reg = get_operator_registry()
official = reg.get_valid_names()
current = VALID_OPERATORS

# 檢查現有運算元是否都在官方清單中
missing_from_official = current - official  # 應為空或極小
added_in_official = official - current      # 應為 ~23 個新運算元
```

**預期結果：**
- `missing_from_official`：可能包含 `pasteurize`, `top_k`, `bottom_k`, `count`, `correlation`, `covariance`, `ts_skewness`, `ts_kurtosis`, `ts_residual`, `ts_percentile`, `group_sum`, `group_std_dev`, `group_quantile`, `decay_exponential`, `decay_custom`, `clip` 等不在官方清單中的名稱
- `added_in_official`：`jump_decay`, `vector_neut`, `kth_element`, `last_diff_value`, `ts_av_diff`, `ts_backfill`, `ts_count_nans`, `ts_scale`, `ts_step`, `vec_avg`, `vec_sum`, `scale_down`, `group_backfill`, `bucket`, `densify`, `hump`, `days_from_last_change`, `quantile`, `normalize`, `signed_power` 等

#### 3.3.2 CoT Prompt 驗證

```python
from alpha_agent.expression_generator import ExpressionGenerator
from alpha_agent.operator_registry import get_operator_registry

reg = get_operator_registry()
# 驗證 prompt 中包含真實運算元定義
assert "group_neutralize" in prompt
assert "rank" in prompt
assert "ts_mean" in prompt
```

**預期結果：**
- CoT prompt 中包含真實運算元定義（非泛稱模板）
- prompt 長度合理（不超過 LLM context 限制）

#### 3.3.3 Rewrite Prompt 驗證

```python
from alpha_agent.repair_engine import RepairEngine
from alpha_agent.operator_registry import get_operator_registry

# 驗證 rewrite prompt 中包含修復相關運算元
assert "group_neutralize" in prompt  # 修復策略對應的運算元
```

**預期結果：**
- Rewrite prompt 中包含與修復策略相關的運算元定義
- 不包含無關運算元

### 3.4 回歸驗證

| 測試檔案 | 測試數 | 預期 |
|---|---|---|
| `test_rag_spec.py` | 23 | 全部通過（VALID_OPERATORS 替換為動態載入） |
| `test_expression_generator.py` | 20 | 全部通過（prompt 增強不影響模板回退） |
| `test_repair_engine.py` | 22 | 全部通過（prompt 增強不影響現有邏輯） |
| `test_operator_registry.py` | 9 | 全部通過（新增） |

---

## 四、檔案變更摘要

```
新增: alpha_agent/operator_registry.py       (~120 行)
新增: tests/test_operator_registry.py        (~80 行)
修改: alpha_agent/rag_spec.py                 (~5 行, 替換 VALID_OPERATORS)
修改: alpha_agent/expression_generator.py     (~20 行, 豐富 CoT prompt)
修改: alpha_agent/repair_engine.py            (~15 行, 豐富 rewrite prompt)
更新: README.md                               (新增 operators 資源說明)
更新: docs/BRAINALPHA_IMPROVEMENT_LOG.md      (記錄變更)
```

**預估工作量：** ~2 小時  
**風險等級：** 低（operator_registry 降級時不影響現有管線）

---

## 五、執行順序

| 步驟 | 內容 | 驗證 |
|---|---|---|
| 1 | 實作 `operator_registry.py` | `test_operator_registry.py` 9 項通過 |
| 2 | 修改 `rag_spec.py` 替換 VALID_OPERATORS | `test_rag_spec.py` 23 項通過 |
| 3 | 修改 `expression_generator.py` 豐富 prompt | `test_expression_generator.py` 20 項通過 |
| 4 | 修改 `repair_engine.py` 豐富 prompt | `test_repair_engine.py` 22 項通過 |
| 5 | 更新文檔 | 手動確認 |
| 6 | 全量測試 | 全部 153 項通過 |

---

## 六、實作狀態

**完成日期：** 2026-05-21  
**測試結果：** 154/154 通過（144 原有 + 10 新增），無回歸

### 6.1 實際檔案變更

```
新增: alpha_agent/operator_registry.py       (148 行)
新增: tests/test_operator_registry.py        (123 行)
修改: alpha_agent/rag_spec.py                (替換 VALID_OPERATORS 為動態載入)
修改: alpha_agent/expression_generator.py    (CoT prompt 含真實運算元定義)
修改: alpha_agent/repair_engine.py           (Rewrite prompt 含策略相關運算元)
更新: README.md                              (新增 operators/datasets 資源說明)
更新: docs/BRAINALPHA_IMPROVEMENT_LOG.md     (簡短註記)
```

### 6.2 實作偏離計畫

| 計畫 | 實際 | 原因 |
|---|---|---|
| `_load_docs()` 回傳 `dict[name] → 純文字` | 回傳 `dict[name] → {text, sequence}` | 需要儲存 sequence 供排序使用 |
| Fallback 返回空註冊表 | 保留舊硬編碼集合作為 fallback | 向後相容，避免完全失效 |
| `get_operators_for_category` 按 sequence 排序 | 初始實作按 doc_text 長度排序（已修正） | Bug，見下方 |

---

## 七、Deep Review & Debug 紀錄

**審查日期：** 2026-05-21  
**方法：** 全檔案閱讀 + 診斷腳本 + 全量測試

### 7.1 已修復 Bug (3)

| # | 檔案 | 問題 | 修正 |
|---|---|---|---|
| **1** | `operator_registry.py:112-114` | `get_operators_for_category` 以 `len(doc_text)` 降序排序，非計畫要求的「按文件 sequence 排序」。導致長文但非核心運算元（如 `ts_regression` doc_len=2544, seq=6606）排在前，基礎運算元（如 `kth_element` doc_len=1835, seq=170）排後 | 改為 `self._doc_sequence(name)` ascending 排序 |
| **2** | `operator_registry.py:36-58` | `_load_docs()` 回傳 `Dict[str, str]` 只儲存純文字，丟棄了 doc 中的 `sequence` 欄位（如 abs:7623, kth_element:170） | `_docs` 改為 `Dict[str, Dict[str, Any]]`，同時儲存 `text` + `sequence`；新增 `_doc_text()` 與 `_doc_sequence()` 輔助方法 |
| **3** | `rag_spec.py:29-42` | `_VALID_OPERATORS_FALLBACK` 含 20+ 個官方不存在的運算元（`pasteurize`, `ts_skewness`, `ts_kurtosis`, `clip`, `top_k`, `bottom_k`, `count`, `decay_exponential`, `decay_custom`, `ts_residual`, `ts_percentile`, `correlation`, `covariance`, `group_sum`, `group_std_dev`, `group_quantile`, `ts_min`, `ts_max`, `ts_argmax`, `ts_argmin`, `ts_correlation`） | 清理為僅含官方 66 個中存在的運算元（26 個） |

### 7.2 已知未修復問題 (Pre-existing，不在本次範圍)

| # | 檔案 | 問題 | 影響 |
|---|---|---|---|
| **4** | `failure_taxonomy.py:109,128` | `pasteurize_add` 與 `pasteurize_wrap` 策略引用 `pasteurize(x)`，但該運算元不在官方 66 個中 | 確定性修復路徑 `wrap_expression` 會生成無效表達式；LLM 路徑不受影響（prompt 不含 pasteurize） |
| **5** | `operators/docs/` | `jump_decay.json`, `scale_down.json`, `vector_neut.json` 有文件但 `operators.json` 無對應條目 | 無害——僅為未使用的孤兒文件 |
| **6** | `operators.json` | `min` / `max` 存在但無 `ts_min` / `ts_max` | 舊 fallback 曾包含 `ts_min` / `ts_max` 等不存在名稱，已於 Bug #3 清理 |

### 7.3 驗證確認

- **類別映射正確性：** 官方 7 個類別（Arithmetic:15, Logical:11, Time Series:24, Group:6, Cross Sectional:6, Transformational:2, Vector:2）與 `ExpressionGenerator._operator_pattern_to_categories`、`RepairEngine._strategy_to_categories` 全部一致
- **CoT Prompt 注入：** `ExpressionGenerator._build_cot_prompt()` 正確注入 "Available operators (Brain official catalog)" 區塊，含 `Use ONLY` 約束
- **Rewrite Prompt 注入：** `RepairEngine._build_rewrite_prompt()` 正確注入 "Available repair operators" 區塊，依策略名稱對應類別
- **HTML 剝離：** `bucket` doc 2583 字，無殘留 `<p>`, `<ul>`, `<li>`, `<b>` 標籤或 HTML entity
- **Fallback 測試：** `operators/` 目錄不存在時 `OperatorRegistry()` 不崩潰，`get_valid_names()` 回傳空集合
- **全量測試：** 154/154 通過，16 秒，無回歸

### 7.4 實際官方類別分佈（修正 1.3）

| 類別 | 數量 | 代表運算元 |
|---|---|---|
| **Arithmetic** | 15 | `abs`, `add`, `log`, `power`, `sign`, `sqrt`, `densify` |
| **Time Series** | 24 | `ts_mean`, `ts_delta`, `ts_rank`, `ts_decay_linear`, `ts_regression`, `ts_std_dev` |
| **Logical** | 11 | `and`, `or`, `if_else`, `is_nan`, `equal`, `greater` |
| **Group** | 6 | `group_neutralize`, `group_rank`, `group_mean`, `group_zscore`, `group_backfill`, `group_scale` |
| **Cross Sectional** | 6 | `rank`, `zscore`, `scale`, `winsorize`, `quantile`, `normalize` |
| **Transformational** | 2 | `bucket`, `trade_when` |
| **Vector** | 2 | `vec_avg`, `vec_sum` |
