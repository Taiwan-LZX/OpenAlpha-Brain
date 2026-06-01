# WorldQuant Alpha Research Agent

**BrainAlpha 復現實作基於論文：** *BrainAlpha: An Autonomous Multi-Agent System for Quantitative Alpha Discovery* (SSRN, Feb 2026)

Planner-driven, tool-using alpha research system for WorldQuant BRAIN with:

- **M0 — 72-cell 結構化探索網格**：8 DatasetCategory × 3 OperatorCategory × 3 Horizon（Analyst, Fundamental, Model, News, Option, Price Volume, Sentiment, Social Media 共 8 大資料類別），學術論文映射驅動（`thesis_map.json`）
- **M1 — RAG 增強規格編碼器**：CSV Description 語義 TF-IDF + 6 規則確定性驗證器（59K+ 欄位）
- **M2 — LLM 表達式生成**：CoT Prompt (T=0.7) + Jaccard 多樣性閘
- **M3 — Brain Oracle**：模擬與檢查執行
- **M3.5 — 錯誤復原層**：3 策略逐步升級重試
- **M4 — 收斂治理修復引擎**：FM-1~6 故障分類 + 3 階段修復流程 + LLM Phase 3 Rewrite (T=0.3) + 5 終止條件
- **共享 LLMClient**：統一 OpenAI 相容 HTTP 層，供 Planner、M2、M4 共用
- stage-aware 候選探索、開發、穩健性測試與收割
- 治理優先的提交模式（`disabled`、`manual`、`auto_approved`）
- 假設記錄與故障感知突變邏輯
- 相關性感知提交就緒與解相關修復候選
- 可重現的 JSON 報告與基線對比評估架構
- 備有內建展示案例、即時控制、視覺分析與經濟邏輯的 Presentation-Ready Streamlit 主控台

Repository: [https://github.com/zeron-G/worldquant-alpha-research-agent](https://github.com/zeron-G/worldquant-alpha-research-agent)

---

## 系統架構總覽

```
                  thesis_map.json（24 組學術論文映射）
                           │
                 ┌──────────▼──────────┐
                 │  M0: 結構化探索網格   │
                 │  72-cell = 論文庫   │
                 │  8 DatasetCategory  │
                 │  × 3 OperatorCategory│
                 │  × 3 Horizon        │
                 └──────────┬──────────┘
                           │ cell 選擇（優先級評分）
                 ┌──────────▼──────────┐
                 │  M1: RAG 規格編碼器  │
                 │  Tfidf 欄位檢索      │
                 │  + D6 確定性驗證器  │
                 └──────────┬──────────┘
                           │ 規格 + 真實欄位列表
                 ┌──────────▼──────────┐
                 │  M2: LLM 表達式生成  │
                 │  CoT (T=0.7) +      │← ─ ─ LLMClient ─ ─ ┐
                 │  Jaccard 多樣性閘   │   thesis 學術引用   │
                 └──────────┬──────────┘                     │
                           │ expression + settings          │
                 ┌──────────▼──────────┐                     │
                 │  M3: Brain Oracle   │  ← Brain CLI       │
                 │  模擬 → 檢查         │                     │
                 └──────┬──────┬──────┘                     │
                   PASS  │      │ FAIL / ERROR               │
                 ┌──────┘      ├──────────────────┐         │
                 │     ┌──────▼──────┐   ┌───────▼──────┐   │
                 │     │ M3.5 錯誤   │   │  M4 修復引擎  │   │
                 │     │ 恢復層       │   │  3-phase     │   │
                 │     │ (3 策略)    │   │  Phase 3:    │── ┘
                 │     └──────┬──────┘   │  LLM (T=0.3) │
                 │      重試/重新認證     └───────┬──────┘
                 ▼                              ▼
           SAO 輸出日誌                    SAO 返回 M0
      (通過的 Alpha)              (無法恢復 → 重選 cell)
```

### 設計原則（對應論文 P1-P4）

| 原則 | 內容 | 實作方式 |
|---|---|---|
| **P1** — 關注點分離 | LLM 負責創造性；Brain Oracle 負責事實 | LLM 生成表達式，WorldQuant API 進行評估 |
| **P2** — 有狀態可觀測性 | 所有管線狀態在每個節點轉換時持久化 | SAO 完整記錄，JSONL 儲存 |
| **P3** — 結構化反饋 | 失敗分類為型別化故障模式後再修復 | FM-1~6 分類 + 結構化修復目錄 |
| **P4** — 資料紮根生成 | M1 檢索真實欄位；M2 強制運算子合法性 | RAG + 6-rule 驗證器 + Jaccard 多樣性閘 |

---

## 專案結構

- `worldquant_brain_cli.py`
  Low-level API client and CLI (auth, simulate, check, submit, metadata fetch).
- `alpha_research_pipeline.py`
  基線啟發式搜尋管線（seed + refine + score + 可選提交）。
- `alpha_research_agent.py`
  新 Planner 驅動的 Agent CLI 入口點。
- `alpha101_ideas.json`
  WorldQuant BRAIN 相容的 101 formulaic alpha 种子转写，用于广泛复制/搜索运行。
- `alpha_agent/`
  Agent 執行期模組：
  - `config.py`: 共享執行期資料類別
  - `planner.py`: 啟發式 Planner + OpenAI JSON Planner
  - `research_logic.py`: 量化風格 Stage 邏輯、新穎性評分、檢查感知與穩健性候選建構器
  - `engine.py`: 編排器迴圈、工具執行、提交閘控、執行報告
  - `evaluation.py`: 基線對比案例套件執行器
  - `datasets_loader.py`: **CSV 欄位中繼資料載入器**（238 個 CSV、59,453 個 field，含 Description/Type/Coverage/Users/Alphas）
  - `field_classifier.py`: **LLM 輔助 field 分類器**（批量分類 → `field_family_proposals.json` → 人為審核 → 匯入 MAP）
   - `exploration_grid.py`: **M0 — 72-cell 結構化探索網格**（8 DatasetCategory × 3 OperatorCategory × 3 Horizon，含 `_ensure_thesis_map()` JSON 載入、`Thesis` dataclass、`GridCell` 優先級評分、`DATASET_FIELD_MAP` 欄位映射）
  - `llm_client.py`: **共享 LLM HTTP 客戶端**（Planner / M2 / M4 統一存取 OpenAI 相容 API）
  - `operator_registry.py`: **官方運算元註冊表**（從 `operators/operators.json` + `docs/*.json` 載入 66 個運算元 + 文件，含優雅降級）
  - `rag_spec.py`: **M1 — RAG 規格編碼器**（CSV Description 語義 TF-IDF + 6-rule 驗證器，VALID_OPERATORS 改為動態從 operator_registry 載入，支援 `from_csv_metadata()` 取代純 ID 比對）
  - `expression_generator.py`: **M2 — 表達式生成器**（LLM CoT T=0.7 + Jaccard 多樣性閘 + 模板回退，CoT prompt 含真實官方運算元定義）
  - `failure_taxonomy.py`: **FM-1~6 故障分類** + 修復目錄
  - `repair_engine.py`: **M4 — 收斂治理修復引擎**（3 階段 + LLM Phase 3 T=0.3 + 5 終止條件，Rewrite prompt 含策略相關運算元定義）
  - `sao.py`: **SAO（Signal Alpha Object）** 資料類別
  - `error_recovery.py`: **M3.5 — 錯誤復原層**（3 策略逐步升級）
- `streamlit_app.py`
  Presentation 主控台，含內建展示案例、即時 Agent 執行、參數調整、視覺診斷與經濟分析。
- `tests/`
  完整測試套件（169 項測試，全部通過：含 M3.5、SAO、LLM、operator_registry、datasets_loader 等全面覆蓋）
- `operators/`
  官方 Brain 運算元資源：
  - `operators.json`: **66 個官方運算元**（含 name, category, definition, description, scope）
  - `docs/*.json`: **56 份運算元文件**（含 HTML 說明、範例計算、模擬設定）
- `datasets/`
  資料集定義（由 `DatasetsLoader` 完整載入，59,453 個 field metadata）：
  - `datasets.sqlite`: SQLite 資料庫，含 2 表格（dataset_fields 7,642 筆欄位 + refresh_runs）
  - 238 個 `*_fields_formatted.csv`：各資料集欄位定義（Field, Description, Type, Coverage, Users, Alphas）
- `docs/`
  設計文件與復現方案：
  - `BRAINALPHA_REPRODUCTION_PLAN.md`: 完整復現方案
  - `BRAINALPHA_IMPROVEMENT_LOG.md`: 錯誤修正歷程（7 項 Bug 修正 + Operators 官方資源整合 + CSV Field Metadata 整合）
  - `BRAINALPHA_EXPANSION_PLAN.md`: P0/P1 擴充改善計畫
  - `OPERATOR_REGISTRY_PLAN.md`: Operators 官方資源整合計畫（66 運算元 → M1/M2/M4）
  - `field-metadata-csv-integration.md`: CSV Field Metadata 整合計畫（238 CSV → DatasetsLoader → ExplorationGrid/RAG/Validator + LLM 輔助分類）
  - `P0_DEEP_REVIEW_LOG.md`: P0 Deep Review 與 14 項改善紀錄
  - `ALPHA101.md`: Alpha101 複製庫說明
  - `PRESENTATION_FRONTEND.md`: Streamlit 主控台使用指南
  - `PIPELINE.md`: 管線架構說明
  - `eval_cases.json`: 起始 replay 案例套件

---

## 主要功能

### 1) M0 — 結構化探索網格（72-Cell Grid）

每個 Cell 代表一個獨特的假說空間，由 **DatasetCategory × OperatorCategory × Horizon** 三維定義：

```
Grid = 8 DatasetCategory × 3 OperatorCategory × 3 Horizon = 72 cells
```

- **8 資料類別**（DatasetCategory）：analyst, fundamental, model, news, option, price_volume, sentiment, social_media（對應 Brain 官方分類）
- **3 運算子類別**（OperatorCategory）：cross_sectional, time_series, group（對應 Brain 官方分類）
- **3 時間尺度**（Horizon）：short (1-10天), medium (10-60天), long (60-252天)
- **學術論文映射**：24 組 (8×3) 論文從 `data/thesis_map.json` 載入，注入 LLM CoT prompt 作為學術依據

Cell 優先級評分（論文公式 1）：
```
Priority(c, a) = 0.5 × Novelty(c) + 0.3 × ExpectedYield(c) + 0.2 × AgentAlignment(c, a)
```

### 2) M1 — RAG 增強規格編碼器

- **CSV Description 語義 TF-IDF**：使用 `DatasetsLoader` 載入 238 個 CSV（59,453 個 field）的 Description 建立 TF-IDF corpus，取代純 field ID 字元比對
- **領域詞典**：將自然語言（"operating profit"）對應到真實欄位（`fnd6_fopo`）
- **D6 確定性驗證器**（零 LLM 成本）：
  ① 表達式非空 ② 包含欄位引用 ③ 使用有效運算元 ④ 複雜度 ≤ 12 運算元、≤ 80 tokens ⑤ 欄位皆為已知 ⑥ 括號平衡
- **CSV metadata fallback**：當 CSV 載入失敗時，自動回退至 `wqb_data_fields_summary.json`

### 3) M2 — LLM 表達式生成（T=0.7 可配置，可選）

- **CoT Prompt**：從 8 個 DatasetCategory 專用模板 + cell 假說 + 學術論文引用（`thesis_map.json`）+ 真實欄位列表拼接 Chain-of-Thought 提示
- **可配置溫度**：`LLMGenConfig.cot_temperature`（預設 0.7），從 config 統一管理
- **LLM 回退**：LLM 不可用或 `cot_enabled=False` 時自動退回模板生成，管線不中斷
- **Jaccard 多樣性閘**（論文公式 2）：確保候選之間運算子 Jaccard < 0.70
- **6-rule 驗證**：正規表達式 + 單詞邊界取代，防止子字串腐蝕

### 4) M4 — 收斂治理修復引擎（3 階段 + 5 終止條件）

**Phase 1 — Recoverability Assessment**：不可修復則返回 M0 重選 Cell

**Phase 2 — Strategy Selection**（完全確定性，從 `REPAIR_CATALOGUE` 依嚴重度選取）：

**Phase 3 — LLM Rewrite**（T=0.3 可配置，可選）：
```
LLM prompt 包含：當前表達式 + 策略名稱 + 所有失敗檢查 + 修復歷史 + Sharpe 歷史 + 原始假說
          ↓ 失敗時回退
wrap_expression() 確定性字串包裹
```

**5 個終止條件**：(a) 所有檢查通過  (b) 達到最大迭代  (c) Sharpe 連續 3 次改善 < 0.02  (d) 差距百分比增大  (e) 所有目錄動作已耗盡

### 5) M3.5 — 錯誤復原層

3 策略逐步升級：① 指數退避重試  ② 重新認證 + 重試  ③ 清除 session 重新登入

### 6) 端到端 Alpha Agent 迴圈

Agent 重複執行：

1. 收集前沿上下文（家族表現、失敗檢查直方圖、Stage、假設）
2. 選擇下一動作（`evaluate_seed`、`evaluate_refine`、`evaluate_diversify`、`evaluate_robustness`、`submit_best`、`stop`）
3. 呼叫模擬/檢查工具
4. 更新排行榜、Stage 和研究筆記本
5. 記錄理由、假設、風險備註與結果

### 7) 量化風格 Stage 政策

- `explore`：在預算限制下最大化家族與表達式多樣性
- `exploit`：針對主要失敗檢查使用檢查感知 refinements
- `robustness`：在宇宙/中性化/截斷擾動下壓力測試頂級候選
- `harvest`：僅當就緒度與治理對齊時才嘗試控制提交

### 8) 提交治理

- `disabled`：永不提交
- `manual`：僅在明確批准回調時提交
- `auto_approved`：當 Planner 選擇時允許无人值守提交

### 9) 可插拔 Planner 後端

- `heuristic`：確定性 Planner（無需模型金鑰）
- `openai`：OpenAI 相容 JSON Planner

### 10) 評估架構

對相同案例套件和預算執行基線管線和 Agent 並比較：

- 逐案例分數與延遲差異
- 總體勝率與平均差異
- JSON 報告用於附錄/展示

---

## 需求

- Python 3.10+
- WorldQuant BRAIN 帳戶訪問權限
- 可選：OpenAI 相容 API 金鑰（Planner LLM / M2 表達式生成 / M4 Phase 3 Rewrite 使用）
- 可選：Streamlit（用於 Web UI）

安裝依賴：

```powershell
pip install -r requirements.txt
```

## 環境變數

從 `.env.example` 複製並本地設定（永不提交機密）：

```powershell
$env:WQB_EMAIL="your_email@example.com"
$env:WQB_PASSWORD="your_password"
# 或
$env:WQB_COOKIE_HEADER="sessionid=...; csrftoken=..."

$env:ALPHA_AGENT_PLANNER_PROVIDER="heuristic"
$env:ALPHA_AGENT_PLANNER_MODEL="gpt5.5"
$env:ALPHA_AGENT_PLANNER_BASE_URL="https://api.openai.com/v1"
$env:ALPHA_AGENT_PLANNER_API_KEY_ENV="OPENAI_API_KEY"

# LLM 生成（M2 CoT + M4 Phase 3 Rewrite）
$env:ALPHA_AGENT_LLM_GEN_ENABLED="false"
$env:ALPHA_AGENT_LLM_GEN_PROVIDER="openai"
$env:ALPHA_AGENT_LLM_GEN_COT_ENABLED="true"
$env:ALPHA_AGENT_LLM_GEN_REWRITE_ENABLED="true"
$env:OPENAI_API_KEY="..."
```

CLI 和 Streamlit 應用在啟動時自動載入 `.env`。

---

## 快速開始

### 從 CLI 執行 Agent

```powershell
python .\alpha_research_agent.py --pretty run --budget 16 --max-iterations 10
```

量化風格調整範例：

```powershell
python .\alpha_research_agent.py --pretty run --budget 20 --refine-top-k 10 --robustness-top-k 4 --robustness-score-threshold 550 --max-family-budget-share 0.4 --min-expression-novelty 0.12
```

聚焦特定家族：

```powershell
python .\alpha_research_agent.py --pretty run --family social_buzz --budget 12
```

從 Alpha101 複製庫啟動：

```powershell
python .\alpha_research_agent.py --pretty --idea-library .\alpha101_ideas.json run --family alpha101 --budget 24 --max-iterations 8
```

使用 OpenAI Planner：

```powershell
python .\alpha_research_agent.py --pretty --planner-provider openai run --budget 16
```

手動提交模式（需要終端機批准）：

```powershell
python .\alpha_research_agent.py --pretty run --submission-mode manual --interactive-approval
```

### 顯示當前排行榜

```powershell
python .\alpha_research_agent.py --pretty leaderboard --limit 10
```

### 執行基線對比評估

```powershell
python .\alpha_research_agent.py --pretty evaluate --cases .\docs\eval_cases.json
```

輸出預設為：
- `<workdir>/evaluation/report.json`

### 執行 Presentation 主控台

```powershell
python -m streamlit run .\streamlit_app.py
```

然後打開 Streamlit 输印的本地 URL，通常是 `http://localhost:8501`。

建議展示流程：

1. 打開應用並保持內建展示案例載入
2. 使用 `Overview` 展示最終 Stage、最佳分數、就緒漏斗與收斂曲線
3. 使用 `Agent Trace` 展示 Planner 如何從探索過渡到相關性修復和穩健性測試
4. 使用 `Economic Logic` 展示約束優化視角並即時調整成本/價值假設
5. 使用 `Architecture` 展示設計選擇與所有可熱修改的執行參數
6. 僅在配置憑證或 Cookie 標頭後才從側邊欄使用 `Run Live Agent`

### 執行測試

```powershell
python -m pytest tests/ -v
```

或使用 unittest：

```powershell
python -m unittest tests\test_planner.py tests\test_research_logic.py tests\test_progress.py tests\test_alpha101_library.py tests\test_repair_engine.py
```

---

## 專案特定功能說明文件

| 文件 | 內容 |
|---|---|---|
| [`docs/BRAINALPHA_REPRODUCTION_PLAN.md`](docs/BRAINALPHA_REPRODUCTION_PLAN.md) | BrainAlpha 完整復現方案 |
| [`docs/BRAINALPHA_IMPROVEMENT_LOG.md`](docs/BRAINALPHA_IMPROVEMENT_LOG.md) | 程式碼審查與錯誤修正歷程 |
| [`docs/BRAINALPHA_EXPANSION_PLAN.md`](docs/BRAINALPHA_EXPANSION_PLAN.md) | 擴充改善計畫 |
| [`docs/grid-reconstruction-plan.md`](docs/grid-reconstruction-plan.md) | 660→72-cell 網格重構計畫（完整紀錄） |
| [`docs/grid-redefinition-discovery.md`](docs/grid-redefinition-discovery.md) | Grid Redefinition 探索與實作過程 |
| [`docs/thesis-mapping.md`](docs/thesis-mapping.md) | 24-cell 學術論文映射驗證 |
| [`docs/pipeline-flowchart.md`](docs/pipeline-flowchart.md) | M0-M4 完整流程圖（中英並行） |
| [`docs/ALPHA101.md`](docs/ALPHA101.md) | Alpha101 複製庫使用說明 |
| [`docs/PRESENTATION_FRONTEND.md`](docs/PRESENTATION_FRONTEND.md) | Streamlit 主控台使用指南 |
| [`docs/PIPELINE.md`](docs/PIPELINE.md) | 管線架構說明 |

---

## 可重現性 artifact

在 Agent 工作目錄（預設 `.alpha_agent`）中：

- `results.jsonl`：每個已評估候選的記錄
- `submissions.jsonl`：每次提交嘗試的記錄
- `state.json`：滾動摘要
- `agent_runs/*.json`：完整每次執行報告（含 Planner 決策和事件日誌）
- `evaluation/report.json`：基線對比 Agent 比較報告

---

## Legacy 基線命令

基線腳本仍然可用：

```powershell
python .\alpha_research_pipeline.py --pretty search --budget 24 --seed-fraction 0.7
python .\alpha_research_pipeline.py --pretty leaderboard --limit 10
python .\alpha_research_pipeline.py --pretty submit-best
```

---

## 安全備註

- 永不提交憑證、Cookie、金鑰或私人帳戶資料
- 使用 `.env` 和被忽略的本地檔案儲存機密
- 研究時保持 `submission_mode=disabled`，除非故意啟用更強模式
- 現場展示和課堂評估時首選手動批准

## 免責聲明

僅在使用授權憑證和權限時負責任地使用。平台行為和可用端點可能隨時間變化。

## License

[MIT](LICENSE)