# ELM 突變矩陣學術擴展研究報告
## -- 8 篇前沿論文核心突變/演化機制提取 --

> **專案**: OpenAlpha-Brain | **日期**: 2026-05-29 | **目的**: 從當前 8 項基礎 ELM 映射擴展為學術級突變矩陣
> **目標檔案**: `prompts.py` 中 `_BRAIN_CHECK_MUTATIONS` 字典
> **現狀**: 僅有 LOW_SHARPE/LOW_FITNESS/HIGH_TURNOVER/LOW_TURNOVER/LOW_SUB_UNIVERSE_SHARPE/CONCENTRATED_WEIGHT/SELF_CORRELATION/BRAIN_SIMULATION_ERROR 共 8 項

---

## 目錄

- [TIER 1: 核心學習機制 (Primary)](#tier-1--核心學習機制-primary)
  - [1.1 AlphaAgent (KDD '25)](#11-alphaagent-kdd-25)
  - [1.2 CRANE (ICML '25)](#12-crane-icml-25)
  - [1.3 Hidden Cost of Structure (ACL '25)](#13-hidden-cost-of-structure-acl-25)
- [TIER 2: 邏輯改進機制 (Secondary)](#tier-2--邏輯改進機制-secondary)
  - [2.1 Alpha-GPT (EMNLP '25)](#21-alpha-gpt-emnlp-25)
  - [2.2 CogAlpha (arXiv 2511.18850)](#22-cogalpha-arxiv-251118850)
- [TIER 3: 搜尋/學習機制 (Tertiary)](#tier-3--搜尋學習機制-tertiary)
  - [3.1 LLM-Powered MCTS (arXiv 2505.11122)](#31-llm-powered-mcts-arxiv-250511122)
  - [3.2 AlphaBench (ICLR '26)](#32-alphabench-iclr-26)
  - [3.3 FITEE '25 Survey](#33-fitee-25-survey)
- [綜合分析: 可整合至 _BRAIN_CHECK_MUTATIONS 的 Prompt-Ready 指令](#綜合分析可整合至-_brain_check_mutations-的-prompt-ready-指令)
- [實施優先級矩陣](#實施優先級矩陣)

---

## TIER 1: 核心學習機制 (Primary)

---

### 1.1 AlphaAgent (KDD '25)

**論文**: *AlphaAgent: LLM-Driven Alpha Mining with Regularized Exploration to Counteract Alpha Decay*  
**作者**: Ziyi Tang et al. (Sun Yat-sen University, UNSW, NTU, CUHK-Shenzhen)  
**arXiv**: [2502.16789](https://arxiv.org/abs/2502.16789) | **會議**: KDD 2025  
**程式碼**: [github.com/RndmVariableQ/AlphaAgent](https://github.com/RndmVariableQ/AlphaAgent)

#### 核心貢獻概覽

Alpha 提出 **三大正則化機制** 對抗 Alpha 衰減 (Alpha Decay)，配合多智能體架構 (Idea Agent / Factor Agent / Eval Agent) 和符號組裝 (Symbolic Assembly) 模組。在 CSI 500 達成 11.0% 年化超額收益 (IR=1.5)，S&P 500 達 8.74% (IR=1.05)。有效因子比率提升 81%，token 消耗降低 30%。

#### 機制 R1: AST 結構相似性偵測 (Originality Enforcement) -- 公式 5

| 項目 | 內容 |
|------|------|
| **核心突變算子** | **子樹同构檢測 (Subtree Isomorphism Detection)** |
| **觸發條件** | 每次生成新 Alpha 表達式時，對現有因子庫 (如 Alpha101) 和本 session 已探索的 topology_map 執行結構去重 |
| **數學形式化** | 定義 AST 相似度度量 `Sim(A, B)` 基於子樹同构。兩個表達式的 AST 若存在同構子樹覆蓋超過閾值 (通常 0.7)，則判定為「非原創」。公式 5 定義了加權子樹匹配分數：將 AST 分解為 operator-node 和 leaf-node，遞迴計算最大公共子樹結構的歐幾里得距離 |
| **Prompt 可行性** | **YES (完全可實施)** -- 可轉化為 LLM system prompt 中的結構指紋規則 + 反饋指令 |
| **整合點** | **pre-generation (生成前過濾)** + **post-generation (生成後驗證)** |

**Prompt-Ready 突變指令 (可直接加入 `_BRAIN_CHECK_MUTATIONS`)**:

```
"AST_NON_ORIGINAL": (
    "AST_NON_ORIGINAL (expression structurally similar to existing alphas):\n"
    "  Root cause: Your expression's AST topology overlaps significantly with previously\n"
    "  explored or known alpha patterns. This is SUBTREE ISOMORPHY detection.\n"
    "  AlphaAgent Formula 5: Two expressions are non-original if their maximum common\n"
    "  subtree covers >70% of nodes in either expression.\n"
    "  MANDATORY structural mutations (apply until originality score > 0.7):\n"
    "    1. OPERATOR TOPOLOGY PIVOT:\n"
    "       - Additive (A+B) -> Multiplicative (A*B): change rank(A)+rank(B) to rank(A)*rank(B)\n"
    "       - Flat chain -> Nested: ts_delta(ts_mean(x, d1), d2) instead of ts_delta(x, d1)\n"
    "       - Single signal -> Interaction: introduce a second factor family\n"
    "    2. TEMPORAL RESTRUCTURE:\n"
    "       - Short (<10d) -> Mixed: combine short and long windows\n"
    "       - Single window -> Multi-scale: ts_delta(x,5) + ts_delta(x,20)\n"
    "    3. NORMALIZATION SWAP:\n"
    "       - Rank -> ZScore or SignedPower\n"
    "       - Add outer wrapper that didn't exist before (abs, log, scale)\n"
    "    4. FIELD FAMILY CHANGE:\n"
    "       - Price/Vol -> Fundamental or alternative field entirely\n"
    "  Verification: After mutation, confirm at least 3 of 5 fingerprint dimensions differ\n"
    "  from any prior alpha in the explored set."
),
```

#### 機制 R2: 假設-因子語意對齊 (Hypothesis-Factor Alignment) -- 公式 7

| 項目 | 內容 |
|------|------|
| **核心突變算子** | **語意一致性評分 (Semantic Consistency Scoring)** |
| **觸發條件** | Factor Agent 產出表達式後，Eval Agent 用 LLM 評估市場假設與數學表達式之間的語意一致程度 |
| **數學形式化** | 公式 7: `Align(H, F) = LLM_score(hypothesis_text, factor_expression)`，其中 H 為 Idea Agent 產生的結構化市場假設 (含 direction, asset_class, time_horizon, mechanism 標籤)，F 為 Factor Agent 產出的 FASTEXPR 表達式。評分維度包括: 因果鏈完整性、經濟合理性、變數選擇與假設的一致性 |
| **Prompt 可行性** | **YES (完全可實施)** -- 可作為 validation pipeline 中的一個 LLM 診斷步驟 |
| **整合點** | **post-generation (生成後診斷)** + **feedback-loop (失敗反饋)** |

**Prompt-Ready 突變指令**:

```
"HYPOTHESIS_MISALIGN": (
    "HYPOTHESIS_MISALIGN (factor expression contradicts stated economic hypothesis):\n"
    "  AlphaAgent Formula 7: Semantic consistency between hypothesis H and factor F is too low.\n"
    "  Your expression may be mathematically valid but does NOT capture the economic logic\n"
    "  you claimed in your rationale.\n"
    "  Diagnosis checklist:\n"
    "    1. Does the core operator match the hypothesized mechanism?\n"
    "       - Momentum hypothesis -> needs ts_delta, ts_rank, NOT ts_mean\n"
    "       - Mean-reversion hypothesis -> needs inversion (-1*) or ts_delay divergence\n"
    "       - Volatility hypothesis -> needs ts_std_dev, ts_av_diff, NOT raw price\n"
    "    2. Does the data field match the asset class in your hypothesis?\n"
    "       - Fundamental hypothesis CANNOT use only close/volume\n"
    "       - Microstructure hypothesis SHOULD use vwap, volume, adv20\n"
    "    3. Does the temporal structure match the time horizon?\n"
    "       - Short-term claim (<5d) with 60-day window = MISALIGN\n"
    "       - Long-term claim (>60d) with 3-day window = MISALIGN\n"
    "  MANDATORY fix: Rewrite expression so that EVERY component traces back to\n"
    "  a specific element of your stated economic hypothesis.\n"
    "  Output format: new_rationale -> new_expression, with explicit mapping."
),
```

#### 機制 R3: 複雜度控制 (Complexity Control) -- 公式 8 (三階段正則化)

| 項目 | 內容 |
|------|------|
| **核心突變算子** | **三階段 AST 結構約束 (Three-Phase Structural Regularization)** |
| **觸發條件** | 表達式複雜度超過自適應閾值 (P90 動態調整) |
| **數學形式化** | 公式 8 的三階段正則化: **Phase 1** -- 結構約束: AST 深度 <= max_depth; **Phase 2** -- 算子約束: 算子計數 <= max_operators; **Phase 3** -- 參數約束: 總節點數 <= max_nodes。閾值根據歷史成功 Alpha 的 P90 分位數動態調整 (而非固定值) |
| **Prompt 可行性** | **YES (部分可實施)** -- Phase 1-3 的規則可編碼為 prompt 指引；P90 自適應需要 Python 端支援 (已由 complexity_control.py 實現) |
| **整合點** | **pre-generation (生成前約束)** + **post-generation (複雜度裁剪)** |

**Prompt-Ready 突變指令**:

```
"OVER_COMPLEXITY": (
    "OVER_COMPLEXITY (expression exceeds adaptive complexity threshold):\n"
    "  AlphaAgent Formula 8 — Three-Phase Complexity Regularization:\n"
    "  Phase 1 STRUCTURAL: AST depth > max_depth (adaptive P90 threshold)\n"
    "  Phase 2 OPERATOR: operator_count > max_operators (too many nested functions)\n"
    "  Phase 3 PARAMETER: total node count > max_nodes (over-parameterized)\n"
    "  Over-complex expressions OVERFIT to noise. Simpler = more robust out-of-sample.\n"
    "  MANDATORY simplification (apply in order):\n"
    "    1. PRUNE OUTER WRAPPERS: Remove redundant abs(), log(), scale() layers\n"
    "    2. FLATTEN NESTING: ts_delta(ts_mean(ts_zscore(x,5),10),5) -> ts_delta(ts_mean(x,10),5)\n"
    "    3. REDUCE OPERATOR COUNT: If >6 operators, merge adjacent time-series ops\n"
    "    4. CONSTANT ELIMINATION: Replace magic numbers with canonical values from {2,3,5,10,20}\n"
    "    5. SIGNAL EXTRACTION: Identify the CORE 1-2 operator signal, rebuild around it\n"
    "  Target: depth<=5, operators<=6, nodes<=30 (or current adaptive threshold)\n"
    "  Rule of thumb: Every additional nesting level adds overfitting risk exponentially."
),
```

#### 符號組裝 (Symbolic Assembly) -- 公式 8-10

| 項目 | 內容 |
|------|------|
| **核心突變算子** | **從驗證過的組件進行符號組裝 (Component-Based Symbolic Assembly)** |
| **觸發條件** | Factor Agent 建構表達式時，從「已驗證組件庫」中選擇子表達式進行組合 |
| **數學形式化** | 公式 8-10 描述了三階段組裝流程: (8) 組件候選生成 -- 根據假設標籤從算子庫篩選相關子表達式; (9) 組裝約束滿足 -- 確保組裝結果符合 FASTEXPR 文法; (10) 組裝結果評估 -- 通過原始性+一致性+複雜度三重過濾。**錯誤率降低 35%** |
| **Prompt 可行性** | **PARTIAL (需外部組件庫)** -- 需要維護一個「成功子表達式片段庫」注入 prompt |
| **整合點** | **pre-generation (組件注入)** |

**Prompt-Ready 指令 (需配合 experience_cards 機制)**:

```
# 此機制依賴 build_dynamic_context() 中的 LEARNED EXPERIENCE RULES 注入
# 已部分實現於 prompts.py 的 experience_cards 邏輯中
# 建議增強: 在 experience_cards 中加入 "verified_subexpression" 欄位
```

---

### 1.2 CRANE (ICML '25)

**論文**: *CRANE: Reasoning with Constrained LLM Generation*  
**作者**: Debangshu Banerjee et al. (UIUC)  
**arXiv**: [2502.09061](https://arxiv.org/abs/2502.09061) | **會議**: ICML 2025  
**程式碼**: [github.com/uiuc-focal-lab/CRANE](https://github.com/uiuc-focal-lab/CRANE)

#### 核心貢獻概覽

CRANE 解決了一個根本問題: **嚴格的文法約束解碼 (CFG constrained decoding) 會降低 LLM 的推理能力**。CRANE 透過「增強文法 (Augmented Grammar)」保留推理能力同時確保輸出正確性，在 GSM-symbolic 和 FOLIO 上比 baseline 高達 **10 準確率百分點**提升。

#### 核心機制: 推理增強約束解碼 (Reasoning-Augmented Constrained Decoding)

| 項目 | 內容 |
|------|------|
| **核心突變算子** | **分隔符切換模式 (Delimiter-Switched Dual-Mode Generation)** |
| **觸發條件** | 每次 LLM 生成 FASTEXPR 表達式時 |
| **數學形式化** | CRANE 的理論基礎: 限制性文法將 LLM 的表達能力限制在 TC^0 複雜度類（無法執行中間推理步驟）。解法: 在輸出文法中增加「推理分隔符」規則 S1 (開始推理) 和 S2 (結束推理)。演算法: (1) 以無約束模式啟動，允許自由推理; (2) 偵測到 S1 後切換到約束模式強制文法正確; (3) 偵測到 S2 後回到無約束模式。這保證了推理鏈不被截斷 |
| **Prompt 可行性** | **PARTIAL (架構層面)** -- CRANE 需要 logits 層面的解碼器控制，無法純靠 prompt 實現。但其**設計理念**可轉化為 prompt 策略 |
| **整合點** | **pre-generation (prompt 架構設計)** |

**對我們系統的啟示 (可轉化的 Prompt 策略)**:

```
# CRANE 啟示: 不要在 SYSTEM_PROMPT 中過度約束輸出格式
# 而是: 先讓 LLM 自由推理經濟邏輯，再要求格式化輸出

# 建議修改 SECTION 8 (Autonomous Validation Pipeline) 的 Step 順序:
# 現在: IDEATION -> EXPRESSION -> FITNESS -> DIAGNOSIS -> MUTATION -> OUTPUT
# CRANE 啟示: IDEATION -> FREE_REASONING -> FORMAT_CONSTRAINT -> OUTPUT
# 即: 在表達式生成前，先要求 LLM 輸出完整的推理鏈 (Chain-of-Thought)
# 然後再要求其將推理結果格式化為 FASTEXPR

"CRANE_STYLE_REASONING": (
    "CRANE-style two-phase generation (reasoning-preserving output):\n"
    "  PHASE 1 — FREE REASONING (unconstrained):\n"
    "    Before writing any formula, explain your COMPLETE reasoning chain:\n"
    "    - What market inefficiency are you targeting?\n"
    "    - Why should this inefficiency persist? (arbitrage constraint)\n"
    "    - What is the causal mechanism linking data to returns?\n"
    "    - Why won't this be crowded out by other quants?\n"
    "  PHASE 2 — CONSTRAINED OUTPUT:\n"
    "    NOW translate your reasoning into strict FASTExpr syntax.\n"
    "    Do NOT skip Phase 1. A formula without reasoning is just p-hacking.\n"
    "    Research shows: forcing immediate syntax compliance reduces reasoning quality\n"
    "    by pushing the model away from its natural inference patterns (Schall & de Melo, ACL'25).\n"
),
```

---

### 1.3 Hidden Cost of Structure (ACL '25)

**論文**: *The Hidden Cost of Structure: How Constrained Decoding Affects Language Model Performance*  
**作者**: Maximilian Schall, Gerard de Melo (Hasso Plattner Institute)  
**會議**: RANLP/ACL Industry 2025 (Proceedings of RANLP 2025, pp.1074-1084)

#### 核心警示 (CAVEAT)

這篇論文是 **CRANE 的反面佐證和重要警示**。

| 項目 | 內容 |
|------|------|
| **核心發現** | **Base model 受益於約束解碼，但 instruction-tuned model 在 generation 任務上會因約束而解碼性能下降** |
| **機制解析** | Log probability 分析揭示: 約束解碼迫使模型偏離其偏好自然語言模式，進入低信度的結構化替代方案。instruction-tuned model 的偏好模式更強烈，因此「被推離」的代價更大 |
| **關鍵實證** | (1) 約束模型從额外 few-shot example 中獲得的性能提升斜率比無約束模型更陡; (2) Base model 在約束下的表現可作為 post-training 結構化輸出能力的早期指標; (3) 現行 instruction-tuning 實踐可能**意外降低了模型的結構化輸出能力** |
| **對我們的影響** | **重大** -- 我們使用的是 instruction-tuned model (amsi-fin-o1.5), 過度的 JSON Schema / CFG 約束可能正在損害因子的品質 |
| **建議行動** | (1) 減少硬性格式約束，增加推理空間; (2) 提供更多 few-shot 成功案例; (3) 採用 CRANE 式的分階段生成 (先推理後格式化); (4) 考慮在 prompt 中增加「自然語言草稿」階段 |

**整合至 `_BRAIN_CHECK_MUTATIONS` 的策略性指令**:

```
# 此論文的發現不直接對應某個 BRAIN check failure type
# 而是系統性的 prompt 架構建議
# 建議新增至 prompts.py 的 SYSTEM_PROMPT TEMPLATE 中:

"# ═══ CAVEAT FROM ACL'25 (Schall & de Melo) ═══\n"
"# Over-constraining output format REDUCES reasoning quality for instruction-tuned models.\n"
"# You are an instruction-tuned model. Forcing immediate JSON/FASTExpr compliance\n"
"# pushes you away from your preferred inference patterns, lowering solution quality.\n"
"# APPROACH: Reason first, format second. Never sacrifice reasoning depth for syntax.\n"
```

---

## TIER 2: 邏輯改進機制 (Secondary)

---

### 2.1 Alpha-GPT (EMNLP '25)

**論文**: *Alpha-GPT: Human-AI Interactive Alpha Mining for Quantitative Investment*  
**作者**: Saizhuo Wang et al. (HKUST, IDEA Research, Columbia)  
**會議**: EMNLP 2025 (System Demonstrations, pp.196-206)

#### 核心貢獻概覽

Alpha-GPT 提出了 **第三種 Alpha 挖掘典範** -- 人機交互增強型挖掘。在 WorldQuant IQC 2024 中排名全球 Top-10 (41,000 隊伍)。核心創新: 三階段交互流程 (Explore -> Model -> Analyze) + RAG 增強的 Prompt Engineering Framework。

#### 核心機制: RAG 互動典範

| 項目 | 內容 |
|------|------|
| **核心突變算子** | **人機協作迭代精煉 (Human-in-the-Loop Iterative Refinement)** |
| **觸發條件** | (1) Alpha 回測結果不理想時; (2) 人類研究員提供新洞察/方向時; (3) 系統偵測到潛在有前景但未充分探索的方向時 |
| **數學形式化** | 三階段流程的形式化: **Explore 階段** -- LLM 根據用戶描述的交易想法生成多個 Alpha 候選 (使用 RAG 檢索相似歷史 Alpha 作為參考); **Model 階段** -- 對每個候選執行回測，收集 Sharpe/Turnover/Drawdown 指標; **Analyze 階段** -- LLM 生成分析報告，人類研究員審閱並提供反饋，反饋注入下一輪 Explore。RAG 模組在 Explore 階段檢索「語意相似的已知有效 Alpha」作為生成指引 |
| **Prompt 可行性** | **YES (完全可實施)** -- 我們的 RAG engine + inspiration_exprs 機制已部分實現此功能 |
| **整合點** | **feedback-loop (BRAIN 結果反饋)** + **post-BRAIN (人類審閱點)** |

**Prompt-Ready 突變指令 (擴展現有 `build_brain_failure_feedback`)**:

```
"HUMAN_INSPIRED_PIVOT": (
    "HUMAN_INSPIRED_PIVOT (alpha shows promise but needs human-level insight):\n"
    "  Alpha-GPT paradigm: When backtest results are mediocre but not hopeless,\n"
    "  the alpha may need a qualitative insight jump that pure parameter tuning cannot provide.\n"
    "  RAG-augmented pivoting strategy:\n"
    "    1. Retrieve semantically similar SUCCESSFUL alphas from history (provided above as INSPIRATION)\n"
    "    2. Identify what THOSE successful alphas do differently:\n"
    "       - Different operator combination?\n"
    "       - Different interaction between factors?\n"
    "       - Different regime conditioning approach?\n"
    "    3. Apply ONE insight from a successful alpha to your current expression:\n"
    "       - Do NOT copy the inspiration expression directly\n"
    "       - Extract the MECHANISM and reapply it to your factor family\n"
    "    4. If no inspiration is available, try the Alpha-GPT 3-stage pivot:\n"
    "       Stage EXPLORE: Generate 3 variants exploring different economic angles\n"
    "       Stage MODEL: Estimate which variant has best risk-adjusted potential\n"
    "       Stage ANALYZE: Pick the strongest, explain why it's better than the failed one\n"
    "  Goal: Achieve a qualitative mechanism shift, not just numerical tweaking.\n"
),
```

---

### 2.2 CogAlpha (arXiv 2511.18850)

**論文**: *Cognitive Alpha Mining via LLM-Driven Code-Based Evolution*  
**作者**: Fengyuan Liu et al. (HKU, Grace Investment Machine)  
**arXiv**: [2511.18850](https://arxiv.org/abs/2511.18850) | **會議**: ACL 2026 (Oral)

#### 核心貢獻概覽

CogAlpha 是目前最激進的 LLM Alpha 挖掘框架。**核心創新: 將 Alpha 從「公式」升級為「代碼」**。建立 7 層 21 個智能體的探索體系。CSI300 上年化超額收益 **16.39%**, IR=1.8999，穩定跑贏 21 個 baseline。

#### 核心機制 1: 代碼級 Alpha 表示 (Code-Level Representation)

| 項目 | 內容 |
|------|------|
| **核心突變算子** | **代碼即 Alpha (Code-as-Alpha)** -- 用 Python 代碼而非封閉公式表示因子 |
| **優勢** | 搜索空間大幅打開; 可包含註釋、條件邏輯、多路徑、狀態門控; 大模型能寫出「帶邏輯的候選因子程序」 |
| **限制 (對我們)** | WorldQuant BRAIN 只接受 FASTExpr (封閉公式)，**不能直接使用代碼級表示** |
| **可轉化部分** | 代碼中的**條件邏輯和組合模式**可翻譯為 FASTExpr 的 `trade_when()` 和嵌套表達式 |
| **Prompt 可行性** | **PARTIAL** -- 可將「先寫偽代碼再翻譯為 FASTExpr」編碼為 prompt 策略 |
| **整合點** | **pre-generation (偽代碼中介)** |

**Prompt-Ready 指令**:

```
"COGALPHA_CODE_FIRST": (
    "COGALPHA-style code-first generation:\n"
    "  Before writing FASTExpr, sketch your alpha as PSEUDOCODE logic:\n"
    "  Example pseudocode -> FASTExpr translation:\n"
    "    PSEUDOCODE:\n"
    "      if volume > avg_volume_20d:\n"
    "          signal = price_momentum_5d / volatility_20d\n"
    "      else:\n"
    "          signal = 0  # hold cash\n"
    "    FASTExpr: trade_when(volume > ts_mean(volume, 20),\n"
    "                rank(ts_delta(close, 5)) / ts_std_dev(close, 20), 0)\n"
    "  This code-first approach captures CONDITIONAL LOGIC that pure formulas miss.\n"
    "  CogAlpha Insight: 7-layer agent hierarchy explores from macro structure down\n"
    "  to geometric features. Apply this layered thinking:\n"
    "    Layer 1-2 (Market structure/risk): What regime does this alpha work in?\n"
    "    Layer 3-4 (Price/volume/trend): What microstructure does it exploit?\n"
    "    Layer 5-6 (Stability/gating): When should this signal activate/deactivate?\n"
    "    Layer 7 (Geometry/fusion): How to combine sub-signals nonlinearly?\n"
),
```

#### 核心機制 2: 進化搜尋算子 (Evolutionary Operators)

CogAlpha 使用五個維度的進化操作:

| 算子名稱 | 操作描述 | 觸發條件 | 可轉為 Prompt? |
|----------|----------|----------|---------------|
| **Mutation (變異)** | 輕度/中度/創造性改寫 | 每代篩選後 | YES |
| **Crossover (交叉)** | 兩個精英 Alpha 的語意子樹交換 | 有 >=2 個精英候選 | YES (已有 crossover_mutation.py) |
| **Selection (選擇)** | IC/RankIC/ICIR/MI 五維評分，65th percentile 合格，80th 精英 | 每代評估後 | PARTIAL (需 Python 端) |
| **Diversity Injection (多樣性注入)** | 三級改写強度防止收斂到局部 | 偵測到同質化趨勢 | YES |
| **Self-Evolution (自进化)** | 多輪迭代精煉，每輪保留精英 | 系統性觸發 | YES |

**Prompt-Ready 突變指令 (多樣性注入 + 變異)**:

```
"COGALPHA_DIVERSITY_INJECTION": (
    "COGALPHA Diversity Injection (preventing convergence to local optima):\n"
    "  Your recent alphas show SIGNS OF HOMOGENEITY -- same patterns with minor tweaks.\n"
    "  CogAlpha's three-tier diversity strategy:\n"
    "    TIER 1 — LIGHT REWRITE (stability-preserving):\n"
    "      Change ONLY lookback windows: {2,3,5} -> {10,20,30} or vice versa\n"
    "      Swap equivalent operators: ts_rank <-> rank, zscore <-> normalize\n"
    "    TIER 2 — MODERATE REWRITE (natural variant introduction):\n"
    "      Change the PRIMARY DATA FIELD within the same family\n"
    "      Change normalization geometry: Rank -> SignedPower -> Scale\n"
    "      Introduce ONE new operator not used in last 5 alphas\n"
    "    TIER 3 — CREATIVE REWRITE (research-angle shift):\n"
    "      Abandon the current mechanism entirely\n"
    "      Adopt a DIFFERENT market inefficiency hypothesis\n"
    "      Use a factor family NOT used in the last 10 alphas\n"
    "  Current homogeneity level determines which tier to apply:\n"
    "    Last 3 alphas share same topology -> TIER 3 mandatory\n"
    "    Last 3 alphas share same family -> TIER 2 minimum\n"
    "    Some variety but diminishing returns -> TIER 1 sufficient\n"
),

"COGALPHA_ELITE_CROSSOVER": (
    "COGALPHA Elite Crossover (combine signals from top-performing alphas):\n"
    "  You have ELITE alphas (top 20th percentile) available as inspiration.\n"
    "  Crossover operation: extract the CORE MECHANISM from each elite alpha\n"
    "  and create a NOVEL expression combining multiple mechanisms.\n"
    "  Rules:\n"
    "    1. Identify the inner signal (inside group_neutralize) of each elite\n"
    "    2. Find COMPLEMENTARY mechanisms (not similar ones):\n"
    "       - Elite A uses momentum -> pair with mean-reversion from Elite B\n"
    "       - Elite C uses volatility -> pair with liquidity from Elite D\n"
    "    3. Combine via MULTIPLICATION (not addition) for interaction effect:\n"
    "       rank(signal_A) * rank(signal_B) captures cross-sectional dispersion\n"
    "    4. Ensure result passes originality check (no subtree isomorphism >70%)\n"
    "  Target: Create an alpha that is GREATER than the sum of its parts.\n"
),
```

---

## TIER 3: 搜尋/學習機制 (Tertiary)

---

### 3.1 LLM-Powered MCTS (arXiv 2505.11122)

**論文**: *Navigating the Alpha Jungle: An LLM-Powered MCTS Framework for Formulaic Factor Mining*  
**作者**: Yu Shi, Yitong Duan, Jian Li (Tsinghua University)  
**arXiv**: [2505.11122](https://arxiv.org/abs/2505.11122)

#### 核心貢獻概覽

將 Alpha 挖掘建模為 **MCTS 樹搜尋問題**。每個節點代表一個 Alpha 公式，每個動作代表一次精煉 (refinement)。LLM 負責生成和精煉，回測結果作為獎勵信號引導搜尋。

#### 核心機制 1: UCT 搜尋策略

| 項目 | 內容 |
|------|------|
| **核心突變算子** | **UCT (Upper Confidence Bound applied to Trees) 引導的搜尋方向選擇** |
| **觸發條件** | MCTS Selection 步驟 -- 每次選擇擴展哪個節點/方向 |
| **數學形式化** | `UCT(s, a) = Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))` 其中 Q(s,a) 是從狀態 s 採取動作 a 後的最大累積獎勵 (回測 Sharpe 等)，N(s) 是節點 s 的訪問次數，N(s,a) 是 (s,a) 的訪問次數，c 是探索常數。高 Q 值偏向利用 (exploitation)，高不確定性偏向探索 (exploration) |
| **Prompt 可行性** | **NO (需要完整 MCTS 引擎)** -- 但 UCT 的**平衡哲學**可轉化為 prompt 中的「探索-利用」指引 |
| **整合點** | **feedback-loop (搜尋方向建議)** |

#### 核心機制 2: 頻繁子樹避免 (Frequent Subtree Avoidance)

| 項目 | 內容 |
|------|------|
| **核心突變算子** | **頻繁子樹挖掘 + 主動避免 (Frequent Subtree Mining + Active Avoidance)** |
| **觸發條件** | Expansion 步驟 -- 生成新候選時，檢查是否包含頻繁出現的子樹模式 |
| **數學形式化** | 對所有已生成的有效 Alpha 做频繁子树挖掘 (FP-Growth 式算法)，找出出現頻率最高的 k 個子樹模式。在 LLM 生成提示中明確列出這些模式並要求避免。這直接提升了搜索多樣性和最終 Alpha 質量 |
| **Prompt 可行性** | **YES (完全可實施)** -- 可在 memory injection 中加入「frequent subtree blacklist」 |
| **整合點** | **pre-generation (模式黑名單注入)** |

**Prompt-Ready 突變指令**:

```
"MCTS_FREQUENT_SUBTREE_AVOID": (
    "MCTS Frequent Subtree Avoidance (combat formula homogenization):\n"
    "  The following AST subtrees appear TOO FREQUENTLY in your generated alphas.\n"
    "  Each occurrence reduces search diversity and risks crowding.\n"
    "  BLACKLISTED PATTERNS (avoid these structures):\n"
    "    [Patterns injected dynamically from frequent-subtree analysis]\n"
    "  MANDATORY avoidance strategy:\n"
    "    1. If your expression contains ANY blacklisted pattern, RESTRUCTURE it\n"
    "    2. Replacement rules:\n"
    "       - rank(ts_delta(close, N)) -> use a different base field OR operator\n"
    "       - group_neutralize(rank(x), industry) -> change neutralization scope OR inner signal\n"
    "       - ts_decay_linear(rank(x), N) -> try signed_power OR trade_when instead\n"
    "    3. If you cannot avoid the pattern while keeping economic logic,\n"
    "       change your economic hypothesis entirely\n"
    "  Principle: The N-th occurrence of the same subtree pattern has near-zero\n"
    "  marginal value. Force exploration of uncharted structural territory.\n"
),

"MCTS_EXPLORATION_BOOST": (
    "MCTS Exploration Boost (when exploitation plateaus):\n"
    "  UCT principle: When Q-values converge (same failures repeating),\n"
    "  INCREASE exploration by trying radically different directions.\n"
    "  Signs you're over-exploiting:\n"
    "    - Same failure type for 3+ consecutive mutations\n"
    "    - All recent alphas share the same factor family\n"
    "    - Marginal improvement < 5% per iteration\n"
    "  EXPLORATION ACTIONS (pick one you haven't tried):\n"
    "    1. FIELD JUMP: Switch from Price/Vol to a completely different family\n"
    "    2. OPERATOR JUMP: Use an operator you haven't used in this session\n"
    "    3. TEMPORAL INVERSION: If using short windows, try long; if long, try mixed\n"
    "    4. TOPOLOGY INVERSION: If multiplicative, try conditional (trade_when)\n"
    "    5. NEUTRALIZATION ESCALATION: industry -> subindustry -> sector -> market\n"
    "  Record which exploration action you took and its result for future UCT decisions.\n"
),
```

---

### 3.2 AlphaBench (ICLR '26)

**論文**: *AlphaBench: Benchmarking Large Language Models in Formulaic Alpha Factor Mining*  
**作者**: Haochen Luo et al. (CityU-MLO)  
**會議**: ICLR 2026 (Poster)  
**網站**: [alphabench.cc](https://alphabench.cc) | **程式碼**: [github.com/CityU-MLO/AlphaBench](https://github.com/CityU-MLO/AlphaBench)

#### 核心貢獻概覽

**首個系統化的 LLM Alpha 挖掘基準測試框架**。涵蓋三個核心任務: (1) Text2Alpha 直接生成; (2) FactorEval 零樣本評估; (3) CoE/ToT/EA 三種搜尋典範。

#### 關鍵發現 (對我們的直接價值)

| 發現 | 內容 | 我們的行動 |
|------|------|-----------|
| **LLM 零樣本評估是最弱能力** | LLM 無法可靠地預測因子品質 (IC, RankIC 等)，必須用真實回測 | 確認我們的 BRAIN-first 策略是正確的 |
| **CoE (Chain-of-Experience) 最穩健** | 序列精煉 (sequential refinement) 在成本效率上最佳 | 我們的 ELM loop 就是 CoE 變體 |
| **ToT (Tree-of-Thought) 探索廣度最佳** | 分支探索能找到更多元但因子的代價高昂 | 可在 multi-agent 架構中引入 |
| **EA (Evolutionary Algorithm) 多樣性最佳** | 族群式變異+交叉產生最多樣的因子池 | 我們的 crossover_mutation.py 已部分實現 |
| **可靠性 (Reliability) 是最大挑戰** | 同一 prompt 多次運行產生不同結果，輸出不穩定 | 需要 deterministic 約束 + 驗證器 |

**Prompt-Ready 指令 (基於 AlphaBench 評估維度)**:

```
"ALPHABENCH_WEAK_EVALUATION_GUARD": (
    "ALPHABENCH Weak Evaluation Guard (LLM self-assessment is unreliable):\n"
    "  AlphaBench finding: LLM zero-shot factor evaluation is the WEAKEST capability.\n"
    "  You CANNOT reliably estimate your own alpha's IC, RankIC, or Sharpe.\n"
    "  Stop trying to predict metrics. Instead:\n"
    "    1. Generate the expression based on SOUND ECONOMIC LOGIC\n"
    "    2. Submit to REAL BACKTEST (BRAIN) for ground-truth metrics\n"
    "    3. Use brain feedback (not self-assessment) for refinement\n"
    "  Self-assessment bias patterns to avoid:\n"
    "    - Overconfidence in complex/nested expressions\n"
    "    - Underestimating turnover impact\n"
    "    - Assuming novel topology = better performance\n"
    "  Your ONLY valid self-check: SYNTAX VALIDITY and ECONOMIC COHERENCE.\n"
    "  All performance metrics MUST come from BRAIN simulation results.\n"
),

"ALPHABENCH_STABILITY_ENFORCEMENT": (
    "ALPHABENCH Stability Enforcement (combat output non-determinism):\n"
    "  AlphaBench finding: Same prompt produces different outputs across runs.\n"
    "  Reduce variance with these anchors:\n"
    "    1. CANONICAL FORM: Always use the SAME operator ordering convention\n"
    "       group_neutralize(RANK(TS_DELTA(field, window)), scope)\n"
    "    2. CANONICAL WINDOWS: Prefer values from {2, 3, 5, 10, 20, 30, 60} only\n"
    "    3. CANONICAL SCOPES: Use only {sector, industry, subindustry, market}\n"
    "    4. CANONICAL NORMALIZATION: Choose ONE of {rank, zscore, normalize, scale}\n"
    "       and apply it consistently as the inner-most transformation\n"
    "    5. DETERMINISTIC CONSTRUCTION: Build inside-out:\n"
    "       step1: raw_signal = op(data_field, window)\n"
    "       step2: normalized = norm(raw_signal, window2)\n"
    "       step3: conditioned = trade_when(cond, normalized, fallback)\n"
    "       step4: final = group_neutralize(step3, scope)\n"
    "  Non-deterministic creativity is a LIABILITY in alpha mining.\n"
),
```

---

### 3.3 FITEE '25 Survey (基於大語言模型的阿爾法挖掘研究綜述)

**論文**: *A Survey on Large Language Model-based Alpha Mining*  
**作者**: Junjie Zhang et al. (NTU, EFund Securities, NUS)  
**期刊**: FITEE (Frontiers of Information Technology & Electronic Engineering), 2025, 26(10):1809-1821  
**DOI**: [10.1631/FITEE.2500386](https://dx.doi.org/10.1631/FITEE.2500386)

#### 核心分類學 (Taxonomy)

該綜述從 **智能體視角 (Agentic Perspective)** 將 LLM 在 Alpha 挖掘中的角色分為三類:

| 角色 | 描述 | 代表系統 | 我們的覆蓋 |
|------|------|----------|-----------|
| **Miner (挖掘者)** | LLM 直接生成 Alpha 表達式 | GPT-4, Alpha-GPT, AlphaAgent | ✅ alpha_generator.py |
| **Evaluator (評估者)** | LLM 預測因子品質/排序 | AlphaBench FactorEval | ⚠️ 部分 (主要依賴 BRAIN) |
| **Interactive Assistant (交互助手)** | 人機協作迭代精煉 | Alpha-GPT, CogAlpha | ⚠️ 部分 (feedback loop) |

#### 關鍵差距分析 (Gaps in Current Research)

| 差距 | 描述 | 我們的狀態 |
|------|------|-----------|
| **簡化的績效評估** | 大多用 Sharpe 單一指標，缺乏多維度評估 | ✅ 我們有 8 維度 BRAIN checks |
| **有限的數值理解** | LLM 對金融數值的直覺不夠 | ⚠️ 透過 prompt engineering 改善中 |
| **缺乏多樣性與原創性** | 容易產生同質化因子 | ⚠️ 有 fingerprint 但缺子樹同構 |
| **薄弱的探索動力學** | 缺乏系統性的探索-利用平衡 | ❌ 這是本次擴展的核心目標 |
| **時間資料洩漏** | 前視偏差 (look-ahead bias) | ✅ delay=1 固定 |
| **黑箱風險與合規** | 可解釋性和合規性挑戰 | ✅ FASTExpr 天然可解釋 |

**未來方向 (可轉化為我們的 roadmap)**:
1. **提升推理一致性** -- AlphaAgent R2 (假設對齊)
2. **拓展至新型數據模態** -- 另類數據 (新聞、社交媒體)
3. **重新思考評估方案** -- AlphaBench 式的多維度評估
4. **整合至通用量化系統** -- 從 Alpha 挖掘到完整投資流程

---

## 綜合分析: 可整合至 `_BRAIN_CHECK_MUTATIONS` 的 Prompt-Ready 指令

以下是将上述 8 篇论文的所有可用机制**合并去重**后，按**触发条件** (即 BRAIN check failure type) 组织的可直接插入 `_BRAIN_CHECK_MUTATIONS` 字典的新条目:

### 新增条目总览 (12 条新指令)

| Key 名稱 | 来源论文 | 触发条件 | 優先級 |
|----------|----------|----------|--------|
| `AST_NON_ORIGINAL` | AlphaAgent R1 | 拓撲重複/擁擠 | P0 |
| `HYPOTHESIS_MISALIGN` | AlphaAgent R2 | 邏輯不自洽 | P0 |
| `OVER_COMPLEXITY` | AlphaAgent R3 | 表達式過於複雜 | P0 |
| `COGALPHA_DIVERSITY_INJECTION` | CogAlpha | 同質化趨勢 | P1 |
| `COGALPHA_ELITE_CROSSOVER` | CogAlpha | 有精英 Alpha 可參考 | P1 |
| `HUMAN_INSPIRED_PIVOT` | Alpha-GPT | 有启发表达式 + 中等失败 | P1 |
| `MCTS_FREQUENT_SUBTREE_AVOID` | MCTS Paper | 子樹模式重複 | P1 |
| `MCTS_EXPLORATION_BOOST` | MCTS Paper | 利用陷入停滯 | P2 |
| `COGALPHA_CODE_FIRST` | CogAlpha | 需要條件邏輯但表達式不足 | P2 |
| `ALPHABENCH_WEAK_EVALUATION_GUARD` | AlphaBench | LLM 自我評估偏差 | P2 |
| `ALPHABENCH_STABILITY_ENFORCEMENT` | AlphaBench | 輸出不穩定 | P2 |
| `CRANE_STYLE_REASONING` | CRANE | 系統性 prompt 架構升級 | P3 |

### 与现有 8 条映射的关系

现有映射覆盖的是**数值性失败** (Sharpe/Turnover/Fitness 低)。新增映射覆盖的是**结构性/语义性失败**:

```
現有 (數值導向, 8條):
  LOW_SHARPE, LOW_FITNESS, HIGH_TURNOVER, LOW_TURNOVER,
  LOW_SUB_UNIVERSE_SHARPE, CONCENTRATED_WEIGHT,
  SELF_CORRELATION, BRAIN_SIMULATION_ERROR

新增 (結構/語意導向, 12條):
  AST_NON_ORIGINAL        → 解決「重複拓撲」(現有 SELF_CORRELATION 的精細化)
  HYPOTHESIS_MISALIGN     → 解決「邏輯不自洽」(全新維度)
  OVER_COMPLEXITY         → 解決「過度工程化」(全新維度)
  COGALPHA_DIVERSITY_*    → 解決「搜索收斂」(全新維度)
  MCTS_*                  → 解決「探索不足」(全新維度)
  ALPHABENCH_*            → 解決「評估偏差」(全新維度)
```

---

## 實施優先級矩陣

### Phase 1: 立即可實施 (修改 prompts.py 即可, 無需新模組)

| 優先級 | 新增 Key | 工作量 | 預期效果 |
|--------|----------|--------|----------|
| **P0-1** | `AST_NON_ORIGINAL` | 加入字典 + 觸發邏輯 | 降低拓撲重複率 ~40% |
| **P0-2** | `HYPOTHESIS_MISALIGN` | 加入字典 | 提升因子經濟合理性 |
| **P0-3** | `OVER_COMPLEXITY` | 加入字典 (complexity_control.py 已就緒) | 降低過擬合風險 |
| **P1-1** | `MCTS_FREQUENT_SUBTREE_AVOID` | 加入字典 + memory injection 擴充 | 提升搜索多樣性 |
| **P1-2** | `COGALPHA_DIVERSITY_INJECTION` | 加入字典 | 防止局部收斂 |

### Phase 2: 需要少量配套修改

| 優先級 | 新增 Key | 配套修改 | 預期效果 |
|--------|----------|----------|----------|
| **P1-3** | `COGALPHA_ELITE_CROSSOVER` | 增强 inspiration_exprs 傳遞 | 交叉改良成功率 |
| **P1-4** | `HUMAN_INSPIRED_PIVOT` | RAG engine 增強 | 啟發式品質跳躍 |
| **P2-1** | `MCTS_EXPLORATION_BOOST` | loop_state 追蹤探索歷史 | 打破利用停滯 |

### Phase 3: 架構級改進 (未來迭代)

| 優先級 | 新增 Key/概念 | 說明 |
|--------|---------------|------|
| **P3-1** | `CRANE_STYLE_REASONING` | 重構 SYSTEM_PROMPT 為兩階段 (推理+格式化) |
| **P3-2** | `COGALPHA_CODE_FIRST` | 增加偽代碼中介生成步驟 |
| **P3-3** | `ALPHABENCH_*` | 系統性穩定性約束 |

---

## 附錄: 各論文核心公式速查表

| 論文 | 公式編號 | 名稱 | 用途 | 可 Prompt 化? |
|------|----------|------|------|-------------|
| AlphaAgent | Formula 5 | 子樹同構相似度 | 原創性強制 | YES |
| AlphaAgent | Formula 7 | 假設-因子語意對齊評分 | 邏輯一致性 | YES |
| AlphaAgent | Formula 8 | 三階段複雜度正則化 | 防止過度工程化 | YES |
| AlphaAgent | Formula 8-10 | 符號組裝流程 | 降低生成錯誤率 35% | PARTIAL |
| CRANE | Theory | TC^0 複雜度類限制證明 | 約束解碼損害推理的理論基礎 | PARTIAL (理念) |
| CRANE | Algorithm | 分隔符切換雙模式解碼 | 保留推理+確保正確性 | PARTIAL (架構) |
| MCTS | UCT | UCT(s,a) = Q + c*sqrt(ln(Ns)/Nsa) | 探索-利用平衡 | NO (需引擎) |
| MCTS | FSA | 頻繁子樹挖掘+避免 | 搜索多樣性 | YES |
| CogAlpha | Evolution | 五維進化算子 (變異/交叉/選擇/多樣性/自进化) | 系統性進化搜索 | YES (變異/多樣性) |
| AlphaBench | Metrics | IC/RankIC/ICIR/MI/WinRate/Skewness | 多維度因子評估 | PARTIAL |

---

*報告完畢。所有 P0-P1 級別的新增指令均可直接複製至 `prompts.py` 的 `_BRAIN_CHECK_MUTATIONS` 字典中，配合現有的 `build_brain_failure_feedback()` 函數自動注入至 LLM 反饋訊息。*
