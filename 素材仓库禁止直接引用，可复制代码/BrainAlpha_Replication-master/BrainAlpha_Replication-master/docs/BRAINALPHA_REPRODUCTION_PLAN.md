# BrainAlpha 復現方案：論文/經濟理論驅動的 Alpha 探索

**基準專案：** [worldquant-alpha-research-agent](https://github.com/zeron-G/worldquant-alpha-research-agent)
**目標論文：** BrainAlpha: An Autonomous Multi-Agent System for Quantitative Alpha Discovery (SSRN, Feb 2026)
**核心原則：** 結構化學術假說覆蓋，非無方向窮舉搜索

---

## 一、總體架構

```
                          論文 / 經濟理論
                               │
                    ┌──────────▼──────────┐
                    │   M0: 結構化探索網格   │  ← 新建 exploration_grid.py
                    │   660-cell = 論文庫   │
                    │   11 家族 × 5 運算子   │
                    │   × 3 時間尺度 × 4 條件│
                    └──────────┬──────────┘
                               │ cell 選擇 (優先級評分)
                    ┌──────────▼──────────┐
                    │   M1: RAG 規格編碼器  │  ← 新建 rag_spec.py
                    │   Tfidf 欄位檢索      │
                    │   + 6-rule 驗證器    │
                    └──────────┬──────────┘
                               │ 規格 + 真實欄位列表
                    ┌──────────▼──────────┐
                    │   M2: LLM 表達式生成  │  ← 新建 expression_generator.py
                    │   CoT + Jaccard 閘   │
                    └──────────┬──────────┘
                               │ expression + settings
                    ┌──────────▼──────────┐
                    │   M3: Brain Oracle  │  ← 現有 worldquant_brain_cli.py ✓
                    │   模擬 → 檢查        │
                    └──────┬──────┬──────┘
                      PASS  │      │ FAIL / ERROR
                    ┌───────┘      ├──────────────────┐
                    │      ┌──────▼──────┐    ┌───────▼──────┐
                    │      │  M3.5 錯誤   │    │   M4 修復引擎 │
                    │      │  恢復層       │    │  3-phase      │
                    │      │  (新建)      │    │  5 終止條件   │
                    │      └──────┬──────┘    │  (新建)       │
                    │       重試/重新認證      └──────┬──────┘
                    │                                │ 修復後 → M3
                    ▼                                ▼
              SAO 輸出日誌                       SAO 返回 M0
         (通過的 Alpha)                    (無法恢復 → 重選 cell)
```

### 設計原則（論文 P1-P4）

| 原則 | 內容 | 實作方式 |
|---|---|---|
| **P1** — 關注點分離 | LLM 負責創造性；Brain Oracle 負責事實 | LLM 生成表達式，WorldQuant API 進行評估 |
| **P2** — 有狀態可觀測性 | 所有管線狀態在每個節點轉換時持久化 | SAO 完整記錄，JSONL 儲存 |
| **P3** — 結構化反饋 | 失敗分類為型別化故障模式後再修復 | FM-1~6 分類 + 結構化修復目錄 |
| **P4** — 資料紮根生成 | M1 檢索真實欄位；M2 強制運算子合法性 | RAG + 6-rule 驗證器 + Jaccard 多樣性閘 |

---

## 二、M0：結構化探索網格（論文 C1）

### 2.1 網格定義——來自學術文獻的分類

```
Grid = 11 SignalFamily × 5 OperatorPattern × 3 Horizon × 4 Conditioning = 660 cells
```

每個維度的每個項目都對應特定的學術文獻。

#### 11 個訊號家族 (SignalFamily)

| # | 家族 | 學術來源 | 對應 Brain 欄位 |
|---|---|---|---|
| 1 | `value` | Fama-French 1992, 1993 (HML) | `bookvalue_ps`, `return_equity`, `ebit`, `debt_lt`, `cashflow_op` |
| 2 | `momentum` | Jegadeesh-Titman 1993 | `returns`, `close` |
| 3 | `quality` | Novy-Marx 2013, Fama-French 2015 (RMW, CMA) | `fnd6_fopo`, `fnd6_mfma*`, `cashflow_op`, `assets` |
| 4 | `low_volatility` | Frazzini-Pedersen 2014 (BAB) | `close`, `returns` |
| 5 | `size` | Banz 1981, Fama-French 1993 (SMB) | `cap`, `sharesout` |
| 6 | `sentiment` | Baker-Wurgler 2006 | `scl12_sentiment`, `snt_value`, `snt_social_value` |
| 7 | `news_attention` | Tetlock 2007, Da-Zhi-Engelberg 2011 | `nws18_relevance`, `news_ratio_vol`, `nws18_nip` |
| 8 | `short_reversal` | Jegadeesh 1990, Lehmann 1990 | `close`, `vwap`, `returns` |
| 9 | `liquidity` | Amihud 2002, Pastor-Stambaugh 2003 | `volume`, `adv20`, `cap` |
| 10 | `growth` | Lakonishok-Shleifer-Vishny 1994 | `sales_growth`, `eps`, `income` |
| 11 | `technical_trend` | Brock-Lakonishok-LeBaron 1992 | `close`, `volume`, `high`, `low` |

#### 5 個運算子模式 (OperatorPattern)

| # | 模式 | 學術含義 | 代表運算子 |
|---|---|---|---|
| 1 | `cross_sectional` | 橫截面排序（價值/規模效應的本質） | `rank()`, `group_rank()`, `group_zscore()` |
| 2 | `time_series` | 時間序列平滑（動量/反轉的本質） | `ts_mean()`, `ts_decay_linear()`, `ts_rank()` |
| 3 | `ratio_spread` | 比率/價差（價值/套利的本質） | `A/B`, `(A-B)/B`, `A-B` |
| 4 | `scale_normalize` | 標準化/去極端值（穩健化的本質） | `zscore()`, `winsorize()`, `ts_std_dev()` |
| 5 | `composite` | 多因子組合（現代因子模型的核心理念） | `A + B`, `A * B`, `trade_when()` |

#### 3 個時間尺度 (Horizon)

| 尺度 | 天數 | 學術含義 |
|---|---|---|
| `short` | 1-10 天 | 短期反轉、流動性壓力、即時資訊反應 |
| `medium` | 10-60 天 | 中期動量、盈餘漂移、公告後效應 |
| `long` | 60-252 天 | 長期價值、基本面動量、宏觀經濟暴露 |

#### 4 個條件化類型 (Conditioning)

| 類型 | 含義 | 學術對應 |
|---|---|---|
| `raw` | 無條件化 | 原始因子暴露 |
| `neutralized` | 行業/子行業中性化 | 純粹異質報酬（Moskowitz-Grinblatt 1999） |
| `residual` | 市場/規模回歸殘差 | Fama-French 殘差（Carhart 1997） |
| `gated` | 流動性/波動率條件 | 考慮交易成本和執行可行性（Amihud 2002） |

### 2.2 核心資料結構

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class CellStatus(Enum):
    EMPTY = "empty"          # 尚未探索
    ASSIGNED = "assigned"    # 已分配給某 agent
    EXPLORED = "explored"    # 至少一個 alpha 已評估
    EXHAUSTED = "exhausted"  # 該 cell 無更多有價值的路徑


class SignalFamily(Enum):
    VALUE = ("value", "Fama-French 1992")
    MOMENTUM = ("momentum", "Jegadeesh-Titman 1993")
    QUALITY = ("quality", "Novy-Marx 2013")
    LOW_VOLATILITY = ("low_volatility", "Frazzini-Pedersen 2014")
    SIZE = ("size", "Banz 1981")
    SENTIMENT = ("sentiment", "Baker-Wurgler 2006")
    NEWS_ATTENTION = ("news_attention", "Tetlock 2007")
    SHORT_REVERSAL = ("short_reversal", "Jegadeesh 1990")
    LIQUIDITY = ("liquidity", "Amihud 2002")
    GROWTH = ("growth", "Lakonishok-Shleifer-Vishny 1994")
    TECHNICAL_TREND = ("technical_trend", "Brock-Lakonishok-LeBaron 1992")


class OperatorPattern(Enum):
    CROSS_SECTIONAL = "cross_sectional"
    TIME_SERIES = "time_series"
    RATIO_SPREAD = "ratio_spread"
    SCALE_NORMALIZE = "scale_normalize"
    COMPOSITE = "composite"


class Horizon(Enum):
    SHORT = ("short", 1, 10)
    MEDIUM = ("medium", 10, 60)
    LONG = ("long", 60, 252)


class Conditioning(Enum):
    RAW = "raw"
    NEUTRALIZED = "neutralized"
    RESIDUAL = "residual"
    GATED = "gated"


@dataclass
class GridCell:
    family: SignalFamily
    operator: OperatorPattern
    horizon: Horizon
    conditioning: Conditioning
    status: CellStatus = CellStatus.EMPTY
    explored_count: int = 0
    pass_count: int = 0
    total_count: int = 0
    last_sharpe: Optional[float] = None
    source_papers: List[str] = field(default_factory=list)
    candidate_fields: List[str] = field(default_factory=list)

    def cell_id(self) -> str:
        """value_cross_sectional_long_neutralized"""
        return f"{self.family.value[0]}_{self.operator.value}_{self.horizon.value[0]}_{self.conditioning.value}"

    def expected_yield(self) -> float:
        if self.total_count == 0:
            return 0.1
        return self.pass_count / max(1, self.total_count)

    def novelty(self) -> float:
        return {
            CellStatus.EMPTY: 1.00, CellStatus.ASSIGNED: 0.50,
            CellStatus.EXPLORED: 0.25, CellStatus.EXHAUSTED: 0.00,
        }[self.status]
```

### 2.3 優先級評分（論文公式 1）

```python
def cell_priority(cell: GridCell, agent_domain: Optional[SignalFamily] = None) -> float:
    """BrainAlpha 論文公式 (1)
    Priority(c, a) = 0.5 * Novelty(c) + 0.3 * ExpectedYield(c) + 0.2 * AgentAlignment(c, a)
    """
    novelty = cell.novelty()
    yield_score = cell.expected_yield()

    alignment = 0.3
    if agent_domain:
        if cell.family == agent_domain:
            alignment = 1.0
        elif cell.family.value[0][:3] == agent_domain.value[0][:3]:
            alignment = 0.7
        elif cell.conditioning == Conditioning.GATED:
            alignment = 0.5

    return 0.5 * novelty + 0.3 * yield_score + 0.2 * alignment
```

### 2.4 現有想法庫到 Grid 的對映

現有 `alpha_pipeline_ideas.json` 中的每個家族可以直接映射到特定的 grid cells：

```
social_buzz       × raw/time_series  → Sentiment × cross_sectional/time_series
price_reversion   × close_vs_mean/   → Short_Reversal × ratio_spread
                    vwap_gap
fundamental_quality × rank/zscore   → Quality × cross_sectional/scale_normalize
news_attention    × mean             → News_Attention × time_series
social_price_combo × news_plus_*    → Sentiment/News × composite
```

好處：現有 `manual_seeds` 仍可直接載入，但每個 seed 被關聯到學術 cell。

---

## 三、M1：RAG 規格編碼器（論文 C2）

### 3.1 架構

```
輸入：自然語言假說（來自 M0 cell 或使用者）
  │
  ▼
┌─────────────────────────────┐
│  TfidfVectorizer +         │
│  character n-grams (3-6)   │  ← 處理縮寫欄位（如 fnd6_fopo）
│  + 領域詞典擴展            │
└─────────────┬───────────────┘
              │ top-k 欄位 × similarity
              ▼
┌─────────────────────────────┐
│  6-rule 確定性驗證器        │  ← 零 LLM 成本
│  ① 非空主欄位               │
│  ② 視窗範圍 1≤min≤max≤120  │
│  ③ 合法中性化策略           │
│  ④ 非空運算子家族           │
│  ⑤ 非空方向（多/空）        │
│  ⑥ 假說長度 ≥ 20 字元       │
└─────────────┬───────────────┘
              │
              ▼
輸出：Specification（帶驗證狀態）
```

### 3.2 核心類別

```python
@dataclass
class Specification:
    primary_fields: List[str]
    neutralization: str              # SECTOR / INDUSTRY / SUBINDUSTRY / NONE
    lookback_min: int
    lookback_max: int
    operator_family: str             # cross_sectional / time_series / ...
    direction: str                   # long / short / long_short
    hypothesis_statement: str        # ≥ 20 chars
    validation_errors: List[str]     # 空 = 通過


class FieldRetriever:
    """TfidfVectorizer 檢索 + 領域詞典輔助"""

    VALID_NEUTRALIZATIONS = {"SECTOR", "INDUSTRY", "SUBINDUSTRY", "NONE"}
    VALID_OPERATORS = {"cross_sectional", "time_series", "ratio_spread",
                       "scale_normalize", "composite"}
    VALID_DIRECTIONS = {"long", "short", "long_short"}

    def __init__(self, fields_json: Path):
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 6),
            max_features=5000,
        )
        self.field_ids: List[str] = []
        self.field_matrix = None
        self._load_fields(fields_json)

    def retrieve(self, hypothesis: str, top_k: int = 15) -> List[Tuple[str, float]]:
        """返回 [(field_id, similarity), ...]"""

    def validate(self, spec: Specification) -> Specification:
        """6-rule 確定性驗證，返回帶 error 列表的 spec"""
```

### 3.3 領域詞典（輔助 Tfidf 處理縮寫欄位）

```python
FIELD_ALIASES = {
    "operating profit":              "fnd6_fopo",
    "free cash flow":                "cashflow_op",
    "book to market":                "bookvalue_ps",
    "return on equity":              "return_equity",
    "social buzz":                   "scl12_buzz",
    "news relevance":                "nws18_relevance",
    "news sentiment score":          "nws18_ssc",
    "enterprise value":              "enterprise_value",
    "working capital":               "working_capital",
    "retained earnings":             "retained_earnings",
    "research development expense":  "rd_expense",
    "selling general admin":         "sga_expense",
    "depreciation amortization":     "depre_amort",
    "current assets":                "assets_curr",
    "current liabilities":           "liabilities_curr",
    "total assets":                  "assets",
    "total debt":                    "debt",
}
```

---

## 四、M2：多樣性強制表達式生成

### 4.1 設計

這是 LLM 的核心價值：**從學術假說 + 真實欄位 → FASTEXPR 表達式**。與現有 `OpenAIJsonPlanner` 共用相同的 LLM 配置基礎設施。

```python
@dataclass
class ExpressionCandidate:
    expression: str
    complexity: int               # AST node count
    operator_tokens: List[str]
    diversity_score: float


class DiversityGate:
    """Jaccard 多樣性閘（論文公式 2）"""

    def __init__(self, threshold: float = 0.70):
        self.threshold = threshold
        self.accepted_ops: List[set] = []

    def accept(self, candidate: ExpressionCandidate) -> bool:
        """|ops(new) ∩ ops(e)| / |ops(new) ∪ ops(e)| < 0.70"""
        ops_new = set(candidate.operator_tokens)
        for ops_existing in self.accepted_ops:
            union = len(ops_new | ops_existing)
            if union == 0:
                continue
            jaccard = len(ops_new & ops_existing) / union
            if jaccard >= self.threshold:
                return False
        self.accepted_ops.append(ops_new)
        return True


class ExpressionGenerator:
    """LLM CoT 表達式生成器（論文 T=0.7）"""

    def generate(
        self,
        cell: GridCell,
        spec: Specification,
        temperature: float = 0.7,
        max_candidates: int = 8,
    ) -> List[ExpressionCandidate]:
        """CoT prompt → LLM → DiversityGate → 候選列表"""
```

### 4.2 CoT Prompt 範本

```
System: You are a quantitative researcher generating WorldQuant FASTEXPR expressions.
Ground every expression in the provided verified field list only.

Cell: [value, cross_sectional_rank, long, neutralized]
Source: Fama-French 1992 HML factor
Hypothesis: "High book-to-market stocks outperform low book-to-market stocks
            in the long run, after neutralizing industry effects."
Verified fields: bookvalue_ps, close, sector, industry, cap
Operators: rank(), group_rank(), ts_mean(), group_neutralize()
Constraint: lookback 60-252, neutralization=INDUSTRY, universe=TOP3000

Step 1: Identify the core signal: bookvalue_ps (high = value)
Step 2: Apply cross-sectional ranking: rank(bookvalue_ps)
Step 3: Neutralize industry: group_neutralize(rank(bookvalue_ps), industry)
Step 4: Smooth for stability: ts_mean(group_neutralize(rank(bookvalue_ps), industry), 120)

Output JSON:
{"expression": "ts_mean(group_neutralize(rank(bookvalue_ps), industry), 120)", ...}
```

---

## 五、FM-1~6 故障模式分類

### 5.1 故障分類 (`alpha_agent/failure_taxonomy.py`)

| 代碼 | 名稱 | 診斷訊號 | Brain 檢查 | 主要修復 |
|---|---|---|---|---|
| **FM-1** | Low Signal Quality | Sharpe < 閾值, 平坦 PnL | `LOW_SHARPE`, `LOW_FITNESS` | 中性化、視窗擴展 |
| **FM-2** | Excessive Turnover | 換手率 > 上限 | `HIGH_TURNOVER` | 衰減平滑、視窗增加 |
| **FM-3** | High Drawdown | 最大回撤 > 閾值 | `HIGH_DRAWDOWN` | Winsorization、波動率條件化 |
| **FM-4** | Signal Reversal | 年化報酬為負 | `NEGATIVE_RETURNS` | Negation (-1*)、反轉 |
| **FM-5** | Low Coverage | 多/空頭數量低於最小閾值 | `LOW_SUB_UNIVERSE_SHARPE` | pasteurize()、過濾放寬 |
| **FM-6** | Low Turnover | 換手率低於下限 | `LOW_TURNOVER` | 視窗減少、截面變異 |

```python
class FailureMode(Enum):
    FM1_LOW_SIGNAL_QUALITY = ("FM-1", "Low Signal Quality")
    FM2_EXCESSIVE_TURNOVER = ("FM-2", "Excessive Turnover")
    FM3_HIGH_DRAWDOWN      = ("FM-3", "High Drawdown")
    FM4_SIGNAL_REVERSAL    = ("FM-4", "Signal Reversal")
    FM5_LOW_COVERAGE       = ("FM-5", "Low Coverage")
    FM6_LOW_TURNOVER       = ("FM-6", "Low Turnover")


CHECK_TO_FM = {
    "LOW_SHARPE":               FailureMode.FM1_LOW_SIGNAL_QUALITY,
    "LOW_FITNESS":              FailureMode.FM1_LOW_SIGNAL_QUALITY,
    "HIGH_TURNOVER":            FailureMode.FM2_EXCESSIVE_TURNOVER,
    "HIGH_DRAWDOWN":            FailureMode.FM3_HIGH_DRAWDOWN,
    "LOW_SUB_UNIVERSE_SHARPE":  FailureMode.FM5_LOW_COVERAGE,
    "LOW_TURNOVER":             FailureMode.FM6_LOW_TURNOVER,
    "CONCENTRATED_WEIGHT":      FailureMode.FM5_LOW_COVERAGE,
}
```

### 5.2 修復目錄（論文 Table 4）

```python
REPAIR_CATALOGUE = {
    FailureMode.FM1_LOW_SIGNAL_QUALITY: [
        ("neutralize_sector",      "group_neutralize(x, sector)"),
        ("neutralize_subindustry", "group_neutralize(x, subindustry)"),
        ("zscore_normalize",       "zscore(x)"),
        ("winsorize_outliers",     "winsorize(x, std=3)"),
        ("extend_lookback_2x",     "Increase all lookback ×2"),
    ],
    FailureMode.FM2_EXCESSIVE_TURNOVER: [
        ("decay_linear_5",         "decay_linear(x, 5)"),
        ("mean_std_ratio",         "ts_mean(x,d) / ts_std_dev(x,d)"),
        ("pasteurize_add",         "pasteurize(x)"),
        ("restructure_rank",       "rank()"),
    ],
    FailureMode.FM3_HIGH_DRAWDOWN: [
        ("winsorize_3std",         "winsorize(x, std=3)"),
        ("volatility_condition",   "Condition on low vol regime"),
    ],
    FailureMode.FM4_SIGNAL_REVERSAL: [
        ("negate_expr",            "-1 * (expr)"),
        ("inverse_operator",       "Invert core operator sign"),
    ],
    FailureMode.FM5_LOW_COVERAGE: [
        ("pasteurize_wrap",        "pasteurize(x)"),
        ("relax_filter",           "Relax universe constraint"),
    ],
    FailureMode.FM6_LOW_TURNOVER: [
        ("reduce_lookback",        "Reduce lookback window"),
        ("cross_sectional_tilt",   "Add cross-sectional variation"),
    ],
}
```

---

## 六、M4：收斂治理修復引擎（論文 C4）

### 6.1 三階段修復流程

```
Phase 1 — Recoverability Assessment:
  ├─ I1: SELF_CORRELATION = FAIL          → 結構性重複，不可修復
  ├─ I2: returns ≤ 0 AND complexity > 20  → 反向無法傳播，不可修復
  ├─ I3: n_failing ≥ 4                     → 不可單一動作修復
  └─ S1: max gap_pct > 0.50               → 差距過大，軟停止
       │ 不可修復
       ▼
    返回 M0，重新選擇 cell

Phase 2 — Strategy Selection (完全確定性，無 LLM):
  └─ 從 REPAIR_CATALOGUE 中選取下一個未嘗試的動作

Phase 3 — LLM Rewrite (T=0.3):
  ├─ 提示：當前表達式
  ├─ 提示：所有失敗檢查值 + 差距百分比
  ├─ 提示：完整修復歷史
  └─ 提示：原始假說
       │
       ▼
    5 個終止條件：
    (a) 所有檢查通過 ✓
    (b) 達到最大迭代次數
    (c) Sharpe 連續 3 次改善 < 0.02
    (d) 差距百分比增大（發散）
    (e) 所有目錄動作已耗盡
```

### 6.2 核心類別

```python
class RepairEngine:
    """
    M4: 收斂治理修復引擎（論文 4.4 節）
    """

    MAX_ITERATIONS = 5
    SHARPE_IMPROVEMENT_THRESHOLD = 0.02  # δ
    SHARPE_STAGNATION_WINDOW = 3

    def assess_recoverability(self, sao: SignalAlphaObject) -> Tuple[bool, Optional[str]]:
        """Phase 1: 返回 (可修復, stop_reason)"""

    def select_next_strategy(self, fm: FailureMode, tried: List[str]) -> Optional[str]:
        """Phase 2: 返回策略名稱或 None（已耗盡）"""

    def rewrite_expression(self, sao: SignalAlphaObject, strategy: str) -> Optional[str]:
        """Phase 3: LLM 重寫 (T=0.3)，返回新的 expression"""

    def check_stop_conditions(self, sao: SignalAlphaObject, iteration: int) -> Tuple[bool, str]:
        """五個終止條件檢查"""
```

---

## 七、M3.5：錯誤恢復層（論文 C5）

```python
class ErrorRecovery:
    """
    三策略逐步升級恢復（論文 M3.5 節）

    Strategy 1: retry with exponential backoff (1s → 2s → 4s)
    Strategy 2: re-authenticate + retry (rebuild BrainClient)
    Strategy 3: clear session, fresh login, retry
    """

    def recover(
        self,
        error: Exception,
        client: BrainClient,
        auth: AuthConfig,
        attempt: int,
    ) -> Tuple[BrainClient, bool]:
        """根據 attempt 選擇策略，返回 (client, recovered)"""
```

---

## 八、SAO 訊號 Alpha 物件（論文 Table 2）

```python
@dataclass
class SignalAlphaObject:
    """完整 SAO：所有 inter-module 資料通過此物件流動"""

    # === Identity (M0) ===
    id: str                          # uuid
    source_cell_id: str              # e.g. "value_cross_sectional_long_neutralized"
    source_paper: str                # e.g. "Fama, French (1992)"

    # === Hypothesis (M0) ===
    statement: str                   # ≥ 20 chars
    category: str
    horizon: str

    # === Specification (M1) ===
    primary_fields: List[str]
    neutralization: str
    lookback_min: int
    lookback_max: int

    # === Expression (M2) ===
    expression: str
    complexity: int
    diversity_score: float

    # === Simulation (M3) ===
    alpha_id: Optional[str] = None
    sharpe: Optional[float] = None
    fitness: Optional[float] = None
    turnover: Optional[float] = None
    returns: Optional[float] = None
    drawdown: Optional[float] = None
    checks: List[Dict] = field(default_factory=list)
    settings: Dict = field(default_factory=dict)

    # === Evaluation (M3) ===
    status: str = "pending"          # pending / ok / error
    failure_mode: Optional[str] = None
    m4_eligible: bool = False

    # === Repair (M4) ===
    repair_iterations: int = 0
    stop_reason: Optional[str] = None
    expression_history: List[str] = field(default_factory=list)
    sharpe_history: List[float] = field(default_factory=list)
    tried_strategies: List[str] = field(default_factory=list)

    def to_candidate(self):
        """向後相容 → 現有 Candidate"""
```

---

## 九、與現有專案的整合

### 9.1 需修改的檔案

| 檔案 | 變更內容 | 影響 |
|---|---|---|
| `alpha_agent/config.py` | 新增 `GridConfig`, `RAGConfig`, `GenConfig` | 低 |
| `alpha_agent/engine.py` | 插入 M0→M1→M2→M3→M3.5/M4 管線 | 中 |
| `alpha_agent/research_logic.py` | 重構修復邏輯使用新 FM + 目錄 | 中 |
| `alpha_agent/planner.py` | 保持不變（規劃器仍在最高層級） | 無 |
| `alpha_research_pipeline.py` | 新增 `build_candidate_from_sao()` | 低 |
| `alpha_research_agent.py` | CLI 新參數：`--enable-grid`, `--enable-rag`, `--enable-llm-gen` | 低 |
| `streamlit_app.py` | 側邊欄新增 Grid 視覺化、RAG 欄位預覽 | 低 |

### 9.2 不需修改的檔案

- `worldquant_brain_cli.py` — API 層完全保持不變
- `alpha_agent/planner.py` — 現有規劃器邏輯有效，僅 context 擴充
- `alpha_pipeline_ideas.json` — 現有想法庫作為 seed data 填充 Grid

### 9.3 新增檔案一覽

| 檔案 | 行數估算 | 依賴 |
|---|---|---|
| `alpha_agent/exploration_grid.py` | ~350 行 | 無 |
| `alpha_agent/rag_spec.py` | ~280 行 | sklearn (TfidfVectorizer) |
| `alpha_agent/failure_taxonomy.py` | ~200 行 | 無 |
| `alpha_agent/repair_engine.py` | ~400 行 | failure_taxonomy, sao |
| `alpha_agent/expression_generator.py` | ~300 行 | OpenAI API |
| `alpha_agent/sao.py` | ~200 行 | Candidate |
| `alpha_agent/error_recovery.py` | ~150 行 | worldquant_brain_cli |
| `tests/test_exploration_grid.py` | ~150 行 | exploration_grid |
| `tests/test_rag_spec.py` | ~120 行 | rag_spec |
| `tests/test_failure_taxonomy.py` | ~100 行 | failure_taxonomy |
| `tests/test_repair_engine.py` | ~200 行 | repair_engine |
| `tests/test_sao.py` | ~80 行 | sao |
| **修改現有檔案** | ~200 行 | 以上所有 |
| **總計** | ~2,830 行 | |

---

## 十、完整資料流範例

```
使用者輸入: 「根據 Fama-French 1992 HML 因子尋找 alpha」

M0 → Grid.select_cells(family=value, alignment=1.0)
    └─ 選中 [value × cross_sectional × long × neutralized]
        [value × ratio_spread × long × raw], ...

M1 → FieldRetriever.retrieve("high book-to-market stocks outperform...", top_k=10)
    └─ 返回 [("bookvalue_ps", 0.92), ("return_equity", 0.78), ...]
    └─ Validator → 通過 ✓

M2 → CoT prompt → LLM (T=0.7)
    └─ expression = "ts_mean(group_neutralize(rank(bookvalue_ps), industry), 120)"
    └─ DiversityGate → Jaccard=0.12 → 通過 ✓

M3 → simulate(expression) → α_id: A-12345
    └─ sharpe: 1.34, fitness: 1.02
    └─ checks: [PROD_CORRELATION=FAIL]

M4 → assess_recoverability()
    └─ PROD_CORRELATION → I1 硬停止: 結構性重複，不可修復
    └─ SAO.返回 M0

M0 → [value × cross_sectional × long × residual] (不同條件化)
M1 → 同一假說，新欄位組合
M2 → "group_zscore(rank(bookvalue_ps), industry)"
M3 → sharpe: 1.21, fitness: 1.08, ALL PASS ✓
    └─ 輸出 SAO → JSONL 儲存
    └─ GridCell.pass_count += 1
```

---

## 十一、實作順序建議

| 階段 | 內容 | 預估工作量 | 狀態 |
|---|---|---|---|---|
| **Phase 1a** | M0: exploration_grid.py + sao.py | 2-3 天 | ✅ 已完成 |
| **Phase 1b** | FM-1~6: failure_taxonomy.py + 修復目錄 | 1-2 天 | ✅ 已完成 |
| **Phase 1c** | M3.5: error_recovery.py | 0.5 天 | ✅ 已完成 |
| **Phase 2a** | M1: rag_spec.py (Tfidf + 6-rule) | 2 天 | ✅ 已完成 |
| **Phase 2b** | M4: repair_engine.py（含 LLM Phase 3 Rewrite T=0.3） | 2-3 天 | ✅ 已完成 |
| **Phase 3** | M2: expression_generator.py（含 LLM CoT T=0.7） | 2 天 | ✅ 已完成 |
| **Phase 4** | 整合 + engine.py 重構 + 測試 | 3 天 | ✅ 已完成 |
| **Phase 5** | Streamlit 擴充 + CLI 參數 | 1 天 | ✅ 已完成 |
| **共計** | | **~15 個工作日** | ✅ 全部完成 |

---

## 十二、與原論文的對照表

| 論文貢獻 | 論文章節 | 實作檔案 | 核心功能 |
|---|---|---|---|
| **C1** — Structure exploration grid | 4.1 M0 | `exploration_grid.py` | 660-cell, 4 階段生命週期, 優先級評分 |
| **C2** — RAG-enhanced specification | 4.2 M1 | `rag_spec.py` | Tfidf 欄位檢索 + 6-rule 驗證器 |
| **C3** — Failure-mode taxonomy | 3.4 (Table 3) | `failure_taxonomy.py` | FM-1~6 + CHECK_TO_FM 映射 |
| **C4** — Convergence-governed repair | 4.4 M4 | `repair_engine.py` | 3-phase + 5 個終止條件 |
| **C5** — Error recovery layer | 3.5 M3.5 | `error_recovery.py` | 3 策略逐步升級 |
| **P1** — Separation of concerns | 3.1 | `engine.py` | LLM 創造性 / Brain Oracle 事實 |
| **P2** — Stateful observability | 3.1 | `sao.py` | SAO 完整記錄 |
| **P3** — Structured feedback | 3.1 | `failure_taxonomy.py` | FM 分類 → 修復 |
| **P4** — Data-grounded generation | 3.1, 4.2, 4.3 | `rag_spec.py`, `expression_generator.py` | 真實欄位 + 運算子驗證 |
| **M2 diversity gate** | 4.3 | `expression_generator.py` | Jaccard diversity < 0.70 |
| **SAO (Table 2)** | 3.3 | `sao.py` | 7 群組欄位, 完整溯源 |

---

*本方案基於 BrainAlpha 論文 (SSRN, February 2026) 與 worldquant-alpha-research-agent 專案 (github.com/zeron-G) 的對比分析，於 2026 年 5 月制定。*
