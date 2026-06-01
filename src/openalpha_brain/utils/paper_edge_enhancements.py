"""Paper Edge Enhancements — 6 篇論文的邊緣概念輕量級實作

整合的論文邊緣概念：
  1. AlphaAgent (KDD'25)  — Economic Rationale Consistency Check
  2. CRANE (ICML'25)     — Grammar Fallback Chain
  3. CogAlpha (arXiv)    — Structural Novelty Scoring
  4. MCTS (arXiv)        — Adaptive Simulation Budget
  5. AlphaBench (ICLR'26) — Cross-Attempt Progress Metric
  6. Alpha-GPT (EMNLP'25) — Failure Pattern Clustering
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import re
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

from openalpha_brain.data import get_data_path

logger = logging.getLogger(__name__)

# ── 共用 regex 模式 ──────────────────────────────────────────────

_OPERATOR_RE = re.compile(
    r"\b(ts_\w+|rank|group_neutralize|group_rank|zscore|group_zscore|"
    r"scale|signed_power|abs|log|sign|delta|correlation|covariance|"
    r"ts_product|ts_sum|ts_min|ts_max|ts_argmax|ts_argmin|"
    r"ts_skewness|ts_kurtosis|ts_av_diff|ts_std_dev|regression)\b",
    re.IGNORECASE,
)
_FIELD_RE = re.compile(
    r"\b(close|open|high|low|vwap|volume|adv\d+|returns|cap|sales|"
    r"assets|equity|revenue|earnings|sharesout|vwap\d+)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\b(\d+)\b")

# 欄位類別映射
_PRICE_FIELDS = {"close", "open", "high", "low", "vwap", "returns"}
_VOLUME_FIELDS = {"volume", "adv20"}
_FUNDAMENTAL_FIELDS = {"cap", "sales", "assets", "equity", "revenue", "earnings", "sharesout", "liabilities", "debt"}

# 技術面運算子（時序差分/排名類）
_TECHNICAL_OPS = {
    "ts_delta",
    "ts_decay_linear",
    "ts_rank",
    "ts_regression",
    "ts_corr",
    "ts_covariance",
}
# 基本面運算子（統計/均值類）
_FUNDAMENTAL_OPS = {
    "ts_mean",
    "ts_std_dev",
    "ts_zscore",
    "ts_sum",
    "ts_product",
    "ts_av_diff",
    "ts_skewness",
    "ts_kurtosis",
}


# =====================================================================
# 2. CRANE: Grammar Fallback Chain
# =====================================================================


def build_grammar_fallback_chain(strict_grammar: str) -> list[str]:
    """建立 GBNF grammar fallback chain。

    當嚴格 grammar 導致 LLM 無法生成有效輸出時，
    逐級放寬約束：
      Level 0: 原始 strict grammar（完整 FASTEXPR）
      Level 1: 移除 group_neutralize 強制要求
      Level 2: 允許任意 root operator（不只是 rank/zscore）
      Level 3: 允許簡化的二元表達式（不需要 ts_op）
      Level 4: 最寬鬆 — 只要求 function(args) 格式

    Args:
        strict_grammar: 原始完整 FASTEXPR GBNF grammar 字串

    Returns:
        list of grammar strings, 從嚴格到寬鬆（長度 = 5）
    """
    chain: list[str] = []

    chain.append(strict_grammar)

    level1 = _relax_group_neutralize(strict_grammar)
    chain.append(level1)

    level2 = _relax_root_operator(level1)
    chain.append(level2)

    level3 = _allow_simplified_binary(level2)
    chain.append(level3)

    level4 = _minimal_function_format()
    chain.append(level4)

    logger.info(
        "Grammar fallback chain built: %d levels (strict → loose)",
        len(chain),
    )
    return chain


def _relax_group_neutralize(grammar: str) -> str:
    """Level 1: 移除 group_neutralize 作為強制 root 的要求。"""
    relaxed = grammar

    relaxed = re.sub(
        r"root\s*::=.*",
        "root ::= expr",
        relaxed,
        count=1,
    )

    if "group_neutralize(" not in relaxed:
        relaxed += (
            "\n\n"
            "# Relaxed: group_neutralize is optional at root\n"
            'root ::= "group_neutralize(" ws expr ws "," ws group-field ws ")" | expr'
        )

    return relaxed


def _relax_root_operator(grammar: str) -> str:
    """Level 2: 允許任意 unary-op 作為 root operator。"""
    relaxed = grammar

    root_match = re.search(r"root\s*::=.+", relaxed, re.MULTILINE)
    if root_match:
        existing_root = root_match.group(0)
        if "unary-op" not in existing_root:
            new_root = existing_root.rstrip() + ' | unary-op "(" ws expr ws ")"'
            relaxed = relaxed.replace(existing_root, new_root, 1)

    return relaxed


def _allow_simplified_binary(grammar: str) -> str:
    """Level 3: 允許簡化的二元表達式（data-field arith-op data-field 或 func-call(data-field)）。"""
    relaxed = grammar

    simplified_rule = (
        "\n\n"
        "# Level 3 relaxation: simplified expressions allowed\n"
        "simple-expr ::= data-field\n"
        '            | unary-op "(" ws simple-expr ws ")"\n'
        "            | simple-expr ws arith-op ws simple-expr\n"
        '            | ts-op "(" ws simple-expr ws "," ws integer ws ")"'
    )

    if "simple-expr" not in relaxed:
        relaxed += simplified_rule

    root_match = re.search(r"root\s*::=.+", relaxed, re.MULTILINE)
    if root_match and "simple-expr" not in root_match.group(0):
        existing_root = root_match.group(0)
        new_root = existing_root.rstrip() + " | simple-expr"
        relaxed = relaxed.replace(existing_root, new_root, 1)

    return relaxed


def _minimal_function_format() -> str:
    """Level 4: 最小 grammar — 只要求 function(args) 格式。"""
    return """# Level 4: Minimal grammar — only requires function(args) format
root ::= function-call
ws ::= " "?
function-call ::= name "(" ws args ws ")"
name ::= [a-z_] [a-z_0-9]*
args ::= arg ("," ws arg)*
arg ::= name | number | "-"? number
number ::= [0-9]+ ("." [0-9]+)?
"""


# =====================================================================
# 3. CogAlpha: Structural Novelty Scoring
# =====================================================================


def compute_structural_novelty_score(
    expression: str,
    known_expressions: list[str],
    parser: Any = None,
) -> float:
    """計算表達式的結構新奇性分數 (0~1)。

    基於 AST 子樹的新穎程度：
      - 計算 expression 的所有子樹 hash
      - 與 known_expressions 的子樹 hash database 比較
      - 新奇性 = 1 - (重複子樹比例)

    可用於：
      - FeatureMap diversity score 的加權因子
      - MCTS simulate 的 bonus 項
      - Selection signal for SignalArbiter

    Args:
        expression: 待評估的 FASTEXPR 表達式
        known_expressions: 已知表達式列表（用於比較基線）
        parser: 可選的 FASTEXPRParser 實例（若無則使用 regex 解析）

    Returns:
        float: 結構新奇性分數，0.0（完全重複）到 1.0（全新結構）
    """
    if not known_expressions:
        return 1.0

    target_subtrees = _extract_subtree_hashes(expression)
    if not target_subtrees:
        return 0.5

    known_db: set[str] = set()
    for known_expr in known_expressions:
        known_db.update(_extract_subtree_hashes(known_expr))

    if not known_db:
        return 1.0

    overlap_count = sum(1 for st in target_subtrees if st in known_db)
    novelty = 1.0 - (overlap_count / len(target_subtrees))

    depth_bonus = _compute_depth_novelty(expression, known_expressions)
    novelty = min(1.0, max(0.0, novelty + depth_bonus * 0.15))

    return round(novelty, 4)


def _extract_subtree_hashes(expression: str) -> list[str]:
    """從表達式中提取所有子樹的 hash 指紋。

    使用括號匹配提取每個函數呼叫及其內容作為「子樹」，
    然後對每個子樹做 hash。
    """
    subtrees: list[str] = []
    expr = expression.strip()

    func_pattern = re.compile(r"([a-zA-Z_]\w*)\s*\([^()]*(?:\([^()]*(?:\([^()]*\)[^()]*)*\)[^()]*)*\)", re.DOTALL)

    for match in func_pattern.finditer(expr):
        subtree = match.group(0).strip()
        normalized = re.sub(r"\s+", " ", subtree).strip()
        hash_val = hashlib.md5(normalized.encode()).hexdigest()[:12]
        subtrees.append(hash_val)

    if not subtrees:
        hash_val = hashlib.md5(re.sub(r"\s+", " ", expr).strip().encode()).hexdigest()[:12]
        subtrees.append(hash_val)

    return subtrees


def _compute_depth_novelty(expression: str, known_expressions: list[str]) -> float:
    """基於巢狀深度的分布計算新奇性加成。

    如果當前表達式的深度在已知表達式中很少見，給予加成。
    """
    target_depth = _compute_nesting_depth(expression)
    known_depths = [_compute_nesting_depth(e) for e in known_expressions]

    if not known_depths:
        return 0.5

    depth_counter = Counter(known_depths)
    total = len(known_depths)
    same_depth_ratio = depth_counter.get(target_depth, 0) / total

    rarity = 1.0 - same_depth_ratio
    return rarity


# =====================================================================
# 4. MCTS: Adaptive Simulation Budget
# =====================================================================


def compute_adaptive_simulation_budget(
    node_visits: int,
    node_reward_variance: float,
    parent_visits: int,
    base_budget: int = 20,
    min_budget: int = 5,
    max_budget: int = 50,
) -> int:
    """根據節點的不確定性自適應分配模擬預算。

    公式（基於 UCT 的 adaptation）::

        budget = base_budget × sqrt(variance / (reward_mean + epsilon)) × log(parent_visits + 1) / scaling

    高 variance + 低 visits → 更多模擬（需要更多信息）
    低 variance + high visits → 更少模擬（已足夠確定）

    Args:
        node_visits: 此節點已被訪問的次數
        node_reward_variance: 此節點獎值的方差
        parent_visits: 父節點的總訪問次數
        base_budget: 基礎預算
        min_budget: 最小預算下限
        max_budget: 最大預算上限

    Returns:
        int: 此節點應分配的模擬次數（clamp 到 [min_budget, max_budget]）
    """
    epsilon = 1e-6

    variance_factor = math.sqrt(node_reward_variance / (abs(node_reward_variance) + epsilon))
    if node_reward_variance < epsilon:
        variance_factor = 0.5

    visit_penalty = 1.0 / (1.0 + math.log(max(1, node_visits)) * 0.3)

    parent_log_factor = math.log(max(1, parent_visits) + 1) / math.log(100)

    raw_budget = base_budget * variance_factor * visit_penalty * parent_log_factor

    budget = int(round(raw_budget))
    budget = max(min_budget, min(max_budget, budget))

    return budget


# =====================================================================
# 5. AlphaBench: Cross-Attempt Progress Tracker
# =====================================================================


@dataclass
class _AttemptRecord:
    sharpe: float
    expression: str
    attempt_number: int


class CrossAttemptTracker:
    """追蹤多次 mutation 嘗試間的漸進改良情況。

    核心指標：
      - monotonicity_score: 分數是否單調改善（-1~1）
      - convergence_rate: 收斂速度（每步平均改善幅度）
      - oscillation_count: 方向反轉次數
      - best_running_sharpe: 歷史最佳

    用途：
      - 判斷是否應該繼續 mutation 還是重新開始
      - StabilityGuard 的輔助輸入
    """

    def __init__(self, window_size: int = 10):
        self._window_size = window_size
        self._records: deque[_AttemptRecord] = deque(maxlen=window_size)
        self._best_sharpe: float = float("-inf")
        self._best_expression: str = ""
        self._total_attempts: int = 0
        self._oscillation_count: int = 0
        self._prev_direction: int = 0

    def record(self, sharpe: float, expression: str) -> dict:
        """記錄一次嘗試結果。

        Args:
            sharpe: 本次嘗試的 Sharpe ratio
            expression: 本次嘗試的表達式

        Returns:
            包含本次分析結果的字典
        """
        self._total_attempts += 1

        if sharpe is not None and sharpe > self._best_sharpe:
            self._best_sharpe = sharpe
            self._best_expression = expression

        rec = _AttemptRecord(
            sharpe=sharpe or 0.0,
            expression=expression,
            attempt_number=self._total_attempts,
        )
        self._records.append(rec)

        if len(self._records) >= 2:
            prev_sharpe = self._records[-2].sharpe
            curr_direction = 1 if (sharpe or 0) > prev_sharpe else (-1 if (sharpe or 0) < prev_sharpe else 0)

            if self._prev_direction != 0 and curr_direction != 0 and curr_direction != self._prev_direction:
                self._oscillation_count += 1

            self._prev_direction = curr_direction

        metrics = self.get_progress_metrics()
        metrics["attempt_number"] = self._total_attempts
        metrics["should_continue"] = self.should_continue_mutating()

        return metrics

    def get_progress_metrics(self) -> dict:
        """返回當前所有進度指標。

        Returns:
            {
                "monotonicity_score": float,   # -1~1
                "convergence_rate": float,      # 平均每步改善幅度
                "oscillation_count": int,       # 方向反轉次數
                "best_running_sharpe": float,   # 歷史最佳 Sharpe
                "best_running_expression": str, # 歷史最佳表達式
                "window_utilization": float,    # 視窗使用率 0~1
                "total_attempts": int,          # 總嘗試次數
            }
        """
        records_list = list(self._records)

        if len(records_list) < 2:
            return {
                "monotonicity_score": 1.0,
                "convergence_rate": 0.0,
                "oscillation_count": 0,
                "best_running_sharpe": self._best_sharpe,
                "best_running_expression": self._best_expression,
                "window_utilization": len(records_list) / self._window_size,
                "total_attempts": self._total_attempts,
            }

        sharpes = [r.sharpe for r in records_list]

        improvements = [sharpes[i] - sharpes[i - 1] for i in range(1, len(sharpes))]

        positive_count = sum(1 for d in improvements if d > 0)
        negative_count = sum(1 for d in improvements if d < 0)
        total_changes = positive_count + negative_count

        monotonicity = ((positive_count - negative_count) / total_changes) if total_changes > 0 else 1.0

        convergence_rate = sum(improvements) / len(improvements) if improvements else 0.0

        return {
            "monotonicity_score": round(monotonicity, 4),
            "convergence_rate": round(convergence_rate, 6),
            "oscillation_count": self._oscillation_count,
            "best_running_sharpe": round(self._best_sharpe, 4),
            "best_running_expression": self._best_expression,
            "window_utilization": round(len(records_list) / self._window_size, 4),
            "total_attempts": self._total_attempts,
        }

    def should_continue_mutating(self) -> bool:
        """判斷是否值得繼續 mutation。

        停止條件（任一滿足即停止）：
          - oscillation_count >= window_size 的 40%
          - monotonicity_score < -0.3 且已嘗試超過 5 次
          - best_running_sharpe == 0 且已嘗試超過 8 次（完全沒改善）

        Returns:
            True 表示值得繼續，False 建議停止或重新開始
        """
        if len(self._records) < 2:
            return True

        metrics = self.get_progress_metrics()
        osc_ratio = metrics["oscillation_count"] / max(1, len(self._records) - 1)

        if osc_ratio >= 0.40:
            return False

        if metrics["monotonicity_score"] < -0.3 and self._total_attempts >= 5:
            return False

        return not (self._best_sharpe <= 0 and self._total_attempts >= 8)

    @property
    def best_sharpe(self) -> float:
        return self._best_sharpe

    @property
    def best_expression(self) -> str:
        return self._best_expression

    def reset(self) -> None:
        """重置 tracker 狀態。"""
        self._records.clear()
        self._best_sharpe = float("-inf")
        self._best_expression = ""
        self._total_attempts = 0
        self._oscillation_count = 0
        self._prev_direction = 0


# =====================================================================
# 6. Alpha-GPT: Failure Pattern Clustering
# =====================================================================


@dataclass
class _FailureFingerprint:
    root_operator: str
    composition_type: str
    nesting_depth_bin: int
    field_category: str
    failure_type: str


_FAILURE_FIX_SUGGESTIONS: dict[tuple[str, ...], str] = {
    ("rank", "nested", "SHARPE"): "降低巢狀深度，改用 ts_decay_linear 平滑後再 rank；考慮加入基本面欄位增加訊號來源",
    ("rank", "nested", "TURNOVER"): "替換外層 rank 為 group_zscore 或 scale，減少換手率；加入 ts_decay_linear 衰減",
    ("ts_delta", "single", "SHARPE"): "單一 ts_delta 訊號太弱，改為組合多個時間視窗的 delta 或加入交叉項",
    ("group_neutralize", "ratio", "FITNESS"): "比率結構可能過擬合，改為加法組合或加入正則化約束",
    ("zscore", "chain", "CORRELATION"): "高相關性：替換部分欄位為不同類別（如從 close 改為 volume-derived）",
    ("ts_std_dev", "single", "SHARPE"): "單一波動率訊號不足，考慮 ts_av_diff 或 skewness 等更高階矩",
    ("scale", "nested", "OVERFITTING"): "深層嵌套 + scale 容易過擬合，簡化結構並使用 group_neutralize 代替",
}


def cluster_failure_patterns(
    failure_history: list[dict],
    n_clusters: int = 5,
) -> list[dict]:
    """將歷史失敗按結構模式聚類。

    每個 failure dict 包含：
      - expression: str
      - failure_type: str (SHARPE/TURNOVER/FITNESS/etc.)
      - sharpe: float | None

    聚類特徵（每個 failure 的 fingerprint）：
      - root_operator
      - composition_type
      - nesting_depth (binned)
      - field_category (price/volume/fundamental/mixed)
      - failure_type

    使用簡易的 k-medoids（不需要 sklearn，純 Python 實作）：
      1. 計算所有 pair-wise distance
      2. 隨機選初始 medoids
      3. 迭代分配+更新直到收斂

    Args:
        failure_history: 失敗記錄列表
        n_clusters: 聚類數量（預設 5）

    Returns:
        [
            {
                "cluster_id": int,
                "medoid": dict,       # 代表性 failure
                "members": list[dict],
                "dominant_failure_type": str,
                "suggested_fix": str,  # 基於 cluster 特徵的建議
                "size": int,
            },
            ...
        ]
    """
    if not failure_history:
        return []

    if len(failure_history) <= 2:
        return [
            {
                "cluster_id": 0,
                "medoid": failure_history[0],
                "members": failure_history,
                "dominant_failure_type": failure_history[0].get("failure_type", "UNKNOWN"),
                "suggested_fix": "樣本不足，建議收集更多失敗案例後再進行聚類",
                "size": len(failure_history),
            }
        ]

    fingerprints: list[_FailureFingerprint] = []
    for fh in failure_history:
        fp = _build_failure_fingerprint(fh)
        fingerprints.append(fp)

    n_clusters = min(n_clusters, len(failure_history))

    medoid_indices = _kmedoids_cluster(fingerprints, n_clusters, max_iter=20)

    clusters: list[dict] = []
    for cid, mid_idx in enumerate(medoid_indices):
        members = []
        for i, fp in enumerate(fingerprints):
            _fingerprint_distance(fp, fingerprints[mid_idx])
            assigned_cid = _find_nearest_medoid_index(i, fingerprints, medoid_indices)
            if assigned_cid == cid:
                members.append(failure_history[i])

        if not members:
            continue

        dominant_ft = Counter(m["failure_type"] for m in members).most_common(1)[0][0]

        medoid_fp = fingerprints[mid_idx]
        suggestion_key = (
            medoid_fp.root_operator,
            medoid_fp.composition_type,
            dominant_ft,
        )
        suggested_fix = _FAILURE_FIX_SUGGESTIONS.get(
            suggestion_key,
            _generate_generic_suggestion(medoid_fp, dominant_ft, members),
        )

        clusters.append(
            {
                "cluster_id": cid,
                "medoid": failure_history[mid_idx],
                "members": members,
                "dominant_failure_type": dominant_ft,
                "suggested_fix": suggested_fix,
                "size": len(members),
            }
        )

    clusters.sort(key=lambda c: c["size"], reverse=True)
    return clusters


def _build_failure_fingerprint(failure: dict) -> _FailureFingerprint:
    """從 failure dict 建構結構指紋。"""
    expression = failure.get("expression", "")

    root_match = re.match(r"\s*(\w+)\s*\(", expression)
    root_operator = (root_match.group(1).lower() if root_match else "unknown")[:20]

    depth = _compute_nesting_depth(expression)
    if depth <= 2:
        depth_bin = 0
    elif depth <= 4:
        depth_bin = 1
    else:
        depth_bin = 2

    expr_fields = [f.lower() for f in _FIELD_RE.findall(expression)]
    field_set = set(expr_fields)
    has_price = bool(field_set & _PRICE_FIELDS)
    has_volume = bool(field_set & _VOLUME_FIELDS)
    has_fundamental = bool(field_set & _FUNDAMENTAL_FIELDS)

    if has_fundamental and (has_price or has_volume):
        field_category = "mixed"
    elif has_fundamental:
        field_category = "fundamental"
    elif has_volume and not has_price:
        field_category = "volume"
    else:
        field_category = "price"

    ops_found = _OPERATOR_RE.findall(expression)
    has_divide = "/" in expression
    has_multiply = "*" in expression
    has_plus = "+" in expression

    if has_divide:
        composition_type = "ratio"
    elif has_multiply:
        composition_type = "multiplicative"
    elif has_plus:
        composition_type = "additive"
    elif len(ops_found) >= 4:
        composition_type = "chain"
    elif depth >= 4:
        composition_type = "nested"
    else:
        composition_type = "single"

    failure_type = str(failure.get("failure_type", "UNKNOWN")).upper()[:20]

    return _FailureFingerprint(
        root_operator=root_operator,
        composition_type=composition_type,
        nesting_depth_bin=depth_bin,
        field_category=field_category,
        failure_type=failure_type,
    )


def _fingerprint_distance(a: _FailureFingerprint, b: _FailureFingerprint) -> float:
    """計算兩個 failure fingerprint 之間的距離（0~1）。"""
    distance = 0.0
    n_dims = 0

    if a.root_operator != b.root_operator:
        distance += 1.0
    n_dims += 1

    if a.composition_type != b.composition_type:
        distance += 1.0
    n_dims += 1

    distance += abs(a.nesting_depth_bin - b.nesting_depth_bin) / 2.0
    n_dims += 1

    if a.field_category != b.field_category:
        distance += 0.5
    n_dims += 1

    if a.failure_type != b.failure_type:
        distance += 0.3
    n_dims += 1

    return distance / n_dims if n_dims > 0 else 0.0


def _kmedoids_cluster(
    fingerprints: list[_FailureFingerprint],
    k: int,
    max_iter: int = 20,
) -> list[int]:
    """簡易 K-medoids 聚類（純 Python 實作）。

    Returns:
        medoid 索引列表
    """
    n = len(fingerprints)
    if k >= n:
        return list(range(n))

    random.seed(42)
    medoid_indices = random.sample(range(n), k)

    for _iteration in range(max_iter):
        clusters: dict[int, list[int]] = {cid: [] for cid in range(k)}

        for i in range(n):
            nearest = _find_nearest_medoid_index(i, fingerprints, medoid_indices)
            clusters[nearest].append(i)

        new_medoids: list[int] = []
        for cid in range(k):
            members = clusters[cid]
            if not members:
                new_medoids.append(medoid_indices[cid])
                continue

            best_total_dist = float("inf")
            best_idx = members[0]

            for candidate in members:
                total_dist = sum(_fingerprint_distance(fingerprints[candidate], fingerprints[m]) for m in members)
                if total_dist < best_total_dist:
                    best_total_dist = total_dist
                    best_idx = candidate

            new_medoids.append(best_idx)

        if new_medoids == medoid_indices:
            break

        medoid_indices = new_medoids

    return medoid_indices


def _find_nearest_medoid_index(
    idx: int,
    fingerprints: list[_FailureFingerprint],
    medoid_indices: list[int],
) -> int:
    """找出距離 idx 最近的 medoid 的 cluster ID。"""
    best_dist = float("inf")
    best_cid = 0

    for cid, mid_idx in enumerate(medoid_indices):
        d = _fingerprint_distance(fingerprints[idx], fingerprints[mid_idx])
        if d < best_dist:
            best_dist = d
            best_cid = cid

    return best_cid


def _generate_generic_suggestion(
    medoid_fp: _FailureFingerprint,
    dominant_failure_type: str,
    members: list[dict],
) -> str:
    """根據 cluster 特徵生成通用修復建議。"""
    parts: list[str] = []

    if medoid_fp.composition_type == "nested":
        parts.append("此集群失敗模式以深層巢狀結構為主，建議簡化表達式深度至 ≤ 4")

    if medoid_fp.field_category == "price":
        parts.append("僅使用價格欄位導致訊號同質化，建議引入 volume 或 fundamental 欄位")

    if medoid_fp.root_operator == "rank":
        parts.append("rank 作為 root operator 可能限制了表達力，嘗試 zscore 或 scale")

    if dominant_failure_type == "TURNOVER":
        parts.append("換手率問題：加入 ts_decay_linear 或延長視窗參數")

    if dominant_failure_type == "SHARPE":
        parts.append("Sharpe 不足：考慮更強的預測變數組合或不同的時間尺度")

    if dominant_failure_type == "FITNESS":
        parts.append("Fitness 不达标：可能需要更多樣化的因子組合")

    if not parts:
        size = len(members)
        parts.append(
            f"集群包含 {size} 個相似失敗案例 "
            f"(root={medoid_fp.root_operator}, type={medoid_fp.composition_type})，"
            "建議嘗試完全不同的結構方向",
        )

    return "；".join(parts)


# ── 共用工具函數 ───────────────────────────────────────────────────


def _compute_nesting_depth(expression: str) -> int:
    """計算表達式的括號巢狀深度。"""
    max_depth = 0
    current_depth = 0
    for ch in expression:
        if ch == "(":
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif ch == ")":
            current_depth = max(0, current_depth - 1)
    return max_depth


# ── 內嵌測試 ───────────────────────────────────────────────────────

if __name__ == "__main__":
    from openalpha_brain.evolution.hypothesis_aligner import HypothesisAligner

    print("=" * 60)
    print("Paper Edge Enhancements — 內嵌測試")
    print("=" * 60)

    aligner = HypothesisAligner()

    # Test 1: Categorical Consistency (via HypothesisAligner)
    print("\n[Test 1] Categorical Consistency (via HypothesisAligner)")

    from openalpha_brain.evolution.hypothesis_aligner import HypothesisAligner

    _aligner = HypothesisAligner()

    expr_good = "group_neutralize(rank(ts_delta(close, 5)), industry)"
    _ops_good = [o.lower() for o in set(_OPERATOR_RE.findall(expr_good))]
    _fields_good = [f.lower() for f in set(_FIELD_RE.findall(expr_good))]
    result_good = _aligner._check_categorical_consistency("momentum_breakout", _ops_good, _fields_good, expr_good)
    assert result_good["is_consistent"], f"Expected consistent, got {result_good}"
    print(f"  OK: {expr_good[:50]}... + momentum_breakout -> {result_good['explanation']}")

    expr_bad = "group_neutralize(rank(log(cap / earnings)), industry)"
    _ops_bad = [o.lower() for o in set(_OPERATOR_RE.findall(expr_bad))]
    _fields_bad = [f.lower() for f in set(_FIELD_RE.findall(expr_bad))]
    result_bad = _aligner._check_categorical_consistency("momentum_breakout", _ops_bad, _fields_bad, expr_bad)
    assert not result_bad["is_consistent"], f"Expected inconsistent, got {result_bad}"
    print(f"  OK: {expr_bad[:50]}... + momentum_breakout -> {result_bad['explanation']}")

    # Test 2: Grammar Fallback Chain
    print("\n[Test 2] Grammar Fallback Chain")
    with open(get_data_path("fastexpr_grammar.gbnf"), encoding="utf-8") as f:
        strict_g = f.read()
    chain = build_grammar_fallback_chain(strict_g)
    assert len(chain) == 5, f"Expected 5 levels, got {len(chain)}"
    for i, g in enumerate(chain):
        print(f"  Level {i}: {len(g)} chars, starts with: {g[:50].split(chr(10))[0]}...")
    print("  OK: 5-level fallback chain built")

    # Test 3: Structural Novelty Score
    print("\n[Test 3] Structural Novelty Score")
    known = [
        "group_neutralize(rank(ts_delta(close, 5)), industry)",
        "group_neutralize(rank(ts_delta(close, 10)), industry)",
        "group_neutralize(rank(ts_delta(vwap, 5)), industry)",
    ]
    novel_expr = "group_neutralize(zscore(ts_regression(earnings, volume, 20)), sector)"
    novelty = compute_structural_novelty_score(novel_expr, known)
    assert 0.0 <= novelty <= 1.0, f"Novelty out of range: {novelty}"
    assert novelty > 0.3, f"Novel expression should score higher, got {novelty}"
    print(f"  OK: novel_expr novelty = {novelty:.3f}")

    duplicate_novelty = compute_structural_novelty_score(known[0], known)
    assert duplicate_novelty < novelty, "Duplicate should have lower novelty"
    print(f"  OK: duplicate novelty = {duplicate_novelty:.3f} (lower as expected)")

    # Test 4: Adaptive Simulation Budget
    print("\n[Test 4] Adaptive Simulation Budget")
    budget_high_var = compute_adaptive_simulation_budget(
        node_visits=2,
        node_reward_variance=0.5,
        parent_visits=10,
    )
    budget_low_var = compute_adaptive_simulation_budget(
        node_visits=50,
        node_reward_variance=0.01,
        parent_visits=200,
    )
    assert 5 <= budget_high_var <= 50
    assert 5 <= budget_low_var <= 50
    print(f"  OK: high_var (visits=2, var=0.5) → budget={budget_high_var}")
    print(f"  OK: low_var (visits=50, var=0.01) → budget={budget_low_var}")

    # Test 5: CrossAttemptTracker
    print("\n[Test 5] CrossAttemptTracker")
    tracker = CrossAttemptTracker(window_size=10)
    sharpes = [0.5, 0.8, 1.2, 1.0, 1.5, 1.3, 1.8]
    for i, s in enumerate(sharpes):
        rec = tracker.record(s, f"expr_{i}")
    metrics = tracker.get_progress_metrics()
    assert metrics["best_running_sharpe"] == 1.8
    assert metrics["oscillation_count"] >= 1
    assert tracker.should_continue_mutating()
    print(
        f"  OK: monotonicity={metrics['monotonicity_score']:.3f}, "
        f"oscillations={metrics['oscillation_count']}, best={metrics['best_running_sharpe']}"
    )

    # Test 6: Failure Pattern Clustering
    print("\n[Test 6] Failure Pattern Clustering")
    failures = [
        {"expression": "rank(ts_delta(close, 5))", "failure_type": "SHARPE", "sharpe": 0.3},
        {"expression": "rank(ts_delta(close, 10))", "failure_type": "SHARPE", "sharpe": 0.4},
        {
            "expression": "group_neutralize(zscore(ts_std_dev(volume, 20)), industry)",
            "failure_type": "TURNOVER",
            "sharpe": 0.9,
        },
        {"expression": "scale(ts_mean(earnings, 60))", "failure_type": "FITNESS", "sharpe": 0.7},
        {"expression": "rank(ts_delta(open, 5))", "failure_type": "SHARPE", "sharpe": 0.35},
        {"expression": "group_neutralize(rank(log(cap/sales)), industry)", "failure_type": "TURNOVER", "sharpe": 0.85},
        {"expression": "zscore(ts_corr(close, volume, 20))", "failure_type": "CORRELATION", "sharpe": 0.6},
    ]
    clusters = cluster_failure_patterns(failures, n_clusters=3)
    assert len(clusters) >= 2, f"Expected ≥2 clusters, got {len(clusters)}"
    total_members = sum(c["size"] for c in clusters)
    assert total_members == len(failures), f"Member count mismatch: {total_members} vs {len(failures)}"
    for c in clusters:
        print(
            f"  Cluster {c['cluster_id']}: size={c['size']}, "
            f"dominant={c['dominant_failure_type']}, fix={c['suggested_fix'][:60]}..."
        )
    print("  OK: clustering completed")

    print("\n" + "=" * 60)
    print("所有測試通過 ✅")
    print("=" * 60)
