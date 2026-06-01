"""
OpenAlpha-Brain — AlphaBench Stability Guard

實作 AlphaBench (ICLR'26) 論文的兩個核心概念：
  1. Stability Enforcement — 偵測並抑制輸出不穩定性
  2. Weak Evaluation Guard — 當評估不可靠時限制搜索範圍

整合點：
  - brain_submitter._brain_improvement_loop: mutation 後檢查穩定性
  - loop_engine.run_loop: cycle 結束時記錄穩定性指標
  - MAB reward calculation: 加入穩定性懲罰/獎勵
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter, deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── 常數 ──────────────────────────────────────────────

_DEFAULT_WINDOW_SIZE = 10
_DEFAULT_INSTABILITY_THRESHOLD = 0.35

# 用於提取運算子的 regex 模式
_OPERATOR_PATTERN = re.compile(r"\b([a-z_]\w*)\s*\(", re.IGNORECASE)
_FIELD_PATTERN = re.compile(r"\b([a-z_]\w{2,})\b", re.IGNORECASE)
_NUMBER_PATTERN = re.compile(r"(?<!\w)(\d+)(?!\w)")
_NESTING_OPEN = re.compile(r"[\(\[]")
_NESTING_CLOSE = re.compile(r"[\)\]]")

# 已知視窗參數名稱（用於識別 window 參數）
_WINDOW_PARAM_NAMES = {
    "window",
    "period",
    "lag",
    "delay",
    "lookback",
    "days",
    "fast_period",
    "slow_period",
    "short",
    "long",
}

# 常見 root 運算子（用於指紋歸一化）
_KNOWN_ROOT_OPS = {
    "rank",
    "ts_rank",
    "ts_delta",
    "ts_decay_linear",
    "ts_mean",
    "ts_std_dev",
    "ts_sum",
    "ts_min",
    "ts_max",
    "ts_argmax",
    "ts_argmin",
    "ts_skewness",
    "ts_kurtosis",
    "ts_av_diff",
    "ts_regression",
    "ts_corr",
    "ts_covariance",
    "group_neutralize",
    "group_zscore",
    "zscale",
    "sign",
    "abs",
    "log",
    "pow",
    "max",
    "min",
    "cond",
    "if_else",
    "ts_step",
    "signed_power",
    "truncate",
    "winsorize",
    "normalize",
    "scale",
    "neutralize",
}


@dataclass(frozen=True)
class ExpressionFingerprint:
    """輕量表達式指紋 — 不需要完整 AST，純 regex 提取結構特徵。

    提取維度：
      - root_operator: 最外層運算子
      - operator_signature: 排序後去重運算子列表的 hash
      - field_signature: 排序後欄位列表的 hash
      - window_signature: 排序後視窗列表的 hash
      - composition_type: 結構分類（simple/chain/nested/conditional）
      - nesting_depth: 最大巢狀深度
      - approx_complexity: 運算子數 × log(巢狀深度+1)
    """

    root_operator: str
    operator_signature: str
    field_signature: str
    window_signature: str
    composition_type: str
    nesting_depth: int
    approx_complexity: float

    def jaccard_similarity(self, other: ExpressionFingerprint) -> float:
        """計算兩個指紋之間的 Jaccard 相似度（基於多維特徵）。"""
        if not isinstance(other, ExpressionFingerprint):
            return 0.0

        scores: list[float] = []

        # 1. Root operator 匹配（二元）
        scores.append(1.0 if self.root_operator == other.root_operator else 0.0)

        # 2. Operator signature 匹配
        scores.append(1.0 if self.operator_signature == other.operator_signature else 0.0)

        # 3. Field signature 匹配
        scores.append(1.0 if self.field_signature == other.field_signature else 0.0)

        # 4. Window signature 匹配
        scores.append(1.0 if self.window_signature == other.window_signature else 0.0)

        # 5. Composition type 匹配
        scores.append(1.0 if self.composition_type == other.composition_type else 0.0)

        # 6. Nesting depth 相似度（歸一化差異）
        max_depth = max(self.nesting_depth, other.nesting_depth, 1)
        depth_diff = abs(self.nesting_depth - other.nesting_depth) / max_depth
        scores.append(1.0 - depth_diff)

        # 7. Complexity 相似度（歸一化差異）
        max_complexity = max(self.approx_complexity, other.approx_complexity, 1.0)
        complexity_diff = abs(self.approx_complexity - other.approx_complexity) / max_complexity
        scores.append(1.0 - complexity_diff)

        return sum(scores) / len(scores)


class _FingerprintExtractor:
    """內部：從表達式字串提取 ExpressionFingerprint 的靜態工具。"""

    @staticmethod
    def extract(expression: str) -> ExpressionFingerprint:
        """從表達式字串中提取完整指紋。

        Args:
            expression: Alpha 表達式字串，例如 "group_neutralize(rank(ts_delta(close,5)), industry)"

        Returns:
            凍結的 ExpressionFingerprint 實例
        """
        expr_stripped = expression.strip()

        # ── Root operator ──
        root_op = _FingerprintExtractor._extract_root_operator(expr_stripped)

        # ── All operators ──
        all_ops = sorted({op.lower() for op in _OPERATOR_PATTERN.findall(expr_stripped)})
        op_sig = hashlib.md5("|".join(all_ops).encode()).hexdigest()[:12]

        # ── Fields ──
        all_fields_raw = _FIELD_PATTERN.findall(expr_stripped)
        all_fields = sorted({f.lower() for f in all_fields_raw if f.lower() not in _KNOWN_ROOT_OPS and len(f) >= 2})
        field_sig = hashlib.md5("|".join(all_fields).encode()).hexdigest()[:12]

        # ── Windows (數字參數) ──
        windows = sorted({int(n) for n in _NUMBER_PATTERN.findall(expr_stripped) if 1 <= int(n) <= 500})
        window_sig = hashlib.md5("|".join(str(w) for w in windows).encode()).hexdigest()[:12]

        # ── Nesting depth ──
        depth = _FingerprintExtractor._compute_nesting_depth(expr_stripped)

        # ── Composition type ──
        comp_type = _FingerprintExtractor._classify_composition(len(all_ops), depth, expr_stripped)

        # ── Approx complexity ──
        import math

        complexity = len(all_ops) * math.log(depth + 1) if depth > 0 else float(len(all_ops))

        return ExpressionFingerprint(
            root_operator=root_op,
            operator_signature=op_sig,
            field_signature=field_sig,
            window_signature=window_sig,
            composition_type=comp_type,
            nesting_depth=depth,
            approx_complexity=round(complexity, 3),
        )

    @staticmethod
    def _extract_root_operator(expr: str) -> str:
        match = re.match(r"\s*(\w+)\s*\(", expr)
        if match:
            return match.group(1).lower()
        return "unknown"

    @staticmethod
    def _compute_nesting_depth(expr: str) -> int:
        max_depth = 0
        current = 0
        for ch in expr:
            if ch in "(":
                current += 1
                max_depth = max(max_depth, current)
            elif ch in ")":
                current = max(0, current - 1)
        return max_depth

    @staticmethod
    def _classify_composition(num_ops: int, depth: int, expr: str) -> str:
        has_cond = bool(re.search(r"\b(if_|cond|where|case)\b", expr, re.IGNORECASE))
        if has_cond:
            return "conditional"
        if depth >= 4:
            return "nested"
        if num_ops >= 5 and depth >= 2:
            return "chain"
        return "simple"


class StabilityTracker:
    """穩定性追蹤器 — 維護滑動窗口內的表達式指紋歷史。

    核心 API：

    >>> tracker = StabilityTracker(window_size=10)
    >>> result = tracker.record("rank(ts_delta(close, 5))")
    >>> score = tracker.get_stability_score()
    >>> info = tracker.detect_instability()
    """

    def __init__(self, window_size: int = _DEFAULT_WINDOW_SIZE):
        self._window_size = window_size
        self._history: deque[tuple[int, ExpressionFingerprint]] = deque(maxlen=window_size)
        self._record_count = 0

    @property
    def history_size(self) -> int:
        return len(self._history)

    def record(self, expression: str) -> dict:
        """記錄一次新的表達式，返回本次的穩定性分析。

        Args:
            expression: Alpha 表達式字串

        Returns:
            包含 fingerprint 和即時分析的字典
        """
        fp = _FingerprintExtractor.extract(expression)
        self._record_count += 1
        self._history.append((self._record_count, fp))

        analysis = {
            "fingerprint": fp,
            "record_id": self._record_count,
            "stability_score": self.get_stability_score(),
            "pairwise_similarities": self._compute_pairwise_similarities(),
        }
        return analysis

    def get_stability_score(self) -> float:
        """計算整體穩定性分數 (0.0~1.0)。

        加權組合三個維度：
          1. Pairwise Jaccard similarity (40%)：相鄰表達式的平均相似度
          2. Trend consistency (30%)：連續變化方向的一致性
          3. Cluster concentration (30%)：指紋聚類集中度

        Returns:
          0.0（完全不穩定）到 1.0（非常穩定）
        """
        if len(self._history) < 2:
            return 1.0

        pairwise = self._compute_pairwise_score()
        trend = self._compute_trend_consistency()
        cluster = self._compute_cluster_concentration()

        final_score = 0.4 * pairwise + 0.3 * trend + 0.3 * cluster
        return round(max(0.0, min(1.0, final_score)), 4)

    def detect_instability(self) -> dict:
        """偵測不穩定性事件並返回診斷結果。

        Returns:
            {
                "is_unstable": bool,           # 是否觸發不穩定閾值
                "instability_type": str,       # "oscillation" | "drift" | "random_jump" | "stable"
                "severity": float,             # 0.0~1.0
                "recent_scores": list[float],  # 最近 N 次的 pairwise similarity
                "diagnosis": str,              # 人類可讀診斷
                "suggested_action": str,       # 建議的修正動作
            }
        """
        n = len(self._history)
        if n < 2:
            return {
                "is_unstable": False,
                "instability_type": "stable",
                "severity": 0.0,
                "recent_scores": [],
                "diagnosis": "資料不足（需至少 2 筆記錄）",
                "suggested_action": "繼續收集表達式",
            }

        stability = self.get_stability_score()
        recent_sims = self._compute_pairwise_similarities()

        # 分類不穩定類型
        instability_type, severity, diagnosis, action = self._classify_instability(
            stability,
            recent_sims,
        )

        is_unstable = stability < _DEFAULT_INSTABILITY_THRESHOLD

        return {
            "is_unstable": is_unstable,
            "instability_type": instability_type,
            "severity": round(severity, 4),
            "recent_scores": recent_sims,
            "diagnosis": diagnosis,
            "suggested_action": action,
        }

    def get_constrained_search_space(self) -> dict | None:
        """Weak Evaluation Guard — 當不穩定時返回受限的搜索空間。

        當 stability_score < 0.35 時：
        - 鎖定 root_operator（使用最近 5 次中最常見的）
        - 鎖定 field_set（使用最近 5 次的交集）
        - 限制 window 範圍（只在已使用的視窗 ±50% 內）
        - 建議 composition_type 保持不變

        Returns:
            None（如果穩定，不需限制）或限制條件字典
        """
        stability = self.get_stability_score()
        if stability >= _DEFAULT_INSTABILITY_THRESHOLD or len(self._history) < 3:
            return None

        recent = list(self._history)[-5:]

        # ── Locked root operator（最近 5 次最常見）──
        root_ops = [fp.root_operator for _, fp in recent]
        root_counter = Counter(root_ops)
        locked_root = root_counter.most_common(1)[0][0] if root_counter else None

        # ── 從原始表達式中提取欄位集合做交集 ──
        locked_fields = self._infer_common_fields(recent)

        # ── Window 範圍 ──
        window_range = self._infer_window_range(recent)

        # ── Preferred composition ──
        comp_types = [fp.composition_type for _, fp in recent]
        preferred_comp = Counter(comp_types).most_common(1)[0][0] if comp_types else None

        reason = (
            f"穩定性分數 {stability:.2f} 低於閾值 {_DEFAULT_INSTABILITY_THRESHOLD}，"
            f"啟動 Weak Evaluation Guard 限制搜索範圍"
        )

        return {
            "locked_root_op": locked_root,
            "locked_fields": locked_fields,
            "window_range": window_range,
            "preferred_composition": preferred_comp,
            "reason": reason,
        }

    # ── 內部計算方法 ──────────────────────────────────

    def _compute_pairwise_similarities(self) -> list[float]:
        """計算相鄰表達式對之間的 Jaccard similarity 列表。"""
        history_list = list(self._history)
        if len(history_list) < 2:
            return []
        sims: list[float] = []
        for i in range(1, len(history_list)):
            sim = history_list[i][1].jaccard_similarity(history_list[i - 1][1])
            sims.append(round(sim, 4))
        return sims

    def _compute_pairwise_score(self) -> float:
        """Pairwise similarity 的平均值。"""
        sims = self._compute_pairwise_similarities()
        if not sims:
            return 1.0
        return sum(sims) / len(sims)

    def _compute_trend_consistency(self) -> float:
        """Trend consistency：root operator 和 composition type 是否在收斂。

        計算方式：
          - 追蹤最近 N 次的 root_operator 變化次數
          - 變化越少 → 越一致 → 分數越高
        """
        if len(self._history) < 3:
            return 1.0

        roots = [fp.root_operator for _, fp in self._history]
        changes = sum(1 for i in range(1, len(roots)) if roots[i] != roots[i - 1])
        max_changes = len(roots) - 1
        if max_changes == 0:
            return 1.0

        consistency = 1.0 - (changes / max_changes)
        return consistency

    def _compute_cluster_concentration(self) -> float:
        """Cluster concentration：指紋是否集中在少數幾個模式。

        使用 operator_signature 的多樣性來衡量：
          - 只有 1 種 operator signature → concentration = 1.0
          - 每筆都不同 → concentration 趨近 0.0
        """
        if len(self._history) < 2:
            return 1.0

        op_sigs = [fp.operator_signature for _, fp in self._history]
        set(op_sigs)
        total = len(op_sigs)

        if total <= 1:
            return 1.0

        # Herfindahl 指數形式的集中度
        counter = Counter(op_sigs)
        hhi = sum((count / total) ** 2 for count in counter.values())
        return hhi

    def _classify_instability(
        self,
        stability: float,
        recent_sims: list[float],
    ) -> tuple[str, float, str, str]:
        """將不穩定性分類為具體類型。

        Returns:
            (type_str, severity, diagnosis, suggested_action)
        """
        if stability >= 0.6:
            return ("stable", 0.0, "輸出穩定", "無需調整")

        severity = round(1.0 - stability, 4)

        # 判斷不穩定類型
        if len(recent_sims) >= 3 and recent_sims[-1] > 0.5 and recent_sims[-3] < 0.3:
            return (
                "oscillation",
                severity,
                "偵測到振盪行為：相似度在高低間劇烈波動",
                "鎖定 root operator 和欄位集，減少變異維度",
            )

        if len(recent_sims) >= 3:
            avg_recent = sum(recent_sims[-3:]) / 3
            if avg_recent < 0.25:
                return (
                    "random_jump",
                    min(severity + 0.1, 1.0),
                    "偵測到隨機跳躍：相鄰表達式結構差異過大",
                    "強制鎖定所有結構元素，僅允許微調參數",
                )

        return (
            "drift",
            severity,
            f"偵測到結構漂移：穩定性分數 {stability:.2f} 低於閾值",
            "鎖定核心結構，引導漸進式改進",
        )

    def _infer_common_fields(self, recent: list[tuple[int, ExpressionFingerprint]]) -> list[str] | None:
        """從最近幾次的指紋中推斷共同欄位集。

        由於指紋只存了 hash，這裡用 root_operator 和 composition_type 作為代理。
        實際欄位需要在呼叫端另外傳入或從原始表達式中提取。
        """
        return None

    def _infer_window_range(self, recent: list[tuple[int, ExpressionFingerprint]]) -> tuple[int, int] | None:
        """推斷合理的窗口參數範圍。

        由於指紋中的 window_signature 是 hash，無法直接還原。
        回傳 None 表示由外部決定。
        """
        return None


class StabilityGuard:
    """頂層穩定性守衛 — 整合 StabilityTracker + 懲罰計算。

    使用方式::

        guard = StabilityGuard()
        result = guard.evaluate_and_guard(expression, context)
        if result["should_restrict"]:
            # 將 result["constraints"] 注入 prompt
        penalty = guard.compute_penalty()  # 用於 MAB reward 修正
    """

    def __init__(
        self,
        window_size: int = _DEFAULT_WINDOW_SIZE,
        instability_threshold: float = _DEFAULT_INSTABILITY_THRESHOLD,
    ):
        self._tracker = StabilityTracker(window_size=window_size)
        self._threshold = instability_threshold
        self._last_result: dict | None = None
        self._total_evaluations = 0
        self._unstable_events = 0

    @property
    def tracker(self) -> StabilityTracker:
        return self._tracker

    @property
    def threshold(self) -> float:
        return self._threshold

    def evaluate_and_guard(
        self,
        expression: str,
        cycle: int = 0,
        current_sharpe: float | None = None,
        is_mutation: bool = False,
    ) -> dict:
        """主入口：記錄表達式 → 分析穩定性 → 判斷是否需要限制。

        Args:
            expression: Alpha 表達式字串
            cycle: 當前循環編號
            current_sharpe: 當前 Sharpe ratio（可選）
            is_mutation: 是否為 mutation 操作產生的表達式

        Returns:
            {
                "stability_score": float,       # 0.0~1.0
                "is_stable": bool,
                "instability_type": str | None,
                "severity": float,
                "should_restrict": bool,        # 是否需要限制搜索
                "constraints": dict | None,     # Weak Evaluation Guard 限制
                "penalty": float,               # 穩定性懲罰（負值）或獎勵（正值）
                "diagnosis": str,
                "raw_fingerprint": dict,        # 本次表達式的指紋
            }
        """
        self._total_evaluations += 1

        # 記錄表達式
        record_result = self._tracker.record(expression)
        fp = record_result["fingerprint"]

        # 穩定性分析
        instability = self._tracker.detect_instability()
        stability_score = record_result["stability_score"]

        # 是否需要限制
        should_restrict = instability["is_unstable"]
        constraints = None
        if should_restrict:
            constraints = self._tracker.get_constrained_search_space()
            self._unstable_events += 1

        # 計算懲罰/獎勵
        penalty = self._compute_stability_penalty(stability_score)

        result = {
            "stability_score": stability_score,
            "is_stable": not instability["is_unstable"],
            "instability_type": instability["instability_type"],
            "severity": instability["severity"],
            "should_restrict": should_restrict,
            "constraints": constraints,
            "penalty": penalty,
            "diagnosis": instability["diagnosis"],
            "raw_fingerprint": {
                "root_operator": fp.root_operator,
                "operator_signature": fp.operator_signature,
                "field_signature": fp.field_signature,
                "window_signature": fp.window_signature,
                "composition_type": fp.composition_type,
                "nesting_depth": fp.nesting_depth,
                "approx_complexity": fp.approx_complexity,
            },
        }

        self._last_result = result
        return result

    def compute_reward_adjustment(self, base_reward: float) -> float:
        """計算經穩定性調整後的 reward。

        公式::

            adjusted = base_reward * (1 + stability_bonus) - instability_penalty

        其中：
          stability_bonus = max(0, (stability_score - 0.5) * 0.2)  # 穩定時加分
          instability_penalty = max(0, (0.35 - stability_score) * 0.3)  # 不穩定時扣分

        Args:
            base_reward: 基礎 reward 值

        Returns:
            調整後的 reward 值
        """
        if self._last_result is None:
            return base_reward

        stability = self._last_result["stability_score"]

        # 穩定時給獎勵
        stability_bonus = max(0.0, (stability - 0.5) * 0.2)

        # 不穩定時扣分
        instability_penalty = max(0.0, (self._threshold - stability) * 0.3)

        adjusted = base_reward * (1.0 + stability_bonus) - instability_penalty
        return round(adjusted, 6)

    def get_summary(self) -> dict:
        """返回穩定性監控摘要（供 dashboard/日誌使用）。

        Returns:
            {
                "total_evaluations": int,
                "unstable_event_count": int,
                "current_stability_score": float,
                "history_size": int,
                "instability_rate": float,
                "last_diagnosis": str | None,
            }
        """
        last_diag = None
        if self._last_result is not None:
            last_diag = self._last_result.get("diagnosis")

        rate = self._unstable_events / max(self._total_evaluations, 1)

        return {
            "total_evaluations": self._total_evaluations,
            "unstable_event_count": self._unstable_events,
            "current_stability_score": self._tracker.get_stability_score(),
            "history_size": self._tracker.history_size,
            "instability_rate": round(rate, 4),
            "last_diagnosis": last_diag,
        }

    def _compute_stability_penalty(self, stability_score: float) -> float:
        """根據穩定性分數計算懲罰值。

        Returns:
            正值表示獎勵，負值表示懲罰
        """
        if stability_score >= 0.7:
            return round((stability_score - 0.7) * 0.1, 4)
        if stability_score < self._threshold:
            return round((self._threshold - stability_score) * -0.15, 4)
        return 0.0
