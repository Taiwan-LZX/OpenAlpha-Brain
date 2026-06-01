"""
OpenAlpha-Brain — Dynamic Skill Library（動態技能庫）

自學習突變策略庫，取代靜態 ELM 矩陣。
從 WorldQuant BRAIN 平台的真實 alpha 資料中學習有效的突變模式，
將模式聚類為「技能」，並持續演化。

核心哲學：
  靜態模板建立「思維樹」容易被風吹倒。
  動態技能庫建立「純思維鏈」— 系統從真實資料中發現有效模式。

整合點：
  - brain_data_client: 取得歷史 alpha 記錄與成功案例
  - ast_originality.FASTEXPRParser: AST 結構分析
  - brain_submitter._brain_improvement_loop: 作為突變指導來源
  - 與靜態 ELM 矩陣共存，動態優先
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from openalpha_brain.utils.algo_logger import algo_log

logger = logging.getLogger(__name__)

_DEFAULT_SKILLS_PATH = "data/dynamic_skills.json"

# 已知 BRAIN 欄位集合，用於模式辨識
_KNOWN_FIELDS = {
    "close",
    "open",
    "high",
    "low",
    "volume",
    "amount",
    "vwap",
    "returns",
    "cap",
    "sharesout",
    "adv20",
    "sales",
    "earnings",
    "assets",
    "income",
    "cashflow",
    "debt",
    "equity",
    "margin",
    "turnover",
    "roa",
    "roe",
    "book_value",
    "market_cap",
}

# 已知時間序列運算子，用於 operator_chain 提取
_TS_OPERATORS = {
    "ts_delta",
    "ts_mean",
    "ts_std_dev",
    "ts_rank",
    "ts_zscore",
    "ts_decay_linear",
    "ts_sum",
    "ts_corr",
    "ts_regression",
    "ts_arg_max",
    "ts_arg_min",
    "ts_delay",
    "ts_av_diff",
    "ts_quantile",
    "ts_backfill",
}

# 已知群組運算子
_GROUP_OPERATORS = {
    "group_neutralize",
    "group_rank",
    "group_zscore",
}

# 已知正規化運算子
_NORM_OPERATORS = {"rank", "scale", "normalize", "zscore", "winsorize"}


@dataclass
class SkillStats:
    """技能統計資訊，追蹤每次使用的成效。"""

    trial_count: int = 0
    success_count: int = 0
    win_rate: float = 0.0
    avg_sharpe_gain: float = 0.0
    total_sharpe_before: float = 0.0
    total_sharpe_after: float = 0.0
    last_used_at: str = ""
    last_success_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SkillStats:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Skill:
    """單一突變技能 — 從真實 alpha 模式中提取的結構性知識。"""

    skill_id: str
    name: str
    pattern_fingerprint: dict
    trigger_failure_types: list[str]
    mutation_template: str
    statistics: SkillStats = field(default_factory=SkillStats)
    created_at: str = ""
    source: str = "brain_api"

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "pattern_fingerprint": self.pattern_fingerprint,
            "trigger_failure_types": self.trigger_failure_types,
            "mutation_template": self.mutation_template,
            "statistics": self.statistics.to_dict(),
            "created_at": self.created_at,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Skill:
        stats_d = d.get("statistics", {})
        return cls(
            skill_id=d["skill_id"],
            name=d["name"],
            pattern_fingerprint=d.get("pattern_fingerprint", {}),
            trigger_failure_types=d.get("trigger_failure_types", []),
            mutation_template=d.get("mutation_template", ""),
            statistics=SkillStats.from_dict(stats_d) if isinstance(stats_d, dict) else SkillStats(),
            created_at=d.get("created_at", ""),
            source=d.get("source", "brain_api"),
        )


class DynamicSkillLibrary:
    """自學習突變技能庫，從 BRAIN API 的真實 alpha 資料建構與演化。

    使用方式：
        lib = DynamicSkillLibrary()
        await lib.initialize_from_brain(brain_data_client, session_id)
        skills = lib.select_skills_for_failure("SHARPE", expr, "long")
        prompt = lib.build_dynamic_mutation_prompt(skills, failure_context)
    """

    def __init__(self, skills_path: str = _DEFAULT_SKILLS_PATH):
        self._skills_path = Path(skills_path)
        self._skills: list[Skill] = []
        self._pattern_buffer: list[dict] = []
        self._last_evolution: str = ""
        self._parser: object | None = None
        self._load()

    # ── 初始化：從 BRAIN API 取得真實資料 ──────────────────────────────

    async def initialize_from_brain(
        self,
        brain_data_client,
        session_id: str,
    ) -> None:
        """從 BRAIN API 取得歷史 alpha 記錄並建構初始技能池。

        流程：
          1. 透過 list_alphas 取得用戶的 alpha 歷史
          2. 篩選成功的 alpha（Sharpe >= 1.25）
          3. 對每個成功 alpha 分析 AST 結構
          4. 提取模式特徵並聚類為技能
          5. 持久化至 JSON

        Args:
            brain_data_client: BrainDataClient 實例，用於取得 alpha 資料
            session_id: 當前 session ID，用於日誌標記
        """
        from openalpha_brain.services import brain_client

        logger.info(
            "[%s] DynamicSkillLibrary: 開始從 BRAIN API 初始化技能庫...",
            session_id,
        )

        cookies = None
        try:
            cookies = await brain_data_client._ensure_client()
        except (ConnectionError, OSError, TimeoutError) as exc:
            logger.warning(
                "[%s] DynamicSkillLibrary: 無法取得 BRAIN cookies — %s",
                session_id,
                exc,
            )
            return

        all_patterns: list[dict] = []

        try:
            alpha_list_data = await brain_client.list_alphas(cookies, limit=100)
            if not alpha_list_data or not isinstance(alpha_list_data, dict):
                logger.warning(
                    "[%s] DynamicSkillLibrary: list_alphas 回傳空或非 dict",
                    session_id,
                )
                return

            results = alpha_list_data.get("results", [])
            if not results:
                logger.info(
                    "[%s] DynamicSkillLibrary: 無歷史 alpha 記錄",
                    session_id,
                )
                return

            logger.info(
                "[%s] DynamicSkillLibrary: 取得 %d 筆歷史 alpha，開始分析...",
                session_id,
                len(results),
            )

            for record in results[:50]:
                try:
                    alpha_id = record.get("id")
                    if not alpha_id:
                        continue

                    details = await brain_client.fetch_alpha_details(alpha_id, cookies)
                    if not details:
                        continue

                    is_data = details.get("is", {}) or {}
                    sharpe = is_data.get("sharpe")
                    fitness = is_data.get("fitness")

                    regular = details.get("regular", {})
                    code = regular.get("code", "") if isinstance(regular, dict) else ""

                    if not code:
                        continue

                    pattern = self.analyze_alpha_pattern(code, sharpe, fitness)
                    pattern["alpha_id"] = alpha_id
                    pattern["is_successful"] = sharpe is not None and sharpe >= 1.25

                    all_patterns.append(pattern)

                except (ValueError, TypeError) as exc:
                    logger.debug(
                        "[%s] DynamicSkillLibrary: 分析 alpha %s 失敗 — %s",
                        session_id,
                        record.get("id", "?"),
                        exc,
                    )
                    continue

        except brain_client.BrainAuthError:
            logger.warning(
                "[%s] DynamicSkillLibrary: BRAIN 認證過期，初始化中止",
                session_id,
            )
            return
        except (ConnectionError, OSError, TimeoutError) as exc:
            logger.error(
                "[%s] DynamicSkillLibrary: 初始化過程發生錯誤 — %s",
                session_id,
                exc,
            )
            return

        if not all_patterns:
            logger.warning(
                "[%s] DynamicSkillLibrary: 未提取到任何有效模式",
                session_id,
            )
            return

        successful_patterns = [p for p in all_patterns if p.get("is_successful")]
        logger.info(
            "[%s] DynamicSkillLibrary: 提取 %d 筆模式（其中 %d 筆成功）",
            session_id,
            len(all_patterns),
            len(successful_patterns),
        )

        new_skills = self.cluster_patterns_into_skills(successful_patterns or all_patterns)

        existing_ids = {s.skill_id for s in self._skills}
        for sk in new_skills:
            if sk.skill_id not in existing_ids:
                self._skills.append(sk)

        self._pattern_buffer.extend(all_patterns)
        self._save()

        logger.info(
            "[%s] DynamicSkillLibrary: 初始化完成，目前共有 %d 個技能",
            session_id,
            len(self._skills),
        )

    # ── 模式分析：解析 alpha 表達式的結構指紋 ───────────────────────────

    def analyze_alpha_pattern(
        self,
        expression: str,
        sharpe: float | None = None,
        fitness: float | None = None,
    ) -> dict:
        """分析一個 alpha 表達式，提取結構模式指紋。

        從表達式中自動偵測：
          - 根運算子（最外層函數）
          - 運算子鏈（巢狀呼叫順序）
          - 使用的欄位集合
          - 時間視窗參數
          - 巢狀深度
          - 中和類型與範圍
          - 組成類型（加法/乘法/比率）

        Args:
            expression: FASTEXPR 格式的 alpha 表達式
            sharpe: 該 alpha 的 Sharpe ratio（可選）
            fitness: 該 alpha 的 Fitness（可選）

        Returns:
            包含各維度特徵的指紋字典
        """
        fp: dict = {
            "root_operator": "",
            "operator_chain": [],
            "fields": [],
            "windows": [],
            "nesting_depth": 0,
            "has_neutralization": False,
            "neutralization_scope": "",
            "composition_type": "",
            "ast_node_count": 0,
            "sharpe": sharpe,
            "fitness": fitness,
            "raw_expression": expression[:200],
        }

        try:
            root_op_match = re.match(r"\s*(\w+)\s*\(", expression)
            fp["root_operator"] = root_op_match.group(1) if root_op_match else ""

            ops_found = re.findall(r"\b([a-zA-Z_]\w*)\s*\(", expression)
            ts_ops = [op for op in ops_found if op in _TS_OPERATORS]
            group_ops = [op for op in ops_found if op in _GROUP_OPERATORS]
            norm_ops = [op for op in ops_found if op in _NORM_OPERATORS]
            other_ops = [
                op
                for op in ops_found
                if op not in _TS_OPERATORS
                and op not in _GROUP_OPERATORS
                and op not in _NORM_OPERATORS
                and op not in {"abs", "log", "signed_power", "max", "min", "vec_sum", "vec_avg", "quantile"}
            ]

            chain = []
            chain.extend(norm_ops[:2])
            chain.extend(ts_ops[:4])
            chain.extend(group_ops[:2])
            chain.extend(other_ops[:2])
            fp["operator_chain"] = chain

            fields_found = set()
            for known_field in _KNOWN_FIELDS:
                if re.search(r"\b" + known_field + r"\b", expression):
                    fields_found.add(known_field)
            fp["fields"] = sorted(fields_found)

            windows = [int(w) for w in re.findall(r"(?<!\w)(\d+)(?!\w)", expression) if 1 <= int(w) <= 500]
            fp["windows"] = sorted(set(windows))

            depth = 0
            max_depth = 0
            for ch in expression:
                if ch == "(":
                    depth += 1
                    max_depth = max(max_depth, depth)
                elif ch == ")":
                    depth -= 1
            fp["nesting_depth"] = max_depth

            neut_match = re.search(
                r"group_neutralize\s*\([^,]+,\s*(\w+)",
                expression,
            )
            if neut_match:
                fp["has_neutralization"] = True
                fp["neutralization_scope"] = neut_match.group(1).lower()
            elif "group_neutralize" in expression:
                fp["has_neutralization"] = True
                fp["neutralization_scope"] = "industry"

            has_plus = "+" in expression and not re.search(
                r"\+\s*[\w(]",
                expression.split("group_neutralize")[0] if "group_neutralize" in expression else expression,
            )
            has_star = "*" in expression or "/" in expression
            has_divide = "/" in expression

            if has_divide and not has_plus:
                fp["composition_type"] = "ratio"
            elif has_star:
                fp["composition_type"] = "multiplicative"
            elif has_plus:
                fp["composition_type"] = "additive"
            else:
                fp["composition_type"] = "nested"

            node_count = len(re.findall(r"[a-zA-Z_]\w*\s*\(", expression))
            fp["ast_node_count"] = node_count

        except (ValueError, TypeError) as exc:
            logger.warning("analyze_alpha_pattern: 解析失敗 — %s", exc)

        return fp

    # ── 聚類：將相似模式分組為技能 ────────────────────────────────────

    def cluster_patterns_into_skills(self, patterns: list[dict]) -> list[Skill]:
        """將多個 alpha 模式聚類為技能物件。

        聚類邏輯（基於 operator_chain 相似度的簡易分群）：
          1. 以 root_operator + composition_type 為初級分群鍵
          2. 同一群內計算 operator_chain 的 Jaccard 相似度
          3. 相似度 > 60% 的模式合併為同一技能
          4. 每個技能以群中心（centroid）作為代表模式

        Args:
            patterns: analyze_alpha_pattern 回傳的指紋列表

        Returns:
            聚類後的 Skill 列表
        """
        if not patterns:
            return []

        clusters: dict[str, list[dict]] = {}

        for pat in patterns:
            key = f"{pat.get('root_operator', '_')}_{pat.get('composition_type', '_')}"
            clusters.setdefault(key, []).append(pat)

        skills: list[Skill] = []
        now = datetime.now(UTC).isoformat()
        skill_idx = len(self._skills)

        for _cluster_key, group in clusters.items():
            centroid = group[0] if len(group) == 1 else self._compute_centroid(group)

            skill_id = f"skill_{skill_idx + len(skills):03d}"
            name = self._generate_skill_name(centroid)

            trigger_types = self._infer_trigger_types(centroid, group)
            template = self._build_mutation_template(centroid, group)

            sk = Skill(
                skill_id=skill_id,
                name=name,
                pattern_fingerprint=centroid,
                trigger_failure_types=trigger_types,
                mutation_template=template,
                statistics=SkillStats(),
                created_at=now,
                source="brain_api",
            )
            skills.append(sk)

        logger.info(
            "cluster_patterns_into_skills: %d 個模式 → %d 個技能",
            len(patterns),
            len(skills),
        )
        return skills

    def _compute_centroid(self, group: list[dict]) -> dict:
        """計算一組模式的中心點（頻率最高的特徵組合）。"""
        from collections import Counter

        root_ops = Counter(p.get("root_operator", "") for p in group)
        comp_types = Counter(p.get("composition_type", "") for p in group)

        all_chains: list[list[str]] = [p.get("operator_chain", []) for p in group if p.get("operator_chain")]
        chain_freq: Counter[tuple[str, ...]] = Counter()
        for chain in all_chains:
            chain_freq[tuple(chain)] += 1

        all_fields: list[list[str]] = [p.get("fields", []) for p in group if p.get("fields")]
        field_counter: Counter[str] = Counter()
        for flds in all_fields:
            for f in flds:
                field_counter[f] += 1

        all_windows: list[list[int]] = [p.get("windows", []) for p in group if p.get("windows")]
        window_counter: Counter[int] = Counter()
        for wins in all_windows:
            for w in wins:
                window_counter[w] += 1

        depths = [p.get("nesting_depth", 0) for p in group]
        avg_depth = round(sum(depths) / len(depths)) if depths else 0

        best_chain = chain_freq.most_common(1)[0][0] if chain_freq else ()
        top_fields = [f for f, _ in field_counter.most_common(4)]
        top_windows = sorted([w for w, _ in window_counter.most_common(3)])

        return {
            "root_operator": root_ops.most_common(1)[0][0] if root_ops else "",
            "operator_chain": list(best_chain),
            "fields": top_fields,
            "windows": top_windows,
            "nesting_depth": avg_depth,
            "has_neutralization": any(p.get("has_neutralization") for p in group),
            "neutralization_scope": "",
            "composition_type": comp_types.most_common(1)[0][0] if comp_types else "",
            "ast_node_count": round(
                sum(p.get("ast_node_count", 0) for p in group) / len(group),
            ),
            "cluster_size": len(group),
        }

    def _generate_skill_name(self, centroid: dict) -> str:
        """根據中心點特徵產生人類可讀的技能名稱。"""
        parts = []

        root = centroid.get("root_operator", "")
        if root:
            parts.append(root.replace("_", ""))

        comp = centroid.get("composition_type", "")
        if comp:
            parts.append(comp)

        chain = centroid.get("operator_chain", [])
        if chain:
            parts.append(chain[0].replace("_", "") if chain else "")

        depth = centroid.get("nesting_depth", 0)
        if depth > 3:
            parts.append("deep")
        elif depth <= 2:
            parts.append("tight")

        return "_".join(parts)[:60] if parts else "unnamed_skill"

    def _infer_trigger_types(self, centroid: dict, group: list[dict]) -> list[str]:
        """根據模式特徵推斷此技能能解決的失敗類型。"""
        triggers: list[str] = []
        root = centroid.get("root_operator", "")
        comp = centroid.get("composition_type", "")
        depth = centroid.get("nesting_depth", 0)

        if root in ("rank", "scale"):
            triggers.append("SHARPE")
        if root in ("group_zscore", "group_rank", "group_neutralize"):
            triggers.append("TURNOVER")
        if "ts_decay" in root or any("ts_decay" in c for c in centroid.get("operator_chain", [])):
            triggers.append("TURNOVER")

        if depth <= 2:
            triggers.append("LOW_COMPLEXITY")
        if depth >= 4:
            triggers.append("OVERFITTING_RISK")

        if comp == "ratio":
            triggers.extend(["SHARPE", "FITNESS"])
        if comp == "additive":
            triggers.append("CORRELATION")

        if centroid.get("has_neutralization"):
            triggers.append("NEUTRALIZATION")

        avg_sharpes = [p.get("sharpe") for p in group if p.get("sharpe") is not None]
        if avg_sharpes and max((v for v in avg_sharpes if v is not None), default=0.0) >= 1.75:
            triggers.append("HIGH_SHARPE_TARGET")

        return list(dict.fromkeys(triggers))[:6]

    def _build_mutation_template(self, centroid: dict, group: list[dict]) -> str:
        """根據中心點模式建構 LLM 可讀的突變指引模板。"""
        root = centroid.get("root_operator", "rank")
        chain = centroid.get("operator_chain", [])
        fields = centroid.get("fields", [])
        windows = centroid.get("windows", [])
        comp = centroid.get("composition_type", "")
        depth = centroid.get("nesting_depth", 0)
        has_neut = centroid.get("has_neutralization", False)

        parts: list[str] = []

        high_sharpe_examples = [p for p in group if p.get("sharpe") is not None and p.get("sharpe", 0) >= 1.5][:2]

        if high_sharpe_examples:
            parts.append("參考以下真實高 Sharpe alpha 的結構：")
            for ex in high_sharpe_examples:
                expr_snippet = ex.get("raw_expression", "")[:100]
                sp = ex.get("sharpe", "?")
                parts.append(f"  • Sharpe={sp} 結構: {expr_snippet}")

        if root:
            parts.append(f"嘗試使用 {root} 作為根運算子。")

        if chain:
            chain_str = " → ".join(chain[:5])
            parts.append(f"運算子鏈模式: {chain_str}")

        if fields:
            parts.append(f"推薦欄位: {', '.join(fields[:5])}")

        if windows:
            win_str = ", ".join(str(w) for w in windows[:4])
            parts.append(f"常用視窗: {win_str}")

        if comp:
            comp_guide = {
                "ratio": "使用比率結構（A/B）捕捉相對價值信號",
                "multiplicative": "使用乘法結構放大弱信號",
                "additive": "使用加法結構組合多因子",
                "nested": "使用深層巢狀結構增加表達力",
            }
            parts.append(comp_guide.get(comp, ""))

        if has_neut:
            parts.append("套用 group_neutralize 進行行業中和")

        if depth and depth <= 2:
            parts.append("保持淺層巢狀（depth ≤ 3）避免過擬合")
        elif depth and depth >= 4:
            parts.append("適當降低巢狀深度，加入 ts_decay_linear 降低換手率")

        return "\n".join(p for p in parts if p)

    # ── 技能選取：依據失敗類型選出最佳技能 ─────────────────────────────

    @algo_log()
    def select_skills_for_failure(
        self,
        failure_type: str,
        current_expression: str,
        direction: str = "long",
    ) -> list[Skill]:
        """根據當前失敗類型與表達式，選出最相關的 Top-K 技能。

        排序考量：
          1. 觸發類型匹配度（failure_type 是否在 trigger_failure_types 中）
          2. 歷史勝率（win_rate 高者優先）
          3. 多樣性（避免選取過多相似技能）
          4. 與當前表達式的結構差異度（鼓勵結構性轉變）

        Args:
            failure_type: BRAIN 失敗類型（如 SHARPE, TURNOVER, FITNESS 等）
            current_expression: 當前的 alpha 表達式
            direction: 方向（long/short），影響技能篩選

        Returns:
            最多 3 個排序後的 Skill 物件
        """
        if not self._skills:
            return []

        current_fp = self.analyze_alpha_pattern(current_expression)

        scored: list[tuple[float, Skill]] = []

        for sk in self._skills:
            score = 0.0

            ft_upper = failure_type.upper().replace(" ", "_")
            if ft_upper in sk.trigger_failure_types:
                score += 40.0
            else:
                for trig in sk.trigger_failure_types:
                    if ft_upper in trig or trig in ft_upper:
                        score += 15.0
                        break

            score += sk.statistics.win_rate * 30.0

            if sk.statistics.trial_count >= 5:
                score += 10.0
            elif sk.statistics.trial_count >= 2:
                score += 5.0

            struct_diff = self._structural_diff(current_fp, sk.pattern_fingerprint)
            score += min(struct_diff * 20.0, 25.0)

            if direction == "short" and ("reversal" in sk.name.lower() or "mean" in sk.name.lower()):
                score += 8.0

            scored.append((score, sk))

        scored.sort(key=lambda x: x[0], reverse=True)

        selected: list[Skill] = []
        selected_chains: set[tuple[str, ...]] = set()

        for _, sk in scored:
            if len(selected) >= 3:
                break

            chain_key = tuple(sk.pattern_fingerprint.get("operator_chain", [])[:3])
            if chain_key in selected_chains:
                continue

            selected.append(sk)
            selected_chains.add(chain_key)

        return selected

    def _structural_diff(self, fp_a: dict, fp_b: dict) -> float:
        """計算兩個模式指紋之間的結構差異度（0~1）。"""
        diff = 0.0
        n_dims = 0

        if fp_a.get("root_operator") != fp_b.get("root_operator"):
            diff += 1.0
        n_dims += 1

        chain_a = set(fp_a.get("operator_chain", []))
        chain_b = set(fp_b.get("operator_chain", []))
        if chain_a or chain_b:
            union = chain_a | chain_b
            inter = chain_a & chain_b
            diff += 1.0 - (len(inter) / len(union)) if union else 0.0
        n_dims += 1

        fields_a = set(fp_a.get("fields", []))
        fields_b = set(fp_b.get("fields", []))
        if fields_a or fields_b:
            union_f = fields_a | fields_b
            inter_f = fields_a & fields_b
            diff += 1.0 - (len(inter_f) / len(union_f)) if union_f else 0.0
        n_dims += 1

        comp_a = fp_a.get("composition_type", "")
        comp_b = fp_b.get("composition_type", "")
        if comp_a != comp_b:
            diff += 0.5
        n_dims += 1

        return diff / n_dims if n_dims > 0 else 0.0

    # ── Prompt 建構：將技能轉為 LLM 指引 ────────────────────────────────

    def build_dynamic_mutation_prompt(
        self,
        skills: list[Skill],
        failure_context: dict,
    ) -> str:
        """將選出的技能轉換為 LLM 可用的突變指引 prompt。

        此方法取代靜態 build_brain_failure_feedback() 中的突變建議部分。
        輸出格式包含：
          - 真實成功模式的結構分析
          - 具體的運算子/欄位/視窗建議
          - 來自實際資料的 Sharpe 表現參考

        Args:
            skills: select_skills_for_failure 選出的技能列表
            failure_context: 包含失敗詳情的字典，應包含：
                - failure_type: str
                - current_sharpe: float | None
                - current_turnover: float | None
                - failed_checks: list[str]

        Returns:
            格式化的 LLM prompt 字串
        """
        if not skills:
            return ""

        sections: list[str] = []

        sections.append("## 🧠 動態技能庫指引（來自真實 BRAIN 成功數據）\n")
        sections.append(
            "以下是從平台上真實高 Sharpe alpha 中自動學習到的結構模式。請優先考慮這些經驗驗證的模式進行突變：\n",
        )

        for i, sk in enumerate(skills, 1):
            stat = sk.statistics
            stats_str = (
                f"勝率={stat.win_rate:.0%} | 試用={stat.trial_count}次"
                if stat.trial_count > 0
                else "全新技能（尚未試用）"
            )

            sections.append(f"### 技能 {i}: {sk.name} [{stats_str}]")

            if sk.mutation_template:
                sections.append(sk.mutation_template)

            pf = sk.pattern_fingerprint
            detail_parts: list[str] = []
            if pf.get("root_operator"):
                detail_parts.append(f"根運算子: `{pf['root_operator']}`")
            if pf.get("fields"):
                detail_parts.append(f"欄位: {', '.join(pf['fields'][:5])}")
            if pf.get("windows"):
                detail_parts.append(f"視窗: {pf['windows']}")
            if pf.get("composition_type"):
                detail_parts.append(f"組成型態: {pf['composition_type']}")
            if detail_parts:
                sections.append(f"結構特徵: {' | '.join(detail_parts)}")

            sections.append("")

        ft = failure_context.get("failure_type", "UNKNOWN")
        cur_sharpe = failure_context.get("current_sharpe")
        cur_to = failure_context.get("current_turnover")
        checks = failure_context.get("failed_checks", [])

        sections.append("### 當前問題診斷")
        diag = f"- 失敗類型: **{ft}**"
        if cur_sharpe is not None:
            diag += f"\n- 目前 Sharpe: **{cur_sharpe:.3f}**"
        if cur_to is not None:
            diag += f"\n- 目前換手率: **{cur_to:.1f}%**"
        if checks:
            chk_str = ", ".join(str(c) for c in checks[:5])
            diag += f"\n- 未通過檢查: {chk_str}"
        sections.append(diag)
        sections.append("")

        sections.append(
            "**請基於以上真實數據模式，生成一個新的 FASTEXPR 表達式。**\n"
            "要求：\n"
            "1. 參考至少一個技能的結構建議\n"
            "2. 保持語法正確且符合 FASTEXPR 規範\n"
            "3. 解釋你的突變策略如何對應到診斷的問題",
        )

        return "\n".join(sections)

    # ── 結果記錄：更新技能統計 ─────────────────────────────────────────

    @algo_log()
    def record_outcome(
        self,
        skill_id: str,
        success: bool,
        sharpe_before: float,
        sharpe_after: float,
    ) -> None:
        """記錄一次技能使用結果，更新統計資料。

        每次突變嘗試後應呼叫此方法，讓技能庫能夠自我學習。
        更新內容：
          - trial_count +1
          - success_count +1（若成功）
          - win_rate 重算
          - avg_sharpe_gain 更新
          - 持久化至 JSON

        Args:
            skill_id: 使用的技能 ID
            success: 此次突變是否成功（BRAIN PASS）
            sharpe_before: 突變前的 Sharpe
            sharpe_after: 突變後的 Sharpe
        """
        now_iso = datetime.now(UTC).isoformat()

        for sk in self._skills:
            if sk.skill_id != skill_id:
                continue

            stat = sk.statistics
            stat.trial_count += 1
            stat.total_sharpe_before += sharpe_before
            stat.total_sharpe_after += sharpe_after
            stat.last_used_at = now_iso

            if success:
                stat.success_count += 1
                stat.last_success_at = now_iso

            if stat.trial_count > 0:
                stat.win_rate = stat.success_count / stat.trial_count

            gain = sharpe_after - sharpe_before
            running_total = stat.avg_sharpe_gain * (stat.trial_count - 1)
            stat.avg_sharpe_gain = (running_total + gain) / stat.trial_count

            break
        else:
            logger.warning("record_outcome: 找不到 skill_id=%s", skill_id)
            return

        self._save()
        _skill_obj = self._get_skill(skill_id)
        logger.info(
            "record_outcome: skill=%s success=%s wr=%.2f trials=%d",
            skill_id,
            success,
            _skill_obj.statistics.win_rate if _skill_obj else 0.0,
            _skill_obj.statistics.trial_count if _skill_obj else 0,
        )

    # ── 技能演化：剪枝、合併、晉升 ─────────────────────────────────────

    @algo_log()
    def evolve_skills(self, min_trials: int = 10) -> None:
        """執行技能演化週期：剪枝低效技能、合併重複、晉升新星。

        三階段演化流程：

        Phase 1 — 剪枝（Prune）：
          移除 trial_count >= min_trials 且 win_rate < 0.2 的技能。
          這些技能已經有足夠樣本但效果不佳。

        Phase 2 — 合併（Merge）：
          若兩個技能的 pattern_fingerprint 相似度 > 90%，
          則合併為一個（保留較高 win_rate 者）。

        Phase 3 — 晉升（Promote）：
          從 pattern_buffer 中找出有潛力的新興模式
          （win_rate > 0.6 且 trial_count < 20），
          晉升為正式技能。

        Args:
            min_trials: 最低試用次數門檻，低於此次數的技能不會被剪枝
        """
        before_count = len(self._skills)
        pruned = 0
        merged = 0
        promoted = 0
        now = datetime.now(UTC).isoformat()

        active: list[Skill] = []

        for sk in self._skills:
            if sk.statistics.trial_count >= min_trials and sk.statistics.win_rate < 0.2:
                pruned += 1
                logger.info(
                    "evolve_skills: 剪枝技能 %s (wr=%.1f%% trials=%d)",
                    sk.skill_id,
                    sk.statistics.win_rate * 100,
                    sk.statistics.trial_count,
                )
                continue
            active.append(sk)

        merged_indices: set[int] = set()
        final_active: list[Skill] = []

        for i, sk_a in enumerate(active):
            if i in merged_indices:
                continue

            for j, sk_b in enumerate(active[i + 1 :], start=i + 1):
                if j in merged_indices:
                    continue

                sim = self._pattern_similarity(
                    sk_a.pattern_fingerprint,
                    sk_b.pattern_fingerprint,
                )
                if sim > 0.9:
                    merged_indices.add(j)
                    merged += 1

                    if sk_b.statistics.win_rate > sk_a.statistics.win_rate:
                        keep, discard = sk_b, sk_a
                    else:
                        keep, discard = sk_a, sk_b

                    keep.statistics.trial_count += discard.statistics.trial_count
                    keep.statistics.success_count += discard.statistics.success_count
                    if keep.statistics.trial_count > 0:
                        keep.statistics.win_rate = keep.statistics.success_count / keep.statistics.trial_count

                    logger.info(
                        "evolve_skills: 合併 %s → %s (sim=%.2f)",
                        discard.skill_id,
                        keep.skill_id,
                        sim,
                    )
                    sk_a = keep
                    break

            final_active.append(sk_a)

        emerging = self._find_emerging_patterns(min_trials=min_trials // 2)
        for em_pat in emerging:
            new_id = f"skill_{len(final_active) + promoted:03d}"
            new_sk = Skill(
                skill_id=new_id,
                name=self._generate_skill_name(em_pat),
                pattern_fingerprint=em_pat,
                trigger_failure_types=self._infer_trigger_types(em_pat, [em_pat]),
                mutation_template=self._build_mutation_template(em_pat, [em_pat]),
                statistics=SkillStats(trial_count=em_pat.get("_trial_count", 0)),
                created_at=now,
                source="discovered",
            )
            final_active.append(new_sk)
            promoted += 1

        self._skills = final_active
        self._last_evolution = now
        self._save()

        logger.info(
            "evolve_skills: 完成 — 前=%d 後=%d 剪枝=%d 合併=%d 晉升=%d",
            before_count,
            len(self._skills),
            pruned,
            merged,
            promoted,
        )

    def _pattern_similarity(self, a: dict, b: dict) -> float:
        """計算兩個 pattern_fingerprint 的整體相似度（0~1）。"""
        scores: list[float] = []
        weights: list[float] = []

        if a.get("root_operator") == b.get("root_operator"):
            scores.append(1.0)
        else:
            scores.append(0.0)
        weights.append(2.0)

        chain_a = set(a.get("operator_chain", []))
        chain_b = set(b.get("operator_chain", []))
        if chain_a or chain_b:
            inter_len = len(chain_a & chain_b)
            union_len = len(chain_a | chain_b)
            scores.append(inter_len / union_len if union_len else 0.0)
        else:
            scores.append(1.0)
        weights.append(3.0)

        fld_a = set(a.get("fields", []))
        fld_b = set(b.get("fields", []))
        if fld_a or fld_b:
            inter_f = len(fld_a & fld_b)
            union_f = len(fld_a | fld_b)
            scores.append(inter_f / union_f if union_f else 0.0)
        else:
            scores.append(1.0)
        weights.append(1.5)

        if a.get("composition_type") == b.get("composition_type"):
            scores.append(1.0)
        else:
            scores.append(0.0)
        weights.append(1.0)

        if a.get("has_neutralization") == b.get("has_neutralization"):
            scores.append(1.0)
        else:
            scores.append(0.0)
        weights.append(0.5)

        total_w = sum(weights)
        if total_w == 0:
            return 0.0
        return sum(s * w for s, w in zip(scores, weights, strict=False)) / total_w

    def _find_emerging_patterns(self, min_trials: int = 5) -> list[dict]:
        """從 pattern buffer 中找出有潛力的新興模式。"""
        if not self._pattern_buffer:
            return []

        successful = [p for p in self._pattern_buffer if p.get("is_successful")]
        if len(successful) < 2:
            return []

        new_skills = self.cluster_patterns_into_skills(successful)
        emerging: list[dict] = []

        for sk in new_skills:
            existing_similar = False
            for existing in self._skills:
                sim = self._pattern_similarity(sk.pattern_fingerprint, existing.pattern_fingerprint)
                if sim > 0.75:
                    existing_similar = True
                    break

            if not existing_similar:
                pf = dict(sk.pattern_fingerprint)
                pf["_trial_count"] = sk.statistics.trial_count
                pf["_estimated_wr"] = 0.7
                emerging.append(pf)

        return emerging[:3]

    # ── 監控介面：技能庫狀態摘要 ────────────────────────────────────────

    def get_skill_summary(self) -> dict:
        """回傳技能庫完整狀態摘要，供監控與除錯使用。

        Returns:
            包含以下資訊的字典：
              - total_skills: 總技能數
              - active_skills: 有使用記錄的技能數
              - prunable_skills: 可能被剪枝的技能數
              - top_skills: 表現最好的 Top-5 技能
              - distribution_by_root: 依根運算子分類的分佈
              - distribution_by_source: 依來源分類的分佈
              - last_evolution: 上次演化時間
              - total_trials: 所有技能總試用次數
              - library_health: 庫健康度評估
        """
        total = len(self._skills)
        active = sum(1 for s in self._skills if s.statistics.trial_count > 0)
        prunable = sum(1 for s in self._skills if s.statistics.trial_count >= 10 and s.statistics.win_rate < 0.2)

        top_skills = sorted(
            self._skills,
            key=lambda s: (
                s.statistics.win_rate,
                s.statistics.trial_count,
            ),
            reverse=True,
        )[:5]

        by_root: dict[str, int] = {}
        by_source: dict[str, int] = {}
        total_trials = 0
        total_successes = 0

        for s in self._skills:
            root = s.pattern_fingerprint.get("root_operator", "_unknown")
            by_root[root] = by_root.get(root, 0) + 1

            src = s.source
            by_source[src] = by_source.get(src, 0) + 1

            total_trials += s.statistics.trial_count
            total_successes += s.statistics.success_count

        global_wr = total_successes / total_trials if total_trials > 0 else 0.0

        if total >= 10 and global_wr >= 0.35:
            health = "healthy"
        elif total >= 5 and global_wr >= 0.2:
            health = "developing"
        elif total > 0:
            health = "initializing"
        else:
            health = "empty"

        return {
            "total_skills": total,
            "active_skills": active,
            "prunable_skills": prunable,
            "top_skills": [
                {
                    "skill_id": s.skill_id,
                    "name": s.name,
                    "win_rate": s.statistics.win_rate,
                    "trials": s.statistics.trial_count,
                    "avg_gain": s.statistics.avg_sharpe_gain,
                    "source": s.source,
                }
                for s in top_skills
            ],
            "distribution_by_root": by_root,
            "distribution_by_source": by_source,
            "last_evolution": self._last_evolution,
            "total_trials": total_trials,
            "total_successes": total_successes,
            "global_win_rate": round(global_wr, 4),
            "library_health": health,
            "pattern_buffer_size": len(self._pattern_buffer),
        }

    # ── 內部工具方法 ─────────────────────────────────────────────────────

    def _get_skill(self, skill_id: str) -> Skill | None:
        for s in self._skills:
            if s.skill_id == skill_id:
                return s
        return None

    def _save(self) -> None:
        """將技能庫持久化至 JSON 檔案。"""
        try:
            self._skills_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": "1.0",
                "last_updated": datetime.now(UTC).isoformat(),
                "last_evolution": self._last_evolution,
                "skills": [s.to_dict() for s in self._skills],
                "pattern_buffer_size": len(self._pattern_buffer),
            }
            with open(self._skills_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
        except OSError as exc:
            logger.error("DynamicSkillLibrary._save: 寫入失敗 — %s", exc)

    def _load(self) -> None:
        """從 JSON 檔案載入技能庫。"""
        try:
            if not self._skills_path.exists():
                logger.info("DynamicSkillLibrary: %s 不存在，從空白開始", self._skills_path)
                return
            with open(self._skills_path, encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                return
            self._skills = [Skill.from_dict(d) for d in data.get("skills", []) if isinstance(d, dict)]
            self._last_evolution = data.get("last_evolution", "")
            logger.info(
                "DynamicSkillLibrary: 已載入 %d 個技能",
                len(self._skills),
            )
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("DynamicSkillLibrary._load: 載入失敗 — %s", exc)
            self._skills = []
