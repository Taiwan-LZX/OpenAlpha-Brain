from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from openalpha_brain.services import brain_client
from openalpha_brain.services import llm_client
from openalpha_brain.utils import extract_json_from_llm as _extract_json_from_llm
from openalpha_brain.cli import session_manager as sm
from openalpha_brain.validation import validator as val
from openalpha_brain.validation.validator import compute_hierarchical_reward, get_reward_level
from openalpha_brain.validation.ast_repair import repair_expression
from openalpha_brain.learning.param_optimizer import expression_hash
from openalpha_brain.config.config import settings
from openalpha_brain.core.models import (
    AlphaFingerprint, AlphaMetrics, AlphaResult,
    BrainSimStatus, BrainSubmissionResult,
    PipelineStatus, SessionStatus,
)
from openalpha_brain.generation.prompts import (
    get_system_prompt,
    build_brain_failure_feedback,
    llm_diagnose_failure,
)
from openalpha_brain.validation.overfit_detector import detect_overfit
from openalpha_brain.utils.volatility_detector import estimate_garch11
from openalpha_brain.utils.algo_logger import algo_log, Timer, log_call
try:
    from openalpha_brain.knowledge.dynamic_skill_library import DynamicSkillLibrary
    _dynamic_skill_lib: DynamicSkillLibrary | None = None
    _DYNAMIC_SKILL_ENABLED = True
except ImportError:
    _dynamic_skill_lib = None
    _DYNAMIC_SKILL_ENABLED = False
from openalpha_brain.core.loop_state import (
    MAX_BRAIN_MUTATIONS, _brain_cookies_lock, _pool,
    _OPS_RE, _FIELDS_RE,
    _param_optimizer, _monitor, _successful_brain_expressions,
    _failure_lib,
    _feature_map, _evo_db,
    _reflection_engine, _tool_factory,
    _algo_tick,
    get_brain_cookies, set_brain_cookies,
    _log, _build_global_blacklist_prompt,
)
from openalpha_brain.generation.alpha_generator import (
    _extract_expression_from_llm,
    _extract_hallucinations_from_failures,
    _extract_brain_hallucinations,
)
from openalpha_brain.evolution.crossover_mutation import _OPERATOR_SWAP_MAP, _FIELD_SWAP_MAP
from openalpha_brain.evolution.hypothesis_aligner import HypothesisAligner
from openalpha_brain.agents.multi_agent import critique_revise_alpha
from openalpha_brain.utils.paper_edge_enhancements import (
    build_grammar_fallback_chain,
    CrossAttemptTracker,
)
from openalpha_brain.utils.resilience import (
    get_circuit_breaker, async_timeout, TaskHealthRegistry,
)
from openalpha_brain.knowledge.rag_engine import (
    ExperienceReplayManager, FactorContext, RepairSuggestion,
)
from openalpha_brain.monitoring.algorithm_telemetry import AlgorithmTelemetryCollector
logger = logging.getLogger(__name__)

try:
    from openalpha_brain.evolution.ea_search import (
        EASearchStrategy, EAConfig, EAMutationType, FactorIndividual,
        extract_block_a, extract_block_c,
    )
    _EA_AVAILABLE = True
except ImportError as _ea_exc:
    _EA_AVAILABLE = False
    logger.debug("[DEFENSIVE_LOG] EASearchStrategy import failed: %s — EA integration disabled", _ea_exc)

try:
    from openalpha_brain.evolution.tot_search import (
        ToTSearchStrategy, ToTConfig,
    )
    _TOT_AVAILABLE = True
except ImportError as _tot_exc:
    _TOT_AVAILABLE = False
    logger.debug("[DEFENSIVE_LOG] ToTSearchStrategy import failed: %s — ToT integration disabled", _tot_exc)


def _extract_root_operator(expression: str) -> str:
    import re
    match = re.match(r'\s*(\w+)\s*\(', expression)
    return match.group(1) if match else "rank"


def _extract_first_window(expression: str) -> str:
    import re
    numbers = re.findall(r'(\d+)', expression)
    return numbers[0] if numbers else "5"


def _apply_turnover_optimization(expression: str, current_turnover: float) -> str:
    """[Brief description of function purpose.]

        Args:
            expression (str): [Description]
            current_turnover (float): [Description]

        Returns:
            str: [Description]
        """
    if "ts_decay_linear" in expression:
        return expression

    if current_turnover > 50:
        window = 15
    elif current_turnover > 35:
        window = 10
    elif current_turnover > 25:
        window = 6
    else:
        return expression

    stripped = expression.strip()
    if stripped.startswith("group_neutralize("):
        depth = 0
        first_comma_pos = None
        for i, ch in enumerate(stripped):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "," and depth == 1:
                first_comma_pos = i
                break

        if first_comma_pos is not None:
            inner = stripped[len("group_neutralize("):first_comma_pos]
            rest = stripped[first_comma_pos:]
            return f"group_neutralize(ts_decay_linear({inner}, {window}){rest}"

    return f"ts_decay_linear({stripped}, {window})"


def _parse_expression_fingerprint(expression: str) -> dict:
    """輕量級解析 FASTEXPR 表達式的結構指紋。

    使用 regex 提取 root_op、operator_chain、fields、composition_type、nesting_depth。
    不依賴完整 AST 解析器，確保在任何環境下都能運作。

    Args:
        expression: FASTEXPR 格式的 alpha 因子表達式

    Returns:
        包含結構指紋的字典：
        - root_operator: 最外層函數名
        - operator_chain: 按出現順序排列的運算子列表
        - fields: 使用的欄位列表（小寫）
        - composition_type: 結構類型（additive/multiplicative/ratio/single）
        - nesting_depth: 巢狀深度估計
        - has_neutralization: 是否包含 group_neutralize/group_zscore
        - windows: 時間視窗參數列表
    """
    import re

    fp: dict = {
        "root_operator": "",
        "operator_chain": [],
        "fields": [],
        "composition_type": "single",
        "nesting_depth": 0,
        "has_neutralization": False,
        "windows": [],
    }

    if not expression or not expression.strip():
        return fp

    expr = expression.strip()

    root_match = re.match(r'\s*(\w+)\s*\(', expr)
    fp["root_operator"] = root_match.group(1) if root_match else ""

    fp["operator_chain"] = list(dict.fromkeys(re.findall(r'\b([a-zA-Z_]\w*)\s*\(', expr)))

    field_pattern = re.compile(r'\b(close|open|high|low|vwap|volume|adv\d+|returns|cap|sales|assets|equity|revenue|earnings|sharesout)\b', re.IGNORECASE)
    fp["fields"] = list(dict.fromkeys(f.lower() for f in field_pattern.findall(expr)))

    fp["windows"] = [int(n) for n in re.findall(r'(?<!\w)(\d+)(?!\w)', expr) if int(n) <= 365]

    depth = 0
    max_depth = 0
    for ch in expr:
        if ch == '(':
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ')':
            depth -= 1
    fp["nesting_depth"] = max_depth

    if 'group_neutralize' in expr or 'group_zscore' in expr:
        fp["has_neutralization"] = True

    if '*' in expr and '/' in expr:
        fp["composition_type"] = "ratio"
    elif '*' in expr or '/' in expr:
        fp["composition_type"] = "multiplicative"
    elif '+' in expr or '-' in expr:
        fp["composition_type"] = "additive"

    return fp


def _collect_session_expressions(_session_id: str, state=None) -> list[str]:
    """從 session state 收集本 session 已生成的所有表達式。

    包含 passed_alphas 中的 expression 以及當前 improvement loop 中產生的表達式。

    Args:
        session_id: session 識別碼
        state: SessionState 實例，用於從 passed_alphas 收集表達式

    Returns:
        本 session 所有歷史表達式字串列表（去重後）
    """
    expressions: list[str] = []

    for expr in _successful_brain_expressions:
        if expr and isinstance(expr, str):
            expressions.append(expr)

    if state is not None:
        passed_alphas = getattr(state, 'passed_alphas', None) or []
        for alpha in passed_alphas:
            expr = getattr(alpha, 'expression', None)
            if expr and isinstance(expr, str) and expr not in expressions:
                expressions.append(expr)

    return list(dict.fromkeys(expressions))


def _apply_diversity_injection(
    current_expression: str,
    existing_expressions: list[str],
    feature_map,
) -> dict:
    """強制多樣性注入 — 當偵測到同質化趨勢時，生成結構性轉變指引。

    邏輯流程：
      1. 解析 current_expression 的結構指紋（root_op, operator_chain, fields, composition_type）
      2. 與 existing_expressions 比較結構相似度
      3. 如果超過 60% 的已有表達式使用相同的 root_op 或 composition_type：
         a. 從 feature_map 找出使用率最低的運算子/欄位組合（探索獎勵）
         b. 生成具體的結構轉變建議（如「將 rank(ts_delta(...)) 改為 zscore(ts_regression(...))」）
         c. 回傳 diversity_prompt 和 forced_structural_change 指引
      4. 如果沒有同質化問題，回傳空 dict

    Args:
        current_expression: 當前正在改進的 alpha 表達式
        existing_expressions: 本 session 已生成的所有表達式列表
        feature_map: FeatureMap 實例（用於查詢低使用率特徵）

    Returns:
        {
            "diversity_needed": bool,       是否需要多樣性注入
            "similarity_score": float,      0~1, 越高越同質
            "diversity_prompt": str,        LLM 可讀的強制轉變指引
            "suggested_operators": list[str], 建議替換的運算子
            "suggested_fields": list[str],     建議替換的欄位
            "underused_features": list[str],   feature_map 中低使用率的特徵
        }
    """
    result: dict = {
        "diversity_needed": False,
        "similarity_score": 0.0,
        "diversity_prompt": "",
        "suggested_operators": [],
        "suggested_fields": [],
        "underused_features": [],
    }

    try:
        current_fp = _parse_expression_fingerprint(current_expression)

        if not existing_expressions or len(existing_expressions) < 2:
            return result

        current_root = current_fp.get("root_operator", "")
        current_comp = current_fp.get("composition_type", "single")
        current_fields_set = set(current_fp.get("fields", []))
        current_ops_set = set(current_fp.get("operator_chain", []))

        same_root_count = 0
        same_comp_count = 0
        field_overlap_scores: list[float] = []
        op_overlap_scores: list[float] = []

        for expr in existing_expressions:
            if not expr or expr == current_expression:
                continue
            fp = _parse_expression_fingerprint(expr)

            if fp.get("root_operator") == current_root:
                same_root_count += 1
            if fp.get("composition_type") == current_comp:
                same_comp_count += 1

            expr_fields = set(fp.get("fields", []))
            expr_ops = set(fp.get("operator_chain", []))

            if current_fields_set and expr_fields:
                intersection = current_fields_set & expr_fields
                union = current_fields_set | expr_fields
                field_overlap_scores.append(len(intersection) / len(union) if union else 0.0)

            if current_ops_set and expr_ops:
                intersection_ops = current_ops_set | expr_ops
                union_ops = current_ops_set | expr_ops
                op_overlap_scores.append(len(intersection_ops) / len(union_ops) if union_ops else 0.0)

        n_compare = max(len(existing_expressions) - 1, 1)
        root_ratio = same_root_count / n_compare
        comp_ratio = same_comp_count / n_compare
        avg_field_overlap = sum(field_overlap_scores) / len(field_overlap_scores) if field_overlap_scores else 0.0
        avg_op_overlap = sum(op_overlap_scores) / len(op_overlap_scores) if op_overlap_scores else 0.0

        similarity_score = (root_ratio * 0.35 + comp_ratio * 0.25 +
                           avg_field_overlap * 0.20 + avg_op_overlap * 0.20)
        result["similarity_score"] = round(similarity_score, 4)

        homogenized = (root_ratio > 0.60 or comp_ratio > 0.60 or
                       similarity_score > 0.55)

        if not homogenized:
            return result

        result["diversity_needed"] = True

        suggested_operators: list[str] = []
        suggested_fields: list[str] = []
        underused_features: list[str] = []

        all_used_ops: set[str] = set()
        all_used_fields: set[str] = set()
        for expr in existing_expressions:
            fp = _parse_expression_fingerprint(expr)
            all_used_ops.update(fp.get("operator_chain", []))
            all_used_fields.update(fp.get("fields", []))

        candidate_ops = [op for op, alts in _OPERATOR_SWAP_MAP.items()
                         if op in current_ops_set]
        for op in candidate_ops:
            alts = _OPERATOR_SWAP_MAP.get(op, [])
            for alt in alts:
                if alt not in all_used_ops and alt not in suggested_operators:
                    suggested_operators.append(alt)

        if not suggested_operators:
            fallback_ops = ["ts_regression", "ts_corr", "ts_av_diff",
                            "ts_skewness", "signed_power"]
            for fo in fallback_ops:
                if fo not in all_used_ops and fo not in suggested_operators:
                    suggested_operators.append(fo)
                    if len(suggested_operators) >= 3:
                        break

        candidate_fields = [f for f, alts in _FIELD_SWAP_MAP.items()
                            if f.lower() in current_fields_set]
        for fld in candidate_fields:
            alts = _FIELD_SWAP_MAP.get(fld, [])
            for alt in alts:
                alt_lower = alt.lower()
                if alt_lower not in all_used_fields and alt_lower not in suggested_fields:
                    suggested_fields.append(alt_lower)

        fundamental_fields = ["cap", "earnings", "sales", "assets", "revenue", "sharesout"]
        for ff in fundamental_fields:
            if ff not in all_used_fields and ff not in suggested_fields:
                suggested_fields.append(ff)
                if len(suggested_fields) >= 3:
                    break

        result["suggested_operators"] = suggested_operators[:4]
        result["suggested_fields"] = suggested_fields[:4]

        if feature_map is not None:
            try:
                div_stats = feature_map.get_diversity_stats()
                dir_cov = div_stats.get("direction_coverage", {})
                underused_dirs = [d for d, cov in dir_cov.items() if cov < 0.25]
                underused_features = [f"方向={d} (覆蓋率 {dir_cov[d]:.0%})"
                                      for d in underused_dirs[:3]]
                unexplored = feature_map.get_unexplored_directions()
                for ud in unexplored[:2]:
                    feat_label = f"未探索方向={ud}"
                    if feat_label not in underused_features:
                        underused_features.append(feat_label)
                result["underused_features"] = underused_features[:5]
            except (OSError, ValueError, RuntimeError) as exc:
                logger.debug("[DEFENSIVE_LOG] _apply_diversity_injection: feature_map 获取未探索方向失败，跳过该增强: %s", exc)

        level = "Level 1 (輕度)"
        if similarity_score > 0.70:
            level = "Level 3 (創新性拓撲變換)"
        elif similarity_score > 0.62:
            level = "Level 2 (中度 — 運算子家族切換 + 新欄位)"

        prompt_parts: list[str] = [
            f"=== COGALPHA DIVERSITY INJECTION ({level}) ===",
            f"同質化偵測：相似度={similarity_score:.1%} "
            f"(root_op一致={root_ratio:.0%}, 結構類型一致={comp_ratio:.0%}, "
            f"欄位重疊={avg_field_overlap:.1%}, 運算子重疊={avg_op_overlap:.1%})",
            "",
            "強制多樣性操作（必須執行其中一項）：",
        ]

        if suggested_operators:
            op_list = ", ".join(suggested_operators[:3])
            prompt_parts.append(
                f"  [運算子替換] 當前使用 '{current_root}' 系列運算子。"
                f"改用以下未被充分探索的運算子：{op_list}"
            )
            prompt_parts.append(
                f"  例如：rank(ts_delta(close, 5)) → zscore(ts_regression({suggested_fields[0] if suggested_fields else 'vwap'}, 10))"
            )

        if suggested_fields:
            fld_list = ", ".join(suggested_fields[:3])
            prompt_parts.append(
                f"  [欄位跳躍] 當前使用價格/成交量欄位 ({', '.join(list(current_fields_set)[:4]) if current_fields_set else 'N/A'})。"
                f"引入基本面/替代資料欄位：{fld_list}"
            )

        if underused_features:
            feat_list = "; ".join(underused_features[:3])
            prompt_parts.append(
                f"  [FeatureMap 探索獎勵] 以下特徵空間幾乎未被佔據：{feat_list}"
            )

        if current_comp == "multiplicative":
            prompt_parts.append(
                "  [拓撲變換] 當前為乘法組成 → 建議改為條件式 (trade_when) 或加法組成"
            )
        elif current_comp == "single":
            prompt_parts.append(
                "  [拓撲變換] 當前為單一訊號 → 建議改為多因子交互作用 (* 或 / 組合)"
            )

        prompt_parts.extend([
            "",
            "參考：CogAlpha (arXiv 2511.18850) — Multi-level diversity injection",
            "=== END DIVERSITY INJECTION ===",
        ])

        result["diversity_prompt"] = "\n".join(prompt_parts)

    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        logger.warning("_apply_diversity_injection 發生異常，回退到原始 prompt: %s", exc)
        return {
            "diversity_needed": False,
            "similarity_score": 0.0,
            "diversity_prompt": "",
            "suggested_operators": [],
            "suggested_fields": [],
            "underused_features": [],
        }

    return result


def _apply_elite_crossover(
    current_expression: str,
    elite_expressions: list[dict],
    max_elites: int = 3,
) -> dict:
    """精英交叉 — 從本 session/歷史中表現最好的 alpha 提取子結構進行交叉。

    選取 top-K 精英 alpha（按 sharpe 排序），分析其與當前表達式的結構差異，
    生成具體的 crossover 操作指引：
    - 平滑方式替換（ts_mean ↔ ts_decay_linear）
    - 欄位引入（close/volume → earnings/cap）
    - 巢狀深度調整（增強表達力）

    Args:
        current_expression: 當前正在改進的 alpha 表達式
        elite_expressions: 高 sharpe 的歷史 alpha 列表，
                           每個元素為 {"expr": str, "sharpe": float}
        max_elites: 最多使用的精英數量（預設 3）

    Returns:
        {
            "crossover_possible": bool,     是否可執行交叉
            "elite_source": str,             來源 alpha 的 expr snippet
            "elite_sharpe": float,           來源 alpha 的 sharpe
            "keep_structure": str,           建議保留的當前部分
            "replace_structure": str,        建議替換為的精英部分
            "crossover_prompt": str,         LLM 可讀的交叉指引
            "expected_improvement": str,      預期改善方向說明
        }
    """
    result: dict = {
        "crossover_possible": False,
        "elite_source": "",
        "elite_sharpe": 0.0,
        "keep_structure": "",
        "replace_structure": "",
        "crossover_prompt": "",
        "expected_improvement": "",
    }

    try:
        if not elite_expressions or not current_expression:
            return result

        sorted_elites = sorted(
            [e for e in elite_expressions if e.get("expr") and e.get("sharpe") is not None],
            key=lambda x: x.get("sharpe", 0.0),
            reverse=True,
        )[:max_elites]

        if not sorted_elites:
            return result

        current_fp = _parse_expression_fingerprint(current_expression)
        current_root = current_fp.get("root_operator", "")
        current_fields = set(current_fp.get("fields", []))
        current_ops = set(current_fp.get("operator_chain", []))
        current_depth = current_fp.get("nesting_depth", 0)

        best_elite = None
        best_diff_score = -1.0

        for elite in sorted_elites:
            elite_expr = elite["expr"]
            elite_sharpe = elite.get("sharpe", 0.0)
            elite_fp = _parse_expression_fingerprint(elite_expr)
            elite_root = elite_fp.get("root_operator", "")
            elite_fields = set(elite_fp.get("fields", []))
            elite_ops = set(elite_fp.get("operator_chain", []))
            elite_depth = elite_fp.get("nesting_depth", 0)

            structural_diff = 0.0
            if elite_root != current_root:
                structural_diff += 0.30
            field_symdiff = current_fields ^ elite_fields
            structural_diff += min(len(field_symdiff) * 0.10, 0.30)
            op_symdiff = current_ops ^ elite_ops
            structural_diff += min(len(op_symdiff) * 0.08, 0.25)
            if abs(elite_depth - current_depth) >= 2:
                structural_diff += 0.15

            has_new_fundamental = bool(elite_fields - current_fields & 
                                       {"cap", "earnings", "sales", "assets", "revenue"})
            if has_new_fundamental:
                structural_diff += 0.10

            if structural_diff > best_diff_score:
                best_diff_score = structural_diff
                best_elite = {
                    "expr": elite_expr,
                    "sharpe": elite_sharpe,
                    "fp": elite_fp,
                    "root": elite_root,
                    "fields": elite_fields,
                    "ops": elite_ops,
                    "depth": elite_depth,
                    "diff_score": structural_diff,
                }

        if best_elite is None or best_diff_score < 0.20:
            return result

        elite_fp = best_elite["fp"]
        elite_root = best_elite["root"]
        elite_fields = best_elite["fields"]
        elite_ops = best_elite["ops"]
        elite_depth = best_elite["depth"]

        keep_parts: list[str] = []
        replace_parts: list[str] = []
        expected_improvements: list[str] = []

        if elite_root != current_root and elite_root in _OPERATOR_SWAP_MAP.get(current_root, []):
            replace_parts.append(
                f"將根運算子從 '{current_root}' 替換為 '{elite_root}' "
                f"(來自 Sharpe={best_elite['sharpe']:.2f} 的精英 alpha)"
            )
            keep_parts.append(f"保留內部子表達式結構")
            expected_improvements.append(
                f"運算子替換：'{current_root}'→'{elite_root}' 可能改變時序平滑特性"
            )
        elif elite_root != current_root:
            replace_parts.append(
                f"考慮使用精英 alpha 的根運算子 '{elite_root}' "
                f"(Sharpe={best_elite['sharpe']:.2f}) 替代或包裝當前的 '{current_root}'"
            )

        new_fundamental_fields = elite_fields - current_fields & \
            {"cap", "earnings", "sales", "assets", "revenue", "sharesout"}
        if new_fundamental_fields:
            fld_names = ", ".join(sorted(new_fundamental_fields)[:3])
            replace_parts.append(
                f"引入精英 alpha 的基本面欄位：{fld_names}"
            )
            expected_improvements.append(
                f"基本面欄位注入可能降低與價格因子的相關性"
            )

        new_price_fields = elite_fields - current_fields & \
            {"open", "high", "low", "vwap"}
        if new_price_fields and not new_fundamental_fields:
            fld_names = ", ".join(sorted(new_price_fields)[:3])
            replace_parts.append(
                f"替換為精英 alpha 使用的價格欄位：{fld_names}"
            )

        smooth_swap_pairs = [
            ("ts_mean", "ts_decay_linear"),
            ("ts_decay_linear", "ts_mean"),
            ("rank", "ts_rank"),
            ("ts_rank", "rank"),
        ]
        for old_op, new_op in smooth_swap_pairs:
            if old_op in current_ops and new_op in elite_ops:
                replace_parts.append(
                    f"平滑方式替換：{old_op} → {new_op} "
                    f"(精英 alpha 使用此方式達到 Sharpe={best_elite['sharpe']:.2f})"
                )
                expected_improvements.append(
                    f"{new_op} 可能有不同的延遲/衰減特性，影響 turnover 和 sharpe"
                )
                break

        if elite_depth > current_depth + 1:
            replace_parts.append(
                f"增加巢狀深度：精英 alpha 深度={elite_depth} > 當前深度={current_depth}，"
                f"建議增加一層包裝（如 ts_zscore 或 ts_std_dev）以增強表達力"
            )
            expected_improvements.append("更深的巢狀可能捕獲更複雜的非線性關係")

        if not replace_parts:
            return result

        if not keep_parts:
            keep_parts.append(f"保留當前表達式的整體架構和中和層")

        result["crossover_possible"] = True
        result["elite_source"] = best_elite["expr"][:120]
        result["elite_sharpe"] = best_elite["sharpe"]
        result["keep_structure"] = "; ".join(keep_parts[:3])
        result["replace_structure"] = "; ".join(replace_parts[:4])
        result["expected_improvement"] = "; ".join(expected_improvements[:3])

        prompt_lines: list[str] = [
            "=== COGALPHA ELITE CROSSOVER ===",
            f"來源精英 Alpha (Sharpe={best_elite['sharpe']:.2f}): {best_elite['expr'][:100]}...",
            f"結構差異分數: {best_diff_score:.2f}/1.00",
            "",
            "交叉操作指引（從以下選擇最適合的一項執行）：",
        ]

        for i, rp in enumerate(replace_parts[:4], 1):
            prompt_lines.append(f"  OP{i}: {rp}")

        prompt_lines.extend([
            "",
            f"保留的部分: {result['keep_structure']}",
            "",
            f"預期改善: {result['expected_improvement']}",
            "",
            "約束條件：產生的表達式必須通過語法驗證和白名單檢查。",
            "參考：CogAlpha — Elite-assisted crossover with fitness inheritance",
            "=== END ELITE CROSSOVER ===",
        ])

        result["crossover_prompt"] = "\n".join(prompt_lines)

    except (ValueError, TypeError, OSError, RuntimeError, KeyError, AttributeError, ImportError) as exc:
        logger.warning("_apply_elite_crossover 發生異常，回退到原始 prompt: %s", exc)
        return {
            "crossover_possible": False,
            "elite_source": "",
            "elite_sharpe": 0.0,
            "keep_structure": "",
            "replace_structure": "",
            "crossover_prompt": "",
            "expected_improvement": "",
        }

    return result


def _build_brain_result_dict(brain_result) -> dict:
    """[Brief description of function purpose.]

        Args:
            brain_result: [Description]

        Returns:
            dict: [Description]
        """
    brain_checks = getattr(brain_result, "brain_checks", []) or []
    failed_check_names = [c["name"] for c in brain_checks if c.get("result") == "FAIL"]
    return {
        "status": brain_result.status.value if brain_result.status else "ERROR",
        "sharpe": brain_result.real_sharpe,
        "fitness": brain_result.real_fitness,
        "turnover": brain_result.real_turnover,
        "checks": failed_check_names,
    }



def _log_brain_result(state, result, expression: str, attempt: int) -> None:
    """[Brief description of function purpose.]

        Args:
            state: [Description]
            result: [Description]
            expression (str): [Description]
            attempt (int): [Description]

        Returns:
            None: [Description]
        """
    checks = getattr(result, "brain_checks", []) or []
    failures = [c for c in checks if c.get("result") == "FAIL"]
    err_msg  = getattr(result, "gate_failures", [])

    if result.status.value == "PASS":
        s = f"{result.real_sharpe:.3f}" if result.real_sharpe is not None else "N/A"
        f = f"{result.real_fitness:.3f}" if result.real_fitness is not None else "N/A"
        t = f"{result.real_turnover:.1f}%" if result.real_turnover is not None else "N/A"
        _log(state, "BRAIN_PASS",
             f"BRAIN PASS — Sharpe={s} Fitness={f} TO={t}",
             {"brain_id": result.alpha_id, "attempt": attempt})
        return

    if failures:
        for chk in failures:
            _log(state, "BRAIN_FAIL",
                 f"BRAIN check FAILED: {chk['name']} "
                 f"(value={chk.get('value')}, limit={chk.get('limit')})",
                 {"check": chk["name"], "value": chk.get("value"),
                  "limit": chk.get("limit"), "attempt": attempt})
    elif err_msg:
        tag = "BRAIN_ERROR" if result.status == BrainSimStatus.ERROR else "BRAIN_FAIL"
        for em in err_msg[:3]:
            _log(state, tag, f"BRAIN simulation {result.status.value}: {em[:120]}",
                 {"attempt": attempt})
        _extract_brain_hallucinations(state, err_msg)

    _log(state, "METRICS",
         f"Real metrics — Sharpe={result.real_sharpe} "
         f"Fitness={result.real_fitness} TO={result.real_turnover}% "
         f"Returns={result.real_returns}%",
         {"sharpe": result.real_sharpe, "fitness": result.real_fitness,
          "turnover": result.real_turnover, "returns": result.real_returns})


_brain_submit_semaphore: asyncio.Semaphore | None = None


def _micro_mutate_expression(expression: str, best_result, attempt: int) -> str:
    """對表達式進行微調變異，用於 improvement loop 的每次迭代。

    策略按 attempt 遞進：
      1. 窗口參數微調（替換數字常量）
      2. 運算子替換（使用 _OPERATOR_SWAP_MAP）
      3. Turnover 優化包裝（ts_decay_linear）
      4. 欄位替換（使用 _FIELD_SWAP_MAP）
      5. 組合策略：窗口 + 運算子同時替換

    Args:
        expression: 當前表達式
        best_result: 目前最佳結果（用於提取 turnover 等指標）
        attempt: 當前嘗試次數（從 1 開始）

    Returns:
        變異後的表達式，若無法變異則返回原表達式
    """
    import re
    if not expression or len(expression) < 5:
        return expression

    strategy = attempt % 5

    if strategy == 1:
        numbers = re.findall(r'\b(\d+)\b', expression)
        if numbers:
            import random
            target = random.choice(numbers)
            old_val = int(target)
            new_val = max(2, old_val + random.choice([-3, -2, -1, 1, 2, 3, 5]))
            mutated = expression.replace(target, str(new_val), 1)
            return mutated

    elif strategy == 2:
        for op, alts in _OPERATOR_SWAP_MAP.items():
            if op in expression and alts:
                import random
                alt = random.choice(alts)
                mutated = expression.replace(op, alt, 1)
                return mutated

    elif strategy == 3:
        turnover = getattr(best_result, 'real_turnover', None)
        if turnover is not None and turnover > 25:
            mutated = _apply_turnover_optimization(expression, turnover)
            if mutated != expression:
                return mutated
        numbers = re.findall(r'\b(\d+)\b', expression)
        if numbers:
            import random
            target = random.choice(numbers)
            old_val = int(target)
            new_val = max(2, old_val + random.choice([-5, -3, 3, 5, 8]))
            return expression.replace(target, str(new_val), 1)

    elif strategy == 4:
        for fld, alts in _FIELD_SWAP_MAP.items():
            pattern = re.compile(re.escape(fld), re.IGNORECASE)
            if pattern.search(expression) and alts:
                import random
                alt = random.choice(alts)
                mutated = pattern.sub(alt, expression, count=1)
                return mutated

    else:
        numbers = re.findall(r'\b(\d+)\b', expression)
        if numbers:
            import random
            target = random.choice(numbers)
            old_val = int(target)
            new_val = max(2, old_val + random.choice([-2, 2, 4]))
            mutated = expression.replace(target, str(new_val), 1)
            for op, alts in _OPERATOR_SWAP_MAP.items():
                if op in mutated and alts:
                    alt = random.choice(alts)
                    mutated = mutated.replace(op, alt, 1)
                    break
            return mutated

    return expression


def _get_brain_semaphore() -> asyncio.Semaphore:
    global _brain_submit_semaphore
    if _brain_submit_semaphore is None:
        _brain_submit_semaphore = asyncio.Semaphore(3)
    return _brain_submit_semaphore


async def create_dedup_mutation_callback(expression: str, result: dict) -> dict | None:
    """Generate a semantic variant of a duplicated alpha to bypass BRAIN uniqueness.

    When BRAIN reports self_correlation > 0.7 for a submitted alpha, this callback
    uses LLM to generate a variant that preserves the core signal logic but replaces
    operators/windows to pass the uniqueness threshold.

    Args:
        expression: The original alpha expression that was flagged as duplicate
        result: BRAIN result dict containing duplication/metadata

    Returns:
        Dict with {"expression": str, "reason": str} on success, or None on failure
    """
    dup_checks = result.get("brain_checks", []) if isinstance(result, dict) else getattr(result, "brain_checks", []) or []
    dup_details = [
        f"{c.get('name', '?')}: {c.get('value', '?')}"
        for c in dup_checks
        if c.get("name", "").upper() in ("SELF_CORRELATION", "CORRELATION", "DUPLICATION")
    ]

    mutation_prompt = f"""You are a quantitative alpha factor engineer. The following alpha was flagged as DUPLICATE (self_correlation > 0.7) by the BRAIN platform.

Original expression: {expression}
Duplication details: {'; '.join(dup_details) if dup_details else 'SELF_CORRELATION > 0.7'}

Generate a SEMANTIC VARIANT that:
1. Preserves the core economic intuition (same signal concept)
2. Replaces at least one operator with a functionally equivalent alternative
3. Changes at least one lookback window parameter
4. Optionally wraps with a different rank/smoothing operator
5. MUST remain syntactically valid for the BRAIN FASTEXPR grammar

Return ONLY a JSON object with the variant:
{{"expression": "the new variant expression", "reason": "brief description of what was changed"}}"""

    try:
        raw = await llm_client.generate(
            mutation_prompt, [], "",
            session_id="dedup_mutation", cycle=0,
        )
        parsed = _extract_json_from_llm(raw)
        if parsed and isinstance(parsed, dict):
            variant_expr = parsed.get("expression", "")
            if variant_expr and variant_expr != expression:
                logger.info(
                    "dedup_mutation: generated variant expr=%s... reason=%s",
                    variant_expr[:80], parsed.get("reason", "")[:80],
                )
                return parsed
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("dedup_mutation: LLM variant generation failed: %s", exc)
    return None


async def _submit_to_brain(alpha, session_id: str, cycle_num: int):
    from openalpha_brain.core import loop_state as _ls
    from openalpha_brain.core.models import BrainSubmissionResult, BrainSimStatus

    eid = None
    tel = AlgorithmTelemetryCollector.get_instance()
    t0 = time.perf_counter()
    try:
        eid = await tel.record_enter("BrainSubmitter", cycle_id=session_id, expr_id=hash(getattr(alpha, 'alpha_id', '')) % 10000)
    except (OSError, ValueError, RuntimeError):
        pass

    _brain_cb = get_circuit_breaker("brain_api", failure_threshold=5, recovery_timeout=30.0)

    if _brain_cb.is_open:
        logger.warning(
            "[%s] cycle=%d BRAIN_CIRCUIT_OPEN: skipping submission, reason=%s",
            session_id, cycle_num, _brain_cb._last_failure_reason[:100],
        )
        return BrainSubmissionResult(
            status=BrainSimStatus.ERROR,
            real_sharpe=None, real_fitness=None, real_turnover=None,
            real_returns=None, real_drawdown=None,
            alpha_id=getattr(alpha, 'alpha_id', ''),
            gate_warnings=[f"Circuit breaker open: {_brain_cb._last_failure_reason[:80]}"],
            brain_checks=[],
        )

    sim_payload = getattr(alpha, 'simulation_payload', None) or {}
    if not sim_payload.get("settings") or not sim_payload.get("regular"):
        logger.warning("[%s] cycle=%d BRAIN_SKIP: no simulation_payload on alpha %s", session_id, cycle_num, getattr(alpha, 'alpha_id', '?'))
        return BrainSubmissionResult(
            status=BrainSimStatus.ERROR,
            real_sharpe=None, real_fitness=None, real_turnover=None,
            real_returns=None, real_drawdown=None,
            alpha_id=getattr(alpha, 'alpha_id', ''),
            gate_warnings=["No simulation payload"],
            brain_checks=[],
        )

    _sem = _get_brain_semaphore()
    async with _sem:
        cookies = _ls._brain_cookies if hasattr(_ls, '_brain_cookies') else None
        if not cookies:
            _auth_attempts = 0
            _max_auth_retries = 3
            _auth_base_delay = 2.0
            while _auth_attempts < _max_auth_retries:
                try:
                    _email = getattr(settings, 'BRAIN_EMAIL', None) or ''
                    _pwd = getattr(settings, 'BRAIN_PASSWORD', None) or ''
                    if not _email or not _pwd:
                        raise brain_client.BrainAuthError("BRAIN_EMAIL/BRAIN_PASSWORD not configured")
                    cookies = await async_timeout(
                        brain_client.authenticate(_email, _pwd),
                        timeout_seconds=15.0,
                        name="brain_auth",
                    )
                    _ls._brain_cookies = cookies
                    break
                except asyncio.TimeoutError:
                    _auth_attempts += 1
                    if _auth_attempts >= _max_auth_retries:
                        logger.error("[%s] cycle=%d BRAIN_AUTH_TIMEOUT after %d retries", session_id, cycle_num, _max_auth_retries)
                        _brain_cb.record_failure("Auth timeout")
                        return BrainSubmissionResult(
                            status=BrainSimStatus.ERROR,
                            real_sharpe=None, real_fitness=None, real_turnover=None,
                            real_returns=None, real_drawdown=None,
                            alpha_id=getattr(alpha, 'alpha_id', ''),
                            gate_warnings=["Auth timeout after 3 retries"],
                            brain_checks=[],
                        )
                    _delay = _auth_base_delay * (2 ** (_auth_attempts - 1))
                    logger.warning("[%s] cycle=%d BRAIN_AUTH_RETRY %d/%d in %.1fs", session_id, cycle_num, _auth_attempts, _max_auth_retries, _delay)
                    await asyncio.sleep(_delay)
                except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError) as e:
                    _auth_attempts += 1
                    if _auth_attempts >= _max_auth_retries:
                        logger.error("[%s] cycle=%d BRAIN_AUTH_FAIL: %s", session_id, cycle_num, e)
                        _brain_cb.record_failure(f"Auth failed: {e}")
                        return BrainSubmissionResult(
                            status=BrainSimStatus.ERROR,
                            real_sharpe=None, real_fitness=None, real_turnover=None,
                            real_returns=None, real_drawdown=None,
                            alpha_id=getattr(alpha, 'alpha_id', ''),
                            gate_warnings=[f"Auth failed: {e}"],
                            brain_checks=[],
                        )
                    _delay = _auth_base_delay * (2 ** (_auth_attempts - 1))
                    logger.warning("[%s] cycle=%d BRAIN_AUTH_RETRY %d/%d: %s", session_id, cycle_num, _auth_attempts, _max_auth_retries, e)
                    await asyncio.sleep(_delay)

        try:
            result = await async_timeout(
                brain_client.submit_and_poll(
                    simulation_payload=sim_payload,
                    cookies=cookies,
                    max_poll_seconds=settings.BRAIN_POLL_TIMEOUT or 300,
                ),
                timeout_seconds=(settings.BRAIN_POLL_TIMEOUT or 300) + 60,
                name="brain_submit_and_poll",
            )
            _brain_cb.record_success()
            _sim_status = BrainSimStatus.PASS if getattr(result, 'passed', False) else BrainSimStatus.FAIL
            if hasattr(result, 'simulation_status') and result.simulation_status == "ERROR":
                _sim_status = BrainSimStatus.ERROR
            return BrainSubmissionResult(
                status=_sim_status,
                real_sharpe=result.sharpe,
                real_fitness=result.fitness,
                real_turnover=result.turnover,
                real_returns=result.returns,
                real_drawdown=result.drawdown,
                alpha_id=result.alpha_id or getattr(alpha, 'alpha_id', ''),
                real_margin=result.margin,
                gate_warnings=result.warnings,
                gate_failures=result.failures or [],
                brain_checks=result.brain_checks,
            )
        except asyncio.TimeoutError:
            logger.error("[%s] cycle=%d BRAIN_SUBMIT_TIMEOUT", session_id, cycle_num)
            _brain_cb.record_failure("Submit timeout")
            return BrainSubmissionResult(
                status=BrainSimStatus.ERROR,
                real_sharpe=None, real_fitness=None, real_turnover=None,
                real_returns=None, real_drawdown=None,
                alpha_id=getattr(alpha, 'alpha_id', ''),
                gate_warnings=["BRAIN submit timeout exceeded"],
                brain_checks=[],
            )
        except brain_client.BrainSubmitError as e:
            _brain_cb.record_failure(f"BrainSubmitError: {e}")
            return BrainSubmissionResult(
                status=BrainSimStatus.FAIL,
                real_sharpe=None, real_fitness=None, real_turnover=None,
                real_returns=None, real_drawdown=None,
                alpha_id=getattr(alpha, 'alpha_id', ''),
                gate_warnings=[str(e)],
                brain_checks=[],
            )
        except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError) as e:
            _brain_cb.record_failure(f"Submit error: {e}")
            return BrainSubmissionResult(
                status=BrainSimStatus.ERROR,
                real_sharpe=None, real_fitness=None, real_turnover=None,
                real_returns=None, real_drawdown=None,
                alpha_id=getattr(alpha, 'alpha_id', ''),
                gate_warnings=[f"Submit error: {e}"],
                brain_checks=[],
            )

    # Semaphore for limiting concurrent BRAIN submissions (max 3, matches BRAIN's 3 concurrent simulation slots)
_brain_submit_semaphore = asyncio.Semaphore(3)


async def _brain_improvement_loop(*, initial_result, session_id, cycle_num=0, expression="",
                                exploration_direction=None, pipeline_orchestrator=None,
                                initial_expression=None, alpha=None, global_cycle=None,
                                grammar=None, resource_dispatcher=None):
    """Run improvement loop for BRAIN alpha with rate-limited submissions.
    
    Args:
        initial_result: Initial BRAIN submission result
        session_id: Session identifier
        cycle_num: Current cycle number
        expression: Alpha expression string
        exploration_direction: Direction for exploration
        pipeline_orchestrator: Pipeline orchestrator instance
        initial_expression: Original expression before modifications
        alpha: Alpha object with metadata
        global_cycle: Global cycle counter
        grammar: Grammar rules for validation
        resource_dispatcher: Resource dispatcher for managing resources
    
    Returns:
        Best result from improvement iterations
    """
    if initial_result is None:
        if not expression and not initial_expression:
            logger.warning("[%s] cycle=%d BRAIN_IMPROVE_SKIP: no initial_result and no expression", session_id, cycle_num)
            return None
        logger.info("[%s] cycle=%d BRAIN_IMPROVE_DEGRADE: entering degradation mode with expression", session_id, cycle_num)
        initial_result = BrainSubmissionResult(
            status=BrainSimStatus.FAIL,
            real_sharpe=None, real_fitness=None, real_turnover=None,
            real_returns=None, real_drawdown=None,
            alpha_id=getattr(alpha, 'alpha_id', '') if alpha else '',
            gate_warnings=["Degradation mode: no initial result"],
            brain_checks=[],
        )

    _stability_guard = None
    try:
        from openalpha_brain.validation.stability_guard import StabilityGuard
        _stability_guard = StabilityGuard()
        logger.info("[%s] cycle=%d [STABILITY] ✓ StabilityGuard initialized for improvement loop", session_id, cycle_num)
    except (ImportError, ValueError, OSError) as stab_init_exc:
        logger.warning("[%s] cycle=%d [STABILITY] ⚠ Init failed (graceful degradation): %s", session_id, cycle_num, stab_init_exc)

    best_result = initial_result
    current_expr = expression or initial_expression or ""
    
    # Check if we should attempt improvements
    if getattr(initial_result, 'status', None) == BrainSimStatus.PASS:
        sharpe = getattr(initial_result, 'real_sharpe', 0) or 0
        if sharpe >= 1.75:
            _log(session_id, "BRAIN_GOOD_ENOUGH",
                 f"Sharpe={sharpe:.3f} already good enough, skipping improvement loop",
                 {"cycle": cycle_num})
            return best_result

    failure_diagnosis = None
    try:
        from openalpha_brain.services import llm_client as _diag_llm_client
        brain_checks = getattr(initial_result, 'brain_checks', []) or []
        hypothesis_text = ""
        if alpha and hasattr(alpha, 'economic_rationale'):
            hypothesis_text = alpha.economic_rationale or ""
        if not hypothesis_text and alpha and hasattr(alpha, 'rationale'):
            hypothesis_text = alpha.rationale or ""
        if not hypothesis_text:
            hypothesis_text = exploration_direction or ""

        async def _diag_llm(prompt: str) -> str:
            return await _diag_llm_client.generate(prompt, [], "", session_id="brain_diagnosis", cycle=cycle_num)

        failure_diagnosis = await llm_diagnose_failure(
            brain_checks=brain_checks,
            expression=expression or initial_expression or "",
            hypothesis_text=hypothesis_text,
            llm_call_fn=_diag_llm,
        )
        if failure_diagnosis:
            _log(session_id, "BRAIN_DIAGNOSIS",
                 f"LLM diagnosed: {failure_diagnosis.get('failure_type', 'unknown')} "
                 f"root_cause={failure_diagnosis.get('root_cause', '')[:100]} "
                 f"confidence={failure_diagnosis.get('confidence', 0):.2f}",
                 {"cycle": cycle_num})
        if failure_diagnosis and _failure_lib and settings.FAILURE_FIX_LIBRARY_ENABLED:
            try:
                await _failure_lib.add_failure(
                    expr=expression or initial_expression or "",
                    failure_type=failure_diagnosis.get("failure_type", "BRAIN_FAIL"),
                    fix_attempt=failure_diagnosis.get("suggested_fix", ""),
                    fix_success=False,
                    direction=exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
                    session_id=session_id,
                )
            except (OSError, ValueError, RuntimeError) as exc:
                logger.debug("[DEFENSIVE_LOG] brain_improvement: failure_lib add_failure 失败，不影响主流程: %s", exc, exc_info=True)
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        logger.warning("[%s] cycle=%d BRAIN_DIAGNOSIS_FAILED: %s", session_id, cycle_num, exc)

    try:
        from openalpha_brain.services import llm_client as _critique_llm
        critique_result = await critique_revise_alpha(
            expression=expression or initial_expression or "",
            brain_result=initial_result,
            llm_generate=_critique_llm.generate,
            hypothesis_text=hypothesis_text if 'hypothesis_text' in dir() else "",
            direction=exploration_direction or settings.DEFAULT_EXPLORATION_DIRECTION,
            max_rounds=3,
        )
        if critique_result and critique_result.get("expression"):
            new_expr = critique_result["expression"]
            if new_expr != current_expr and len(new_expr) >= 5:
                _log(session_id, "BRAIN_CRITIQUE_REVISE",
                     f"3-round critique-revise: {critique_result['rounds_completed']} rounds, "
                     f"orig={current_expr[:40]}... → revised={new_expr[:40]}...",
                     {"cycle": cycle_num, "rounds": critique_result["rounds_completed"]})
                current_expr = new_expr
                if alpha is not None:
                    alpha.expression = current_expr
                    if alpha.simulation_payload is not None:
                        alpha.simulation_payload["regular"] = current_expr
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        logger.warning("[%s] cycle=%d critique_revise_alpha failed: %s", session_id, cycle_num, exc)

    max_attempts = min(MAX_BRAIN_MUTATIONS, 5)

    classifier = FailureClassifier()
    mutator = SegmentLockedMutator()

    _exp_replay: ExperienceReplayManager | None = None
    try:
        _exp_replay = ExperienceReplayManager(llm_client=llm_client)
        logger.info("[%s] cycle=%d EXP_REPLAY initialized", session_id, cycle_num)
    except (ValueError, TypeError, OSError) as exc:
        logger.warning("[%s] cycle=%d EXP_REPLAY_INIT_FAILED: %s — continuing without experience replay", session_id, cycle_num, exc)

    _current_exp_card_id: str | None = None

    try:
        from openalpha_brain.knowledge.field_proxy_map import FieldProxyMap as _FPM
        _fpm_instance = _FPM()
        if not _fpm_instance._loaded:
            _fpm_instance.load()
    except (OSError, FileNotFoundError, ValueError) as exc:
        _fpm_instance = None

    try:
        from openalpha_brain.services import llm_client as _refl_llm_client
        async def _refl_llm(prompt: str, messages: list, system: str, **kwargs):
            return await _refl_llm_client.generate(prompt, messages, system, **kwargs)
    except (ImportError, AttributeError):
        _refl_llm = None

    proxy_evaluator = ProxyEvaluator(field_proxy_map=_fpm_instance)
    _prefilter = SignalQualityPreFilter()
    reflexion_engine = ReflexionEngine(
        llm_generate_fn=_refl_llm,
        proxy_evaluator=proxy_evaluator,
    )

    _ea_strategy: EASearchStrategy | None = None
    if _EA_AVAILABLE:
        try:
            _ea_config = EAConfig(
                population_size=6,
                max_generations=2,
                elite_ratio=0.25,
                mutation_rate=0.6,
                crossover_rate=0.3,
                llm_mutation_prob=0.2,
                diversity_threshold=0.6,
                timeout_seconds=180,
            )
            _ea_strategy = EASearchStrategy(config=_ea_config)

            from openalpha_brain.evolution.near_pass_improver import NearPassImprover
            _ea_strategy.initialize_dependencies(
                near_pass_improver=NearPassImprover(),
                llm_client=_refl_llm_client if '_refl_llm_client' in dir() else None,
                prefilter=_prefilter,
            )
            logger.info("[%s] cycle=%d [EA] ✅ EASearchStrategy initialized (pop=%d, gen=%d)",
                        session_id, cycle_num, _ea_config.population_size, _ea_config.max_generations)
        except (ValueError, TypeError, OSError) as _ea_init_exc:
            logger.warning("[%s] cycle=%d [EA] ⚠️ Init failed: %s — falling back to standard loop",
                           session_id, cycle_num, _ea_init_exc)
            _ea_strategy = None

    _tot_strategy: ToTSearchStrategy | None = None
    if _TOT_AVAILABLE:
        try:
            _tot_config = ToTConfig(
                max_depth=3,
                branch_factor=4,
                top_k_survivors=2,
                accept_threshold=0.0,
                llm_expand_ratio=0.5,
                mutation_ratio=0.3,
                crossover_ratio=0.2,
                timeout_seconds=240,
                max_total_nodes=50,
                enforce_diversity=True,
            )
            _tot_strategy = ToTSearchStrategy(config=_tot_config)

            from openalpha_brain.evolution.near_pass_improver import NearPassImprover
            _tot_strategy.initialize_dependencies(
                near_pass_improver=NearPassImprover(),
                llm_client=_refl_llm_client if '_refl_llm_client' in dir() else None,
                prefilter=_prefilter,
            )
            logger.info("[%s] cycle=%d [TOT] ✅ ToTSearchStrategy initialized (depth=%d, branch=%d)",
                        session_id, cycle_num, _tot_config.max_depth, _tot_config.branch_factor)
        except (ValueError, TypeError, OSError) as _tot_init_exc:
            _tot_strategy = None

    for attempt in range(max_attempts):
        try:
            if _stability_guard is not None and attempt > 0 and current_expr:
                try:
                    prev_sharpe = getattr(best_result, 'real_sharpe', 0) or 0
                    stab_eval = _stability_guard.evaluate_and_guard(
                        expression=current_expr,
                        cycle=cycle_num * 100 + attempt,
                        current_sharpe=prev_sharpe,
                        is_mutation=True,
                    )
                    _log(session_id, "STABILITY_EVAL",
                         f"Attempt {attempt + 1}: stability_score={stab_eval['stability_score']:.2f} "
                         f"is_stable={stab_eval['is_stable']} "
                         f"should_restrict={stab_eval['should_restrict']} "
                         f"type={stab_eval.get('instability_type', 'N/A')}",
                         {"cycle": cycle_num, "attempt": attempt + 1})

                    if stab_eval["should_restrict"]:
                        _log(session_id, "STABILITY_RESTRICTED",
                             f"Attempt {attempt + 1}: Expression pool unstable, applying constraints",
                             {"cycle": cycle_num, "attempt": attempt + 1,
                              "severity": stab_eval["severity"],
                              "diagnosis": stab_eval.get("diagnosis", "")})
                except (OSError, ValueError, RuntimeError) as stab_loop_exc:
                    logger.debug("[%s] cycle=%d [STABILITY] loop evaluation failed attempt=%d: %s",
                                 session_id, cycle_num, attempt + 1, stab_loop_exc)

            if attempt > 0:
                metrics = {
                    'sharpe': getattr(best_result, 'real_sharpe', 0) or 0,
                    'turnover': getattr(best_result, 'real_turnover', 0) or 0,
                    'margin': getattr(best_result, 'real_margin', 0.05) or 0.05,
                    'fitness': getattr(best_result, 'real_fitness', 0) or 0,
                }
                error_msg = "; ".join(getattr(best_result, 'gate_failures', []) or [])

                diagnosis = classifier.classify(metrics, error_msg)

                _exp_replay_applied = False
                if _exp_replay is not None:
                    try:
                        exp_suggestion = await _exp_replay.get_repair_suggestion(
                            failure_type=diagnosis.failure_type.value,
                            expr=current_expr,
                            metrics_snapshot=metrics,
                            min_confidence=0.7,
                        )
                        if exp_suggestion and exp_suggestion.confidence > 0.7:
                            new_expr = exp_suggestion.suggested_expr
                            if new_expr and new_expr != current_expr and len(new_expr) >= 5:
                                _current_exp_card_id = exp_suggestion.card_id
                                _log(session_id, "EXP_REPLAY_APPLIED",
                                     f"Attempt {attempt + 1}: [EXP_REPLAY] Applied historical repair "
                                     f"card_id={exp_suggestion.card_id[:8]} confidence={exp_suggestion.confidence:.3f} "
                                     f"action={exp_suggestion.suggested_action} → {new_expr[:50]}...",
                                     {"cycle": cycle_num, "attempt": attempt + 1,
                                      "exp_card_id": exp_suggestion.card_id[:8],
                                      "confidence": exp_suggestion.confidence,
                                      "action": exp_suggestion.suggested_action})
                                current_expr = new_expr
                                _exp_replay_applied = True
                                if alpha is not None:
                                    alpha.expression = current_expr
                                    if alpha.simulation_payload is not None:
                                        alpha.simulation_payload["regular"] = current_expr
                    except (OSError, ValueError, RuntimeError) as exc:
                        logger.debug("[DEFENSIVE_LOG] brain_improvement: exp_replay_query_failed attempt=%d error=%s — falling back to segment_mutator", attempt + 1, exc)

                if not _exp_replay_applied:
                    new_expr, diagnosis = mutator.improve(current_expr, metrics, error_msg)

                if new_expr and new_expr != current_expr and len(new_expr) >= 5:
                    _log(session_id, "SEGMENT_LOCKED_MUTATE",
                         f"Attempt {attempt + 1}: [{diagnosis.failure_type.value}] "
                         f"{current_expr[:50]}... → {new_expr[:50]}...",
                         {"cycle": cycle_num, "attempt": attempt + 1,
                          "failure_type": diagnosis.failure_type.value,
                          "action": diagnosis.suggested_action,
                          "confidence": diagnosis.confidence})

                    try:
                        reflexion_result = await reflexion_engine.reflect_and_improve(
                            original_expr=current_expr,
                            mutated_expr=new_expr,
                            failure=diagnosis,
                            max_reflections=getattr(settings, 'REFLEXION_MAX_ROUNDS', 2),
                        )

                        if reflexion_result.passed_proxy:
                            final_expr = reflexion_result.final_expr
                            if final_expr and final_expr != current_expr and len(final_expr) >= 5:
                                _log(session_id, "REFLEXION_IMPROVED",
                                     f"Attempt {attempt + 1}: {reflexion_result.reflection_count} rounds, "
                                     f"score={reflexion_result.final_verdict.overall_score:.2f}, "
                                     f"{new_expr[:40]}... → {final_expr[:40]}...",
                                     {"cycle": cycle_num, "attempt": attempt + 1,
                                      "reflection_count": reflexion_result.reflection_count,
                                      "proxy_score": reflexion_result.final_verdict.overall_score})
                                new_expr = final_expr
                            else:
                                _log(session_id, "REFLEXION_NO_CHANGE",
                                     f"Attempt {attempt + 1}: reflection completed but no effective change",
                                     {"cycle": cycle_num, "attempt": attempt + 1})
                        else:
                            _log(session_id, "REFLEXION_REJECTED",
                                 f"Attempt {attempt + 1}: factor rejected by proxy (score={reflexion_result.final_verdict.overall_score:.2f}), skipping submission",
                                 {"cycle": cycle_num, "attempt": attempt + 1,
                                  "proxy_score": reflexion_result.final_verdict.overall_score,
                                  "reason": reflexion_result.final_verdict.decision_reason[:100]})
                            proxy_evaluator.record_submission(new_expr)
                            continue
                    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, json.JSONDecodeError) as refl_exc:
                        logger.warning("[%s] cycle=%d REFLEXION_FAILED attempt=%d: %s",
                                       session_id, cycle_num, attempt + 1, refl_exc)
                        _log(session_id, "REFLEXION_FALLBACK",
                             f"Attempt {attempt + 1}: reflexion failed, using mutator output directly",
                             {"cycle": cycle_num, "attempt": attempt + 1, "error": str(refl_exc)[:80]})

                    current_expr = new_expr
                    if alpha is not None:
                        alpha.expression = current_expr
                        if alpha.simulation_payload is not None:
                            alpha.simulation_payload["regular"] = current_expr

                _should_run_ea = False
                _ea_trigger_reason = ""

                if _ea_strategy is not None and attempt >= 1:
                    _current_sharpe = getattr(best_result, 'real_sharpe', 0) or 0
                    _current_fitness = getattr(best_result, 'real_fitness', 0) or 0

                    if 0.8 <= _current_sharpe < 1.25:
                        _should_run_ea = True
                        _ea_trigger_reason = f"near_pass(sharpe={_current_sharpe:.3f})"

                    elif attempt >= 2 and _current_sharpe < 1.0:
                        _should_run_ea = True
                        _ea_trigger_reason = f"stuck(sharpe={_current_sharpe:.3f}, attempt={attempt+1})"

                if _should_run_ea and _ea_strategy is not None:
                    _log(session_id, "EA_TRIGGERED",
                         f"🧬 Triggering EA search: reason={_ea_trigger_reason}, seed_expr={current_expr[:60]}...",
                         {"cycle": cycle_num, "attempt": attempt + 1, "trigger_reason": _ea_trigger_reason})

                    try:
                        import asyncio as _asyncio_for_ea

                        _ea_best, _ea_history = await _asyncio_for_ea.wait_for(
                            _ea_strategy.search(
                                seed_expression=current_expr,
                                target_sharpe=1.25,
                                initial_sharpe=_current_sharpe,
                                context={
                                    "session_id": session_id,
                                    "cycle": cycle_num,
                                    "attempt": attempt + 1,
                                    "trigger_reason": _ea_trigger_reason,
                                },
                            ),
                            timeout=_ea_strategy.config.timeout_seconds if hasattr(_ea_strategy, 'config') else 180,
                        )

                        if _ea_best and _ea_best.fitness > _current_sharpe:
                            _ea_new_expr = _ea_best.expression
                            if _ea_new_expr and _ea_new_expr != current_expr and len(_ea_new_expr) >= 5:
                                _log(session_id, "EA_SUCCESS",
                                     f"✅ EA found better solution: Sharpe {_current_sharpe:.3f} → {_ea_best.fitness:.3f}, "
                                     f"gen={_ea_best.generation}, mutation={_ea_best.mutation_type.value}, "
                                     f"expr={_ea_new_expr[:60]}...",
                                     {"cycle": cycle_num, "attempt": attempt + 1,
                                      "ea_fitness": _ea_best.fitness,
                                      "ea_generation": _ea_best.generation,
                                      "ea_mutation_type": _ea_best.mutation_type.value,
                                      "ea_pop_size": len(_ea_history) if _ea_history else 0})

                                current_expr = _ea_new_expr
                                if alpha is not None:
                                    alpha.expression = current_expr
                                    if alpha.simulation_payload is not None:
                                        alpha.simulation_payload["regular"] = current_expr

                                try:
                                    from openalpha_brain.knowledge.graph_experience_db import GraphBasedExperienceDB
                                    _graph_db_local = GraphBasedExperienceDB()
                                    _graph_db_local.load()
                                    _graph_db_local.add_factor_experience(
                                        expression=current_expr,
                                        wq_feedback={
                                            "sharpe": _current_sharpe,
                                            "fitness": _current_fitness,
                                            "turnover": getattr(best_result, 'real_turnover', None),
                                            "checks": getattr(best_result, 'brain_checks', []),
                                        },
                                        improvement_result={
                                            "strategy": "ea_search",
                                            "new_expression": _ea_new_expr,
                                            "result": f"sharpe {_current_sharpe:.3f} -> {_ea_best.fitness:.3f}",
                                            "success": True,
                                        },
                                        category="near_pass" if 0.8 <= _current_sharpe < 1.25 else "improved",
                                    )
                                    _graph_db_local.save()
                                    _log(session_id, "EA_EXPERIENCE_RECORDED",
                                         f"EA result recorded to graph DB (category=near_pass/improved)",
                                         {"cycle": cycle_num, "attempt": attempt + 1})
                                except (OSError, ValueError, RuntimeError) as _graph_exc:
                                    logger.debug("[DEFENSIVE_LOG] EA graph_db recording failed: %s", _graph_exc)
                        else:
                            _log(session_id, "EA_NO_BETTER",
                                 f"EA completed but no better solution found (best_fitness={_ea_best.fitness if _ea_best else 0:.3f}), continuing standard loop",
                                 {"cycle": cycle_num, "attempt": attempt + 1})

                    except asyncio.TimeoutError:
                        _log(session_id, "EA_TIMEOUT",
                             f"⏰ EA search timed out after {getattr(_ea_strategy, 'config', type('', (), {'timeout_seconds': 180})).timeout_seconds}s, falling back",
                             {"cycle": cycle_num, "attempt": attempt + 1})
                    except (ValueError, TypeError, OSError, RuntimeError, KeyError, AttributeError, ImportError) as exc:
                        _log(session_id, "EA_ERROR",
                             f"EA search failed: {str(_ea_exc)[:100]}, continuing standard loop",
                             {"cycle": cycle_num, "attempt": attempt + 1, "error": str(_ea_exc)[:100]})

                _should_run_tot = False
                _tot_trigger_reason = ""

                if _tot_strategy is not None and attempt >= 2:
                    _current_sharpe_tot = getattr(best_result, 'real_sharpe', 0) or 0
                    _current_fitness_tot = getattr(best_result, 'real_fitness', 0) or 0

                    if 0.9 <= _current_sharpe_tot < 1.25 and _current_fitness_tot < 1.0:
                        _should_run_tot = True
                        _tot_trigger_reason = f"deep_search(sharpe={_current_sharpe_tot:.3f}, fitness={_current_fitness_tot:.3f})"

                if _should_run_tot and _tot_strategy is not None:
                    _log(session_id, "TOT_TRIGGERED",
                         f"🌳 Triggering ToT tree search: reason={_tot_trigger_reason}, seed_expr={current_expr[:60]}...",
                         {"cycle": cycle_num, "attempt": attempt + 1, "trigger_reason": _tot_trigger_reason})

                    try:
                        import asyncio as _asyncio_for_tot

                        _tot_result = await _asyncio_for_tot.wait_for(
                            _tot_strategy.search(
                                seed_expression=current_expr,
                                target_fitness=1.0,
                                initial_fitness=_current_fitness_tot,
                                context={
                                    "session_id": session_id,
                                    "cycle": cycle_num,
                                    "attempt": attempt + 1,
                                    "trigger_reason": _tot_trigger_reason,
                                },
                            ),
                            timeout=_tot_strategy.config.timeout_seconds if hasattr(_tot_strategy, 'config') else 240,
                        )

                        _tot_best_expr = _tot_result.get_best_expression()
                        _tot_best_fitness = _tot_result.get_best_fitness()

                        if _tot_best_expr and _tot_best_fitness > _current_fitness_tot:
                            if _tot_best_expr != current_expr and len(_tot_best_expr) >= 5:
                                _log(session_id, "TOT_SUCCESS",
                                     f"✅ ToT found better solution: Fitness {_current_fitness_tot:.3f} → {_tot_best_fitness:.3f}, "
                                     f"depth={_tot_result.total_depth_reached}, nodes={_tot_result.total_nodes_explored}, "
                                     f"expr={_tot_best_expr[:60]}...",
                                     {"cycle": cycle_num, "attempt": attempt + 1,
                                      "tot_fitness": _tot_best_fitness,
                                      "tot_depth": _tot_result.total_depth_reached,
                                      "tot_nodes": _tot_result.total_nodes_explored})

                                current_expr = _tot_best_expr
                                if alpha is not None:
                                    alpha.expression = current_expr
                                    if alpha.simulation_payload is not None:
                                        alpha.simulation_payload["regular"] = current_expr

                                try:
                                    from openalpha_brain.knowledge.graph_experience_db import GraphBasedExperienceDB
                                    _graph_db_tot = GraphBasedExperienceDB()
                                    _graph_db_tot.load()
                                    _graph_db_tot.add_factor_experience(
                                        expression=current_expr,
                                        wq_feedback={
                                            "sharpe": _current_sharpe_tot,
                                            "fitness": _current_fitness_tot,
                                            "turnover": getattr(best_result, 'real_turnover', None),
                                            "checks": getattr(best_result, 'brain_checks', []),
                                        },
                                        improvement_result={
                                            "strategy": "tot_search",
                                            "new_expression": _tot_best_expr,
                                            "result": f"fitness {_current_fitness_tot:.3f} -> {_tot_best_fitness:.3f}",
                                            "success": True,
                                            "tot_depth": _tot_result.total_depth_reached,
                                            "tot_nodes": _tot_result.total_nodes_explored,
                                        },
                                        category="near_pass",
                                    )
                                    _graph_db_tot.save()
                                    _log(session_id, "TOT_EXPERIENCE_RECORDED",
                                         f"ToT result recorded to graph DB",
                                         {"cycle": cycle_num, "attempt": attempt + 1})
                                except (OSError, ValueError, RuntimeError) as _tot_graph_exc:
                                    logger.debug("[DEFENSIVE_LOG] ToT graph_db recording failed: %s", _tot_graph_exc)
                        else:
                            _log(session_id, "TOT_NO_BETTER",
                                 f"ToT completed but no better solution (best_fitness={_tot_best_fitness:.3f}), continuing",
                                 {"cycle": cycle_num, "attempt": attempt + 1})

                    except asyncio.TimeoutError:
                        _log(session_id, "TOT_TIMEOUT",
                             f"⏰ ToT search timed out after {getattr(_tot_strategy, 'config', type('', (), {'timeout_seconds': 240})).timeout_seconds}s, falling back",
                             {"cycle": cycle_num, "attempt": attempt + 1})
                    except (ValueError, TypeError, OSError, RuntimeError, KeyError, AttributeError, ImportError) as _tot_exc:
                        _log(session_id, "TOT_ERROR",
                             f"ToT search failed: {str(_tot_exc)[:100]}, continuing standard loop",
                             {"cycle": cycle_num, "attempt": attempt + 1, "error": str(_tot_exc)[:100]})
                else:
                    _log(session_id, "BRAIN_MUTATE",
                         f"Attempt {attempt + 1}: mutated expr={current_expr[:60]}...",
                         {"cycle": cycle_num, "attempt": attempt + 1})

            _log(session_id, "BRAIN_IMPROVE_ATTEMPT",
                 f"Improvement attempt {attempt + 1}/{max_attempts} for expr={current_expr[:60]}...",
                 {"cycle": cycle_num, "attempt": attempt + 1})

            pf_result = _prefilter.prefilter(current_expr)
            if not pf_result.passed:
                _log(session_id, "PREFILTER_REJECTED",
                     f"PreFilter rejected: {pf_result.reason} (confidence={pf_result.confidence_score:.2f})",
                     {"cycle": cycle_num, "attempt": attempt + 1,
                      "reject_reason": pf_result.reason,
                      "confidence": pf_result.confidence_score})
                continue

            improved_result = await _submit_to_brain(
                alpha=alpha,
                session_id=session_id,
                cycle_num=cycle_num,
            )

            if improved_result is None:
                continue

            proxy_evaluator.record_submission(current_expr)

            _log_brain_result(session_id, improved_result, current_expr, attempt + 1)

            new_sharpe = getattr(improved_result, 'real_sharpe', 0) or 0
            best_sharpe = getattr(best_result, 'real_sharpe', 0) or 0

            if new_sharpe > best_sharpe:
                best_result = improved_result
                _log(session_id, "BRAIN_IMPROVED",
                     f"Improved Sharpe: {best_sharpe:.3f} -> {new_sharpe:.3f}",
                     {"cycle": cycle_num, "attempt": attempt + 1})

            if getattr(improved_result, 'status', None) == BrainSimStatus.PASS:
                _log(session_id, "BRAIN_PASS_EARLY",
                     f"Got PASS on attempt {attempt + 1}, stopping early",
                     {"cycle": cycle_num})
                break

        except (ValueError, TypeError, OSError, RuntimeError, KeyError, AttributeError, ImportError) as exc:
            continue

    if _exp_replay is not None and _current_exp_card_id:
        try:
            final_status = "PASS" if getattr(best_result, 'status', None) == BrainSimStatus.PASS else ("IMPROVED" if (getattr(best_result, 'real_sharpe', 0) or 0) > (getattr(initial_result, 'real_sharpe', 0) or 0) else "FAIL")
            final_metrics_after = {
                'sharpe': getattr(best_result, 'real_sharpe', None),
                'turnover': getattr(best_result, 'real_turnover', None),
                'margin': getattr(best_result, 'real_margin', None),
                'fitness': getattr(best_result, 'real_fitness', None),
            }
            _exp_replay.update_card_outcome(
                card_id=_current_exp_card_id,
                outcome=final_status,
                metrics=final_metrics_after,
            )
            _log(session_id, "EXP_REPLAY_RECORDED",
                 f"Experience replay card updated: card_id={_current_exp_card_id[:8]} outcome={final_status}",
                 {"cycle": cycle_num, "card_id": _current_exp_card_id[:8], "outcome": final_status})
        except (OSError, ValueError, RuntimeError) as exc:
            logger.debug("[DEFENSIVE_LOG] brain_improvement: exp_replay_record_failed error=%s — outcome not recorded", exc)

    if _stability_guard is not None:
        try:
            final_sharpe = getattr(best_result, 'real_sharpe', 0) or 0
            base_reward = final_sharpe
            adjusted_reward = _stability_guard.compute_reward_adjustment(base_reward)
            _log(session_id, "STABILITY_REWARD_ADJUSTMENT",
                 f"Final: base_reward={base_reward:.3f} → adjusted_reward={adjusted_reward:.3f} "
                 f"(Δ={adjusted_reward - base_reward:+.3f})",
                 {"cycle": cycle_num, "stability_score": _stability_guard.get_summary().get("current_stability_score", "N/A")})

            if hasattr(_param_optimizer, '_scheduler') and _param_optimizer._scheduler is not None:
                try:
                    direction = _extract_root_operator(current_expr or expression or initial_expression or "")
                    _param_optimizer._scheduler.update(
                        arm=direction,
                        reward=adjusted_reward,
                    )
                    _log(session_id, "STABILITY_MAB_UPDATE",
                         f"MAB reward updated with stability adjustment | direction={direction} reward={adjusted_reward:.3f}",
                         {"cycle": cycle_num})
                except (OSError, ValueError, RuntimeError) as mab_exc:
                    logger.debug("[%s] cycle=%d [STABILITY] MAB update failed: %s", session_id, cycle_num, mab_exc)

            stab_summary = _stability_guard.get_summary()
            _log(session_id, "STABILITY_SUMMARY",
                 f"Improvement loop stability summary | evaluations={stab_summary['total_evaluations']} "
                 f"unstable_events={stab_summary['unstable_event_count']} "
                 f"instability_rate={stab_summary['instability_rate']:.2%} "
                 f"current_score={stab_summary['current_stability_score']:.2f}",
                 {"cycle": cycle_num})
        except (OSError, ValueError, RuntimeError) as stab_final_exc:
            logger.debug("[%s] cycle=%d [STABILITY] final reward adjustment failed: %s", session_id, cycle_num, stab_final_exc)

    return best_result


async def _run_param_optimization(expression: str, session_id: str, 
                                   cycle_num: int, **kwargs) -> dict | None:
    """Run parameter optimization on an alpha expression.
    
    Args:
        expression: Alpha expression to optimize
        session_id: Session identifier
        cycle_num: Current cycle number
        **kwargs: Additional optimization parameters
    
    Returns:
        Optimization results dict or None if optimization failed/skipped
    """
    if not expression or not expression.strip():
        return None
    
    try:
        expr_hash = expression_hash(expression)
        _log(session_id, "PARAM_OPT_START",
             f"Starting param optimization for expr_hash={expr_hash[:12]}...",
             {"cycle": cycle_num})
        
        # Use parameter optimizer if available
        if _param_optimizer is not None:
            optimized = await _param_optimizer.optimize(
                expression=expression,
                session_id=session_id,
                **kwargs
            )
            if optimized:
                _log(session_id, "PARAM_OPT_COMPLETE",
                     f"Param optimization completed successfully",
                     {"cycle": cycle_num, "expr_hash": expr_hash[:12]})
                return optimized
        
        _log(session_id, "PARAM_OPT_SKIPPED",
             f"Param optimizer not available or returned None",
             {"cycle": cycle_num})
        return None
        
    except (ValueError, TypeError, OSError, RuntimeError, KeyError, AttributeError) as exc:
        logger.warning("[%s] cycle=%d PARAM_OPT_ERROR: %s", session_id, cycle_num, exc)
        return None


class FailureType(Enum):
    HIGH_TURNOVER = "high_turnover"
    LOW_SHARPE = "low_sharpe"
    OVERFIT = "overfit"
    DECAY_FAST = "decay_fast"
    CORRELATION_HIGH = "correlation_high"
    SYNTAX_ERROR = "syntax_error"
    UNKNOWN = "unknown"


@dataclass
class FailureDiagnosis:
    failure_type: FailureType
    confidence: float
    raw_metrics: dict
    suggested_action: str
    reason: str


@dataclass
class ProxyVerdict:
    overall_score: float
    syntax_score: float
    structure_score: float
    field_score: float
    param_score: float
    history_score: float
    syntax_details: str
    structure_details: str
    field_details: str
    param_details: str
    history_details: str
    should_submit: bool
    decision_reason: str


@dataclass
class Reflection:
    round_num: int
    analysis: str
    suggested_fix: str
    target_block: str
    confidence: float


@dataclass
class ReflexionResult:
    final_expr: str
    passed_proxy: bool
    reflection_count: int
    log: list
    final_verdict: ProxyVerdict = None


EPSILON = 1e-6

_ALLOWED_OPERATORS = {
    'ts_rank', 'ts_mean', 'ts_std_dev', 'ts_sum', 'ts_max', 'ts_min',
    'ts_delta', 'ts_corr', 'ts_covariance', 'ts_regression',
    'ts_decay_linear', 'group_neutralize', 'rank', 'signed_power',
    'log', 'abs', 'sign', 'ts_zscore', 'ts_skewness', 'ts_kurtosis',
    'ts_argmax', 'ts_argmin', 'ts_product', 'ts_av_diff', 'ts_percentage',
    'vec_avg', 'vec_sum', 'vec_norm',
}

_HIGH_CORRELATION_FIELD_PAIRS = [
    ('close', 'vwap'),
    ('volume', 'sharesout'),
    ('sales', 'revenue'),
    ('net_income', 'earnings'),
]

_DECAY_WINDOW_RANGE = (5, 30)
_RANK_WINDOW_MIN = 2
_EXTREME_PARAM_THRESHOLD = 100


class ProxyEvaluator:
    def __init__(self, field_proxy_map=None, failure_fix_library=None):
        self._fpm = field_proxy_map
        self._ffl = failure_fix_library
        self._recent_submissions: list[tuple[str, float]] = []
        self._max_recent_history = 50

    @algo_log()
    def evaluate(self, expr: str, context: dict = None) -> ProxyVerdict:
        with Timer("proxy_evaluate"):
            if not expr or not expr.strip():
                return self._build_zero_verdict("表达式为空")

            expr = expr.strip()

            syntax_score, syntax_details = self._eval_syntax_viability(expr)
            structure_score, structure_details = self._eval_structural_compliance(expr)
            field_score, field_details = self._eval_field_plausibility(expr)
            param_score, param_details = self._eval_parameter_sanity(expr)
            history_score, history_details = self._eval_historical_similarity(expr)

            weighted_score = (
                syntax_score * 0.30 +
                structure_score * 0.25 +
                field_score * 0.20 +
                param_score * 0.15 +
                history_score * 0.10
            )

            should_sub, decision_reason = self.should_submit(weighted_score)

            verdict = ProxyVerdict(
                overall_score=weighted_score,
                syntax_score=syntax_score,
                structure_score=structure_score,
                field_score=field_score,
                param_score=param_score,
                history_score=history_score,
                syntax_details=syntax_details,
                structure_details=structure_details,
                field_details=field_details,
                param_details=param_details,
                history_details=history_details,
                should_submit=should_sub,
                decision_reason=decision_reason,
            )

            logger.info("[DEFENSIVE_LOG] proxy_evaluator: expr=%.60s... score=%.3f submit=%s reason=%s",
                       expr[:60], weighted_score, should_sub, decision_reason[:80])
            return verdict

    def _eval_syntax_viability(self, expr: str) -> tuple[float, str]:
        score = 100.0
        issues = []

        try:
            parsed = val.validate_expression(expr, strict=False)
            if not parsed.get('is_valid', False):
                score -= 40
                issues.append(f"AST解析失败")
        except (ValueError, SyntaxError, TypeError) as exc:
            score -= 40
            issues.append(f"AST解析异常: {str(exc)[:50]}")

        tokens = re.findall(r'\b([a-zA-Z_]\w*)\s*\(', expr)
        unknown_ops = [t for t in tokens if t not in _ALLOWED_OPERATORS]
        if unknown_ops:
            penalty = min(30, len(unknown_ops) * 10)
            score -= penalty
            issues.append(f"未知算子({len(unknown_ops)}): {unknown_ops[:3]}")

        depth = 0
        max_depth = 0
        for ch in expr:
            if ch == '(':
                depth += 1
                max_depth = max(max_depth, depth)
            elif ch == ')':
                depth -= 1
        if max_depth > 4:
            penalty = min(20, (max_depth - 4) * 5)
            score -= penalty
            issues.append(f"嵌套深度={max_depth}>4")

        open_count = expr.count('(')
        close_count = expr.count(')')
        if open_count != close_count:
            score -= 25
            issues.append(f"括号不平衡 ({open_count}/{close_count})")

        return max(0, score), "; ".join(issues) if issues else "OK"

    def _eval_structural_compliance(self, expr: str) -> tuple[float, str]:
        score = 100.0
        issues = []

        has_decay = "ts_decay_linear" in expr
        has_neutralize = "group_neutralize" in expr

        if not (has_decay or has_neutralize):
            score -= 15
            issues.append("缺少三段式结构（无decay/neutralize）")
        else:
            if has_decay and has_neutralize:
                decay_match = re.search(r'ts_decay_linear\s*\(', expr)
                neutralize_match = re.search(r'group_neutralize\s*\(', expr)
                if decay_match and neutralize_match:
                    if decay_match.start() < neutralize_match.start():
                        score -= 20
                        issues.append("decay应在外层，当前neutralize在外层")
                else:
                    score -= 10
                    issues.append("三段式解析异常")

            if has_neutralize:
                neut_match = re.search(r'group_neutralize\s*\([^,]+,\s*(\w+)\s*\)', expr)
                if not neut_match:
                    score -= 15
                    issues.append("neutralize格式异常")

        signal_content = expr
        if has_neutralize:
            neut_match = re.search(r'group_neutralize\s*\(\s*(.+?)\s*,\s*\w+\s*\)', expr, re.DOTALL)
            if neut_match:
                signal_content = neut_match.group(1)
        if has_decay:
            decay_match = re.search(r'ts_decay_linear\s*\(\s*(.+?)\s*,\s*\d+\s*\)', expr, re.DOTALL)
            if decay_match:
                signal_content = decay_match.group(1)

        signal_clean = re.sub(r'\b\d+\b', '', signal_content)
        signal_clean = re.sub(r'[+\-*/(),]', '', signal_clean).strip()
        if len(signal_clean) < 5:
            score -= 25
            issues.append("信号段内容过短或接近常量")

        return max(0, score), "; ".join(issues) if issues else "OK"

    def _eval_field_plausibility(self, expr: str) -> tuple[float, str]:
        score = 100.0
        issues = []

        fields_found = re.findall(r'\b([a-z][a-z0-9_]*)\b', expr)
        operators = _ALLOWED_OPERATORS | {'rank', 'group'}
        fields_found = [f for f in fields_found if f not in operators and len(f) > 2]

        if not fields_found:
            score -= 20
            issues.append("未检测到数据字段")
        elif self._fpm and self._fpm._loaded:
            known_fields = 0
            cold_fields = 0
            for fld in fields_found:
                info = self._fpm.get_field_info(fld)
                if info:
                    known_fields += 1
                    if info.get('is_cold', False):
                        cold_fields += 1
                else:
                    pass

            if known_fields == 0:
                score -= 25
                issues.append(f"所有字段({len(fields_found)})均不在FieldProxyMap中")
            else:
                cold_ratio = cold_fields / max(known_fields, 1)
                if cold_ratio > 0.5:
                    penalty = min(20, int(cold_ratio * 30))
                    score -= penalty
                    issues.append(f"冷字段比例过高: {cold_ratio:.0%}")

        for pair in _HIGH_CORRELATION_FIELD_PAIRS:
            if all(p in expr for p in pair):
                score -= 15
                issues.append(f"高相关字段组合: {pair[0]}/{pair[1]}")

        return max(0, score), "; ".join(issues) if issues else "OK"

    def _eval_parameter_sanity(self, expr: str) -> tuple[float, str]:
        score = 100.0
        issues = []

        decay_match = re.search(r'ts_decay_linear\s*\([^,]+,\s*(\d+)\s*\)', expr)
        if decay_match:
            window = int(decay_match.group(1))
            if window < _DECAY_WINDOW_RANGE[0] or window > _DECAY_WINDOW_RANGE[1]:
                penalty = 15
                score -= penalty
                issues.append(f"decay_window={window}超出合理范围[{_DECAY_WINDOW_RANGE[0]},{_DECAY_WINDOW_RANGE[1]}]")
        else:
            pass

        numbers = re.findall(r'\b(\d+)\b', expr)
        for num_str in numbers:
            num = int(num_str)
            if num <= 0:
                score -= 10
                issues.append(f"非正参数: {num}")
            elif num > _EXTREME_PARAM_THRESHOLD:
                penalty = min(15, num // 50)
                score -= penalty
                issues.append(f"极端参数值: {num}")

        rank_matches = re.findall(r'(?:ts_rank|ts_mean|ts_std_dev)\s*\([^,]+,\s*(\d+)\s*\)', expr)
        for rm in rank_matches:
            w = int(rm)
            if w < _RANK_WINDOW_MIN:
                score -= 8
                issues.append(f"回看窗口过小: {w}")

        return max(0, score), "; ".join(issues) if issues else "OK"

    def _eval_historical_similarity(self, expr: str) -> tuple[float, str]:
        score = 100.0
        issues = []

        if not self._recent_submissions:
            return score, "无历史记录"

        expr_hash = expression_hash(expr) if callable(expression_hash) else hash(expr)
        similar_count = 0
        for hist_expr, hist_time in self._recent_submissions[-20:]:
            try:
                hist_hash = expression_hash(hist_expr) if callable(expression_hash) else hash(hist_expr)
                similarity = 1.0 if expr_hash == hist_hash else 0.0
                if similarity > 0.9:
                    similar_count += 1
            except (OSError, ValueError, TypeError):
                pass

        if similar_count > 0:
            penalty = min(40, similar_count * 15)
            score -= penalty
            issues.append(f"与近期提交相似度过高: {similar_count}个")

        return max(0, score), "; ".join(issues) if issues else "OK"

    @algo_log()
    def record_submission(self, expr: str, timestamp: float = None) -> None:
        with Timer("proxy_record_submission"):
            import time as _time
            ts = timestamp or _time.time()
            self._recent_submissions.append((expr, ts))
            if len(self._recent_submissions) > self._max_recent_history:
                self._recent_submissions = self._recent_submissions[-self._max_recent_history:]

    def should_submit(self, score: float, threshold: float = 0.55) -> tuple[bool, str]:
        if score >= 0.75:
            return True, f"高质量因子(score={score:.2f}≥0.75)，直接提交"
        elif score >= threshold:
            return True, f"边界因子(score={score:.2f}∈[{threshold:.2f},0.75))，可提交但需关注"
        else:
            return False, f"低质量因子(score={score:.2f}<{threshold:.2f})，拒绝提交建议改进"

    def _build_zero_verdict(self, reason: str) -> ProxyVerdict:
        return ProxyVerdict(
            overall_score=0.0,
            syntax_score=0.0,
            structure_score=0.0,
            field_score=0.0,
            param_score=0.0,
            history_score=0.0,
            syntax_details=reason,
            structure_details="N/A",
            field_details="N/A",
            param_details="N/A",
            history_details="N/A",
            should_submit=False,
            decision_reason=reason,
        )


class ReflexionEngine:
    def __init__(self, llm_generate_fn=None, proxy_evaluator: ProxyEvaluator = None,
                 failure_fix_library=None):
        self._llm = llm_generate_fn
        self._proxy = proxy_evaluator or ProxyEvaluator()
        self._ffl = failure_fix_library

    @algo_log()
    async def reflect_and_improve(self, original_expr: str, mutated_expr: str,
                                   failure: FailureDiagnosis,
                                   max_reflections: int = 2) -> ReflexionResult:
        with Timer("reflexion_reflect_and_improve"):
            current_expr = mutated_expr
            reflection_log = []

            for round_i in range(max_reflections):
                verdict = self._proxy.evaluate(current_expr)

                if verdict.overall_score >= 0.70:
                    logger.info("[REFLEXION] Round %d: proxy passed (score=%.3f≥0.70), expr=%.60s...",
                               round_i + 1, verdict.overall_score, current_expr[:60])
                    return ReflexionResult(
                        final_expr=current_expr,
                        passed_proxy=True,
                        reflection_count=round_i + 1,
                        log=reflection_log,
                        final_verdict=verdict,
                    )

                logger.info("[REFLEXION] Round %d: proxy score=%.3f<0.70, triggering LLM reflection",
                           round_i + 1, verdict.overall_score)

                try:
                    reflection = await self._llm_reflect(
                        expr=current_expr,
                        verdict=verdict,
                        failure=failure,
                        previous_reflections=reflection_log,
                    )
                except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, json.JSONDecodeError) as exc:
                    logger.warning("[REFLEXION] Round %d: LLM reflect failed: %s", round_i + 1, exc)
                    reflection = Reflection(
                        round_num=round_i + 1,
                        analysis=f"LLM反思失败: {str(exc)[:80]}",
                        suggested_fix="",
                        target_block="A",
                        confidence=0.0,
                    )

                reflection_log.append(reflection)

                if not reflection.suggested_fix or reflection.confidence < 0.3:
                    logger.info("[REFLEXION] Round %d: LLM无有效建议或置信度过低(conf=%.2f)，停止反思",
                               round_i + 1, reflection.confidence)
                    break

                current_expr = self._apply_reflection_suggestion(
                    current_expr, reflection.suggested_fix, reflection.target_block
                )

                if current_expr != mutated_expr:
                    integrity = validate_block_integrity(current_expr)
                    if not integrity.get("valid", True):
                        logger.warning("[REFLEXION] Round %d: 反思后完整性验证失败，回滚到上一版本",
                                      round_i + 1)
                        current_expr = mutated_expr
                        break

            final_verdict = self._proxy.evaluate(current_expr)
            passed = final_verdict.overall_score >= 0.55

            logger.info("[REFLEXION] Completed: %d rounds, final_score=%.3f, passed=%s",
                       len(reflection_log), final_verdict.overall_score, passed)

            return ReflexionResult(
                final_expr=current_expr,
                passed_proxy=passed,
                reflection_count=len(reflection_log),
                log=reflection_log,
                final_verdict=final_verdict,
            )

    async def _llm_reflect(self, expr: str, verdict: ProxyVerdict,
                            failure: FailureDiagnosis,
                            previous_reflections: list) -> Reflection:
        with Timer("reflexion_llm_reflect"):
            system_prompt = """你是 WorldQuant BRAIN 平台的因子审查专家。
你的任务是根据本地代理验证器的评估结果，分析因子表达式的潜在问题，
并提出**仅针对信号段(Block A)**的修改建议。

绝对禁止：
- 修改中性段(Block B/group_neutralize)
- 修改衰减段(Block C/ts_decay_linear)
- 改变整体三段式结构

只允许修改：
- Block A 内的字段选择
- Block A 内的窗口参数
- Block A 内的算子替换

请用中文回答，并以JSON格式返回修改建议。"""

            history_summary = ""
            if previous_reflections:
                history_parts = [f"Round {r.round_num}: {r.analysis[:100]}" for r in previous_reflections[-3:]]
                history_summary = "\n历史反思记录:\n" + "\n".join(history_parts)

            user_prompt = f"""当前因子表达式: {expr}

代理验证器评估结果:
- 总分: {verdict.overall_score:.1f}/100
- 语法存活率: {verdict.syntax_score:.1f}/100 (详情: {verdict.syntax_details})
- 结构合规性: {verdict.structure_score:.1f}/100 (详情: {verdict.structure_details})
- 字段合理性: {verdict.field_score:.1f}/100 (详情: {verdict.field_details})
- 参数合理性: {verdict.param_score:.1f}/100 (详情: {verdict.param_details})
- 历史相似度: {verdict.history_score:.1f}/100 (详情: {verdict.history_details})

原始失败原因: {failure.failure_type.value} - {failure.reason}
{history_summary}

请分析该因子最主要的问题是什么（优先关注得分最低的维度），
并提出一个具体的信号段修改方案。

JSON格式返回:
{{
  "analysis": "问题分析（100字以内）",
  "suggested_fix": "具体的Block A修改后的完整表达式片段",
  "target_block": "A",
  "confidence": 0.0-1.0
}}"""

            try:
                if self._llm is None:
                    raise RuntimeError("LLM generate function not configured")

                response = await self._llm(user_prompt, [], "", session_id="reflexion_engine", cycle=0)

                import json as _json
                cleaned = _extract_json_from_llm(response) if callable(_extract_json_from_llm) else response
                result = _json.loads(cleaned) if isinstance(cleaned, str) else cleaned

                return Reflection(
                    round_num=len(previous_reflections) + 1,
                    analysis=result.get("analysis", "无分析"),
                    suggested_fix=result.get("suggested_fix", ""),
                    target_block=result.get("target_block", "A"),
                    confidence=max(0.0, min(1.0, float(result.get("confidence", 0.5)))),
                )
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                logger.error("[DEFENSIVE_LOG] reflexion_engine: llm_reflect_parse_failed error=%s — returning fallback reflection", exc)
                return Reflection(
                    round_num=len(previous_reflections) + 1,
                    analysis=f"解析失败: {str(exc)[:100]}",
                    suggested_fix="",
                    target_block="A",
                    confidence=0.0,
                )

    def _apply_reflection_suggestion(self, original_expr: str, suggestion: str,
                                     target_block: str = "A") -> str:
        with Timer("reflexion_apply_suggestion"):
            if not suggestion or not suggestion.strip():
                return original_expr

            suggestion = suggestion.strip()

            if target_block.upper() != "A":
                logger.warning("[REFLEXION] 非法目标段: %s，仅允许修改Block A", target_block)
                return original_expr

            mutator = SegmentLockedMutator()
            blocks = mutator.parse_blocks(original_expr)

            if not blocks.get("structure_valid", False):
                logger.warning("[REFLEXION] 无法解析表达式结构，保持原样")
                return original_expr

            blocks["block_a"] = suggestion
            new_expr = mutator.reassemble(blocks)

            integrity = validate_block_integrity(new_expr)
            if integrity.get("valid", False):
                log_call("reflexion_applied",
                        extra={"old": original_expr[:50], "new": new_expr[:50]})
                return new_expr
            else:
                logger.warning("[REFLEXION] 应用建议后完整性验证失败: %s",
                              integrity.get("warnings", []))
                return original_expr


class FailureClassifier:
    def __init__(self):
        self._rules = [
            (lambda m: m.get('turnover', 0) > 0.35, FailureType.HIGH_TURNOVER, 0.9, "adjust_C"),
            (lambda m: m.get('sharpe', 0) < 0.3 and m.get('turnover', 0) < 0.25, FailureType.LOW_SHARPE, 0.85, "mutate_A"),
            (lambda m: m.get('margin', 0) < 0.02, FailureType.OVERFIT, 0.7, "simplify_A"),
            (lambda m: m.get('decay_rate', 1.0) < 0.5, FailureType.DECAY_FAST, 0.8, "adjust_C"),
            (lambda m: m.get('correlation', 0) > 0.85, FailureType.CORRELATION_HIGH, 0.75, "mutate_A"),
        ]

    @algo_log()
    def classify(self, metrics: dict, error_msg: str = "") -> FailureDiagnosis:
        with Timer("failure_classify"):
            for condition, ftype, confidence, action in self._rules:
                if condition(metrics):
                    return FailureDiagnosis(
                        failure_type=ftype,
                        confidence=confidence,
                        raw_metrics=metrics.copy(),
                        suggested_action=action,
                        reason=self._build_reason(ftype, metrics),
                    )
            if error_msg and ("syntax" in error_msg.lower() or "parse" in error_msg.lower()):
                return FailureDiagnosis(
                    failure_type=FailureType.SYNTAX_ERROR,
                    confidence=0.95,
                    raw_metrics=metrics.copy(),
                    suggested_action="resubmit",
                    reason=f"语法错误: {error_msg[:100]}",
                )
            return FailureDiagnosis(
                failure_type=FailureType.UNKNOWN,
                confidence=0.3,
                raw_metrics=metrics.copy(),
                suggested_action="mutate_A",
                reason="无法分类，保守处理：变异 Block A",
            )

    def _build_reason(self, ftype: FailureType, metrics: dict) -> str:
        reasons = {
            FailureType.HIGH_TURNOVER: f"换手率={metrics.get('turnover', 0):.2%}超标(>35%)",
            FailureType.LOW_SHARPE: f"Sharpe={metrics.get('sharpe', 0):.3f}过低(<0.3)，换手率正常",
            FailureType.OVERFIT: f"边际收益={metrics.get('margin', 0):.4f}过低(<2%)，疑似过拟合",
            FailureType.DECAY_FAST: f"衰减率={metrics.get('decay_rate', 1.0):.2f}过快(<50%)",
            FailureType.CORRELATION_HIGH: f"相关性={metrics.get('correlation', 0):.2%}过高(>85%)",
        }
        return reasons.get(ftype, "未知原因")


class SegmentLockedMutator:
    def __init__(self, llm_generate_fn=None):
        self._llm = llm_generate_fn

    @algo_log()
    def parse_blocks(self, expression: str) -> dict:
        with Timer("parse_blocks"):
            result = {
                "block_a": "",
                "block_b": "",
                "block_c": "",
                "structure_valid": False,
            }
            if not expression or not expression.strip():
                return result

            expr = expression.strip()

            has_decay = "ts_decay_linear" in expr
            has_neutralize = "group_neutralize" in expr

            if has_decay and has_neutralize:
                decay_match = re.match(
                    r'ts_decay_linear\s*\(\s*(.+?)\s*,\s*(\d+)\s*\)',
                    expr,
                    re.DOTALL,
                )
                if decay_match:
                    inner_expr = decay_match.group(1)
                    decay_window = decay_match.group(2)
                    result["block_c"] = f"ts_decay_linear(..., {decay_window})"

                    neutralize_match = re.match(
                        r'group_neutralize\s*\(\s*(.+?)\s*,\s*(\w+)\s*\)',
                        inner_expr,
                        re.DOTALL,
                    )
                    if neutralize_match:
                        signal_expr = neutralize_match.group(1)
                        group_field = neutralize_match.group(2)
                        result["block_b"] = f"group_neutralize(..., {group_field})"
                        result["block_a"] = signal_expr
                        result["structure_valid"] = True
                    else:
                        result["block_a"] = inner_expr
                        result["block_b"] = ""
                        result["structure_valid"] = True
            elif has_neutralize:
                neutralize_match = re.match(
                    r'group_neutralize\s*\(\s*(.+?)\s*,\s*(\w+)\s*\)',
                    expr,
                    re.DOTALL,
                )
                if neutralize_match:
                    result["block_a"] = neutralize_match.group(1)
                    result["block_b"] = f"group_neutralize(..., {neutralize_match.group(2)})"
                    result["structure_valid"] = True
            else:
                result["block_a"] = expr
                result["structure_valid"] = True

            log_call("blocks_parsed", extra={k: v[:40] if isinstance(v, str) else v for k, v in result.items()})
            return result

    @algo_log()
    def mutate_block_a(self, block_a: str, failure: FailureDiagnosis, field_proxy_map=None) -> str:
        with Timer("mutate_block_a"):
            if not block_a:
                return block_a

            if failure.failure_type == FailureType.LOW_SHARPE:
                return self._apply_low_sharpe_mutation(block_a, field_proxy_map)
            elif failure.failure_type == FailureType.OVERFIT:
                return self._apply_overfit_simplification(block_a)
            elif failure.failure_type == FailureType.CORRELATION_HIGH:
                return self._apply_correlation_mutation(block_a, field_proxy_map)
            else:
                return self._generic_mutation(block_a)

    def _apply_low_sharpe_mutation(self, block_a: str, field_proxy_map=None) -> str:
        import random
        numbers = re.findall(r'\b(\d+)\b', block_a)
        if numbers:
            target = random.choice(numbers)
            old_val = int(target)
            new_val = max(2, old_val + random.choice([-3, -2, 2, 3, 5]))
            mutated = block_a.replace(target, str(new_val), 1)
            log_call("low_sharpe_mutate", extra={"window_change": f"{old_val}->{new_val}"})
            return mutated

        for op, alts in _OPERATOR_SWAP_MAP.items():
            if op in block_a and alts:
                alt = random.choice(alts)
                mutated = block_a.replace(op, alt, 1)
                log_call("low_sharpe_mutate", operator_swap=f"{op}->{alt}")
                return mutated

        return block_a

    def _apply_overfit_simplification(self, block_a: str) -> str:
        depth = 0
        max_depth = 0
        for ch in block_a:
            if ch == '(':
                depth += 1
                max_depth = max(max_depth, depth)
            elif ch == ')':
                depth -= 1

        if max_depth > 4:
            outer_match = re.match(r'(\w+)\s*\((.+)\)$', block_a, re.DOTALL)
            if outer_match:
                simplified = outer_match.group(2)
                log_call("overfit_simplify", removed_outer=outer_match.group(1), old_depth=max_depth)
                return simplified

        import random
        numbers = re.findall(r'\b(\d+)\b', block_a)
        if len(numbers) > 2:
            target = random.choice(numbers[1:])
            new_val = max(5, int(target) * 2)
            mutated = block_a.replace(target, str(new_val), 1)
            log_call("overfit_simplify", extra={"window_expand": f"{target}->{new_val}"})
            return mutated

        return block_a

    def _apply_correlation_mutation(self, block_a: str, field_proxy_map=None) -> str:
        import random
        for fld, alts in _FIELD_SWAP_MAP.items():
            pattern = re.compile(re.escape(fld), re.IGNORECASE)
            if pattern.search(block_a) and alts:
                alt = random.choice(alts)
                mutated = pattern.sub(alt, block_a, count=1)
                log_call("correlation_mutate", field_swap=f"{fld}->{alt}")
                return mutated

        for op, alts in _OPERATOR_SWAP_MAP.items():
            if op in block_a and alts:
                alt = random.choice(alts)
                mutated = block_a.replace(op, alt, 1)
                log_call("correlation_mutate", extra={"operator_swap": f"{op}->{alt}"})
                return mutated

        return block_a

    def _generic_mutation(self, block_a: str) -> str:
        import random
        numbers = re.findall(r'\b(\d+)\b', block_a)
        if numbers:
            target = random.choice(numbers)
            old_val = int(target)
            new_val = max(2, old_val + random.choice([-2, 2, 3]))
            return block_a.replace(target, str(new_val), 1)

        for op, alts in _OPERATOR_SWAP_MAP.items():
            if op in block_a and alts:
                alt = random.choice(alts)
                return block_a.replace(op, alt, 1)

        return block_a

    @algo_log()
    def adjust_block_c(self, block_c: str, failure: FailureDiagnosis) -> str:
        with Timer("adjust_block_c"):
            if not block_c or "ts_decay_linear" not in block_c:
                return block_c

            window_match = re.search(r'(\d+)', block_c)
            if not window_match:
                return block_c

            current_window = int(window_match.group(1))

            if failure.failure_type in (FailureType.HIGH_TURNOVER, FailureType.DECAY_FAST):
                if current_window < 20:
                    new_window = min(60, current_window * 2)
                else:
                    new_window = min(60, current_window + 10)
            else:
                new_window = min(60, current_window + 5)

            new_block_c = re.sub(r'\d+', str(new_window), block_c, count=1)
            log_call("block_c_adjusted", extra={"old_window": current_window, "new_window": new_window, "reason": failure.failure_type.value})
            return new_block_c

    @algo_log()
    def reassemble(self, blocks: dict) -> str:
        with Timer("reassemble"):
            block_a = blocks.get("block_a", "")
            block_b = blocks.get("block_b", "")
            block_c = blocks.get("block_c", "")

            if not block_a:
                return ""

            if block_b and block_c:
                inner_b = block_b.replace("group_neutralize(...,", "group_neutralize(").replace("...", block_a, 1)
                full_c = block_c.replace("ts_decay_linear(...,", "ts_decay_linear(").replace("...", inner_b, 1)
                return full_c
            elif block_b:
                assembled = block_b.replace("group_neutralize(...,", "group_neutralize(").replace("...", block_a, 1)
                return assembled
            elif block_c:
                assembled = block_c.replace("ts_decay_linear(...,", "ts_decay_linear(").replace("...", block_a, 1)
                return assembled
            else:
                return block_a

    @algo_log()
    def improve(self, expression: str, metrics: dict, error_msg: str = "") -> tuple:
        with Timer("segment_locked_improve"):
            classifier = FailureClassifier()
            diagnosis = classifier.classify(metrics, error_msg)

            log_call("failure_classified", extra={"type": diagnosis.failure_type.value, "action": diagnosis.suggested_action, "confidence": diagnosis.confidence})

            blocks = self.parse_blocks(expression)

            if diagnosis.suggested_action == "adjust_C":
                if blocks.get("block_c"):
                    blocks["block_c"] = self.adjust_block_c(blocks["block_c"], diagnosis)
                    log_call("block_c_adjusted", extra={"non_llm_mutation": True})
                else:
                    log_call("block_c_missing", extra={"fallback_to": "mutate_A"})
                    blocks["block_a"] = self.mutate_block_a(blocks["block_a"], diagnosis)

            elif diagnosis.suggested_action in ("mutate_A", "simplify_A"):
                blocks["block_a"] = self.mutate_block_a(blocks["block_a"], diagnosis)
                log_call("block_a_mutated", extra={"llm_used": False})

            elif diagnosis.suggested_action == "resubmit":
                log_call("syntax_error_resubmit", extra={"no_mutation": True})

            else:
                logger.warning("未知失败类型: %s，跳过改进", diagnosis.failure_type.value)
                return expression, diagnosis

            new_expression = self.reassemble(blocks)

            integrity = validate_block_integrity(new_expression)
            if not integrity.get("valid"):
                logger.warning("段锁重组后完整性验证失败: %s", integrity.get("warnings", []))
                log_call("integrity_check_failed", warnings=integrity.get("warnings", []))

            return new_expression, diagnosis


def validate_block_integrity(expr: str) -> dict:
    result = {
        "valid": True,
        "has_block_b": False,
        "has_block_c": False,
        "block_a_complexity": 0,
        "warnings": [],
    }

    if not expr or not expr.strip():
        result["valid"] = False
        result["warnings"].append("表达式为空")
        return result

    expr = expr.strip()

    if "group_neutralize" in expr:
        result["has_block_b"] = True
    if "ts_decay_linear" in expr:
        result["has_block_c"] = True

    depth = 0
    max_depth = 0
    for ch in expr:
        if ch == '(':
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ')':
            depth -= 1
    result["block_a_complexity"] = max_depth

    open_count = expr.count('(')
    close_count = expr.count(')')
    if open_count != close_count:
        result["valid"] = False
        result["warnings"].append(f"括号不匹配: ({open_count} vs ){close_count}")

    if max_depth > 8:
        result["warnings"].append(f"嵌套深度过高: {max_depth}")

    log_call("integrity_validated", extra=result)
    return result


@dataclass
class PreFilterResult:
    passed: bool
    reason: str
    confidence_score: float


class SignalQualityPreFilter:
    """
    提交前预筛选 — 在 ProxyEvaluator 之前运行 (Enhanced v2)
    目的: 拦截明显没有 alpha 信号的噪音表达式

    基于 5 层增强检查:
    Layer 1: 算子组合黑名单 — 检测 trivial/拥挤模式并拒绝
    Layer 2: 复杂度门槛提升 — ≥5个算子才通过，最佳范围5-12
    Layer 3: WQ 格式预校验 — lookback参数、字符串字面量、group_neutralize格式
    Layer 4: 字段多样性增强 — 禁止纯price/vol字段、鼓励跨数据族
    Layer 5: 结构指纹去重 — AST拓扑相似度检测，拒绝重复结构
    """

    _CROWDED_FIELDS = {'close', 'open', 'high', 'low', 'vwap', 'volume'}
    _CORE_PRICE_VOL_FIELDS = {'close', 'open', 'high', 'low', 'vwap', 'volume'}
    _INVALID_OPERATOR_COMBINATIONS = [
        (r'rank\s*\(\s*rank\s*\(', "rank(rank(x)) 无效嵌套"),
        (r'ts_rank\s*\(\s*ts_rank\s*\(', "ts_rank(ts_rank(x)) 无效嵌套"),
        (r'log\s*\(\s*abs\s*\(', "log(abs(x)) 可能导致数值问题"),
        (r'sign\s*\(\s*sign\s*\(', "sign(sign(x)) 冗余操作"),
    ]
    _EXTREME_PARAMS = [
        (r'ts_decay_linear\s*\([^,]+,\s*(1)\s*\)', "decay_window=1 过小"),
        (r'(?:ts_rank|ts_mean|ts_delta|ts_av_diff)\s*\([^,]+,\s*(0|1)\s*\)', "回看窗口<=1 无意义"),
    ]
    _OPERATOR_BLACKLIST_PATTERNS = [
        (r'^\s*-?\s*ts_delta\s*\(\s*(?:close|open|high|low|vwap)\s*,\s*\d+\s*\)\s*$',
         "ts_delta(price) 单独使用 - 过度拥挤信号"),
        (r'^\s*-?\s*ts_delta\s*\(\s*volume\s*,\s*\d+\s*\)\s*$',
         "ts_delta(volume) 单独使用 - 过度拥挤信号"),
        (r'rank\s*\(\s*ts_mean\s*\([^)]+\)\s*\)',
         "rank(ts_mean(x)) 结构过于简单 - 仅2个算子"),
        (r'group_neutralize\s*\(\s*rank\s*\(\s*[a-zA-Z_]\w*\s*\)\s*,\s*\w+\s*\)',
         "group_neutralize(rank(单一基础字段)) - 字段太简单"),
        (r'rank\s*\([a-zA-Z_]\w*\)\s*\+\s*rank\s*\([a-zA-Z_]\w*\)',
         "纯 Additive 结构 rank(A)+rank(B) - Anti-Crowding 规则"),
        (r'^(?!.*(?:ts_decay_linear|ts_av_diff)).*$',
         "缺少 ts_decay_linear/ts_av_diff 的纯动量表达式 - 换手控制不足"),
    ]

    def __init__(self):
        self._recent_expressions: list[str] = []
        self._max_history = 50
        self._failed_topologies: list[str] = []
        self._max_failed_topology_history = 10
        try:
            from openalpha_brain.knowledge.operator_registry import get_operator_registry
            self._op_reg = get_operator_registry()
            logger.info("[DEFENSIVE_LOG] SignalQualityPreFilter: OperatorRegistry initialized with %d operators", len(self._op_reg.get_all_operator_names()))
        except (ImportError, OSError, ValueError) as exc:
            self._op_reg = None

    def prefilter(self, expr: str, context: dict = None) -> PreFilterResult:
        if not expr or not expr.strip():
            return PreFilterResult(
                passed=False,
                reason="表达式为空",
                confidence_score=1.0,
            )

        expr = expr.strip()

        layer1_result = self._check_operator_blacklist(expr)
        if not layer1_result.passed:
            logger.info("[DEFENSIVE_LOG] SignalQualityPreFilter [Layer1]: 算子黑名单未通过 - %s", layer1_result.reason)
            self._record_failed_topology(expr)
            return layer1_result

        layer2_result = self._check_complexity(expr)
        if not layer2_result.passed:
            logger.info("[DEFENSIVE_LOG] SignalQualityPreFilter [Layer2]: 复杂度检查未通过 - %s", layer2_result.reason)
            self._record_failed_topology(expr)
            return layer2_result

        layer3_result = self._check_wq_format(expr)
        if not layer3_result.passed:
            logger.info("[DEFENSIVE_LOG] SignalQualityPreFilter [Layer3]: WQ格式预校验未通过 - %s", layer3_result.reason)
            self._record_failed_topology(expr)
            return layer3_result

        layer4_result = self._check_field_diversity(expr)
        if not layer4_result.passed:
            logger.info("[DEFENSIVE_LOG] SignalQualityPreFilter [Layer4]: 字段多样性检查未通过 - %s", layer4_result.reason)
            self._record_failed_topology(expr)
            return layer4_result

        layer5_result = self._check_topology_dedup(expr)
        if not layer5_result.passed:
            logger.info("[DEFENSIVE_LOG] SignalQualityPreFilter [Layer5]: 结构指纹去重未通过 - %s", layer5_result.reason)
            return layer5_result

        operator_result = self._check_operator_combinations(expr)
        if not operator_result.passed:
            logger.info("[DEFENSIVE_LOG] SignalQualityPreFilter: 算子组合检查未通过 - %s", operator_result.reason)
            return operator_result

        param_result = self._check_parameter_sanity(expr)
        if not param_result.passed:
            logger.info("[DEFENSIVE_LOG] SignalQualityPreFilter: 参数合理性检查未通过 - %s", param_result.reason)
            return param_result

        history_result = self._check_historical_similarity(expr)
        if not history_result.passed:
            logger.info("[DEFENSIVE_LOG] SignalQualityPreFilter: 历史相似度检查未通过 - %s", history_result.reason)
            return history_result

        final_confidence = min(layer2_result.confidence_score, layer3_result.confidence_score,
                               layer4_result.confidence_score, layer5_result.confidence_score)
        logger.debug("[DEFENSIVE_LOG] SignalQualityPreFilter: 表达式通过所有预筛选检查 (confidence=%.2f)", final_confidence)
        return PreFilterResult(
            passed=True,
            reason="通过所有增强预筛选",
            confidence_score=final_confidence,
        )

    def _check_complexity(self, expr: str) -> PreFilterResult:
        if self._op_reg is not None:
            operator_count = self._op_reg.count_operators(expr)
            logger.debug("[DEFENSIVE_LOG] SignalQualityPreFilter: Using OperatorRegistry.count_operators() = %d", operator_count)
        else:
            operators = re.findall(r'\b([a-zA-Z_]\w*)\s*\(', expr)
            operator_count = len(operators)

        depth = 0
        max_depth = 0
        for ch in expr:
            if ch == '(':
                depth += 1
                max_depth = max(max_depth, depth)
            elif ch == ')':
                depth -= 1

        if operator_count < 5:
            return PreFilterResult(
                passed=False,
                reason=f"表达式过于简单（仅{operator_count}个算子<5），缺乏信号复杂度",
                confidence_score=0.92,
            )

        if operator_count > 15:
            logger.warning("[DEFENSIVE_LOG] SignalQualityPreFilter: 算子数过多（%d>15），可能过拟合", operator_count)

        if max_depth > 8:
            return PreFilterResult(
                passed=False,
                reason=f"嵌套深度过高（{max_depth}层>8），可能过拟合",
                confidence_score=0.85,
            )

        if 5 <= operator_count <= 12:
            confidence = 0.88
        else:
            confidence = 0.78

        return PreFilterResult(
            passed=True,
            reason=f"复杂度正常（{operator_count}个算子，{max_depth}层嵌套）",
            confidence_score=confidence,
        )

    def _check_field_diversity(self, expr: str) -> PreFilterResult:
        field_pattern = re.compile(
            r'\b(close|open|high|low|vwap|volume|adv\d+|returns|cap|sales|'
            r'assets|equity|revenue|earnings|sharesout|debt|enterprise_value)\b',
            re.IGNORECASE,
        )
        fields_found = set(f.lower() for f in field_pattern.findall(expr))

        if not fields_found:
            return PreFilterResult(
                passed=False,
                reason="未检测到任何数据字段",
                confidence_score=0.95,
            )

        core_price_vol_only = fields_found.issubset(self._CORE_PRICE_VOL_FIELDS)
        if core_price_vol_only and len(fields_found) <= 2:
            return PreFilterResult(
                passed=False,
                reason=f"仅使用核心价格/成交量字段（{sorted(fields_found)}≤2个），信号过于拥挤",
                confidence_score=0.88,
            )

        crowded_only = fields_found.issubset(self._CROWDED_FIELDS)
        if crowded_only and len(fields_found) >= 2:
            return PreFilterResult(
                passed=False,
                reason=f"仅使用拥挤字段 {sorted(fields_found)}，信号可能过度挖掘",
                confidence_score=0.75,
            )

        has_fundamental = bool(fields_found & {'cap', 'earnings', 'sales', 'assets',
                                               'revenue', 'sharesout', 'debt', 'enterprise_value'})
        if has_fundamental:
            return PreFilterResult(
                passed=True,
                reason=f"包含基本面字段，跨数据族使用良好",
                confidence_score=0.88,
            )

        if core_price_vol_only:
            return PreFilterResult(
                passed=True,
                reason=f"仅包含价格/成交量字段 - 降低置信度",
                confidence_score=0.50,
            )

        return PreFilterResult(
            passed=True,
            reason=f"字段组合可接受: {sorted(fields_found)}",
            confidence_score=0.72,
        )

    def _check_operator_blacklist(self, expr: str) -> PreFilterResult:
        if self._op_reg is not None:
            patterns = self._op_reg.get_forbidden_patterns()
            logger.debug("[DEFENSIVE_LOG] SignalQualityPreFilter: Using OperatorRegistry forbidden patterns (count=%d)", len(patterns))
            for pattern, reason in patterns:
                if re.search(pattern, expr, re.IGNORECASE | re.MULTILINE):
                    return PreFilterResult(
                        passed=False,
                        reason=f"[黑名单] {reason}",
                        confidence_score=0.93,
                    )
        for pattern, reason in self._OPERATOR_BLACKLIST_PATTERNS:
            if re.search(pattern, expr, re.IGNORECASE | re.MULTILINE):
                return PreFilterResult(
                    passed=False,
                    reason=f"[黑名单] {reason}",
                    confidence_score=0.93,
                )

        has_rank = bool(re.search(r'\brank\s*\(', expr))
        has_negation = bool(re.search(r'^\s*-|-\s*rank\s*\(\s*-', expr))
        if not has_rank:
            return PreFilterResult(
                passed=False,
                reason="缺少 rank() 外层包裹 - 参考过审因子都使用 rank() 归一化",
                confidence_score=0.85,
            )

        if not has_negation:
            logger.debug("[DEFENSIVE_LOG] SignalQualityPreFilter: 未检测到 -1 反转信号（可选但推荐）")

        return PreFilterResult(
            passed=True,
            reason="算子组合通过黑名单检查",
            confidence_score=0.86,
        )

    def _check_wq_format(self, expr: str) -> PreFilterResult:
        string_literals = re.findall(r'"([^"]*)"', expr)
        if string_literals:
            return PreFilterResult(
                passed=False,
                reason=f"包含字符串字面量 {string_literals} - WQ 不支持字符串参数",
                confidence_score=0.95,
            )

        ts_operators = re.finditer(
            r'(?:ts_decay_linear|ts_mean|ts_rank|ts_delta|ts_av_diff|ts_std_dev|ts_corr)\s*\(([^)]+)',
            expr,
        )
        for match in ts_operators:
            params_str = match.group(1)
            param_parts = [p.strip() for p in params_str.split(',')]
            if len(param_parts) >= 2:
                lookback_param = param_parts[-1].strip()
                if not re.match(r'^-?\d+$', lookback_param):
                    return PreFilterResult(
                        passed=False,
                        reason=f"{match.group(0)[:30]}... 的 lookback 参数 '{lookback_param}' 不是整数",
                        confidence_score=0.94,
                    )
                lookback_val = int(lookback_param)
                if lookback_val <= 0:
                    return PreFilterResult(
                        passed=False,
                        reason=f"{match.group(0)[:30]}... 的 lookback={lookback_val} 必须为正整数",
                        confidence_score=0.92,
                    )

        group_neutralize_match = re.search(r'group_neutralize\s*\(([^)]+)\)', expr)
        if group_neutralize_match:
            params = group_neutralize_match.group(1)
            param_count = len([p.strip() for p in params.split(',') if p.strip()])
            if param_count < 2:
                return PreFilterResult(
                    passed=False,
                    reason="group_neutralize 缺少第二个参数（需要 industry/subindustry/sector）",
                    confidence_score=0.91,
                )

        return PreFilterResult(
            passed=True,
            reason="WQ 格式预校验通过",
            confidence_score=0.87,
        )

    def _extract_topology_signature(self, expr: str) -> str:
        stack = []
        signature_chars = []
        for ch in expr:
            if ch == '(':
                stack.append('(')
                signature_chars.append('(')
            elif ch == ')':
                if stack:
                    stack.pop()
                signature_chars.append(')')
        return ''.join(signature_chars)

    def _check_topology_dedup(self, expr: str) -> PreFilterResult:
        current_topology = self._extract_topology_signature(expr)

        if len(self._failed_topologies) < 3:
            return PreFilterResult(
                passed=True,
                reason="失败拓扑样本不足，跳过去重检查",
                confidence_score=0.60,
            )

        for failed_topo in self._failed_topologies[-10:]:
            if current_topology == failed_topo:
                return PreFilterResult(
                    passed=False,
                    reason=f"与最近失败表达式拓扑结构完全相同，拒绝重复提交",
                    confidence_score=0.90,
                )

        similarity_count = sum(
            1 for failed_topo in self._failed_topologies[-10:]
            if self._calculate_topology_similarity(current_topology, failed_topo) > 0.85
        )
        if similarity_count >= 3:
            return PreFilterResult(
                passed=False,
                reason=f"与{similarity_count}个最近失败表达式拓扑相似度>85%，可能存在结构性缺陷",
                confidence_score=0.82,
            )

        return PreFilterResult(
            passed=True,
            reason=f"拓扑结构独特（与最近10个失败因子无高度相似）",
            confidence_score=0.80,
        )

    @staticmethod
    def _calculate_topology_similarity(topo1: str, topo2: str) -> float:
        if not topo1 or not topo2:
            return 0.0
        max_len = max(len(topo1), len(topo2))
        if max_len == 0:
            return 1.0
        matches = sum(1 for a, b in zip(topo1, topo2) if a == b)
        return matches / max_len

    def _record_failed_topology(self, expr: str) -> None:
        topology = self._extract_topology_signature(expr)
        self._failed_topologies.append(topology)
        if len(self._failed_topologies) > self._max_failed_topology_history:
            self._failed_topologies = self._failed_topologies[-self._max_failed_topology_history:]

    def _check_operator_combinations(self, expr: str) -> PreFilterResult:
        for pattern, reason in self._INVALID_OPERATOR_COMBINATIONS:
            if re.search(pattern, expr, re.IGNORECASE):
                return PreFilterResult(
                    passed=False,
                    reason=f"无效算子组合: {reason}",
                    confidence_score=0.92,
                )

        return PreFilterResult(
            passed=True,
            reason="算子组合有效",
            confidence_score=0.88,
        )

    def _check_parameter_sanity(self, expr: str) -> PreFilterResult:
        for pattern, reason in self._EXTREME_PARAMS:
            if re.search(pattern, expr):
                return PreFilterResult(
                    passed=False,
                    reason=f"极端参数值: {reason}",
                    confidence_score=0.90,
                )

        numbers = re.findall(r'\b(\d+)\b', expr)
        for num_str in numbers:
            num = int(num_str)
            if num == 0 and 'threshold' in expr.lower():
                return PreFilterResult(
                    passed=False,
                    reason="rank_threshold=0 无区分能力",
                    confidence_score=0.88,
                )
            if num > 500:
                return PreFilterResult(
                    passed=False,
                    reason=f"参数过大({num}>500)，可能导致计算问题",
                    confidence_score=0.78,
                )

        return PreFilterResult(
            passed=True,
            reason="参数范围合理",
            confidence_score=0.83,
        )

    def _check_historical_similarity(self, expr: str) -> PreFilterResult:
        if len(self._recent_expressions) < 5:
            return PreFilterResult(
                passed=True,
                reason="历史样本不足，跳过相似度检查",
                confidence_score=0.60,
            )

        try:
            expr_hash_val = expression_hash(expr) if callable(expression_hash) else hash(expr)
        except (OSError, ValueError, TypeError):
            expr_hash_val = hash(expr)

        similar_count = 0
        max_similarity = 0.0

        for hist_expr in self._recent_expressions[-50:]:
            try:
                hist_hash = expression_hash(hist_expr) if callable(expression_hash) else hash(hist_expr)
                similarity = 1.0 if expr_hash_val == hist_hash else 0.0
                max_similarity = max(max_similarity, similarity)
                if similarity > 0.90:
                    similar_count += 1
            except (OSError, ValueError, TypeError):
                pass

        if similar_count > 0 or max_similarity > 0.90:
            return PreFilterResult(
                passed=False,
                reason=f"与近期提交高度相似（最大相似度={max_similarity:.1%}, "
                       f"重复数={similar_count}）",
                confidence_score=0.87,
            )

        return PreFilterResult(
            passed=True,
            reason=f"与历史因子差异足够（最大相似度={max_similarity:.1%}）",
            confidence_score=0.75,
        )

    def record_submission(self, expr: str) -> None:
        """记录已提交的表达式用于后续相似度检查"""
        if expr and isinstance(expr, str):
            self._recent_expressions.append(expr)
            if len(self._recent_expressions) > self._max_history:
                self._recent_expressions = self._recent_expressions[-self._max_history:]

