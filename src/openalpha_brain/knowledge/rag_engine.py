"""
OpenAlpha-Brain — RAG Retrieval Engine
Loads three separate vector stores (operators, fields, financial logic),
retrieves relevant context based on exploration direction, and assembles
dynamic context for LLM injection.

Also contains Success-Failure Dual RAG knowledge base (CoSTEER core mechanism):
- SuccessCaseLibrary: stores BRAIN PASS alpha cases with vector search
- FailureFixLibrary: stores BRAIN FAIL cases + fix attempts
- auto_debug_loop: CoSTEER iterative debug-and-fix mechanism

Experience Replay System (Success-Based Case Retrieval for Code Repair):
- ExperienceCard: single experience record with input features, repair actions, outcomes
- FactorContext: factor-aware context for targeted RAG retrieval
- ExperienceReplayManager: multi-dimensional weighted similarity matching + LLM judgment
"""

from __future__ import annotations

import copy
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np

from openalpha_brain.knowledge.vector_index import VectorStore
from openalpha_brain.utils.algo_logger import Timer, algo_log

logger = logging.getLogger(__name__)

from openalpha_brain.data import VEC_STORE_DIR as _DEFAULT_VEC_DIR

_CACHE_TTL = 300
FIELD_RETRIEVAL_OVERFETCH_FACTOR = 2
FEEDBACK_BOOST_STANDARD = 1.3
FEEDBACK_BOOST_ALTERNATIVE = 1.5

_budget_tracker = None


def set_budget_tracker(tracker) -> None:
    global _budget_tracker
    _budget_tracker = tracker


class RAGEngine:
    def __init__(
        self,
        vec_dir: str | Path | None = None,
        embed_fn=None,
        top_k_ops: int = 15,
        top_k_fields: int = 40,
        top_k_finlogic: int = 5,
    ) -> None:
        self._vec_dir = Path(vec_dir) if vec_dir else _DEFAULT_VEC_DIR
        self._embed_fn = embed_fn
        self._top_k_ops = top_k_ops
        self._top_k_fields = top_k_fields
        self._top_k_finlogic = top_k_finlogic
        self._ops_store: VectorStore | None = None
        self._fields_store: VectorStore | None = None
        self._finlogic_store: VectorStore | None = None
        self._cache: dict[str, tuple[float, dict]] = {}
        self._eliminated_fields: set[str] = set()
        self._feedback_weights: dict[str, dict[str, float]] = {}
        self._whitelist_manager = None

    def load_indexes(self) -> None:
        ops_path = self._vec_dir / "vec_operators.json"
        fields_path = self._vec_dir / "vec_fields.json"
        finlogic_path = self._vec_dir / "vec_finlogic.json"
        if ops_path.exists():
            try:
                self._ops_store = VectorStore.load_index(ops_path)
            except (OSError, FileNotFoundError) as exc:
                logger.error("Failed to load operator vector index '%s': %s", ops_path, exc)
        else:
            logger.warning("Operator vector index not found: %s", ops_path)
        if fields_path.exists():
            try:
                self._fields_store = VectorStore.load_index(fields_path)
            except (OSError, FileNotFoundError) as exc:
                logger.error("Failed to load field vector index '%s': %s", fields_path, exc)
        else:
            logger.warning("Field vector index not found: %s", fields_path)
        if finlogic_path.exists():
            try:
                self._finlogic_store = VectorStore.load_index(finlogic_path)
            except (OSError, FileNotFoundError) as exc:
                logger.error("Failed to load financial logic vector index '%s': %s", finlogic_path, exc)
        else:
            logger.warning("Financial logic vector index not found: %s", finlogic_path)

    @property
    def is_ready(self) -> bool:
        return self._ops_store is not None and self._fields_store is not None

    def set_eliminated_fields(self, fields: set[str]) -> None:
        self._eliminated_fields = fields

    async def _embed(self, text: str) -> np.ndarray:
        if self._embed_fn is not None:
            result = await self._embed_fn(text)
            if isinstance(result, list):
                result = np.array(result, dtype=np.float32)
            return result
        raise RuntimeError("No embedding function configured. Call set_embed_fn() first.")

    def set_embed_fn(self, fn) -> None:
        self._embed_fn = fn

    def set_whitelist_manager(self, wm) -> None:
        self._whitelist_manager = wm

    async def retrieve(
        self,
        exploration_direction: str,
        top_k_ops: int | None = None,
        top_k_fields: int | None = None,
        factor_context: FactorContext | None = None,
    ) -> dict[str, Any]:
        """升级版检索：支持因子上下文定向匹配

        当提供 factor_context 时，会启用定向检索模式：
        1. 如果有 failure_type，优先检索相关的失败修复案例
        2. 将经验回放结果与通用检索结果融合
        3. 返回增强的检索结果

        Args:
            exploration_direction: 探索方向描述
            top_k_ops: 运算子返回数量
            top_k_fields: 字段返回数量
            factor_context: 因子感知上下文（新增）

        Returns:
            检索结果字典，包含 operators、fields、financial_logic 和可选的 experience_replay
        """
        cache_key = exploration_direction.lower().strip()
        if factor_context and factor_context.failure_type:
            cache_key = f"{cache_key}::targeted::{factor_context.failure_type}"

        now = time.time()
        if cache_key in self._cache:
            cached_time, cached_result = self._cache[cache_key]
            if now - cached_time < _CACHE_TTL:
                logger.debug("RAG cache hit for direction: %s", exploration_direction)
                return cached_result

        if _budget_tracker is not None and not _budget_tracker.can_search():
            logger.info("RAG budget exceeded for direction: %s, returning cached/empty", exploration_direction)
            logger.warning(
                "[DEFENSIVE] rag_engine: budget_exhausted direction=%s — request silently dropped, returning empty result",  # noqa: E501
                exploration_direction,
            )
            if cache_key in self._cache:
                _, cached = self._cache[cache_key]
                return cached
            logger.warning(
                "[DEFENSIVE] rag_engine: no_cache_fallback direction=%s — returning completely empty result (0 ops, 0 fields, 0 finlogic)",  # noqa: E501
                exploration_direction,
            )
            return {
                "direction": exploration_direction,
                "operators": [],
                "fields": [],
                "financial_logic": [],
            }

        k_ops = top_k_ops or self._top_k_ops
        k_fields = top_k_fields or self._top_k_fields

        try:
            query_vec = await self._embed(exploration_direction)
        except (TimeoutError, aiohttp.ClientError, ValueError, json.JSONDecodeError) as exc:
            logger.error("RAG retrieve: embedding failed for direction '%s': %s", exploration_direction, exc)
            if cache_key in self._cache:
                _, cached = self._cache[cache_key]
                return cached
            return {
                "direction": exploration_direction,
                "operators": [],
                "fields": [],
                "financial_logic": [],
            }

        ops_results = []
        if self._ops_store and self._ops_store.count > 0:
            try:
                raw = self._ops_store.query(query_vec, top_k=k_ops)
                ops_results = [{"id": did, "score": score, "meta": meta} for did, score, meta in raw]
            except (ValueError, TypeError, OSError) as exc:
                logger.error(
                    "RAG retrieve: operator store query failed for direction '%s': %s", exploration_direction, exc
                )

        fields_results = []
        if self._fields_store and self._fields_store.count > 0:
            try:
                raw = self._fields_store.query(
                    query_vec, top_k=k_fields * FIELD_RETRIEVAL_OVERFETCH_FACTOR, exclude_ids=self._eliminated_fields
                )
                fields_results = [
                    {"id": did, "score": score, "meta": meta}
                    for did, score, meta in raw
                    if did not in self._eliminated_fields
                ][:k_fields]
            except (ValueError, TypeError, OSError) as exc:
                logger.error(
                    "RAG retrieve: fields store query failed for direction '%s': %s", exploration_direction, exc
                )

        finlogic_results = []
        if self._finlogic_store and self._finlogic_store.count > 0:
            try:
                raw = self._finlogic_store.query(query_vec, top_k=self._top_k_finlogic)
                finlogic_results = [{"id": did, "score": score, "meta": meta} for did, score, meta in raw]
            except (ValueError, TypeError, OSError) as exc:
                logger.error(
                    "RAG retrieve: finlogic store query failed for direction '%s': %s", exploration_direction, exc
                )

        result = {
            "direction": exploration_direction,
            "operators": ops_results,
            "fields": fields_results,
            "financial_logic": finlogic_results,
        }

        experience_replay_results = None
        if (
            factor_context
            and factor_context.failure_type
            and hasattr(self, "_experience_replay")
            and self._experience_replay is not None
        ):
            try:
                experience_replay_results = await self._experience_replay.get_repair_suggestion(
                    failure_type=factor_context.failure_type,
                    expr=factor_context.current_expression or "",
                    metrics_snapshot=factor_context.metrics_snapshot,
                    min_confidence=0.5,
                )
                if experience_replay_results:
                    result["experience_replay"] = {
                        "card_id": experience_replay_results.card_id,
                        "confidence": experience_replay_results.confidence,
                        "suggested_action": experience_replay_results.suggested_action,
                        "suggested_target": experience_replay_results.suggested_target,
                        "suggested_expr": experience_replay_results.suggested_expr,
                        "reasoning": experience_replay_results.reasoning,
                        "historical_success_rate": experience_replay_results.historical_success_rate,
                        "similar_cases": experience_replay_results.similar_cases,
                    }
                    logger.info(
                        "[RAG] targeted_retrieval enabled failure_type=%s expReplayConfidence=%.3f",
                        factor_context.failure_type,
                        experience_replay_results.confidence,
                    )
            except (ValueError, TypeError, OSError, RuntimeError) as exc:
                logger.warning(
                    "[DEFENSIVE_LOG] rag_engine: targeted_retrieval_failed failure_type=%s error=%s — using generic results only",  # noqa: E501
                    factor_context.failure_type,
                    exc,
                )

        try:
            self._cache[cache_key] = (now, copy.deepcopy(result))
        except (ValueError, TypeError, OSError) as exc:
            logger.error("RAG retrieve: cache deepcopy failed for direction '%s': %s", exploration_direction, exc)

        try:
            result = self._rerank_with_feedback(result, exploration_direction)
        except (ValueError, TypeError, OSError) as exc:
            logger.error("RAG retrieve: rerank failed for direction '%s': %s", exploration_direction, exc)
            logger.warning(
                "[DEFENSIVE] rag_engine: rerank_fallback direction=%s error=%s — using original ranking without feedback boost",  # noqa: E501
                exploration_direction,
                exc,
            )
        logger.info(
            "RAG retrieve '%s': %d ops, %d fields, %d finlogic%s",
            exploration_direction,
            len(ops_results),
            len(fields_results),
            len(finlogic_results),
            f", {len(experience_replay_results.similar_cases)} exp_replay cases" if experience_replay_results else "",
        )
        if len(ops_results) == 0 and len(fields_results) == 0:
            logger.warning(
                "[DEFENSIVE] rag_engine: empty_retrieval direction=%s — no operators or fields retrieved (vector stores may be unloaded or query mismatch)",  # noqa: E501
                exploration_direction,
            )
        elif len(ops_results) == 0:
            logger.warning(
                "[DEFENSIVE] rag_engine: empty_ops direction=%s — operator store returned 0 results (store may be empty or embedding mismatch)",  # noqa: E501
                exploration_direction,
            )
        elif len(fields_results) == 0:
            logger.warning(
                "[DEFENSIVE] rag_engine: empty_fields direction=%s — field store returned 0 results (store may be empty or all fields eliminated)",  # noqa: E501
                exploration_direction,
            )
        return result

    def set_experience_replay(self, exp_replay_manager: ExperienceReplayManager) -> None:
        """设置经验回放管理器实例

        Args:
            exp_replay_manager: ExperienceReplayManager 实例
        """
        self._experience_replay = exp_replay_manager
        logger.info("[RAG] experience_replay_manager attached to RAGEngine")

    def assemble_context(self, retrieval: dict[str, Any]) -> dict[str, Any]:
        ops = retrieval.get("operators", [])
        fields = retrieval.get("fields", [])
        finlogic = retrieval.get("financial_logic", [])

        top_ops_detailed = ops[: self._top_k_ops]
        all_op_names = [o["id"] for o in ops]
        detailed_op_names = {o["id"] for o in top_ops_detailed}
        remaining_op_names = [n for n in all_op_names if n not in detailed_op_names]

        field_ids = [f["id"] for f in fields]

        finlogic_texts = [fl["id"] for fl in finlogic]

        return {
            "exploration_direction": retrieval.get("direction", ""),
            "top_ops_detailed": [
                {
                    "name": o["id"],
                    "category": o["meta"].get("category", ""),
                    "definition": o["meta"].get("definition", ""),
                    "description": o["meta"].get("description", ""),
                }
                for o in top_ops_detailed
            ],
            "remaining_op_names": remaining_op_names,
            "field_ids": field_ids,
            "financial_logic_ids": finlogic_texts,
        }

    def update_weights_from_feedback(self, direction: str, brain_checks: list[dict]) -> None:
        weights = self._feedback_weights.setdefault(direction, {})
        for chk in brain_checks:
            name = chk.get("name", "") if isinstance(chk, dict) else str(chk)
            result = chk.get("result", "FAIL") if isinstance(chk, dict) else "FAIL"
            if result != "FAIL":
                continue
            if "SHARPE" in name.upper():
                weights["ts_ops_boost"] = weights.get("ts_ops_boost", 1.0) * FEEDBACK_BOOST_STANDARD
            elif "TURNOVER" in name.upper() and "LOW" not in name.upper():
                weights["smoothing_ops_boost"] = weights.get("smoothing_ops_boost", 1.0) * FEEDBACK_BOOST_STANDARD
            elif "TURNOVER" in name.upper() and "LOW" in name.upper():
                weights["volatility_ops_boost"] = weights.get("volatility_ops_boost", 1.0) * FEEDBACK_BOOST_STANDARD
            elif "CORRELATION" in name.upper():
                weights["alternative_ops_boost"] = (
                    weights.get("alternative_ops_boost", 1.0) * FEEDBACK_BOOST_ALTERNATIVE
                )
            elif "FITNESS" in name.upper():
                weights["normalization_ops_boost"] = (
                    weights.get("normalization_ops_boost", 1.0) * FEEDBACK_BOOST_STANDARD
                )
            elif "CONCENTRATED" in name.upper():
                weights["diversification_ops_boost"] = (
                    weights.get("diversification_ops_boost", 1.0) * FEEDBACK_BOOST_STANDARD
                )

    def _rerank_with_feedback(self, retrieval_result: dict, direction: str) -> dict:
        weights = self._feedback_weights.get(direction, {})
        if not weights:
            return retrieval_result
        top_ops = retrieval_result.get("top_ops_detailed", [])
        if top_ops:
            for op in top_ops:
                name = op.get("name", "")
                cat = op.get("category", "").lower()
                boost = 1.0
                if "ts_ops_boost" in weights and "ts_" in name:
                    boost *= weights["ts_ops_boost"]
                if "smoothing_ops_boost" in weights and "decay" in name.lower():
                    boost *= weights["smoothing_ops_boost"]
                if "alternative_ops_boost" in weights:
                    boost *= weights["alternative_ops_boost"]
                if "normalization_ops_boost" in weights and cat in ("normalization", "ranking"):
                    boost *= weights["normalization_ops_boost"]
                op["_feedback_boost"] = boost
            top_ops.sort(key=lambda x: x.get("_feedback_boost", 1.0), reverse=True)
        fields = retrieval_result.get("fields", [])
        if fields and self._whitelist_manager is not None:
            for f in fields:
                fid = f.get("id", "")
                arm = self._whitelist_manager.solidified_fields.get(fid)
                if arm is not None:
                    f["_beta_expectation"] = arm.expectation
                    f["_final_score"] = f.get("score", 0) * (1.0 + arm.expectation)
                else:
                    f["_beta_expectation"] = 0.5
                    f["_final_score"] = f.get("score", 0)
            fields.sort(key=lambda x: x.get("_final_score", 0), reverse=True)
            retrieval_result["fields"] = fields
        return retrieval_result

    def save_feedback_weights(self, path: str) -> None:
        try:
            import json

            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._feedback_weights, f, indent=2)
        except OSError as exc:
            logger.warning(
                "[DEFENSIVE] rag_engine: save_feedback_weights_failed path=%s error=%s — weights not persisted, will be lost on restart",  # noqa: E501
                path,
                exc,
            )

    def load_feedback_weights(self, path: str) -> None:
        try:
            import json
            from pathlib import Path

            p = Path(path)
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    self._feedback_weights = json.load(f)
        except (OSError, FileNotFoundError, json.JSONDecodeError) as exc:
            logger.warning(
                "[DEFENSIVE] rag_engine: load_feedback_weights_failed path=%s error=%s — using default empty weights",
                path,
                exc,
            )

    def health_check(self) -> dict[str, Any]:
        return {
            "module": "RAGEngine",
            "status": "active" if self.is_ready else "not_ready",
            "cache_size": len(self._cache),
            "ops_store_loaded": self._ops_store is not None,
            "fields_store_loaded": self._fields_store is not None,
            "finlogic_store_loaded": self._finlogic_store is not None,
        }

    def clear_cache(self) -> None:
        self._cache.clear()


@dataclass
class ExperienceCard:
    """单条经验卡片 — 记录一次完整的失败→修复→结果循环

    Attributes:
        card_id: 唯一标识符
        timestamp: 记录时间
        failure_type: 失败类型 (HIGH_TURNOVER/LOW_SHARPE/...)
        expression_structure: AST 结构指纹 (如 "rank-decay-neutralize")
        field_family: 涉及的字段族 (如 "price_volume")
        metrics_snapshot: 失败时的指标快照 {sharpe, turnover, ...}
        repair_action: 具体做了什么 (如 "decay_window 10→20")
        repair_target: 修改了哪个段 ("block_c" / "block_a")
        before_expr: 修复前表达式
        after_expr: 修复后表达式
        outcome: 结果 ("PASS" / "FAIL" / "IMPROVED")
        improvement_delta: 指标改善量 {sharpe: +0.3, turnover: -0.15}
        source_loop_id: 来源循环轮次
        usage_count: 被检索使用次数
        success_rate: 基于后续使用的成功率
    """

    card_id: str = ""
    timestamp: str = ""
    failure_type: str = ""
    expression_structure: str = ""
    field_family: str = ""
    metrics_snapshot: dict = field(default_factory=dict)
    repair_action: str = ""
    repair_target: str = ""
    before_expr: str = ""
    after_expr: str = ""
    outcome: str = ""
    improvement_delta: dict = field(default_factory=dict)
    source_loop_id: int = 0
    usage_count: int = 0
    success_rate: float = 0.0


@dataclass
class FactorContext:
    """因子感知上下文 — 用于 RAG 定向检索

    Attributes:
        current_expression: 当前因子表达式
        failure_type: 当前失败类型（如果有）
        field_family_in_use: 当前使用的字段族
        recent_history: 最近 N 次提交历史
        metrics_snapshot: 当前指标快照
    """

    current_expression: str = ""
    failure_type: str | None = None
    field_family_in_use: str = ""
    recent_history: list = field(default_factory=list)
    metrics_snapshot: dict = field(default_factory=dict)


@dataclass
class RepairSuggestion:
    """修复建议 — 由 ExperienceReplayManager 生成

    Attributes:
        card_id: 来源经验卡片 ID
        confidence: 置信度 (0~1)
        suggested_action: 建议的修复动作
        suggested_target: 建议修改的目标段
        suggested_expr: 建议的表达式
        reasoning: 推理依据
        historical_success_rate: 历史成功率
        similar_cases: 相似案例列表
    """

    card_id: str = ""
    confidence: float = 0.0
    suggested_action: str = ""
    suggested_target: str = ""
    suggested_expr: str = ""
    reasoning: str = ""
    historical_success_rate: float = 0.0
    similar_cases: list = field(default_factory=list)


class SuccessCaseLibrary:
    _PERSIST_PATH = Path(__file__).resolve().parent / "success_cases.json"

    def __init__(self, embed_fn: Callable[..., Awaitable] | None = None) -> None:
        self._embed_fn = embed_fn
        self._store = VectorStore()
        self._cases: list[dict] = []

    def set_embed_fn(self, fn: Callable[..., Awaitable]) -> None:
        self._embed_fn = fn

    async def _embed(self, text: str) -> np.ndarray:
        if self._embed_fn is None:
            raise RuntimeError("No embedding function configured for SuccessCaseLibrary")
        result = await self._embed_fn(text)
        if isinstance(result, list):
            result = np.array(result, dtype=np.float32)
        return result

    async def add_case(
        self,
        expr: str,
        hypothesis: str,
        sharpe: float,
        fitness: float,
        turnover: float,
        direction: str,
        session_id: str,
    ) -> None:
        case_id = str(uuid.uuid4())
        case = {
            "id": case_id,
            "expr": expr,
            "hypothesis": hypothesis,
            "sharpe": sharpe,
            "fitness": fitness,
            "turnover": turnover,
            "direction": direction,
            "session_id": session_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        try:
            vec = await self._embed(f"{hypothesis} {expr}")
            self._store.add_documents(
                ids=[case_id],
                vectors=[vec],
                metas=[{k: v for k, v in case.items() if k != "id"}],
            )
        except (TimeoutError, aiohttp.ClientError, ValueError, json.JSONDecodeError):
            logger.warning("SuccessCaseLibrary: embedding failed for case %s", case_id)
        self._cases.append(case)
        self._persist()

    async def search_similar(self, query: str, top_k: int = 3) -> list[dict]:
        if self._store.count == 0:
            logger.warning(
                "[DEFENSIVE] success_case_library: empty_store query=%s — no cases available for search", query
            )
            return []
        try:
            vec = await self._embed(query)
            raw = self._store.query(vec, top_k=top_k)
            return [{"id": did, "score": score, **meta} for did, score, meta in raw]
        except (TimeoutError, aiohttp.ClientError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("SuccessCaseLibrary: search_similar failed")
            logger.warning(
                "[DEFENSIVE] success_case_library: search_similar_failed query=%s error=%s — returning empty results",
                query,
                exc,
            )
            return []

    def get_recent(self, n: int = 5) -> list[dict]:
        return list(self._cases[-n:])

    def _persist(self) -> None:
        try:
            data = {
                "cases": self._cases,
                "store_path": str(self._PERSIST_PATH.with_suffix(".vec.json")),
            }
            self._PERSIST_PATH.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
            if self._store.count > 0:
                self._store.save_index(self._PERSIST_PATH.with_suffix(".vec.json"))
        except OSError:
            logger.warning("SuccessCaseLibrary: persist failed")

    def load(self) -> None:
        try:
            if self._PERSIST_PATH.exists():
                data = json.loads(self._PERSIST_PATH.read_text(encoding="utf-8"))
                self._cases = data.get("cases", [])
                vec_path = Path(data.get("store_path", str(self._PERSIST_PATH.with_suffix(".vec.json"))))
                if vec_path.exists():
                    self._store = VectorStore.load_index(vec_path)
        except (OSError, FileNotFoundError, json.JSONDecodeError):
            logger.warning("SuccessCaseLibrary: load failed")
            self._cases = []


class FailureFixLibrary:
    from openalpha_brain.data import get_data_path

    _PERSIST_PATH = get_data_path("failure_fixes.json")

    def __init__(self, embed_fn: Callable[..., Awaitable] | None = None) -> None:
        self._embed_fn = embed_fn
        self._store = VectorStore()
        self._failures: list[dict] = []

    def set_embed_fn(self, fn: Callable[..., Awaitable]) -> None:
        self._embed_fn = fn

    async def _embed(self, text: str) -> np.ndarray:
        if self._embed_fn is None:
            raise RuntimeError("No embedding function configured for FailureFixLibrary")
        result = await self._embed_fn(text)
        if isinstance(result, list):
            result = np.array(result, dtype=np.float32)
        return result

    async def add_failure(
        self,
        expr: str,
        failure_type: str,
        fix_attempt: str | None,
        fix_success: bool,
        direction: str,
        session_id: str,
    ) -> None:
        failure_id = str(uuid.uuid4())
        failure = {
            "id": failure_id,
            "expr": expr,
            "failure_type": failure_type,
            "fix_attempt": fix_attempt,
            "fix_success": fix_success,
            "direction": direction,
            "session_id": session_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        try:
            vec = await self._embed(f"{failure_type} {expr}")
            self._store.add_documents(
                ids=[failure_id],
                vectors=[vec],
                metas=[{k: v for k, v in failure.items() if k != "id"}],
            )
        except (TimeoutError, aiohttp.ClientError, ValueError, json.JSONDecodeError):
            logger.warning("FailureFixLibrary: embedding failed for failure %s", failure_id)
        self._failures.append(failure)
        self._persist()

    async def search_fix(self, failure_type: str, top_k: int = 3) -> list[dict]:
        if self._store.count == 0:
            logger.warning(
                "[DEFENSIVE] failure_fix_library: empty_store failure_type=%s — no failures available for fix search",
                failure_type,
            )
            return []
        try:
            vec = await self._embed(failure_type)
            raw = self._store.query(vec, top_k=top_k * 2)
            results = []
            for did, score, meta in raw:
                if meta.get("fix_success"):
                    results.append({"id": did, "score": score, **meta})
                if len(results) >= top_k:
                    break
            if not results:
                results = [{"id": did, "score": score, **meta} for did, score, meta in raw[:top_k]]
            return results
        except (TimeoutError, aiohttp.ClientError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("FailureFixLibrary: search_fix failed")
            logger.warning(
                "[DEFENSIVE] failure_fix_library: search_fix_failed failure_type=%s error=%s — returning empty results",
                failure_type,
                exc,
            )
            return []

    async def search_by_expr_similarity(self, query_expr: str, top_k: int = 3) -> list[dict]:
        if self._store.count == 0:
            logger.warning(
                "[DEFENSIVE] failure_fix_library: empty_store query_expr=%s — no failures available for similarity search",  # noqa: E501
                query_expr[:50],
            )
            return []
        try:
            vec = await self._embed(query_expr)
            raw = self._store.query(vec, top_k=top_k)
            return [{"id": did, "score": score, **meta} for did, score, meta in raw]
        except (TimeoutError, aiohttp.ClientError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("FailureFixLibrary: search_by_expr_similarity failed")
            logger.warning(
                "[DEFENSIVE] failure_fix_library: search_by_expr_similarity_failed query=%s error=%s — returning empty results",  # noqa: E501
                query_expr[:50],
                exc,
            )
            return []

    def _persist(self) -> None:
        try:
            data = {
                "failures": self._failures,
                "store_path": str(self._PERSIST_PATH.with_suffix(".vec.json")),
            }
            self._PERSIST_PATH.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
            if self._store.count > 0:
                self._store.save_index(self._PERSIST_PATH.with_suffix(".vec.json"))
        except OSError:
            logger.warning("FailureFixLibrary: persist failed")

    def load(self) -> None:
        try:
            if self._PERSIST_PATH.exists():
                data = json.loads(self._PERSIST_PATH.read_text(encoding="utf-8"))
                self._failures = data.get("failures", [])
                vec_path = Path(data.get("store_path", str(self._PERSIST_PATH.with_suffix(".vec.json"))))
                if vec_path.exists():
                    self._store = VectorStore.load_index(vec_path)
        except (OSError, FileNotFoundError, json.JSONDecodeError):
            logger.warning("FailureFixLibrary: load failed")
            self._failures = []


class ExperienceReplayManager:
    """经验回放管理器 — 基于 Success-Based Case Retrieval 的智能修复建议系统

    核心能力：
    1. record(): 记录每次修复经验（失败→修复动作→结果）
    2. query_similar(): 多维度加权融合检索相似历史案例
    3. get_repair_suggestion(): 结合向量检索 + LLM 判断生成修复建议
    4. update_card_outcome(): 根据后续反馈更新卡片成功率
    5. persist/load(): JSON 持久化

    匹配维度权重配置：
    - failure_type 精确匹配 (weight 0.35)
    - expression_structure 语义相似 (weight 0.25)
    - metrics_snapshot 范围相似 (weight 0.20)
    - field_family 匹配 (weight 0.20)
    """

    _DEFAULT_PERSIST_PATH = Path(__file__).resolve().parent.parent.parent / "experience_cards.json"

    def __init__(self, storage_path: str | Path | None = None, llm_client=None):
        self._storage_path = Path(storage_path) if storage_path else self._DEFAULT_PERSIST_PATH
        self._cards: list[ExperienceCard] = []
        self._llm = llm_client
        self._embed_fn = None

        self._matching_weights = {
            "failure_type": 0.35,
            "expression_structure": 0.25,
            "metrics_snapshot": 0.20,
            "field_family": 0.20,
        }

        self.load()

    def set_embed_fn(self, fn: Callable[..., Awaitable]) -> None:
        """设置嵌入函数用于向量化"""
        self._embed_fn = fn

    def set_llm_client(self, llm_client) -> None:
        """设置 LLM 客户端"""
        self._llm = llm_client

    @algo_log()
    def record(
        self,
        failure_type: str,
        before_expr: str,
        repair_action: str,
        after_expr: str,
        repair_target: str = "",
        outcome: str = "UNKNOWN",
        expression_structure: str = "",
        field_family: str = "",
        metrics_before: dict | None = None,
        metrics_after: dict | None = None,
        source_loop_id: int = 0,
    ) -> ExperienceCard:
        """记录一次修复经验

        Args:
            failure_type: 失败类型 (HIGH_TURNOVER/LOW_SHARPE/...)
            before_expr: 修复前表达式
            repair_action: 具体修复动作 (如 "decay_window 10→20")
            after_expr: 修复后表达式
            repair_target: 修改的目标段 ("block_c"/"block_a")
            outcome: 结果 ("PASS"/"FAIL"/"IMPROVED")
            expression_structure: AST 结构指纹
            field_family: 字段族
            metrics_before: 修复前指标
            metrics_after: 修复后指标
            source_loop_id: 来源循环轮次

        Returns:
            创建的 ExperienceCard 实例
        """
        with Timer("exp_replay_record"):
            card = ExperienceCard(
                card_id=str(uuid.uuid4()),
                timestamp=datetime.now(UTC).isoformat(),
                failure_type=failure_type,
                expression_structure=expression_structure or self._extract_structure_fingerprint(before_expr),
                field_family=field_family or self._extract_field_family(before_expr),
                metrics_snapshot=metrics_before or {},
                repair_action=repair_action,
                repair_target=repair_target,
                before_expr=before_expr,
                after_expr=after_expr,
                outcome=outcome,
                improvement_delta=self._compute_improvement_delta(metrics_before, metrics_after),
                source_loop_id=source_loop_id,
                usage_count=0,
                success_rate=1.0 if outcome == "PASS" else 0.0,
            )

            self._cards.append(card)

            logger.info(
                "[EXP_REPLAY] recorded card_id=%s failure_type=%s action=%s outcome=%s",
                card.card_id[:8],
                failure_type,
                repair_action,
                outcome,
            )

            if len(self._cards) % 10 == 0:
                self.persist()

            return card

    @algo_log()
    async def query_similar(
        self, failure_type: str, expr: str, metrics_snapshot: dict | None = None, top_k: int = 3
    ) -> list[ExperienceCard]:
        """核心方法：给定当前失败，检索历史上相似的修复经验

        匹配维度（加权融合）：
        - failure_type 精确匹配 (weight 0.35)
        - expression_structure 语义相似 (weight 0.25)
        - metrics_snapshot 范围相似 (weight 0.20)
        - field_family 匹配 (weight 0.20)

        Args:
            failure_type: 当前失败类型
            expr: 当前因子表达式
            metrics_snapshot: 当前指标快照
            top_k: 返回最相似的 K 个案例

        Returns:
            按相似度排序的 ExperienceCard 列表
        """
        with Timer("exp_replay_query_similar"):
            if not self._cards:
                logger.warning(
                    "[DEFENSIVE_LOG] exp_replay: empty_library failure_type=%s expr=%s — no experience cards available",
                    failure_type,
                    expr[:50],
                )
                return []

            current_structure = self._extract_structure_fingerprint(expr)
            current_field_family = self._extract_field_family(expr)
            current_metrics = metrics_snapshot or {}

            scored_cards: list[tuple[ExperienceCard, float]] = []

            for card in self._cards:
                try:
                    score = self._compute_similarity_score(
                        card=card,
                        target_failure_type=failure_type,
                        target_structure=current_structure,
                        target_field_family=current_field_family,
                        target_metrics=current_metrics,
                    )
                    scored_cards.append((card, score))
                except (ValueError, TypeError, OSError, RuntimeError) as exc:
                    logger.debug(
                        "[DEFENSIVE_LOG] exp_replay: score_computation_failed card_id=%s error=%s",
                        card.card_id[:8],
                        exc,
                    )
                    continue

            scored_cards.sort(key=lambda x: x[1], reverse=True)
            top_results = scored_cards[:top_k]

            for card, score in top_results:
                card.usage_count += 1
                logger.info(
                    "[EXP_REPLAY] similar_found card_id=%s score=%.3f action=%s success_rate=%.2f",
                    card.card_id[:8],
                    score,
                    card.repair_action,
                    card.success_rate,
                )

            return [card for card, _ in top_results]

    @algo_log()
    async def get_repair_suggestion(
        self, failure_type: str, expr: str, metrics_snapshot: dict | None = None, min_confidence: float = 0.6
    ) -> RepairSuggestion | None:
        """综合查询结果 + LLM 判断，给出修复建议

        流程：
        1. 调用 query_similar() 获取相似案例
        2. 如果有高置信度案例，直接返回
        3. 如果置信度不足，调用 LLM 综合判断
        4. 返回 RepairSuggestion 或 None

        Args:
            failure_type: 当前失败类型
            expr: 当前因子表达式
            metrics_snapshot: 当前指标快照
            min_confidence: 最小置信度阈值

        Returns:
            RepairSuggestion 实例或 None（如果没有足够好的建议）
        """
        with Timer("exp_replay_get_suggestion"):
            similar_cards = await self.query_similar(
                failure_type=failure_type,
                expr=expr,
                metrics_snapshot=metrics_snapshot,
                top_k=3,
            )

            if not similar_cards:
                logger.info("[EXP_REPLAY] no_similar_cases failure_type=%s — returning None", failure_type)
                return None

            best_card = similar_cards[0]
            base_confidence = self._compute_confidence_from_cards(similar_cards, metrics_snapshot or {})

            if base_confidence >= min_confidence and best_card.success_rate >= 0.6:
                suggestion = RepairSuggestion(
                    card_id=best_card.card_id,
                    confidence=base_confidence,
                    suggested_action=best_card.repair_action,
                    suggested_target=best_card.repair_target,
                    suggested_expr=self._apply_repair_to_expression(expr, best_card),
                    reasoning=f"基于 {len(similar_cards)} 个相似历史案例，最佳匹配 card_id={best_card.card_id[:8]}",
                    historical_success_rate=best_card.success_rate,
                    similar_cases=[
                        {"card_id": c.card_id[:8], "action": c.repair_action, "success_rate": c.success_rate}
                        for c in similar_cards
                    ],
                )

                logger.info(
                    "[EXP_REPLAY] suggestion_generated card_id=%s confidence=%.3f action=%s",
                    suggestion.card_id[:8],
                    suggestion.confidence,
                    suggestion.suggested_action,
                )
                return suggestion

            if self._llm is not None:
                try:
                    llm_suggestion = await self._call_llm_for_judgment(
                        failure_type=failure_type,
                        expr=expr,
                        metrics=metrics_snapshot or {},
                        similar_cards=similar_cards,
                    )
                    if llm_suggestion and llm_suggestion.confidence >= min_confidence:
                        logger.info(
                            "[EXP_REPLAY] llm_suggestion_accepted confidence=%.3f action=%s",
                            llm_suggestion.confidence,
                            llm_suggestion.suggested_action,
                        )
                        return llm_suggestion
                except (TimeoutError, aiohttp.ClientError, ValueError, json.JSONDecodeError) as exc:
                    logger.warning(
                        "[DEFENSIVE_LOG] exp_replay: llm_judgment_failed error=%s — falling back to best match", exc
                    )

            if base_confidence > 0.4:
                return RepairSuggestion(
                    card_id=best_card.card_id,
                    confidence=base_confidence,
                    suggested_action=best_card.repair_action,
                    suggested_target=best_card.repair_target,
                    suggested_expr=self._apply_repair_to_expression(expr, best_card),
                    reasoning=f"低置信度回退：基于最佳匹配案例 (confidence={base_confidence:.2f})",
                    historical_success_rate=best_card.success_rate,
                    similar_cases=[
                        {"card_id": c.card_id[:8], "action": c.repair_action, "success_rate": c.success_rate}
                        for c in similar_cards
                    ],
                )

            logger.info(
                "[EXP_REPLAY] no_suitable_suggestion failure_type=%s best_confidence=%.3f < threshold=%.2f",
                failure_type,
                base_confidence,
                min_confidence,
            )
            return None

    @algo_log()
    def update_card_outcome(self, card_id: str, outcome: str, metrics: dict | None = None) -> bool:
        """根据后续 WQ 反馈更新卡片成功率

        Args:
            card_id: 经验卡片 ID
            outcome: 新的结果 ("PASS"/"FAIL"/"IMPROVED")
            metrics: 更新后的指标（可选）

        Returns:
            是否成功更新
        """
        with Timer("exp_replay_update_outcome"):
            for card in self._cards:
                if card.card_id == card_id:
                    old_outcome = card.outcome
                    card.outcome = outcome

                    if outcome == "PASS":
                        new_rate = (card.success_rate * card.usage_count + 1.0) / (card.usage_count + 1)
                    elif outcome == "FAIL":
                        new_rate = (card.success_rate * card.usage_count + 0.0) / (card.usage_count + 1)
                    else:
                        new_rate = (card.success_rate * card.usage_count + 0.5) / (card.usage_count + 1)

                    card.success_rate = max(1e-6, min(1.0, new_rate))

                    if metrics:
                        card.improvement_delta = metrics

                    logger.info(
                        "[EXP_REPLAY] card_updated card_id=%s oldOutcome=%s newOutcome=%s newSuccessRate=%.3f",
                        card_id[:8],
                        old_outcome,
                        outcome,
                        card.success_rate,
                    )
                    return True

            logger.warning("[DEFENSIVE_LOG] exp_replay: card_not_found card_id=%s — cannot update outcome", card_id[:8])
            return False

    @algo_log()
    def persist(self) -> None:
        """持久化到 JSON 文件"""
        with Timer("exp_replay_persist"):
            try:
                data = {
                    "version": "1.0.0",
                    "updated_at": datetime.now(UTC).isoformat(),
                    "total_cards": len(self._cards),
                    "cards": [
                        {
                            "card_id": c.card_id,
                            "timestamp": c.timestamp,
                            "failure_type": c.failure_type,
                            "expression_structure": c.expression_structure,
                            "field_family": c.field_family,
                            "metrics_snapshot": c.metrics_snapshot,
                            "repair_action": c.repair_action,
                            "repair_target": c.repair_target,
                            "before_expr": c.before_expr,
                            "after_expr": c.after_expr,
                            "outcome": c.outcome,
                            "improvement_delta": c.improvement_delta,
                            "source_loop_id": c.source_loop_id,
                            "usage_count": c.usage_count,
                            "success_rate": c.success_rate,
                        }
                        for c in self._cards
                    ],
                }

                self._storage_path.write_text(
                    json.dumps(data, ensure_ascii=False, default=str, indent=2),
                    encoding="utf-8",
                )

                logger.info(
                    "[EXP_REPLAY] persisted path=%s totalCards=%d",
                    self._storage_path,
                    len(self._cards),
                )
            except OSError as exc:
                logger.warning(
                    "[DEFENSIVE_LOG] exp_replay: persist_failed path=%s error=%s — data not saved",
                    self._storage_path,
                    exc,
                )

    @algo_log()
    def load(self) -> None:
        """从 JSON 加载"""
        with Timer("exp_replay_load"):
            try:
                if not self._storage_path.exists():
                    logger.info(
                        "[EXP_REPLAY] no_existing_file path=%s — starting with empty library", self._storage_path
                    )
                    return

                data = json.loads(self._storage_path.read_text(encoding="utf-8"))
                cards_data = data.get("cards", [])

                self._cards = [
                    ExperienceCard(
                        card_id=c.get("card_id", ""),
                        timestamp=c.get("timestamp", ""),
                        failure_type=c.get("failure_type", ""),
                        expression_structure=c.get("expression_structure", ""),
                        field_family=c.get("field_family", ""),
                        metrics_snapshot=c.get("metrics_snapshot", {}),
                        repair_action=c.get("repair_action", ""),
                        repair_target=c.get("repair_target", ""),
                        before_expr=c.get("before_expr", ""),
                        after_expr=c.get("after_expr", ""),
                        outcome=c.get("outcome", ""),
                        improvement_delta=c.get("improvement_delta", {}),
                        source_loop_id=c.get("source_loop_id", 0),
                        usage_count=c.get("usage_count", 0),
                        success_rate=max(1e-6, min(1.0, c.get("success_rate", 0.0))),
                    )
                    for c in cards_data
                ]

                logger.info(
                    "[EXP_REPLAY] loaded path=%s totalCards=%d version=%s",
                    self._storage_path,
                    len(self._cards),
                    data.get("version", "unknown"),
                )
            except (OSError, FileNotFoundError, json.JSONDecodeError) as exc:
                logger.warning(
                    "[DEFENSIVE_LOG] exp_replay: load_failed path=%s error=%s — starting with empty library",
                    self._storage_path,
                    exc,
                )
                self._cards = []

    def get_stats(self) -> dict:
        """获取统计信息"""
        if not self._cards:
            return {"total_cards": 0, "avg_success_rate": 0.0, "failure_types": {}, "top_actions": []}

        from collections import Counter

        failure_types = Counter(c.failure_type for c in self._cards)
        actions = Counter(c.repair_action for c in self._cards)
        avg_success_rate = sum(c.success_rate for c in self._cards) / max(len(self._cards), 1)

        return {
            "total_cards": len(self._cards),
            "avg_success_rate": round(avg_success_rate, 4),
            "failure_types": dict(failure_types.most_common(5)),
            "top_actions": actions.most_common(5),
        }

    def _compute_similarity_score(
        self,
        card: ExperienceCard,
        target_failure_type: str,
        target_structure: str,
        target_field_family: str,
        target_metrics: dict,
    ) -> float:
        """计算多维加权融合相似度分数"""
        epsilon = 1e-6

        type_score = 1.0 if card.failure_type == target_failure_type else 0.0

        structure_score = self._compute_structure_similarity(card.expression_structure, target_structure)

        field_score = (
            1.0 if card.field_family == target_field_family or not card.field_family or not target_field_family else 0.3
        )

        metrics_score = self._compute_metrics_similarity(card.metrics_snapshot, target_metrics)

        weights = self._matching_weights
        total_score = (
            type_score * weights["failure_type"]
            + structure_score * weights["expression_structure"]
            + metrics_score * weights["metrics_snapshot"]
            + field_score * weights["field_family"]
        )

        success_boost = card.success_rate * 0.1
        final_score = max(epsilon, min(1.0, total_score + success_boost))

        return final_score

    def _compute_structure_similarity(self, struct1: str, struct2: str) -> float:
        """计算结构指纹相似度（简单的 token 重叠）"""
        if not struct1 or not struct2:
            return 0.5

        tokens1 = set(struct1.split("-"))
        tokens2 = set(struct2.split("-"))

        if not tokens1 or not tokens2:
            return 0.5

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / max(len(union), 1)

    def _compute_metrics_similarity(self, metrics1: dict, metrics2: dict) -> float:
        """计算指标范围相似度"""
        if not metrics1 or not metrics2:
            return 0.5

        score = 0.0
        count = 0

        for key in ["sharpe", "turnover", "margin", "fitness"]:
            v1 = metrics1.get(key)
            v2 = metrics2.get(key)
            if v1 is not None and v2 is not None:
                try:
                    v1_float = float(v1)
                    v2_float = float(v2)
                    diff = abs(v1_float - v2_float)
                    max_val = max(abs(v1_float), abs(v2_float), 1e-6)
                    normalized_diff = diff / max_val
                    similarity = max(0.0, 1.0 - normalized_diff)
                    score += similarity
                    count += 1
                except (ValueError, TypeError):
                    continue

        return score / max(count, 1)

    def _compute_confidence_from_cards(self, cards: list[ExperienceCard], current_metrics: dict) -> float:
        """从多个相似卡片计算综合置信度"""
        if not cards:
            return 0.0

        base_scores = []
        for card in cards:
            weight = card.success_rate
            base_scores.append(weight * (1.0 + card.usage_count * 0.05))

        if not base_scores:
            return 0.0

        raw_confidence = sum(base_scores) / len(base_scores)
        adjusted = min(1.0, raw_confidence * (1.0 + len(cards) * 0.1))

        return max(1e-6, adjusted)

    async def _call_llm_for_judgment(
        self, failure_type: str, expr: str, metrics: dict, similar_cards: list[ExperienceCard]
    ) -> RepairSuggestion | None:
        """调用 LLM 进行综合判断"""
        if self._llm is None:
            return None

        cards_summary = "\n".join(
            [
                f"- Card {i + 1}: action={c.repair_action}, target={c.repair_target}, "
                f"success_rate={c.success_rate:.2f}, outcome={c.outcome}"
                for i, c in enumerate(similar_cards[:3])
            ]
        )

        prompt = f"""你是一个量化因子修复专家。当前因子遇到 {failure_type} 类型失败。

当前表达式: {expr}
当前指标: sharpe={metrics.get("sharpe", "N/A")}, turnover={metrics.get("turnover", "N/A")}%

历史上相似的修复案例:
{cards_summary}

请根据以上信息，判断是否应该采用历史案例中的某个修复方案。

返回JSON格式:
{{
    "should_apply": true/false,
    "recommended_action": "推荐的修复动作",
    "recommended_target": "修改目标段(block_a/block_c)",
    "confidence": 0.0-1.0,
    "reasoning": "推理依据"
}}"""

        try:
            response = await self._llm.generate(prompt, [], "", session_id="exp_replay_judgment", cycle=0)

            import re

            json_match = re.search(r"\{[^{}]+\}", response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                if parsed.get("should_apply") and parsed.get("confidence", 0) >= 0.5:
                    best_card = similar_cards[0]
                    return RepairSuggestion(
                        card_id=best_card.card_id,
                        confidence=parsed.get("confidence", 0.5),
                        suggested_action=parsed.get("recommended_action", best_card.repair_action),
                        suggested_target=parsed.get("recommended_target", best_card.repair_target),
                        suggested_expr=self._apply_repair_to_expression(expr, best_card),
                        reasoning=parsed.get("reasoning", "LLM judgment based on historical cases"),
                        historical_success_rate=best_card.success_rate,
                        similar_cards=[{"card_id": c.card_id[:8], "action": c.repair_action} for c in similar_cards],
                    )
        except json.JSONDecodeError as exc:
            logger.warning("[DEFENSIVE_LOG] exp_replay: llm_parse_failed error=%s", exc)

        return None

    def _extract_structure_fingerprint(self, expr: str) -> str:
        """从表达式中提取结构指纹"""
        import re

        if not expr:
            return ""

        operators = sorted(set(re.findall(r"\b([a-zA-Z_]\w*)\s*\(", expr)))
        structure_parts = [
            op
            for op in operators
            if op in ("rank", "ts_decay_linear", "group_neutralize", "ts_mean", "ts_delta", "zscore")
        ]

        return "-".join(structure_parts) if structure_parts else "unknown"

    def _extract_field_family(self, expr: str) -> str:
        """从表达式中提取字段族"""
        import re

        if not expr:
            return ""

        price_fields = {"close", "open", "high", "low", "vwap"}
        volume_fields = {"volume", "adv"}
        fundamental_fields = {"cap", "earnings", "sales", "assets", "revenue", "sharesout"}

        found_price = bool(price_fields & set(re.findall(r"\b(close|open|high|low|vwap)\b", expr, re.IGNORECASE)))
        found_volume = bool(volume_fields & set(re.findall(r"\b(volume|adv\d+)\b", expr, re.IGNORECASE)))
        found_fundamental = bool(
            fundamental_fields
            & set(re.findall(r"\b(cap|earnings|sales|assets|revenue|sharesout)\b", expr, re.IGNORECASE))
        )

        families = []
        if found_price:
            families.append("price")
        if found_volume:
            families.append("volume")
        if found_fundamental:
            families.append("fundamental")

        return "_".join(families) if families else "unknown"

    def _compute_improvement_delta(self, metrics_before: dict | None, metrics_after: dict | None) -> dict:
        """计算改善量"""
        if not metrics_before or not metrics_after:
            return {}

        delta = {}
        for key in ["sharpe", "turnover", "margin", "fitness"]:
            v_before = metrics_before.get(key)
            v_after = metrics_after.get(key)
            if v_before is not None and v_after is not None:
                try:
                    delta[key] = float(v_after) - float(v_before)
                except (ValueError, TypeError):
                    continue

        return delta

    def _apply_repair_to_expression(self, current_expr: str, card: ExperienceCard) -> str:
        """将历史修复方案应用到当前表达式"""
        import re

        if not current_expr or not card.before_expr or not card.after_expr:
            return current_expr

        if card.repair_target == "block_c" and "ts_decay_linear" in current_expr:
            window_match = re.search(r"(\d+)", card.after_expr)
            if window_match:
                new_window = window_match.group(1)
                return re.sub(r"(\d+)", new_window, current_expr, count=1)

        if card.repair_action.startswith("window") and "→" in card.repair_action:
            parts = card.repair_action.split("→")
            if len(parts) == 2:
                old_val = parts[0].strip().split()[-1]
                new_val = parts[1].strip()
                if old_val.isdigit():
                    return current_expr.replace(old_val, new_val, 1)

        return current_expr


async def auto_debug_loop(
    generate_fn: Callable[..., Awaitable[str]],
    validate_fn: Callable[[str], Any],
    initial_expr: str,
    max_rounds: int = 3,
) -> tuple[str, bool]:
    best_expr = initial_expr
    current_expr = initial_expr
    last_error = ""

    for round_num in range(1, max_rounds + 1):
        validation = validate_fn(current_expr)
        passed = getattr(validation, "passed", False)
        if isinstance(validation, bool):
            passed = validation

        if passed:
            return current_expr, True

        if hasattr(validation, "failures") and validation.failures:
            last_error = "; ".join(str(f) for f in validation.failures)
        elif hasattr(validation, "error_message"):
            last_error = str(validation.error_message)
        else:
            last_error = str(validation)

        if round_num >= max_rounds:
            break

        try:
            fix_prompt = (
                f"The following alpha expression failed validation:\n"
                f"Expression: {current_expr}\n"
                f"Error: {last_error}\n\n"
                f"Please generate a corrected expression that fixes the above error."
            )
            current_expr = await generate_fn(fix_prompt)
            if not current_expr or not isinstance(current_expr, str):
                current_expr = best_expr
        except (TimeoutError, aiohttp.ClientError, ValueError, json.JSONDecodeError):
            logger.warning("auto_debug_loop: generate_fn failed on round %d", round_num + 1)
            break

    return best_expr, False
