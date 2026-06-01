from __future__ import annotations

import json
import logging
import math
import random
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvolutionRecord:
    record_id: str = ""
    expr: str = ""
    sharpe: float | None = None
    fitness: float | None = None
    turnover: float | None = None
    direction: str = ""
    category: str | None = None
    parent_id: str | None = None
    mutation_type: str | None = None
    timestamp: float = 0.0
    evaluation_detail: dict | None = None
    session_id: str | None = None
    status: str = "FAIL"
    embedding: list[float] | None = None

    def __post_init__(self):
        if not self.record_id:
            self.record_id = uuid.uuid4().hex[:12]
        if not self.timestamp:
            self.timestamp = time.time()


class EvolutionDatabase:
    def __init__(self, path: str = "runtime/evolution_db.json", embed_fn: Callable[..., Awaitable] | None = None):
        self._path = Path(path)
        self._records: dict[str, EvolutionRecord] = {}
        self._lock = threading.Lock()
        self._embed_fn = embed_fn
        self._load()

    def set_embed_fn(self, fn: Callable[..., Awaitable]) -> None:
        self._embed_fn = fn

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._records = {}
            for rid, rec_data in data.get("records", {}).items():
                self._records[rid] = EvolutionRecord(**rec_data)
            logger.info("EvolutionDatabase: loaded %d records from %s", len(self._records), self._path)
        except OSError:
            logger.warning("EvolutionDatabase: failed to load from %s", self._path, exc_info=True)
            self._records = {}

    def _save(self) -> None:
        try:
            data = {"records": {rid: asdict(rec) for rid, rec in self._records.items()}}
            self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError:
            logger.warning("EvolutionDatabase: failed to save", exc_info=True)

    async def add_record(
        self,
        expr: str,
        sharpe: float | None = None,
        fitness: float | None = None,
        turnover: float | None = None,
        direction: str = "",
        category: str | None = None,
        parent_id: str | None = None,
        mutation_type: str | None = None,
        evaluation_detail: dict | None = None,
        session_id: str | None = None,
        status: str = "FAIL",
    ) -> str:
        embedding = None
        if self._embed_fn is not None:
            try:
                vec = await self._embed_fn(expr)
                if isinstance(vec, list):
                    if len(vec) > 0 and isinstance(vec[0], list):
                        embedding = vec[0]
                    else:
                        embedding = vec
                elif isinstance(vec, np.ndarray):
                    embedding = vec.tolist()
            except (ValueError, TypeError, OSError):
                logger.warning("EvolutionDatabase: embed failed for record")
        with self._lock:
            rec = EvolutionRecord(
                expr=expr,
                sharpe=sharpe,
                fitness=fitness,
                turnover=turnover,
                direction=direction,
                category=category,
                parent_id=parent_id,
                mutation_type=mutation_type,
                evaluation_detail=evaluation_detail,
                session_id=session_id,
                status=status,
                embedding=embedding,
            )
            self._records[rec.record_id] = rec
            self._save()
            return rec.record_id

    def sample_inspiration(self, n: int = 3, min_sharpe: float = 0.5) -> list[EvolutionRecord]:
        with self._lock:
            candidates = [
                rec for rec in self._records.values()
                if rec.sharpe is not None and rec.sharpe >= min_sharpe
            ]
        if not candidates:
            with self._lock:
                candidates = list(self._records.values())
        if not candidates:
            return []
        weights = []
        for rec in candidates:
            w = max(rec.sharpe or 0.1, 0.1)
            weights.append(w)
        total = sum(weights)
        weights = [w / total for w in weights]
        n = min(n, len(candidates))
        return random.choices(candidates, weights=weights, k=n)

    def get_elite_by_features(
        self,
        direction: str | None = None,
        category: str | None = None,
        min_sharpe: float | None = None,
        limit: int = 10,
    ) -> list[EvolutionRecord]:
        """Query elite records filtered by direction, category, and/or minimum sharpe.

        Returns records sorted by sharpe (descending), limited to *limit* results.
        Useful for finding the best-performing alphas that match specific feature
        criteria, e.g. all momentum alphas with sharpe >= 1.0.
        """
        with self._lock:
            results = list(self._records.values())
        if direction:
            results = [r for r in results if r.direction == direction]
        if category:
            results = [r for r in results if r.category == category]
        if min_sharpe is not None:
            results = [r for r in results if r.sharpe is not None and r.sharpe >= min_sharpe]
        results.sort(key=lambda r: r.sharpe or 0, reverse=True)
        return results[:limit]

    def get_record(self, record_id: str) -> EvolutionRecord | None:
        with self._lock:
            return self._records.get(record_id)

    def get_lineage(self, record_id: str, depth: int = 5) -> list[EvolutionRecord]:
        """Trace the parent chain of a record up to *depth* ancestors.

        Returns a list starting from the given record, followed by each parent
        in succession. Useful for understanding how an alpha was derived through
        successive mutations or crossovers.
        """
        lineage = []
        current_id = record_id
        for _ in range(depth):
            with self._lock:
                rec = self._records.get(current_id)
            if rec is None:
                break
            lineage.append(rec)
            if rec.parent_id is None:
                break
            current_id = rec.parent_id
        return lineage

    def get_stats(self) -> dict:
        """Return summary statistics of the evolution database.

        Includes total/pass/fail counts, average and top sharpe, and direction
        distribution. Useful for dashboard displays and monitoring evolution progress.
        """
        with self._lock:
            records = list(self._records.values())
        total = len(records)
        if total == 0:
            return {"total": 0, "pass_count": 0, "fail_count": 0, "avg_sharpe": 0.0, "top_sharpe": 0.0, "direction_distribution": {}}
        pass_count = sum(1 for r in records if r.status == "PASS")
        fail_count = sum(1 for r in records if r.status != "PASS")
        sharpes = [r.sharpe for r in records if r.sharpe is not None]
        avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0.0
        top_sharpe = max(sharpes) if sharpes else 0.0
        direction_dist: dict[str, int] = {}
        for r in records:
            direction_dist[r.direction] = direction_dist.get(r.direction, 0) + 1
        return {
            "total": total,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "avg_sharpe": round(avg_sharpe, 4),
            "top_sharpe": round(top_sharpe, 4),
            "direction_distribution": direction_dist,
        }

    async def find_similar_by_embedding(self, query_expr: str, top_k: int = 3) -> list[dict]:
        if self._embed_fn is None:
            return []
        with self._lock:
            records_with_emb = [r for r in self._records.values() if r.embedding is not None]
        if not records_with_emb:
            return []
        try:
            query_vec = await self._embed_fn(query_expr)
            if isinstance(query_vec, np.ndarray):
                query_vec = query_vec.tolist()
            if isinstance(query_vec, list) and len(query_vec) > 0 and isinstance(query_vec[0], list):
                query_vec = query_vec[0]
            if not query_vec:
                return []
            results = []
            for r in records_with_emb:
                if r.embedding is None:
                    continue
                dot = sum(a * b for a, b in zip(query_vec, r.embedding))
                norm_a = math.sqrt(sum(a * a for a in query_vec))
                norm_b = math.sqrt(sum(b * b for b in r.embedding))
                if norm_a == 0 or norm_b == 0:
                    continue
                sim = dot / (norm_a * norm_b)
                results.append({"record": r, "similarity": round(sim, 4)})
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
        except (ValueError, TypeError, OSError):
            logger.warning("EvolutionDatabase: find_similar_by_embedding failed")
            return []

    async def portfolio_level_evaluation(self, new_expr: str, top_k: int = 5) -> dict:
        similar = await self.find_similar_by_embedding(new_expr, top_k=top_k)
        if not similar:
            return {
                "diversity_score": None,
                "max_similarity": 0.0,
                "similar_count": 0,
                "priority_adjustment": 0.0,
                "diversity_computed": False,
            }
        max_similarity = max(s.get("similarity", 0.0) for s in similar)
        diversity_score = round(1.0 - max_similarity, 4)
        priority_adjustment = 0.0
        if diversity_score > 0.5:
            priority_adjustment = 0.1
        elif diversity_score < 0.2:
            priority_adjustment = -0.1
        return {
            "diversity_score": diversity_score,
            "max_similarity": round(max_similarity, 4),
            "similar_count": len(similar),
            "priority_adjustment": priority_adjustment,
            "diversity_computed": True,
        }
