"""
OpenAlpha-Brain — Lightweight Vector Store
Stores document embeddings with metadata, supports cosine-similarity retrieval.
Persistence via JSON (vectors serialised as base64).
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _encode_vec(v: np.ndarray) -> str:
    return base64.b64encode(v.astype(np.float32).tobytes()).decode("ascii")


def _decode_vec(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.float32)


class VectorStore:
    def __init__(self, dim: int = 0) -> None:
        self.dim = dim
        self._ids: list[str] = []
        self._vectors: np.ndarray | None = None
        self._metas: list[dict[str, Any]] = []
        self._id_to_idx: dict[str, int] = {}

    @property
    def count(self) -> int:
        return len(self._ids)

    def add_documents(
        self,
        ids: list[str],
        vectors: list[np.ndarray],
        metas: list[dict[str, Any]] | None = None,
    ) -> None:
        if not ids:
            return
        arr = np.stack(vectors).astype(np.float32)
        if self.dim == 0:
            self.dim = arr.shape[1]
        if arr.shape[1] != self.dim:
            raise ValueError(f"Vector dim mismatch: expected {self.dim}, got {arr.shape[1]}")
        if metas is None:
            metas = [{} for _ in ids]
        if self._vectors is None:
            self._vectors = arr
            self._ids = list(ids)
            self._metas = list(metas)
            self._id_to_idx = {did: i for i, did in enumerate(ids)}
        else:
            for i, did in enumerate(ids):
                if did in self._id_to_idx:
                    idx = self._id_to_idx[did]
                    self._vectors[idx] = arr[i]
                    self._metas[idx] = metas[i]
                else:
                    self._id_to_idx[did] = len(self._ids)
                    self._ids.append(did)
                    self._metas.append(metas[i])
                    self._vectors = np.concatenate([self._vectors, arr[i : i + 1]], axis=0)

    def query(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        exclude_ids: set[str] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        if self._vectors is None or self.count == 0:
            return []
        q = query_vec.astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-9:
            return []
        q = q / q_norm
        norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        normed = self._vectors / norms
        sims = normed @ q
        exclude = exclude_ids or set()
        results: list[tuple[str, float, dict[str, Any]]] = []
        for idx in np.argsort(-sims):
            did = self._ids[int(idx)]
            if did in exclude:
                continue
            results.append((did, float(sims[int(idx)]), self._metas[int(idx)]))
            if len(results) >= top_k:
                break
        return results

    def save_index(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "dim": self.dim,
            "count": self.count,
            "ids": self._ids,
            "metas": self._metas,
        }
        if self._vectors is not None:
            data["vectors_b64"] = [_encode_vec(self._vectors[i]) for i in range(self.count)]
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved vector index (%d docs) → %s", self.count, path)

    @classmethod
    def load_index(cls, path: str | Path) -> VectorStore:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Vector index not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        store = cls(dim=data.get("dim", 0))
        store._ids = data.get("ids", [])
        store._metas = data.get("metas", [])
        store._id_to_idx = {did: i for i, did in enumerate(store._ids)}
        vecs_b64 = data.get("vectors_b64", [])
        if vecs_b64:
            store._vectors = np.stack([_decode_vec(v) for v in vecs_b64])
        store.count
        logger.info("Loaded vector index (%d docs) ← %s", store.count, path)
        return store

    def get_by_id(self, doc_id: str) -> dict[str, Any] | None:
        idx = self._id_to_idx.get(doc_id)
        if idx is None:
            return None
        return {"id": doc_id, "meta": self._metas[idx]}

    def get_all_ids(self) -> list[str]:
        return list(self._ids)
