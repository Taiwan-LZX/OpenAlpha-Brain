from __future__ import annotations

import json
import logging
import math
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

import numpy as np

from openalpha_brain.utils.algo_logger import algo_log

logger = logging.getLogger(__name__)


@dataclass
class AlphaTool:
    tool_id: str = ""
    name: str = ""
    parameters: list[str] = field(default_factory=list)
    applicable_conditions: str = ""
    fix_logic: str = ""
    economic_explanation: str = ""
    success_count: int = 0
    fail_count: int = 0
    last_used: float = 0.0
    created_at: float = 0.0
    embedding: list[float] | None = None

    def __post_init__(self):
        if not self.tool_id:
            self.tool_id = uuid.uuid4().hex[:12]
        if not self.created_at:
            self.created_at = time.time()


class ToolFactory:
    def __init__(self, path: str = "runtime/alpha_tools.json", embed_fn: Callable[..., Awaitable] | None = None):
        self._path = Path(path)
        self._tools: dict[str, AlphaTool] = {}
        self._embed_fn = embed_fn
        self._pattern_counter: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._tools = {}
            for tid, tool_data in data.get("tools", {}).items():
                self._tools[tid] = AlphaTool(**tool_data)
            self._pattern_counter = data.get("pattern_counter", {})
            logger.info("ToolFactory: loaded %d tools from %s", len(self._tools), self._path)
        except OSError:
            logger.warning("ToolFactory: failed to load from %s", self._path, exc_info=True)
            self._tools = {}
            self._pattern_counter = {}

    def _save(self) -> None:
        try:
            data = {
                "tools": {tid: asdict(tool) for tid, tool in self._tools.items()},
                "pattern_counter": self._pattern_counter,
            }
            self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError:
            logger.warning("ToolFactory: failed to save", exc_info=True)

    def record_fix_pattern(
        self,
        failure_type: str,
        fix_attempt: str,
        fix_success: bool,
        direction: str = "",
    ) -> None:
        pattern_key = f"{failure_type}::{fix_attempt[:100]}"
        self._pattern_counter[pattern_key] = self._pattern_counter.get(pattern_key, 0) + 1
        count = self._pattern_counter[pattern_key]
        if count >= 2 and fix_success:
            has_tool = any(
                t.applicable_conditions == f"{failure_type} + {direction}" and t.fix_logic == fix_attempt
                for t in self._tools.values()
            )
            if not has_tool:
                import asyncio

                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(
                        self.create_tool_from_pattern(failure_type, fix_attempt, direction),
                    )
                except RuntimeError:
                    asyncio.run(
                        self.create_tool_from_pattern(failure_type, fix_attempt, direction),
                    )
        self._save()

    async def create_tool_from_pattern(
        self,
        failure_type: str,
        fix_attempt: str,
        direction: str = "",
    ) -> AlphaTool | None:
        tool_name = f"fix_{failure_type.lower()}_{direction.lower()}"
        tool_name = tool_name.replace(" ", "_").replace(";", "").replace(",", "")
        import re

        parameters = []
        param_patterns = [
            r"change\s+(\w+)\s+to\s+(\w+)",
            r"add\s+(\w+)",
            r"replace\s+(\w+)\s+with\s+(\w+)",
            r"set\s+(\w+)\s+to\s+(\w+)",
        ]
        for pat in param_patterns:
            for m in re.finditer(pat, fix_attempt, re.IGNORECASE):
                for g in m.groups():
                    if g and g not in parameters:
                        parameters.append(g)
        if not parameters:
            parameters = ["expression", "lookback_range"]

        applicable_conditions = f"{failure_type} + {direction}" if direction else failure_type
        fix_logic = fix_attempt
        economic_explanation = f"This fix addresses {failure_type} by {fix_logic}"

        embedding = None
        if self._embed_fn is not None:
            try:
                vec = await self._embed_fn(fix_logic)
                if isinstance(vec, list):
                    if len(vec) > 0 and isinstance(vec[0], list):
                        embedding = vec[0]
                    else:
                        embedding = vec
                elif isinstance(vec, np.ndarray):
                    embedding = vec.tolist()
            except (ValueError, TypeError, OSError):
                logger.warning("ToolFactory: embed failed for tool")

        for existing in self._tools.values():
            if existing.name == tool_name:
                return None
            if embedding is not None and existing.embedding is not None:
                sim = _cosine_similarity(embedding, existing.embedding)
                if sim > 0.9:
                    return None

        tool = AlphaTool(
            name=tool_name,
            parameters=parameters,
            applicable_conditions=applicable_conditions,
            fix_logic=fix_logic,
            economic_explanation=economic_explanation,
            embedding=embedding,
        )
        self._tools[tool.tool_id] = tool
        self._save()
        logger.info("ToolFactory: created tool %s (%s)", tool.name, tool.tool_id)
        return tool

    @algo_log()
    async def search_tools(self, query: str, top_k: int = 3) -> list[dict]:
        if self._embed_fn is None or not self._tools:
            return []
        try:
            query_vec = await self._embed_fn(query)
            if isinstance(query_vec, np.ndarray):
                query_vec = query_vec.tolist()
            if isinstance(query_vec, list) and len(query_vec) > 0 and isinstance(query_vec[0], list):
                query_vec = query_vec[0]
            if not query_vec:
                return []

            now = time.time()
            results: list[dict[str, object]] = []
            for tool in self._tools.values():
                if tool.embedding is None:
                    continue
                sim = _cosine_similarity(query_vec, tool.embedding)
                age_days = (now - tool.last_used) / 86400 if tool.last_used > 0 else 0
                if age_days < 30:
                    decay = 1.0
                elif age_days < 60:
                    decay = 0.5
                else:
                    decay = 0.2
                decayed_sim = sim * decay
                deprecated = age_days >= 60
                results.append(
                    {
                        "tool": tool,
                        "similarity": round(sim, 4),
                        "decayed_similarity": round(decayed_sim, 4),
                        "deprecated": deprecated,
                    }
                )

            def _sim_key(item: dict[str, object]) -> float:
                return cast(float, item["decayed_similarity"])

            results.sort(key=_sim_key, reverse=True)
            return results[:top_k]
        except (ValueError, TypeError, OSError):
            logger.warning("ToolFactory: search_tools failed")
            return []

    @algo_log()
    def apply_tool(self, tool_id: str, success: bool) -> None:
        tool = self._tools.get(tool_id)
        if tool is None:
            return
        if success:
            tool.success_count += 1
        else:
            tool.fail_count += 1
        tool.last_used = time.time()
        self._save()

    def detect_conflicts(self) -> list[tuple[str, str]]:
        conflicts: list[tuple[str, str]] = []
        tools_with_emb = [t for t in self._tools.values() if t.embedding is not None]
        for i in range(len(tools_with_emb)):
            for j in range(i + 1, len(tools_with_emb)):
                a = tools_with_emb[i]
                b = tools_with_emb[j]
                _emb_a: list[float] = a.embedding if a.embedding is not None else []
                _emb_b: list[float] = b.embedding if b.embedding is not None else []
                sim = _cosine_similarity(_emb_a, _emb_b)
                if sim > 0.9:
                    conflicts.append((a.tool_id, b.tool_id))
                    total_a = a.success_count + a.fail_count
                    total_b = b.success_count + b.fail_count
                    rate_a = a.success_count / total_a if total_a > 0 else 0.0
                    rate_b = b.success_count / total_b if total_b > 0 else 0.0
                    weaker_id = b.tool_id if rate_a >= rate_b else a.tool_id
                    self._tools.pop(weaker_id, None)
        if conflicts:
            self._save()
        return conflicts


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
