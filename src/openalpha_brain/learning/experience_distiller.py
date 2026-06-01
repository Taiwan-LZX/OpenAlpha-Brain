from __future__ import annotations

import json
import logging
import math
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from openalpha_brain.utils.algo_logger import algo_log

logger = logging.getLogger(__name__)


@dataclass
class ExperienceCard:
    rule_id: str = ""
    failure_pattern: str = ""
    fix_strategy: str = ""
    applicable_conditions: str = ""
    confidence: float = 0.0
    usage_count: int = 0
    success_count: int = 0
    last_used: float = 0.0
    created_at: float = 0.0
    embedding: list[float] | None = None

    def __post_init__(self):
        if not self.rule_id:
            self.rule_id = uuid.uuid4().hex[:12]
        if not self.created_at:
            self.created_at = time.time()


class ExperienceDistiller:
    def __init__(self, path: str = "runtime/experience_cards.json", embed_fn: Callable[..., Awaitable] | None = None):
        self._path = Path(path)
        self._cards: list[ExperienceCard] = []
        self._embed_fn = embed_fn
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._cards = []
            for card_data in data.get("cards", []):
                self._cards.append(ExperienceCard(**card_data))
            logger.info("ExperienceDistiller: loaded %d cards from %s", len(self._cards), self._path)
        except (OSError, json.JSONDecodeError, ValueError, RuntimeError):
            logger.warning("ExperienceDistiller: failed to load from %s", self._path, exc_info=True)
            self._cards = []

    def _save(self) -> None:
        try:
            data = {"cards": [asdict(c) for c in self._cards]}
            self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except (OSError, ValueError, RuntimeError):
            logger.warning("ExperienceDistiller: failed to save", exc_info=True)

    async def _compute_embedding(self, text: str) -> list[float] | None:
        if self._embed_fn is None:
            return None
        try:
            vec = await self._embed_fn(text)
            if isinstance(vec, list):
                if len(vec) > 0 and isinstance(vec[0], list):
                    return vec[0]
                return vec
            if isinstance(vec, np.ndarray):
                return vec.tolist()
        except (ValueError, TypeError, OSError, RuntimeError):
            logger.warning("ExperienceDistiller: embed failed for text")
        return None

    @algo_log(log_args_to_skip=("self", "reflection_engine", "failure_lib"))
    async def distill_from_failures(
        self,
        reflection_engine,
        failure_lib,
        min_occurrences: int = 3,
    ) -> list[ExperienceCard]:
        patterns = reflection_engine.get_failure_patterns()
        new_cards: list[ExperienceCard] = []

        for pattern, count in patterns.items():
            if count < min_occurrences:
                continue

            try:
                similar_fixes = await failure_lib.search_fix(pattern, top_k=5)
            except (TimeoutError, ValueError, TypeError, ConnectionError, OSError):
                similar_fixes = []

            success_fixes = [f for f in similar_fixes if f.get("fix_success")]
            if not success_fixes:
                continue

            fix_strategies: list[str] = []
            for fix in success_fixes:
                attempt = fix.get("fix_attempt", "")
                if attempt:
                    fix_strategies.append(attempt)

            if not fix_strategies:
                continue

            best_fix = fix_strategies[0]
            direction = success_fixes[0].get("direction", "")

            card = ExperienceCard(
                failure_pattern=pattern,
                fix_strategy=best_fix,
                applicable_conditions=f"When {pattern} failure occurs"
                + (f" in {direction} direction" if direction else ""),
                confidence=min(0.9, 0.5 + 0.1 * len(success_fixes)),
            )
            if self._embed_fn is not None:
                embedding = await self._compute_embedding(f"{card.fix_strategy} {card.applicable_conditions}")
                if embedding is not None:
                    card.embedding = embedding
            new_cards.append(card)

        merged = await self._merge_cards(new_cards)
        self._cards.extend(merged)
        self._save()
        return merged

    async def _merge_cards(self, new_cards: list[ExperienceCard]) -> list[ExperienceCard]:
        result: list[ExperienceCard] = []
        for card in new_cards:
            conflict_idx = None
            for i, existing in enumerate(self._cards):
                if existing.failure_pattern == card.failure_pattern:
                    conflict_idx = i
                    break

            if conflict_idx is not None:
                if card.confidence > self._cards[conflict_idx].confidence:
                    self._cards[conflict_idx] = card
                continue

            result.append(card)

        use_semantic = self._embed_fn is not None and any(c.embedding is not None for c in result)

        deduped: list[ExperienceCard] = []
        for card in result:
            is_dup = False
            for kept in deduped:
                if use_semantic and card.embedding is not None and kept.embedding is not None:
                    sim = _cosine_similarity(card.embedding, kept.embedding)
                    threshold = 0.92
                else:
                    sim = self._string_similarity(card.fix_strategy, kept.fix_strategy)
                    threshold = 0.8
                if sim > threshold:
                    if card.confidence > kept.confidence:
                        deduped.remove(kept)
                        deduped.append(card)
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(card)

        if use_semantic:
            try:
                from openalpha_brain.core.loop_state import _algo_tick

                _algo_tick("experience_semantic_dedup")
            except (ImportError, AttributeError):
                pass

        return deduped

    @staticmethod
    def _string_similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        a_set = set(a.lower().split())
        b_set = set(b.lower().split())
        if not a_set or not b_set:
            return 0.0
        intersection = a_set & b_set
        union = a_set | b_set
        return len(intersection) / len(union)

    async def distill_from_evidence(
        self,
        alpha_logic_lib,
        min_evidence: int = 5,
    ) -> list[ExperienceCard]:
        new_cards: list[ExperienceCard] = []

        for logic in alpha_logic_lib.all_logics():
            if logic.evidence_count < min_evidence:
                continue

            passing_exprs: list[str] = []
            failing_exprs: list[str] = []
            for rec in logic.evidence_records:
                expr = rec.get("expression", "")
                if not expr:
                    continue
                if rec.get("fix_success"):
                    passing_exprs.append(expr)
                else:
                    failing_exprs.append(expr)

            if not passing_exprs:
                continue

            common_features = self._extract_common_features(passing_exprs)
            if not common_features:
                continue

            direction = logic.category
            card = ExperienceCard(
                failure_pattern=f"LOW_EVIDENCE+{logic.logic_id}",
                fix_strategy=f"Use {common_features}",
                applicable_conditions=f"For {direction} direction, {common_features} tend to succeed",
                confidence=min(0.9, 0.4 + 0.05 * len(passing_exprs)),
            )
            if self._embed_fn is not None:
                embedding = await self._compute_embedding(f"{card.fix_strategy} {card.applicable_conditions}")
                if embedding is not None:
                    card.embedding = embedding
            new_cards.append(card)

        merged = await self._merge_cards(new_cards)
        self._cards.extend(merged)
        self._save()
        return merged

    @staticmethod
    def _extract_common_features(expressions: list[str]) -> str:
        if not expressions:
            return ""
        token_sets: list[set[str]] = []
        for expr in expressions:
            tokens = set()
            for tok in expr.replace("(", " ").replace(")", " ").replace(",", " ").split():
                tok = tok.strip()
                if tok and not tok.isdigit():
                    tokens.add(tok)
            token_sets.append(tokens)

        if not token_sets:
            return ""

        common = token_sets[0]
        for s in token_sets[1:]:
            common = common & s

        if not common:
            return ""

        return ", ".join(sorted(common)[:5])

    @algo_log()
    async def get_applicable_cards(self, context: str, top_k: int = 3) -> list[ExperienceCard]:
        if not context:
            return []

        if self._embed_fn is not None:
            cards_with_emb = [c for c in self._cards if c.embedding is not None]
            if cards_with_emb:
                try:
                    query_emb = await self._compute_embedding(context)
                    if query_emb is not None:
                        scored: list[tuple[float, ExperienceCard]] = []
                        for card in cards_with_emb:
                            sim = _cosine_similarity(query_emb, card.embedding or [])
                            if sim > 0.7:
                                score = sim * card.confidence * (1 + 0.1 * card.usage_count)
                                scored.append((score, card))
                        scored.sort(key=lambda x: x[0], reverse=True)
                        try:
                            from openalpha_brain.core.loop_state import _algo_tick

                            _algo_tick("experience_semantic_retrieval")
                        except (ImportError, AttributeError):
                            pass
                        return [card for _, card in scored[:top_k]]
                except (ValueError, TypeError, OSError, RuntimeError):
                    logger.warning("ExperienceDistiller: semantic retrieval failed, falling back to keyword matching")

        context_lower = context.lower()
        scored_kw: list[tuple[float, ExperienceCard]] = []

        for card in self._cards:
            pattern_lower = card.failure_pattern.lower()
            cond_lower = card.applicable_conditions.lower()

            match = False
            for keyword in pattern_lower.replace("+", " ").split():
                if keyword in context_lower:
                    match = True
                    break
            if not match:
                for keyword in cond_lower.split():
                    if len(keyword) > 2 and keyword in context_lower:
                        match = True
                        break

            if match:
                score = card.confidence * (1 + 0.1 * card.usage_count)
                scored_kw.append((score, card))

        scored_kw.sort(key=lambda x: x[0], reverse=True)
        return [card for _, card in scored_kw[:top_k]]

    async def _store_pattern(
        self,
        pattern_type: str,
        pattern_data: dict,
        confidence: float = 0.5,
        strategy_id: str = "",
    ) -> None:
        card = ExperienceCard(
            failure_pattern=pattern_type,
            fix_strategy=json.dumps(pattern_data, ensure_ascii=False),
            applicable_conditions=f"direction={strategy_id}" if strategy_id else "",
            confidence=confidence,
        )
        self._cards.append(card)
        self._save()

    def record_card_usage(self, rule_id: str, success: bool) -> None:
        for card in self._cards:
            if card.rule_id == rule_id:
                card.usage_count += 1
                if success:
                    card.success_count += 1
                    card.confidence = card.success_count / card.usage_count
                card.last_used = time.time()
                self._save()
                return

    @algo_log(log_args_to_skip=("self",))
    def record_single_failure(
        self, failure_pattern: str, fix_strategy: str = "", direction: str = "", expression: str = ""
    ) -> ExperienceCard | None:
        for existing in self._cards:
            if existing.failure_pattern == failure_pattern:
                existing.usage_count += 1
                if fix_strategy and fix_strategy not in (existing.fix_strategy or ""):
                    existing.fix_strategy = fix_strategy
                self._save()
                return existing
        card = ExperienceCard(
            failure_pattern=failure_pattern,
            fix_strategy=fix_strategy or f"Avoid {failure_pattern}",
            applicable_conditions=f"When {failure_pattern} occurs" + (f" in {direction}" if direction else ""),
            confidence=0.3,
        )
        self._cards.append(card)
        self._save()
        logger.info(
            "ExperienceDistiller: recorded new failure pattern '%s' (total cards=%d)", failure_pattern, len(self._cards)
        )
        return card

    def save_cards(self, path: str | Path | None = None) -> bool:
        """保存经验卡片到磁盘（公开接口）

        Args:
            path (str | Path | None): 保存路径，默认使用初始化时的路径

        Returns:
            bool: 是否保存成功
        """
        target = Path(path) if path else self._path
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "cards": [asdict(c) for c in self._cards],
                "saved_at": time.time(),
                "version": "1.0",
            }
            target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info(
                "[DEFENSIVE_LOG] ExperienceDistiller::save_cards 保存成功 → %s (cards=%d)",
                target,
                len(self._cards),
            )
            return True
        except (OSError, ValueError, RuntimeError) as exc:
            logger.warning("[DEFENSIVE_LOG] ExperienceDistiller::save_cards 保存失败: %s", exc)
            return False

    def get_top_cards(self, n: int = 5) -> list[dict]:
        """Return top-N cards by confidence score as dicts for prompt injection."""
        if not self._cards:
            return []
        sorted_cards = sorted(self._cards, key=lambda c: c.confidence, reverse=True)
        return [asdict(c) for c in sorted_cards[:n]]

    def load_cards(self, path: str | Path | None = None) -> int:
        """从磁盘加载经验卡片（公开接口）

        Args:
            path (str | Path | None): 加载路径，默认使用初始化时的路径

        Returns:
            int: 加载的卡片数量，失败返回 0
        """
        target = Path(path) if path else self._path
        if not target.exists():
            logger.debug("[DEFENSIVE_LOG] ExperienceDistiller::load_cards 文件不存在: %s", target)
            return 0
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            cards_data = data.get("cards", [])
            loaded_cards = []
            for card_data in cards_data:
                try:
                    card = ExperienceCard(**card_data)
                    loaded_cards.append(card)
                except (ValueError, TypeError):
                    logger.warning("[DEFENSIVE_LOG] ExperienceDistiller::load_cards 跳过无效卡片数据")
                    continue
            self._cards = loaded_cards
            saved_at = data.get("saved_at", "unknown")
            logger.info(
                "[DEFENSIVE_LOG] ExperienceDistiller::load_cards 加载成功 ← %s (saved_at=%s, cards=%d)",
                target,
                saved_at,
                len(self._cards),
            )
            return len(self._cards)
        except (OSError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
            logger.warning("[DEFENSIVE_LOG] ExperienceDistiller::load_cards 加载失败: %s", exc)
            self._cards = []
            return 0


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
