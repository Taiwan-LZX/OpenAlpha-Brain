from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class HallucinationEntry:
    content: str
    count: int = 1
    first_seen: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_seen: datetime = field(default_factory=lambda: datetime.now(UTC))
    source_sessions: list[str] = field(default_factory=list)


class GlobalKnowledge:
    def __init__(self, path: str = "global_knowledge.json"):
        self._path = Path(path)
        self._entries: dict[str, HallucinationEntry] = {}

    def load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._entries = {}
            for key, entry_data in data.get("entries", {}).items():
                self._entries[key] = HallucinationEntry(
                    content=entry_data["content"],
                    count=entry_data["count"],
                    first_seen=datetime.fromisoformat(entry_data["first_seen"]),
                    last_seen=datetime.fromisoformat(entry_data["last_seen"]),
                    source_sessions=entry_data.get("source_sessions", []),
                )
            logger.info("Loaded %d global knowledge entries from %s", len(self._entries), self._path)
        except (ValueError, TypeError, OSError):
            logger.warning("Failed to load global knowledge from %s", self._path, exc_info=True)
            self._entries = {}

    def save(self) -> None:
        try:
            data = self.to_dict()
            self._path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Saved %d global knowledge entries to %s", len(self._entries), self._path)
        except (ValueError, TypeError, OSError):
            logger.warning("Failed to save global knowledge to %s", self._path, exc_info=True)

    def merge_session(self, session_id: str, hallucination_log: list[str]) -> None:
        now = datetime.now(UTC)
        for content in hallucination_log:
            key = content.lower().strip()
            if not key:
                continue
            if key in self._entries:
                entry = self._entries[key]
                entry.count += 1
                entry.last_seen = now
                if session_id not in entry.source_sessions:
                    entry.source_sessions.append(session_id)
            else:
                self._entries[key] = HallucinationEntry(
                    content=content.strip(),
                    count=1,
                    first_seen=now,
                    last_seen=now,
                    source_sessions=[session_id],
                )

    def get_high_frequency(self, threshold: int = 3) -> list[str]:
        return [entry.content for entry in self._entries.values() if entry.count >= threshold]

    def get_all_entries(self) -> list[HallucinationEntry]:
        return sorted(self._entries.values(), key=lambda e: e.count, reverse=True)

    def get_never_use_list(self, threshold: int = 3) -> list[str]:
        return self.get_high_frequency(threshold)

    def get_rag_context_entries(self) -> list[dict]:
        return [
            {
                "content": entry.content,
                "count": entry.count,
                "last_seen": entry.last_seen.isoformat(),
            }
            for entry in self.get_all_entries()
        ]

    def to_dict(self) -> dict:
        return {
            "entries": {
                key: {
                    "content": entry.content,
                    "count": entry.count,
                    "first_seen": entry.first_seen.isoformat(),
                    "last_seen": entry.last_seen.isoformat(),
                    "source_sessions": entry.source_sessions,
                }
                for key, entry in self._entries.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> GlobalKnowledge:
        gk = cls()
        for key, entry_data in d.get("entries", {}).items():
            gk._entries[key] = HallucinationEntry(
                content=entry_data["content"],
                count=entry_data["count"],
                first_seen=datetime.fromisoformat(entry_data["first_seen"]),
                last_seen=datetime.fromisoformat(entry_data["last_seen"]),
                source_sessions=entry_data.get("source_sessions", []),
            )
        return gk
