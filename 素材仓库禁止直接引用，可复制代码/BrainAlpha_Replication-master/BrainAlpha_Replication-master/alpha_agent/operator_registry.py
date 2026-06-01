from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_HERE = Path(__file__).resolve().parent
_OPERATORS_DIR = _HERE.parent / "operators"
_OPERATORS_JSON = _OPERATORS_DIR / "operators.json"
_DOCS_DIR = _OPERATORS_DIR / "docs"


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_ENTITIES_RE = re.compile(r"&[a-zA-Z]+;")
_MULTI_SPACE_RE = re.compile(r" {2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def _strip_html(html: str) -> str:
    text = _HTML_TAG_RE.sub(" ", html)
    text = _HTML_ENTITIES_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def _load_operators() -> Dict[str, Dict[str, Any]]:
    if not _OPERATORS_JSON.exists():
        return {}
    with open(_OPERATORS_JSON, encoding="utf-8") as f:
        raw = json.load(f)
    return {op["name"]: op for op in raw}


def _load_docs() -> Dict[str, Dict[str, Any]]:
    if not _DOCS_DIR.is_dir():
        return {}
    docs: Dict[str, Dict[str, Any]] = {}
    for p in sorted(_DOCS_DIR.glob("*.json")):
        name = p.stem
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            blocks = []
            for item in data.get("content", []):
                val = item.get("value", "")
                if isinstance(val, str):
                    blocks.append(_strip_html(val))
                elif isinstance(val, dict):
                    if "settings" in val:
                        blocks.append(_strip_html(json.dumps(val["settings"], ensure_ascii=False)))
                    else:
                        blocks.append(_strip_html(json.dumps(val, ensure_ascii=False)))
            docs[name] = {
                "text": "\n\n".join(blocks),
                "sequence": data.get("sequence", 9999),
            }
        except Exception:
            docs[name] = {"text": "", "sequence": 9999}
    return docs


class OperatorRegistry:
    def __init__(self) -> None:
        self._operators = _load_operators()
        self._docs = _load_docs()

    def _doc_text(self, name: str) -> str:
        entry = self._docs.get(name, {})
        return entry.get("text", "")

    def _doc_sequence(self, name: str) -> int:
        entry = self._docs.get(name, {})
        return entry.get("sequence", 9999)

    def get_valid_names(self) -> Set[str]:
        return set(self._operators.keys())

    def get_operator(self, name: str) -> Optional[Dict[str, Any]]:
        op = self._operators.get(name)
        if op is None:
            return None
        result = {
            "name": op.get("name", name),
            "definition": op.get("definition", ""),
            "description": op.get("description", ""),
            "category": op.get("category", ""),
            "scope": op.get("scope", []),
            "level": op.get("level", ""),
            "doc_text": self._doc_text(name),
        }
        return result

    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        return [
            op
            for op in self._operators.values()
            if op.get("category", "").lower() == category.lower()
        ]

    def get_categories(self) -> List[str]:
        cats: List[str] = []
        seen: Set[str] = set()
        for op in self._operators.values():
            c = op.get("category", "")
            if c and c not in seen:
                seen.add(c)
                cats.append(c)
        return cats

    def get_relevant(self, pattern: str) -> List[Dict[str, Any]]:
        pattern_lower = pattern.lower()
        result = []
        for op in self._operators.values():
            name = str(op.get("name", "")).lower()
            cat = str(op.get("category", "")).lower()
            desc = str(op.get("description", "")).lower()
            if pattern_lower in name or pattern_lower in cat or pattern_lower in desc:
                result.append(op)
        return result

    def get_operators_for_category(self, category: str, top_k: int = 8) -> List[Dict[str, Any]]:
        cat_ops = self.get_by_category(category)
        cat_ops.sort(key=lambda op: self._doc_sequence(op.get("name", "")))
        return cat_ops[:top_k]

    def format_operators_for_prompt(self, operator_names: List[str]) -> str:
        lines = []
        for name in operator_names:
            op = self.get_operator(name)
            if op is None:
                continue
            defn = op.get("definition", "")
            desc = op.get("description", "")
            if defn:
                lines.append(f"- {defn}: {desc}")
        return "\n".join(lines)


_REGISTRY: Optional[OperatorRegistry] = None


def get_operator_registry() -> OperatorRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = OperatorRegistry()
    return _REGISTRY
