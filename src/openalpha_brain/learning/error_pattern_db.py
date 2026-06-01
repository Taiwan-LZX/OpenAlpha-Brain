"""ErrorPatternDB — BRAIN rejection 错误模式学习数据库。

Inspired by worldquant-miner Generation Two AST error learning + Alpha-GPT Review phase feedback loop.

每次 BRAIN 提交被拒绝时，提取错误模式并持久化。
高频错误模式在下一次 cycle 注入 prompt 作为负面约束，
形成持续改进闭环。

数据流:
BRAIN rejection → extract_error_pattern() → store → aggregate ->
build_negative_constraints() -> 注入 prompts.py build_dynamic_context()
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import Any

from openalpha_brain.data import get_data_path

logger = logging.getLogger(__name__)

_ERROR_PATTERN_RULES: list[tuple[str, re.Pattern[str], str]] = [
    (
        "unknown_field",
        re.compile(r"unknown.*?field.*?(\w+)", re.IGNORECASE),
        "use returns or SAFE_FIELDS only",
    ),
    (
        "invalid_operator",
        re.compile(
            r"['\"](\w+)['\"].*?\bnot\s+recognized\b|invalid\s+operator.*?['\"](\w+)['\"]"
            r"|operator\s+(\w+)\s+not\s+recognized",
            re.IGNORECASE,
        ),
        "use ts_delay instead",
    ),
    (
        "syntax_error",
        re.compile(r"\bsyntax\b|\bparse\b|\bunexpected\s+(token|character|symbol)\b", re.IGNORECASE),
        "check parentheses balance and operator syntax",
    ),
    (
        "timeout",
        re.compile(r"timeout|exceeded", re.IGNORECASE),
        "reduce expression complexity or use shorter lookback windows",
    ),
    (
        "correlation_too_high",
        re.compile(r"correlation|overlap|similar|duplicate", re.IGNORECASE),
        "change operator family or dataset to reduce structural similarity",
    ),
    (
        "low_sharpe",
        re.compile(r"sharpe.*?low|below.*?threshold", re.IGNORECASE),
        "add normalization (ts_zscore) or tighten neutralization scope",
    ),
    (
        "high_turnover",
        re.compile(r"turnover.*?high|exceeds", re.IGNORECASE),
        "apply ts_decay_linear with window 5-15 to smooth signal",
    ),
]


class ErrorPatternDB:
    """BRAIN rejection 错误模式学习数据库。

    从 BRAIN 拒绝响应中提取结构化错误模式，
    持久化到 JSON 文件，并支持查询高频模式用于 prompt 约束注入。
    """

    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            db_path = get_data_path("error_patterns.json")
        self._db_path = Path(db_path)
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        try:
            if self._db_path.exists():
                with open(self._db_path, encoding="utf-8") as f:
                    return json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("ErrorPatternDB: failed to load %s: %s", self._db_path, exc)
        return {"patterns": {}, "stats": {"total_patterns": 0, "total_rejections": 0, "unique_types": 0}}

    def _save(self) -> bool:
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._db_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            return True
        except (OSError, TypeError) as exc:
            logger.error("ErrorPatternDB: failed to save %s: %s", self._db_path, exc)
            return False

    def _update_stats(self) -> None:
        patterns = self._data.get("patterns", {})
        total_patterns = len(patterns)
        total_rejections = sum(p.get("count", 0) for p in patterns.values())
        unique_types = len({p.get("type", "") for p in patterns.values()})
        self._data["stats"] = {
            "total_patterns": total_patterns,
            "total_rejections": total_rejections,
            "unique_types": unique_types,
        }

    def extract_error_pattern(
        self,
        expression: str,
        error_msg: str,
        brain_result: dict | None = None,
    ) -> dict:
        """从 BRAIN rejection 中提取结构化错误模式。

        Args:
            expression: 被拒绝的 alpha 表达式
            error_msg: BRAIN 返回的错误消息
            brain_result: 可选的完整 BRAIN 结果字典

        Returns:
            结构化的错误模式字典，包含 type, content, fix 等字段
        """
        pattern_type = "other"
        content = ""
        fix = "review expression structure and BRAIN documentation"

        for ptype, regex, default_fix in _ERROR_PATTERN_RULES:
            match = regex.search(error_msg)
            if match:
                pattern_type = ptype
                content = match.group(1) if match.lastindex and match.group(1) else error_msg[:80]
                fix = default_fix
                break

        if not content:
            content = error_msg[:100] if error_msg else "unknown error"

        today = date.today().isoformat()
        result: dict[str, Any] = {
            "type": pattern_type,
            "content": content,
            "count": 1,
            "last_seen": today,
            "fix": fix,
            "expression_preview": expression[:100] if expression else "",
        }

        if brain_result:
            if brain_result.get("status"):
                result["brain_status"] = brain_result["status"]
            if brain_result.get("gate_failures"):
                result["gate_failures"] = brain_result["gate_failures"][:3]
            if brain_result.get("sharpe") is not None:
                result["sharpe"] = brain_result["sharpe"]

        return result

    def store(self, pattern: dict) -> bool:
        """追加错误模式到 JSON 数据库。

        Args:
            pattern: 由 extract_error_pattern() 返回的错误模式字典

        Returns:
            是否成功保存
        """
        try:
            ptype = pattern.get("type", "other")
            pcontent = pattern.get("content", "")
            pattern_key = f"{ptype}:{pcontent}"

            existing = self._data.setdefault("patterns", {}).get(pattern_key)
            if existing:
                existing["count"] = existing.get("count", 0) + 1
                existing["last_seen"] = pattern.get("last_seen", date.today().isoformat())
                if pattern.get("fix"):
                    existing["fix"] = pattern["fix"]
            else:
                self._data["patterns"][pattern_key] = {
                    "type": ptype,
                    "content": pcontent,
                    "count": pattern.get("count", 1),
                    "last_seen": pattern.get("last_seen", date.today().isoformat()),
                    "fix": pattern.get("fix", ""),
                    "expression_preview": pattern.get("expression_preview", ""),
                }

            self._update_stats()
            return self._save()
        except Exception as exc:
            logger.error("ErrorPatternDB.store: unexpected error: %s", exc)
            return False

    def get_top_patterns(self, n: int = 10, min_count: int = 2) -> list[dict]:
        """获取最高频错误模式。

        Args:
            n: 返回的最大模式数量
            min_count: 最小出现次数阈值

        Returns:
            按计数降序排列的错误模式列表
        """
        patterns = self._data.get("patterns", {})
        filtered = [p for p in patterns.values() if p.get("count", 0) >= min_count]
        sorted_patterns = sorted(filtered, key=lambda x: x.get("count", 0), reverse=True)
        return sorted_patterns[:n]

    def build_negative_constraints(self, top_n: int = 5) -> str:
        """格式化为 prompt 可用的负面约束文本。

        Args:
            top_n: 包含的顶级错误模式数量

        Returns:
            可直接注入 prompt 的约束文本字符串
        """
        top_patterns = self.get_top_patterns(n=top_n, min_count=1)
        if not top_patterns:
            return ""

        lines = [
            "\n▶ LEARNED ERROR PATTERNS (from BRAIN rejections — AVOID these):",
            "  The following patterns have been repeatedly rejected by BRAIN:",
        ]

        for i, p in enumerate(top_patterns[:top_n], 1):
            ptype = p.get("type", "?")
            content = p.get("content", "?")
            count = p.get("count", 0)
            fix = p.get("fix", "")
            last_seen = p.get("last_seen", "?")

            lines.append(f"\n  {i}. [{ptype}] '{content}' (rejected {count}x, last: {last_seen})")
            if fix:
                lines.append(f"     Recommended fix: {fix}")

        lines.append("\n  IMPORTANT: Do NOT generate expressions that trigger these error types.")
        return "\n".join(lines)

    def increment_pattern(self, pattern_key: str) -> None:
        """增加指定模式的计数。

        Args:
            pattern_key: 格式为 'type:content' 的模式键
        """
        patterns = self._data.setdefault("patterns", {})
        if pattern_key in patterns:
            patterns[pattern_key]["count"] = patterns[pattern_key].get("count", 0) + 1
            patterns[pattern_key]["last_seen"] = date.today().isoformat()
            self._update_stats()
            self._save()

    def get_stats(self) -> dict:
        """返回统计信息。

        Returns:
            包含 total_patterns, total_rejections, unique_types 的字典
        """
        self._update_stats()
        return dict(self._data.get("stats", {}))

    def get_patterns_by_type(self, pattern_type: str) -> list[dict]:
        """按类型获取所有错误模式。

        Args:
            pattern_type: 错误类型名称

        Returns:
            匹配指定类型的错误模式列表
        """
        patterns = self._data.get("patterns", {})
        return [p for p in patterns.values() if p.get("type") == pattern_type]

    def clear(self) -> bool:
        """清空所有错误模式。

        Returns:
            是否成功清空并保存
        """
        self._data = {"patterns": {}, "stats": {"total_patterns": 0, "total_rejections": 0, "unique_types": 0}}
        return self._save()
