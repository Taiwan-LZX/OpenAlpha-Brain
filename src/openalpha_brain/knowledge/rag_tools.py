from __future__ import annotations

import re
from typing import Any


class RAGBudgetTracker:
    def __init__(self, budget: int = 3):
        self._budget = budget
        self._used = 0

    def can_search(self) -> bool:
        return self._used < self._budget

    def record_search(self) -> None:
        self._used += 1

    def reset(self) -> None:
        self._used = 0

    @property
    def remaining(self) -> int:
        return self._budget - self._used

    def health_check(self) -> dict[str, Any]:
        return {
            "module": "RAGBudgetTracker",
            "status": "active" if self.can_search() else "exhausted",
            "budget": self._budget,
            "used": self._used,
            "remaining": self.remaining,
        }


_budget_tracker: RAGBudgetTracker | None = None


def set_budget_tracker(tracker: RAGBudgetTracker) -> None:
    global _budget_tracker
    _budget_tracker = tracker


def check_budget() -> bool:
    if _budget_tracker is None:
        return True
    if not _budget_tracker.can_search():
        return False
    _budget_tracker.record_search()
    return True


RAG_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_operators",
            "description": "Search for valid BRAIN operators by exploration direction. MUST call this before writing any expression to get the list of allowed operators.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "description": "Exploration direction: momentum, value, volatility, mean_reversion, quality, growth",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of operators to return",
                        "default": 15,
                    },
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_fields",
            "description": "Search for valid BRAIN data fields by exploration direction. MUST call this before writing any expression to get the list of allowed data fields. Using any field NOT in this list will cause BRAIN ERROR.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "description": "Exploration direction: momentum, value, volatility, mean_reversion, quality, growth",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of fields to return",
                        "default": 25,
                    },
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_financial_logic",
            "description": "Retrieve financial logic context for a given exploration direction. Returns relevant financial reasoning patterns, factor construction logic, and domain knowledge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "description": "Exploration direction: momentum, value, volatility, mean_reversion, quality, growth",
                    },
                },
                "required": ["direction"],
            },
        },
    },
]

_BRAIN_KEYWORDS = frozenset({
    "if", "else", "then", "and", "or", "not", "abs", "max", "min",
    "log", "sign", "sqrt", "pow", "exp",
})

_OPERATOR_PATTERN = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
_IDENTIFIER_PATTERN = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b")


async def execute_rag_tool(
    tool_name: str,
    arguments: dict[str, Any],
    rag_engine: Any,
) -> dict[str, Any]:
    direction = arguments.get("direction", "")

    if tool_name == "search_operators":
        if not check_budget():
            return {"operators": [], "token_count": 0}
        top_k = arguments.get("top_k", 15)
        retrieval = await rag_engine.retrieve(direction, top_k_ops=top_k)
        context = rag_engine.assemble_context(retrieval)
        ops_detailed = context.get("top_ops_detailed", [])
        remaining = context.get("remaining_op_names", [])
        operators = []
        for op in ops_detailed:
            operators.append({
                "name": op.get("name", ""),
                "signature": op.get("definition", ""),
                "category": op.get("category", ""),
            })
        for name in remaining:
            operators.append({"name": name, "signature": "", "category": ""})
        token_count = sum(len(op.get("name", "")) + len(op.get("signature", "")) + len(op.get("category", "")) for op in operators)
        return {"operators": operators, "token_count": token_count}

    if tool_name == "search_fields":
        if not check_budget():
            return {"fields": [], "eliminated": [], "token_count": 0}
        top_k = arguments.get("top_k", 25)
        retrieval = await rag_engine.retrieve(direction, top_k_fields=top_k)
        context = rag_engine.assemble_context(retrieval)
        field_ids = context.get("field_ids", [])
        eliminated = list(rag_engine._eliminated_fields) if rag_engine._eliminated_fields else []
        token_count = sum(len(f) for f in field_ids) + sum(len(e) for e in eliminated)
        return {"fields": field_ids, "eliminated": eliminated, "token_count": token_count}

    if tool_name == "search_financial_logic":
        if not check_budget():
            return {"financial_logic": [], "token_count": 0}
        retrieval = await rag_engine.retrieve(direction)
        context = rag_engine.assemble_context(retrieval)
        finlogic_ids = context.get("financial_logic_ids", [])
        token_count = sum(len(fl) for fl in finlogic_ids)
        return {"financial_logic": finlogic_ids, "token_count": token_count}

    return {"error": f"Unknown tool: {tool_name}"}


def validate_expression_fields(
    expression: str,
    allowed_fields: set[str],
) -> tuple[bool, list[str]]:
    if not expression or not allowed_fields:
        return True, []

    called_ops = set(_OPERATOR_PATTERN.findall(expression))
    all_identifiers = set(_IDENTIFIER_PATTERN.findall(expression))

    candidate_fields = set()
    for ident in all_identifiers:
        if ident in called_ops:
            continue
        if ident.isdigit():
            continue
        if ident in _BRAIN_KEYWORDS:
            continue
        if ident[0].isupper():
            continue
        candidate_fields.add(ident)

    invalid = sorted(candidate_fields - allowed_fields)
    is_valid = len(invalid) == 0
    return is_valid, invalid


def find_closest_field(invalid_field: str, allowed_fields: set[str]) -> str | None:
    if not allowed_fields:
        return None

    invalid_lower = invalid_field.lower()
    best_match = None
    best_score = -1

    for field in allowed_fields:
        field_lower = field.lower()
        if field_lower == invalid_lower:
            return field

        common_prefix = 0
        for a, b in zip(invalid_lower, field_lower):
            if a == b:
                common_prefix += 1
            else:
                break

        score = common_prefix
        if invalid_lower in field_lower or field_lower in invalid_lower:
            score += 10

        parts_invalid = set(invalid_lower.split("_"))
        parts_field = set(field_lower.split("_"))
        overlap = len(parts_invalid & parts_field)
        score += overlap * 5

        if score > best_score:
            best_score = score
            best_match = field

    if best_score < 2:
        return None
    return best_match


def auto_repair_expression(
    expression: str,
    allowed_fields: set[str],
    invalid_fields: list[str],
) -> str:
    repaired = expression
    for bad_field in invalid_fields:
        replacement = find_closest_field(bad_field, allowed_fields)
        if replacement is None:
            continue
        pattern = r"\b" + re.escape(bad_field) + r"\b"
        repaired = re.sub(pattern, replacement, repaired)
    return repaired
