"""
OpenAlpha-Brain — AST Subtree Isomorphism Originality Detection

Implements the originality enforcement mechanism described in AlphaAgent
paper Section 3.2 (Originality Enforcement).

Pipeline:
  1. Parse FASTEXPR expression into an AST
  2. Extract all subtrees and compute normalized hashes
  3. Compare new factor's subtree hashes against the fingerprint database
  4. Originality score = 1 - (isomorphic_subtree_count / total_subtree_count)

Score < 0.3 indicates excessive structural similarity to existing factors.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AST Node
# ---------------------------------------------------------------------------

def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


@dataclass
class ASTNode:
    """AST node for a FASTEXPR expression.

    Leaf nodes:  op=""  children=[]  value="close" | "5" | ...
    Func call:   op="rank"  children=[...]  value=""
    Binary op:   op="+"  children=[left, right]  value=""
    Unary neg:   op="neg"  children=[child]  value=""
    """

    op: str = ""
    children: list[ASTNode] = field(default_factory=list)
    value: str = ""

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_field(self) -> bool:
        return self.is_leaf and self.op == "" and bool(self.value) and not _is_number(self.value)

    @property
    def is_number(self) -> bool:
        return self.is_leaf and self.op == "" and bool(self.value) and _is_number(self.value)

    def __repr__(self) -> str:
        if self.is_leaf:
            return f"ASTNode(value={self.value!r})"
        return f"ASTNode(op={self.op!r}, children={self.children!r})"


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class _TT(Enum):
    IDENT = auto()
    NUMBER = auto()
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    EOF = auto()


@dataclass
class _Token:
    type: _TT
    value: str
    pos: int = 0


def _tokenize(expr: str) -> list[_Token]:
    tokens: list[_Token] = []
    i = 0
    n = len(expr)
    while i < n:
        ch = expr[i]
        if ch in " \t\n\r":
            i += 1
            continue
        if ch == "(":
            tokens.append(_Token(_TT.LPAREN, "(", i))
            i += 1
        elif ch == ")":
            tokens.append(_Token(_TT.RPAREN, ")", i))
            i += 1
        elif ch == ",":
            tokens.append(_Token(_TT.COMMA, ",", i))
            i += 1
        elif ch == "+":
            tokens.append(_Token(_TT.PLUS, "+", i))
            i += 1
        elif ch == "-":
            tokens.append(_Token(_TT.MINUS, "-", i))
            i += 1
        elif ch == "*":
            tokens.append(_Token(_TT.STAR, "*", i))
            i += 1
        elif ch == "/":
            tokens.append(_Token(_TT.SLASH, "/", i))
            i += 1
        elif ch.isdigit() or (ch == "." and i + 1 < n and expr[i + 1].isdigit()):
            start = i
            while i < n and (expr[i].isdigit() or expr[i] == "."):
                i += 1
            tokens.append(_Token(_TT.NUMBER, expr[start:i], start))
        elif ch.isalpha() or ch == "_":
            start = i
            while i < n and (expr[i].isalnum() or expr[i] == "_"):
                i += 1
            tokens.append(_Token(_TT.IDENT, expr[start:i], start))
        else:
            raise ValueError(f"Unexpected character at position {i}: {ch!r}")
    tokens.append(_Token(_TT.EOF, "", n))
    return tokens


# ---------------------------------------------------------------------------
# Recursive-descent parser
# ---------------------------------------------------------------------------
# Grammar (precedence low → high):
#   expr    → term (('+' | '-') term)*
#   term    → unary (('*' | '/') unary)*
#   unary   → '-' unary | primary
#   primary → IDENT '(' args ')' | NUMBER | IDENT | '(' expr ')'
#   args    → ε | expr (',' expr)*
# ---------------------------------------------------------------------------

class _ExprParser:
    def __init__(self, tokens: list[_Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    def _cur(self) -> _Token:
        return self._tokens[self._pos]

    def _advance(self) -> _Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, tt: _TT) -> _Token:
        tok = self._cur()
        if tok.type != tt:
            raise ValueError(
                f"Expected {tt.name}, got {tok.type.name} ({tok.value!r}) "
                f"at position {tok.pos}",
            )
        return self._advance()

    def _peek(self) -> _TT:
        return self._cur().type

    # ---- grammar rules ----------------------------------------------------

    def _expr(self) -> ASTNode:
        left = self._term()
        while self._peek() in (_TT.PLUS, _TT.MINUS):
            op = self._advance().value
            right = self._term()
            left = ASTNode(op=op, children=[left, right])
        return left

    def _term(self) -> ASTNode:
        left = self._unary()
        while self._peek() in (_TT.STAR, _TT.SLASH):
            op = self._advance().value
            right = self._unary()
            left = ASTNode(op=op, children=[left, right])
        return left

    def _unary(self) -> ASTNode:
        if self._peek() == _TT.MINUS:
            self._advance()
            child = self._unary()
            return ASTNode(op="neg", children=[child])
        return self._primary()

    def _primary(self) -> ASTNode:
        tok = self._cur()
        if tok.type == _TT.NUMBER:
            self._advance()
            return ASTNode(value=tok.value)
        if tok.type == _TT.IDENT:
            self._advance()
            if self._peek() == _TT.LPAREN:
                return self._func_call(tok.value)
            return ASTNode(value=tok.value)
        if tok.type == _TT.LPAREN:
            self._advance()
            node = self._expr()
            self._expect(_TT.RPAREN)
            return node
        raise ValueError(
            f"Unexpected token {tok.type.name} ({tok.value!r}) at position {tok.pos}",
        )

    def _func_call(self, name: str) -> ASTNode:
        self._expect(_TT.LPAREN)
        args: list[ASTNode] = []
        if self._peek() != _TT.RPAREN:
            args.append(self._expr())
            while self._peek() == _TT.COMMA:
                self._advance()
                args.append(self._expr())
        self._expect(_TT.RPAREN)
        return ASTNode(op=name, children=args)


# ---------------------------------------------------------------------------
# FASTEXPR Parser (public API)
# ---------------------------------------------------------------------------

class FASTEXPRParser:
    """FASTEXPR expression parser — tokenizes then builds an AST via
    recursive descent."""

    def parse(self, expr: str) -> ASTNode:
        """Parse a FASTEXPR expression string into an AST.

        Supports: operator calls, arithmetic (+, -, *, /), parentheses,
        field names, and numeric constants.
        """
        tokens = _tokenize(expr)
        parser = _ExprParser(tokens)
        node = parser._expr()
        if parser._cur().type != _TT.EOF:
            raise ValueError(
                f"Unexpected token at position {parser._cur().pos}: "
                f"{parser._cur().value!r}",
            )
        return node

    def normalize(self, node: ASTNode) -> ASTNode:
        """Return a deep copy with field names replaced by _F1, _F2, ...

        Number literals are kept as-is.  The mapping is assigned in
        left-to-right depth-first order of first appearance.

        Example:
            rank(ts_delta(close, 5))  →  rank(ts_delta(_F1, 5))
        """
        field_map: dict[str, str] = {}
        counter = [0]
        return self._normalize_rec(node, field_map, counter)

    def _normalize_rec(
        self, node: ASTNode, field_map: dict[str, str], counter: list[int],
    ) -> ASTNode:
        if node.is_field:
            if node.value not in field_map:
                counter[0] += 1
                field_map[node.value] = f"_F{counter[0]}"
            return ASTNode(value=field_map[node.value])
        if node.is_number:
            return ASTNode(value=node.value)
        norm_children = [
            self._normalize_rec(c, field_map, counter) for c in node.children
        ]
        return ASTNode(op=node.op, children=norm_children)

    def extract_subtrees(self, node: ASTNode) -> list[ASTNode]:
        """Extract every subtree rooted at *node* or any of its descendants
        (inclusive of *node* itself)."""
        result: list[ASTNode] = []
        self._collect_subtrees(node, result)
        return result

    def _collect_subtrees(self, node: ASTNode, result: list[ASTNode]) -> None:
        result.append(node)
        for child in node.children:
            self._collect_subtrees(child, result)

    def subtree_hash(self, node: ASTNode) -> str:
        """Compute a normalized SHA-256 hash for the subtree rooted at *node*.

        The subtree is first independently normalized (field names → _F1,
        _F2, … scoped to this subtree), then rendered to a canonical
        string, and finally hashed.
        """
        normalized = self.normalize(node)
        canonical = self._canonical_str(normalized)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def _canonical_str(self, node: ASTNode) -> str:
        if node.is_leaf:
            return node.value
        child_strs = ",".join(self._canonical_str(c) for c in node.children)
        return f"{node.op}({child_strs})"


# ---------------------------------------------------------------------------
# Originality Checker
# ---------------------------------------------------------------------------

class OriginalityChecker:
    """AST subtree isomorphism originality checker.

    Maintains a persistent fingerprint database (JSON file) that maps
    ``alpha_id`` → list of subtree hashes.  When a new alpha is evaluated,
    its subtree hashes are compared against all existing hashes to compute
    an originality score.
    """

    ORIGINALITY_THRESHOLD: float = 0.3

    def __init__(self, fingerprint_path: str = "ast_fingerprints.json") -> None:
        self._fingerprints: dict[str, list[str]] = {}
        self._fingerprint_path = fingerprint_path
        self._parser = FASTEXPRParser()
        self._load()

    # ---- public API -------------------------------------------------------

    def check_originality(self, alpha_id: str, expr: str) -> float:
        """Check expression originality against the fingerprint database.

        Returns a score in [0, 1]:
            1.0 = completely novel
            0.0 = every subtree already exists in the database

        A score below ``ORIGINALITY_THRESHOLD`` (0.3) indicates the factor
        is too similar to existing ones.
        """
        try:
            ast = self._parser.parse(expr)
        except ValueError as exc:
            logger.warning("check_originality: parse error for %s — %s", alpha_id, exc)
            return 1.0

        subtrees = self._parser.extract_subtrees(ast)
        if not subtrees:
            return 1.0

        new_hashes = {self._parser.subtree_hash(st) for st in subtrees}

        existing_hashes: set[str] = set()
        for aid, hashes in self._fingerprints.items():
            if aid != alpha_id:
                existing_hashes.update(hashes)

        if not existing_hashes:
            return 1.0

        isomorphic_count = sum(1 for h in new_hashes if h in existing_hashes)
        total_count = len(new_hashes)
        return 1.0 - (isomorphic_count / total_count)

    def register_alpha(self, alpha_id: str, expr: str) -> None:
        """Register an alpha expression into the fingerprint database.

        After registration the fingerprint file is persisted to disk.
        If the expression cannot be parsed the call is silently skipped.
        """
        try:
            ast = self._parser.parse(expr)
        except ValueError as exc:
            logger.warning("register_alpha: parse error for %s — %s", alpha_id, exc)
            return

        subtrees = self._parser.extract_subtrees(ast)
        hashes = [self._parser.subtree_hash(st) for st in subtrees]
        self._fingerprints[alpha_id] = hashes
        self._save()

    # ---- persistence ------------------------------------------------------

    def _save(self) -> None:
        try:
            path = Path(self._fingerprint_path)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self._fingerprints, fh, indent=2, ensure_ascii=False)
        except OSError as exc:
            logger.error("_save: failed to write fingerprints — %s", exc)

    def _load(self) -> None:
        try:
            path = Path(self._fingerprint_path)
            if not path.exists():
                return
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                self._fingerprints = {
                    k: v for k, v in data.items() if isinstance(v, list)
                }
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("_load: failed to read fingerprints — %s", exc)
            self._fingerprints = {}
