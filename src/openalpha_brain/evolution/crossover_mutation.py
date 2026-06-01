from __future__ import annotations

import copy
import json
import logging
import math
import random
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, cast

from openalpha_brain.evolution.evolution_types import AlphaTrajectory
from openalpha_brain.evolution.trajectory_mutation import TrajectoryMutationResult, TrajectoryMutationV2
from openalpha_brain.generation.ast_originality import ASTNode, FASTEXPRParser, OriginalityChecker
from openalpha_brain.utils import extract_json_from_llm as _extract_json_from_llm

logger = logging.getLogger(__name__)

from openalpha_brain.cli.algo_monitor import AlgoMonitor
from openalpha_brain.utils.algo_logger import Timer, algo_log, log_call

_monitor = AlgoMonitor.get_instance()

_PRICE_FIELDS = ["close", "open", "high", "low", "vwap"]
_VOLUME_FIELDS = ["volume", "adv20", "cap"]
_FUNDAMENTAL_FIELDS = ["sales", "assets", "liabilities", "revenue", "equity", "debt"]
_ALL_FIELDS = _PRICE_FIELDS + _VOLUME_FIELDS + _FUNDAMENTAL_FIELDS

_RANK_OPS = {"rank", "ts_rank", "group_rank"}
_TS_SINGLE_OPS = {"ts_delta", "ts_sum", "ts_mean", "ts_std_dev", "ts_zscore",
                   "ts_delay", "ts_decay_linear", "ts_rank", "ts_arg_max", "ts_arg_min"}
_TS_DUAL_OPS = {"ts_corr", "ts_regression"}
_NEUTRALIZE_OPS = {"group_neutralize", "group_zscore"}
_SCALAR_OPS = {"abs", "log", "signed_power", "scale", "zscore", "normalize",
               "winsorize", "hump", "max", "min"}

_OPERATOR_SWAP_MAP: dict[str, list[str]] = {
    "rank": ["ts_rank", "group_rank", "zscore", "normalize"],
    "ts_rank": ["rank", "group_rank", "zscore"],
    "ts_delta": ["ts_sum", "ts_mean", "ts_delay"],
    "ts_sum": ["ts_delta", "ts_mean", "ts_decay_linear"],
    "ts_mean": ["ts_sum", "ts_decay_linear", "ts_delta"],
    "ts_std_dev": ["ts_zscore"],
    "ts_zscore": ["ts_std_dev", "zscore", "normalize"],
    "ts_decay_linear": ["ts_mean", "ts_sum"],
    "ts_delay": ["ts_delta"],
    "ts_corr": ["ts_regression"],
    "ts_regression": ["ts_corr"],
    "group_neutralize": ["group_zscore"],
    "group_zscore": ["group_neutralize"],
    "zscore": ["normalize", "rank", "ts_zscore"],
    "normalize": ["zscore", "rank", "scale"],
    "scale": ["normalize", "zscore"],
    "signed_power": ["abs", "log"],
    "abs": ["signed_power", "log"],
    "log": ["abs", "signed_power"],
    "winsorize": ["hump", "zscore"],
    "hump": ["winsorize"],
}

_FIELD_SWAP_MAP: dict[str, list[str]] = {
    "close": ["open", "high", "low", "vwap"],
    "open": ["close", "high", "low", "vwap"],
    "high": ["close", "open", "low", "vwap"],
    "low": ["close", "open", "high", "vwap"],
    "vwap": ["close", "open", "high", "low"],
    "volume": ["adv20", "cap"],
    "adv20": ["volume", "cap"],
    "cap": ["volume", "adv20"],
}

_LOOKBACK_SCALES = [2, 3, 5, 10, 20, 30, 60]

_BRAIN_FEEDBACK_OPERATOR_HINTS: dict[str, dict[str, list[str]]] = {
    "low_sharpe": {
        "rank": ["ts_rank", "signed_power"],
        "ts_mean": ["ts_delta", "ts_decay_linear"],
        "ts_delta": ["ts_decay_linear", "ts_sum"],
    },
    "high_turnover": {
        "ts_delta": ["ts_mean", "ts_decay_linear"],
        "rank": ["ts_decay_linear", "ts_mean"],
        "ts_sum": ["ts_mean", "ts_decay_linear"],
    },
}

_DIRECTION_KEYS = ["momentum", "mean_reversion", "volatility", "statistical", "volume", "interaction"]


@dataclass
class CrossoverResult:
    child_expression: str
    parent1_id: str
    parent2_id: str
    crossover_point: str
    originality_score: float


@dataclass
class MutationResult:
    mutated_expression: str
    original_id: str
    mutation_type: str
    mutation_description: str
    originality_score: float


@dataclass
class TrajectoryCrossoverResult:
    child_trajectory: AlphaTrajectory
    parent1_id: str
    parent2_id: str
    crossover_strategy: str
    complementary_segments: list[str] = dataclass_field(default_factory=list)
    llm_analysis: str = ""


class TrajectoryCrossover:

    MAX_CHILDREN: int = 3

    def __init__(self, llm_generate_fn: Callable[..., Awaitable[str]] | None = None,
                 originality_checker: OriginalityChecker | None = None):
        self._llm = llm_generate_fn
        self._originality_checker = originality_checker

    def _serialize_trajectory(self, traj: AlphaTrajectory, label: str) -> dict[str, Any]:
        return {
            "label": label,
            "hypothesis_direction": traj.hypothesis_direction,
            "hypothesis_mechanism": traj.hypothesis_mechanism,
            "final_status": traj.final_status,
            "final_sharpe": traj.final_sharpe,
            "expression_evolution": [
                {"version": i + 1, "expression": expr}
                for i, expr in enumerate(traj.expression_versions)
            ],
            "decision_points": traj.decision_points,
            "key_feedback": list(traj.brain_feedbacks) if traj.brain_feedbacks else [],
        }

    def _build_crossover_prompt(self, traj_a: AlphaTrajectory, traj_b: AlphaTrajectory) -> str:
        ser_a = json.dumps(self._serialize_trajectory(traj_a, "PARENT_A"), ensure_ascii=False, indent=2)
        ser_b = json.dumps(self._serialize_trajectory(traj_b, "PARENT_B"), ensure_ascii=False, indent=2)

        return f"""You are an expert quantitative researcher analyzing two alpha mining trajectories.
Each trajectory represents a complete end-to-end mining run: hypothesis → factor construction → backtest evaluation.

Your task: Identify complementary segments between the two trajectories and propose crossover strategies
to generate hybrid children that inherit the best from both parents.

Consider these segment types:
1. **hypothesis** — The economic intuition and market mechanism
2. **factor_structure** — The operator composition and data field choices
3. **construction_pattern** — How the expression was built step by step
4. **repair_behavior** — How failures were diagnosed and fixed

**PARENT A trajectory:**
{ser_a}

**PARENT B trajectory:**
{ser_b}

Analyze the trajectories and output a JSON with this structure:
```json
{{{{
  "complementarity_analysis": "Brief analysis of why these two trajectories are complementary",
  "complementary_segments": ["hypothesis", "factor_structure"],
  "crossover_strategies": [
    {{{{
      "strategy": "hypothesis_from_A_factor_from_B",
      "description": "What this strategy does",
      "merged_hypothesis": "Combined economic intuition",
      "merged_direction": "momentum",
      "merged_mechanism": "Combined mechanism description",
      "suggested_expression_pattern": "High-level description of the expected factor structure",
      "expected_benefit": "Why this combination should outperform individual parents"
    }}}}
  ]
}}}}
```

Output only the JSON, no other text."""

    async def crossover_trajectories(self,
                                     traj_a: AlphaTrajectory,
                                     traj_b: AlphaTrajectory,
                                     id_a: str = "",
                                     id_b: str = "",
                                     ) -> list[TrajectoryCrossoverResult]:
        results: list[TrajectoryCrossoverResult] = []

        if self._llm is not None:
            try:
                prompt = self._build_crossover_prompt(traj_a, traj_b)
                response = await self._llm(prompt)

                analysis = _extract_json_from_llm(response)
                if not analysis or not isinstance(analysis, dict):
                    logger.warning("CrossoverMutationEngine: failed to parse crossover analysis JSON")
                    return results

                strategies = analysis.get("crossover_strategies", [])
                complementary = cast(list[str], analysis.get("complementary_segments", []))

                for strategy in strategies[:self.MAX_CHILDREN]:
                    child = AlphaTrajectory(
                        hypothesis_direction=str(strategy.get("merged_direction", traj_a.hypothesis_direction)),
                        hypothesis_mechanism=str(strategy.get("merged_mechanism", "")),
                        expression_versions=[],
                        final_status="PENDING",
                    )

                    child.add_decision("crossover_strategy", str(strategy.get("strategy", "combined")), [])
                    child.add_decision("parent_a_direction", traj_a.hypothesis_direction, [])
                    child.add_decision("parent_b_direction", traj_b.hypothesis_direction, [])

                    if traj_a.expression_versions:
                        child.add_expression_version(f"/* A: {traj_a.expression_versions[-1][:80]}... */")
                    if traj_b.expression_versions:
                        child.add_expression_version(f"/* B: {traj_b.expression_versions[-1][:80]}... */")

                    results.append(TrajectoryCrossoverResult(
                        child_trajectory=child,
                        parent1_id=id_a,
                        parent2_id=id_b,
                        crossover_strategy=str(strategy.get("strategy", "unknown")),
                        complementary_segments=complementary,
                        llm_analysis=str(analysis.get("complementarity_analysis", "")),
                    ))

                    _monitor.record("STEP", "trajectory_crossover", "llm_crossover",
                                    f"parents={id_a},{id_b} strategy={strategy.get('strategy')}")

                if results:
                    logger.info(
                        "TrajectoryCrossover: generated %d children from parents %s, %s — complementary=%s",
                        len(results), id_a, id_b, complementary,
                    )

            except (OSError, ValueError, RuntimeError) as exc:
                logger.warning("TrajectoryCrossover: LLM analysis failed, using deterministic fallback — %s", exc)
                results = self._deterministic_crossover(traj_a, traj_b, id_a, id_b)
        else:
            results = self._deterministic_crossover(traj_a, traj_b, id_a, id_b)

        return results

    def _deterministic_crossover(self,
                                  traj_a: AlphaTrajectory,
                                  traj_b: AlphaTrajectory,
                                  id_a: str = "",
                                  id_b: str = "",
                                  ) -> list[TrajectoryCrossoverResult]:
        results: list[TrajectoryCrossoverResult] = []

        child1 = AlphaTrajectory(
            hypothesis_direction=traj_a.hypothesis_direction,
            hypothesis_mechanism=traj_b.hypothesis_mechanism,
            final_status="PENDING",
        )
        for expr in traj_a.expression_versions[:1]:
            child1.add_expression_version(expr)
        for expr in traj_b.expression_versions[-1:]:
            child1.add_expression_version(expr)
        results.append(TrajectoryCrossoverResult(
            child_trajectory=child1,
            parent1_id=id_a,
            parent2_id=id_b,
            crossover_strategy="hypothesis_from_A_mechanism_from_B",
            complementary_segments=["hypothesis", "factor_mechanism"],
        ))

        child2 = AlphaTrajectory(
            hypothesis_direction=traj_b.hypothesis_direction,
            hypothesis_mechanism=traj_a.hypothesis_mechanism,
            final_status="PENDING",
        )
        for expr in traj_b.expression_versions[:1]:
            child2.add_expression_version(expr)
        for expr in traj_a.expression_versions[-1:]:
            child2.add_expression_version(expr)
        results.append(TrajectoryCrossoverResult(
            child_trajectory=child2,
            parent1_id=id_b,
            parent2_id=id_a,
            crossover_strategy="hypothesis_from_B_mechanism_from_A",
            complementary_segments=["hypothesis", "factor_mechanism"],
        ))

        if traj_a.decision_points and traj_b.decision_points:
            child3 = AlphaTrajectory(
                hypothesis_direction=traj_a.hypothesis_direction,
                hypothesis_mechanism=f"{traj_a.hypothesis_mechanism} × {traj_b.hypothesis_mechanism}",
                decision_points=traj_a.decision_points[:1] + traj_b.decision_points[-1:],
                final_status="PENDING",
            )
            results.append(TrajectoryCrossoverResult(
                child_trajectory=child3,
                parent1_id=id_a,
                parent2_id=id_b,
                crossover_strategy="combined_decision_patterns",
                complementary_segments=["decision_patterns"],
            ))

        _monitor.record("STEP", "trajectory_crossover", "deterministic_crossover",
                        f"parents={id_a},{id_b} children={len(results)}")

        return results[:self.MAX_CHILDREN]


def _ast_to_string(node: ASTNode) -> str:
    if node.is_leaf:
        return node.value
    if node.op == "neg":
        return f"-{_ast_to_string(node.children[0])}"
    child_strs = [_ast_to_string(c) for c in node.children]
    if node.op in ("+", "-", "*", "/"):
        left = child_strs[0]
        right = child_strs[1]
        if node.op in ("+", "-"):
            return f"{left} {node.op} {right}"
        return f"{left} {node.op} {right}"
    return f"{node.op}({', '.join(child_strs)})"


def _collect_subtree_paths(node: ASTNode, path: tuple[int, ...] = (),
                           result: list[tuple[ASTNode, tuple[int, ...]]] | None = None
                           ) -> list[tuple[ASTNode, tuple[int, ...]]]:
    if result is None:
        result = []
    result.append((node, path))
    for i, child in enumerate(node.children):
        _collect_subtree_paths(child, path + (i,), result)
    return result


def _get_node_at_path(root: ASTNode, path: tuple[int, ...]) -> ASTNode:
    node = root
    for idx in path:
        node = node.children[idx]
    return node


def _set_node_at_path(root: ASTNode, path: tuple[int, ...], replacement: ASTNode) -> ASTNode:
    new_root = copy.deepcopy(root)
    if not path:
        return copy.deepcopy(replacement)
    node = new_root
    for idx in path[:-1]:
        node = node.children[idx]
    node.children[path[-1]] = copy.deepcopy(replacement)
    return new_root


def _operator_category(op: str) -> str | None:
    if op in _RANK_OPS:
        return "rank"
    if op in _TS_SINGLE_OPS:
        return "ts_single"
    if op in _TS_DUAL_OPS:
        return "ts_dual"
    if op in _NEUTRALIZE_OPS:
        return "neutralize"
    if op in _SCALAR_OPS:
        return "scalar"
    if op in ("+", "-", "*", "/"):
        return "arithmetic"
    return None


def _subtrees_compatible(sub1: ASTNode, sub2: ASTNode) -> bool:
    if sub1.is_leaf and sub2.is_leaf:
        if sub1.is_field and sub2.is_field:
            return True
        if sub1.is_number and sub2.is_number:
            return True
        return False
    if sub1.is_leaf or sub2.is_leaf:
        return False
    cat1 = _operator_category(sub1.op)
    cat2 = _operator_category(sub2.op)
    if cat1 is not None and cat1 == cat2:
        return True
    if cat1 == "scalar" and cat2 == "rank":
        return True
    if cat1 == "rank" and cat2 == "scalar":
        return True
    return False


def _compute_originality(expr: str, checker: OriginalityChecker | None) -> float:
    if checker is None:
        return 0.5
    try:
        return checker.check_originality("__crossover_mut__", expr)
    except (OSError, ValueError, RuntimeError):
        return 0.5


class SemanticCrossover:

    MAX_CHILDREN: int = 3
    EPSILON: float = 1e-6

    def __init__(self, originality_checker: OriginalityChecker | None = None,
                 llm_generate_fn: Callable[..., Awaitable[str]] | None = None):
        self._originality_checker = originality_checker
        self._parser = FASTEXPRParser()
        self._llm = llm_generate_fn
        self._success_stats: dict[str, float] = {"total_calls": 0.0, "successful_calls": 0.0, "total_children": 0.0}
        self._skip_stats: dict[str, int] = {"duplicate_parent": 0, "duplicate_seen": 0, "llm_fallback": 0}

    @algo_log()
    async def crossover(self, expr1: str, expr2: str,
                  id1: str = "", id2: str = "", context: dict | None = None) -> list[CrossoverResult]:
        self._success_stats["total_calls"] += 1.0
        logger.info(
            "[SEMANTIC_XOVER] 开始语义交叉 | parents=[%s:%s, %s:%s] | expr1长度=%d expr2长度=%d",
            id1, expr1[:60], id2, expr2[:60], len(expr1), len(expr2)
        )

        _monitor.record("STEP", "crossover", "semantic_crossover", f"parents={id1},{id2}")

        try:
            with Timer("parse_blocks", "crossover"):
                blocks_a = self._extract_signal_block(expr1)
                blocks_b = self._extract_signal_block(expr2)

            if not blocks_a.get("structure_valid") or not blocks_b.get("structure_valid"):
                logger.warning(
                    "[DEFENSIVE_LOG] 段解析失败 | parent1_valid=%s parent2_valid=%s | "
                    "使用确定性回退",
                    blocks_a.get("structure_valid", False),
                    blocks_b.get("structure_valid", False),
                )
                return self._deterministic_fallback(expr1, expr2, id1, id2)

            logger.info(
                "[SEMANTIC_XOVER] 段解析成功 | parent1_block_a=%s | parent2_block_a=%s",
                blocks_a["block_a"][:60],
                blocks_b["block_a"][:60],
            )

            results: list[CrossoverResult] = []
            seen_exprs: set[str] = set()

            if self._llm is not None:
                with Timer("llm_semantic_analysis", "crossover"):
                    llm_results = await self._llm_semantic_analysis(
                        signal_a=blocks_a["block_a"],
                        signal_b=blocks_b["block_a"],
                        context=context or {},
                        parent_a_full=expr1,
                        parent_b_full=expr2,
                    )

                for child_signal in llm_results[:self.MAX_CHILDREN]:
                    child_expr = self._reassemble_blocks(
                        new_block_a=child_signal,
                        block_b=blocks_a["block_b"],
                        block_c=blocks_a["block_c"],
                        original_expr=expr1,
                    )

                    validation = self._validate_crossover_result(
                        child=child_expr,
                        parent_a=expr1,
                        parent_b=expr2,
                        seen=seen_exprs,
                    )

                    if not validation["valid"]:
                        if validation["reason"] == "duplicate_parent":
                            self._skip_stats["duplicate_parent"] += 1
                            logger.warning(
                                "[SEMANTIC_XOVER] ⚠️ 子代与父代相同，跳过 | child_expr=%s",
                                child_expr[:80],
                            )
                        elif validation["reason"] == "duplicate_seen":
                            self._skip_stats["duplicate_seen"] += 1
                            logger.debug(
                                "[DEFENSIVE_LOG] 跳过重复已见变体 | child_expr=%s",
                                child_expr[:80],
                            )
                        continue

                    seen_exprs.add(child_expr)
                    orig_score = _compute_originality(child_expr, self._originality_checker)

                    results.append(CrossoverResult(
                        child_expression=child_expr,
                        parent1_id=id1,
                        parent2_id=id2,
                        crossover_point=f"LLM semantic: {child_signal[:50]}...",
                        originality_score=orig_score,
                    ))

                if not results:
                    self._skip_stats["llm_fallback"] += 1
                    logger.warning(
                        "[SEMANTIC_XOVER] LLM 未产生有效子代，使用适应度回退"
                    )
                    return self._fitness_fallback(expr1, expr2, id1, id2, context)
            else:
                self._skip_stats["llm_fallback"] += 1
                logger.info("[SEMANTIC_XOVER] 未配置LLM，使用确定性回退")
                return self._deterministic_fallback(expr1, expr2, id1, id2)

            if results:
                self._success_stats["successful_calls"] += 1.0
                self._success_stats["total_children"] += float(len(results))

            orig_scores = [r.originality_score for r in results]
            score_range = f"{min(orig_scores):.3f}-{max(orig_scores):.3f}" if orig_scores else "N/A"

            logger.info(
                "[SEMANTIC_XOVER] 完成 | 输出变体数=%d | 质量分数范围=%s | "
                "跳过(与父代相同)=%d | 跳过(重复已见)=%d | 成功率=%.1f%%",
                len(results),
                score_range,
                self._skip_stats["duplicate_parent"],
                self._skip_stats["duplicate_seen"],
                (self._success_stats["successful_calls"] / max(1.0, self._success_stats["total_calls"])) * 100,
            )

            _monitor.record("PASS" if results else "FAIL", "crossover", "semantic_crossover",
                           f"children={len(results)} method=llm_semantic")

            return results

        except (OSError, ValueError, RuntimeError) as exc:
            logger.error("[SEMANTIC_XOVER] 异常 | error=%s", exc, exc_info=True)
            return []

    @algo_log()
    async def _llm_semantic_analysis(self, signal_a: str, signal_b: str,
                                     context: dict, parent_a_full: str = "",
                                     parent_b_full: str = "") -> list[str]:
        prompt = self._build_semantic_prompt(signal_a, signal_b, context,
                                            parent_a_full, parent_b_full)

        try:
            response = await self._llm(prompt)
            if not response:
                logger.warning("[SEMANTIC_XOVER] LLM 返回空响应")
                return []

            parsed = _extract_json_from_llm(response)
            if not parsed or not isinstance(parsed, dict):
                logger.warning("[SEMANTIC_XOVER] 无法解析 LLM JSON 响应")
                return []

            child_signals = parsed.get("child_signals", [])
            if not child_signals or not isinstance(child_signals, list):
                logger.warning("[SEMANTIC_XOVER] LLM 响应中无有效子代信号")
                return []

            validated_signals = []
            for signal in child_signals:
                if not isinstance(signal, str) or not signal.strip():
                    continue
                try:
                    self._parser.parse(signal.strip())
                    validated_signals.append(signal.strip())
                except ValueError:
                    logger.debug("[DEFENSIVE_LOG] LLM 生成的信号段 AST 校验失败: %s", signal[:80])
                    continue

            logger.info(
                "[SEMANTIC_XOVER] LLM 语义分析完成 | 输入信号数=2 | 有效输出信号数=%d",
                len(validated_signals),
            )

            return validated_signals

        except (OSError, ValueError, RuntimeError) as exc:
            logger.warning("[SEMANTIC_XOVER] LLM 调用异常 | error=%s", exc)
            return []

    def _build_semantic_prompt(self, signal_a: str, signal_b: str,
                               context: dict, parent_a_full: str = "",
                               parent_b_full: str = "") -> str:
        direction = context.get("direction", "unknown")
        sharpe_a = context.get("sharpe_a", "N/A")
        sharpe_b = context.get("sharpe_b", "N/A")

        return f"""You are a quantitative finance researcher specializing in alpha factor design for the WorldQuant platform.

## Task: Semantic Crossover of Signal Blocks (Block A Only)

You must perform **semantic crossover** on two parent alpha expressions, following these strict rules:

### INPUT DATA
**Parent A Signal Block (Block A):**
```
{signal_a}
```
Full Expression: `{parent_a_full[:100]}...`
Sharpe: {sharpe_a}

**Parent B Signal Block (Block A):**
```
{signal_b}
```
Full Expression: `{parent_b_full[:100]}...`
Sharpe: {sharpe_b}

Hypothesis Direction: {direction}

### YOUR ANALYSIS TASKS
1. **Semantic Analysis**: Identify what financial logic each signal encodes:
   - Momentum (趋势跟踪)?
   - Mean Reversion (均值回归)?
   - Value (价值因子)?
   - Volatility (波动率)?
   - Volume (成交量)?
   - Statistical Arbitrage (统计套利)?

2. **Field Family Complementarity**: Check if the two signals use complementary data fields:
   - Price fields: close, open, high, low, vwap
   - Volume fields: volume, adv20, cap
   - Fundamental fields: sales, assets, revenue, equity, debt
   - Alternative data: analyst estimates, supply chain, news sentiment

3. **Generate Hybrid Child Signals**: Create 1-3 NEW signal block expressions that combine the best semantic properties of both parents.

### STRICT CONSTRAINTS (MUST FOLLOW)
1. **ONLY modify Block A (Signal Block)** — NEVER touch Block B (neutralize) or Block C (decay)
2. Output MUST be valid WorldQuant expression syntax using operators like:
   - Time-series: ts_delta, ts_mean, ts_rank, ts_decay_linear, ts_zscore, ts_corr
   - Cross-sectional: rank, group_neutralize, group_zscore
   - Scalar: abs, log, signed_power, scale, normalize
   - Arithmetic: +, -, *, /
3. Each child signal must be **semantically meaningful** (not random operator combinations)
4. Child should inherit strengths from BOTH parents (complementary recombination)
5. Do NOT simply copy one parent's signal — must be a genuine hybrid

### OUTPUT FORMAT (JSON ONLY)
```json
{{
  "semantic_analysis": {{
    "parent_a_type": "momentum|mean_reversion|value|volatility|volume|statistical",
    "parent_b_type": "momentum|mean_reversion|value|volatility|volume|statistical",
    "field_complementarity": "description of how fields complement each other",
    "recombination_strategy": "brief explanation of the crossover strategy used"
  }},
  "child_signals": [
    "valid WQ expression for hybrid child signal 1",
    "valid WQ expression for hybrid child signal 2"
  ]
}}
```

Output ONLY valid JSON. No explanations outside JSON."""

    @algo_log()
    def _extract_signal_block(self, expression: str) -> dict:
        result = {
            "block_a": "",
            "block_b": "",
            "block_c": "",
            "structure_valid": False,
        }

        if not expression or not expression.strip():
            return result

        expr = expression.strip()

        has_decay = "ts_decay_linear" in expr
        has_neutralize = "group_neutralize" in expr

        if has_decay and has_neutralize:
            decay_match = re.match(
                r'ts_decay_linear\s*\(\s*(.+?)\s*,\s*(\d+)\s*\)',
                expr,
                re.DOTALL,
            )
            if decay_match:
                inner_expr = decay_match.group(1)
                decay_window = decay_match.group(2)
                result["block_c"] = f"ts_decay_linear(..., {decay_window})"

                neutralize_match = re.match(
                    r'group_neutralize\s*\(\s*(.+?)\s*,\s*(\w+)\s*\)',
                    inner_expr,
                    re.DOTALL,
                )
                if neutralize_match:
                    signal_expr = neutralize_match.group(1)
                    group_field = neutralize_match.group(2)
                    result["block_b"] = f"group_neutralize(..., {group_field})"
                    result["block_a"] = signal_expr
                    result["structure_valid"] = True
                else:
                    result["block_a"] = inner_expr
                    result["block_b"] = ""
                    result["structure_valid"] = True
        elif has_neutralize:
            neutralize_match = re.match(
                r'group_neutralize\s*\(\s*(.+?)\s*,\s*(\w+)\s*\)',
                expr,
                re.DOTALL,
            )
            if neutralize_match:
                result["block_a"] = neutralize_match.group(1)
                result["block_b"] = f"group_neutralize(..., {neutralize_match.group(2)})"
                result["structure_valid"] = True
        else:
            result["block_a"] = expr
            result["structure_valid"] = True

        return result

    @algo_log()
    def _reassemble_blocks(self, new_block_a: str, block_b: str,
                          block_c: str, original_expr: str) -> str:
        if block_c and block_b:
            return f"ts_decay_linear(group_neutralize({new_block_a}, industry), 20)"
        elif block_b:
            return f"group_neutralize({new_block_a}, industry)"
        elif block_c:
            return f"ts_decay_linear({new_block_a}, 20)"
        else:
            return new_block_a

    @algo_log()
    def _validate_crossover_result(self, child: str, parent_a: str,
                                   parent_b: str, seen: set[str]) -> dict:
        if child == parent_a or child == parent_b:
            return {"valid": False, "reason": "duplicate_parent"}

        if child in seen:
            return {"valid": False, "reason": "duplicate_seen"}

        try:
            self._parser.parse(child)
        except ValueError:
            return {"valid": False, "reason": "parse_error"}

        return {"valid": True, "reason": "ok"}

    @algo_log()
    def _deterministic_fallback(self, expr1: str, expr2: str,
                                id1: str = "", id2: str = "") -> list[CrossoverResult]:
        logger.info("[SEMANTIC_XOVER] 使用确定性回退策略")

        blocks1 = self._extract_signal_block(expr1)
        blocks2 = self._extract_signal_block(expr2)

        results: list[CrossoverResult] = []
        seen: set[str] = set()

        if blocks1.get("structure_valid") and blocks2.get("structure_valid"):
            hybrid_signal_1 = f"ts_rank({blocks1['block_a']}, 5) + ts_delta({blocks2['block_a']}, 5)"
            child1 = self._reassemble_blocks(hybrid_signal_1, blocks1["block_b"],
                                            blocks1["block_c"], expr1)

            validation1 = self._validate_crossover_result(child1, expr1, expr2, seen)
            if validation1["valid"]:
                seen.add(child1)
                orig1 = _compute_originality(child1, self._originality_checker)
                results.append(CrossoverResult(
                    child_expression=child1,
                    parent1_id=id1,
                    parent2_id=id2,
                    crossover_point="deterministic: rank(A) + delta(B)",
                    originality_score=orig1,
                ))

            hybrid_signal_2 = f"ts_corr({blocks1['block_a']}, {blocks2['block_a']}, 10)"
            child2 = self._reassemble_blocks(hybrid_signal_2, blocks2["block_b"],
                                            blocks2["block_c"], expr2)

            validation2 = self._validate_crossover_result(child2, expr1, expr2, seen)
            if validation2["valid"]:
                seen.add(child2)
                orig2 = _compute_originality(child2, self._originality_checker)
                results.append(CrossoverResult(
                    child_expression=child2,
                    parent1_id=id1,
                    parent2_id=id2,
                    crossover_point="deterministic: corr(A, B)",
                    originality_score=orig2,
                ))

        _monitor.record("PASS" if results else "WARN", "crossover", "semantic_crossover",
                       f"children={len(results)} method=deterministic_fallback")

        return results[:self.MAX_CHILDREN]

    @algo_log()
    def _fitness_fallback(self, expr1: str, expr2: str, id1: str = "", id2: str = "",
                         context: dict | None = None) -> list[CrossoverResult]:
        logger.info("[SEMANTIC_XOVER] 使用适应度优先回退")

        sharpe_a = (context or {}).get("sharpe_a", 0.0) or 0.0
        sharpe_b = (context or {}).get("sharpe_b", 0.0) or 0.0

        better_expr = expr1 if sharpe_a >= sharpe_b else expr2
        better_id = id1 if sharpe_a >= sharpe_b else id2

        logger.info(
            "[SEMANTIC_XOVER] 选择更高适应度父代 | selected=%s | sharpe_a=%.2f sharpe_b=%.2f",
            better_id, sharpe_a, sharpe_b,
        )

        orig_score = _compute_originality(better_expr, self._originality_checker)

        return [CrossoverResult(
            child_expression=better_expr,
            parent1_id=id1,
            parent2_id=id2,
            crossover_point=f"fitness_fallback: better_parent ({better_id})",
            originality_score=orig_score,
        )]

    async def semantic_crossover_via_llm(
        self,
        llm_generate: Callable,
        expr1: str, expr2: str,
        context1: dict | None = None,
        context2: dict | None = None,
    ) -> list[CrossoverResult]:
        context1 = context1 or {}
        context2 = context2 or {}

        merged_context = {
            "direction": context1.get("direction", context2.get("direction", "unknown")),
            "sharpe_a": context1.get("sharpe", "N/A"),
            "sharpe_b": context2.get("sharpe", "N/A"),
        }

        _monitor.record("STEP", "crossover", "llm_semantic_crossover_legacy", "parents")

        try:
            return await self.crossover(
                expr1=expr1,
                expr2=expr2,
                id1=context1.get("id", ""),
                id2=context2.get("id", ""),
                context=merged_context,
            )
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, json.JSONDecodeError):
            return []


class GradientMutation:

    UCB_C: float = 1.41
    DIVERSITY_WINDOW: int = 10
    DIVERSITY_LOW_THRESHOLD: float = 0.15
    DIVERSITY_EPSILON: float = 1e-6
    MAX_OPERATOR_SWAP_VARIANTS: int = 3
    MAX_FIELD_SWAP_VARIANTS: int = 3
    MAX_PARAMETER_TUNE_VARIANTS: int = 2
    MAX_STRUCTURE_CHANGE_VARIANTS: int = 4
    FITNESS_WINDOW_MIN_SAMPLES: int = 3
    FITNESS_WINDOW_MAX_MULTIPLIER: int = 3
    BASE_VARIANT_LIMIT: dict[str, int] = {
        "operator_swap": 3,
        "field_swap": 3,
        "parameter_tune": 2,
        "structure_change": 4,
    }
    BOOST_VARIANT_LIMIT: dict[str, int] = {
        "operator_swap": 5,
        "field_swap": 5,
        "parameter_tune": 3,
        "structure_change": 6,
    }

    def __init__(self, originality_checker: OriginalityChecker | None = None):
        self._originality_checker = originality_checker
        self._parser = FASTEXPRParser()
        self._operator_stats: dict[str, dict[str, float]] = {
            "operator_swap": {"visits": 0.0, "reward": 0.0},
            "field_swap": {"visits": 0.0, "reward": 0.0},
            "parameter_tune": {"visits": 0.0, "reward": 0.0},
            "structure_change": {"visits": 0.0, "reward": 0.0},
        }
        self._fitness_window: list[float] = []
        self._total_visits: float = 0.0
        self._validate_params()

    def _validate_params(self) -> None:
        if self.UCB_C <= 0:
            raise ValueError(f"UCB_C must be positive, got {self.UCB_C}")
        if self.DIVERSITY_WINDOW < 2:
            raise ValueError(f"DIVERSITY_WINDOW must be >= 2, got {self.DIVERSITY_WINDOW}")
        if not (0.0 < self.DIVERSITY_LOW_THRESHOLD <= 1.0):
            raise ValueError(f"DIVERSITY_LOW_THRESHOLD must be in (0,1], got {self.DIVERSITY_LOW_THRESHOLD}")
        if self.DIVERSITY_EPSILON <= 0 or self.DIVERSITY_EPSILON >= 0.01:
            raise ValueError(f"DIVERSITY_EPSILON must be in (0, 0.01), got {self.DIVERSITY_EPSILON}")
        if self.FITNESS_WINDOW_MIN_SAMPLES < 2:
            raise ValueError(f"FITNESS_WINDOW_MIN_SAMPLES must be >= 2, got {self.FITNESS_WINDOW_MIN_SAMPLES}")
        for name, limit in self.BASE_VARIANT_LIMIT.items():
            if limit < 1:
                raise ValueError(f"BASE_VARIANT_LIMIT[{name}] must be >= 1, got {limit}")
        for name, limit in self.BOOST_VARIANT_LIMIT.items():
            if limit < self.BASE_VARIANT_LIMIT.get(name, 1):
                raise ValueError(f"BOOST_VARIANT_LIMIT[{name}] must be >= BASE_VARIANT_LIMIT[{name}], got {limit}")

    def record_population_fitness(self, fitness_values: list[float]) -> None:
        self._fitness_window.extend(fitness_values)
        max_len = self.DIVERSITY_WINDOW * self.FITNESS_WINDOW_MAX_MULTIPLIER
        if len(self._fitness_window) > max_len:
            self._fitness_window = self._fitness_window[-max_len:]

    def _compute_diversity_score(self) -> float:
        if len(self._fitness_window) < self.FITNESS_WINDOW_MIN_SAMPLES:
            return 1.0
        window = self._fitness_window[-self.DIVERSITY_WINDOW:]
        mean_val = sum(window) / len(window)
        if mean_val == 0:
            logger.warning(
                "GradientMutation._compute_diversity_score: fitness mean=0, "
                "diversity=0, using default strategy branch"
            )
            return 0.0
        variance = sum((x - mean_val) ** 2 for x in window) / len(window)
        cv = (variance ** 0.5) / (abs(mean_val) + 1e-8)
        diversity = min(1.0, cv)
        safe_diversity = max(diversity, self.DIVERSITY_EPSILON)
        if diversity == 0.0:
            logger.warning(
                "GradientMutation._compute_diversity_score: diversity=0 detected, "
                "safe_diversity clamped to %.1e, falling back to default strategy",
                safe_diversity,
            )
        return diversity

    def _safe_diversity(self) -> float:
        raw = self._compute_diversity_score()
        return max(raw, self.DIVERSITY_EPSILON)

    def _ucb1_select_operators(self) -> list[tuple[str, float]]:
        scores: list[tuple[str, float]] = []
        log_total = math.log(max(1.0, self._total_visits))
        for op_name, stats in self._operator_stats.items():
            if stats["visits"] == 0:
                scores.append((op_name, float("inf")))
            else:
                exploitation = stats["reward"] / stats["visits"]
                exploration = self.UCB_C * math.sqrt(log_total / stats["visits"])
                scores.append((op_name, exploitation + exploration))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def record_operator_result(self, op_type: str, success: bool) -> None:
        if op_type not in self._operator_stats:
            return
        self._operator_stats[op_type]["visits"] += 1.0
        self._total_visits += 1.0
        if success:
            self._operator_stats[op_type]["reward"] += 1.0

    @algo_log(log_args_to_skip=("self", "brain_feedback"))
    def mutate(self, expr: str, brain_feedback: dict | None = None,
               original_id: str = "") -> list[MutationResult]:
        results: list[MutationResult] = []
        seen: set[str] = set()

        _monitor.record("STEP", "mutation", "gradient_mutation", f"id={original_id}")

        diversity_score = self._compute_diversity_score()
        safe_diversity = self._safe_diversity()
        is_low_diversity = diversity_score < self.DIVERSITY_LOW_THRESHOLD
        variant_limits = self.BOOST_VARIANT_LIMIT if is_low_diversity else self.BASE_VARIANT_LIMIT

        ranked_operators = self._ucb1_select_operators()

        log_call(
            "GradientMutation.mutate",
            input={"expr_len": len(expr), "diversity": round(diversity_score, 4),
                   "is_low_diversity": is_low_diversity},
            extra={"strategy_order": [n for n, _ in ranked_operators]},
        )

        if is_low_diversity:
            logger.info(
                "mutation: low diversity detected (cv=%.4f), boosting variant limits "
                "operator_swap=%d field_swap=%d parameter_tune=%d structure_change=%d",
                diversity_score,
                variant_limits["operator_swap"],
                variant_limits["field_swap"],
                variant_limits["parameter_tune"],
                variant_limits["structure_change"],
            )

        op_methods: dict[str, object] = {
            "operator_swap": self._operator_swap,
            "field_swap": self._field_swap,
            "parameter_tune": self._parameter_tune,
            "structure_change": self._structure_change,
        }

        for op_name, ucb_score in ranked_operators:
            method = op_methods.get(op_name)
            if method is None:
                continue
            limit = variant_limits.get(op_name, 3)
            method_fn = cast(Callable[..., list[MutationResult]], method)

            self._operator_stats[op_name]["visits"] += 1.0
            self._total_visits += 1.0

            prev_count = len(results)
            try:
                for result in method_fn(expr, brain_feedback, original_id):
                    if result.mutated_expression not in seen:
                        seen.add(result.mutated_expression)
                        results.append(result)
                        if len(results) - prev_count >= limit:
                            break
            except (OSError, ValueError, RuntimeError) as exc:
                logger.warning(
                    "GradientMutation.mutate: operator %s failed with error, skipping — %s",
                    op_name, exc,
                )
                log_call(
                    f"GradientMutation.{op_name}",
                    error=str(exc),
                    elapsed_ms=0,
                    extra={"recovery": "skipped_operator"},
                )

        log_call(
            "GradientMutation.mutate",
            output={"variants": len(results), "diversity": round(safe_diversity, 4)},
            extra={"unique_exprs": len(seen)},
        )

        _monitor.record(
            "PASS" if results else "FAIL", "mutation", "gradient_mutation",
            f"variants={len(results)} diversity={diversity_score:.3f}"
        )

        return results

    def _operator_swap(self, expr: str, brain_feedback: dict | None,
                       original_id: str) -> list[MutationResult]:
        try:
            ast = self._parser.parse(expr)
        except ValueError as exc:
            logger.warning("GradientMutation._operator_swap: AST parse failed — %s", exc)
            log_call("GradientMutation._operator_swap", error=f"parse_failed:{exc}", elapsed_ms=0)
            return []

        results: list[MutationResult] = []
        subtrees = _collect_subtree_paths(ast)

        feedback_hints: dict[str, list[str]] = {}
        if brain_feedback:
            sharpe = brain_feedback.get("sharpe")
            turnover = brain_feedback.get("turnover")
            if sharpe is not None and sharpe < 1.0:
                feedback_hints = _BRAIN_FEEDBACK_OPERATOR_HINTS.get("low_sharpe", {})
            if turnover is not None and turnover > 0.7:
                for k, v in _BRAIN_FEEDBACK_OPERATOR_HINTS.get("high_turnover", {}).items():
                    feedback_hints.setdefault(k, []).extend(v)

        for sub, path in subtrees:
            if sub.is_leaf or not sub.op:
                continue
            candidates = list(feedback_hints.get(sub.op, []))
            default_candidates = _OPERATOR_SWAP_MAP.get(sub.op, [])
            for c in default_candidates:
                if c not in candidates:
                    candidates.append(c)

            if not candidates:
                continue

            swap_to = random.choice(candidates)
            try:
                new_sub = ASTNode(op=swap_to, children=[copy.deepcopy(c) for c in sub.children])
                new_ast = _set_node_at_path(ast, path, new_sub)
                new_expr = _ast_to_string(new_ast)
            except (OSError, ValueError, RuntimeError) as exc:
                logger.warning("GradientMutation._operator_swap: AST rebuild failed for %s → %s — %s",
                               sub.op, swap_to, exc)
                continue

            orig_score = _compute_originality(new_expr, self._originality_checker)

            results.append(MutationResult(
                mutated_expression=new_expr,
                original_id=original_id,
                mutation_type="operator_swap",
                mutation_description=f"{sub.op} → {swap_to}",
                originality_score=orig_score,
            ))

            if len(results) >= self.MAX_OPERATOR_SWAP_VARIANTS:
                break

        return results

    def _field_swap(self, expr: str, brain_feedback: dict | None,
                    original_id: str) -> list[MutationResult]:
        try:
            ast = self._parser.parse(expr)
        except ValueError as exc:
            logger.warning("GradientMutation._field_swap: AST parse failed — %s", exc)
            log_call("GradientMutation._field_swap", error=f"parse_failed:{exc}", elapsed_ms=0)
            return []

        results: list[MutationResult] = []
        subtrees = _collect_subtree_paths(ast)

        for sub, path in subtrees:
            if not sub.is_field:
                continue
            field_name = sub.value.lower()
            alternatives = _FIELD_SWAP_MAP.get(field_name)
            if not alternatives:
                continue

            swap_to = random.choice(alternatives)
            try:
                new_sub = ASTNode(value=swap_to)
                new_ast = _set_node_at_path(ast, path, new_sub)
                new_expr = _ast_to_string(new_ast)
            except (OSError, ValueError, RuntimeError) as exc:
                logger.warning("GradientMutation._field_swap: AST rebuild failed for %s → %s — %s",
                               field_name, swap_to, exc)
                continue

            orig_score = _compute_originality(new_expr, self._originality_checker)

            results.append(MutationResult(
                mutated_expression=new_expr,
                original_id=original_id,
                mutation_type="field_swap",
                mutation_description=f"{field_name} → {swap_to}",
                originality_score=orig_score,
            ))

            if len(results) >= self.MAX_FIELD_SWAP_VARIANTS:
                break

        return results

    def _parameter_tune(self, expr: str, brain_feedback: dict | None,
                        original_id: str) -> list[MutationResult]:
        try:
            ast = self._parser.parse(expr)
        except ValueError as exc:
            logger.warning("GradientMutation._parameter_tune: AST parse failed — %s", exc)
            log_call("GradientMutation._parameter_tune", error=f"parse_failed:{exc}", elapsed_ms=0)
            return []

        results: list[MutationResult] = []
        subtrees = _collect_subtree_paths(ast)

        for sub, path in subtrees:
            if not sub.is_number:
                continue
            try:
                current_val = float(sub.value)
            except (ValueError, TypeError):
                continue

            if current_val <= 0 or current_val != int(current_val):
                continue

            current_int = int(current_val)
            candidates = []
            for scale in _LOOKBACK_SCALES:
                if scale != current_int:
                    candidates.append(scale)

            if not candidates:
                continue

            new_val = random.choice(candidates)
            try:
                new_sub = ASTNode(value=str(new_val))
                new_ast = _set_node_at_path(ast, path, new_sub)
                new_expr = _ast_to_string(new_ast)
            except (ValueError, SyntaxError, TypeError):
                pass
                continue

            orig_score = _compute_originality(new_expr, self._originality_checker)

            results.append(MutationResult(
                mutated_expression=new_expr,
                original_id=original_id,
                mutation_type="parameter_tune",
                mutation_description=f"lookback {current_int} → {new_val}",
                originality_score=orig_score,
            ))

            if len(results) >= self.MAX_PARAMETER_TUNE_VARIANTS:
                break

        return results

    def _structure_change(self, expr: str, brain_feedback: dict | None,
                          original_id: str) -> list[MutationResult]:
        try:
            ast = self._parser.parse(expr)
        except ValueError as exc:
            logger.warning("GradientMutation._structure_change: AST parse failed — %s", exc)
            log_call("GradientMutation._structure_change", error=f"parse_failed:{exc}", elapsed_ms=0)
            return []

        results: list[MutationResult] = []

        try:
            wrapped = ASTNode(op="rank", children=[copy.deepcopy(ast)])
            wrapped_expr = _ast_to_string(wrapped)
            orig_score = _compute_originality(wrapped_expr, self._originality_checker)
            results.append(MutationResult(
                mutated_expression=wrapped_expr,
                original_id=original_id,
                mutation_type="structure_change",
                mutation_description="added rank() wrapper",
                originality_score=orig_score,
            ))
        except (OSError, ValueError, RuntimeError) as exc:
            logger.warning("GradientMutation._structure_change: rank wrapper failed — %s", exc)

        if ast.op == "group_neutralize" and len(ast.children) >= 1:
            try:
                inner = copy.deepcopy(ast.children[0])
                unwrapped_expr = _ast_to_string(inner)
                if unwrapped_expr != expr:
                    orig_score2 = _compute_originality(unwrapped_expr, self._originality_checker)
                    results.append(MutationResult(
                        mutated_expression=unwrapped_expr,
                        original_id=original_id,
                        mutation_type="structure_change",
                        mutation_description="removed group_neutralize() wrapper",
                        originality_score=orig_score2,
                    ))
            except (OSError, ValueError, RuntimeError) as exc:
                logger.warning("GradientMutation._structure_change: unwrap neutralize failed — %s", exc)

        if ast.op != "group_neutralize":
            try:
                neutralized = ASTNode(
                    op="group_neutralize",
                    children=[copy.deepcopy(ast), ASTNode(value="industry")],
                )
                neutralized_expr = _ast_to_string(neutralized)
                orig_score3 = _compute_originality(neutralized_expr, self._originality_checker)
                results.append(MutationResult(
                    mutated_expression=neutralized_expr,
                    original_id=original_id,
                    mutation_type="structure_change",
                    mutation_description="added group_neutralize(..., industry)",
                    originality_score=orig_score3,
                ))
            except (ValueError, SyntaxError, TypeError):
                pass

        if ast.op == "rank" and len(ast.children) >= 1:
            try:
                inner = copy.deepcopy(ast.children[0])
                unwrapped_expr = _ast_to_string(inner)
                if unwrapped_expr != expr:
                    orig_score4 = _compute_originality(unwrapped_expr, self._originality_checker)
                    results.append(MutationResult(
                        mutated_expression=unwrapped_expr,
                        original_id=original_id,
                        mutation_type="structure_change",
                        mutation_description="removed rank() wrapper",
                        originality_score=orig_score4,
                    ))
            except (OSError, ValueError, RuntimeError) as exc:
                logger.warning("GradientMutation._structure_change: unwrap rank failed — %s", exc)

        return results


class CrossoverMutationEngine:

    TOURNAMENT_SIZE: int = 3
    ELITE_COUNT: int = 2

    DIRECTION_WEIGHTS: dict[str, dict[str, float]] = {
        "momentum": {"sharpe": 0.60, "turnover": 0.20, "complexity": 0.20},
        "mean_reversion": {"sharpe": 0.55, "turnover": 0.30, "complexity": 0.15},
        "volatility": {"sharpe": 0.50, "turnover": 0.25, "complexity": 0.25},
        "statistical": {"sharpe": 0.55, "turnover": 0.20, "complexity": 0.25},
        "volume": {"sharpe": 0.50, "turnover": 0.30, "complexity": 0.20},
        "interaction": {"sharpe": 0.45, "turnover": 0.25, "complexity": 0.30},
    }
    WEIGHT_MIN: float = 0.10
    WEIGHT_MAX: float = 0.70
    ADAPT_STEP: float = 0.05
    ADAPT_BATCH_SIZE: int = 20
    PARETO_TIE_THRESHOLD: float = 0.10

    def __init__(self, originality_checker: OriginalityChecker | None = None,
                 llm_generate_fn: Callable[..., Awaitable[str]] | None = None):
        self._crossover = SemanticCrossover(originality_checker, llm_generate_fn=llm_generate_fn)
        self._mutation = GradientMutation(originality_checker)
        self._trajectory_crossover = TrajectoryCrossover(
            llm_generate_fn=llm_generate_fn,
            originality_checker=originality_checker,
        )
        self._trajectory_mutation_v2 = TrajectoryMutationV2(llm_generate_fn=llm_generate_fn)
        self._trajectory_pool: list[tuple[AlphaTrajectory, dict[str, Any]]] = []
        self._direction_weights: dict[str, dict[str, float]] = {
            k: dict(v) for k, v in self.DIRECTION_WEIGHTS.items()
        }
        self._direction_rejections: dict[str, list[str]] = {
            d: [] for d in _DIRECTION_KEYS
        }
        self._direction_alpha_count: dict[str, int] = dict.fromkeys(_DIRECTION_KEYS, 0)

    def _tournament_select(self, population: list[dict], k: int | None = None) -> dict:
        k = k or self.TOURNAMENT_SIZE
        candidates = random.sample(population, min(k, len(population)))
        return max(candidates, key=lambda c: c.get("fitness", c.get("sharpe", 0.0)))

    def compute_fitness(self, sharpe: float, turnover: float | None,
                        complexity: int, direction: str) -> float:
        weights = self._direction_weights.get(
            direction, self._direction_weights["momentum"]
        )
        w_sharpe = weights["sharpe"]
        w_turnover = weights["turnover"]
        w_complexity = weights["complexity"]

        turnover_score = max(0.0, 1.0 - (turnover or 0.5))
        complexity_penalty = max(0.0, 1.0 - complexity / 12.0)

        return w_sharpe * max(0.0, sharpe) + w_turnover * turnover_score + w_complexity * complexity_penalty

    def record_alpha_outcome(self, direction: str, sharpe: float, turnover: float | None,
                             complexity: int, accepted: bool,
                             reject_reason: str | None = None) -> None:
        if direction not in self._direction_alpha_count:
            return

        self._direction_alpha_count[direction] += 1
        reason = reject_reason or ("accepted" if accepted else "unknown")
        self._direction_rejections[direction].append(reason)

        if self._direction_alpha_count[direction] % self.ADAPT_BATCH_SIZE == 0:
            self._adapt_weights(direction)

    def _adapt_weights(self, direction: str) -> None:
        recent = self._direction_rejections[direction][-self.ADAPT_BATCH_SIZE:]
        weights = self._direction_weights[direction]

        consecutive_turnover = 0
        consecutive_complexity = 0
        consecutive_sharpe = 0

        turnover_keywords = ("turnover", "换手率", "high_turnover")
        complexity_keywords = ("overfit", "complexity", "过拟合", "复杂度")
        sharpe_keywords = ("sharpe", "sharpe不足", "low_sharpe")

        for r in recent:
            r_lower = r.lower()
            if any(kw in r_lower for kw in turnover_keywords):
                consecutive_turnover += 1
                consecutive_complexity = 0
                consecutive_sharpe = 0
            elif any(kw in r_lower for kw in complexity_keywords):
                consecutive_complexity += 1
                consecutive_turnover = 0
                consecutive_sharpe = 0
            elif any(kw in r_lower for kw in sharpe_keywords):
                consecutive_sharpe += 1
                consecutive_turnover = 0
                consecutive_complexity = 0
            else:
                consecutive_turnover = 0
                consecutive_complexity = 0
                consecutive_sharpe = 0

        changed = False
        if consecutive_turnover >= 3:
            weights["turnover"] = min(self.WEIGHT_MAX, weights["turnover"] + self.ADAPT_STEP)
            weights["sharpe"] = max(self.WEIGHT_MIN, weights["sharpe"] - self.ADAPT_STEP)
            changed = True
            logger.info("engine: dir=%s turnover penalty boosted to turnover=%.2f sharpe=%.2f",
                        direction, weights["turnover"], weights["sharpe"])
        elif consecutive_complexity >= 3:
            weights["complexity"] = min(self.WEIGHT_MAX, weights["complexity"] + self.ADAPT_STEP)
            weights["sharpe"] = max(self.WEIGHT_MIN, weights["sharpe"] - self.ADAPT_STEP)
            changed = True
            logger.info("engine: dir=%s complexity penalty boosted to complexity=%.2f sharpe=%.2f",
                        direction, weights["complexity"], weights["sharpe"])
        elif consecutive_sharpe >= 3:
            weights["sharpe"] = min(self.WEIGHT_MAX, weights["sharpe"] + self.ADAPT_STEP)
            weights["turnover"] = max(self.WEIGHT_MIN, weights["turnover"] - self.ADAPT_STEP / 2)
            weights["complexity"] = max(self.WEIGHT_MIN, weights["complexity"] - self.ADAPT_STEP / 2)
            changed = True
            logger.info("engine: dir=%s sharpe focus boosted to sharpe=%.2f turnover=%.2f complexity=%.2f",
                        direction, weights["sharpe"], weights["turnover"], weights["complexity"])

        if changed:
            total_sum = weights["sharpe"] + weights["turnover"] + weights["complexity"]
            if total_sum > 0:
                weights["sharpe"] = round(weights["sharpe"] / total_sum, 4)
                weights["turnover"] = round(weights["turnover"] / total_sum, 4)
                weights["complexity"] = round(weights["complexity"] / total_sum, 4)

    def _pareto_dominates(self, a: dict, b: dict) -> bool:
        a_sharpe = a.get("sharpe", a.get("fitness", 0.0))
        b_sharpe = b.get("sharpe", b.get("fitness", 0.0))
        a_turnover = a.get("turnover")
        b_turnover = b.get("turnover")
        a_complexity = a.get("complexity", len(a.get("expression", "")))
        b_complexity = b.get("complexity", len(b.get("expression", "")))

        if a_turnover is not None and b_turnover is not None:
            turnover_worse = a_turnover > b_turnover
        else:
            turnover_worse = False

        a_better_or_equal = (
            a_sharpe >= b_sharpe
            and a_complexity <= b_complexity
            and (not turnover_worse or a_turnover is None)
        )
        a_strictly_better = (
            a_sharpe > b_sharpe
            or a_complexity < b_complexity
            or (b_turnover is not None and a_turnover is not None
                and a_turnover < b_turnover)
        )

        return a_better_or_equal and a_strictly_better

    def _pareto_tie_break_sort(self, candidates: list[dict]) -> list[dict]:
        if len(candidates) <= 1:
            return candidates

        sorted_list = sorted(
            candidates,
            key=lambda v: v.get("fitness", v.get("sharpe", 0.0)),
            reverse=True,
        )

        result = [sorted_list[0]]
        for i in range(1, len(sorted_list)):
            prev = result[-1]
            curr = sorted_list[i]

            prev_fitness = prev.get("fitness", 0.0)
            curr_fitness = curr.get("fitness", 0.0)
            max_fit = max(abs(prev_fitness), abs(curr_fitness), 1e-8)

            if abs(prev_fitness - curr_fitness) / max_fit < self.PARETO_TIE_THRESHOLD:
                if self._pareto_dominates(curr, prev):
                    result[-1], curr = curr, prev

            result.append(curr)

        return result

    def generate_variants(self,
                          successful_alphas: list[dict],
                          brain_feedback: dict | None = None,
                          max_variants: int = 5,
                          direction: str = "") -> list[dict]:
        variants: list[dict] = []

        _monitor.record("STEP", "crossover_mutation_engine", "generate_variants", f"alphas={len(successful_alphas)}")

        sorted_alphas = sorted(
            successful_alphas,
            key=lambda a: a.get("fitness", a.get("sharpe", 0.0)),
            reverse=True,
        )

        elites = sorted_alphas[:self.ELITE_COUNT]
        for elite in elites:
            variants.append({
                "expression": elite["expression"],
                "source": "elite",
                "parent": elite.get("id", ""),
                "originality_score": elite.get("originality_score", 0.8),
                "fitness": elite.get("fitness", elite.get("sharpe", 0.0)),
            })
            if len(variants) >= max_variants:
                seen: set[str] = {v["expression"] for v in variants}
                _monitor.record("PASS", "crossover_mutation_engine", "generate_variants", f"elites_only={len(variants)}")
                return variants

        if len(successful_alphas) >= 2:
            crossover_rounds = min(3, max_variants - len(variants))
            for _ in range(crossover_rounds):
                if len(variants) >= max_variants:
                    break
                parent1 = self._tournament_select(successful_alphas)
                remaining = [a for a in successful_alphas if a is not parent1]
                if not remaining:
                    continue
                parent2 = self._tournament_select(remaining, min(self.TOURNAMENT_SIZE, len(remaining)))

                results = self._crossover.crossover(
                    parent1["expression"],
                    parent2["expression"],
                    parent1.get("id", ""),
                    parent2.get("id", ""),
                )
                for r in results:
                    if r.originality_score >= 0.3:
                        child_dir = direction or parent1.get("direction", "")
                        variants.append({
                            "expression": r.child_expression,
                            "source": "crossover",
                            "parents": [r.parent1_id, r.parent2_id],
                            "originality_score": r.originality_score,
                            "parent1_fitness": parent1.get("fitness", parent1.get("sharpe", 0.0)),
                            "parent2_fitness": parent2.get("fitness", parent2.get("sharpe", 0.0)),
                            "direction": child_dir,
                        })
                        if len(variants) >= max_variants:
                            break

        mutation_count = min(3, len(sorted_alphas))
        for i in range(mutation_count):
            if len(variants) >= max_variants:
                break
            alpha = self._tournament_select(sorted_alphas)
            mutation_results = self._mutation.mutate(
                alpha["expression"],
                brain_feedback=brain_feedback,
                original_id=alpha.get("id", ""),
            )
            for r in mutation_results:
                if r.originality_score >= 0.3:
                    child_dir = direction or alpha.get("direction", "")
                    variants.append({
                        "expression": r.mutated_expression,
                        "source": "mutation",
                        "type": r.mutation_type,
                        "parent": r.original_id,
                        "originality_score": r.originality_score,
                        "parent_fitness": alpha.get("fitness", alpha.get("sharpe", 0.0)),
                        "direction": child_dir,
                    })
                    if len(variants) >= max_variants:
                        break

        seen: set[str] = set()
        unique: list[dict] = []
        for v in variants:
            if v["expression"] not in seen:
                seen.add(v["expression"])
                unique.append(v)

        unique = self._pareto_tie_break_sort(unique)

        _monitor.record("PASS", "crossover_mutation_engine", "generate_variants", f"variants={len(unique[:max_variants])}")

        return unique[:max_variants]

    def record_trajectory(self, trajectory: AlphaTrajectory, meta: dict[str, Any] | None = None) -> None:
        self._trajectory_pool.append((trajectory, meta or {}))
        if len(self._trajectory_pool) > 50:
            self._trajectory_pool = self._trajectory_pool[-50:]

    async def crossover_trajectories(self,
                                     id_a: str = "",
                                     id_b: str = "",
                                     ) -> list[TrajectoryCrossoverResult]:
        eligible = [(t, m) for t, m in self._trajectory_pool
                    if t.final_status != "FAILED" and t.expression_versions and t.hypothesis_direction]

        if len(eligible) < 2:
            logger.info("TrajectoryCrossover: not enough eligible trajectories (%d)", len(eligible))
            return []

        if id_a and id_b:
            candidates_a = [(t, m) for t, m in eligible if m.get("id") == id_a]
            candidates_b = [(t, m) for t, m in eligible if m.get("id") == id_b]
            if candidates_a and candidates_b:
                traj_a, meta_a = candidates_a[0]
                traj_b, meta_b = candidates_b[0]
            else:
                traj_a, meta_a = eligible[0]
                traj_b, meta_b = eligible[1]
        else:
            import random as _random
            indices = _random.sample(range(len(eligible)), min(2, len(eligible)))
            traj_a, meta_a = eligible[indices[0]]
            traj_b, meta_b = eligible[indices[1]]

        _monitor.record("STEP", "crossover_mutation_engine", "trajectory_crossover",
                        f"pool_size={len(self._trajectory_pool)} parents={meta_a.get('id')},{meta_b.get('id')}")

        return await self._trajectory_crossover.crossover_trajectories(
            traj_a, traj_b,
            id_a=meta_a.get("id", ""),
            id_b=meta_b.get("id", ""),
        )

    async def mutate_trajectory(self,
                                trajectory_id: str = "",
                                ) -> list[TrajectoryMutationResult]:
        eligible = [(t, m) for t, m in self._trajectory_pool
                    if t.final_status != "FAILED"]

        if not eligible:
            logger.info("TrajectoryMutationV2: no eligible trajectories to mutate")
            return []

        if trajectory_id:
            candidates = [(t, m) for t, m in eligible if m.get("id") == trajectory_id]
            if candidates:
                traj, meta = candidates[0]
            else:
                traj, meta = eligible[0]
        else:
            import random as _random
            traj, meta = _random.choice(eligible)

        _monitor.record("STEP", "crossover_mutation_engine", "trajectory_mutation_v2",
                        f"pool_size={len(self._trajectory_pool)} target={meta.get('id')}")

        return await self._trajectory_mutation_v2.mutate_trajectory(
            traj, original_id=meta.get("id", ""),
        )

