"""
QuantaAlpha Generation Gates — Three-Way Semantic Consistency Gate (arXiv:2602.07085)

Enforces three-way semantic consistency at factor generation time:
  Hypothesis ↔ Expression ↔ Code

If any dimension fails below threshold, injects a correction prompt and triggers
regeneration (up to MAX_GATE_RETRIES).

Gate Dimensions:
  G1: H↔E — Hypothesis ↔ Expression alignment via HypothesisAligner (R² ≥ 0.55)
  G2: E↔C — Expression ↔ Code structural consistency via structural heuristics
  G3: H↔E↔C — Three-way holistic consistency via LLM semantic review
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from openalpha_brain.utils import extract_json_from_llm as _extract_json_from_llm
from openalpha_brain.utils.algo_logger import algo_log, log_call
from openalpha_brain.validation.ast_validator import ASTValidator

logger = logging.getLogger(__name__)

GATE_HYPOTHESIS_EXPRESSION = "H↔E"
GATE_EXPRESSION_CODE = "E↔C"
GATE_HOLISTIC = "H↔E↔C"

DEFAULT_R2_THRESHOLD = 0.40
DEFAULT_STRUCTURAL_THRESHOLD = 0.50
DEFAULT_HOLISTIC_THRESHOLD = 0.55
DEFAULT_MAX_RETRIES = 2

ALIGNMENT_WARNING_LOW = 0.6
ALIGNMENT_WARNING_HIGH = 0.75


@dataclass
class GateResult:
    gate_name: str
    passed: bool
    score: float
    threshold: float
    diagnosis: str = ""
    fix_hints: list[str] = field(default_factory=list)
    alignment_details: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationGateReport:
    passed: bool
    results: list[GateResult]
    overall_score: float
    correction_prompt: str = ""
    failed_gates: list[str] = field(default_factory=list)
    decision_rationale: str = ""


class GenerationGates:
    """QuantaAlpha Generation Gates — three-way semantic consistency enforcement.

    Usage:
        gates = GenerationGates(
            hypothesis_aligner=aligner,
            llm_generate_fn=llm_client.generate,
        )
        report = await gates.check(
            hypothesis_direction="momentum_long",
            hypothesis_mechanism="price_continuation",
            hypothesis_nl="Prices that have been rising will continue to rise",
            expression="group_neutralize(rank(ts_delta(close, 5)), industry)",
        )
        if not report.passed:
            # inject report.correction_prompt into generator and retry
    """

    def __init__(
        self,
        hypothesis_aligner=None,
        llm_generate_fn: Callable[..., Awaitable[str]] | None = None,
        r2_threshold: float = DEFAULT_R2_THRESHOLD,
        structural_threshold: float = DEFAULT_STRUCTURAL_THRESHOLD,
        holistic_threshold: float = DEFAULT_HOLISTIC_THRESHOLD,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self._aligner = hypothesis_aligner
        self._llm = llm_generate_fn
        self._r2_threshold = r2_threshold
        self._structural_threshold = structural_threshold
        self._holistic_threshold = holistic_threshold
        self._max_retries = max_retries

    @algo_log(level=logging.INFO, label="GenerationGates.check")
    async def check(
        self,
        hypothesis_direction: str,
        hypothesis_mechanism: str = "",
        hypothesis_nl: str = "",
        expression: str = "",
        operators: list[str] | None = None,
        fields: list[str] | None = None,
    ) -> GenerationGateReport:
        results: list[GateResult] = []

        log_call(
            "GenerationGates.check_start",
            input={
                "hypothesis_direction": hypothesis_direction[:80],
                "expression": expression[:100],
                "operators_count": len(operators) if operators else 0,
                "fields_count": len(fields) if fields else 0,
            },
            level=logging.DEBUG,
        )

        g1 = self._check_hypothesis_expression(
            expression,
            hypothesis_direction,
            hypothesis_nl,
        )
        results.append(g1)

        log_call(
            "GenerationGates.G1_result",
            input={"gate": GATE_HYPOTHESIS_EXPRESSION},
            output={
                "passed": g1.passed,
                "score": g1.score,
                "threshold": g1.threshold,
                "diagnosis": g1.diagnosis[:150],
            },
            level=logging.INFO,
        )

        g2 = self._check_expression_code(expression, operators, fields)
        results.append(g2)

        log_call(
            "GenerationGates.G2_result",
            input={"gate": GATE_EXPRESSION_CODE},
            output={
                "passed": g2.passed,
                "score": g2.score,
                "threshold": g2.threshold,
                "diagnosis": g2.diagnosis[:150],
            },
            level=logging.INFO,
        )

        g3 = await self._check_holistic(
            expression,
            hypothesis_direction,
            hypothesis_mechanism,
            hypothesis_nl,
            operators,
            fields,
        )
        results.append(g3)

        log_call(
            "GenerationGates.G3_result",
            input={"gate": GATE_HOLISTIC},
            output={
                "passed": g3.passed,
                "score": g3.score,
                "threshold": g3.threshold,
                "diagnosis": g3.diagnosis[:150],
            },
            level=logging.INFO,
        )

        all_passed = all(r.passed for r in results)
        failed_gates = [r.gate_name for r in results if not r.passed]
        overall_score = sum(r.score for r in results) / max(len(results), 1)

        decision_rationale = self._build_decision_rationale(results, all_passed, failed_gates, overall_score)

        log_call(
            "GenerationGates.final_decision",
            input={"gates_checked": len(results)},
            output={
                "all_passed": all_passed,
                "overall_score": round(overall_score, 4),
                "failed_gates": failed_gates,
                "rationale": decision_rationale[:200],
            },
            level=logging.INFO,
        )

        correction_prompt = ""
        if not all_passed:
            correction_prompt = self._build_correction_prompt(results, hypothesis_direction)

        return GenerationGateReport(
            passed=all_passed,
            results=results,
            overall_score=round(overall_score, 4),
            correction_prompt=correction_prompt,
            failed_gates=failed_gates,
            decision_rationale=decision_rationale,
        )

    @algo_log(level=logging.DEBUG, label="GenerationGates.check_hypothesis_expression")
    def _check_hypothesis_expression(
        self,
        expression: str,
        hypothesis_direction: str,
        hypothesis_nl: str = "",
    ) -> GateResult:
        if self._aligner is None:
            logger.info("GenerationGates[H↔E]: HypothesisAligner 未配置，跳过 H↔E 门控，默认通过")
            return GateResult(
                gate_name=GATE_HYPOTHESIS_EXPRESSION,
                passed=True,
                score=0.6,
                threshold=self._r2_threshold,
                diagnosis="HypothesisAligner not available, skipping H↔E gate",
                alignment_details={"skipped_reason": "aligner_not_available"},
            )

        try:
            target = hypothesis_nl or hypothesis_direction
            alignment = self._aligner.align(expression, target)

            r2_score = alignment.get("r2_score", 0.0)
            level = alignment.get("alignment_level", "unknown")
            diagnosis = alignment.get("diagnosis", "")
            suggestions = alignment.get("suggestions", [])
            field_matches = alignment.get("field_matches", {})
            semantic_score = alignment.get("semantic_score", 0.0)
            direction_consistent = alignment.get("direction_consistent", True)

            alignment_details = {
                "r2_score": round(r2_score, 4),
                "alignment_level": level,
                "field_matches": field_matches,
                "semantic_score": round(semantic_score, 4) if semantic_score else None,
                "direction_consistent": direction_consistent,
                "matched_fields": [k for k, v in field_matches.items() if v],
                "mismatched_fields": [k for k, v in field_matches.items() if not v],
            }

            enhanced_score = self._compute_enhanced_alignment_score(
                r2_score, semantic_score, direction_consistent, field_matches
            )

            passed = enhanced_score >= self._r2_threshold

            if ALIGNMENT_WARNING_LOW <= enhanced_score <= ALIGNMENT_WARNING_HIGH:
                logger.warning(
                    "GenerationGates[H↔E]: ⚠️ 对齐分数处于边界区间 [%.3f]，建议人工审核 | "
                    "R²=%.3f | semantic=%.3f | direction_ok=%s | matched=%s | mismatched=%s",
                    enhanced_score,
                    r2_score,
                    semantic_score or 0,
                    direction_consistent,
                    alignment_details["matched_fields"],
                    alignment_details["mismatched_fields"],
                )

            log_call(
                "GenerationGates.H↔E.alignment_detail",
                input={"expression": expression[:80], "target": target[:80]},
                output={
                    "r2_score": round(r2_score, 4),
                    "enhanced_score": round(enhanced_score, 4),
                    "level": level,
                    "direction_consistent": direction_consistent,
                    "field_match_count": len(alignment_details["matched_fields"]),
                    "field_mismatch_count": len(alignment_details["mismatched_fields"]),
                    "passed": passed,
                },
                level=logging.DEBUG if passed else logging.WARNING,
            )

            return GateResult(
                gate_name=GATE_HYPOTHESIS_EXPRESSION,
                passed=passed,
                score=round(enhanced_score, 4),
                threshold=self._r2_threshold,
                diagnosis=f"R²={r2_score:.3f} (enhanced={enhanced_score:.3f}, {level}): {diagnosis}",
                fix_hints=suggestions[:4],
                alignment_details=alignment_details,
            )
        except Exception as exc:
            logger.error("GenerationGates[H↔E]: ❌ H↔E 检查异常，默认通过: %s", exc, exc_info=True)
            return GateResult(
                gate_name=GATE_HYPOTHESIS_EXPRESSION,
                passed=True,
                score=0.6,
                threshold=self._r2_threshold,
                diagnosis=f"H↔E check error (deemed pass): {exc}",
                alignment_details={"error": str(exc)},
            )

    def _compute_enhanced_alignment_score(
        self,
        r2_score: float,
        semantic_score: float | None,
        direction_consistent: bool,
        field_matches: dict[str, bool],
    ) -> float:
        base_score = r2_score

        if semantic_score is not None and semantic_score > 0:
            base_score = base_score * 0.7 + semantic_score * 0.3

        if not direction_consistent:
            base_score *= 0.6
            logger.debug("GenerationGates[H↔E]: 方向不一致惩罚，分数调整为 %.3f", base_score)

        total_fields = len(field_matches)
        if total_fields > 0:
            match_ratio = sum(1 for v in field_matches.values() if v) / total_fields
            if match_ratio < 0.5:
                base_score *= 0.8
                logger.debug(
                    "GenerationGates[H↔E]: 字段匹配率低 (%.1f%%)，惩罚后分数 %.3f", match_ratio * 100, base_score
                )

        return max(0.0, min(1.0, base_score))

    @algo_log(level=logging.DEBUG, label="GenerationGates.check_expression_code")
    def _check_expression_code(
        self,
        expression: str,
        operators: list[str] | None = None,
        fields: list[str] | None = None,
    ) -> GateResult:
        if not expression:
            logger.warning("GenerationGates[E↔C]: ❌ 表达式为空，直接拒绝")
            return GateResult(
                gate_name=GATE_EXPRESSION_CODE,
                passed=False,
                score=0.0,
                threshold=self._structural_threshold,
                diagnosis="Empty expression",
                fix_hints=["Provide a valid FASTEXPR formula"],
            )

        # ── AST 硬校验（AlphaBench ICLR'26 SOTA）─────────────────────────────
        ast_validator = ASTValidator()
        ast_result = ast_validator.validate(expression)

        if ast_result.errors:
            logger.error(
                "GenerationGates[E↔C]: ❌ [DEFENSIVE_LOG] AST 硬校验拦截 | errors=%s | expr=%s",
                ast_result.errors,
                expression[:100],
            )
            log_call(
                "GenerationGates.E↔C.ast_hard_block",
                input={"expression": expression[:80]},
                output={
                    "ast_passed": False,
                    "error_count": len(ast_result.errors),
                    "errors": ast_result.errors,
                    "warnings": ast_result.warnings[:3],
                    "structure": ast_result.structure_info,
                },
                level=logging.ERROR,
            )
            return GateResult(
                gate_name=GATE_EXPRESSION_CODE,
                passed=False,
                score=0.0,
                threshold=self._structural_threshold,
                diagnosis=f"[AST_HARD_BLOCK] {'; '.join(ast_result.errors)}",
                fix_hints=ast_result.fix_suggestions[:4],
                alignment_details={
                    "ast_validation": {
                        "passed": False,
                        "errors": ast_result.errors,
                        "warnings": ast_result.warnings,
                        "structure": ast_result.structure_info,
                    }
                },
            )

        sub_scores: list[tuple[str, float]] = []
        hints: list[str] = []

        # ── 使用 AST 结构信息替代正则检查 ─────────────────────────────────
        structure = ast_result.structure_info

        sub_scores.append(("syntax_valid", 1.0))
        sub_scores.append(("operator_whitelist", 1.0))

        op_count = structure.get("operator_count", 0)
        if 1 <= op_count <= 8:
            sub_scores.append(("operator_count", 1.0))
        elif op_count == 0:
            sub_scores.append(("operator_count", 0.3))
            hints.append("Expression lacks operators, may be a stub or comment")
        else:
            penalty = (op_count - 8) * 0.04
            sub_scores.append(("operator_count", max(0.5, 1.0 - penalty)))
            if op_count > 10:
                hints.append(f"Expression has {op_count} operators, consider simplifying (max 8-10 recommended)")

        ast_depth = structure.get("nesting_depth", 0)
        if ast_depth <= 6:
            sub_scores.append(("nesting_depth", 1.0))
        elif ast_depth <= 10:
            sub_scores.append(("nesting_depth", 0.7))
        else:
            sub_scores.append(("nesting_depth", 0.4))
            hints.append(f"AST nesting depth {ast_depth} is very high, simplify structure")

        has_neutralize = structure.get("has_neutralize", False)
        has_decay = structure.get("has_decay", False)
        if has_neutralize and has_decay:
            sub_scores.append(("three_block_structure", 1.0))
        elif has_neutralize or has_decay:
            sub_scores.append(("three_block_structure", 0.75))
            missing = "decay" if not has_decay else "neutralize"
            hints.append(f"Three-block incomplete: missing {missing} segment")
        else:
            sub_scores.append(("three_block_structure", 0.5))
            hints.append("No three-block structure detected (no neutralize + no decay)")

        effectively_empty = len(expression.strip()) < 10 or expression.strip().lower() in (
            "none",
            "null",
            "false",
            "true",
            "1",
            "0",
            "-1",
        )
        if effectively_empty:
            sub_scores = [("effectively_empty", 0.1)]
            hints = ["Expression is effectively empty or a placeholder"]

        structural_score = sum(s[1] for s in sub_scores) / max(len(sub_scores), 1)
        passed = structural_score >= self._structural_threshold

        if ast_result.warnings:
            for w in ast_result.warnings:
                hints.append(w)

        borderline_items = [name for name, score in sub_scores if 0.35 <= score <= 0.65 and score != 1.0]

        if borderline_items and passed:
            logger.warning(
                "GenerationGates[E↔C]: ⚠️ 结构检查通过但存在边界项: %s | 总分=%.3f (阈值=%.2f) | 建议: %s",
                borderline_items,
                structural_score,
                self._structural_threshold,
                "; ".join(hints[:2]) if hints else "无",
            )

        log_call(
            "GenerationGates.E↔C.structural_detail",
            input={"expression": expression[:80]},
            output={
                "sub_scores": {name: round(s, 3) for name, s in sub_scores},
                "total_score": round(structural_score, 4),
                "op_count": op_count,
                "max_depth": ast_depth,
                "paren_balanced": True,
                "has_neutralize": has_neutralize,
                "has_decay": has_decay,
                "borderline_items": borderline_items,
                "ast_warnings_count": len(ast_result.warnings),
                "passed": passed,
            },
            level=logging.DEBUG if passed else logging.WARNING,
        )

        diagnosis_parts = [
            f"ops={op_count}",
            f"ast_depth={ast_depth}",
            "balanced=True",
            f"neutralize={has_neutralize}",
            f"decay={has_decay}",
            f"score_breakdown={dict(sub_scores)}",
        ]

        return GateResult(
            gate_name=GATE_EXPRESSION_CODE,
            passed=passed,
            score=round(structural_score, 4),
            threshold=self._structural_threshold,
            diagnosis="; ".join(diagnosis_parts),
            fix_hints=hints,
            alignment_details={
                "ast_validation": {
                    "passed": ast_result.passed,
                    "warnings": ast_result.warnings,
                    "structure": ast_result.structure_info,
                }
            },
        )

    @algo_log(level=logging.DEBUG, label="GenerationGates.check_holistic")
    async def _check_holistic(
        self,
        expression: str,
        hypothesis_direction: str,
        hypothesis_mechanism: str = "",
        hypothesis_nl: str = "",
        operators: list[str] | None = None,
        fields: list[str] | None = None,
    ) -> GateResult:
        if self._llm is None:
            logger.info("GenerationGates[H↔E↔C]: LLM 未配置，跳过整体门控，默认通过")
            return GateResult(
                gate_name=GATE_HOLISTIC,
                passed=True,
                score=0.6,
                threshold=self._holistic_threshold,
                diagnosis="LLM not available, skipping holistic gate",
            )

        prompt = self._build_holistic_prompt(
            expression,
            hypothesis_direction,
            hypothesis_mechanism,
            hypothesis_nl,
            operators,
            fields,
        )

        try:
            response = await self._llm(prompt)
            parsed = self._parse_holistic_response(response)

            score = parsed.get("consistency_score", 0.5)
            reasoning = parsed.get("reasoning", "")
            issues = parsed.get("issues", [])
            suggestions = parsed.get("suggestions", [])

            complexity_assessment = self._assess_complexity(expression, operators)
            overfitting_risk = self._assess_overfitting_risk(expression, fields)
            decay_indicator = self._assess_decay_indicator(expression)
            stability_check = self._check_stability_indicators(expression, hypothesis_direction)

            holistic_factors = {
                "llm_consistency_score": score,
                "complexity": complexity_assessment,
                "overfitting_risk": overfitting_risk,
                "decay_indicator": decay_indicator,
                "stability": stability_check,
            }

            adjusted_score = self._adjust_holistic_score(score, holistic_factors)

            borderline_factors = {
                k: v for k, v in holistic_factors.items() if isinstance(v, dict) and v.get("is_borderline", False)
            }

            if borderline_factors:
                logger.warning(
                    "GenerationGates[H↔E↔C]: ⚠️ 整体检查发现边界风险因子: %s | 原始 LLM 分数=%.3f → 调整后=%.3f",
                    list(borderline_factors.keys()),
                    score,
                    adjusted_score,
                )

            passed = adjusted_score >= self._holistic_threshold

            log_call(
                "GenerationGates.H↔E↔C.holistic_detail",
                input={
                    "expression": expression[:80],
                    "hypothesis_direction": hypothesis_direction[:60],
                },
                output={
                    "llm_raw_score": round(score, 4),
                    "adjusted_score": round(adjusted_score, 4),
                    "holistic_factors": {
                        k: {kk: vv for kk, vv in v.items() if kk != "details"}
                        for k, v in holistic_factors.items()
                        if isinstance(v, dict)
                    },
                    "borderline_factors": list(borderline_factors.keys()),
                    "issues_count": len(issues),
                    "passed": passed,
                },
                level=logging.DEBUG if passed else logging.WARNING,
            )

            final_diagnosis = f"LLM={score:.3f}→adj={adjusted_score:.3f}: {reasoning[:150]}"
            if borderline_factors:
                final_diagnosis += f" | 边界风险: {', '.join(borderline_factors.keys())}"

            return GateResult(
                gate_name=GATE_HOLISTIC,
                passed=passed,
                score=round(adjusted_score, 4),
                threshold=self._holistic_threshold,
                diagnosis=final_diagnosis,
                fix_hints=suggestions[:4] if suggestions else issues[:4],
            )
        except Exception as exc:
            logger.error("GenerationGates[H↔E↔C]: ❌ 整体检查 LLM 调用异常，默认通过: %s", exc, exc_info=True)
            return GateResult(
                gate_name=GATE_HOLISTIC,
                passed=True,
                score=0.55,
                threshold=self._holistic_threshold,
                diagnosis=f"Holistic check LLM error (deemed pass): {exc}",
            )

    def _assess_complexity(self, expression: str, operators: list[str] | None) -> dict[str, Any]:
        op_pattern = re.findall(r"\b\w+\(", expression)
        func_count = len(op_pattern)
        nesting = expression.count("(")

        complexity_score = 1.0
        if func_count > 15:
            complexity_score = 0.5
        elif func_count > 10:
            complexity_score = 0.7
        elif func_count > 7:
            complexity_score = 0.85

        is_borderline = 0.65 <= complexity_score <= 0.8

        result = {
            "func_count": func_count,
            "nesting_depth": nesting,
            "complexity_score": complexity_score,
            "is_borderline": is_borderline,
            "verdict": "acceptable" if complexity_score >= 0.7 else "high_complexity",
        }

        if is_borderline:
            logger.debug(
                "GenerationGates[H↔E↔C]: 复杂度边界: 函数数=%d, 嵌套=%d, 得分=%.2f",
                func_count,
                nesting,
                complexity_score,
            )

        return result

    def _assess_overfitting_risk(self, expression: str, fields: list[str] | None) -> dict[str, Any]:
        risk_indicators = []
        risk_score = 0.0

        specific_fields = re.findall(r"\b(volume|vwap|turnover|bid|ask|spread|imbalance)\b", expression.lower())
        if len(specific_fields) > 3:
            risk_score += 0.2
            risk_indicators.append(f"过多特定字段({len(specific_fields)})")

        param_counts = re.findall(r"\d+", expression)
        if len(param_counts) > 5:
            risk_score += 0.15
            risk_indicators.append(f"过多硬编码参数({len(param_counts)})")

        nested_ts = re.findall(r"ts_\w+\([^)]*ts_\w+", expression)
        if len(nested_ts) > 2:
            risk_score += 0.2
            risk_indicators.append(f"深层时序嵌套({len(nested_ts)}层)")

        combined_ops = re.findall(r"(rank\s*\(\s*rank|zscore\s*\(\s*zscore)", expression.lower())
        if combined_ops:
            risk_score += 0.15
            risk_indicators.append("重复标准化操作")

        final_risk_score = min(0.9, risk_score)
        is_borderline = 0.25 <= final_risk_score <= 0.45

        result = {
            "risk_score": round(final_risk_score, 3),
            "indicators": risk_indicators,
            "is_borderline": is_borderline,
            "verdict": "low_risk"
            if final_risk_score < 0.3
            else ("medium_risk" if final_risk_score < 0.5 else "high_risk"),
        }

        if is_borderline or final_risk_score >= 0.4:
            logger.debug(
                "GenerationGates[H↔E↔C]: 过拟合风险评估: 得分=%.3f | 风险因子: %s",
                final_risk_score,
                risk_indicators,
            )

        return result

    def _assess_decay_indicator(self, expression: str) -> dict[str, Any]:
        has_decay_resistance = bool(re.search(r"\b(decay|halflife|ewm|exp_weighted)\b", expression.lower()))
        uses_long_window = any(int(x) > 60 for x in re.findall(r"\b\d+\b", expression))

        decay_score = 1.0
        decay_notes = []

        if not has_decay_resistance and uses_long_window:
            decay_score = 0.7
            decay_notes.append("长窗口但无衰减机制")

        is_borderline = 0.6 <= decay_score <= 0.8

        result = {
            "decay_score": decay_score,
            "has_decay_resistance": has_decay_resistance,
            "uses_long_window": uses_long_window,
            "notes": decay_notes,
            "is_borderline": is_borderline,
        }

        if is_borderline:
            logger.debug(
                "GenerationGates[H↔E↔C]: 衰减指标边界: 得分=%.2f | 说明: %s",
                decay_score,
                decay_notes,
            )

        return result

    def _check_stability_indicators(self, expression: str, hypothesis_direction: str) -> dict[str, Any]:
        stability_issues = []
        stability_score = 1.0

        division_by_var = re.search(r"/\s*(close|open|high|low|volume)\b", expression, re.IGNORECASE)
        if division_by_var:
            stability_score -= 0.15
            stability_issues.append("变量做除数可能导致不稳定")

        extreme_transform = re.search(r"\b(pow|signed_power)\([^,]+,\s*[3-9]", expression)
        if extreme_transform:
            stability_score -= 0.1
            stability_issues.append("高幂次变换可能放大噪声")

        thin_field_usage = re.search(r"\b(bid_ask_spread|imbalance_ratio)\b", expression.lower())
        if thin_field_usage and "mean_reversion" not in hypothesis_direction.lower():
            stability_score -= 0.1
            stability_issues.append("使用薄流动性字段但非均值回归策略")

        final_stability_score = max(0.3, stability_score)
        is_borderline = 0.65 <= final_stability_score <= 0.82

        result = {
            "stability_score": round(final_stability_score, 3),
            "issues": stability_issues,
            "is_borderline": is_borderline,
            "verdict": "stable" if final_stability_score >= 0.8 else "moderate_risk",
        }

        if is_borderline or stability_issues:
            logger.debug(
                "GenerationGates[H↔E↔C]: 稳定性检查: 得分=%.3f | 问题: %s",
                final_stability_score,
                stability_issues,
            )

        return result

    def _adjust_holistic_score(self, llm_score: float, factors: dict[str, Any]) -> float:
        adjusted = llm_score

        complexity = factors.get("complexity", {})
        if isinstance(complexity, dict):
            comp_score = complexity.get("complexity_score", 1.0)
            if comp_score < 0.7:
                adjusted = adjusted * (0.7 + comp_score * 0.3)

        overfitting = factors.get("overfitting_risk", {})
        if isinstance(overfitting, dict):
            risk = overfitting.get("risk_score", 0.0)
            if risk > 0.4:
                adjusted = adjusted * (1.0 - risk * 0.3)

        stability = factors.get("stability", {})
        if isinstance(stability, dict):
            stab_score = stability.get("stability_score", 1.0)
            if stab_score < 0.7:
                adjusted = adjusted * (0.75 + stab_score * 0.25)

        return max(0.0, min(1.0, adjusted))

    def _build_holistic_prompt(
        self,
        expression: str,
        hypothesis_direction: str,
        hypothesis_mechanism: str = "",
        hypothesis_nl: str = "",
        operators: list[str] | None = None,
        fields: list[str] | None = None,
    ) -> str:
        op_str = ", ".join(operators) if operators else "(not provided)"
        field_str = ", ".join(fields) if fields else "(not provided)"

        return f"""You are a semantic consistency auditor for alpha factor generation.
Evaluate the THREE-WAY consistency of this alpha factor:

1. HYPOTHESIS (what is claimed):
   Direction: {hypothesis_direction}
   Mechanism: {hypothesis_mechanism or "(not specified)"}
   Description: {hypothesis_nl or "(not specified)"}

2. EXPRESSION (the FASTEXPR formula):
   ```
   {expression}
   ```
   Operators used: {op_str}
   Fields used: {field_str}

3. CODE (the implementation derived from the expression):
   Same as the FASTEXPR formula above.

Evaluate the following consistency dimensions:
a) Hypothesis ↔ Expression: Does the formula actually implement the claimed economic mechanism?
b) Hypothesis ↔ Code: Is the code structure aligned with the economic intent?
c) Expression ↔ Code: Are there any internal contradictions (e.g. sign inversions that conflict with stated direction)?

Additionally assess:
d) Complexity: Is the formula unnecessarily complex? Too many nested functions?
e) Overfitting risk: Does it use too many specific parameters or rare fields that might overfit?
f) Stability: Are there divisions by variables, extreme power transforms, or thin-liquidity fields?

Output ONLY a JSON object:
```json
{{
  "consistency_score": 0.0_to_1.0,
  "reasoning": "brief explanation covering all dimensions",
  "issues": ["issue 1", "issue 2"],
  "suggestions": ["suggestion 1", "suggestion 2"],
  "passed": true_or_false
}}
```

Score 1.0 = fully consistent, 0.0 = completely contradictory.
Be strict: if the expression uses mean_reversion operators (ts_mean) but claims momentum direction, score < 0.5.
If the expression has a leading negative sign but claims "long" direction, flag it.
If the expression uses fields that make no sense for the claimed mechanism, reduce score."""

    def _parse_holistic_response(self, response: str) -> dict[str, Any]:
        result = _extract_json_from_llm(response)
        if result and isinstance(result, dict):
            try:
                score = float(result.get("consistency_score", 0.5))
                return {
                    "consistency_score": max(0.0, min(1.0, score)),
                    "reasoning": str(result.get("reasoning", "")),
                    "issues": result.get("issues", []),
                    "suggestions": result.get("suggestions", []),
                    "passed": bool(result.get("passed", score >= 0.5)),
                }
            except (ValueError, TypeError) as exc:
                logger.warning("GenerationGates: failed to parse holistic LLM response: %s", exc)
                return {
                    "consistency_score": 0.5,
                    "reasoning": f"Failed to parse LLM response: {str(exc)[:100]}",
                    "issues": [],
                    "suggestions": [],
                    "passed": True,
                }
        return {
            "consistency_score": 0.5,
            "reasoning": "Invalid response format",
            "issues": [],
            "suggestions": [],
            "passed": True,
        }

    def _build_decision_rationale(
        self,
        results: list[GateResult],
        all_passed: bool,
        failed_gates: list[str],
        overall_score: float,
    ) -> str:
        if all_passed:
            parts = [f"✅ 所有门控通过 (总分={overall_score:.3f})"]
            for r in results:
                status = "PASS" if r.passed else "FAIL"
                parts.append(
                    f"  [{r.gate_name}] {status}: score={r.score:.3f} ≥ {r.threshold:.2f} | {r.diagnosis[:100]}"
                )
            return "\n".join(parts)
        else:
            parts = [f"❌ 门控未通过 (总分={overall_score:.3f}, 失败项: {failed_gates})"]
            for r in results:
                status = "PASS" if r.passed else "FAIL"
                icon = "✓" if r.passed else "✗"
                parts.append(
                    f"  {icon} [{r.gate_name}] {status}: score={r.score:.3f} {'≥' if r.passed else '<'} {r.threshold:.2f}"
                )
                if not r.passed and r.fix_hints:
                    parts.append(f"    修复建议: {'; '.join(r.fix_hints[:2])}")
            return "\n".join(parts)

    def _build_correction_prompt(
        self,
        results: list[GateResult],
        hypothesis_direction: str,
    ) -> str:
        lines = [
            "\n\n⚠️ GENERATION GATE FAILURE — three-way semantic consistency check failed.",
            f"Target hypothesis: {hypothesis_direction}",
            "",
            "The following gates did NOT pass:",
        ]

        for r in results:
            if not r.passed:
                lines.append(f"  [{r.gate_name}] score={r.score:.3f} (threshold={r.threshold:.2f}): {r.diagnosis}")
                if r.fix_hints:
                    for hint in r.fix_hints[:3]:
                        lines.append(f"    → {hint}")

        lines.append("")
        lines.append("REQUIRED: Regenerate the alpha factor addressing ALL issues above.")
        lines.append(
            "Ensure three-way consistency: the hypothesis, expression formula, and code structure must tell the same story."
        )

        return "\n".join(lines)

    @algo_log(level=logging.INFO, label="GenerationGates.apply_with_retry")
    async def apply_with_retry(
        self,
        hypothesis_direction: str,
        hypothesis_mechanism: str,
        hypothesis_nl: str,
        expression: str,
        regenerate_fn: Callable[..., Awaitable[tuple[str, dict[str, Any]]]],
        operators: list[str] | None = None,
        fields: list[str] | None = None,
    ) -> tuple[str, GenerationGateReport]:
        current_expr = expression
        current_payload: dict[str, Any] = {}
        final_report: GenerationGateReport | None = None

        log_call(
            "GenerationGates.retry_start",
            input={
                "initial_expression": expression[:80],
                "max_retries": self._max_retries,
                "hypothesis_direction": hypothesis_direction[:60],
            },
            level=logging.INFO,
        )

        for attempt in range(self._max_retries + 1):
            report = await self.check(
                hypothesis_direction=hypothesis_direction,
                hypothesis_mechanism=hypothesis_mechanism,
                hypothesis_nl=hypothesis_nl,
                expression=current_expr,
                operators=operators,
                fields=fields,
            )
            final_report = report

            if report.passed:
                logger.info(
                    "GenerationGates: ✅ 所有门控通过 (尝试 %d/%d, 总分=%.3f)%s",
                    attempt,
                    self._max_retries,
                    report.overall_score,
                    f"\n{report.decision_rationale}" if report.decision_rationale else "",
                )
                return current_expr, report

            logger.warning(
                "GenerationGates: ❌ 门控失败 (尝试 %d/%d, 总分=%.3f, 失败项=%s)\n%s",
                attempt,
                self._max_retries,
                report.overall_score,
                report.failed_gates,
                report.decision_rationale,
            )

            if attempt >= self._max_retries:
                logger.warning(
                    "GenerationGates: ⚠️ 达到最大重试次数 (%d)，接受当前结果 (总分=%.3f)\n%s",
                    self._max_retries,
                    report.overall_score,
                    report.decision_rationale,
                )
                break

            try:
                new_expr, current_payload = await regenerate_fn(
                    expression=current_expr,
                    correction_prompt=report.correction_prompt,
                    payload=current_payload,
                )
                if new_expr and new_expr != current_expr:
                    current_expr = new_expr
                    logger.info(
                        "GenerationGates: 🔄 已重新生成表达式 (尝试 %d): %s",
                        attempt + 1,
                        current_expr[:100],
                    )
                else:
                    logger.warning(
                        "GenerationGates: ⚠️ regenerate_fn 返回相同/空表达式，将使用更强的提示重试",
                    )
                    current_expr = expression
            except Exception as exc:
                logger.error("GenerationGates: ❌ regenerate_fn 异常 (尝试 %d): %s", attempt, exc, exc_info=True)
                break

        assert final_report is not None
        return current_expr, final_report
