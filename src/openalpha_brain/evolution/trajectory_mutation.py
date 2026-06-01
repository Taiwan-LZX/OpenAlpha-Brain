"""
TrajectoryMutation — 轨迹级变异策略模块。

包含两个变异引擎：
  1. TemplateTrajectoryMutation — 基于固定模板的确定性变异（保留作为降级回退）
  2. TrajectoryMutationV2 — QuantaAlpha LLM自我反思式定向变异（主引擎）

TrajectoryMutationV2 基于 QuantaAlpha (arXiv:2602.07085) 的核心思想：
  - LLM 自我反思诊断轨迹中哪个步骤导致次优结果
  - 仅定向重写失败段，保留轨迹其余部分不变
  - 避免全量重生成导致的语义漂移

多级 Fallback 策略（参考 OpenEvolve 框架）：
  Level 1: LLM 轨迹级变异（原有逻辑，增强错误处理）
  Level 2: AST-based 表达式级变异（当 LLM 失败时自动降级）
  Level 3: 参数微调变异（当 AST 也失败时的最后手段）
  Level 4: 返回原始轨迹 + 警告日志（保底方案，绝不返回空）
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from openalpha_brain.evolution.evolution_types import AlphaTrajectory
from openalpha_brain.utils import extract_json_from_llm as _extract_json_from_llm
from openalpha_brain.utils.algo_logger import Timer, algo_log, log_call

logger = logging.getLogger(__name__)


class TemplateTrajectoryMutation:
    """基于固定模板的确定性轨迹变异（降级回退/算法参考）。

    已被 TrajectoryMutationV2（LLM 驱动的智能定向变异）替代作为主要变异引擎。
    保留此实现作为 LLM 不可用时的确定性回退。
    """

    def __init__(self, operator_swap_map: dict | None = None):
        self._swap_map = operator_swap_map or {}

    def mutate_trajectory(self, trajectory: AlphaTrajectory) -> list[AlphaTrajectory]:
        if not trajectory.decision_points:
            return []

        variants = []
        for i, decision in enumerate(trajectory.decision_points):
            for alt in decision.get("alternatives", [])[:2]:
                new_trajectory = AlphaTrajectory(
                    hypothesis_direction=trajectory.hypothesis_direction,
                    hypothesis_mechanism=trajectory.hypothesis_mechanism,
                    expression_versions=list(trajectory.expression_versions),
                    brain_feedbacks=list(trajectory.brain_feedbacks),
                    decision_points=list(trajectory.decision_points),
                    final_status="PENDING",
                )
                new_trajectory.decision_points[i] = {
                    "type": decision["type"],
                    "chosen": alt,
                    "alternatives": decision["alternatives"],
                }
                last_expr = trajectory.expression_versions[-1] if trajectory.expression_versions else ""
                if last_expr and decision["type"] == "operator":
                    mutated = last_expr.replace(decision["chosen"], alt, 1)
                    new_trajectory.expression_versions.append(mutated)
                elif last_expr and decision["type"] == "parameter":
                    pattern = r"(\d+)"
                    matches = list(re.finditer(pattern, last_expr))
                    if matches:
                        mutated = last_expr[: matches[0].start()] + str(alt) + last_expr[matches[0].end() :]
                        new_trajectory.expression_versions.append(mutated)
                    else:
                        new_trajectory.expression_versions.append(last_expr)
                else:
                    new_trajectory.expression_versions.append(last_expr)
                variants.append(new_trajectory)

        return variants[:5]

    def crossover_trajectories(self, traj_a: AlphaTrajectory, traj_b: AlphaTrajectory) -> list[AlphaTrajectory]:
        if not traj_a.decision_points or not traj_b.decision_points:
            return []

        children = []
        min_len = min(len(traj_a.decision_points), len(traj_b.decision_points))
        for i in range(min_len):
            child = AlphaTrajectory(
                hypothesis_direction=traj_a.hypothesis_direction,
                hypothesis_mechanism=traj_b.hypothesis_mechanism,
                expression_versions=list(traj_a.expression_versions[:1]) if traj_a.expression_versions else [],
                decision_points=traj_a.decision_points[: i + 1] + traj_b.decision_points[i + 1 :],
                final_status="PENDING",
            )
            children.append(child)

        return children[:3]


TrajectoryMutation = TemplateTrajectoryMutation


_SEGMENT_TYPES = ("hypothesis", "factor_structure", "parameter_choice", "repair_behavior")


@dataclass
class TrajectoryMutationResult:
    mutated_trajectory: AlphaTrajectory
    original_id: str
    mutation_type: str
    weak_segment: str
    diagnosis: str
    revision_summary: str


class TrajectoryMutationV2:
    """QuantaAlpha 轨迹级定向变异。

    基于 LLM 自我反思：
    1. 诊断轨迹中最薄弱的段（假设/因子结构/参数/修复行为）
    2. 仅定向重写该段，保留其余有效部分
    3. 生成 1-3 个针对性修复的变体轨迹

    多级 Fallback 策略：
      Level 1: LLM 轨迹级变异（主引擎）
      Level 2: AST-based 表达式级变异（LLM 失败时自动降级）
      Level 3: 参数微调变异（AST 失败时的最后手段）
      Level 4: 返回原始轨迹 + 警告日志（保底方案）

    淘汰并替换了旧版 TemplateTrajectoryMutation 的模板式全量变异。
    """

    DEFAULT_MAX_VARIANTS: int = 3
    DEFAULT_MUTATION_DEPTH: float = 0.5

    def __init__(self, llm_generate_fn: Callable[..., Awaitable[str]] | None = None):
        self._llm = llm_generate_fn
        self._template = TemplateTrajectoryMutation()

    def _compute_dynamic_max_variants(self, trajectory: AlphaTrajectory) -> int:
        steps_count = len(trajectory.decision_points)
        dynamic_max = max(3, steps_count // 2)
        return min(dynamic_max, self.DEFAULT_MAX_VARIANTS * 2)

    def _compute_mutation_depth(self, trajectory: AlphaTrajectory) -> float:
        steps_count = len(trajectory.decision_points)
        if steps_count <= 3:
            return 0.8
        elif steps_count <= 6:
            return 0.5
        else:
            return 0.3

    def _serialize_for_diagnosis(self, traj: AlphaTrajectory) -> dict[str, Any]:
        return {
            "hypothesis_direction": traj.hypothesis_direction,
            "hypothesis_mechanism": traj.hypothesis_mechanism,
            "final_status": traj.final_status,
            "final_sharpe": traj.final_sharpe,
            "expression_versions": [{"v": i + 1, "expr": e} for i, e in enumerate(traj.expression_versions)],
            "decision_points": [
                {"step": i, "type": d.get("type"), "chosen": d.get("chosen"), "alternatives": d.get("alternatives", [])}
                for i, d in enumerate(traj.decision_points)
            ],
            "brain_feedbacks": [
                {"sharpe": f.get("sharpe"), "status": f.get("status"), "feedback": str(f.get("feedback", ""))[:200]}
                for f in traj.brain_feedbacks[-5:]
            ],
        }

    def _build_diagnosis_prompt(self, traj: AlphaTrajectory) -> str:
        ser = json.dumps(self._serialize_for_diagnosis(traj), ensure_ascii=False, indent=2)

        return f"""You are an expert quantitative researcher performing a post-mortem on an alpha mining trajectory.

A "trajectory" is the complete end-to-end mining run: hypothesis → factor construction → backtest evaluation.

Your task: Diagnose which specific segment of this trajectory is the WEAKEST and caused suboptimal performance.

Analyze these four segment types:
1. **hypothesis** — The economic intuition; was it flawed or not well-matched to the market mechanism?
2. **factor_structure** — The operator composition and data transformations; was there a better way to capture the hypothesis?  # noqa: E501
3. **parameter_choice** — Lookback windows, thresholds, scaling factors; were suboptimal values chosen?
4. **repair_behavior** — How failures were diagnosed and fixed; could a different repair strategy have worked?

**TRAJECTORY TO DIAGNOSE:**
{ser}

Output a JSON with this structure:
```json
{{{{
  "weakest_segment": "hypothesis | factor_structure | parameter_choice | repair_behavior",
  "diagnosis": "Detailed analysis of why this segment is the bottleneck",
  "evidence": "Specific evidence from the trajectory supporting this diagnosis",
  "severity": "critical | moderate | minor",
  "potential_revisions": [
    "Specific revision idea 1",
    "Specific revision idea 2",
    "Specific revision idea 3"
  ]
}}}}
```

Output only the JSON, no other text."""

    def _build_rewrite_prompt(self, traj: AlphaTrajectory, diagnosis: dict[str, Any]) -> str:
        ser = json.dumps(self._serialize_for_diagnosis(traj), ensure_ascii=False, indent=2)
        diag_json = json.dumps(diagnosis, ensure_ascii=False, indent=2)

        return f"""You are an expert quantitative researcher performing TARGETED REVISION on one segment of an alpha mining trajectory.  # noqa: E501

**DIAGNOSIS** (the weakest segment has been identified):
{diag_json}

**ORIGINAL TRAJECTORY:**
{ser}

Your task: Rewrite ONLY the diagnosed weak segment. Keep all other segments EXACTLY as they are.

For the rewrite, output a JSON with this structure:
```json
{{{{
  "rewrites": [
    {{{{
      "strategy": "Name of this revision approach",
      "description": "What changed and why",
      "new_hypothesis_direction": "momentum | mean_reversion | volatility | statistical | volume | interaction",
      "new_hypothesis_mechanism": "Revised mechanism description",
      "new_expression_suggestion": "High-level description of the revised expression"
    }}}}
  ]
}}}}
```

Output only the JSON, no other text."""

    def _log_trajectory_summary(self, trajectory: AlphaTrajectory, label: str = "input") -> None:
        log_call(
            f"trajectory_mutation_{label}",
            input={
                "steps_count": len(trajectory.decision_points),
                "expressions_count": len(trajectory.expression_versions),
                "current_sharpe": getattr(trajectory, "final_sharpe", None),
                "direction": trajectory.hypothesis_direction,
                "mechanism": trajectory.hypothesis_mechanism[:100] if trajectory.hypothesis_mechanism else None,
            },
            level=logging.INFO,
        )

    def _fallback_level_2_ast_mutation(
        self,
        trajectory: AlphaTrajectory,
        original_id: str,
        max_variants: int,
        trigger_reason: str,
    ) -> list[TrajectoryMutationResult]:
        results: list[TrajectoryMutationResult] = []

        try:
            with Timer("fallback_level2_ast_mutation"):
                last_expr = trajectory.expression_versions[-1] if trajectory.expression_versions else ""

                if not last_expr:
                    logger.warning("TrajectoryMutationV2: Level 2 fallback failed — 无表达式可变异")
                    return []

                ast_variants = self._apply_ast_mutations(last_expr, max_variants)

                if not ast_variants:
                    logger.warning("TrajectoryMutationV2: Level 2 fallback failed — AST 变异未产生结果")
                    return []

                for mutated_expr in ast_variants[:max_variants]:
                    child = AlphaTrajectory(
                        hypothesis_direction=trajectory.hypothesis_direction,
                        hypothesis_mechanism=trajectory.hypothesis_mechanism,
                        expression_versions=list(trajectory.expression_versions) + [mutated_expr],
                        brain_feedbacks=list(trajectory.brain_feedbacks),
                        decision_points=list(trajectory.decision_points),
                        final_status="PENDING",
                    )
                    child.add_decision("ast_mutation", "expression_rewritten", [])
                    child.add_decision("fallback_level", "2", [])

                    results.append(
                        TrajectoryMutationResult(
                            mutated_trajectory=child,
                            original_id=original_id,
                            mutation_type="ast_based_fallback",
                            weak_segment="expression",
                            diagnosis=f"Level 2 fallback (AST): {trigger_reason}",
                            revision_summary=f"AST-based expression mutation: {mutated_expr[:100]}",
                        )
                    )

                log_call(
                    "fallback_level2_complete",
                    input={"trigger_reason": trigger_reason},
                    output={"variants_generated": len(results)},
                    level=logging.INFO,
                )

                logger.info(
                    "TrajectoryMutationV2: Level 2 fallback 成功 — 生成 %d 个 AST 变体 (原因: %s)",
                    len(results),
                    trigger_reason,
                )

        except (ValueError, TypeError, RuntimeError) as exc:
            log_call(
                "fallback_level2_failed",
                error=str(exc),
                level=logging.WARNING,
            )
            logger.warning("TrajectoryMutationV2: Level 2 fallback 异常 — %s", exc)

        return results

    def _apply_ast_mutations(self, expression: str, max_variants: int) -> list[str]:
        variants = []

        try:
            operator_swaps = {
                "rank": "zscore",
                "zscore": "rank",
                "ts_mean": "ts_decay_linear",
                "ts_decay_linear": "ts_mean",
                "scale": "sign",
                "sign": "scale",
            }

            window_adjustments = ["5", "10", "20", "60"]

            for old_op, new_op in list(operator_swaps.items())[:max_variants]:
                if old_op in expression.lower():
                    mutated = re.sub(r"\b" + re.escape(old_op) + r"\b", new_op, expression, count=1)
                    if mutated != expression:
                        variants.append(mutated)

            if len(variants) < max_variants:
                for window in window_adjustments[: max_variants - len(variants)]:
                    pattern = r"(\d+)(?=\))"
                    match = re.search(pattern, expression)
                    if match:
                        mutated = expression[: match.start()] + window + expression[match.end() :]
                        if mutated != expression and mutated not in variants:
                            variants.append(mutated)

        except (ValueError, TypeError, RuntimeError) as exc:
            logger.warning("TrajectoryMutationV2: AST 变异执行异常 — %s", exc)

        return variants[:max_variants]

    def _fallback_level_3_parameter_tweak(
        self,
        trajectory: AlphaTrajectory,
        original_id: str,
        max_variants: int,
        trigger_reason: str,
    ) -> list[TrajectoryMutationResult]:
        results: list[TrajectoryMutationResult] = []

        try:
            with Timer("fallback_level3_parameter_tweak"):
                if not trajectory.decision_points:
                    logger.warning("TrajectoryMutationV2: Level 3 fallback failed — 无决策点")
                    return []

                tweak_variants = self._apply_parameter_tweaks(trajectory, max_variants)

                if not tweak_variants:
                    logger.warning("TrajectoryMutationV2: Level 3 fallback failed — 参数微调未产生结果")
                    return []

                for tweaked_traj in tweak_variants[:max_variants]:
                    tweaked_traj.add_decision("parameter_tweak", "value_adjusted", [])
                    tweaked_traj.add_decision("fallback_level", "3", [])

                    results.append(
                        TrajectoryMutationResult(
                            mutated_trajectory=tweaked_traj,
                            original_id=original_id,
                            mutation_type="parameter_tweak_fallback",
                            weak_segment="parameter",
                            diagnosis=f"Level 3 fallback (参数微调): {trigger_reason}",
                            revision_summary="Parameter value adjustment based on alternatives",
                        )
                    )

                log_call(
                    "fallback_level3_complete",
                    input={"trigger_reason": trigger_reason},
                    output={"variants_generated": len(results)},
                    level=logging.INFO,
                )

                logger.info(
                    "TrajectoryMutationV2: Level 3 fallback 成功 — 生成 %d 个参数微调变体 (原因: %s)",
                    len(results),
                    trigger_reason,
                )

        except (ValueError, TypeError, RuntimeError) as exc:
            log_call(
                "fallback_level3_failed",
                error=str(exc),
                level=logging.WARNING,
            )
            logger.warning("TrajectoryMutationV2: Level 3 fallback 异常 — %s", exc)

        return results

    def _apply_parameter_tweaks(self, trajectory: AlphaTrajectory, max_variants: int) -> list[AlphaTrajectory]:
        variants = []
        tweak_count = 0

        for i, decision in enumerate(trajectory.decision_points):
            if tweak_count >= max_variants:
                break

            alternatives = decision.get("alternatives", [])
            chosen = decision.get("chosen", "")

            for alt in alternatives[:2]:
                if tweak_count >= max_variants:
                    break

                if alt != chosen:
                    new_trajectory = AlphaTrajectory(
                        hypothesis_direction=trajectory.hypothesis_direction,
                        hypothesis_mechanism=trajectory.hypothesis_mechanism,
                        expression_versions=list(trajectory.expression_versions),
                        brain_feedbacks=list(trajectory.brain_feedbacks),
                        decision_points=list(trajectory.decision_points),
                        final_status="PENDING",
                    )

                    new_decision = dict(decision)
                    new_decision["chosen"] = alt
                    new_trajectory.decision_points[i] = new_decision

                    last_expr = trajectory.expression_versions[-1] if trajectory.expression_versions else ""
                    if last_expr and decision.get("type") == "operator":
                        mutated = last_expr.replace(chosen, alt, 1)
                        new_trajectory.expression_versions.append(mutated)
                    elif last_expr and decision.get("type") == "parameter":
                        pattern = r"(\d+)"
                        matches = list(re.finditer(pattern, last_expr))
                        if matches:
                            mutated = last_expr[: matches[0].start()] + str(alt) + last_expr[matches[0].end() :]
                            new_trajectory.expression_versions.append(mutated)
                        else:
                            new_trajectory.expression_versions.append(last_expr)
                    else:
                        new_trajectory.expression_versions.append(last_expr)

                    variants.append(new_trajectory)
                    tweak_count += 1

        return variants

    def _fallback_level_4_return_original(
        self,
        trajectory: AlphaTrajectory,
        original_id: str,
        trigger_reason: str,
    ) -> list[TrajectoryMutationResult]:
        log_call(
            "fallback_level4_original",
            input={
                "trigger_reason": trigger_reason,
                "original_steps": len(trajectory.decision_points),
            },
            level=logging.WARNING,
        )

        logger.warning(
            "TrajectoryMutationV2: 所有变异策略均失败，返回原始轨迹作为保底 (原因: %s)",
            trigger_reason,
        )

        original_copy = AlphaTrajectory(
            hypothesis_direction=trajectory.hypothesis_direction,
            hypothesis_mechanism=trajectory.hypothesis_mechanism,
            expression_versions=list(trajectory.expression_versions),
            brain_feedbacks=list(trajectory.brain_feedbacks),
            decision_points=list(trajectory.decision_points),
            final_status=trajectory.final_status,
        )

        original_copy.add_decision("fallback_preserved", "original_trajectory", [])
        original_copy.add_decision("fallback_level", "4", [])
        original_copy.add_decision("fallback_reason", trigger_reason[:200], [])

        return [
            TrajectoryMutationResult(
                mutated_trajectory=original_copy,
                original_id=original_id,
                mutation_type="original_preserved_fallback",
                weak_segment="none",
                diagnosis=f"Level 4 fallback (保底): {trigger_reason}",
                revision_summary="Original trajectory preserved due to all mutation strategies failing",
            )
        ]

    @algo_log(log_args_to_skip=("self",))
    async def mutate_trajectory(
        self,
        trajectory: AlphaTrajectory,
        original_id: str = "",
    ) -> list[TrajectoryMutationResult]:
        """执行轨迹级定向变异（含多级 Fallback 策略）。

        Fallback 层级：
          Level 1: LLM 轨迹级变异（主引擎）
          Level 2: AST-based 表达式级变异（LLM 失败时降级）
          Level 3: 参数微调变异（AST 失败时降级）
          Level 4: 返回原始轨迹 + 警告日志（保底，绝不返回空）
        """
        results: list[TrajectoryMutationResult] = []

        dynamic_max_variants = self._compute_dynamic_max_variants(trajectory)
        mutation_depth = self._compute_mutation_depth(trajectory)

        self._log_trajectory_summary(trajectory, label="input")

        log_call(
            "mutation_config",
            input={
                "dynamic_max_variants": dynamic_max_variants,
                "mutation_depth": mutation_depth,
                "steps_in_trajectory": len(trajectory.decision_points),
                "llm_available": self._llm is not None,
            },
            level=logging.DEBUG,
        )

        if self._llm is None:
            logger.info("TrajectoryMutationV2: LLM unavailable, starting fallback chain")
            results = self._try_fallback_chain(
                trajectory=trajectory,
                original_id=original_id,
                max_variants=dynamic_max_variants,
                start_level=2,
                trigger_reason="LLM function not provided",
            )
            self._log_output_summary(results, "llm_unavailable")
            return results

        try:
            with Timer("level1_llm_mutation"):
                results = await self._execute_level1_llm_mutation(
                    trajectory=trajectory,
                    original_id=original_id,
                    max_variants=dynamic_max_variants,
                    mutation_depth=mutation_depth,
                )

            if results:
                self._log_output_summary(results, "level1_success")
                return results

            logger.info("TrajectoryMutationV2: Level 1 未产生结果，启动 fallback chain")

        except (ValueError, TypeError, RuntimeError) as exc:
            log_call(
                "level1_llm_exception",
                error=str(exc),
                level=logging.WARNING,
            )
            logger.warning("TrajectoryMutationV2: Level 1 LLM 异常 — %s", exc)

        results = self._try_fallback_chain(
            trajectory=trajectory,
            original_id=original_id,
            max_variants=dynamic_max_variants,
            start_level=2,
            trigger_reason="Level 1 LLM mutation failed or returned empty",
        )

        self._log_output_summary(results, "fallback_chain_result")
        return results

    async def _execute_level1_llm_mutation(
        self,
        trajectory: AlphaTrajectory,
        original_id: str,
        max_variants: int,
        mutation_depth: float,
    ) -> list[TrajectoryMutationResult]:
        results: list[TrajectoryMutationResult] = []

        try:
            diag_prompt = self._build_diagnosis_prompt(trajectory)

            log_call(
                "llm_diagnosis_request",
                input={"trajectory_steps": len(trajectory.decision_points)},
                level=logging.DEBUG,
            )

            diag_response = await self._llm(diag_prompt)

            log_call(
                "llm_diagnosis_response",
                input={
                    "response_length": len(diag_response) if diag_response else 0,
                    "response_preview": (diag_response or "")[:100],
                },
                level=logging.DEBUG,
            )

            diagnosis = _extract_json_from_llm(diag_response)

            if not diagnosis or not isinstance(diagnosis, dict):
                log_call(
                    "llm_diagnosis_parse_failed",
                    input={"raw_response_preview": (diag_response or "")[:200]},
                    level=logging.WARNING,
                )
                logger.warning("TrajectoryMutationV2: 无法解析诊断 JSON，触发 Level 2 fallback")
                return []

            weak_segment = str(diagnosis.get("weakest_segment", "unknown"))
            severity = str(diagnosis.get("severity", "moderate"))
            diag_text = str(diagnosis.get("diagnosis", ""))

            log_call(
                "llm_diagnosis_parsed",
                input={
                    "weak_segment": weak_segment,
                    "severity": severity,
                    "diagnosis_preview": diag_text[:150],
                },
                level=logging.INFO,
            )

            logger.info(
                "TrajectoryMutationV2: 诊断完成 weakest_segment=%s severity=%s — %s",
                weak_segment,
                severity,
                diag_text[:100],
            )

            if severity == "minor" and mutation_depth < 0.7:
                logger.info(
                    "TrajectoryMutationV2: severity=minor 且 mutation_depth=%.2f < 0.7，跳过重写", mutation_depth
                )
                return []

            rewrite_prompt = self._build_rewrite_prompt(trajectory, diagnosis)

            log_call(
                "llm_rewrite_request",
                input={
                    "weak_segment": weak_segment,
                    "target_variants": max_variants,
                },
                level=logging.DEBUG,
            )

            rewrite_response = await self._llm(rewrite_prompt)

            log_call(
                "llm_rewrite_response",
                input={
                    "response_length": len(rewrite_response) if rewrite_response else 0,
                    "response_preview": (rewrite_response or "")[:100],
                },
                level=logging.DEBUG,
            )

            rewrite_plan = _extract_json_from_llm(rewrite_response)

            if not rewrite_plan or not isinstance(rewrite_plan, dict):
                log_call(
                    "llm_rewrite_parse_failed",
                    input={"raw_response_preview": (rewrite_response or "")[:200]},
                    level=logging.WARNING,
                )
                logger.warning("TrajectoryMutationV2: 无法解析重写计划 JSON，触发 Level 2 fallback")
                return []

            rewrites = rewrite_plan.get("rewrites", [])

            actual_variants = min(len(rewrites), max_variants)

            for rw in rewrites[:actual_variants]:
                child = AlphaTrajectory(
                    hypothesis_direction=str(rw.get("new_hypothesis_direction", trajectory.hypothesis_direction)),
                    hypothesis_mechanism=str(rw.get("new_hypothesis_mechanism", trajectory.hypothesis_mechanism)),
                    expression_versions=list(trajectory.expression_versions),
                    brain_feedbacks=list(trajectory.brain_feedbacks),
                    decision_points=list(trajectory.decision_points),
                    final_status="PENDING",
                )

                child.add_decision("targeted_mutation", str(rw.get("strategy", "")), [])
                child.add_decision("weak_segment", weak_segment, [])
                child.add_decision("diagnosis_summary", diag_text[:200], [])
                child.add_decision("mutation_level", "1", [])

                revision_desc = str(rw.get("description", ""))
                if revision_desc:
                    child.add_decision("revision_description", revision_desc[:200], [])

                results.append(
                    TrajectoryMutationResult(
                        mutated_trajectory=child,
                        original_id=original_id,
                        mutation_type=f"targeted_{weak_segment}",
                        weak_segment=weak_segment,
                        diagnosis=diag_text,
                        revision_summary=revision_desc,
                    )
                )

            if results:
                log_call(
                    "level1_success",
                    input={
                        "variants_generated": len(results),
                        "weak_segment": weak_segment,
                    },
                    level=logging.INFO,
                )
                logger.info(
                    "TrajectoryMutationV2: Level 1 成功 — 生成 %d 个定向修订变体 (weak_segment=%s)",
                    len(results),
                    weak_segment,
                )

        except TimeoutError as exc:
            log_call(
                "llm_timeout",
                error=str(exc),
                level=logging.WARNING,
            )
            logger.warning("TrajectoryMutationV2: LLM 调用超时 — %s", exc)
            raise

        except ConnectionError as exc:
            log_call(
                "llm_connection_error",
                error=str(exc),
                level=logging.WARNING,
            )
            logger.warning("TrajectoryMutationV2: LLM 连接错误 — %s", exc)
            raise

        except ValueError as exc:
            log_call(
                "llm_value_error",
                error=str(exc),
                level=logging.WARNING,
            )
            logger.warning("TrajectoryMutationV2: LLM 返回值错误 — %s", exc)
            raise

        except Exception as exc:
            log_call(
                "llm_unexpected_error",
                error=str(exc),
                level=logging.ERROR,
            )
            logger.error("TrajectoryMutationV2: LLM 未预期异常 — %s", exc, exc_info=True)
            raise

        return results

    def _try_fallback_chain(
        self,
        trajectory: AlphaTrajectory,
        original_id: str,
        max_variants: int,
        start_level: int,
        trigger_reason: str,
    ) -> list[TrajectoryMutationResult]:
        results: list[TrajectoryMutationResult] = []

        if start_level <= 2:
            logger.info("TrajectoryMutationV2: 尝试 Level 2 fallback (AST-based) — %s", trigger_reason)
            results = self._fallback_level_2_ast_mutation(
                trajectory=trajectory,
                original_id=original_id,
                max_variants=max_variants,
                trigger_reason=trigger_reason,
            )

            if results:
                return results

        if start_level <= 3:
            logger.info("TrajectoryMutationV2: 尝试 Level 3 fallback (参数微调) — %s", trigger_reason)
            results = self._fallback_level_3_parameter_tweak(
                trajectory=trajectory,
                original_id=original_id,
                max_variants=max_variants,
                trigger_reason=trigger_reason,
            )

            if results:
                return results

        logger.warning("TrajectoryMutationV2: 触发 Level 4 保底机制 — %s", trigger_reason)
        results = self._fallback_level_4_return_original(
            trajectory=trajectory,
            original_id=original_id,
            trigger_reason=trigger_reason,
        )

        return results

    def _log_output_summary(self, results: list[TrajectoryMutationResult], source: str) -> None:
        if not results:
            log_call(
                "mutation_output_empty",
                input={"source": source},
                level=logging.WARNING,
            )
            return

        mutation_types = [r.mutation_type for r in results]
        weak_segments = [r.weak_segment for r in results]

        log_call(
            "mutation_output_summary",
            input={
                "source": source,
                "variants_count": len(results),
                "mutation_types": mutation_types,
                "weak_segments": weak_segments,
            },
            level=logging.INFO,
        )

        logger.info(
            "TrajectoryMutationV2: 输出摘要 — source=%s, 变体数=%d, 类型=%s, 薄弱段=%s",
            source,
            len(results),
            mutation_types,
            weak_segments,
        )

    @algo_log(log_args_to_skip=("self",))
    def mutate_sync(self, trajectory: AlphaTrajectory, original_id: str = "") -> list[TrajectoryMutationResult]:
        """同步回退方法，使用多级 fallback 策略。"""
        logger.info("TrajectoryMutationV2: sync fallback requested, using fallback chain")

        dynamic_max_variants = self._compute_dynamic_max_variants(trajectory)

        self._log_trajectory_summary(trajectory, label="sync_input")

        results = self._try_fallback_chain(
            trajectory=trajectory,
            original_id=original_id,
            max_variants=dynamic_max_variants,
            start_level=2,
            trigger_reason="sync context requested",
        )

        self._log_output_summary(results, "sync_fallback_result")
        return results
