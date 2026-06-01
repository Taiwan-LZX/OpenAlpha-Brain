from __future__ import annotations

import logging
from typing import Any

from openalpha_brain.core import loop_state as _ls
from openalpha_brain.utils.resilience import TaskHealthRegistry, async_timeout

logger = logging.getLogger(__name__)


async def _periodic_decay_check(session_id: str) -> None:
    _health = TaskHealthRegistry.get("decay_check")
    import asyncio as _asyncio_decay

    while True:
        await _asyncio_decay.sleep(300)
        if _ls._decay_detector is None:
            continue
        try:
            _results = await async_timeout(
                _ls._decay_detector.check_all_alphas(),
                timeout_seconds=60.0,
                name="decay_check_all",
            )
            summary = await async_timeout(
                _ls._decay_detector.get_decay_summary(),
                timeout_seconds=10.0,
                name="decay_summary",
            )
            _health.record_success()
            logger.info(
                "[%s] decay_check: total=%d levels=%s blacklisted=%d",
                session_id,
                summary.get("total_tracked", 0),
                summary.get("by_level", {}),
                len(summary.get("blacklisted_directions", [])),
            )
            decay_prompt = _ls._decay_detector.build_decay_prompt_injection()
            if decay_prompt:
                _ls._decay_state = _ls.__dict__.get("_decay_state", {})
                _ls._decay_state["prompt_injection"] = decay_prompt
                logger.info("[%s] decay_prompt_injection: %d chars", session_id, len(decay_prompt))
        except TimeoutError:
            _health.record_failure("timeout")
            logger.warning("[%s] periodic decay check timed out", session_id)
        except (OSError, ValueError, RuntimeError) as exc:
            _health.record_failure(str(exc))
            logger.warning("[%s] periodic decay check failed: %s", session_id, exc)


async def _periodic_trajectory_crossover(session_id: str) -> None:
    _health = TaskHealthRegistry.get("trajectory_crossover")
    import asyncio as _asyncio_traj

    await _asyncio_traj.sleep(60)
    if not hasattr(_ls, "_trajectory_crossover_insights"):
        _ls._trajectory_crossover_insights = []
    if not hasattr(_ls, "_crossover_exploration_proposals"):
        _ls._crossover_exploration_proposals = []
    if not hasattr(_ls, "_weak_segment_alerts"):
        _ls._weak_segment_alerts = []
    while True:
        await _asyncio_traj.sleep(120)
        _eng = getattr(_ls, "_crossover_engine", None)
        if _eng is None:
            continue
        try:
            results = await async_timeout(
                _eng.crossover_trajectories(),
                timeout_seconds=45.0,
                name="crossover_trajectories",
            )
            if results:
                logger.info(
                    "[%s] TrajectoryCrossover: %d children generated",
                    session_id,
                    len(results),
                )
                _crossover_insights: list[dict[str, Any]] = []
                for _r in results:
                    _crossover_insights.append(
                        {
                            "strategy": _r.crossover_strategy,
                            "direction": _r.child_trajectory.hypothesis_direction,
                            "mechanism": _r.child_trajectory.hypothesis_mechanism,
                            "parent1": _r.parent1_id,
                            "parent2": _r.parent2_id,
                            "complementary": _r.complementary_segments,
                            "analysis": _r.llm_analysis,
                        }
                    )
                    logger.info(
                        "[%s] TrajectoryCrossover child: dir=%s mechanism=%s strategy=%s",
                        session_id,
                        _r.child_trajectory.hypothesis_direction,
                        _r.child_trajectory.hypothesis_mechanism[:80],
                        _r.crossover_strategy,
                    )
                _ls._trajectory_crossover_insights = _crossover_insights
                _ls._algo_tick("trajectory_crossover_done")
                logger.info(
                    "[%s] TrajectoryCrossover: %d insights published to build_dynamic_context",
                    session_id,
                    len(_crossover_insights),
                )

                _mut_results = await _eng.mutate_trajectory()
                if _mut_results:
                    _ls._algo_tick("trajectory_mutation_v2_done")
                    _mut_insights: list[dict[str, Any]] = []
                    for _mr in _mut_results:
                        _mut_insights.append(
                            {
                                "mutation_type": _mr.mutation_type,
                                "weak_segment": _mr.weak_segment,
                                "diagnosis": _mr.diagnosis,
                                "revision": _mr.revision_summary,
                                "direction": _mr.mutated_trajectory.hypothesis_direction,
                                "mechanism": _mr.mutated_trajectory.hypothesis_mechanism,
                            }
                        )
                        logger.info(
                            "[%s] TrajectoryMutationV2: weak=%s type=%s -> dir=%s",
                            session_id,
                            _mr.weak_segment,
                            _mr.mutation_type,
                            _mr.mutated_trajectory.hypothesis_direction,
                        )
                    _ls._trajectory_mutation_insights = _mut_insights

                _exploration_proposals: list[dict[str, Any]] = []
                for _r in results:
                    _proposal_dir = _r.child_trajectory.hypothesis_direction
                    _proposal_mechanism = _r.child_trajectory.hypothesis_mechanism
                    if _proposal_dir:
                        _proposal = {
                            "direction": _proposal_dir,
                            "mechanism": _proposal_mechanism,
                            "strategy": _r.crossover_strategy,
                            "complementary_segments": _r.complementary_segments,
                            "source": "trajectory_crossover",
                        }
                        _exploration_proposals.append(_proposal)
                        if _ls._scheduler is not None:
                            try:
                                _ls._scheduler.record_direction_result(_proposal_dir, reward=0.03)
                            except (OSError, ValueError, RuntimeError):
                                pass
                if _exploration_proposals:
                    _existing = getattr(_ls, "_crossover_exploration_proposals", [])
                    _existing.extend(_exploration_proposals)
                    _ls._crossover_exploration_proposals = _existing[-10:]
                    logger.info(
                        "[%s] TrajectoryCrossover: %d exploration proposals registered (total=%d), consumed by select_exploration_arm",
                        session_id,
                        len(_exploration_proposals),
                        len(_ls._crossover_exploration_proposals),
                    )

                _weak_segment_alerts: list[dict[str, Any]] = []
                for _mr in _mut_results:
                    if _mr.weak_segment and _mr.diagnosis:
                        _weak_segment_alerts.append(
                            {
                                "weak_segment": _mr.weak_segment,
                                "diagnosis": _mr.diagnosis,
                                "direction": _mr.mutated_trajectory.hypothesis_direction,
                            }
                        )
                if _weak_segment_alerts:
                    _ls._weak_segment_alerts = _weak_segment_alerts
                    logger.info(
                        "[%s] TrajectoryCrossover: %d weak segment alerts published to build_dynamic_context",
                        session_id,
                        len(_weak_segment_alerts),
                    )
            _health.record_success()
        except TimeoutError:
            _health.record_failure("timeout")
            logger.warning("[%s] periodic trajectory crossover timed out", session_id)
        except (OSError, ValueError, RuntimeError) as exc:
            _health.record_failure(str(exc))
            logger.warning("[%s] periodic trajectory crossover failed: %s", session_id, exc)
