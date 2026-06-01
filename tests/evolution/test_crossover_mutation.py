import pytest
from openalpha_brain.evolution.evolution_types import AlphaTrajectory
from openalpha_brain.evolution.trajectory_mutation import TemplateTrajectoryMutation


class TestAlphaTrajectory:
    def test_create_trajectory(self):
        t = AlphaTrajectory(
            hypothesis_direction="momentum",
            hypothesis_mechanism="price momentum",
        )
        assert t.hypothesis_direction == "momentum"
        assert t.final_status == "PENDING"
        assert len(t.expression_versions) == 0

    def test_add_decision(self):
        t = AlphaTrajectory(hypothesis_direction="momentum", hypothesis_mechanism="test")
        t.add_decision("operator", "ts_delta", ["ts_zscore", "ts_rank"])
        assert len(t.decision_points) == 1
        assert t.decision_points[0]["chosen"] == "ts_delta"

    def test_add_expression_version(self):
        t = AlphaTrajectory(hypothesis_direction="momentum", hypothesis_mechanism="test")
        t.add_expression_version("ts_delta(close, 5)")
        assert len(t.expression_versions) == 1

    def test_add_brain_feedback(self):
        t = AlphaTrajectory(hypothesis_direction="momentum", hypothesis_mechanism="test")
        t.add_brain_feedback({"name": "LOW_SHARPE", "result": "FAIL"})
        assert len(t.brain_feedbacks) == 1


class TestTrajectoryMutation:
    def test_mutate_trajectory(self):
        t = AlphaTrajectory(hypothesis_direction="momentum", hypothesis_mechanism="test")
        t.add_expression_version("ts_delta(close, 5)")
        t.add_decision("operator", "ts_delta", ["ts_zscore", "ts_rank"])

        mutator = TemplateTrajectoryMutation()
        variants = mutator.mutate_trajectory(t)
        assert len(variants) > 0
        assert variants[0].hypothesis_direction == "momentum"

    def test_mutate_empty_trajectory(self):
        t = AlphaTrajectory(hypothesis_direction="momentum", hypothesis_mechanism="test")
        mutator = TemplateTrajectoryMutation()
        variants = mutator.mutate_trajectory(t)
        assert len(variants) == 0

    def test_crossover_trajectories(self):
        t1 = AlphaTrajectory(hypothesis_direction="momentum", hypothesis_mechanism="test1")
        t1.add_decision("operator", "ts_delta", ["ts_zscore"])
        t1.add_expression_version("ts_delta(close, 5)")

        t2 = AlphaTrajectory(hypothesis_direction="value", hypothesis_mechanism="test2")
        t2.add_decision("operator", "rank", ["zscore"])
        t2.add_expression_version("rank(cap)")

        mutator = TemplateTrajectoryMutation()
        children = mutator.crossover_trajectories(t1, t2)
        assert len(children) > 0
