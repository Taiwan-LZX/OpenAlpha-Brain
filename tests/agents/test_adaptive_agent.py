import time

from openalpha_brain.agents.adaptive_agent import AdaptiveAgentFactory


class TestCreateSpecialistAgent:
    def test_creates_originality_agent_for_self_correlation(self):
        factory = AdaptiveAgentFactory()
        agent = factory.create_specialist_agent("SELF_CORRELATION", direction="momentum")

        assert agent is not None
        assert agent.agent_type == "originality"
        assert agent.specialty == "SELF_CORRELATION"

    def test_creates_sharpe_optimizer_for_low_sharpe(self):
        factory = AdaptiveAgentFactory()
        agent = factory.create_specialist_agent("LOW_SHARPE", direction="momentum")

        assert agent is not None
        assert agent.agent_type == "sharpe_optimizer"

    def test_creates_logic_verifier_for_unknown_failure(self):
        factory = AdaptiveAgentFactory()
        agent = factory.create_specialist_agent("UNKNOWN_FAILURE", direction="value")

        assert agent is not None
        assert agent.agent_type == "logic_verifier"

    def test_reuses_existing_agent_of_same_type(self):
        factory = AdaptiveAgentFactory()
        agent1 = factory.create_specialist_agent("SELF_CORRELATION")
        agent2 = factory.create_specialist_agent("SELF_CORRELATION")

        assert agent1.agent_id == agent2.agent_id
        assert factory.agent_count == 1


class TestGetSpecialistPrompt:
    def test_originality_prompt(self):
        factory = AdaptiveAgentFactory()
        prompt = factory.get_specialist_prompt("originality", "ts_delta(close, 5)", "too similar")

        assert "SELF_CORRELATION" in prompt
        assert "ts_delta(close, 5)" in prompt

    def test_sharpe_optimizer_prompt(self):
        factory = AdaptiveAgentFactory()
        prompt = factory.get_specialist_prompt("sharpe_optimizer", "rank(cap)", "sharpe 0.3")

        assert "LOW_SHARPE" in prompt
        assert "rank(cap)" in prompt

    def test_logic_verifier_prompt_as_default(self):
        factory = AdaptiveAgentFactory()
        prompt = factory.get_specialist_prompt("logic_verifier", "some_expr", "bad logic")

        assert "some_expr" in prompt
        assert "logic" in prompt.lower()


class TestRecordAgentResult:
    def test_tracks_success(self):
        factory = AdaptiveAgentFactory()
        agent = factory.create_specialist_agent("SELF_CORRELATION")

        factory.record_agent_result(agent.agent_id, success=True)

        assert agent.success_count == 1
        assert agent.total_tasks == 1

    def test_tracks_failure(self):
        factory = AdaptiveAgentFactory()
        agent = factory.create_specialist_agent("SELF_CORRELATION")

        factory.record_agent_result(agent.agent_id, success=False)

        assert agent.success_count == 0
        assert agent.total_tasks == 1

    def test_promotes_to_expert_after_three_consecutive_successes(self):
        factory = AdaptiveAgentFactory()
        agent = factory.create_specialist_agent("SELF_CORRELATION")

        for _ in range(3):
            factory.record_agent_result(agent.agent_id, success=True)

        assert agent.is_expert is True

    def test_resets_consecutive_on_failure(self):
        factory = AdaptiveAgentFactory()
        agent = factory.create_specialist_agent("SELF_CORRELATION")

        factory.record_agent_result(agent.agent_id, success=True)
        factory.record_agent_result(agent.agent_id, success=True)
        factory.record_agent_result(agent.agent_id, success=False)
        factory.record_agent_result(agent.agent_id, success=True)

        assert agent.is_expert is False
        assert agent._consecutive_successes == 1

    def test_noop_for_unknown_agent_id(self):
        factory = AdaptiveAgentFactory()
        factory.record_agent_result("nonexistent", success=True)
        assert factory.agent_count == 0


class TestCleanupIdleAgents:
    def test_removes_idle_agents(self):
        factory = AdaptiveAgentFactory()
        agent = factory.create_specialist_agent("SELF_CORRELATION")

        agent.last_used = time.time() - 11 * 60

        removed = factory.cleanup_idle_agents(max_idle_cycles=10)
        assert removed == 1
        assert factory.agent_count == 0

    def test_keeps_active_agents(self):
        factory = AdaptiveAgentFactory()
        factory.create_specialist_agent("SELF_CORRELATION")

        removed = factory.cleanup_idle_agents(max_idle_cycles=10)
        assert removed == 0
        assert factory.agent_count == 1
