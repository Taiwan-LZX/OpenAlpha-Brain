from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_EPSILON = 1e-6


@dataclass
class SpecialistAgent:
    agent_id: str = ""
    agent_type: str = ""
    specialty: str = ""
    success_count: int = 0
    total_tasks: int = 0
    is_expert: bool = False
    created_at: float = 0.0
    last_used: float = 0.0
    _consecutive_successes: int = field(default=0, repr=False)
    dynamic_weight: float = field(default=1.0, repr=False)
    mab_signal: float = field(default=0.0, repr=False)
    regime_boost: float = field(default=1.0, repr=False)
    sharpe_trend: float = field(default=0.0, repr=False)

    def __post_init__(self):
        if not self.agent_id:
            self.agent_id = uuid.uuid4().hex[:12]
        if not self.created_at:
            self.created_at = time.time()
        if not self.last_used:
            self.last_used = time.time()


_FAILURE_TYPE_MAP: dict[str, str] = {
    "SELF_CORRELATION": "originality",
    "LOW_SHARPE": "sharpe_optimizer",
}

_ORIGINALITY_PROMPT = (
    "The expression '{expression}' failed due to SELF_CORRELATION — it is too similar to existing alphas.\n"
    "Failure details: {failure_info}\n\n"
    "Your task: Make this expression more unique and original. Strategies:\n"
    "1. Change the neutralization method (e.g., from INDUSTRY to SUBINDUSTRY or MARKET)\n"
    "2. Use different data fields (e.g., replace close with vwap, or add volume-based terms)\n"
    "3. Add interaction terms (e.g., multiply or divide two signals)\n"
    "4. Apply a different time-series transformation (e.g., ts_zscore instead of ts_rank)\n"
    "5. Combine complementary signals from different directions\n\n"
    "Output ONLY the improved FASTEXPR expression — no JSON, no markdown, no explanation."
)

_SHARPE_OPTIMIZER_PROMPT = (
    "The expression '{expression}' failed due to LOW_SHARPE — its risk-adjusted return is insufficient.\n"
    "Failure details: {failure_info}\n\n"
    "Your task: Improve the Sharpe ratio of this expression. Strategies:\n"
    "1. Adjust lookback windows (try 5, 10, 20, 60 instead of current values)\n"
    "2. Add volatility normalization (divide by ts_std_dev or wrap with ts_zscore)\n"
    "3. Combine complementary signals (e.g., momentum + value interaction)\n"
    "4. Add ts_decay_linear for turnover control and signal smoothing\n"
    "5. Use group_neutralize to remove sector noise\n\n"
    "Output ONLY the improved FASTEXPR expression — no JSON, no markdown, no explanation."
)

_LOGIC_VERIFIER_PROMPT = (
    "The expression '{expression}' may not correctly implement the intended market logic.\n"
    "Failure details: {failure_info}\n\n"
    "Your task: Verify and fix the expression so it matches the hypothesis logic. Check:\n"
    "1. Does the operator choice match the stated mechanism? (e.g., momentum → ts_delta/ts_regression)\n"
    "2. Is the direction correct? (e.g., mean reversion should be negative of momentum)\n"
    "3. Are the lookback windows appropriate for the time horizon?\n"
    "4. Is the neutralization removing the right sources of noise?\n"
    "5. Does the expression capture the economic rationale described in the hypothesis?\n\n"
    "Output ONLY the corrected FASTEXPR expression — no JSON, no markdown, no explanation."
)


class AdaptiveAgentFactory:
    def __init__(self, llm_generate_fn=None):
        self._agents: dict[str, SpecialistAgent] = {}
        self._llm_generate_fn = llm_generate_fn

    def create_specialist_agent(self, failure_type: str, direction: str = "") -> SpecialistAgent | None:
        agent_type = _FAILURE_TYPE_MAP.get(failure_type, "logic_verifier")

        matching = [a for a in self._agents.values() if a.agent_type == agent_type]
        if matching:
            for a in matching:
                a.dynamic_weight = self._compute_agent_score(a, direction)
            matching.sort(key=lambda a: a.dynamic_weight, reverse=True)
            best = matching[0]
            if best.is_expert:
                logger.info(
                    "Expert agent %s available for %s (weight=%.3f), reusing instead of creating new",
                    best.agent_id, failure_type, best.dynamic_weight,
                )
            best.last_used = time.time()
            return best

        agent = SpecialistAgent(
            agent_type=agent_type,
            specialty=failure_type,
            created_at=time.time(),
            last_used=time.time(),
        )
        self._agents[agent.agent_id] = agent
        return agent

    def get_specialist_prompt(
        self, agent_type: str, expression: str, failure_info: str, agent: SpecialistAgent | None = None
    ) -> str:
        base_prompt = ""
        if agent_type == "originality":
            base_prompt = _ORIGINALITY_PROMPT.format(expression=expression, failure_info=failure_info)
        elif agent_type == "sharpe_optimizer":
            base_prompt = _SHARPE_OPTIMIZER_PROMPT.format(expression=expression, failure_info=failure_info)
        else:
            base_prompt = _LOGIC_VERIFIER_PROMPT.format(expression=expression, failure_info=failure_info)

        if agent is not None:
            expertise_parts = []
            if agent.is_expert:
                expertise_parts.append("You are an EXPERT in this domain with a proven track record.")
            if agent.success_count > 0:
                expertise_parts.append(f"You have {agent.success_count} successful experiences to draw upon.")
            if expertise_parts:
                base_prompt = " ".join(expertise_parts) + " Apply your deepest insights.\n\n" + base_prompt

        return base_prompt

    def record_agent_result(self, agent_id: str, success: bool) -> None:
        agent = self._agents.get(agent_id)
        if agent is None:
            return
        agent.total_tasks += 1
        if success:
            agent.success_count += 1
            agent._consecutive_successes += 1
            if agent._consecutive_successes >= 3:
                agent.is_expert = True
        else:
            agent._consecutive_successes = 0
        agent.last_used = time.time()

    def cleanup_idle_agents(self, max_idle_cycles: int = 10) -> int:
        now = time.time()
        threshold = max_idle_cycles * 60
        to_remove = [aid for aid, agent in self._agents.items() if (now - agent.last_used) > threshold]
        for aid in to_remove:
            del self._agents[aid]
        return len(to_remove)

    def _compute_agent_score(self, agent: SpecialistAgent, direction: str = "") -> float:
        _base = 0.3
        if agent.is_expert:
            _base += 0.3
        if agent.total_tasks > 0:
            _base += 0.2 * (agent.success_count / max(agent.total_tasks, 1))
        _consec_bonus = min(0.1, agent._consecutive_successes * 0.03)
        _base += _consec_bonus
        _base *= agent.regime_boost
        _base += agent.mab_signal * 0.15
        _base += agent.sharpe_trend * 0.10
        return max(_EPSILON, min(2.0, _base))

    def update_agent_signals(
        self,
        agent_type: str,
        mab_success_rate: float | None = None,
        regime: str = "",
        recent_sharpe_trend: float = 0.0,
    ) -> None:
        _mab_weight = 1.0
        if mab_success_rate is not None and mab_success_rate > 0:
            _mab_weight = 0.5 + min(1.5, mab_success_rate * 2)
        _regime_map = {
            "high_volatility": {"sharpe_optimizer": 1.3, "originality": 1.1, "logic_verifier": 1.0},
            "trending": {"sharpe_optimizer": 1.4, "originality": 0.9, "logic_verifier": 1.1},
            "low_volatility": {"sharpe_optimizer": 1.0, "originality": 1.2, "logic_verifier": 1.1},
            "crash_risk": {"sharpe_optimizer": 0.7, "originality": 1.4, "logic_verifier": 1.2},
        }
        _regime_boost = _regime_map.get(regime, {}).get(agent_type, 1.0)
        for agent in self._agents.values():
            if agent.agent_type == agent_type:
                agent.mab_signal = mab_success_rate or 0.0
                agent.regime_boost = _regime_boost
                agent.sharpe_trend = max(-1.0, min(1.0, recent_sharpe_trend))
        logger.info(
            "[DEFENSIVE_LOG] ADAPTIVE_AGENT::SIGNAL_UPDATE type=%s "
            "mab=%.3f regime=%s boost=%.2f sharpe_trend=%.3f",
            agent_type, mab_success_rate or 0, regime, _regime_boost, recent_sharpe_trend,
        )

    def get_agent(self, agent_id: str) -> SpecialistAgent | None:
        """Look up a specialist agent by its ID.

        Returns None if no agent with the given ID exists. Useful for
        inspecting agent state (success count, expertise status, etc.).
        """
        return self._agents.get(agent_id)

    @property
    def agent_count(self) -> int:
        """Return the number of currently active specialist agents."""
        return len(self._agents)
