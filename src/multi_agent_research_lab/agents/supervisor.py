"""Supervisor / router agent.

Uses a deterministic routing policy: cheaper, more predictable, and easier to
benchmark than an LLM router for this lab. The next route is appended to
``state.route_history`` via :meth:`ResearchState.record_route`.
"""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, max_iterations: int | None = None) -> None:
        settings = get_settings()
        self._max_iterations = max_iterations or settings.max_iterations

    def decide(self, state: ResearchState) -> str:
        """Pure routing decision. Returns one of: researcher, analyst, writer, critic, done."""

        if state.iteration >= self._max_iterations:
            return "done"

        # Fallback: too many consecutive errors → stop and let writer salvage if possible.
        if len(state.errors) >= 3:
            return "writer" if state.final_answer is None else "done"

        if not state.sources and state.research_notes is None:
            return "researcher"
        if state.analysis_notes is None:
            return "analyst"
        if state.final_answer is None:
            return "writer"
        # Final answer present: run critic once before finishing.
        if state.critic_attempts == 0:
            return "critic"
        return "done"

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("agent.supervisor", {}) as span:
            route = self.decide(state)
            state.record_route(route)
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.SUPERVISOR,
                    content=route,
                    metadata={"iteration": state.iteration},
                )
            )
            span["attributes"]["route"] = route
            state.add_trace_event("supervisor", {"route": route, "iteration": state.iteration})
            logger.debug("supervisor route=%s iteration=%d", route, state.iteration)
        return state
