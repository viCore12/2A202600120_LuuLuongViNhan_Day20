"""LangGraph multi-agent workflow.

The graph alternates: supervisor -> worker -> supervisor -> ... -> END. Worker
exceptions are recorded into ``state.errors`` so the supervisor can decide to
fall back instead of crashing the run.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


def _safe_run(agent_name: str, agent: Any, state: ResearchState) -> ResearchState:
    try:
        return agent.run(state)
    except AgentExecutionError as exc:
        msg = f"{agent_name} failed: {exc}"
        logger.warning(msg)
        state.errors.append(msg)
        state.add_trace_event(f"{agent_name}.error", {"error": str(exc)})
        return state


class MultiAgentWorkflow:
    """Builds and runs the supervisor-coordinated graph."""

    def __init__(
        self,
        supervisor: SupervisorAgent | None = None,
        researcher: ResearcherAgent | None = None,
        analyst: AnalystAgent | None = None,
        writer: WriterAgent | None = None,
        critic: CriticAgent | None = None,
    ) -> None:
        self.supervisor = supervisor or SupervisorAgent()
        self._researcher = researcher
        self._analyst = analyst
        self._writer = writer
        self._critic = critic or CriticAgent()
        self._compiled: Any | None = None

    def _researcher_node(self, state: ResearchState) -> ResearchState:
        if self._researcher is None:
            self._researcher = ResearcherAgent()
        return _safe_run("researcher", self._researcher, state)

    def _analyst_node(self, state: ResearchState) -> ResearchState:
        if self._analyst is None:
            self._analyst = AnalystAgent()
        return _safe_run("analyst", self._analyst, state)

    def _writer_node(self, state: ResearchState) -> ResearchState:
        if self._writer is None:
            self._writer = WriterAgent()
        return _safe_run("writer", self._writer, state)

    def _critic_node(self, state: ResearchState) -> ResearchState:
        return _safe_run("critic", self._critic, state)

    def _route_from_supervisor(self, state: ResearchState) -> str:
        if not state.route_history:
            return "done"
        last = state.route_history[-1]
        if last in {"researcher", "analyst", "writer", "critic"}:
            return last
        return "done"

    def build(self) -> Any:
        graph = StateGraph(ResearchState)
        graph.add_node("supervisor", self.supervisor.run)
        graph.add_node("researcher", self._researcher_node)
        graph.add_node("analyst", self._analyst_node)
        graph.add_node("writer", self._writer_node)
        graph.add_node("critic", self._critic_node)

        graph.set_entry_point("supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                "done": END,
            },
        )
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")
        graph.add_edge("critic", "supervisor")

        self._compiled = graph.compile()
        return self._compiled

    def run(self, state: ResearchState) -> ResearchState:
        if self._compiled is None:
            self.build()
        assert self._compiled is not None
        # Allow generous step budget; supervisor enforces real stop via max_iterations.
        result = self._compiled.invoke(
            state,
            config={"recursion_limit": 50},
        )
        if isinstance(result, ResearchState):
            return result
        return ResearchState.model_validate(result)
