"""Analyst agent: turns research notes into structured analysis."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a critical analyst. Given research notes with [n] citations, produce a "
    "structured analysis. Sections: 1) Key claims, 2) Agreements, 3) Disagreements / "
    "open questions, 4) Weak evidence or gaps. Preserve [n] citations from the input."
)


class AnalystAgent(BaseAgent):
    """Reads research_notes and produces analysis_notes."""

    name = "analyst"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient(temperature=0.1)

    def run(self, state: ResearchState) -> ResearchState:
        if not state.research_notes:
            msg = "analyst requires research_notes; none present"
            state.errors.append(msg)
            raise AgentExecutionError(msg)

        with trace_span("agent.analyst", {}) as span:
            user_prompt = (
                f"Query: {state.request.query}\n"
                f"Audience: {state.request.audience}\n\n"
                f"Research notes:\n{state.research_notes}\n\n"
                "Produce the structured analysis. Be concise."
            )
            response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
            state.analysis_notes = response.content
            metadata = {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
                "model": response.model,
            }
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.ANALYST,
                    content=response.content,
                    metadata=metadata,
                )
            )
            span["attributes"].update(metadata)
            state.add_trace_event("analyst", metadata)
        return state
