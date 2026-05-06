"""Researcher agent: collects sources and writes structured notes."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a meticulous research assistant. Given a user query and a numbered list of "
    "search snippets, produce concise research notes. Keep facts grounded in the snippets, "
    "cite each fact with [n] referencing the source index, and flag any gap explicitly."
)


class ResearcherAgent(BaseAgent):
    """Collects sources via SearchClient and produces grounded notes via LLM."""

    name = "researcher"

    def __init__(
        self,
        llm: LLMClient | None = None,
        search: SearchClient | None = None,
    ) -> None:
        self._llm = llm or LLMClient(temperature=0.2)
        self._search = search or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("agent.researcher", {"query": state.request.query}) as span:
            try:
                sources = self._search.search(
                    state.request.query, max_results=state.request.max_sources
                )
            except Exception as exc:
                msg = f"researcher.search failed: {exc}"
                state.errors.append(msg)
                logger.warning(msg)
                raise AgentExecutionError(msg) from exc

            state.sources = sources
            if not sources:
                state.research_notes = "No sources returned from search."
                state.agent_results.append(
                    AgentResult(
                        agent=AgentName.RESEARCHER,
                        content=state.research_notes,
                        metadata={"sources": 0},
                    )
                )
                state.add_trace_event("researcher", {"sources": 0})
                return state

            snippets = "\n".join(
                f"[{i + 1}] {doc.title} ({doc.url}): {doc.snippet}"
                for i, doc in enumerate(sources)
            )
            user_prompt = (
                f"Query: {state.request.query}\n"
                f"Audience: {state.request.audience}\n"
                f"Sources:\n{snippets}\n\n"
                "Write 6-10 bullet points of grounded notes. Each bullet must end with [n]."
            )
            response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
            state.research_notes = response.content
            metadata = {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
                "model": response.model,
                "sources": len(sources),
            }
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content=response.content,
                    metadata=metadata,
                )
            )
            span["attributes"].update(metadata)
            state.add_trace_event("researcher", metadata)
        return state
