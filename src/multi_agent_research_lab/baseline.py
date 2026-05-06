"""Single-agent baseline runner.

The baseline performs one search and one LLM call: research, analyze, and write
all in one prompt. This isolates the *orchestration* variable when benchmarking
against the multi-agent workflow.
"""

from __future__ import annotations

import logging

from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    ResearchQuery,
)
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

_BASELINE_SYSTEM = (
    "You are a single research assistant. Given a query and search snippets, you must "
    "(1) extract grounded facts, (2) analyze them, and (3) write a clear final answer "
    "in one pass. Cite each fact with [n] referencing the source index. End with a "
    "'Sources' list mapping [n] -> URL."
)


def run_baseline(
    query: str,
    max_sources: int = 5,
    audience: str = "technical learners",
) -> ResearchState:
    """Execute the single-agent baseline and return the final state."""

    state = ResearchState(
        request=ResearchQuery(query=query, max_sources=max_sources, audience=audience)
    )

    with trace_span("baseline.single_agent", {"query": query}):
        try:
            search = SearchClient()
            sources = search.search(query, max_results=max_sources)
        except Exception as exc:
            logger.warning("baseline.search failed: %s", exc)
            state.errors.append(f"baseline.search failed: {exc}")
            sources = []
        state.sources = sources

        snippets = "\n".join(
            f"[{i + 1}] {doc.title} ({doc.url}): {doc.snippet}"
            for i, doc in enumerate(sources)
        ) or "(no sources)"

        user_prompt = (
            f"Query: {query}\n"
            f"Audience: {audience}\n"
            f"Sources:\n{snippets}\n\n"
            "Write the final answer (markdown, ~400-600 words). End with a 'Sources' list."
        )
        try:
            llm = LLMClient(temperature=0.3)
            response = llm.complete(_BASELINE_SYSTEM, user_prompt)
        except Exception as exc:
            raise AgentExecutionError(f"baseline LLM call failed: {exc}") from exc

        state.final_answer = response.content
        metadata = {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
            "model": response.model,
            "sources": len(sources),
        }
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata=metadata,
            )
        )
        state.add_trace_event("baseline", metadata)
    return state
