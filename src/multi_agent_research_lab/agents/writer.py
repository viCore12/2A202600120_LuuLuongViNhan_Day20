"""Writer agent: synthesizes final answer with citations."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a technical writer. Synthesize a clear, well-structured answer for the "
    "given audience. Use the research notes and analysis as your only ground truth. "
    "Preserve [n] citations and append a 'Sources' list mapping [n] -> URL at the end."
)


class WriterAgent(BaseAgent):
    """Produces final_answer from research_notes and analysis_notes."""

    name = "writer"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient(temperature=0.4)

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("agent.writer", {}) as span:
            sources_block = "\n".join(
                f"[{i + 1}] {doc.title} - {doc.url or 'no-url'}"
                for i, doc in enumerate(state.sources)
            ) or "(no sources)"

            critic_block = (
                f"\nReviewer feedback (must address):\n{state.critic_feedback}\n"
                if state.critic_feedback
                else ""
            )
            user_prompt = (
                f"Query: {state.request.query}\n"
                f"Audience: {state.request.audience}\n\n"
                f"Research notes:\n{state.research_notes or '(none)'}\n\n"
                f"Analysis:\n{state.analysis_notes or '(none)'}\n\n"
                f"Source index:\n{sources_block}\n"
                f"{critic_block}\n"
                "Write the final answer (markdown, ~400-600 words). End with a 'Sources' list. "
                "Cite EVERY source from the index at least once using [n]."
            )
            response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
            state.final_answer = response.content
            metadata = {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
                "model": response.model,
            }
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
                    content=response.content,
                    metadata=metadata,
                )
            )
            span["attributes"].update(metadata)
            state.add_trace_event("writer", metadata)
        return state
