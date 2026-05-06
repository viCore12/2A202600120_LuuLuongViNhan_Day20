"""Critic agent: enforces citation coverage on the writer's output.

The critic computes citation coverage on ``state.final_answer``. If coverage is
below ``threshold`` and we have not yet retried, it clears ``final_answer`` and
records actionable feedback so the supervisor routes back to the writer with
hints. Otherwise it leaves the answer in place and lets the supervisor finish.
A bounded ``critic_attempts`` counter guarantees the loop terminates.
"""

from __future__ import annotations

import logging
import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")


def _citation_coverage(state: ResearchState) -> tuple[float, set[int], set[int]]:
    """Return (coverage, cited_indices, missing_indices)."""

    if not state.final_answer or not state.sources:
        return 0.0, set(), set()
    n = len(state.sources)
    valid = set(range(1, n + 1))
    cited = {int(m) for m in _CITATION_RE.findall(state.final_answer) if 1 <= int(m) <= n}
    missing = valid - cited
    return len(cited) / n, cited, missing


class CriticAgent(BaseAgent):
    """Citation-coverage gate. Retries the writer at most ``max_retries`` times."""

    name = "critic"

    def __init__(self, threshold: float = 1.0, max_retries: int = 1) -> None:
        self.threshold = threshold
        self.max_retries = max_retries

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("agent.critic", {}) as span:
            coverage, _, missing = _citation_coverage(state)
            state.critic_attempts += 1

            should_retry = (
                coverage < self.threshold
                and state.critic_attempts <= self.max_retries
                and bool(missing)
            )

            if should_retry:
                missing_list = sorted(missing)
                feedback = (
                    f"Citation coverage {coverage:.0%} is below {self.threshold:.0%}. "
                    f"Missing source indices: {missing_list}. "
                    "Rewrite the answer to cite every source at least once."
                )
                state.critic_feedback = feedback
                state.final_answer = None  # forces supervisor → writer again
                outcome = "retry"
            else:
                state.critic_feedback = (
                    f"Approved at coverage {coverage:.0%}."
                    if coverage >= self.threshold
                    else f"Stopped retrying after {state.critic_attempts} attempt(s); "
                    f"coverage {coverage:.0%}."
                )
                outcome = "approved" if coverage >= self.threshold else "stop_max_retries"

            metadata = {
                "coverage": coverage,
                "missing": sorted(missing),
                "attempts": state.critic_attempts,
                "outcome": outcome,
            }
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.CRITIC,
                    content=state.critic_feedback or "",
                    metadata=metadata,
                )
            )
            span["attributes"].update(metadata)
            state.add_trace_event("critic", metadata)
            logger.debug("critic outcome=%s coverage=%.2f", outcome, coverage)
        return state
