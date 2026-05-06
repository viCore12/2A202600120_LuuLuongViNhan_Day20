"""CriticAgent tests."""

from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def _state_with_sources(answer: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query="bonus critic test"))
    state.sources = [
        SourceDocument(title=f"s{i}", url=f"http://x/{i}", snippet="x")
        for i in range(3)
    ]
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    state.final_answer = answer
    return state


def test_critic_approves_full_coverage() -> None:
    state = _state_with_sources("Cite [1], [2], and [3].")
    out = CriticAgent().run(state)
    assert out.final_answer is not None
    assert out.critic_attempts == 1
    assert "Approved" in (out.critic_feedback or "")


def test_critic_clears_answer_on_missing_citation() -> None:
    state = _state_with_sources("Only [1] cited.")
    out = CriticAgent().run(state)
    assert out.final_answer is None
    assert out.critic_attempts == 1
    assert "Missing" in (out.critic_feedback or "")


def test_critic_stops_after_max_retries() -> None:
    state = _state_with_sources("Only [1] cited.")
    state.critic_attempts = 1  # already retried once
    out = CriticAgent(max_retries=1).run(state)
    assert out.final_answer is not None  # not cleared
    assert out.critic_attempts == 2
    assert "Stopped retrying" in (out.critic_feedback or "")


def test_supervisor_routes_to_critic_after_writer() -> None:
    supervisor = SupervisorAgent()
    state = _state_with_sources("answer with [1]")
    assert supervisor.decide(state) == "critic"


def test_supervisor_done_after_critic_attempts() -> None:
    supervisor = SupervisorAgent()
    state = _state_with_sources("answer with [1]")
    state.critic_attempts = 1
    assert supervisor.decide(state) == "done"
