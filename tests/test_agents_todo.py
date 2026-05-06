"""Supervisor routing tests."""

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def _make_state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))


def test_supervisor_routes_to_researcher_first() -> None:
    supervisor = SupervisorAgent()
    state = _make_state()
    assert supervisor.decide(state) == "researcher"


def test_supervisor_routes_to_analyst_after_research() -> None:
    supervisor = SupervisorAgent()
    state = _make_state()
    state.sources = [SourceDocument(title="t", url="http://x", snippet="s")]
    state.research_notes = "notes"
    assert supervisor.decide(state) == "analyst"


def test_supervisor_routes_to_writer_after_analysis() -> None:
    supervisor = SupervisorAgent()
    state = _make_state()
    state.sources = [SourceDocument(title="t", url="http://x", snippet="s")]
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    assert supervisor.decide(state) == "writer"


def test_supervisor_routes_to_critic_when_final_answer_present() -> None:
    supervisor = SupervisorAgent()
    state = _make_state()
    state.sources = [SourceDocument(title="t", url="http://x", snippet="s")]
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    state.final_answer = "answer"
    assert supervisor.decide(state) == "critic"


def test_supervisor_done_after_critic_ran() -> None:
    supervisor = SupervisorAgent()
    state = _make_state()
    state.sources = [SourceDocument(title="t", url="http://x", snippet="s")]
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    state.final_answer = "answer"
    state.critic_attempts = 1
    assert supervisor.decide(state) == "done"


def test_supervisor_enforces_max_iterations() -> None:
    supervisor = SupervisorAgent(max_iterations=2)
    state = _make_state()
    state.iteration = 2
    assert supervisor.decide(state) == "done"
