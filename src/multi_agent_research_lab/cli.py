"""Command-line entrypoint for the lab.

``load_dotenv`` runs before importing ``langgraph`` / ``langsmith`` so their
auto-instrumentation can read ``LANGSMITH_TRACING`` and ``LANGSMITH_API_KEY``.
``# noqa: E402`` is intentional here.
"""
# ruff: noqa: E402

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv

load_dotenv()

import typer
import yaml
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.baseline import run_baseline
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    run_benchmark,
    summarize_state,
)
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _run_multi_agent(query: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    return workflow.run(state)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the single-agent baseline."""

    _init()
    state = run_baseline(query)
    console.print(Panel.fit(state.final_answer or "(no answer)", title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""

    _init()
    state = _run_multi_agent(query)
    console.print(
        Panel.fit(state.final_answer or "(no answer)", title="Multi-Agent Final Answer")
    )
    console.print(f"[dim]route_history: {state.route_history}[/dim]")


@app.command()
def benchmark(
    config: Annotated[
        Path, typer.Option("--config", "-c", help="YAML benchmark config")
    ] = Path("configs/lab_default.yaml"),
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Markdown report path")
    ] = Path("reports/benchmark_report.md"),
) -> None:
    """Run baseline + multi-agent on a list of queries and write a report."""

    _init()
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    queries: list[str] = cfg["benchmark"]["queries"]

    store = LocalArtifactStore(root=output.parent)
    metrics_all = []
    summaries: list[dict] = []

    for i, query in enumerate(queries):
        console.rule(f"Query {i + 1}/{len(queries)}: {query}")

        console.print("[cyan]Running baseline...[/cyan]")
        state_b, metrics_b = run_benchmark("baseline", query, run_baseline)
        metrics_all.append(metrics_b)
        summaries.append({"query": query, **summarize_state("baseline", state_b, metrics_b)})

        console.print("[cyan]Running multi-agent...[/cyan]")
        state_m, metrics_m = run_benchmark("multi-agent", query, _run_multi_agent)
        metrics_all.append(metrics_m)
        summaries.append({"query": query, **summarize_state("multi-agent", state_m, metrics_m)})

        store.write_text(
            f"trace_q{i + 1}_baseline.json",
            json.dumps(state_b.model_dump(mode="json"), indent=2, ensure_ascii=False),
        )
        store.write_text(
            f"trace_q{i + 1}_multi.json",
            json.dumps(state_m.model_dump(mode="json"), indent=2, ensure_ascii=False),
        )

    report = render_markdown_report(metrics_all, summaries=summaries)
    path = store.write_text(output.name, report)
    console.print(f"[green]Wrote report:[/green] {path}")


if __name__ == "__main__":
    app()
