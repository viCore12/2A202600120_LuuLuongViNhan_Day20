"""Benchmark utilities for single-agent vs multi-agent."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]

_CITATION_RE = re.compile(r"\[(\d+)\]")


def _aggregate_tokens_cost(state: ResearchState) -> tuple[int, int, float]:
    in_tok = 0
    out_tok = 0
    cost = 0.0
    for r in state.agent_results:
        meta = r.metadata or {}
        in_tok += int(meta.get("input_tokens") or 0)
        out_tok += int(meta.get("output_tokens") or 0)
        cost += float(meta.get("cost_usd") or 0.0)
    return in_tok, out_tok, cost


def _citation_coverage(state: ResearchState) -> float | None:
    """Fraction of distinct citation indices in [1..N] that appear in final_answer."""

    if not state.final_answer or not state.sources:
        return None
    n = len(state.sources)
    cited = {int(m) for m in _CITATION_RE.findall(state.final_answer) if 1 <= int(m) <= n}
    return len(cited) / n if n else None


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, tokens, cost, citation coverage."""

    started = perf_counter()
    failure = False
    try:
        state = runner(query)
    except Exception as exc:
        logger.exception("benchmark runner failed: %s", exc)
        state = ResearchState(request=ResearchQuery(query=query))
        state.errors.append(str(exc))
        failure = True

    latency = perf_counter() - started
    in_tok, out_tok, cost = _aggregate_tokens_cost(state)
    coverage = _citation_coverage(state)
    notes_parts = [f"tokens={in_tok}+{out_tok}", f"sources={len(state.sources)}"]
    if coverage is not None:
        notes_parts.append(f"citation_cov={coverage:.0%}")
    if state.route_history:
        notes_parts.append(f"routes={'-'.join(state.route_history)}")
    if failure or state.errors:
        notes_parts.append(f"errors={len(state.errors)}")

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=cost if cost else None,
        notes=", ".join(notes_parts),
    )
    return state, metrics


def summarize_state(
    run_name: str, state: ResearchState, metrics: BenchmarkMetrics
) -> dict[str, Any]:
    in_tok, out_tok, cost = _aggregate_tokens_cost(state)
    return {
        "run": run_name,
        "latency_s": metrics.latency_seconds,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": cost,
        "sources": len(state.sources),
        "citation_coverage": _citation_coverage(state),
        "errors": len(state.errors),
        "route_history": state.route_history,
        "final_answer_preview": (state.final_answer or "")[:200],
    }
