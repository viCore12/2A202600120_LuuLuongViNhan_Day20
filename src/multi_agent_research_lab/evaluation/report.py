"""Benchmark report rendering."""

from __future__ import annotations

from typing import Any

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def _fmt_optional(value: Any, fmt: str = "{:.4f}") -> str:
    if value is None:
        return "-"
    try:
        return fmt.format(value)
    except (TypeError, ValueError):
        return str(value)


def render_markdown_report(
    metrics: list[BenchmarkMetrics],
    summaries: list[dict[str, Any]] | None = None,
) -> str:
    """Render benchmark metrics + per-query summaries to markdown."""

    lines: list[str] = ["# Benchmark Report", ""]

    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append("| Run | Latency (s) | Cost (USD) | Quality | Notes |")
    lines.append("|---|---:|---:|---:|---|")
    for item in metrics:
        cost = _fmt_optional(item.estimated_cost_usd, "{:.4f}")
        quality = _fmt_optional(item.quality_score, "{:.1f}")
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | {item.notes} |"
        )

    if summaries:
        lines += ["", "## Per-query comparison", ""]
        # Group summaries by query (each query has 2 entries: baseline + multi-agent).
        by_query: dict[str, list[dict[str, Any]]] = {}
        for s in summaries:
            by_query.setdefault(s["query"], []).append(s)

        for q, runs in by_query.items():
            lines.append(f"### {q}")
            lines.append("")
            header = (
                "| Run | Latency (s) | In tok | Out tok | Cost (USD) | "
                "Sources | Citation cov | Errors | Routes |"
            )
            lines.append(header)
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
            for r in runs:
                cov = (
                    f"{r['citation_coverage']:.0%}"
                    if r.get("citation_coverage") is not None
                    else "-"
                )
                routes = "-".join(r.get("route_history") or []) or "-"
                row = (
                    f"| {r['run']} | {r['latency_s']:.2f} | {r['input_tokens']} | "
                    f"{r['output_tokens']} | {r['cost_usd']:.4f} | {r['sources']} | "
                    f"{cov} | {r['errors']} | {routes} |"
                )
                lines.append(row)
            # Delta if both runs present
            if len(runs) == 2:
                a, b = runs[0], runs[1]
                d_lat = b["latency_s"] - a["latency_s"]
                d_cost = b["cost_usd"] - a["cost_usd"]
                lines.append("")
                lines.append(
                    f"_Delta ({b['run']} - {a['run']}): "
                    f"latency {d_lat:+.2f}s, cost {d_cost:+.4f} USD_"
                )
            lines.append("")
            for r in runs:
                lines.append(f"**{r['run']} preview:**")
                lines.append("")
                lines.append("> " + (r.get("final_answer_preview") or "(empty)").replace("\n", " "))
                lines.append("")

    lines += [
        "## Notes",
        "",
        "- Latency = wall-clock seconds.",
        "- Cost is an estimate based on the OpenAI public price table; treat as relative.",
        "- Citation coverage = distinct sources cited in `final_answer` / total sources fetched.",
        "- Routes show the supervisor decisions for the multi-agent run.",
        "",
    ]
    return "\n".join(lines) + "\n"
