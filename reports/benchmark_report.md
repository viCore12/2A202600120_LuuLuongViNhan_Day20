# Benchmark Report

## Aggregate metrics

| Run | Latency (s) | Cost (USD) | Quality | Notes |
|---|---:|---:|---:|---|
| baseline | 13.64 | 0.0007 | - | tokens=1386+801, sources=5, citation_cov=100% |
| multi-agent | 28.02 | 0.0015 | - | tokens=3006+1699, sources=5, citation_cov=100%, routes=researcher-analyst-writer-critic-done |
| baseline | 13.27 | 0.0008 | - | tokens=1580+879, sources=5, citation_cov=100% |
| multi-agent | 27.32 | 0.0015 | - | tokens=3143+1689, sources=5, citation_cov=100%, routes=researcher-analyst-writer-critic-done |
| baseline | 12.98 | 0.0008 | - | tokens=1519+954, sources=5, citation_cov=100% |
| multi-agent | 26.58 | 0.0015 | - | tokens=3026+1716, sources=5, citation_cov=100%, routes=researcher-analyst-writer-critic-done |

## Per-query comparison

### Research GraphRAG state-of-the-art and write a 500-word summary

| Run | Latency (s) | In tok | Out tok | Cost (USD) | Sources | Citation cov | Errors | Routes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 13.64 | 1386 | 801 | 0.0007 | 5 | 100% | 0 | - |
| multi-agent | 28.02 | 3006 | 1699 | 0.0015 | 5 | 100% | 0 | researcher-analyst-writer-critic-done |

_Delta (multi-agent - baseline): latency +14.38s, cost +0.0008 USD_

**baseline preview:**

> ## Understanding GraphRAG: A State-of-the-Art Approach to Retrieval-Augmented Generation  GraphRAG is an innovative framework developed by Microsoft Research that enhances the traditional Retrieval-Au

**multi-agent preview:**

> ### GraphRAG: A State-of-the-Art Approach to Retrieval Augmented Generation  GraphRAG is an innovative framework designed to enhance the capabilities of language models in Retrieval Augmented Generati

### Compare single-agent and multi-agent workflows for customer support

| Run | Latency (s) | In tok | Out tok | Cost (USD) | Sources | Citation cov | Errors | Routes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 13.27 | 1580 | 879 | 0.0008 | 5 | 100% | 0 | - |
| multi-agent | 27.32 | 3143 | 1689 | 0.0015 | 5 | 100% | 0 | researcher-analyst-writer-critic-done |

_Delta (multi-agent - baseline): latency +14.05s, cost +0.0007 USD_

**baseline preview:**

> ## Comparison of Single-Agent and Multi-Agent Workflows for Customer Support  When designing customer support systems, organizations often face the choice between single-agent and multi-agent workflow

**multi-agent preview:**

> # Comparing Single-Agent and Multi-Agent Workflows for Customer Support  In the realm of customer support, the choice between single-agent and multi-agent workflows can significantly impact efficiency

### Summarize production guardrails for LLM agents

| Run | Latency (s) | In tok | Out tok | Cost (USD) | Sources | Citation cov | Errors | Routes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 12.98 | 1519 | 954 | 0.0008 | 5 | 100% | 0 | - |
| multi-agent | 26.58 | 3026 | 1716 | 0.0015 | 5 | 100% | 0 | researcher-analyst-writer-critic-done |

_Delta (multi-agent - baseline): latency +13.60s, cost +0.0007 USD_

**baseline preview:**

> ### Production Guardrails for LLM Agents  As the deployment of Large Language Models (LLMs) becomes increasingly prevalent in various applications, establishing robust production guardrails is essenti

**multi-agent preview:**

> # Production Guardrails for LLM Agents  In the realm of Large Language Models (LLMs), implementing robust guardrails is essential to ensure safe and effective deployment in production environments. Th

## Notes

- Latency = wall-clock seconds.
- Cost is an estimate based on the OpenAI public price table; treat as relative.
- Citation coverage = distinct sources cited in `final_answer` / total sources fetched.
- Routes show the supervisor decisions for the multi-agent run.

