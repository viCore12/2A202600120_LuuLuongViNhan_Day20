"""Search client abstraction for ResearcherAgent."""

from __future__ import annotations

import logging

from tavily import TavilyClient
from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Tavily-backed search client."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.tavily_api_key:
            raise AgentExecutionError("TAVILY_API_KEY is not configured")
        self._client = TavilyClient(api_key=settings.tavily_api_key)

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Return up to ``max_results`` source documents for ``query``."""

        logger.debug("Tavily search query=%r max_results=%d", query, max_results)
        response = self._client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=False,
        )
        results = response.get("results", []) if isinstance(response, dict) else []
        documents: list[SourceDocument] = []
        for item in results[:max_results]:
            documents.append(
                SourceDocument(
                    title=item.get("title") or item.get("url") or "Untitled",
                    url=item.get("url"),
                    snippet=(item.get("content") or "").strip(),
                    metadata={"score": item.get("score")},
                )
            )
        return documents
