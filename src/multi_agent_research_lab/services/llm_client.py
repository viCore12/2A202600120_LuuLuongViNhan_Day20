"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError

try:  # Optional LangSmith tracing wrapper. Falls back to identity if unavailable.
    from langsmith.wrappers import wrap_openai
except Exception:  # pragma: no cover - optional dep
    def wrap_openai(client):  # type: ignore[no-redef]
        return client

logger = logging.getLogger(__name__)


# Approximate USD per 1K tokens. Used for benchmark cost estimates only.
_PRICE_TABLE: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.00060),
    "gpt-4o": (0.0025, 0.0100),
    "gpt-4.1-mini": (0.00040, 0.00160),
    "gpt-4.1": (0.0020, 0.0080),
}


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    model: str | None = None


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    price = _PRICE_TABLE.get(model)
    if price is None:
        return None
    in_price, out_price = price
    return (input_tokens / 1000.0) * in_price + (output_tokens / 1000.0) * out_price


class LLMClient:
    """Provider-agnostic LLM client backed by OpenAI."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.2,
        timeout: float | None = None,
    ) -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise AgentExecutionError("OPENAI_API_KEY is not configured")
        self._client = wrap_openai(OpenAI(api_key=settings.openai_api_key))
        self.model = model or settings.openai_model
        self.temperature = temperature
        self.timeout = timeout if timeout is not None else float(settings.timeout_seconds)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
    )
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion. Retries up to 3 times with exponential backoff."""

        logger.debug("LLM call model=%s temp=%s", self.model, self.temperature)
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            timeout=self.timeout,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        choice = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        output_tokens = getattr(usage, "completion_tokens", None) if usage else None
        cost = (
            _estimate_cost(self.model, input_tokens, output_tokens)
            if input_tokens is not None and output_tokens is not None
            else None
        )
        return LLMResponse(
            content=choice.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            model=self.model,
        )
