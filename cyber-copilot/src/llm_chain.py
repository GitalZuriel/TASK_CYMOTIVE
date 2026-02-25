"""
LLM processing with LangChain supporting multiple models.

Supports GPT-4o, GPT-4o-mini, and Claude Sonnet with latency/token tracking.
Includes exponential-backoff retry for transient API errors.
"""

import json
import time
import logging
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError
from langchain_core.messages import SystemMessage, HumanMessage

import config

logger = logging.getLogger(__name__)

# Transient exceptions worth retrying (import errors are non-fatal —
# the retry loop simply won't match them and will re-raise immediately).
_RETRYABLE_EXCEPTIONS: list[type[Exception]] = []
try:
    from openai import RateLimitError as OpenAIRateLimit, APITimeoutError as OpenAITimeout, APIConnectionError as OpenAIConn
    _RETRYABLE_EXCEPTIONS.extend([OpenAIRateLimit, OpenAITimeout, OpenAIConn])
except ImportError:
    pass
try:
    from anthropic import RateLimitError as AnthropicRateLimit, APITimeoutError as AnthropicTimeout, APIConnectionError as AnthropicConn
    _RETRYABLE_EXCEPTIONS.extend([AnthropicRateLimit, AnthropicTimeout, AnthropicConn])
except ImportError:
    pass

_RETRYABLE_EXCEPTIONS_TUPLE = tuple(_RETRYABLE_EXCEPTIONS)

# Retry settings
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds; doubles each attempt

# Supported model identifiers
SUPPORTED_MODELS = {
    "gpt-4o": {"provider": "openai", "model_id": "gpt-4o"},
    "gpt-4o-mini": {"provider": "openai", "model_id": "gpt-4o-mini"},
    "claude-sonnet": {"provider": "anthropic", "model_id": "claude-sonnet-4-20250514"},
}


@dataclass
class LLMResponse:
    """Structured response from the LLM chain."""
    content: str
    model: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float


def _create_llm(model_name: str):
    """
    Create a LangChain chat model instance.

    Args:
        model_name: One of "gpt-4o", "gpt-4o-mini", "claude-sonnet".

    Returns:
        A LangChain BaseChatModel instance.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Supported: {list(SUPPORTED_MODELS.keys())}"
        )

    spec = SUPPORTED_MODELS[model_name]

    if spec["provider"] == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=spec["model_id"],
            temperature=0.1,
            api_key=config.OPENAI_API_KEY,
        )
    elif spec["provider"] == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=spec["model_id"],
            temperature=0.1,
            api_key=config.ANTHROPIC_API_KEY,
        )


def _estimate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD based on model pricing."""
    model_key = SUPPORTED_MODELS.get(model_name, {}).get("model_id", model_name)
    costs = config.MODEL_COSTS.get(model_key, {"input": 0, "output": 0})
    return (input_tokens / 1000 * costs["input"]) + (output_tokens / 1000 * costs["output"])


def _count_tokens_from_response(response) -> tuple[int, int]:
    """Extract token counts from LangChain response metadata."""
    usage = getattr(response, "usage_metadata", None)
    if usage:
        return (
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )

    # Fallback: try response_metadata
    meta = getattr(response, "response_metadata", {})
    token_usage = meta.get("token_usage", {})
    if token_usage:
        return (
            token_usage.get("prompt_tokens", 0),
            token_usage.get("completion_tokens", 0),
        )

    logger.warning("Could not extract token usage from LLM response — cost/token stats will be zero")
    return (0, 0)


class LLMChain:
    """LangChain-based LLM processor with multi-model support and usage tracking."""

    def __init__(self, model_name: str | None = None) -> None:
        """
        Initialize the LLM chain.

        Args:
            model_name: Model identifier. Defaults to config.DEFAULT_LLM.
        """
        self._model_name = model_name or config.DEFAULT_LLM
        self._llm = _create_llm(self._model_name)
        logger.info("LLM chain initialized with model: %s", self._model_name)

    def _invoke_with_retry(self, messages: list):
        """Call LLM with exponential-backoff retry for transient API errors."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self._llm.invoke(messages)
            except _RETRYABLE_EXCEPTIONS_TUPLE as exc:
                if attempt == MAX_RETRIES:
                    logger.error(
                        "LLM call failed after %d attempts: %s", MAX_RETRIES, exc,
                    )
                    raise
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Transient LLM error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt, MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)
        # Unreachable, but keeps type checkers happy
        raise RuntimeError("Retry loop exited unexpectedly")

    def invoke(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """
        Send a system + user message to the LLM and return a structured response.

        Args:
            system_prompt: The system-level instruction.
            user_prompt: The user-level content (context + query).

        Returns:
            LLMResponse with content, token counts, latency, and cost estimate.
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        start = time.perf_counter()
        response = self._invoke_with_retry(messages)
        latency = time.perf_counter() - start

        input_tokens, output_tokens = _count_tokens_from_response(response)
        total_tokens = input_tokens + output_tokens
        cost = _estimate_cost(self._model_name, input_tokens, output_tokens)

        result = LLMResponse(
            content=response.content,
            model=self._model_name,
            latency_seconds=round(latency, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=round(cost, 6),
        )

        logger.info(
            "LLM response: model=%s latency=%.2fs tokens=%d cost=$%.4f",
            self._model_name,
            result.latency_seconds,
            result.total_tokens,
            result.estimated_cost_usd,
        )
        return result

    def invoke_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        max_retries: int = 1,
    ) -> tuple[BaseModel | None, LLMResponse, bool]:
        """
        Invoke LLM and parse + validate the response as a Pydantic model.

        Returns:
            (parsed_model or None, raw LLMResponse, is_structured).
            Falls back to None + raw text if parsing fails after retries.
        """
        last_response = None
        last_error = None

        for attempt in range(1 + max_retries):
            if attempt == 0:
                resp = self.invoke(system_prompt, user_prompt)
            else:
                retry_prompt = (
                    f"{user_prompt}\n\n"
                    f"## PREVIOUS ATTEMPT FAILED\n"
                    f"Your previous response was not valid JSON. Error: {last_error}\n"
                    f"Please respond with ONLY a valid JSON object matching the schema. "
                    f"No markdown fences, no commentary."
                )
                resp = self.invoke(system_prompt, retry_prompt)

            last_response = resp
            raw = resp.content.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                first_nl = raw.index("\n") if "\n" in raw else len(raw)
                raw = raw[first_nl + 1:]
                if raw.endswith("```"):
                    raw = raw[:-3].strip()

            try:
                data = json.loads(raw)
                parsed = response_model.model_validate(data)
                logger.info(
                    "Structured output parsed successfully (attempt %d)", attempt + 1,
                )
                return (parsed, resp, True)
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = str(e)[:300]
                logger.warning(
                    "Structured output parse failed (attempt %d/%d): %s",
                    attempt + 1, 1 + max_retries, last_error,
                )

        logger.warning("Structured output failed after %d attempts, falling back to raw text", 1 + max_retries)
        return (None, last_response, False)

    @property
    def model_name(self) -> str:
        return self._model_name
