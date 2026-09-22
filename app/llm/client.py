"""
LLM Client — OpenAI-compatible API wrapper.

Supports any OpenAI-compatible endpoint (Databricks, OpenAI, Anthropic via proxy,
local Ollama, etc.) by configuring base_url and api_key.
"""

import time
import structlog
from openai import OpenAI

from app.config import get_settings

logger = structlog.get_logger(__name__)


class LLMClient:
    """
    Wrapper around an OpenAI-compatible chat completions API.

    Usage:
        client = LLMClient()
        response = client.complete(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.1,
        )
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
    ):
        settings = get_settings()
        self.base_url = base_url or settings.llm_base_url
        self.api_key = api_key or settings.llm_api_key
        self.model_name = model_name or settings.llm_model_name
        self.default_temperature = settings.llm_temperature
        self.default_max_tokens = settings.llm_max_tokens

        if not self.base_url:
            raise ValueError(
                "LLM_BASE_URL is not configured. Set it in .env or pass it directly."
            )
        if not self.api_key:
            raise ValueError(
                "LLM_API_KEY is not configured. Set it in .env or pass it directly."
            )

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

        logger.info(
            "llm_client_initialized",
            base_url=self.base_url,
            model=self.model_name,
        )

    def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        """
        Send a chat completion request and return the response text.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in the response.
            json_mode: If True, request JSON response format.

        Returns:
            The assistant's response text.

        Raises:
            Exception: On API errors after logging.
        """
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        start_time = time.time()

        try:
            response = self._client.chat.completions.create(**kwargs)
            elapsed = time.time() - start_time

            result = response.choices[0].message.content or ""

            # Log usage statistics
            usage = response.usage
            logger.info(
                "llm_completion",
                model=self.model_name,
                elapsed_s=round(elapsed, 2),
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                total_tokens=usage.total_tokens if usage else None,
                json_mode=json_mode,
            )

            return result

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                "llm_completion_failed",
                model=self.model_name,
                elapsed_s=round(elapsed, 2),
                error=str(e),
            )
            raise

    def complete_json(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Convenience method: complete with JSON mode enabled.

        Returns:
            Raw JSON string from the LLM (caller must parse/validate).
        """
        return self.complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
