"""
LLM Retry & JSON Parsing Utilities.

Handles:
- Extracting JSON from LLM responses (even when wrapped in markdown code blocks)
- Retrying with correction prompts on parse failures
- Pydantic validation of parsed JSON
"""

import json
import re
from typing import TypeVar, Type

import structlog
from pydantic import BaseModel, ValidationError

from app.llm.client import LLMClient

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


def extract_json_from_text(text: str) -> str:
    """
    Extract JSON from LLM response text.

    Handles cases where the LLM wraps JSON in markdown code blocks
    or includes preamble text before/after the JSON.

    Args:
        text: Raw LLM response text.

    Returns:
        Cleaned JSON string.

    Raises:
        ValueError: If no JSON object/array can be found.
    """
    # Try direct parse first
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return stripped

    # Try extracting from markdown code blocks
    # Match ```json ... ``` or ``` ... ```
    code_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    if matches:
        for match in matches:
            match = match.strip()
            if match.startswith("{") or match.startswith("["):
                return match

    # Try finding JSON object/array with brace matching
    # Find the first { or [ and the last matching } or ]
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start_idx = text.find(start_char)
        if start_idx == -1:
            continue
        end_idx = text.rfind(end_char)
        if end_idx == -1 or end_idx <= start_idx:
            continue
        candidate = text[start_idx : end_idx + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not extract JSON from LLM response: {text[:200]}...")


def parse_json_response(text: str) -> dict | list:
    """
    Parse JSON from LLM response text.

    Args:
        text: Raw LLM response.

    Returns:
        Parsed JSON as dict or list.

    Raises:
        ValueError: If parsing fails.
    """
    json_str = extract_json_from_text(text)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Try fixing common JSON issues
        # Remove trailing commas before } or ]
        fixed = re.sub(r",\s*([}\]])", r"\1", json_str)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse JSON: {e}") from e


def validate_with_model(data: dict | list, model: Type[T]) -> T:
    """
    Validate parsed JSON against a Pydantic model.

    Args:
        data: Parsed JSON dict.
        model: Pydantic model class.

    Returns:
        Validated Pydantic model instance.

    Raises:
        ValidationError: If validation fails.
    """
    return model.model_validate(data)


def complete_and_parse(
    client: LLMClient,
    messages: list[dict],
    response_model: Type[T],
    max_retries: int = 3,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> T:
    """
    Send a completion request, parse JSON, and validate against a Pydantic model.
    Retries with correction prompts on failure.

    Args:
        client: LLM client instance.
        messages: Chat messages.
        response_model: Pydantic model to validate against.
        max_retries: Maximum retry attempts.
        temperature: Sampling temperature.
        max_tokens: Maximum response tokens.

    Returns:
        Validated Pydantic model instance.

    Raises:
        ValueError: If all retries fail.
    """
    last_error = None
    current_messages = list(messages)  # Copy to avoid mutating original

    for attempt in range(1, max_retries + 1):
        logger.info(
            "llm_parse_attempt",
            attempt=attempt,
            max_retries=max_retries,
            model=response_model.__name__,
        )

        try:
            # Request completion (try JSON mode first, fall back to plain)
            try:
                raw_response = client.complete_json(
                    messages=current_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception:
                # Some models don't support JSON mode — fall back
                raw_response = client.complete(
                    messages=current_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=False,
                )

            # Parse JSON
            parsed = parse_json_response(raw_response)

            # Validate with Pydantic
            result = validate_with_model(parsed, response_model)

            logger.info(
                "llm_parse_success",
                attempt=attempt,
                model=response_model.__name__,
            )
            return result

        except (ValueError, ValidationError, json.JSONDecodeError) as e:
            last_error = e
            logger.warning(
                "llm_parse_failed",
                attempt=attempt,
                error=str(e),
                model=response_model.__name__,
            )

            if attempt < max_retries:
                # Build correction prompt
                correction_message = {
                    "role": "user",
                    "content": (
                        f"Your previous response could not be parsed correctly.\n"
                        f"Error: {str(e)}\n\n"
                        f"Please respond with ONLY valid JSON matching the required schema. "
                        f"Do not include any text before or after the JSON.\n"
                        f"Do not wrap the JSON in code blocks.\n"
                        f"Required schema: {response_model.model_json_schema()}"
                    ),
                }

                # Add the failed response and correction to messages
                current_messages.append(
                    {"role": "assistant", "content": raw_response}
                )
                current_messages.append(correction_message)

    raise ValueError(
        f"Failed to get valid response after {max_retries} attempts. "
        f"Last error: {last_error}"
    )
