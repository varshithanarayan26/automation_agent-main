import time
from typing import Any, Dict, Optional, Type, TypeVar
import httpx
from google import genai
from google.genai import types
from google.genai.errors import APIError
from loguru import logger
from pydantic import BaseModel

from app.core.config import settings

# Generic Type variable for Pydantic model response casting
T = TypeVar("T", bound=BaseModel)


def extract_retry_delay(e: APIError) -> Optional[float]:
    """Helper to parse Google API's RetryInfo and return wait time in seconds."""
    try:
        details_list = None
        # e.details might be the json dictionary or a list
        if isinstance(e.details, dict):
            details_list = e.details.get("error", {}).get("details", [])
            if not details_list:
                details_list = e.details.get("details", [])
        elif isinstance(e.details, list):
            details_list = e.details

        if details_list and isinstance(details_list, list):
            for detail in details_list:
                if (
                    isinstance(detail, dict)
                    and detail.get("@type")
                    == "type.googleapis.com/google.rpc.RetryInfo"
                ):
                    delay_str = detail.get("retryDelay") or detail.get(
                        "retry_delay"
                    )
                    if (
                        delay_str
                        and isinstance(delay_str, str)
                        and delay_str.endswith("s")
                    ):
                        return float(delay_str[:-1])
    except Exception as parse_err:
        logger.warning(
            f"Failed to parse retry delay from API error: {parse_err}"
        )
    return None


class GeminiClient:
    """Wrapper client for the Google Gemini API using the google-genai SDK.

    Acts as the single point of contact for all LLM queries, providing retry policies,
    error mapping, and structured output parsing.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initializes the Gemini GenAI Client.

        Reads from configuration if no key is directly passed.
        """
        # Resolve key from parameters or singleton settings
        resolved_key = api_key or settings.GEMINI_API_KEY

        if not resolved_key or not resolved_key.strip():
            logger.warning(
                "Gemini Client initialized without a GEMINI_API_KEY. Live API calls will fail."
            )
            self._client = None
        else:
            logger.info("Initializing Google GenAI Client.")
            try:
                self._client = genai.Client(api_key=resolved_key)
            except Exception as e:
                logger.error(f"Failed to initialize google-genai client: {e}")
                self._client = None

    def _get_client(self) -> genai.Client:
        """Returns the active genai.Client instance, raising ValueError if unconfigured."""
        if not self._client:
            raise ValueError(
                "Gemini API Client is not configured. "
                "Please configure GEMINI_API_KEY in your environment or .env file."
            )
        return self._client

    def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """Generates raw text content from the Gemini LLM.

        Retries on HTTP 429 (Resource Exhausted) errors, respecting RetryInfo or fallback backoff.

        Args:
            prompt: The string content prompt.
            model: Optional model string override (e.g. 'gemini-2.5-flash').
            system_instruction: Optional developer guidelines set for the model.
            temperature: Creativity control parameter (0.0 to 2.0).

        Returns:
            The raw text response from the model.
        """
        client = self._get_client()
        model_name = model or settings.DEFAULT_MODEL
        logger.debug(
            f"Sending text generation request to model: {model_name}."
        )

        max_attempts = 5
        attempt = 0
        backoff = 2.0

        while True:
            try:
                # Build configuration parameters
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                )

                # Call SDK generate_content method
                response = client.models.generate_content(
                    model=model_name, contents=prompt, config=config
                )

                if not response.text:
                    logger.warning(
                        f"Received empty response from Gemini using model: {model_name}."
                    )
                    return ""

                return response.text

            except APIError as e:
                # 429 Quota Exceeded (Resource Exhausted)
                if e.code == 429:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(
                            f"Gemini API 429 Quota Exceeded. Max attempts ({max_attempts}) reached. Raising error."
                        )
                        raise e

                    delay = extract_retry_delay(e)
                    if delay is None:
                        delay = backoff
                        backoff *= 2.0  # Exponential backoff

                    logger.warning(
                        f"Gemini API 429 Quota Exceeded (Resource Exhausted). "
                        f"Retrying task (attempt {attempt}/{max_attempts}) after waiting {delay}s..."
                    )
                    time.sleep(delay)
                    continue

                logger.error(
                    f"Gemini APIError encountered [Status Code: {e.code}]: {e.message}. Detailed response: {getattr(e, 'details', 'None')}"
                )
                # Differentiate auth error from generic errors
                if e.code == 401:
                    raise ValueError(
                        "Authentication failed (HTTP 401 Unauthorized): Google Cloud API Gateway rejected the API key "
                        "as unauthenticated and expected an OAuth 2.0 access token (ACCESS_TOKEN_TYPE_UNSUPPORTED). "
                        "Please verify that your GEMINI_API_KEY is active and valid in Google AI Studio."
                    ) from e
                if e.code == 403 or "API_KEY_INVALID" in str(e.message):
                    raise ValueError(
                        "Invalid Google Gemini API key provided."
                    ) from e
                raise e
            except Exception as e:
                logger.error(f"Unexpected error in Gemini generate_text: {e}")
                raise e

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
    ) -> T:
        """Generates JSON structured content validated against a Pydantic schema.

        Retries on HTTP 429 (Resource Exhausted) errors, respecting RetryInfo or fallback backoff.

        Args:
            prompt: The text prompt describing what to extract/generate.
            response_schema: The Pydantic BaseModel class to validate against.
            model: Optional model string override.
            system_instruction: Optional system persona instruction.
            temperature: Creativity control parameter (defaults to low temperature for extraction).

        Returns:
            An instance of the response_schema BaseModel.
        """
        client = self._get_client()
        model_name = model or settings.DEFAULT_MODEL
        logger.debug(
            f"Sending structured content request to model: {model_name}."
        )

        max_attempts = 5
        attempt = 0
        backoff = 2.0

        while True:
            try:
                # Configure structured output with Pydantic schema integration
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                )

                response = client.models.generate_content(
                    model=model_name, contents=prompt, config=config
                )

                # Check if google-genai successfully auto-parsed the response into the schema
                if response.parsed is not None:
                    return response.parsed

                text_output = response.text
                if not text_output:
                    raise ValueError(
                        "Received empty structured response text from Gemini."
                    )

                # Fallback parsing and validation in case auto-parsing was bypassed
                return response_schema.model_validate_json(text_output)

            except APIError as e:
                # 429 Quota Exceeded (Resource Exhausted)
                if e.code == 429:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(
                            f"Gemini API 429 Quota Exceeded. Max attempts ({max_attempts}) reached. Raising error."
                        )
                        raise e

                    delay = extract_retry_delay(e)
                    if delay is None:
                        delay = backoff
                        backoff *= 2.0  # Exponential backoff

                    logger.warning(
                        f"Gemini API 429 Quota Exceeded (Resource Exhausted). "
                        f"Retrying task (attempt {attempt}/{max_attempts}) after waiting {delay}s..."
                    )
                    time.sleep(delay)
                    continue

                logger.error(
                    f"Gemini APIError in structured output [Status Code: {e.code}]: {e.message}. Detailed response: {getattr(e, 'details', 'None')}"
                )
                if e.code == 401:
                    raise ValueError(
                        "Authentication failed (HTTP 401 Unauthorized): Google Cloud API Gateway rejected the API key "
                        "as unauthenticated and expected an OAuth 2.0 access token (ACCESS_TOKEN_TYPE_UNSUPPORTED). "
                        "Please verify that your GEMINI_API_KEY is active and valid in Google AI Studio."
                    ) from e
                if e.code == 403 or "API_KEY_INVALID" in str(e.message):
                    raise ValueError(
                        "Invalid Google Gemini API key provided."
                    ) from e
                raise e
            except Exception as e:
                logger.error(
                    f"Unexpected error in Gemini generate_structured: {e}"
                )
                raise e
