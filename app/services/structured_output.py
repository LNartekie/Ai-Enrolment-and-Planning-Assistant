import json
import os

import requests
from dotenv import load_dotenv
from pydantic import ValidationError

from app.prompts.forecast_prompts import (
    SYSTEM_PROMPT,
    build_forecast_prompt,
)

from app.schemas.forecast_response import (
    ForecastExplanation,
)

from app.services.structured_validation import (
    validate_forecast_semantics,
)


# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()


OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
).rstrip("/")


OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:1.7b",
)


# --------------------------------------------------
# SEND ONE STRUCTURED REQUEST TO OLLAMA
# --------------------------------------------------

def _request_structured_response(
    user_prompt: str,
    response_schema: dict,
    model_name: str,
) -> ForecastExplanation:
    """
    Send one structured request to Ollama.

    This function performs:
    1. Ollama API communication
    2. JSON extraction
    3. Pydantic structural validation

    Semantic validation is handled separately.
    """

    payload = {
        "model": model_name,

        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],

        "think": False,

        "stream": False,

        "format": response_schema,

        "options": {
            "temperature": 0,
        },
    }


    # --------------------------------------------------
    # CALL OLLAMA
    # --------------------------------------------------

    try:

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=120,
        )

        response.raise_for_status()

    except requests.RequestException as error:

        raise RuntimeError(
            "Structured AI generation failed because "
            "the Ollama service was unavailable: "
            f"{error}"
        ) from error


    # --------------------------------------------------
    # EXTRACT RESPONSE CONTENT
    # --------------------------------------------------

    try:

        response_data = response.json()

        content = (
            response_data[
                "message"
            ][
                "content"
            ]
        )

    except (
        ValueError,
        KeyError,
        TypeError,
    ) as error:

        raise RuntimeError(
            "Ollama returned a malformed API response."
        ) from error


    if not content.strip():

        raise RuntimeError(
            "Ollama returned an empty structured response."
        )


    # --------------------------------------------------
    # PYDANTIC STRUCTURAL VALIDATION
    # --------------------------------------------------

    try:

        validated_response = (
            ForecastExplanation.model_validate_json(
                content
            )
        )

    except ValidationError as error:

        raise RuntimeError(
            "The AI response failed Pydantic "
            "schema validation."
        ) from error


    return validated_response


# --------------------------------------------------
# GENERATE FULLY VALIDATED FORECAST EXPLANATION
# --------------------------------------------------

def generate_structured_forecast_explanation(
    trend_summary: dict,
    forecast_result: dict,
    capacity: float | None = None,
    model_name: str | None = None,
) -> ForecastExplanation:
    """
    Generate a forecast explanation that must pass:

    1. Ollama JSON-schema constraints
    2. Pydantic structural validation
    3. Semantic validation

    If semantic validation fails, one corrective
    retry is allowed.

    If the retry also fails, the response is rejected.
    """

    selected_model = (
        model_name
        if model_name is not None
        else OLLAMA_MODEL
    )


    # --------------------------------------------------
    # 1. BUILD VERIFIED FORECAST PROMPT
    # --------------------------------------------------

    base_prompt = build_forecast_prompt(
        trend_summary=trend_summary,
        forecast_result=forecast_result,
        capacity=capacity,
        variant="C",
    )


    # --------------------------------------------------
    # 2. GENERATE JSON SCHEMA FROM PYDANTIC
    # --------------------------------------------------

    response_schema = (
        ForecastExplanation.model_json_schema()
    )


    # --------------------------------------------------
    # 3. BUILD INITIAL STRUCTURED PROMPT
    # --------------------------------------------------

    initial_prompt = f"""
{base_prompt}

STRUCTURED OUTPUT REQUIREMENT

Return only valid JSON.

The JSON must match the supplied schema exactly.

Do not include Markdown.
Do not include code fences.
Do not include explanatory text outside the JSON.

Do not describe model accuracy or error as:
high,
moderate,
low,
good,
or poor.

Do not say that the model or forecast was validated.

The backtest evidence label refers only to the
amount of historical evaluation available.

Do not reinterpret the meaning of the supplied
backtest evidence label.

Do not calculate new values.

JSON SCHEMA:

{json.dumps(response_schema, indent=2)}
"""


    # --------------------------------------------------
    # 4. FIRST GENERATION ATTEMPT
    # --------------------------------------------------

    first_response = (
        _request_structured_response(
            user_prompt=initial_prompt,
            response_schema=response_schema,
            model_name=selected_model,
        )
    )


    # --------------------------------------------------
    # 5. FIRST SEMANTIC VALIDATION
    # --------------------------------------------------

    first_semantic_errors = (
        validate_forecast_semantics(
            response=first_response,
            forecast_result=forecast_result,
            capacity=capacity,
        )
    )


    # --------------------------------------------------
    # 6. RETURN IMMEDIATELY IF VALID
    # --------------------------------------------------

    if not first_semantic_errors:

        return first_response


    # --------------------------------------------------
    # 7. PREPARE ONE CORRECTIVE RETRY
    # --------------------------------------------------

    correction_details = "\n".join(
        f"- {error}"
        for error in first_semantic_errors
    )


    retry_prompt = f"""
{base_prompt}

STRUCTURED OUTPUT REQUIREMENT

Return only valid JSON matching the supplied
schema exactly.

Do not include Markdown.
Do not include code fences.
Do not include text outside the JSON.

Your previous attempt failed semantic validation.

The detected problems were:

{correction_details}

CORRECTION RULES

Correct all of the problems above.

Do not describe accuracy or error as high,
moderate, low, good, or poor.

Do not claim that the model or forecast
was validated.

Do not calculate new values.

Use only the supplied deterministic facts.

The backtest evidence label describes only
the amount of historical evaluation available.

Forecasts remain planning estimates.

Human review must remain required.

JSON SCHEMA:

{json.dumps(response_schema, indent=2)}
"""


    # --------------------------------------------------
    # 8. SECOND AND FINAL GENERATION ATTEMPT
    # --------------------------------------------------

    retry_response = (
        _request_structured_response(
            user_prompt=retry_prompt,
            response_schema=response_schema,
            model_name=selected_model,
        )
    )


    # --------------------------------------------------
    # 9. VALIDATE RETRY SEMANTICALLY
    # --------------------------------------------------

    retry_semantic_errors = (
        validate_forecast_semantics(
            response=retry_response,
            forecast_result=forecast_result,
            capacity=capacity,
        )
    )


    # --------------------------------------------------
    # 10. FAIL CLOSED IF RETRY IS STILL INVALID
    # --------------------------------------------------

    if retry_semantic_errors:

        error_message = " | ".join(
            retry_semantic_errors
        )

        raise RuntimeError(
            "The AI response failed semantic "
            "validation after one corrective retry: "
            f"{error_message}"
        )


    # --------------------------------------------------
    # 11. RETURN CORRECTED VALIDATED RESPONSE
    # --------------------------------------------------

    return retry_response