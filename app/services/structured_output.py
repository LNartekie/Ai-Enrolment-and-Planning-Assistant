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
# GENERATE STRUCTURED FORECAST EXPLANATION
# --------------------------------------------------

def generate_structured_forecast_explanation(
    trend_summary: dict,
    forecast_result: dict,
    capacity: float | None = None,
    model_name: str | None = None,
) -> ForecastExplanation:
    """
    Generate an AI forecast explanation that must pass:

    1. Ollama JSON-schema constraints
    2. Pydantic structural validation
    3. Deterministic semantic validation
    """

    selected_model = (
        model_name
        if model_name is not None
        else OLLAMA_MODEL
    )


    # --------------------------------------------------
    # 1. BUILD VERIFIED PROMPT
    # --------------------------------------------------

    base_prompt = build_forecast_prompt(
        trend_summary=trend_summary,
        forecast_result=forecast_result,
        capacity=capacity,
        variant="C",
    )


    # --------------------------------------------------
    # 2. CREATE JSON SCHEMA
    # --------------------------------------------------

    response_schema = (
        ForecastExplanation.model_json_schema()
    )


    # --------------------------------------------------
    # 3. STRUCTURED OUTPUT INSTRUCTIONS
    # --------------------------------------------------

    user_prompt = f"""
{base_prompt}

STRUCTURED OUTPUT REQUIREMENT

Return only valid JSON.

The JSON must match the supplied schema exactly.

Do not include Markdown.
Do not include code fences.
Do not include text outside the JSON.

Do not describe model accuracy or error as high,
moderate, low, good, or poor.

Do not say that the model or forecast was validated.

A backtest evidence label describes only the amount
of historical evaluation available.

JSON SCHEMA:

{json.dumps(response_schema, indent=2)}
"""


    # --------------------------------------------------
    # 4. BUILD OLLAMA REQUEST
    # --------------------------------------------------

    payload = {
        "model": selected_model,

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
    # 5. CALL OLLAMA
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
    # 6. EXTRACT RAW RESPONSE
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
    # 7. PYDANTIC STRUCTURAL VALIDATION
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


    # --------------------------------------------------
    # 8. SEMANTIC VALIDATION
    # --------------------------------------------------

    semantic_errors = (
        validate_forecast_semantics(
            response=validated_response,
            forecast_result=forecast_result,
            capacity=capacity,
        )
    )


    if semantic_errors:

        error_message = (
            " | ".join(
                semantic_errors
            )
        )

        raise RuntimeError(
            "The AI response passed structural "
            "validation but failed semantic "
            "validation: "
            f"{error_message}"
        )


    # --------------------------------------------------
    # 9. RETURN ONLY FULLY VALIDATED OUTPUT
    # --------------------------------------------------

    return validated_response