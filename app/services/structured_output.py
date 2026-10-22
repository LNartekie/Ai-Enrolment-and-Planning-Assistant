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
    Ask Ollama to return a forecast explanation
    that matches the ForecastExplanation
    Pydantic schema.

    The returned JSON is validated before it is
    accepted by the application.
    """

    selected_model = (
        model_name
        if model_name is not None
        else OLLAMA_MODEL
    )


    # --------------------------------------------------
    # 1. BUILD THE VERIFIED PROMPT
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
    # 3. ADD STRUCTURED OUTPUT INSTRUCTIONS
    # --------------------------------------------------

    user_prompt = f"""
{base_prompt}

STRUCTURED OUTPUT REQUIREMENT

Return only valid JSON.

The JSON must match the supplied schema exactly.

Do not include Markdown.

Do not include code fences.

Do not include explanatory text outside the JSON.

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

        # Ollama will attempt to constrain the
        # model response to this JSON schema.
        "format": response_schema,

        # Lower temperature gives more
        # deterministic structured responses.
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
    # 6. EXTRACT MODEL RESPONSE
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
    # 7. VALIDATE JSON WITH PYDANTIC
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
    # 8. RETURN ONLY VALIDATED DATA
    # --------------------------------------------------

    return validated_response