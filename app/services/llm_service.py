import os

import requests
from dotenv import load_dotenv

from app.prompts.forecast_prompts import (
    SYSTEM_PROMPT,
    build_forecast_prompt,
    build_verified_context,
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
# CHECK OLLAMA CONNECTION
# --------------------------------------------------

def check_ollama_connection() -> bool:
    """
    Check whether the local Ollama service
    is available.
    """

    try:

        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=5,
        )

        return response.status_code == 200

    except requests.RequestException:

        return False


# --------------------------------------------------
# GENERIC OLLAMA CHAT REQUEST
# --------------------------------------------------

def call_ollama_chat(
    messages: list,
    model_name: str | None = None,
    temperature: float = 0.2,
) -> str:
    """
    Send a list of chat messages to Ollama
    and return the assistant's response.
    """

    selected_model = (
        model_name
        if model_name is not None
        else OLLAMA_MODEL
    )


    payload = {
        "model": selected_model,

        "messages": messages,

        # Do not expose Qwen's reasoning process.
        "think": False,

        # Return one complete response.
        "stream": False,

        # Low temperature gives more consistent,
        # less creative administrative responses.
        "options": {
            "temperature": temperature,
        },
    }


    try:

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=120,
        )

        response.raise_for_status()

    except requests.RequestException as error:

        raise RuntimeError(
            "The AI service is currently unavailable: "
            f"{error}"
        ) from error


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
            "Ollama returned an unexpected or "
            "malformed response."
        ) from error


    if not content.strip():

        raise RuntimeError(
            "Ollama returned an empty response."
        )


    return content.strip()


# --------------------------------------------------
# GENERATE INITIAL FORECAST EXPLANATION
# --------------------------------------------------

def generate_forecast_explanation(
    trend_summary: dict,
    forecast_result: dict,
    capacity: float | None = None,
    model_name: str | None = None,
    prompt_variant: str = "C",
) -> str:
    """
    Generate the initial administrator-facing
    forecast explanation.

    Prompt C is the production prompt selected
    after prompt evaluation.
    """

    user_prompt = build_forecast_prompt(
        trend_summary=trend_summary,
        forecast_result=forecast_result,
        capacity=capacity,
        variant=prompt_variant,
    )


    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


    return call_ollama_chat(
        messages=messages,
        model_name=model_name,
        temperature=0.2,
    )


# --------------------------------------------------
# GENERATE MULTI-TURN FOLLOW-UP RESPONSE
# --------------------------------------------------

def generate_followup_response(
    trend_summary: dict,
    forecast_result: dict,
    conversation_history: list,
    user_message: str,
    capacity: float | None = None,
    model_name: str | None = None,
) -> str:
    """
    Answer a follow-up administrator question
    using verified analytical results and recent
    conversation history.
    """

    # --------------------------------------------------
    # 1. CHECK USER MESSAGE
    # --------------------------------------------------

    if not user_message.strip():

        raise ValueError(
            "The user message cannot be empty."
        )


    # --------------------------------------------------
    # 2. BUILD VERIFIED ANALYTICAL CONTEXT
    # --------------------------------------------------

    verified_context = build_verified_context(
        trend_summary=trend_summary,
        forecast_result=forecast_result,
        capacity=capacity,
    )


    # --------------------------------------------------
    # 3. BUILD CONVERSATION SYSTEM PROMPT
    # --------------------------------------------------

    conversation_system_prompt = (
        SYSTEM_PROMPT
        + """

You are answering a follow-up question in a
multi-turn conversation.

Use the verified analytical context as the
authoritative factual source for all numerical claims.

Conversation history may help you understand what
the administrator is referring to, but previous
assistant messages must not override the verified
analytical results.

If the administrator asks for information that is
not available in the verified context, clearly say
that the information is unavailable.

Do not invent institutional policies, student demand,
staffing requirements, budgets, facilities, causes,
or future events.

Keep the response focused on the user's question.
"""
    )


    # --------------------------------------------------
    # 4. START MESSAGE LIST
    # --------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": conversation_system_prompt,
        },

        {
            "role": "user",
            "content": (
                "Use the following verified analytical "
                "results as the factual context for this "
                "conversation:\n\n"
                + verified_context
            ),
        },

        {
            "role": "assistant",
            "content": (
                "I will use these verified analytical "
                "results when answering follow-up questions."
            ),
        },
    ]


    # --------------------------------------------------
    # 5. ADD RECENT CONVERSATION HISTORY
    # --------------------------------------------------

    recent_history = (
        conversation_history[-8:]
    )


    for message in recent_history:

        role = message.get(
            "role"
        )

        content = message.get(
            "content"
        )


        if (
            role in {
                "user",
                "assistant",
            }
            and content
        ):

            messages.append({
                "role": role,
                "content": content,
            })


    # --------------------------------------------------
    # 6. ADD CURRENT USER QUESTION
    # --------------------------------------------------

    messages.append({
        "role": "user",
        "content": user_message,
    })


    # --------------------------------------------------
    # 7. GENERATE RESPONSE
    # --------------------------------------------------

    return call_ollama_chat(
        messages=messages,
        model_name=model_name,
        temperature=0.2,
    )