"""Validated Ollama tool-calling workflow for institutional capacity analysis."""

import json
import os

import requests
from dotenv import load_dotenv

from app.services.planning_tools import CapacityAnalysisInput, analyse_capacity


load_dotenv()

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
).rstrip("/")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")


# --------------------------------------------------
# TOOL DEFINITION EXPOSED TO QWEN
# --------------------------------------------------

CAPACITY_TOOL = {
    "type": "function",
    "function": {
        "name": "analyse_capacity",
        "description": (
            "Calculate the capacity gap, capacity utilisation percentage, "
            "and planning status using forecast enrolment and approved capacity."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "forecast_enrolment": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Forecast student enrolment for the institution.",
                },
                "approved_capacity": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Formally approved institutional capacity.",
                },
            },
            "required": ["forecast_enrolment", "approved_capacity"],
            "additionalProperties": False,
        },
    },
}


# --------------------------------------------------
# OLLAMA CHAT HELPER
# --------------------------------------------------

def _chat(messages, model_name, tools=None):
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": 0},
    }

    if tools is not None:
        payload["tools"] = tools

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        message = response.json()["message"]

        if not isinstance(message, dict):
            raise ValueError("Expected an assistant message object.")

        return message
    except (requests.RequestException, KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Ollama tool-calling request failed.") from error


# --------------------------------------------------
# MAIN TOOL-CALLING WORKFLOW
# --------------------------------------------------

def explain_capacity_with_tool(
    forecast_enrolment: int,
    approved_capacity: int,
    model_name: str | None = None,
) -> dict:
       
    """Let Qwen request an approved tool; run it using validated inputs."""

    # 1. Validate the numbers supplied by the application.
    inputs = CapacityAnalysisInput(
        forecast_enrolment=forecast_enrolment,
        approved_capacity=approved_capacity,
    )
    selected_model = model_name or OLLAMA_MODEL

    messages = [
        {
            "role": "system",
            "content": (
                "You are an institutional enrolment planning assistant. "
                "You MUST request the analyse_capacity tool before explaining "
                "capacity metrics. Pass the exact two values from the user. "
                "Do not invent numerical values. A forecast does not grant "
                "permission to exceed approved capacity."
            ),
        },
        {
            "role": "user",
            "content": (
                "Use the analyse_capacity tool to evaluate this scenario: "
                f"forecast_enrolment={inputs.forecast_enrolment}; "
                f"approved_capacity={inputs.approved_capacity}. "
                "Request the tool now."
            ),
        },
    ]

    # 2. Ask the model to request the tool.
    first_message = _chat(
        messages=messages,
        model_name=selected_model,
        tools=[CAPACITY_TOOL],
    )

    tool_calls = first_message.get("tool_calls") or []
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        raise RuntimeError(
            "Qwen did not request exactly one capacity-analysis tool call."
        )

    tool_call = tool_calls[0]
    if not isinstance(tool_call, dict):
        raise RuntimeError("Qwen returned an invalid tool call.")

    function = tool_call.get("function", {})
    if not isinstance(function, dict) or function.get("name") != "analyse_capacity":
        raise RuntimeError("Qwen requested an unapproved tool.")

    # 3. Read arguments. Ollama may return a dict or a JSON string.
    arguments = function.get("arguments", {})

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise RuntimeError("Invalid JSON in tool-call arguments.") from error

    if not isinstance(arguments, dict):
        raise RuntimeError("Tool arguments must be a JSON object.")

    # Do not allow the model to override trusted application data.
    try:
        tool_inputs = CapacityAnalysisInput.model_validate(arguments)
    except ValueError as error:
        raise RuntimeError(
            "Qwen supplied invalid capacity tool arguments."
        ) from error

    if (
        tool_inputs.forecast_enrolment != inputs.forecast_enrolment
        or tool_inputs.approved_capacity != inputs.approved_capacity
    ):
        raise RuntimeError(
            "Qwen's tool arguments do not match the validated application inputs."
        )

    # 4. Execute the allowlisted Python tool using trusted values.
    tool_result = analyse_capacity(
        forecast_enrolment=inputs.forecast_enrolment,
        approved_capacity=inputs.approved_capacity,
    )

    # 5. Give the tool's calculated result back to Ollama.
    messages.append({
        "role": "assistant",
        "content": first_message.get("content") or "",
        "tool_calls": tool_calls,
    })
    messages.append({
        "role": "tool",
        "tool_name": "analyse_capacity",
        "content": json.dumps(tool_result),
    })
    messages.append({
        "role": "system",
        "content": (
            "Explain the verified tool result clearly and briefly. "
            "Use the exact values returned by Python. "
            "Do not invent figures or institutional approvals."
        ),
    })

    final_message = _chat(messages=messages, model_name=selected_model)

    if final_message.get("tool_calls"):
        raise RuntimeError("Unexpected additional tool call.")

    explanation = final_message.get("content", "")
    if not isinstance(explanation, str) or not explanation.strip():
        raise RuntimeError("Qwen returned no explanation.")

    return {
        "tool_called": "analyse_capacity",
        "tool_result": tool_result,
        "explanation": explanation.strip(),
    }
