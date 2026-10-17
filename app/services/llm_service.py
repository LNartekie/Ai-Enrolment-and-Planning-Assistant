import os

import requests
from dotenv import load_dotenv


# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:1.7b"
)


# --------------------------------------------------
# SYSTEM PROMPT
# --------------------------------------------------

SYSTEM_PROMPT = """
You are an institutional enrolment planning assistant.

Your role is to explain analytical and forecasting
results supplied by deterministic Python tools.

Important rules:

1. Do not invent enrolment values, percentages,
   forecast metrics, capacity figures, or trends.

2. Use only the numerical information supplied
   in the user message.

3. Do not recalculate or change supplied values.

4. Clearly distinguish historical facts from forecasts.

5. Explain forecasts as planning estimates rather
   than guaranteed outcomes.

6. If a capacity gap exists, explain its planning
   implication without making an autonomous decision.

7. Staffing, budgeting, admissions, programme,
   and capacity decisions must remain subject
   to human review.

8. If information is unavailable, say that it is
   unavailable rather than guessing.

Use clear professional language suitable for
a university administrator.
"""


# --------------------------------------------------
# CHECK OLLAMA CONNECTION
# --------------------------------------------------

def check_ollama_connection() -> bool:
    """
    Check whether the Ollama service is available.
    """

    try:

        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=5
        )

        return response.status_code == 200

    except requests.RequestException:

        return False


# --------------------------------------------------
# GENERATE EXPLANATION
# --------------------------------------------------

def generate_forecast_explanation(
    trend_summary: dict,
    forecast_result: dict,
    capacity: float | None = None
) -> str:
    """
    Ask Ollama to explain deterministic
    enrolment analysis and forecast results.
    """

    forecast_year = (
        forecast_result["forecast_year"]
    )

    predicted_enrolment = (
        forecast_result[
            "predicted_enrolment"
        ]
    )

    selected_model = (
        forecast_result[
            "selected_model"
        ]
    )

    mae = (
        forecast_result["mae"]
    )

    rmse = (
        forecast_result["rmse"]
    )

    mape = (
        forecast_result["mape"]
    )

    evidence = (
        forecast_result[
            "backtest_evidence"
        ]
    )

    latest_enrolment = (
        trend_summary[
            "latest_enrolment"
        ]
    )

    overall_direction = (
        trend_summary[
            "overall_direction"
        ]
    )

    recent_direction = (
        trend_summary[
            "recent_direction"
        ]
    )


    # --------------------------------------------------
    # OPTIONAL CAPACITY INFORMATION
    # --------------------------------------------------

    if capacity is not None:

        capacity_gap = (
            predicted_enrolment
            - capacity
        )

        capacity_text = f"""
Current supplied capacity: {capacity:.0f}
Forecast capacity gap: {capacity_gap:.0f}
"""

    else:

        capacity_text = """
No capacity information was supplied.
"""


    # --------------------------------------------------
    # FORMAT MAPE
    # --------------------------------------------------

    if mape is not None:

        mape_text = f"{mape:.2f}%"

    else:

        mape_text = "Unavailable"


    # --------------------------------------------------
    # USER PROMPT
    # --------------------------------------------------

    user_prompt = f"""
Explain the following institutional enrolment
analysis to a university administrator.

HISTORICAL ANALYSIS

Latest historical enrolment:
{latest_enrolment:.0f}

Overall historical direction:
{overall_direction}

Recent direction:
{recent_direction}


FORECAST

Forecast year:
{forecast_year}

Predicted enrolment:
{predicted_enrolment:.0f}

Selected forecasting model:
{selected_model}


BACKTEST PERFORMANCE

MAE:
{mae:.2f} students

RMSE:
{rmse:.2f} students

MAPE:
{mape_text}

Backtest evidence:
{evidence}


CAPACITY INFORMATION

{capacity_text}


Provide:

- a short explanation of the historical trend;
- a clear explanation of the forecast;
- an interpretation of the model error;
- the capacity implication, if capacity is available;
- an explicit reminder that the forecast requires
  human review.

Do not create any numerical values that are not
provided above.
"""


    # --------------------------------------------------
    # SEND REQUEST TO OLLAMA
    # --------------------------------------------------

    payload = {
    "model": OLLAMA_MODEL,

    "messages": [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ],

    "think": False,
    "stream": False
}


    try:

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        response_data = (
            response.json()
        )

        explanation = (
            response_data[
                "message"
            ][
                "content"
            ]
        )

        return explanation.strip()


    except requests.RequestException as error:

        raise RuntimeError(
            "The AI explanation service is "
            f"currently unavailable: {error}"
        )