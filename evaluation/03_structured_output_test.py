from pathlib import Path
import sys

import pandas as pd


# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# --------------------------------------------------
# APPLICATION IMPORTS
# --------------------------------------------------

from app.services.data_validation import (
    validate_enrolment_data,
)

from app.services.trend_analysis import (
    analyse_enrolment_trend,
)

from app.services.forecasting import (
    run_forecasting_pipeline,
)

from app.services.structured_output import (
    generate_structured_forecast_explanation,
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

DEMO_FILE = (
    PROJECT_ROOT
    / "data"
    / "demo"
    / "demo_institution_enrolment.csv"
)


# --------------------------------------------------
# LOAD DEMO DATA
# --------------------------------------------------

data = pd.read_csv(
    DEMO_FILE
)


# --------------------------------------------------
# VALIDATE DATA
# --------------------------------------------------

validation = validate_enrolment_data(
    data
)


if not validation["valid"]:

    raise ValueError(
        validation["errors"]
    )


cleaned_data = validation[
    "cleaned_data"
]


# --------------------------------------------------
# TREND ANALYSIS
# --------------------------------------------------

trend_result = analyse_enrolment_trend(
    cleaned_data
)

trend_summary = trend_result[
    "summary"
]


# --------------------------------------------------
# FORECASTING
# --------------------------------------------------

forecast_result = run_forecasting_pipeline(
    cleaned_data
)


# --------------------------------------------------
# GET LATEST CAPACITY
# --------------------------------------------------

capacity = None


if "capacity" in cleaned_data.columns:

    valid_capacity = (
        cleaned_data["capacity"]
        .dropna()
    )

    if not valid_capacity.empty:

        capacity = float(
            valid_capacity.iloc[-1]
        )


# --------------------------------------------------
# GENERATE STRUCTURED AI RESPONSE
# --------------------------------------------------

print(
    "\nGenerating structured response...\n"
)


structured_response = (
    generate_structured_forecast_explanation(
        trend_summary=trend_summary,
        forecast_result=forecast_result,
        capacity=capacity,
    )
)


# --------------------------------------------------
# DISPLAY VALIDATED RESULT
# --------------------------------------------------

print(
    "======================================"
)

print(
    "VALIDATED STRUCTURED RESPONSE"
)

print(
    "======================================\n"
)


print(
    structured_response.model_dump_json(
        indent=2
    )
)


# --------------------------------------------------
# DISPLAY IMPORTANT FIELD TYPES
# --------------------------------------------------

print(
    "\n======================================"
)

print(
    "VALIDATION CHECK"
)

print(
    "======================================\n"
)


print(
    "Object type:",
    type(structured_response).__name__,
)


print(
    "Capacity status:",
    structured_response.capacity_status,
)


print(
    "Human review required:",
    structured_response.human_review_required,
)