from pathlib import Path
import sys
import time

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
    validate_enrolment_data
)

from app.services.trend_analysis import (
    analyse_enrolment_trend
)

from app.services.forecasting import (
    run_forecasting_pipeline
)

from app.services.llm_service import (
    generate_forecast_explanation
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

RESULTS_FOLDER = (
    PROJECT_ROOT
    / "results"
)

RESULTS_FOLDER.mkdir(
    exist_ok=True
)


# --------------------------------------------------
# MODEL AND PROMPT VARIATIONS
# --------------------------------------------------

MODEL_NAME = "qwen3:1.7b"

PROMPT_VARIANTS = [
    "A",
    "B",
    "C",
]


# --------------------------------------------------
# LOAD DEMO DATA
# --------------------------------------------------

data = pd.read_csv(
    DEMO_FILE
)


# --------------------------------------------------
# VALIDATE DATA
# --------------------------------------------------

validation = (
    validate_enrolment_data(
        data
    )
)

if not validation["valid"]:

    raise ValueError(
        validation["errors"]
    )


cleaned_data = (
    validation["cleaned_data"]
)


# --------------------------------------------------
# RUN DETERMINISTIC ANALYSIS
# --------------------------------------------------

trend_result = (
    analyse_enrolment_trend(
        cleaned_data
    )
)

trend_summary = (
    trend_result["summary"]
)

forecast_result = (
    run_forecasting_pipeline(
        cleaned_data
    )
)


# --------------------------------------------------
# GET CAPACITY
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
# RUN PROMPT VARIATIONS
# --------------------------------------------------

results = []


print(
    "\n====================================="
)

print(
    "QWEN3 PROMPT VARIATION EXPERIMENT"
)

print(
    "====================================="
)


for variant in PROMPT_VARIANTS:

    print(
        f"\nRunning prompt variant {variant}"
    )

    start_time = time.time()

    try:

        response = (
            generate_forecast_explanation(
                trend_summary=trend_summary,
                forecast_result=forecast_result,
                capacity=capacity,
                model_name=MODEL_NAME,
                prompt_variant=variant,
            )
        )

        elapsed_time = (
            time.time()
            - start_time
        )

        results.append({
            "variant": variant,
            "model": MODEL_NAME,
            "status": "Success",
            "response_time_seconds":
                elapsed_time,
            "response": response,
        })

        print(
            f"Completed in "
            f"{elapsed_time:.2f} seconds"
        )


    except Exception as error:

        elapsed_time = (
            time.time()
            - start_time
        )

        results.append({
            "variant": variant,
            "model": MODEL_NAME,
            "status": "Failed",
            "response_time_seconds":
                elapsed_time,
            "response": str(error),
        })

        print(
            f"Failed: {error}"
        )


# --------------------------------------------------
# SAVE CSV RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    RESULTS_FOLDER
    / "prompt_variation_results.csv",
    index=False,
)


# --------------------------------------------------
# SAVE READABLE MARKDOWN REPORT
# --------------------------------------------------

report_path = (
    RESULTS_FOLDER
    / "prompt_variation_results.md"
)


with open(
    report_path,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "# Qwen3 Prompt Variation Experiment\n\n"
    )

    file.write(
        "Model: qwen3:1.7b\n\n"
    )

    file.write(
        "All prompt variations received the same "
        "deterministic analytical and forecasting "
        "results.\n\n"
    )

    for result in results:

        file.write(
            f"## Prompt Variant "
            f"{result['variant']}\n\n"
        )

        file.write(
            f"**Status:** "
            f"{result['status']}\n\n"
        )

        file.write(
            f"**Response time:** "
            f"{result['response_time_seconds']:.2f} "
            f"seconds\n\n"
        )

        file.write(
            result["response"]
        )

        file.write(
            "\n\n---\n\n"
        )


print(
    "\n====================================="
)

print(
    "PROMPT EXPERIMENT COMPLETE"
)

print(
    "====================================="
)

print(
    "\nSaved:"
)

print(
    RESULTS_FOLDER
    / "prompt_variation_results.csv"
)

print(
    report_path
)