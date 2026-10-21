from pathlib import Path
import sys
import time

import pandas as pd


# --------------------------------------------------
# ADD PROJECT ROOT TO PYTHON PATH
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
# 1. PROJECT PATHS
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
# 2. MODELS TO COMPARE
# --------------------------------------------------

MODELS = [
    "qwen3:1.7b",
    "qwen:latest",
    "mistral:latest",
    "llama3.2:latest",
]


# --------------------------------------------------
# 3. LOAD DEMO DATA
# --------------------------------------------------

data = pd.read_csv(
    DEMO_FILE
)


# --------------------------------------------------
# 4. VALIDATE DATA
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
    validation[
        "cleaned_data"
    ]
)


# --------------------------------------------------
# 5. RUN DETERMINISTIC ANALYSIS
# --------------------------------------------------

trend_result = (
    analyse_enrolment_trend(
        cleaned_data
    )
)

trend_summary = (
    trend_result[
        "summary"
    ]
)

forecast_result = (
    run_forecasting_pipeline(
        cleaned_data
    )
)


# --------------------------------------------------
# 6. GET CAPACITY
# --------------------------------------------------

capacity = None

if "capacity" in cleaned_data.columns:

    valid_capacity = (
        cleaned_data[
            "capacity"
        ]
        .dropna()
    )

    if not valid_capacity.empty:

        capacity = float(
            valid_capacity.iloc[-1]
        )


# --------------------------------------------------
# 7. RUN SAME PROMPT THROUGH EACH MODEL
# --------------------------------------------------

results = []


print(
    "\n====================================="
)
print(
    "LLM MODEL COMPARISON"
)
print(
    "====================================="
)


for model_name in MODELS:

    print(
        f"\nRunning: {model_name}"
    )

    start_time = time.time()

    try:

        response = (
            generate_forecast_explanation(
                trend_summary=trend_summary,
                forecast_result=forecast_result,
                capacity=capacity,
                model_name=model_name
            )
        )

        elapsed_time = (
            time.time()
            - start_time
        )

        results.append({
            "model":
                model_name,

            "status":
                "Success",

            "response_time_seconds":
                elapsed_time,

            "response":
                response
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
            "model":
                model_name,

            "status":
                "Failed",

            "response_time_seconds":
                elapsed_time,

            "response":
                str(error)
        })

        print(
            f"Failed: {error}"
        )


# --------------------------------------------------
# 8. SAVE CSV SUMMARY
# --------------------------------------------------

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    RESULTS_FOLDER
    / "llm_model_comparison.csv",
    index=False
)


# --------------------------------------------------
# 9. SAVE READABLE MARKDOWN REPORT
# --------------------------------------------------

report_file = (
    RESULTS_FOLDER
    / "llm_model_comparison.md"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "# LLM Model Comparison\n\n"
    )

    file.write(
        "All models received the same deterministic "
        "forecast results, system instructions, and "
        "user prompt.\n\n"
    )

    for result in results:

        file.write(
            f"## {result['model']}\n\n"
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
    "COMPARISON COMPLETE"
)
print(
    "====================================="
)

print(
    "\nSaved:"
)

print(
    RESULTS_FOLDER
    / "llm_model_comparison.csv"
)

print(
    RESULTS_FOLDER
    / "llm_model_comparison.md"
)