from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import Holt


# --------------------------------------------------
# 1. LOCATE PROJECT FILES
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting_base_2001_2013.csv"
)

RESULTS_FOLDER = PROJECT_ROOT / "results"
RESULTS_FOLDER.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)


# --------------------------------------------------
# 3. SELECT SAME REPRESENTATIVE INSTITUTION
# --------------------------------------------------

institution_medians = (
    df.groupby("UNITID")["UGDS"]
    .median()
)

overall_median = institution_medians.median()

representative_id = (
    (institution_medians - overall_median)
    .abs()
    .idxmin()
)

institution_df = (
    df[df["UNITID"] == representative_id]
    .sort_values("year")
    .copy()
)

institution_name = institution_df.iloc[-1]["INSTNM"]


# --------------------------------------------------
# 4. CREATE GENERIC FORMAT
# --------------------------------------------------

series_df = (
    institution_df[
        ["year", "UGDS"]
    ]
    .rename(
        columns={
            "UGDS": "enrolment"
        }
    )
    .reset_index(drop=True)
)


# --------------------------------------------------
# 5. HOLT FORECAST FUNCTION
# --------------------------------------------------

def holt_forecast(history):

    model = Holt(
        history["enrolment"],
        initialization_method="estimated"
    )

    fitted_model = model.fit(
        optimized=True
    )

    prediction = fitted_model.forecast(1).iloc[0]

    return prediction


# --------------------------------------------------
# 6. WALK-FORWARD BACKTEST
# --------------------------------------------------

test_years = [2011, 2012, 2013]

predictions = []

for test_year in test_years:

    history = series_df[
        series_df["year"] < test_year
    ].copy()

    actual_row = series_df[
        series_df["year"] == test_year
    ]

    if history.empty or actual_row.empty:
        continue

    actual = actual_row["enrolment"].iloc[0]

    prediction = holt_forecast(history)

    # Enrolment cannot realistically be negative
    prediction = max(0, prediction)

    error = actual - prediction

    absolute_error = abs(error)

    percentage_error = (
        absolute_error / actual * 100
        if actual != 0
        else np.nan
    )

    predictions.append({
        "test_year": test_year,
        "actual": actual,
        "prediction": prediction,
        "error": error,
        "absolute_error": absolute_error,
        "absolute_percentage_error": percentage_error
    })


# --------------------------------------------------
# 7. RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(predictions)

print("\n=====================================")
print("HOLT TREND PREDICTIONS")
print("=====================================")

print(results_df)


# --------------------------------------------------
# 8. EVALUATION METRICS
# --------------------------------------------------

mae = results_df["absolute_error"].mean()

rmse = np.sqrt(
    np.mean(
        results_df["error"] ** 2
    )
)

mape = (
    results_df[
        "absolute_percentage_error"
    ]
    .mean()
)


print("\n=====================================")
print("HOLT TREND PERFORMANCE")
print("=====================================")

print(f"MAE:  {mae:.2f} students")
print(f"RMSE: {rmse:.2f} students")
print(f"MAPE: {mape:.2f}%")


# --------------------------------------------------
# 9. LOAD PREVIOUS RESULTS
# --------------------------------------------------

comparison_file = (
    RESULTS_FOLDER
    / "baseline_vs_linear_trend.csv"
)

previous_models = pd.read_csv(
    comparison_file
)


# --------------------------------------------------
# 10. ADD HOLT TO MODEL COMPARISON
# --------------------------------------------------

holt_results = pd.DataFrame({
    "model": ["Holt Linear Trend"],
    "mae": [mae],
    "rmse": [rmse],
    "mape": [mape]
})

comparison_df = pd.concat(
    [
        previous_models,
        holt_results
    ],
    ignore_index=True
)


print("\n=====================================")
print("MODEL COMPARISON")
print("=====================================")

print(
    comparison_df.sort_values("mae")
)


# --------------------------------------------------
# 11. SAVE RESULTS
# --------------------------------------------------

results_df.to_csv(
    RESULTS_FOLDER
    / "holt_single_institution_predictions.csv",
    index=False
)

comparison_df.to_csv(
    RESULTS_FOLDER
    / "single_institution_model_comparison.csv",
    index=False
)


print("\nFORECASTING STEP 3 COMPLETE")