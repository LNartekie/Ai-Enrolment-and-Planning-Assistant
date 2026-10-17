from pathlib import Path

import numpy as np
import pandas as pd


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
# 2. LOAD BENCHMARK DATA
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

print("\n=====================================")
print("FORECASTING DATA")
print("=====================================")

print(f"Rows: {len(df):,}")
print(f"Institutions: {df['UNITID'].nunique():,}")


# --------------------------------------------------
# 3. SELECT ONE REPRESENTATIVE INSTITUTION
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

# Use the most recent institution name
institution_name = (
    institution_df
    .sort_values("year")
    .iloc[-1]["INSTNM"]
)


# --------------------------------------------------
# 4. CREATE GENERIC FORECASTING FORMAT
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

print("\n=====================================")
print("REPRESENTATIVE INSTITUTION")
print("=====================================")

print(f"UNITID: {representative_id}")
print(f"Institution: {institution_name}")
print(f"Typical enrolment: {overall_median:.1f}")

print("\nHistorical data:")
print(series_df)


# --------------------------------------------------
# 5. DEFINE NAIVE FORECAST FUNCTION
# --------------------------------------------------

def naive_forecast(history):
    """
    Predict next year's enrolment using the most
    recent observed enrolment value.
    """

    return history["enrolment"].iloc[-1]


# --------------------------------------------------
# 6. WALK-FORWARD BACKTEST
# --------------------------------------------------

test_years = [2011, 2012, 2013]

predictions = []

for test_year in test_years:

    # Only use information available before the test year
    history = series_df[
        series_df["year"] < test_year
    ].copy()

    # Actual enrolment for the year being predicted
    actual_row = series_df[
        series_df["year"] == test_year
    ]

    if history.empty or actual_row.empty:
        continue

    actual = actual_row["enrolment"].iloc[0]

    prediction = naive_forecast(history)

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
# 7. CONVERT RESULTS TO DATAFRAME
# --------------------------------------------------

results_df = pd.DataFrame(predictions)


print("\n=====================================")
print("NAIVE BASELINE PREDICTIONS")
print("=====================================")

print(results_df)


# --------------------------------------------------
# 8. CALCULATE EVALUATION METRICS
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
print("NAIVE BASELINE PERFORMANCE")
print("=====================================")

print(f"MAE:  {mae:.2f} students")
print(f"RMSE: {rmse:.2f} students")
print(f"MAPE: {mape:.2f}%")


# --------------------------------------------------
# 9. SAVE RESULTS
# --------------------------------------------------

results_df.to_csv(
    RESULTS_FOLDER
    / "baseline_single_institution_predictions.csv",
    index=False
)

metrics_df = pd.DataFrame({
    "model": ["Naive Baseline"],
    "institution_id": [representative_id],
    "institution_name": [institution_name],
    "mae": [mae],
    "rmse": [rmse],
    "mape": [mape]
})

metrics_df.to_csv(
    RESULTS_FOLDER
    / "baseline_single_institution_metrics.csv",
    index=False
)


print("\n=====================================")
print("FORECASTING STEP 1 COMPLETE")
print("=====================================")