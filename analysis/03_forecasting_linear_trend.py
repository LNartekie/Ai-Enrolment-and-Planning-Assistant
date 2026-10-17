from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


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
# 4. GENERIC FORECASTING FORMAT
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
# 5. LINEAR TREND FORECAST FUNCTION
# --------------------------------------------------

def linear_trend_forecast(history, forecast_year):

    # X must be 2-dimensional for scikit-learn
    X = history[["year"]]

    # y is the value we want to model
    y = history["enrolment"]

    model = LinearRegression()

    model.fit(X, y)

    future_year = pd.DataFrame({
        "year": [forecast_year]
    })

    prediction = model.predict(future_year)[0]

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

    prediction = linear_trend_forecast(
        history,
        test_year
    )

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
print("LINEAR TREND PREDICTIONS")
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
print("LINEAR TREND PERFORMANCE")
print("=====================================")

print(f"MAE:  {mae:.2f} students")
print(f"RMSE: {rmse:.2f} students")
print(f"MAPE: {mape:.2f}%")


# --------------------------------------------------
# 9. COMPARE WITH NAIVE BASELINE
# --------------------------------------------------

baseline_file = (
    RESULTS_FOLDER
    / "baseline_single_institution_metrics.csv"
)

baseline_df = pd.read_csv(baseline_file)

comparison_df = pd.DataFrame({
    "model": [
        "Naive Baseline",
        "Linear Trend"
    ],

    "mae": [
        baseline_df.loc[0, "mae"],
        mae
    ],

    "rmse": [
        baseline_df.loc[0, "rmse"],
        rmse
    ],

    "mape": [
        baseline_df.loc[0, "mape"],
        mape
    ]
})


print("\n=====================================")
print("MODEL COMPARISON")
print("=====================================")

print(comparison_df)


# --------------------------------------------------
# 10. SAVE RESULTS
# --------------------------------------------------

results_df.to_csv(
    RESULTS_FOLDER
    / "linear_trend_single_institution_predictions.csv",
    index=False
)

comparison_df.to_csv(
    RESULTS_FOLDER
    / "baseline_vs_linear_trend.csv",
    index=False
)


print("\nFORECASTING STEP 2 COMPLETE")