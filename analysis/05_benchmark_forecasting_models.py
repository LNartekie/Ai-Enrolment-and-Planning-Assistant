from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tools.sm_exceptions import ConvergenceWarning


# --------------------------------------------------
# 1. PROJECT PATHS
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
# 2. SETTINGS
# --------------------------------------------------

TEST_YEARS = [2011, 2012, 2013]

warnings.simplefilter(
    "ignore",
    ConvergenceWarning
)


# --------------------------------------------------
# 3. LOAD BENCHMARK DATA
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

df = df.sort_values(
    ["UNITID", "year"]
).reset_index(drop=True)

print("\n=====================================")
print("BENCHMARK DATA")
print("=====================================")

print(f"Rows: {len(df):,}")
print(
    f"Institutions: "
    f"{df['UNITID'].nunique():,}"
)
print(
    f"Years: "
    f"{df['year'].min()}-"
    f"{df['year'].max()}"
)


# --------------------------------------------------
# 4. CREATE INSTITUTION SIZE GROUPS
# --------------------------------------------------

institution_sizes = (
    df.groupby("UNITID")["UGDS"]
    .median()
    .rename("median_enrolment")
    .reset_index()
)

q1 = institution_sizes[
    "median_enrolment"
].quantile(0.25)

q3 = institution_sizes[
    "median_enrolment"
].quantile(0.75)


def classify_size(value):

    if value <= q1:
        return "Small"

    elif value <= q3:
        return "Medium"

    else:
        return "Large"


institution_sizes["size_category"] = (
    institution_sizes[
        "median_enrolment"
    ].apply(classify_size)
)

print("\n=====================================")
print("SIZE THRESHOLDS")
print("=====================================")

print(f"Small:  <= {q1:.2f}")
print(
    f"Medium: > {q1:.2f} "
    f"and <= {q3:.2f}"
)
print(f"Large:  > {q3:.2f}")


# --------------------------------------------------
# 5. FORECAST FUNCTIONS
# --------------------------------------------------

def naive_forecast(history):

    prediction = (
        history["enrolment"]
        .iloc[-1]
    )

    return float(prediction)


def linear_forecast(
    history,
    forecast_year
):

    X = history[["year"]]

    y = history["enrolment"]

    model = LinearRegression()

    model.fit(X, y)

    future = pd.DataFrame({
        "year": [forecast_year]
    })

    prediction = (
        model.predict(future)[0]
    )

    # Enrolment cannot be negative
    return max(
        0.0,
        float(prediction)
    )


def holt_forecast(history):

    model = Holt(
        history["enrolment"],
        initialization_method="estimated"
    )

    fitted_model = model.fit(
        optimized=True
    )

    prediction = (
        fitted_model
        .forecast(1)
        .iloc[0]
    )

    # Enrolment cannot be negative
    return max(
        0.0,
        float(prediction)
    )


# --------------------------------------------------
# 6. WALK-FORWARD BENCHMARK
# --------------------------------------------------

all_predictions = []

failed_forecasts = []

institution_groups = df.groupby(
    "UNITID"
)

total_institutions = (
    df["UNITID"].nunique()
)

print("\n=====================================")
print("RUNNING BENCHMARK")
print("=====================================")

for count, (
    unitid,
    institution_df
) in enumerate(
    institution_groups,
    start=1
):

    institution_df = (
        institution_df
        .sort_values("year")
        .copy()
    )

    institution_name = (
        institution_df
        .iloc[-1]["INSTNM"]
    )

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

    for test_year in TEST_YEARS:

        history = series_df[
            series_df["year"]
            < test_year
        ].copy()

        actual_row = series_df[
            series_df["year"]
            == test_year
        ]

        if (
            history.empty
            or actual_row.empty
        ):
            continue

        actual = float(
            actual_row[
                "enrolment"
            ].iloc[0]
        )

        # ------------------------------
        # NAIVE
        # ------------------------------

        try:

            prediction = (
                naive_forecast(
                    history
                )
            )

            all_predictions.append({
                "UNITID": unitid,
                "INSTNM":
                    institution_name,
                "test_year":
                    test_year,
                "model":
                    "Naive Baseline",
                "actual":
                    actual,
                "prediction":
                    prediction
            })

        except Exception as error:

            failed_forecasts.append({
                "UNITID": unitid,
                "test_year":
                    test_year,
                "model":
                    "Naive Baseline",
                "error":
                    str(error)
            })

        # ------------------------------
        # LINEAR TREND
        # ------------------------------

        try:

            prediction = (
                linear_forecast(
                    history,
                    test_year
                )
            )

            all_predictions.append({
                "UNITID": unitid,
                "INSTNM":
                    institution_name,
                "test_year":
                    test_year,
                "model":
                    "Linear Trend",
                "actual":
                    actual,
                "prediction":
                    prediction
            })

        except Exception as error:

            failed_forecasts.append({
                "UNITID": unitid,
                "test_year":
                    test_year,
                "model":
                    "Linear Trend",
                "error":
                    str(error)
            })

        # ------------------------------
        # HOLT TREND
        # ------------------------------

        try:

            prediction = (
                holt_forecast(
                    history
                )
            )

            all_predictions.append({
                "UNITID": unitid,
                "INSTNM":
                    institution_name,
                "test_year":
                    test_year,
                "model":
                    "Holt Linear Trend",
                "actual":
                    actual,
                "prediction":
                    prediction
            })

        except Exception as error:

            failed_forecasts.append({
                "UNITID": unitid,
                "test_year":
                    test_year,
                "model":
                    "Holt Linear Trend",
                "error":
                    str(error)
            })

    # Progress message every 500 institutions
    if (
        count % 500 == 0
        or count == total_institutions
    ):

        print(
            f"Processed "
            f"{count:,} / "
            f"{total_institutions:,} "
            f"institutions"
        )


# --------------------------------------------------
# 7. CREATE PREDICTION DATAFRAME
# --------------------------------------------------

predictions_df = pd.DataFrame(
    all_predictions
)

print("\n=====================================")
print("PREDICTIONS CREATED")
print("=====================================")

print(
    f"Prediction rows: "
    f"{len(predictions_df):,}"
)

print(
    f"Failed forecasts: "
    f"{len(failed_forecasts):,}"
)


# --------------------------------------------------
# 8. CALCULATE ERRORS
# --------------------------------------------------

predictions_df["error"] = (
    predictions_df["actual"]
    - predictions_df["prediction"]
)

predictions_df[
    "absolute_error"
] = predictions_df[
    "error"
].abs()

predictions_df[
    "squared_error"
] = (
    predictions_df["error"]
    ** 2
)

predictions_df[
    "absolute_percentage_error"
] = np.where(
    predictions_df["actual"] != 0,

    (
        predictions_df[
            "absolute_error"
        ]
        /
        predictions_df["actual"]
    ) * 100,

    np.nan
)


# --------------------------------------------------
# 9. OVERALL MODEL METRICS
# --------------------------------------------------

def calculate_metrics(group):

    return pd.Series({

        "predictions":
            len(group),

        "mae":
            group[
                "absolute_error"
            ].mean(),

        "rmse":
            np.sqrt(
                group[
                    "squared_error"
                ].mean()
            ),

        "mape":
            group[
                "absolute_percentage_error"
            ].mean()

    })


overall_metrics = (
    predictions_df
    .groupby("model")
    .apply(
        calculate_metrics,
        include_groups=False
    )
    .reset_index()
    .sort_values("mae")
)


print("\n=====================================")
print("OVERALL MODEL PERFORMANCE")
print("=====================================")

print(overall_metrics)


# --------------------------------------------------
# 10. PERFORMANCE BY TEST YEAR
# --------------------------------------------------

year_metrics = (
    predictions_df
    .groupby(
        ["model", "test_year"]
    )
    .apply(
        calculate_metrics,
        include_groups=False
    )
    .reset_index()
)


print("\n=====================================")
print("PERFORMANCE BY TEST YEAR")
print("=====================================")

print(year_metrics)


# --------------------------------------------------
# 11. MERGE SIZE CATEGORIES
# --------------------------------------------------

predictions_df = (
    predictions_df
    .merge(
        institution_sizes[
            [
                "UNITID",
                "median_enrolment",
                "size_category"
            ]
        ],
        on="UNITID",
        how="left"
    )
)


# --------------------------------------------------
# 12. PERFORMANCE BY INSTITUTION SIZE
# --------------------------------------------------

size_metrics = (
    predictions_df
    .groupby(
        ["size_category", "model"]
    )
    .apply(
        calculate_metrics,
        include_groups=False
    )
    .reset_index()
    .sort_values(
        [
            "size_category",
            "mae"
        ]
    )
)


print("\n=====================================")
print("PERFORMANCE BY INSTITUTION SIZE")
print("=====================================")

print(size_metrics)


# --------------------------------------------------
# 13. METRICS FOR EACH INSTITUTION + MODEL
# --------------------------------------------------

institution_model_metrics = (
    predictions_df
    .groupby(
        [
            "UNITID",
            "INSTNM",
            "size_category",
            "model"
        ]
    )
    .apply(
        calculate_metrics,
        include_groups=False
    )
    .reset_index()
)


# --------------------------------------------------
# 14. FIND BEST MODEL PER INSTITUTION
# --------------------------------------------------

best_model_index = (
    institution_model_metrics
    .groupby("UNITID")["mae"]
    .idxmin()
)

best_models = (
    institution_model_metrics
    .loc[best_model_index]
    .reset_index(drop=True)
)


best_model_counts = (
    best_models["model"]
    .value_counts()
    .rename_axis("model")
    .reset_index(
        name="institution_count"
    )
)

best_model_counts[
    "percentage"
] = (
    best_model_counts[
        "institution_count"
    ]
    /
    best_models[
        "UNITID"
    ].nunique()
    * 100
)


print("\n=====================================")
print("BEST MODEL PER INSTITUTION")
print("=====================================")

print(best_model_counts)


# --------------------------------------------------
# 15. BEST MODEL BY SIZE CATEGORY
# --------------------------------------------------

best_model_by_size = (
    best_models
    .groupby(
        ["size_category", "model"]
    )
    .size()
    .reset_index(
        name="institution_count"
    )
)

best_model_by_size[
    "percentage_within_size"
] = (
    best_model_by_size[
        "institution_count"
    ]
    /
    best_model_by_size
    .groupby(
        "size_category"
    )[
        "institution_count"
    ]
    .transform("sum")
    * 100
)


print("\n=====================================")
print("BEST MODEL BY SIZE")
print("=====================================")

print(best_model_by_size)


# --------------------------------------------------
# 16. SAVE RESULTS
# --------------------------------------------------

predictions_df.to_csv(
    RESULTS_FOLDER
    / "benchmark_all_predictions.csv",
    index=False
)

overall_metrics.to_csv(
    RESULTS_FOLDER
    / "benchmark_overall_metrics.csv",
    index=False
)

year_metrics.to_csv(
    RESULTS_FOLDER
    / "benchmark_metrics_by_year.csv",
    index=False
)

size_metrics.to_csv(
    RESULTS_FOLDER
    / "benchmark_metrics_by_size.csv",
    index=False
)

institution_model_metrics.to_csv(
    RESULTS_FOLDER
    / "benchmark_institution_model_metrics.csv",
    index=False
)

best_models.to_csv(
    RESULTS_FOLDER
    / "benchmark_best_model_per_institution.csv",
    index=False
)

best_model_counts.to_csv(
    RESULTS_FOLDER
    / "benchmark_best_model_counts.csv",
    index=False
)

best_model_by_size.to_csv(
    RESULTS_FOLDER
    / "benchmark_best_model_by_size.csv",
    index=False
)

if failed_forecasts:

    pd.DataFrame(
        failed_forecasts
    ).to_csv(
        RESULTS_FOLDER
        / "benchmark_failed_forecasts.csv",
        index=False
    )


print("\n=====================================")
print("BENCHMARK COMPLETE")
print("=====================================")