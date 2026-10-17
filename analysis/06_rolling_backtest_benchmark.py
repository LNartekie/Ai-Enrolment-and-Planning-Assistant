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

MIN_TRAIN_YEARS = 6

TEST_YEARS = list(
    range(2007, 2014)
)

# If a simpler model performs within 5% of the best
# MAE, prefer the simpler model.
MODEL_TOLERANCE = 0.05

MODEL_COMPLEXITY = {
    "Naive Baseline": 1,
    "Linear Trend": 2,
    "Holt Linear Trend": 3
}

warnings.simplefilter(
    "ignore",
    ConvergenceWarning
)


# --------------------------------------------------
# 3. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

df = (
    df.sort_values(
        ["UNITID", "year"]
    )
    .reset_index(drop=True)
)

print("\n=====================================")
print("ROLLING BENCHMARK DATA")
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

print(
    f"Backtest years: "
    f"{TEST_YEARS}"
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

q1 = (
    institution_sizes[
        "median_enrolment"
    ]
    .quantile(0.25)
)

q3 = (
    institution_sizes[
        "median_enrolment"
    ]
    .quantile(0.75)
)


def classify_size(value):

    if value <= q1:
        return "Small"

    elif value <= q3:
        return "Medium"

    return "Large"


institution_sizes[
    "size_category"
] = (
    institution_sizes[
        "median_enrolment"
    ]
    .apply(classify_size)
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

    future_year = pd.DataFrame({
        "year": [forecast_year]
    })

    prediction = (
        model.predict(
            future_year
        )[0]
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
# 6. RUN 7-FOLD WALK-FORWARD BENCHMARK
# --------------------------------------------------

all_predictions = []

failed_forecasts = []

institution_groups = df.groupby(
    "UNITID"
)

total_institutions = (
    df["UNITID"]
    .nunique()
)


print("\n=====================================")
print("RUNNING 7-FOLD BENCHMARK")
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

        history = (
            series_df[
                series_df["year"]
                < test_year
            ]
            .copy()
        )

        actual_row = (
            series_df[
                series_df["year"]
                == test_year
            ]
        )

        # Check enough training history exists
        if len(history) < MIN_TRAIN_YEARS:
            continue

        if actual_row.empty:
            continue

        actual = float(
            actual_row[
                "enrolment"
            ].iloc[0]
        )

        training_start = int(
            history["year"].min()
        )

        training_end = int(
            history["year"].max()
        )

        number_training_years = (
            len(history)
        )

        # ------------------------------
        # NAIVE BASELINE
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
                "train_start_year":
                    training_start,
                "train_end_year":
                    training_end,
                "training_years":
                    number_training_years,
                "model":
                    "Naive Baseline",
                "actual":
                    actual,
                "prediction":
                    prediction
            })

        except Exception as error:

            failed_forecasts.append({
                "UNITID":
                    unitid,
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
                "train_start_year":
                    training_start,
                "train_end_year":
                    training_end,
                "training_years":
                    number_training_years,
                "model":
                    "Linear Trend",
                "actual":
                    actual,
                "prediction":
                    prediction
            })

        except Exception as error:

            failed_forecasts.append({
                "UNITID":
                    unitid,
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
                "train_start_year":
                    training_start,
                "train_end_year":
                    training_end,
                "training_years":
                    number_training_years,
                "model":
                    "Holt Linear Trend",
                "actual":
                    actual,
                "prediction":
                    prediction
            })

        except Exception as error:

            failed_forecasts.append({
                "UNITID":
                    unitid,
                "test_year":
                    test_year,
                "model":
                    "Holt Linear Trend",
                "error":
                    str(error)
            })

    # Progress report
    if (
        count % 250 == 0
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
# 8. CALCULATE FORECAST ERRORS
# --------------------------------------------------

predictions_df["error"] = (
    predictions_df["actual"]
    - predictions_df["prediction"]
)

predictions_df[
    "absolute_error"
] = (
    predictions_df["error"]
    .abs()
)

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
    )
    * 100,

    np.nan
)


# --------------------------------------------------
# 9. METRIC HELPER FUNCTION
# --------------------------------------------------

def calculate_metrics(group):

    return {
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
    }


# --------------------------------------------------
# 10. OVERALL MODEL PERFORMANCE
# --------------------------------------------------

overall_records = []

for model_name, group in (
    predictions_df.groupby("model")
):

    metrics = calculate_metrics(
        group
    )

    metrics["model"] = model_name

    overall_records.append(
        metrics
    )


overall_metrics = (
    pd.DataFrame(
        overall_records
    )
    [
        [
            "model",
            "predictions",
            "mae",
            "rmse",
            "mape"
        ]
    ]
    .sort_values("mae")
)


print("\n=====================================")
print("7-FOLD OVERALL MODEL PERFORMANCE")
print("=====================================")

print(overall_metrics)


# --------------------------------------------------
# 11. PERFORMANCE BY TEST YEAR
# --------------------------------------------------

year_records = []

for (
    model_name,
    test_year
), group in predictions_df.groupby(
    ["model", "test_year"]
):

    metrics = calculate_metrics(
        group
    )

    metrics["model"] = model_name

    metrics["test_year"] = (
        test_year
    )

    year_records.append(
        metrics
    )


year_metrics = pd.DataFrame(
    year_records
)

year_metrics = year_metrics[
    [
        "model",
        "test_year",
        "predictions",
        "mae",
        "rmse",
        "mape"
    ]
]


# --------------------------------------------------
# 12. ADD SIZE CATEGORIES
# --------------------------------------------------

predictions_df = (
    predictions_df
    .merge(
        institution_sizes,
        on="UNITID",
        how="left"
    )
)


# --------------------------------------------------
# 13. PERFORMANCE BY SIZE
# --------------------------------------------------

size_records = []

for (
    size_category,
    model_name
), group in predictions_df.groupby(
    [
        "size_category",
        "model"
    ]
):

    metrics = calculate_metrics(
        group
    )

    metrics[
        "size_category"
    ] = size_category

    metrics["model"] = (
        model_name
    )

    size_records.append(
        metrics
    )


size_metrics = pd.DataFrame(
    size_records
)

size_metrics = (
    size_metrics[
        [
            "size_category",
            "model",
            "predictions",
            "mae",
            "rmse",
            "mape"
        ]
    ]
    .sort_values(
        [
            "size_category",
            "mae"
        ]
    )
)


print("\n=====================================")
print("7-FOLD PERFORMANCE BY SIZE")
print("=====================================")

print(size_metrics)


# --------------------------------------------------
# 14. MODEL METRICS FOR EACH INSTITUTION
# --------------------------------------------------

institution_records = []

for (
    unitid,
    institution_name,
    size_category,
    model_name
), group in predictions_df.groupby(
    [
        "UNITID",
        "INSTNM",
        "size_category",
        "model"
    ]
):

    metrics = calculate_metrics(
        group
    )

    metrics["UNITID"] = unitid
    metrics["INSTNM"] = (
        institution_name
    )
    metrics[
        "size_category"
    ] = size_category
    metrics["model"] = model_name

    institution_records.append(
        metrics
    )


institution_model_metrics = (
    pd.DataFrame(
        institution_records
    )
)

institution_model_metrics = (
    institution_model_metrics[
        [
            "UNITID",
            "INSTNM",
            "size_category",
            "model",
            "predictions",
            "mae",
            "rmse",
            "mape"
        ]
    ]
)


# --------------------------------------------------
# 15. SELECT BEST MODEL PER INSTITUTION
#     WITH SIMPLICITY TIE-BREAK RULE
# --------------------------------------------------

selected_models = []

for unitid, group in (
    institution_model_metrics
    .groupby("UNITID")
):

    group = group.copy()

    # Raw lowest MAE
    raw_best_row = (
        group
        .sort_values("mae")
        .iloc[0]
    )

    best_mae = float(
        raw_best_row["mae"]
    )

    raw_best_model = (
        raw_best_row["model"]
    )

    # Models considered practically similar
    tolerance_limit = (
        best_mae
        * (1 + MODEL_TOLERANCE)
    )

    candidates = (
        group[
            group["mae"]
            <= tolerance_limit
        ]
        .copy()
    )

    # Prefer simpler model where performance
    # is within the tolerance
    candidates[
        "complexity_rank"
    ] = (
        candidates["model"]
        .map(MODEL_COMPLEXITY)
    )

    selected = (
        candidates
        .sort_values(
            [
                "complexity_rank",
                "mae"
            ]
        )
        .iloc[0]
    )

    selected_models.append({
        "UNITID":
            unitid,

        "INSTNM":
            selected["INSTNM"],

        "size_category":
            selected[
                "size_category"
            ],

        "raw_best_model":
            raw_best_model,

        "raw_best_mae":
            best_mae,

        "selected_model":
            selected["model"],

        "selected_mae":
            selected["mae"],

        "selected_rmse":
            selected["rmse"],

        "selected_mape":
            selected["mape"],

        "within_5_percent_rule":
            selected["model"]
            != raw_best_model
    })


selected_models_df = pd.DataFrame(
    selected_models
)


# --------------------------------------------------
# 16. SELECTED MODEL COUNTS
# --------------------------------------------------

selected_model_counts = (
    selected_models_df[
        "selected_model"
    ]
    .value_counts()
    .rename_axis(
        "model"
    )
    .reset_index(
        name="institution_count"
    )
)

selected_model_counts[
    "percentage"
] = (
    selected_model_counts[
        "institution_count"
    ]
    /
    selected_models_df[
        "UNITID"
    ].nunique()
    * 100
)


print("\n=====================================")
print("SELECTED MODEL PER INSTITUTION")
print("=====================================")

print(selected_model_counts)


# --------------------------------------------------
# 17. RAW BEST MODEL COUNTS
# --------------------------------------------------

raw_best_model_counts = (
    selected_models_df[
        "raw_best_model"
    ]
    .value_counts()
    .rename_axis("model")
    .reset_index(
        name="institution_count"
    )
)

raw_best_model_counts[
    "percentage"
] = (
    raw_best_model_counts[
        "institution_count"
    ]
    /
    selected_models_df[
        "UNITID"
    ].nunique()
    * 100
)


print("\n=====================================")
print("RAW BEST MODEL BY MAE")
print("=====================================")

print(raw_best_model_counts)


# --------------------------------------------------
# 18. MODEL SELECTION BY SIZE
# --------------------------------------------------

selection_by_size = (
    selected_models_df
    .groupby(
        [
            "size_category",
            "selected_model"
        ]
    )
    .size()
    .reset_index(
        name="institution_count"
    )
)

selection_by_size[
    "percentage_within_size"
] = (
    selection_by_size[
        "institution_count"
    ]
    /
    selection_by_size
    .groupby(
        "size_category"
    )[
        "institution_count"
    ]
    .transform("sum")
    * 100
)


print("\n=====================================")
print("SELECTED MODEL BY SIZE")
print("=====================================")

print(selection_by_size)


# --------------------------------------------------
# 19. TIE-BREAK SUMMARY
# --------------------------------------------------

tie_break_count = (
    selected_models_df[
        "within_5_percent_rule"
    ]
    .sum()
)

print("\n=====================================")
print("SIMPLICITY TIE-BREAK SUMMARY")
print("=====================================")

print(
    "Institutions where a simpler "
    "model was selected because it "
    "was within 5% of the lowest MAE:"
)

print(
    f"{tie_break_count:,} / "
    f"{len(selected_models_df):,}"
)


# --------------------------------------------------
# 20. SAVE OUTPUTS
# --------------------------------------------------

predictions_df.to_csv(
    RESULTS_FOLDER
    / "rolling_7fold_all_predictions.csv",
    index=False
)

overall_metrics.to_csv(
    RESULTS_FOLDER
    / "rolling_7fold_overall_metrics.csv",
    index=False
)

year_metrics.to_csv(
    RESULTS_FOLDER
    / "rolling_7fold_metrics_by_year.csv",
    index=False
)

size_metrics.to_csv(
    RESULTS_FOLDER
    / "rolling_7fold_metrics_by_size.csv",
    index=False
)

institution_model_metrics.to_csv(
    RESULTS_FOLDER
    / "rolling_7fold_institution_model_metrics.csv",
    index=False
)

selected_models_df.to_csv(
    RESULTS_FOLDER
    / "rolling_7fold_selected_models.csv",
    index=False
)

selected_model_counts.to_csv(
    RESULTS_FOLDER
    / "rolling_7fold_selected_model_counts.csv",
    index=False
)

raw_best_model_counts.to_csv(
    RESULTS_FOLDER
    / "rolling_7fold_raw_best_model_counts.csv",
    index=False
)

selection_by_size.to_csv(
    RESULTS_FOLDER
    / "rolling_7fold_selected_model_by_size.csv",
    index=False
)

if failed_forecasts:

    pd.DataFrame(
        failed_forecasts
    ).to_csv(
        RESULTS_FOLDER
        / "rolling_7fold_failed_forecasts.csv",
        index=False
    )


print("\n=====================================")
print("7-FOLD BENCHMARK COMPLETE")
print("=====================================")