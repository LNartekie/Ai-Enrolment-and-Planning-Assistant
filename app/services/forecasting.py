import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tools.sm_exceptions import ConvergenceWarning


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MIN_TRAIN_YEARS = 6

MODEL_TOLERANCE = 0.05

MODEL_COMPLEXITY = {
    "Naive Baseline": 1,
    "Linear Trend": 2,
    "Holt Linear Trend": 3,
}

warnings.simplefilter(
    "ignore",
    ConvergenceWarning
)


# --------------------------------------------------
# 1. PREPARE FORECASTING DATA
# --------------------------------------------------

def prepare_forecasting_data(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepare validated enrolment data for forecasting.

    Required columns:
        year
        enrolment

    Forecasting requires:
        - at least 7 observations
        - consecutive annual observations
    """

    if data is None or data.empty:
        raise ValueError(
            "No enrolment data was provided."
        )

    required_columns = {
        "year",
        "enrolment"
    }

    if not required_columns.issubset(
        data.columns
    ):
        raise ValueError(
            "Forecasting data must contain "
            "'year' and 'enrolment' columns."
        )

    df = (
        data[
            ["year", "enrolment"]
        ]
        .copy()
        .sort_values("year")
        .reset_index(drop=True)
    )

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce"
    )

    df["enrolment"] = pd.to_numeric(
        df["enrolment"],
        errors="coerce"
    )

    if (
        df["year"].isna().any()
        or df["enrolment"].isna().any()
    ):
        raise ValueError(
            "Forecasting data contains "
            "missing or invalid values."
        )

    if (
        df["enrolment"] < 0
    ).any():
        raise ValueError(
            "Enrolment cannot be negative."
        )

    if (
        df["year"]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Duplicate years are not allowed "
            "for forecasting."
        )

    if len(df) < (
        MIN_TRAIN_YEARS + 1
    ):
        raise ValueError(
            f"Forecasting requires at least "
            f"{MIN_TRAIN_YEARS + 1} years of "
            "historical data."
        )

    # Check that the annual series is continuous.
    year_differences = (
        df["year"]
        .diff()
        .dropna()
    )

    if not (
        year_differences == 1
    ).all():
        raise ValueError(
            "Forecasting requires consecutive "
            "annual observations. Missing years "
            "must be reviewed before forecasting."
        )

    df["year"] = (
        df["year"]
        .astype(int)
    )

    return df


# --------------------------------------------------
# 2. NAIVE MODEL
# --------------------------------------------------

def naive_forecast(
    history: pd.DataFrame,
    forecast_year: int
) -> float:
    """
    Predict next year's enrolment using
    the most recent observed value.
    """

    prediction = (
        history["enrolment"]
        .iloc[-1]
    )

    return max(
        0.0,
        float(prediction)
    )


# --------------------------------------------------
# 3. LINEAR TREND MODEL
# --------------------------------------------------

def linear_trend_forecast(
    history: pd.DataFrame,
    forecast_year: int
) -> float:
    """
    Fit a linear regression between
    year and enrolment.
    """

    X = history[
        ["year"]
    ]

    y = history[
        "enrolment"
    ]

    model = LinearRegression()

    model.fit(
        X,
        y
    )

    future = pd.DataFrame({
        "year": [
            forecast_year
        ]
    })

    prediction = (
        model.predict(
            future
        )[0]
    )

    return max(
        0.0,
        float(prediction)
    )


# --------------------------------------------------
# 4. HOLT LINEAR TREND MODEL
# --------------------------------------------------

def holt_forecast(
    history: pd.DataFrame,
    forecast_year: int
) -> float:
    """
    Forecast one annual period ahead using
    Holt's linear trend method.
    """

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

    return max(
        0.0,
        float(prediction)
    )


# --------------------------------------------------
# 5. MODEL REGISTRY
# --------------------------------------------------

MODEL_FUNCTIONS = {
    "Naive Baseline":
        naive_forecast,

    "Linear Trend":
        linear_trend_forecast,

    "Holt Linear Trend":
        holt_forecast,
}


# --------------------------------------------------
# 6. CALCULATE METRICS
# --------------------------------------------------

def calculate_metrics(
    results: pd.DataFrame
) -> dict:
    """
    Calculate MAE, RMSE and MAPE.
    """

    errors = (
        results["actual"]
        - results["prediction"]
    )

    absolute_errors = (
        errors.abs()
    )

    squared_errors = (
        errors ** 2
    )

    mae = float(
        absolute_errors.mean()
    )

    rmse = float(
        np.sqrt(
            squared_errors.mean()
        )
    )

    non_zero_mask = (
        results["actual"] != 0
    )

    if non_zero_mask.any():

        mape = float(
            (
                absolute_errors[
                    non_zero_mask
                ]
                /
                results.loc[
                    non_zero_mask,
                    "actual"
                ]
            )
            .mean()
            * 100
        )

    else:
        mape = None

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }


# --------------------------------------------------
# 7. WALK-FORWARD BACKTEST
# --------------------------------------------------

def backtest_models(
    data: pd.DataFrame
):
    """
    Perform expanding-window walk-forward
    backtesting for all approved models.
    """

    df = prepare_forecasting_data(
        data
    )

    prediction_records = []

    failure_records = []

    # Example with 13 years:
    #
    # 2001-2006 -> predict 2007
    # 2001-2007 -> predict 2008
    # ...
    # 2001-2012 -> predict 2013

    for test_index in range(
        MIN_TRAIN_YEARS,
        len(df)
    ):

        history = (
            df.iloc[
                :test_index
            ]
            .copy()
        )

        actual_row = (
            df.iloc[
                test_index
            ]
        )

        test_year = int(
            actual_row["year"]
        )

        actual = float(
            actual_row[
                "enrolment"
            ]
        )

        for (
            model_name,
            model_function
        ) in MODEL_FUNCTIONS.items():

            try:

                prediction = (
                    model_function(
                        history,
                        test_year
                    )
                )

                prediction_records.append({
                    "model":
                        model_name,

                    "test_year":
                        test_year,

                    "actual":
                        actual,

                    "prediction":
                        prediction,

                    "training_years":
                        len(history),
                })

            except Exception as error:

                failure_records.append({
                    "model":
                        model_name,

                    "test_year":
                        test_year,

                    "error":
                        str(error),
                })

    predictions_df = pd.DataFrame(
        prediction_records
    )

    if predictions_df.empty:
        raise ValueError(
            "No backtest forecasts could "
            "be generated."
        )

    return (
        predictions_df,
        failure_records
    )


# --------------------------------------------------
# 8. SUMMARISE EACH MODEL
# --------------------------------------------------

def summarise_model_performance(
    predictions_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate metrics for each forecasting model.
    """

    records = []

    for (
        model_name,
        group
    ) in predictions_df.groupby(
        "model"
    ):

        metrics = calculate_metrics(
            group
        )

        records.append({
            "model":
                model_name,

            "backtest_predictions":
                len(group),

            "mae":
                metrics["mae"],

            "rmse":
                metrics["rmse"],

            "mape":
                metrics["mape"],
        })

    metrics_df = (
        pd.DataFrame(
            records
        )
        .sort_values("mae")
        .reset_index(drop=True)
    )

    return metrics_df


# --------------------------------------------------
# 9. SELECT MODEL
# --------------------------------------------------

def select_forecasting_model(
    metrics_df: pd.DataFrame
) -> dict:
    """
    Select the forecasting model using MAE.

    If another simpler model is within 5%
    of the lowest MAE, prefer the simpler model.
    """

    if metrics_df.empty:
        raise ValueError(
            "No model performance metrics "
            "were provided."
        )

    comparison = (
        metrics_df
        .copy()
    )

    raw_best = (
        comparison
        .sort_values("mae")
        .iloc[0]
    )

    raw_best_model = (
        raw_best["model"]
    )

    best_mae = float(
        raw_best["mae"]
    )

    tolerance_limit = (
        best_mae
        * (
            1
            + MODEL_TOLERANCE
        )
    )

    candidates = (
        comparison[
            comparison["mae"]
            <= tolerance_limit
        ]
        .copy()
    )

    candidates[
        "complexity_rank"
    ] = (
        candidates["model"]
        .map(
            MODEL_COMPLEXITY
        )
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

    return {
        "raw_best_model":
            raw_best_model,

        "raw_best_mae":
            best_mae,

        "selected_model":
            selected["model"],

        "selected_mae":
            float(
                selected["mae"]
            ),

        "selected_rmse":
            float(
                selected["rmse"]
            ),

        "selected_mape":
            (
                float(
                    selected["mape"]
                )
                if pd.notna(
                    selected["mape"]
                )
                else None
            ),

        "simplicity_rule_applied":
            (
                selected["model"]
                != raw_best_model
            ),
    }


# --------------------------------------------------
# 10. FINAL FUTURE FORECAST
# --------------------------------------------------

def forecast_next_year(
    data: pd.DataFrame,
    model_name: str
) -> dict:
    """
    Refit the selected model using all
    available history and forecast the
    next calendar year.
    """

    df = prepare_forecasting_data(
        data
    )

    if model_name not in (
        MODEL_FUNCTIONS
    ):
        raise ValueError(
            f"Unknown forecasting model: "
            f"{model_name}"
        )

    latest_year = int(
        df["year"].max()
    )

    forecast_year = (
        latest_year + 1
    )

    model_function = (
        MODEL_FUNCTIONS[
            model_name
        ]
    )

    prediction = (
        model_function(
            df,
            forecast_year
        )
    )

    return {
        "forecast_year":
            forecast_year,

        "predicted_enrolment":
            float(prediction),

        "selected_model":
            model_name,
    }


# --------------------------------------------------
# 11. COMPLETE FORECASTING PIPELINE
# --------------------------------------------------

def run_forecasting_pipeline(
    data: pd.DataFrame
) -> dict:
    """
    Complete forecasting workflow:

    1. Prepare data
    2. Backtest models
    3. Calculate model metrics
    4. Apply model-selection rule
    5. Refit selected model
    6. Forecast next year
    """

    df = prepare_forecasting_data(
        data
    )

    (
        predictions_df,
        failures
    ) = backtest_models(
        df
    )

    metrics_df = (
        summarise_model_performance(
            predictions_df
        )
    )

    selection = (
        select_forecasting_model(
            metrics_df
        )
    )

    final_forecast = (
        forecast_next_year(
            df,
            selection[
                "selected_model"
            ]
        )
    )

    number_of_backtests = (
        predictions_df[
            "test_year"
        ]
        .nunique()
    )

    # Give the application some context
    # about how much historical evaluation
    # evidence was available.
    if number_of_backtests >= 5:

        evidence_strength = "Strong"

    elif number_of_backtests >= 3:

        evidence_strength = "Moderate"

    else:

        evidence_strength = "Limited"

    return {
        "forecast_year":
            final_forecast[
                "forecast_year"
            ],

        "predicted_enrolment":
            final_forecast[
                "predicted_enrolment"
            ],

        "selected_model":
            selection[
                "selected_model"
            ],

        "raw_best_model":
            selection[
                "raw_best_model"
            ],

        "mae":
            selection[
                "selected_mae"
            ],

        "rmse":
            selection[
                "selected_rmse"
            ],

        "mape":
            selection[
                "selected_mape"
            ],

        "simplicity_rule_applied":
            selection[
                "simplicity_rule_applied"
            ],

        "backtest_folds":
            int(
                number_of_backtests
            ),

        "backtest_evidence":
            evidence_strength,

        "model_metrics":
            metrics_df,

        "backtest_predictions":
            predictions_df,

        "model_failures":
            failures,
    }