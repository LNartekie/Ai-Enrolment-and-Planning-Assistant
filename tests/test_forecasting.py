import pandas as pd
import pytest

from app.services.forecasting import (
    naive_forecast,
    linear_trend_forecast,
    prepare_forecasting_data,
    run_forecasting_pipeline,
)


# --------------------------------------------------
# SAMPLE DATA
# --------------------------------------------------

def sample_data():

    return pd.DataFrame({
        "year": [
            2017,
            2018,
            2019,
            2020,
            2021,
            2022,
            2023,
            2024,
            2025,
        ],

        "enrolment": [
            1000,
            1050,
            1100,
            1080,
            1150,
            1200,
            1250,
            1300,
            1350,
        ]
    })


# --------------------------------------------------
# TEST 1: DATA PREPARATION
# --------------------------------------------------

def test_prepare_forecasting_data():

    df = sample_data()

    result = (
        prepare_forecasting_data(
            df
        )
    )

    assert len(result) == 9

    assert (
        result.iloc[0]["year"]
        == 2017
    )

    assert (
        result.iloc[-1]["year"]
        == 2025
    )


# --------------------------------------------------
# TEST 2: NAIVE MODEL
# --------------------------------------------------

def test_naive_forecast():

    df = sample_data()

    prediction = (
        naive_forecast(
            df,
            2026
        )
    )

    assert prediction == 1350


# --------------------------------------------------
# TEST 3: LINEAR MODEL NON-NEGATIVE
# --------------------------------------------------

def test_linear_forecast_non_negative():

    df = sample_data()

    prediction = (
        linear_trend_forecast(
            df,
            2026
        )
    )

    assert prediction >= 0


# --------------------------------------------------
# TEST 4: COMPLETE PIPELINE
# --------------------------------------------------

def test_forecasting_pipeline():

    df = sample_data()

    result = (
        run_forecasting_pipeline(
            df
        )
    )

    assert (
        result["forecast_year"]
        == 2026
    )

    assert (
        result[
            "predicted_enrolment"
        ]
        >= 0
    )

    assert (
        result["selected_model"]
        in [
            "Naive Baseline",
            "Linear Trend",
            "Holt Linear Trend",
        ]
    )

    assert (
        result["backtest_folds"]
        == 3
    )

    assert (
        result["backtest_evidence"]
        == "Moderate"
    )


# --------------------------------------------------
# TEST 5: TOO LITTLE HISTORY
# --------------------------------------------------

def test_insufficient_history():

    df = pd.DataFrame({
        "year": [
            2021,
            2022,
            2023,
            2024,
            2025,
            2026,
        ],

        "enrolment": [
            1000,
            1050,
            1100,
            1150,
            1200,
            1250,
        ]
    })

    with pytest.raises(
        ValueError
    ):
        run_forecasting_pipeline(
            df
        )


# --------------------------------------------------
# TEST 6: MISSING YEAR
# --------------------------------------------------

def test_missing_year_rejected():

    df = pd.DataFrame({
        "year": [
            2017,
            2018,
            2019,
            2021,
            2022,
            2023,
            2024,
        ],

        "enrolment": [
            1000,
            1050,
            1100,
            1200,
            1250,
            1300,
            1350,
        ]
    })

    with pytest.raises(
        ValueError,
        match="consecutive"
    ):
        run_forecasting_pipeline(
            df
        )


# --------------------------------------------------
# TEST 7: NEGATIVE ENROLMENT
# --------------------------------------------------

def test_negative_enrolment_rejected():

    df = sample_data()

    df.loc[
        3,
        "enrolment"
    ] = -100

    with pytest.raises(
        ValueError,
        match="negative"
    ):
        run_forecasting_pipeline(
            df
        )


# --------------------------------------------------
# TEST 8: FINAL FORECAST IS NEXT YEAR
# --------------------------------------------------

def test_forecast_year_is_next_year():

    df = sample_data()

    result = (
        run_forecasting_pipeline(
            df
        )
    )

    assert (
        result["forecast_year"]
        == (
            df["year"].max()
            + 1
        )
    )