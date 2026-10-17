import pandas as pd
import pytest

from app.services.trend_analysis import (
    analyse_enrolment_trend
)


# --------------------------------------------------
# TEST 1: INCREASING TREND
# --------------------------------------------------

def test_increasing_trend():

    df = pd.DataFrame({
        "year": [
            2018,
            2019,
            2020,
            2021,
            2022,
            2023
        ],

        "enrolment": [
            1000,
            1100,
            1200,
            1300,
            1400,
            1500
        ]
    })

    result = analyse_enrolment_trend(
        df
    )

    summary = result["summary"]

    assert (
        summary[
            "overall_direction"
        ]
        == "Increasing"
    )

    assert (
        summary[
            "recent_direction"
        ]
        == "Increasing"
    )

    assert (
        summary[
            "absolute_change"
        ]
        == 500
    )


# --------------------------------------------------
# TEST 2: DECREASING TREND
# --------------------------------------------------

def test_decreasing_trend():

    df = pd.DataFrame({
        "year": [
            2018,
            2019,
            2020,
            2021,
            2022,
            2023
        ],

        "enrolment": [
            1500,
            1400,
            1300,
            1200,
            1100,
            1000
        ]
    })

    result = analyse_enrolment_trend(
        df
    )

    assert (
        result[
            "summary"
        ][
            "overall_direction"
        ]
        == "Decreasing"
    )


# --------------------------------------------------
# TEST 3: STABLE TREND
# --------------------------------------------------

def test_stable_trend():

    df = pd.DataFrame({
        "year": [
            2018,
            2019,
            2020,
            2021,
            2022,
            2023
        ],

        "enrolment": [
            1000,
            1000,
            1000,
            1000,
            1000,
            1000
        ]
    })

    result = analyse_enrolment_trend(
        df
    )

    summary = result["summary"]

    assert (
        summary[
            "overall_direction"
        ]
        == "Stable"
    )

    assert (
        summary[
            "recent_direction"
        ]
        == "Stable"
    )


# --------------------------------------------------
# TEST 4: OVERALL UP BUT RECENT DOWN
# --------------------------------------------------

def test_recent_decline():

    df = pd.DataFrame({
        "year": [
            2018,
            2019,
            2020,
            2021,
            2022,
            2023
        ],

        "enrolment": [
            1000,
            1300,
            1600,
            1800,
            1700,
            1500
        ]
    })

    result = analyse_enrolment_trend(
        df
    )

    summary = result["summary"]

    assert (
        summary[
            "overall_direction"
        ]
        == "Increasing"
    )

    assert (
        summary[
            "recent_direction"
        ]
        == "Decreasing"
    )


# --------------------------------------------------
# TEST 5: ZERO START VALUE
# --------------------------------------------------

def test_zero_start_enrolment():

    df = pd.DataFrame({
        "year": [
            2018,
            2019,
            2020,
            2021,
            2022,
            2023
        ],

        "enrolment": [
            0,
            100,
            200,
            300,
            400,
            500
        ]
    })

    result = analyse_enrolment_trend(
        df
    )

    summary = result["summary"]

    assert (
        summary[
            "percentage_change"
        ]
        is None
    )

    assert (
        summary[
            "cagr"
        ]
        is None
    )


# --------------------------------------------------
# TEST 6: EMPTY DATA
# --------------------------------------------------

def test_empty_data():

    df = pd.DataFrame()

    with pytest.raises(
        ValueError
    ):
        analyse_enrolment_trend(
            df
        )