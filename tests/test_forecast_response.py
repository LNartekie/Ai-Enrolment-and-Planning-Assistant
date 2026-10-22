import pytest

from pydantic import ValidationError

from app.schemas.forecast_response import (
    ForecastExplanation,
)


# --------------------------------------------------
# VALID STRUCTURED RESPONSE
# --------------------------------------------------

def test_valid_forecast_explanation():

    response = ForecastExplanation(
        summary=(
            "The forecast indicates a small "
            "capacity gap for the next year."
        ),

        historical_trend=(
            "Historical enrolment is increasing."
        ),

        forecast_statement=(
            "The supplied forecast predicts "
            "1832 students in 2026."
        ),

        backtest_interpretation=(
            "The supplied historical backtest "
            "metrics describe past forecasting "
            "error and do not guarantee future "
            "performance."
        ),

        capacity_status="Exceeded",

        capacity_statement=(
            "The supplied forecast is above the "
            "reported institutional capacity."
        ),

        limitations=[
            (
                "The forecast is based on "
                "historical enrolment patterns."
            ),
            (
                "Future conditions may differ "
                "from historical patterns."
            ),
        ],

        human_review_required=True,
    )

    assert (
        response.capacity_status
        == "Exceeded"
    )

    assert (
        response.human_review_required
        is True
    )


# --------------------------------------------------
# INVALID CAPACITY STATUS
# --------------------------------------------------

def test_invalid_capacity_status():

    with pytest.raises(
        ValidationError
    ):

        ForecastExplanation(
            summary=(
                "The forecast provides planning "
                "information for the institution."
            ),

            historical_trend=(
                "Historical enrolment is increasing."
            ),

            forecast_statement=(
                "A next-year forecast is available."
            ),

            backtest_interpretation=(
                "Historical errors do not guarantee "
                "future forecasting accuracy."
            ),

            capacity_status="Probably exceeded",

            capacity_statement=(
                "Capacity information is available."
            ),

            limitations=[
                "Future conditions may change."
            ],

            human_review_required=True,
        )


# --------------------------------------------------
# HUMAN REVIEW MUST BE TRUE
# --------------------------------------------------

def test_human_review_cannot_be_false():

    with pytest.raises(
        ValidationError
    ):

        ForecastExplanation(
            summary=(
                "The forecast provides planning "
                "information for the institution."
            ),

            historical_trend=(
                "Historical enrolment is increasing."
            ),

            forecast_statement=(
                "A next-year forecast is available."
            ),

            backtest_interpretation=(
                "Historical errors do not guarantee "
                "future forecasting accuracy."
            ),

            capacity_status="Exceeded",

            capacity_statement=(
                "The forecast exceeds capacity."
            ),

            limitations=[
                "Future conditions may change."
            ],

            human_review_required=False,
        )


# --------------------------------------------------
# EXTRA FIELDS ARE REJECTED
# --------------------------------------------------

def test_extra_field_is_rejected():

    with pytest.raises(
        ValidationError
    ):

        ForecastExplanation(
            summary=(
                "The forecast provides planning "
                "information for the institution."
            ),

            historical_trend=(
                "Historical enrolment is increasing."
            ),

            forecast_statement=(
                "A next-year forecast is available."
            ),

            backtest_interpretation=(
                "Historical errors do not guarantee "
                "future forecasting accuracy."
            ),

            capacity_status="Exceeded",

            capacity_statement=(
                "The forecast exceeds capacity."
            ),

            limitations=[
                "Future conditions may change."
            ],

            human_review_required=True,

            invented_recommendation=(
                "Build new classrooms."
            ),
        )