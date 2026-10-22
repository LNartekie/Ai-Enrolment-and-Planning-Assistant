from app.schemas.forecast_response import (
    ForecastExplanation,
)

from app.services.structured_validation import (
    validate_forecast_semantics,
)


# --------------------------------------------------
# SAMPLE FORECAST RESULT
# --------------------------------------------------

FORECAST_RESULT = {
    "predicted_enrolment": 1832,
}


# --------------------------------------------------
# VALID RESPONSE
# --------------------------------------------------

def test_valid_semantic_response():

    response = ForecastExplanation(
        summary=(
            "The forecast indicates enrolment "
            "above the supplied capacity."
        ),

        historical_trend=(
            "Historical enrolment is increasing."
        ),

        forecast_statement=(
            "The next-year forecast is a planning "
            "estimate rather than a guarantee."
        ),

        backtest_interpretation=(
            "The supplied MAE, RMSE and MAPE "
            "describe historical backtest error "
            "and do not guarantee future accuracy."
        ),

        capacity_status="Exceeded",

        capacity_statement=(
            "The forecast is above the supplied "
            "institutional capacity."
        ),

        limitations=[
            (
                "Future conditions may differ "
                "from historical patterns."
            ),
        ],

        human_review_required=True,
    )


    errors = validate_forecast_semantics(
        response=response,
        forecast_result=FORECAST_RESULT,
        capacity=1800,
    )


    assert errors == []


# --------------------------------------------------
# WRONG CAPACITY STATUS
# --------------------------------------------------

def test_wrong_capacity_status_is_detected():

    response = ForecastExplanation(
        summary=(
            "The forecast provides information "
            "for institutional planning."
        ),

        historical_trend=(
            "Historical enrolment is increasing."
        ),

        forecast_statement=(
            "The forecast is a planning estimate."
        ),

        backtest_interpretation=(
            "Historical error metrics do not "
            "guarantee future performance."
        ),

        capacity_status="Within capacity",

        capacity_statement=(
            "Capacity information was supplied."
        ),

        limitations=[
            "Future conditions may change."
        ],

        human_review_required=True,
    )


    errors = validate_forecast_semantics(
        response=response,
        forecast_result=FORECAST_RESULT,
        capacity=1800,
    )


    assert len(errors) > 0


# --------------------------------------------------
# UNSUPPORTED ACCURACY LABEL
# --------------------------------------------------

def test_unsupported_accuracy_label_is_detected():

    response = ForecastExplanation(
        summary=(
            "The forecast provides information "
            "for institutional planning."
        ),

        historical_trend=(
            "Historical enrolment is increasing."
        ),

        forecast_statement=(
            "The forecast is a planning estimate."
        ),

        backtest_interpretation=(
            "The model has moderate accuracy."
        ),

        capacity_status="Exceeded",

        capacity_statement=(
            "The forecast exceeds capacity."
        ),

        limitations=[
            "Future conditions may change."
        ],

        human_review_required=True,
    )


    errors = validate_forecast_semantics(
        response=response,
        forecast_result=FORECAST_RESULT,
        capacity=1800,
    )


    assert len(errors) > 0


# --------------------------------------------------
# MODEL VALIDATION OVERCLAIM
# --------------------------------------------------

def test_validation_overclaim_is_detected():

    response = ForecastExplanation(
        summary=(
            "The forecast provides information "
            "for institutional planning."
        ),

        historical_trend=(
            "Historical enrolment is increasing."
        ),

        forecast_statement=(
            "The forecast is a planning estimate."
        ),

        backtest_interpretation=(
            "The model was validated using "
            "historical folds."
        ),

        capacity_status="Exceeded",

        capacity_statement=(
            "The forecast exceeds capacity."
        ),

        limitations=[
            "Future conditions may change."
        ],

        human_review_required=True,
    )


    errors = validate_forecast_semantics(
        response=response,
        forecast_result=FORECAST_RESULT,
        capacity=1800,
    )


    assert len(errors) > 0