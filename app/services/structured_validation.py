from app.schemas.forecast_response import (
    ForecastExplanation,
)


# --------------------------------------------------
# SEMANTIC VALIDATION
# --------------------------------------------------

def validate_forecast_semantics(
    response: ForecastExplanation,
    forecast_result: dict,
    capacity: float | None = None,
) -> list[str]:
    """
    Check a structurally valid LLM response against
    deterministic analytical results.

    Returns a list of semantic validation errors.
    An empty list means the response passed.
    """

    errors = []


    # --------------------------------------------------
    # 1. DETERMINE EXPECTED CAPACITY STATUS
    # --------------------------------------------------

    predicted_enrolment = float(
        forecast_result[
            "predicted_enrolment"
        ]
    )


    if capacity is None:

        expected_capacity_status = (
            "Unavailable"
        )

    else:

        capacity_gap = (
            predicted_enrolment
            - capacity
        )

        if capacity_gap > 0:

            expected_capacity_status = (
                "Exceeded"
            )

        else:

            expected_capacity_status = (
                "Within capacity"
            )


    # --------------------------------------------------
    # 2. CHECK CAPACITY STATUS
    # --------------------------------------------------

    if (
        response.capacity_status
        != expected_capacity_status
    ):

        errors.append(
            "The AI capacity status does not match "
            "the deterministic capacity calculation."
        )


    # --------------------------------------------------
    # 3. COMBINE NARRATIVE TEXT
    # --------------------------------------------------

    narrative_text = " ".join([
        response.summary,
        response.historical_trend,
        response.forecast_statement,
        response.backtest_interpretation,
        response.capacity_statement,
        *response.limitations,
    ]).lower()


    # --------------------------------------------------
    # 4. REJECT UNSUPPORTED ACCURACY LABELS
    # --------------------------------------------------

    unsupported_accuracy_phrases = [
        "moderate error",
        "moderate accuracy",
        "high accuracy",
        "low accuracy",
        "good accuracy",
        "poor accuracy",
    ]


    for phrase in unsupported_accuracy_phrases:

        if phrase in narrative_text:

            errors.append(
                "The AI introduced an unsupported "
                f"accuracy classification: '{phrase}'."
            )


    # --------------------------------------------------
    # 5. REJECT VALIDATION OVERCLAIMS
    # --------------------------------------------------

    validation_overclaims = [
        "model was validated",
        "forecast was validated",
        "validated forecast",
        "proven accurate",
        "guaranteed accurate",
    ]


    for phrase in validation_overclaims:

        if phrase in narrative_text:

            errors.append(
                "The AI made an unsupported model "
                f"validation claim: '{phrase}'."
            )


    # --------------------------------------------------
    # 6. CHECK HUMAN REVIEW
    # --------------------------------------------------

    if response.human_review_required is not True:

        errors.append(
            "Human review must remain required."
        )


    return errors