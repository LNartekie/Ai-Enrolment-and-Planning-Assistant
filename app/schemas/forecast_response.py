from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


# --------------------------------------------------
# STRUCTURED FORECAST EXPLANATION
# --------------------------------------------------

class ForecastExplanation(BaseModel):
    """
    Validated structure for an AI-generated
    enrolment forecast explanation.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


    summary: str = Field(
        min_length=10,
        max_length=500,
        description=(
            "A concise administrator-facing summary "
            "of the forecast."
        ),
    )


    historical_trend: str = Field(
        min_length=5,
        max_length=500,
        description=(
            "Explanation of the supplied historical "
            "enrolment direction."
        ),
    )


    forecast_statement: str = Field(
        min_length=5,
        max_length=500,
        description=(
            "Explanation of the supplied next-year "
            "forecast."
        ),
    )


    backtest_interpretation: str = Field(
        min_length=5,
        max_length=700,
        description=(
            "Interpretation of the supplied MAE, "
            "RMSE, and MAPE without guaranteeing "
            "future accuracy."
        ),
    )


    capacity_status: Literal[
        "Exceeded",
        "Within capacity",
        "Unavailable",
    ] = Field(
        description=(
            "Capacity status supplied by the "
            "deterministic Python layer."
        ),
    )


    capacity_statement: str = Field(
        min_length=5,
        max_length=500,
        description=(
            "Explanation of the supplied capacity "
            "information without prescribing actions."
        ),
    )


    limitations: list[str] = Field(
        min_length=1,
        max_length=5,
        description=(
            "Important limitations or uncertainties "
            "that apply to the forecast."
        ),
    )


    human_review_required: Literal[True] = Field(
        description=(
            "Must always be true because final "
            "institutional decisions require "
            "authorised human review."
        ),
    )