
from pydantic import BaseModel, Field, ConfigDict


class CapacityAnalysisInput(BaseModel):
    """
    Validate the numbers supplied to
    the capacity-analysis tool.
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    forecast_enrolment: int = Field(ge=0)

    approved_capacity: int = Field(gt=0)


def analyse_capacity(
    forecast_enrolment: int,
    approved_capacity: int,
) -> dict:
    """
    Calculate enrolment-capacity metrics using
    deterministic Python arithmetic.
    """

    inputs = CapacityAnalysisInput(
        forecast_enrolment=forecast_enrolment,
        approved_capacity=approved_capacity,
    )

    forecast = inputs.forecast_enrolment
    capacity = inputs.approved_capacity

    capacity_gap = forecast - capacity

    utilisation_percent = (
        forecast / capacity
    ) * 100

    if capacity_gap > 0:
        status = "over_capacity"
        students_over_capacity = capacity_gap
        remaining_places = 0

    elif capacity_gap == 0:
        status = "at_capacity"
        students_over_capacity = 0
        remaining_places = 0

    else:
        status = "within_capacity"
        students_over_capacity = 0
        remaining_places = abs(capacity_gap)

    return {
        "forecast_enrolment": forecast,
        "approved_capacity": capacity,
        "capacity_gap": capacity_gap,
        "utilisation_percent": round(
            utilisation_percent, 2
        ),
        "students_over_capacity": students_over_capacity,
        "remaining_places": remaining_places,
        "status": status,
    }
