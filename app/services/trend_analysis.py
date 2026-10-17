import numpy as np
import pandas as pd


def analyse_enrolment_trend(
    data: pd.DataFrame
) -> dict:
    """
    Analyse historical enrolment trends.

    Expected columns:
        year
        enrolment

    Returns:
        summary statistics
        overall trend
        recent trend
        annual changes
    """

    # --------------------------------------------------
    # 1. BASIC SAFETY CHECKS
    # --------------------------------------------------

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
            "Data must contain "
            "'year' and 'enrolment' columns."
        )


    # --------------------------------------------------
    # 2. PREPARE DATA
    # --------------------------------------------------

    df = (
        data[
            ["year", "enrolment"]
        ]
        .copy()
        .sort_values("year")
        .reset_index(drop=True)
    )


    # --------------------------------------------------
    # 3. BASIC VALUES
    # --------------------------------------------------

    first_year = int(
        df.iloc[0]["year"]
    )

    last_year = int(
        df.iloc[-1]["year"]
    )

    first_enrolment = float(
        df.iloc[0]["enrolment"]
    )

    latest_enrolment = float(
        df.iloc[-1]["enrolment"]
    )


    # --------------------------------------------------
    # 4. OVERALL CHANGE
    # --------------------------------------------------

    absolute_change = (
        latest_enrolment
        - first_enrolment
    )

    if first_enrolment != 0:

        percentage_change = (
            absolute_change
            / first_enrolment
            * 100
        )

    else:

        percentage_change = None


    # --------------------------------------------------
    # 5. CAGR
    # --------------------------------------------------

    number_of_periods = (
        last_year
        - first_year
    )

    if (
        first_enrolment > 0
        and latest_enrolment >= 0
        and number_of_periods > 0
    ):

        cagr = (
            (
                latest_enrolment
                / first_enrolment
            )
            ** (
                1 / number_of_periods
            )
            - 1
        ) * 100

    else:

        cagr = None


    # --------------------------------------------------
    # 6. OVERALL DIRECTION
    # --------------------------------------------------

    if latest_enrolment > first_enrolment:

        overall_direction = "Increasing"

    elif latest_enrolment < first_enrolment:

        overall_direction = "Decreasing"

    else:

        overall_direction = "Stable"


    # --------------------------------------------------
    # 7. ANNUAL CHANGES
    # --------------------------------------------------

    annual_changes = df.copy()

    annual_changes[
        "absolute_change"
    ] = (
        annual_changes[
            "enrolment"
        ]
        .diff()
    )

    annual_changes[
        "percentage_change"
    ] = (
        annual_changes[
            "enrolment"
        ]
        .pct_change()
        * 100
    )

    # Replace infinite values that can occur
    # after a zero enrolment year.
    annual_changes[
        "percentage_change"
    ] = (
        annual_changes[
            "percentage_change"
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )


    # --------------------------------------------------
    # 8. RECENT TREND
    # --------------------------------------------------

    # Use the latest three observations when available.
    recent_df = df.tail(3)

    recent_start = float(
        recent_df.iloc[0]["enrolment"]
    )

    recent_end = float(
        recent_df.iloc[-1]["enrolment"]
    )

    if recent_end > recent_start:

        recent_direction = "Increasing"

    elif recent_end < recent_start:

        recent_direction = "Decreasing"

    else:

        recent_direction = "Stable"


    # --------------------------------------------------
    # 9. SUMMARY STATISTICS
    # --------------------------------------------------

    average_enrolment = float(
        df["enrolment"].mean()
    )

    median_enrolment = float(
        df["enrolment"].median()
    )

    minimum_enrolment = float(
        df["enrolment"].min()
    )

    maximum_enrolment = float(
        df["enrolment"].max()
    )


    # --------------------------------------------------
    # 10. AVERAGE ANNUAL ABSOLUTE CHANGE
    # --------------------------------------------------

    average_annual_change = float(
        annual_changes[
            "absolute_change"
        ]
        .dropna()
        .mean()
    )


    # --------------------------------------------------
    # 11. RETURN STRUCTURED RESULT
    # --------------------------------------------------

    return {

        "summary": {

            "first_year":
                first_year,

            "last_year":
                last_year,

            "first_enrolment":
                first_enrolment,

            "latest_enrolment":
                latest_enrolment,

            "average_enrolment":
                average_enrolment,

            "median_enrolment":
                median_enrolment,

            "minimum_enrolment":
                minimum_enrolment,

            "maximum_enrolment":
                maximum_enrolment,

            "absolute_change":
                float(
                    absolute_change
                ),

            "percentage_change":
                (
                    float(
                        percentage_change
                    )
                    if percentage_change
                    is not None
                    else None
                ),

            "cagr":
                (
                    float(cagr)
                    if cagr is not None
                    else None
                ),

            "average_annual_change":
                average_annual_change,

            "overall_direction":
                overall_direction,

            "recent_direction":
                recent_direction,
        },

        "annual_changes":
            annual_changes
    }