import pandas as pd
import streamlit as st

from app.services.data_validation import validate_enrolment_data
from app.services.trend_analysis import analyse_enrolment_trend
from app.services.forecasting import run_forecasting_pipeline


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Enrolment Forecasting Assistant",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title(
    "Institutional Enrolment Forecasting "
    "and Planning Assistant"
)

st.write(
    """
    Upload your institution's historical enrolment data
    to analyse trends and generate a next-year enrolment
    forecast.
    """
)

st.info(
    "The current version uses deterministic analytical "
    "and forecasting tools. AI-generated explanations "
    "will be added in a later stage."
)


# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload enrolment data",
    type=["csv", "xlsx"]
)


# --------------------------------------------------
# READ UPLOADED FILE
# --------------------------------------------------

if uploaded_file is not None:

    try:

        if uploaded_file.name.lower().endswith(
            ".csv"
        ):
            raw_data = pd.read_csv(
                uploaded_file
            )

        else:
            raw_data = pd.read_excel(
                uploaded_file
            )

    except Exception as error:

        st.error(
            f"Unable to read the uploaded file: "
            f"{error}"
        )

        st.stop()


    # --------------------------------------------------
    # DISPLAY ORIGINAL DATA
    # --------------------------------------------------

    st.subheader(
        "Uploaded Data"
    )

    st.dataframe(
        raw_data,
        use_container_width=True
    )


    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    st.subheader(
        "Data Validation"
    )

    validation_result = (
        validate_enrolment_data(
            raw_data
        )
    )


    # --------------------------------------------------
    # SHOW VALIDATION ERRORS
    # --------------------------------------------------

    if not validation_result["valid"]:

        st.error(
            "The uploaded data cannot currently "
            "be used for analysis."
        )

        for error in (
            validation_result["errors"]
        ):
            st.error(error)

        st.stop()


    # --------------------------------------------------
    # VALID DATA
    # --------------------------------------------------

    st.success(
        "The uploaded data passed the required "
        "validation checks."
    )

    cleaned_data = (
        validation_result[
            "cleaned_data"
        ]
    )


    # --------------------------------------------------
    # SHOW WARNINGS
    # --------------------------------------------------

    for warning in (
        validation_result["warnings"]
    ):
        st.warning(warning)


    # --------------------------------------------------
    # VALIDATION SUMMARY
    # --------------------------------------------------

    validation_summary = (
        validation_result["summary"]
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Historical Years",
        validation_summary[
            "number_of_years"
        ]
    )

    col2.metric(
        "First Year",
        validation_summary[
            "first_year"
        ]
    )

    col3.metric(
        "Latest Year",
        validation_summary[
            "last_year"
        ]
    )

    col4.metric(
        "Latest Enrolment",
        f"{validation_summary['latest_enrolment']:,.0f}"
    )


    # --------------------------------------------------
    # HISTORICAL ENROLMENT CHART
    # --------------------------------------------------

    st.subheader(
        "Historical Enrolment"
    )

    chart_data = (
        cleaned_data[
            ["year", "enrolment"]
        ]
        .set_index("year")
    )

    st.line_chart(
        chart_data
    )


    # --------------------------------------------------
    # TREND ANALYSIS
    # --------------------------------------------------

    st.subheader(
        "Historical Trend Analysis"
    )

    try:

        trend_result = (
            analyse_enrolment_trend(
                cleaned_data
            )
        )

        trend = trend_result[
            "summary"
        ]

        t1, t2, t3, t4 = (
            st.columns(4)
        )

        t1.metric(
            "Overall Trend",
            trend[
                "overall_direction"
            ]
        )

        t2.metric(
            "Recent Trend",
            trend[
                "recent_direction"
            ]
        )

        t3.metric(
            "Average Enrolment",
            f"{trend['average_enrolment']:,.0f}"
        )

        if (
            trend["percentage_change"]
            is not None
        ):

            percentage_text = (
                f"{trend['percentage_change']:.2f}%"
            )

        else:

            percentage_text = "N/A"

        t4.metric(
            "Overall Change",
            percentage_text
        )


        # More detail
        with st.expander(
            "View detailed trend statistics"
        ):

            st.write(
                f"**Minimum enrolment:** "
                f"{trend['minimum_enrolment']:,.0f}"
            )

            st.write(
                f"**Maximum enrolment:** "
                f"{trend['maximum_enrolment']:,.0f}"
            )

            st.write(
                f"**Median enrolment:** "
                f"{trend['median_enrolment']:,.0f}"
            )

            st.write(
                f"**Absolute change:** "
                f"{trend['absolute_change']:,.0f}"
            )

            if trend["cagr"] is not None:

                st.write(
                    f"**CAGR:** "
                    f"{trend['cagr']:.2f}%"
                )

            st.write(
                f"**Average annual change:** "
                f"{trend['average_annual_change']:,.2f}"
            )


        # Annual change table
        with st.expander(
            "View annual changes"
        ):

            st.dataframe(
                trend_result[
                    "annual_changes"
                ],
                use_container_width=True
            )

    except Exception as error:

        st.error(
            f"Trend analysis could not be completed: "
            f"{error}"
        )


    # --------------------------------------------------
    # FORECASTING
    # --------------------------------------------------

    st.subheader(
        "Enrolment Forecast"
    )

    try:

        forecast_result = (
            run_forecasting_pipeline(
                cleaned_data
            )
        )

        f1, f2, f3 = (
            st.columns(3)
        )

        f1.metric(
            "Forecast Year",
            forecast_result[
                "forecast_year"
            ]
        )

        f2.metric(
            "Predicted Enrolment",
            f"{forecast_result['predicted_enrolment']:,.0f}"
        )

        f3.metric(
            "Selected Model",
            forecast_result[
                "selected_model"
            ]
        )


        # --------------------------------------------------
        # PERFORMANCE METRICS
        # --------------------------------------------------

        st.markdown(
            "### Backtest Performance"
        )

        m1, m2, m3, m4 = (
            st.columns(4)
        )

        m1.metric(
            "MAE",
            f"{forecast_result['mae']:.2f}"
        )

        m2.metric(
            "RMSE",
            f"{forecast_result['rmse']:.2f}"
        )

        if (
            forecast_result["mape"]
            is not None
        ):

            mape_text = (
                f"{forecast_result['mape']:.2f}%"
            )

        else:

            mape_text = "N/A"

        m3.metric(
            "MAPE",
            mape_text
        )

        m4.metric(
            "Backtest Evidence",
            forecast_result[
                "backtest_evidence"
            ]
        )


        # --------------------------------------------------
        # MODEL COMPARISON
        # --------------------------------------------------

        with st.expander(
            "Compare forecasting models"
        ):

            st.dataframe(
                forecast_result[
                    "model_metrics"
                ],
                use_container_width=True
            )


        # --------------------------------------------------
        # MODEL SELECTION INFORMATION
        # --------------------------------------------------

        if forecast_result[
            "simplicity_rule_applied"
        ]:

            st.info(
                "The simplest model within 5% of "
                "the best MAE was selected."
            )


        # --------------------------------------------------
        # CAPACITY CHECK
        # --------------------------------------------------

        if "capacity" in cleaned_data.columns:

            valid_capacity = (
                cleaned_data["capacity"]
                .dropna()
            )

            if not valid_capacity.empty:

                latest_capacity = float(
                    valid_capacity.iloc[-1]
                )

                predicted_enrolment = (
                    forecast_result[
                        "predicted_enrolment"
                    ]
                )

                capacity_gap = (
                    predicted_enrolment
                    - latest_capacity
                )

                st.markdown(
                    "### Capacity Check"
                )

                c1, c2 = (
                    st.columns(2)
                )

                c1.metric(
                    "Latest Capacity",
                    f"{latest_capacity:,.0f}"
                )

                c2.metric(
                    "Forecast Capacity Gap",
                    f"{capacity_gap:,.0f}"
                )

                if capacity_gap > 0:

                    st.warning(
                        "Forecast enrolment exceeds "
                        "the supplied capacity."
                    )

                else:

                    st.success(
                        "Forecast enrolment remains "
                        "within the supplied capacity."
                    )


        # --------------------------------------------------
        # HUMAN REVIEW
        # --------------------------------------------------

        st.warning(
            "Forecasts are planning estimates and "
            "should be reviewed by an authorised "
            "institutional decision-maker before "
            "staffing, budget, or capacity decisions "
            "are made."
        )


    except Exception as error:

        st.warning(
            "Historical analysis is available, but "
            "forecasting could not be completed."
        )

        st.error(str(error))


# --------------------------------------------------
# NO FILE YET
# --------------------------------------------------

else:

    st.write(
        "Upload a CSV or Excel file to begin."
    )

    st.caption(
        "Required columns: year, enrolment. "
        "Optional column: capacity."
    )