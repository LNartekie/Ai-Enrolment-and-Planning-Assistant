import hashlib

import pandas as pd
import streamlit as st

from app.services.data_validation import (
    validate_enrolment_data,
)

from app.services.trend_analysis import (
    analyse_enrolment_trend,
)

from app.services.forecasting import (
    run_forecasting_pipeline,
)

from app.services.llm_service import (
    check_ollama_connection,
    generate_forecast_explanation,
    generate_followup_response,
)

from app.services.structured_output import (
    generate_structured_forecast_explanation,
)

from app.services.document_processing import (
    process_uploaded_document,
)

from app.services.structured_rag_service import (
    answer_document_question_structured,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Enrolment Forecasting Assistant",
    page_icon="📊",
    layout="wide",
)


# --------------------------------------------------
# INITIALISE SESSION STATE
# --------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


if "forecast_explanation" not in st.session_state:
    st.session_state.forecast_explanation = None

if "structured_explanation" not in st.session_state:
    st.session_state.structured_explanation = None

if "dataset_signature" not in st.session_state:
    st.session_state.dataset_signature = None

if "knowledge_chunks" not in st.session_state:
    st.session_state.knowledge_chunks = []

if "knowledge_upload_version" not in st.session_state:
    st.session_state.knowledge_upload_version = 0


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header(
        "Session Controls"
    )

    st.write(
        "Conversation memory is stored only in the "
        "current Streamlit session."
    )

    if st.button(
        "New Session",
        use_container_width=True,
    ):

        st.session_state.chat_history = []

        st.session_state.forecast_explanation = None

        st.session_state.dataset_signature = None

        st.session_state.structured_explanation = None

        st.session_state.knowledge_chunks = []

        st.session_state.knowledge_upload_version += 1

        st.rerun()


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
    to analyse trends, compare forecasting methods,
    generate a next-year forecast, and receive
    AI-assisted explanations.
    """
)

st.info(
    "Numerical calculations are performed by "
    "deterministic Python tools. The language model "
    "is used for explanation and conversation."
)


# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload enrolment data",
    type=[
        "csv",
        "xlsx",
    ],
)


# --------------------------------------------------
# NO FILE
# --------------------------------------------------

if uploaded_file is None:

    st.write(
        "Upload a CSV or Excel file to begin."
    )

    st.caption(
        "Required columns: year, enrolment. "
        "Optional column: capacity."
    )

    st.stop()


# --------------------------------------------------
# CREATE DATASET SIGNATURE
# --------------------------------------------------

file_bytes = (
    uploaded_file.getvalue()
)

dataset_signature = (
    hashlib.sha256(
        file_bytes
    ).hexdigest()
)


# --------------------------------------------------
# CLEAR OLD MEMORY IF DATASET CHANGES
# --------------------------------------------------

if (
    st.session_state.dataset_signature
    != dataset_signature
):

    st.session_state.dataset_signature = (
        dataset_signature
    )

    st.session_state.chat_history = []

    st.session_state.forecast_explanation = None

    st.session_state.structured_explanation = None

    st.session_state.knowledge_chunks = [] 


# --------------------------------------------------
# READ FILE
# --------------------------------------------------

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
        "Unable to read the uploaded file: "
        f"{error}"
    )

    st.stop()


# --------------------------------------------------
# DISPLAY UPLOADED DATA
# --------------------------------------------------

st.subheader(
    "Uploaded Data"
)

st.dataframe(
    raw_data,
    use_container_width=True,
)


# --------------------------------------------------
# DATA VALIDATION
# --------------------------------------------------

st.subheader(
    "Data Validation"
)

validation_result = (
    validate_enrolment_data(
        raw_data
    )
)


if not validation_result["valid"]:

    st.error(
        "The uploaded data cannot currently "
        "be used for analysis."
    )

    for error in (
        validation_result["errors"]
    ):

        st.error(
            error
        )

    st.stop()


st.success(
    "The uploaded data passed the required "
    "validation checks."
)


cleaned_data = (
    validation_result[
        "cleaned_data"
    ]
)


for warning in (
    validation_result["warnings"]
):

    st.warning(
        warning
    )


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
    ],
)


col2.metric(
    "First Year",
    validation_summary[
        "first_year"
    ],
)


col3.metric(
    "Latest Year",
    validation_summary[
        "last_year"
    ],
)


col4.metric(
    "Latest Enrolment",
    (
        f"{validation_summary['latest_enrolment']:,.0f}"
    ),
)


# --------------------------------------------------
# HISTORICAL CHART
# --------------------------------------------------

st.subheader(
    "Historical Enrolment"
)


chart_data = (
    cleaned_data[
        [
            "year",
            "enrolment",
        ]
    ]
    .set_index(
        "year"
    )
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

    trend = (
        trend_result[
            "summary"
        ]
    )


except Exception as error:

    st.error(
        "Trend analysis could not be completed: "
        f"{error}"
    )

    st.stop()


t1, t2, t3, t4 = (
    st.columns(4)
)


t1.metric(
    "Overall Trend",
    trend[
        "overall_direction"
    ],
)


t2.metric(
    "Recent Trend",
    trend[
        "recent_direction"
    ],
)


t3.metric(
    "Average Enrolment",
    (
        f"{trend['average_enrolment']:,.0f}"
    ),
)


if (
    trend[
        "percentage_change"
    ]
    is not None
):

    percentage_text = (
        f"{trend['percentage_change']:.2f}%"
    )

else:

    percentage_text = "N/A"


t4.metric(
    "Overall Change",
    percentage_text,
)


with st.expander(
    "View detailed trend statistics"
):

    st.write(
        "**Minimum enrolment:** "
        f"{trend['minimum_enrolment']:,.0f}"
    )

    st.write(
        "**Maximum enrolment:** "
        f"{trend['maximum_enrolment']:,.0f}"
    )

    st.write(
        "**Median enrolment:** "
        f"{trend['median_enrolment']:,.0f}"
    )

    st.write(
        "**Absolute change:** "
        f"{trend['absolute_change']:,.0f}"
    )

    if trend["cagr"] is not None:

        st.write(
            "**CAGR:** "
            f"{trend['cagr']:.2f}%"
        )

    st.write(
        "**Average annual change:** "
        f"{trend['average_annual_change']:,.2f}"
    )


with st.expander(
    "View annual changes"
):

    st.dataframe(
        trend_result[
            "annual_changes"
        ],
        use_container_width=True,
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


except Exception as error:

    st.warning(
        "Historical analysis is available, but "
        "forecasting could not be completed."
    )

    st.error(
        str(error)
    )

    st.stop()


# --------------------------------------------------
# FORECAST SUMMARY
# --------------------------------------------------

f1, f2, f3 = (
    st.columns(3)
)


f1.metric(
    "Forecast Year",
    forecast_result[
        "forecast_year"
    ],
)


f2.metric(
    "Predicted Enrolment",
    (
        f"{forecast_result['predicted_enrolment']:,.0f}"
    ),
)


f3.metric(
    "Selected Model",
    forecast_result[
        "selected_model"
    ],
)


# --------------------------------------------------
# BACKTEST PERFORMANCE
# --------------------------------------------------

st.markdown(
    "### Backtest Performance"
)


m1, m2, m3, m4 = (
    st.columns(4)
)


m1.metric(
    "MAE",
    f"{forecast_result['mae']:.2f}",
)


m2.metric(
    "RMSE",
    f"{forecast_result['rmse']:.2f}",
)


if forecast_result["mape"] is not None:

    mape_text = (
        f"{forecast_result['mape']:.2f}%"
    )

else:

    mape_text = "N/A"


m3.metric(
    "MAPE",
    mape_text,
)


m4.metric(
    "Backtest Evidence",
    forecast_result[
        "backtest_evidence"
    ],
)


with st.expander(
    "Compare forecasting models"
):

    st.dataframe(
        forecast_result[
            "model_metrics"
        ],
        use_container_width=True,
    )


if forecast_result[
    "simplicity_rule_applied"
]:

    st.info(
        "The simplest model within 5% "
        "of the lowest MAE was selected."
    )


# --------------------------------------------------
# CAPACITY
# --------------------------------------------------

latest_capacity = None


if "capacity" in cleaned_data.columns:

    valid_capacity = (
        cleaned_data[
            "capacity"
        ]
        .dropna()
    )


    if not valid_capacity.empty:

        latest_capacity = float(
            valid_capacity.iloc[-1]
        )

        capacity_gap = (
            forecast_result[
                "predicted_enrolment"
            ]
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
            f"{latest_capacity:,.0f}",
        )


        c2.metric(
            "Forecast Capacity Gap",
            f"{capacity_gap:,.0f}",
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
# INSTITUTIONAL KNOWLEDGE DOCUMENTS
# --------------------------------------------------

st.divider()

st.subheader(
    "Institutional Knowledge Documents"
)

st.write(
    "Upload planning documents, policies or reports "
    "to prepare institution-specific knowledge "
    "for the AI assistant."
)

st.caption(
    "Supported formats: PDF, DOCX and TXT. "
    "Maximum file size: 10 MB per document. "
    "Maximum documents: 5."
)


# Give the uploader a different identity when
# the dataset changes or a new session begins.

document_widget_key = (
    f"institution_docs_"
    f"{st.session_state.knowledge_upload_version}_"
    f"{dataset_signature}"
)


uploaded_documents = st.file_uploader(
    "Upload institutional documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    key=document_widget_key,
)


# Rebuild the active session's chunks from
# the currently uploaded documents.

st.session_state.knowledge_chunks = []


if uploaded_documents:

    if len(uploaded_documents) > 5:

        st.error(
            "Please upload no more than five documents."
        )

    else:

        seen_sources = set()

        for uploaded_document in uploaded_documents:

            result = process_uploaded_document(
                uploaded_document
            )

            if not result["success"]:

                st.error(
                    f"Could not process "
                    f"{uploaded_document.name}"
                )

                for error in result["errors"]:
                    st.write(error)

                continue


            # Avoid including identical file contents twice.

            if result["source_id"] in seen_sources:

                st.warning(
                    f"{uploaded_document.name} is a "
                    "duplicate of a document already uploaded."
                )

                continue

            seen_sources.add(
                result["source_id"]
            )


            # Keep successfully processed chunks
            # in the current Streamlit session.

            st.session_state.knowledge_chunks.extend(
                result["chunks"]
            )


            st.success(
                f"Processed: {uploaded_document.name}"
            )

            st.write(
                f"Chunks created: "
                f"{result['summary']['total_chunks']}"
            )


            for warning in result["warnings"]:
                st.warning(warning)


            # Show metadata without automatically
            # exposing the full document text.

            with st.expander(
                f"View metadata: {uploaded_document.name}"
            ):

                st.json({
                    "source_name": result["source_name"],
                    "source_id": result["source_id"],
                    "file_type": result["file_type"],
                    "total_chunks": result["summary"]["total_chunks"],
                    "units_with_text": result["summary"]["units_with_text"],
                })


if st.session_state.knowledge_chunks:

    st.info(
        f"Total knowledge chunks prepared: "
        f"{len(st.session_state.knowledge_chunks)}"
    )

else:

    st.info(
        "No usable institutional knowledge is currently "
        "available. Document-based answers will require "
        "uploaded and retrievable evidence."
    )

# --------------------------------------------------
# STRUCTURED AI FORECAST EXPLANATION
# --------------------------------------------------

st.divider()

st.subheader(
    "AI-Assisted Forecast Explanation"
)

st.write(
    "The explanation is generated by Qwen3 and "
    "must pass JSON-schema, Pydantic, and semantic "
    "validation before it is displayed."
)


if st.button(
    "Generate Validated AI Explanation",
    type="primary",
):

    if not check_ollama_connection():

        st.error(
            "Ollama is not available. "
            "Make sure Ollama is running."
        )

    else:

        with st.spinner(
            "Generating and validating explanation..."
        ):

            try:

                structured_explanation = (
                    generate_structured_forecast_explanation(
                        trend_summary=trend,
                        forecast_result=forecast_result,
                        capacity=latest_capacity,
                    )
                )


                st.session_state[
                    "structured_explanation"
                ] = structured_explanation


            except Exception as error:

                st.session_state[
                    "structured_explanation"
                ] = None

                st.error(
                    "The AI response could not pass "
                    "the required validation checks."
                )

                st.error(
                    str(error)
                )


# --------------------------------------------------
# DISPLAY VALIDATED EXPLANATION
# --------------------------------------------------

if (
    st.session_state.structured_explanation
    is not None
):

    explanation = (
        st.session_state.structured_explanation
    )


    st.success(
        "AI response passed structural and "
        "semantic validation."
    )


    st.markdown(
        "### Summary"
    )

    st.write(
        explanation.summary
    )


    st.markdown(
        "### Historical Trend"
    )

    st.write(
        explanation.historical_trend
    )


    st.markdown(
        "### Forecast"
    )

    st.write(
        explanation.forecast_statement
    )


    st.markdown(
        "### Historical Backtest"
    )

    st.write(
        explanation.backtest_interpretation
    )


    st.markdown(
        "### Capacity"
    )

    st.write(
        explanation.capacity_statement
    )


    st.metric(
        "Validated Capacity Status",
        explanation.capacity_status,
    )


    st.markdown(
        "### Limitations"
    )


    for limitation in (
        explanation.limitations
    ):

        st.write(
            f"- {limitation}"
        )


    if explanation.human_review_required:

        st.warning(
            "Human review is required before "
            "institutional decisions are made."
        )


# --------------------------------------------------
# MULTI-TURN CONVERSATION
# --------------------------------------------------

st.divider()

st.subheader(
    "Ask Follow-Up Questions"
)


st.caption(
    "Conversation history is retained within "
    "this Streamlit session and is cleared when "
    "you start a new session or upload different data."
)


# --------------------------------------------------
# DISPLAY CONVERSATION HISTORY
# --------------------------------------------------

for message in (
    st.session_state.chat_history
):

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

user_question = st.chat_input(
    "Ask a question about the enrolment analysis..."
)


if user_question:

    # Store user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_question,
    })


    # Display user message
    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_question
        )


    # Check Ollama
    if not check_ollama_connection():

        assistant_response = (
            "The AI service is currently unavailable. "
            "Please ensure Ollama is running."
        )


    else:

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Thinking..."
            ):

                try:

                    assistant_response = (
                        generate_followup_response(
                            trend_summary=trend,
                            forecast_result=forecast_result,

                            # Do not include the newly
                            # appended current user message
                            # twice.
                            conversation_history=(
                                st.session_state
                                .chat_history[:-1]
                            ),

                            user_message=user_question,

                            capacity=latest_capacity,
                        )
                    )


                except Exception as error:

                    assistant_response = (
                        "I could not answer the question: "
                        f"{error}"
                    )


            st.markdown(
                assistant_response
            )


    # Store assistant response
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": assistant_response,
    })


# --------------------------------------------------
# HUMAN REVIEW
# --------------------------------------------------

st.divider()

st.warning(
    "Forecasts and AI explanations are planning "
    "support only. Final institutional decisions "
    "remain subject to authorised human review."
)