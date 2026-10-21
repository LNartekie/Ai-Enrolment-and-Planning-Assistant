SYSTEM_PROMPT = """
You are an institutional enrolment planning assistant.

Your role is to explain analytical and forecasting
results supplied by deterministic Python tools.

Important rules:

1. Do not invent enrolment values, percentages,
   forecast metrics, capacity figures, trends,
   or causes.

2. Use only the numerical and categorical information
   supplied in the user message.

3. Do not recalculate, alter, or replace supplied values.

4. Clearly distinguish historical facts from forecasts.

5. Forecasts are planning estimates, not guaranteed
   future outcomes.

6. Do not infer student demand, institutional growth,
   staffing needs, budget needs, facility requirements,
   or other causes unless supporting evidence is
   explicitly supplied.

7. "Backtest evidence" describes the amount of
   historical evaluation available. It does not prove
   that a forecast is correct or certain.

8. MAE, RMSE, and MAPE describe historical backtesting
   performance. They do not guarantee future accuracy.

9. When capacity is exceeded, explain that the
   administrator should review the capacity gap.
   Do not prescribe staffing, budgeting, admissions,
   facility, or programme changes unless relevant
   evidence has been supplied.

10. Institutional decisions must remain subject
    to human review.

11. If information is unavailable, explicitly state
    that it is unavailable rather than guessing.

Use clear professional language suitable for
a university administrator.
"""


def build_verified_context(
    trend_summary: dict,
    forecast_result: dict,
    capacity: float | None = None
) -> str:
    """
    Convert deterministic analytical results into
    verified context for the LLM.
    """

    latest_enrolment = (
        trend_summary["latest_enrolment"]
    )

    overall_direction = (
        trend_summary["overall_direction"]
    )

    recent_direction = (
        trend_summary["recent_direction"]
    )

    forecast_year = (
        forecast_result["forecast_year"]
    )

    predicted_enrolment = (
        forecast_result["predicted_enrolment"]
    )

    selected_model = (
        forecast_result["selected_model"]
    )

    mae = forecast_result["mae"]
    rmse = forecast_result["rmse"]
    mape = forecast_result["mape"]

    backtest_folds = (
        forecast_result["backtest_folds"]
    )

    backtest_evidence = (
        forecast_result["backtest_evidence"]
    )


    if mape is not None:

        mape_text = (
            f"{mape:.2f}%"
        )

    else:

        mape_text = "Unavailable"


    # --------------------------------------------------
    # DETERMINISTIC CAPACITY STATUS
    # --------------------------------------------------

    if capacity is not None:

        capacity_gap = (
            predicted_enrolment
            - capacity
        )

        if capacity_gap > 0:

            capacity_status = "Exceeded"

        else:

            capacity_status = (
                "Within capacity"
            )

        capacity_text = f"""
Supplied capacity:
{capacity:.0f}

Forecast capacity gap:
{capacity_gap:.0f} students

Capacity status:
{capacity_status}
"""

    else:

        capacity_text = """
No capacity information was supplied.
"""


    context = f"""
VERIFIED ANALYTICAL RESULTS

HISTORICAL ANALYSIS

Latest historical enrolment:
{latest_enrolment:.0f}

Overall historical direction:
{overall_direction}

Recent historical direction:
{recent_direction}


FORECAST

Forecast year:
{forecast_year}

Predicted enrolment:
{predicted_enrolment:.0f}

Selected forecasting model:
{selected_model}


HISTORICAL BACKTEST PERFORMANCE

MAE:
{mae:.2f} students

RMSE:
{rmse:.2f} students

MAPE:
{mape_text}

Number of backtest folds:
{backtest_folds}

Backtest evidence label:
{backtest_evidence}

Important interpretation:
The backtest evidence label refers only to the
amount of historical evaluation available.
It does not prove that the future forecast is
correct or certain.


CAPACITY INFORMATION

{capacity_text}
"""

    return context


def build_forecast_prompt(
    trend_summary: dict,
    forecast_result: dict,
    capacity: float | None = None,
    variant: str = "A"
) -> str:
    """
    Build one of three prompt variations using
    exactly the same verified analytical context.
    """

    context = build_verified_context(
        trend_summary,
        forecast_result,
        capacity
    )


    # --------------------------------------------------
    # VARIANT A
    # Structured administrator explanation
    # --------------------------------------------------

    if variant == "A":

        instructions = """
TASK

Explain these results to a university administrator.

Use these sections:

1. Historical Trend
2. Forecast
3. Historical Backtest Performance
4. Capacity Implication
5. Human Review

Keep the explanation professional and easy to
understand.

Do not introduce any numerical value or factual
claim that does not appear in the verified results.

Do not claim that the forecast has been validated
or guaranteed.
"""


    # --------------------------------------------------
    # VARIANT B
    # Evidence-first explanation
    # --------------------------------------------------

    elif variant == "B":

        instructions = """
TASK

Prepare an evidence-first enrolment planning brief.

Begin by clearly distinguishing:

KNOWN HISTORICAL FACTS
FORECAST ESTIMATE
HISTORICAL MODEL PERFORMANCE
CAPACITY STATUS

After presenting those facts, provide a short
planning interpretation.

Clearly explain that historical backtest performance
does not guarantee future forecasting accuracy.

Do not infer causes for enrolment changes.

Do not recommend specific staffing, budgeting,
facility, admissions, or programme actions unless
supporting evidence has been provided.

End with a clear human-review statement.
"""


    # --------------------------------------------------
    # VARIANT C
    # Concise decision-support explanation
    # --------------------------------------------------

    elif variant == "C":

        instructions = """
TASK

Write a concise decision-support explanation for
a university administrator.

Start with a one-sentence summary.

Then briefly explain:

- the historical direction;
- the next-year forecast;
- what the MAE, RMSE, and MAPE indicate about
  historical model performance;
- the supplied capacity status;
- why human review is still required.

Keep the response concise and avoid repetition.

Do not generate new calculations or numbers.

Do not infer causes, demand, staffing needs,
budget needs, or facility requirements.
"""


    else:

        raise ValueError(
            f"Unknown prompt variant: {variant}"
        )


    return (
        context
        + "\n"
        + instructions
    )