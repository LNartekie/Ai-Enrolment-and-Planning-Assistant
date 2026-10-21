# LLM Model Comparison

All models received the same deterministic forecast results, system instructions, and user prompt.

## qwen3:1.7b

**Status:** Success

**Response time:** 33.33 seconds

### Institutional Enrolment Analysis Explanation

---

#### **1. Historical Trend**  
The historical enrolment data shows a consistent upward trend, with the latest enrolment value being **1815**. The overall direction of enrolment is **increasing**, indicating that the institution is experiencing steady growth in student numbers over time.

---

#### **2. Forecast**  
Based on the linear trend model, the forecasted enrolment for the year **2026** is **1832**. This projection assumes that the trend observed in the historical data will continue, with no significant changes in the underlying factors influencing student enrollment.

---

#### **3. Historical Backtest Error**  
The backtest performance indicates that the forecasted enrolment of **1832** has an **MAE (Mean Absolute Error)** of **43.98 students**, an **RMSE (Root Mean Square Error)** of **46.59 students**, and an **MAPE (Mean Absolute Percentage Error)** of **2.65%**. These metrics reflect the accuracy of the forecast relative to the historical data. A **strong backtest evidence** suggests that the model has been validated against the available historical data, indicating a reasonable level of confidence in the forecast.

---

#### **4. Capacity Implication**  
The current supplied capacity is **1800**, and the forecasted enrolment for 2026 is **1832**, resulting in a **capacity gap of 32 students**. This gap indicates that the institution will need to plan for additional resources, such as classrooms, staff, or facilities, to accommodate the projected increase in enrolment. However, the capacity gap is **not yet exceeded**, and no immediate action is required. The forecast remains subject to human review to ensure alignment with institutional planning and resource allocation.

---

#### **5. Forecast Requires Human Review**  
The forecasted enrolment of **1832** is a **planning estimate** and not a guaranteed outcome. Therefore, the decision regarding staffing, budgeting, admissions, or program planning should be made **subject to human review** to ensure that all factors are considered and that the forecast aligns with institutional goals and available resources.

--- 

**Note:** The forecast is based solely on the provided historical data and the linear trend model. No additional assumptions or external factors have been considered.

---

## qwen:latest

**Status:** Success

**Response time:** 25.75 seconds

Historical analysis:
The historical trend is increasing.

Forecast:
The forecast year is 2026.
The predicted enrolment is 1832.

Selected forecasting model:
The selected forecasting model is Linear Trend.

Backtest performance:
MAE: 43.98 students
RMSE: 46.59 students
MAPE: 2.65%

---

## mistral:latest

**Status:** Success

**Response time:** 90.42 seconds

In the provided analysis, we observe a historical trend of increasing enrolment for the institution. The latest enrolment count stands at 1815 students, and this trend has continued in the recent past.

The forecast for the year 2026 indicates a predicted enrolment of 1832 students. This forecast is based on a linear trend model, which is a deterministic method that extrapolates historical data into the future.

In terms of historical backtesting error, the Mean Absolute Error (MAE) is 43.98 students, the Root Mean Squared Error (RMSE) is 46.59 students, and the Mean Absolute Percentage Error (MAPE) is 2.65%. A stronger backtest evidence is associated with this model, suggesting that it has performed well in the past when predicting enrolment trends. However, it's essential to remember that these error values do not guarantee the accuracy of the forecast for the year 2026.

Looking at the capacity information, the current supplied capacity is 1800, and the forecast capacity gap is 32 students. If capacity is available, this gap may suggest the need for additional infrastructure, staff, or resources to accommodate the predicted enrolment in 2026. However, it's crucial to emphasize that this capacity gap requires human review and further analysis before any decisions can be made regarding staffing, budgeting, facility, admissions, or program changes.

---

## llama3.2:latest

**Status:** Success

**Response time:** 46.07 seconds

Here's a clear explanation of the institutional enrolment analysis:

Historical Trend:
The university has experienced a steady increase in enrolment over the past few years, with a consistent upward trend. This is a positive indicator of the institution's growth and demand for higher education.

Forecast:
Based on a linear trend model, the forecast predicts that enrolment will continue to rise, with a predicted value of 1832 students for the year 2026. This represents a 6% increase from the current year.

Interpretation of Backtest Error:
The historical backtest error, measured by the Mean Absolute Error (MAE), RMSE, and Mean Absolute Percentage Error (MAPE), indicates that the forecasting model has not perfectly captured the historical data. The MAE and RMSE values suggest that the model has struggled to accurately predict enrolment fluctuations, with a moderate error rate. However, the strong backtest evidence suggests that the model has performed relatively well in the past, despite some room for improvement.

Capacity Implication:
The current supplied capacity is 1800, but the forecasted capacity gap is 32, indicating that the institution will need to accommodate approximately 32 additional students in the year 2026. While this may seem manageable, it's essential to consider the capacity implications and ensure that the institution can provide adequate facilities and resources to support the increased student load.

Reminder:
Please note that this forecast is a planning estimate and not a guarantee of future enrolment. The forecast requires human review and consideration to ensure that it aligns with institutional goals, budget constraints, and available resources. It's crucial to review and refine this forecast before making any decisions regarding admissions, staffing, or resource allocation.

---

