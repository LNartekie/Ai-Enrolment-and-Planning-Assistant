# Qwen3 Prompt Variation Experiment

Model: qwen3:1.7b

All prompt variations received the same deterministic analytical and forecasting results.

## Prompt Variant A

**Status:** Success

**Response time:** 30.58 seconds

1. **Historical Trend**  
The institution has shown a consistent upward trend in enrolment over time, with a recent acceleration in growth. The latest historical enrolment figure is 1,815 students, indicating a steady increase in student numbers.

2. **Forecast**  
Based on a linear trend model, the forecasted enrolment for the year 2026 is predicted to be 1,832 students. This forecast reflects the continuation of the historical growth pattern, with the expectation that enrolment will continue to increase.

3. **Historical Backtest Performance**  
The historical backtest performance indicates that the model has been trained on a sufficient dataset, with a Mean Absolute Error (MAE) of 43.98 students, a Root Mean Square Error (RMSE) of 46.59 students, and a Mean Absolute Percentage Error (MAPE) of 2.65%. These metrics suggest that the model has performed well in capturing the historical trends, with a low error rate relative to the actual values.

4. **Capacity Implication**  
The supplied capacity for the institution is 1,800 students. The forecasted enrolment of 1,832 students exceeds this capacity, resulting in a capacity gap of 32 students. This indicates that the institution may need to review its capacity planning, including potential adjustments to facilities, staffing, or admissions policies, to accommodate the forecasted enrolment.

5. **Human Review**  
The results are based on the provided data and forecasting model. While the forecast reflects the historical trend, it is a planning estimate and not guaranteed to be accurate. The decision to review capacity and make adjustments will be subject to further human review and consideration of additional factors before implementation.

---

## Prompt Variant B

**Status:** Success

**Response time:** 24.52 seconds

### Enrolment Planning Brief

---

### **KNOWN HISTORICAL FACTS**

- **Latest historical enrolment:** 1815 students  
- **Overall historical direction:** Increasing  
- **Recent historical direction:** Increasing  

---

### **FORECAST ESTIMATE**

- **Forecast year:** 2026  
- **Predicted enrolment:** 1832 students  
- **Forecasting model used:** Linear Trend  

---

### **HISTORICAL MODEL PERFORMANCE**

- **MAE (Mean Absolute Error):** 43.98 students  
- **RMSE (Root Mean Square Error):** 46.59 students  
- **MAPE (Mean Absolute Percentage Error):** 2.65%  
- **Number of backtest folds:** 5  
- **Backtest evidence label:** Strong  

**Important Note:** The backtest evidence label refers only to the amount of historical evaluation available. It does not prove that the future forecast is correct or certain.

---

### **CAPACITY STATUS**

- **Supplied capacity:** 1800 students  
- **Forecast capacity gap:** 32 students  
- **Capacity status:** Exceeded  

---

### **PLANNING INTERPRETATION**

The historical data shows a consistent upward trend in enrolment, with the most recent direction also indicating growth. The linear trend model predicts an increase of 18 students over the forecasted year, which is slightly below the supplied capacity. While the model performs well in terms of backtesting (with a low MAPE of 2.65%), it is important to note that forecasting is inherently uncertain and does not guarantee future outcomes.

The capacity gap of 32 students indicates that the forecasted enrolment exceeds the available capacity. However, this is a planning estimate and does not constitute a definitive requirement for staffing, budgeting, or facility planning. Institutional decisions should remain subject to human review and should not be based solely on forecasted numbers.

---

**Human Review Statement:**  
This brief is based on historical data and forecasting estimates. The capacity status reflects a planning gap, not a definitive requirement for institutional action. Final decisions should be made in consultation with relevant stakeholders.

---

## Prompt Variant C

**Status:** Success

**Response time:** 11.03 seconds

**One-sentence summary:** The institution is forecasting an increase in enrolment in 2026, with a capacity gap of 32 students, but the current capacity is insufficient to meet projected demand.

- **Historical direction:** Enrolment is increasing over time.  
- **Next-year forecast:** Enrolment is predicted to reach 1832 students.  
- **Model performance:** The linear trend model shows a MAE of 43.98 students, RMSE of 46.59 students, and MAPE of 2.65%, indicating moderate historical accuracy.  
- **Capacity status:** Current capacity is exceeded by 32 students.  
- **Human review required:** Institutional decisions must remain subject to human review, even with forecasting results, to ensure appropriate action is taken.

---

