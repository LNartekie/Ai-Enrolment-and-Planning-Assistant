from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. LOCATE PROJECT FILES
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clean_enrolment_2001_2013.csv"
)

RESULTS_FOLDER = PROJECT_ROOT / "results"

# Create results folder if it does not exist
RESULTS_FOLDER.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. LOAD THE CLEAN DATASET
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)


# --------------------------------------------------
# 3. BASIC DATASET OVERVIEW
# --------------------------------------------------

print("\n=====================================")
print("DATASET OVERVIEW")
print("=====================================")

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print(
    f"Unique institutions: "
    f"{df['UNITID'].nunique():,}"
)

print(
    f"Year range: "
    f"{df['year'].min()} - {df['year'].max()}"
)

print("\nColumn names:")
print(df.columns.tolist())


# --------------------------------------------------
# 4. DATA TYPES
# --------------------------------------------------

print("\n=====================================")
print("DATA TYPES")
print("=====================================")

print(df.dtypes)


# --------------------------------------------------
# 5. MISSING VALUES
# --------------------------------------------------

missing_values = df.isna().sum()

print("\n=====================================")
print("MISSING VALUES")
print("=====================================")

print(missing_values)


# --------------------------------------------------
# 6. BASIC ENROLMENT STATISTICS
# --------------------------------------------------

enrolment_stats = df["UGDS"].describe()

print("\n=====================================")
print("UGDS ENROLMENT STATISTICS")
print("=====================================")

print(enrolment_stats)


# --------------------------------------------------
# 7. ZERO ENROLMENT CHECK
# --------------------------------------------------

zero_records = df[df["UGDS"] == 0]

zero_institutions = zero_records["UNITID"].nunique()

print("\n=====================================")
print("ZERO ENROLMENT CHECK")
print("=====================================")

print(
    f"Zero-enrolment rows: "
    f"{len(zero_records):,}"
)

print(
    f"Institutions with at least one "
    f"zero-enrolment record: "
    f"{zero_institutions:,}"
)


# --------------------------------------------------
# 8. CHECK FOR DUPLICATES
# --------------------------------------------------

duplicate_count = df.duplicated(
    subset=["UNITID", "year"]
).sum()

print("\n=====================================")
print("DUPLICATE CHECK")
print("=====================================")

print(
    f"Duplicate UNITID-year records: "
    f"{duplicate_count:,}"
)


# --------------------------------------------------
# 9. INSTITUTION TYPE DISTRIBUTION
# --------------------------------------------------

control_counts = (
    df[["UNITID", "CONTROL"]]
    .drop_duplicates(subset=["UNITID"])
    ["CONTROL"]
    .value_counts()
    .sort_index()
)

print("\n=====================================")
print("INSTITUTION TYPE DISTRIBUTION")
print("=====================================")

print(control_counts)


# --------------------------------------------------
# 10. SAVE SUMMARY RESULTS
# --------------------------------------------------

summary = pd.DataFrame({
    "metric": [
        "total_rows",
        "total_columns",
        "unique_institutions",
        "minimum_year",
        "maximum_year",
        "missing_UGDS",
        "zero_enrolment_rows",
        "institutions_with_zero_enrolment",
        "duplicate_institution_year_records"
    ],

    "value": [
        len(df),
        len(df.columns),
        df["UNITID"].nunique(),
        df["year"].min(),
        df["year"].max(),
        df["UGDS"].isna().sum(),
        len(zero_records),
        zero_institutions,
        duplicate_count
    ]
})

summary.to_csv(
    RESULTS_FOLDER / "eda_dataset_summary.csv",
    index=False
)


# Save descriptive statistics
enrolment_stats.to_csv(
    RESULTS_FOLDER / "eda_enrolment_statistics.csv"
)


print("\n=====================================")
print("EDA STEP 1 COMPLETE")
print("=====================================")

print(
    "\nResults saved in:"
    f"\n{RESULTS_FOLDER}"
)

# --------------------------------------------------
# 11. YEARLY ENROLMENT TRENDS
# --------------------------------------------------

yearly_summary = (
    df.groupby("year")
      .agg(
          total_enrolment=("UGDS", "sum"),
          average_enrolment=("UGDS", "mean"),
          median_enrolment=("UGDS", "median")
      )
      .reset_index()
)

print("\n=====================================")
print("YEARLY ENROLMENT SUMMARY")
print("=====================================")

print(yearly_summary)


# --------------------------------------------------
# 12. YEAR-ON-YEAR TOTAL ENROLMENT CHANGE
# --------------------------------------------------

yearly_summary["total_change"] = (
    yearly_summary["total_enrolment"].diff()
)

yearly_summary["total_percentage_change"] = (
    yearly_summary["total_enrolment"]
    .pct_change()
    .mul(100)
)

print("\n=====================================")
print("YEAR-ON-YEAR CHANGE")
print("=====================================")

print(
    yearly_summary[
        [
            "year",
            "total_enrolment",
            "total_change",
            "total_percentage_change"
        ]
    ]
)


# --------------------------------------------------
# 13. SAVE YEARLY SUMMARY
# --------------------------------------------------

yearly_summary.to_csv(
    RESULTS_FOLDER / "eda_yearly_enrolment_summary.csv",
    index=False
)


# --------------------------------------------------
# 14. CREATE TREND CHART
# --------------------------------------------------

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.plot(
    yearly_summary["year"],
    yearly_summary["total_enrolment"],
    marker="o"
)

plt.title("Total Undergraduate Enrolment, 2001–2013")
plt.xlabel("Year")
plt.ylabel("Total Undergraduate Enrolment")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_FOLDER / "total_enrolment_trend.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 15. TOP 10 LARGEST ENROLMENT RECORDS
# --------------------------------------------------

largest_records = (
    df.nlargest(10, "UGDS")[
        ["UNITID", "INSTNM", "STABBR", "CONTROL", "year", "UGDS"]
    ]
)

print("\n=====================================")
print("TOP 10 LARGEST ENROLMENT RECORDS")
print("=====================================")

print(largest_records)


# --------------------------------------------------
# 16. INSTITUTIONS WITH ZERO ENROLMENT
# --------------------------------------------------

zero_case_details = (
    df[df["UGDS"] == 0]
    [
        ["UNITID", "INSTNM", "STABBR", "CONTROL", "year", "UGDS"]
    ]
    .sort_values(["UNITID", "year"])
)

print("\n=====================================")
print("ZERO ENROLMENT CASES")
print("=====================================")

print(zero_case_details)


# --------------------------------------------------
# 17. ENROLMENT HISTORY FOR ZERO-CASE INSTITUTIONS
# --------------------------------------------------

zero_ids = zero_case_details["UNITID"].unique()

zero_history = (
    df[df["UNITID"].isin(zero_ids)]
    [
        ["UNITID", "INSTNM", "year", "UGDS"]
    ]
    .sort_values(["UNITID", "year"])
)

print("\n=====================================")
print("FULL HISTORY OF ZERO-CASE INSTITUTIONS")
print("=====================================")

print(zero_history)


# --------------------------------------------------
# 18. SAVE OUTLIER AND ZERO-CASE RESULTS
# --------------------------------------------------

largest_records.to_csv(
    RESULTS_FOLDER / "eda_largest_enrolment_records.csv",
    index=False
)

zero_case_details.to_csv(
    RESULTS_FOLDER / "eda_zero_enrolment_cases.csv",
    index=False
)

zero_history.to_csv(
    RESULTS_FOLDER / "eda_zero_enrolment_history.csv",
    index=False
)


print("\nEDA STEP 3 COMPLETE")


# --------------------------------------------------
# 19. CREATE MAIN FORECASTING BASE DATASET
# --------------------------------------------------

# Identify institutions that contain at least one zero UGDS value
zero_institution_ids = (
    df.loc[df["UGDS"] == 0, "UNITID"]
    .unique()
)

# Exclude those institutions from the primary forecasting dataset
forecasting_df = (
    df[~df["UNITID"].isin(zero_institution_ids)]
    .copy()
)

print("\n=====================================")
print("FORECASTING BASE DATASET")
print("=====================================")

print(f"Rows: {len(forecasting_df):,}")

print(
    f"Institutions: "
    f"{forecasting_df['UNITID'].nunique():,}"
)

print(
    f"Zero UGDS values: "
    f"{(forecasting_df['UGDS'] == 0).sum():,}"
)


# Save the modelling base dataset
forecasting_df.to_csv(
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting_base_2001_2013.csv",
    index=False
)


# --------------------------------------------------
# 20. GET 2001 AND 2013 ENROLMENT
# --------------------------------------------------

start_data = (
    forecasting_df[
        forecasting_df["year"] == 2001
    ][
        ["UNITID", "UGDS"]
    ]
    .rename(
        columns={
            "UGDS": "enrolment_2001"
        }
    )
)

end_data = (
    forecasting_df[
        forecasting_df["year"] == 2013
    ][
        [
            "UNITID",
            "INSTNM",
            "STABBR",
            "CONTROL",
            "UGDS"
        ]
    ]
    .rename(
        columns={
            "UGDS": "enrolment_2013"
        }
    )
)


# --------------------------------------------------
# 21. COMBINE START AND END VALUES
# --------------------------------------------------

institution_change = pd.merge(
    start_data,
    end_data,
    on="UNITID",
    how="inner"
)


# --------------------------------------------------
# 22. CALCULATE CHANGE
# --------------------------------------------------

institution_change["absolute_change"] = (
    institution_change["enrolment_2013"]
    - institution_change["enrolment_2001"]
)

institution_change["percentage_change"] = (
    (
        institution_change["enrolment_2013"]
        - institution_change["enrolment_2001"]
    )
    / institution_change["enrolment_2001"]
    * 100
)


# --------------------------------------------------
# 23. CALCULATE CAGR
# --------------------------------------------------

institution_change["cagr"] = (
    (
        institution_change["enrolment_2013"]
        / institution_change["enrolment_2001"]
    ) ** (1 / 12)
    - 1
) * 100


# --------------------------------------------------
# 24. CLASSIFY INSTITUTION TREND
# --------------------------------------------------

institution_change["trend_category"] = "Stable"

institution_change.loc[
    institution_change["percentage_change"] > 5,
    "trend_category"
] = "Growth"

institution_change.loc[
    institution_change["percentage_change"] < -5,
    "trend_category"
] = "Decline"


# --------------------------------------------------
# 25. COUNT TREND CATEGORIES
# --------------------------------------------------

trend_counts = (
    institution_change["trend_category"]
    .value_counts()
)

print("\n=====================================")
print("INSTITUTION TREND CATEGORIES")
print("=====================================")

print(trend_counts)


# --------------------------------------------------
# 26. SUMMARY STATISTICS
# --------------------------------------------------

print("\n=====================================")
print("INSTITUTION PERCENTAGE CHANGE")
print("=====================================")

print(
    institution_change[
        "percentage_change"
    ].describe()
)


# --------------------------------------------------
# 27. SAVE RESULTS
# --------------------------------------------------

institution_change.to_csv(
    RESULTS_FOLDER
    / "eda_institution_growth_decline.csv",
    index=False
)

trend_counts.to_csv(
    RESULTS_FOLDER
    / "eda_institution_trend_counts.csv"
)


print("\nEDA STEP 4 COMPLETE")

# --------------------------------------------------
# 28. TOP 15 HIGHEST PERCENTAGE GROWTH
# --------------------------------------------------

highest_growth = (
    institution_change
    .nlargest(15, "percentage_change")
    [
        [
            "UNITID",
            "INSTNM",
            "STABBR",
            "CONTROL",
            "enrolment_2001",
            "enrolment_2013",
            "absolute_change",
            "percentage_change",
            "cagr"
        ]
    ]
)

print("\n=====================================")
print("TOP 15 HIGHEST PERCENTAGE GROWTH")
print("=====================================")

print(highest_growth)


# --------------------------------------------------
# 29. TOP 15 LARGEST DECLINES
# --------------------------------------------------

largest_decline = (
    institution_change
    .nsmallest(15, "percentage_change")
    [
        [
            "UNITID",
            "INSTNM",
            "STABBR",
            "CONTROL",
            "enrolment_2001",
            "enrolment_2013",
            "absolute_change",
            "percentage_change",
            "cagr"
        ]
    ]
)

print("\n=====================================")
print("TOP 15 LARGEST PERCENTAGE DECLINES")
print("=====================================")

print(largest_decline)


# --------------------------------------------------
# 30. CHECK SMALL STARTING ENROLMENT
# --------------------------------------------------

small_start_counts = {
    "10_or_less": (
        institution_change["enrolment_2001"] <= 10
    ).sum(),

    "50_or_less": (
        institution_change["enrolment_2001"] <= 50
    ).sum(),

    "100_or_less": (
        institution_change["enrolment_2001"] <= 100
    ).sum()
}

print("\n=====================================")
print("SMALL STARTING ENROLMENT COUNTS")
print("=====================================")

for label, count in small_start_counts.items():
    print(f"{label}: {count:,}")


# --------------------------------------------------
# 31. CHECK STARTING SIZE OF TOP 100 GROWTH CASES
# --------------------------------------------------

top_100_growth = (
    institution_change
    .nlargest(100, "percentage_change")
)

print("\n=====================================")
print("TOP 100 GROWTH CASES - STARTING SIZE")
print("=====================================")

print(
    top_100_growth["enrolment_2001"]
    .describe()
)


# --------------------------------------------------
# 32. SAVE EXTREME CASES
# --------------------------------------------------

highest_growth.to_csv(
    RESULTS_FOLDER / "eda_highest_growth_institutions.csv",
    index=False
)

largest_decline.to_csv(
    RESULTS_FOLDER / "eda_largest_decline_institutions.csv",
    index=False
)

print("\nEDA STEP 5 COMPLETE")