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