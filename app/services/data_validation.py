import pandas as pd


REQUIRED_COLUMNS = ["year", "enrolment"]

# We decided that forecasting requires at least
# six historical observations.
MINIMUM_YEARS = 6


def validate_enrolment_data(df: pd.DataFrame) -> dict:
    """
    Validate uploaded historical enrolment data.

    Expected minimum columns:
        year
        enrolment

    Optional columns:
        institution_name
        capacity

    Returns:
        {
            "valid": bool,
            "errors": list,
            "warnings": list,
            "cleaned_data": DataFrame | None,
            "summary": dict | None
        }
    """

    errors = []
    warnings = []

    # --------------------------------------------------
    # 1. CHECK THAT DATA WAS PROVIDED
    # --------------------------------------------------

    if df is None:
        return {
            "valid": False,
            "errors": ["No dataset was provided."],
            "warnings": [],
            "cleaned_data": None,
            "summary": None,
        }

    if df.empty:
        return {
            "valid": False,
            "errors": ["The uploaded dataset is empty."],
            "warnings": [],
            "cleaned_data": None,
            "summary": None,
        }

    # Work on a copy so the original upload
    # is not modified.
    data = df.copy()

    # --------------------------------------------------
    # 2. STANDARDISE COLUMN NAMES
    # --------------------------------------------------

    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------
    # 3. CHECK REQUIRED COLUMNS
    # --------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        errors.append(
            "Missing required column(s): "
            + ", ".join(missing_columns)
        )

        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "cleaned_data": None,
            "summary": None,
        }

    # --------------------------------------------------
    # 4. KEEP REQUIRED AND SUPPORTED OPTIONAL COLUMNS
    # --------------------------------------------------

    optional_columns = [
        column
        for column in [
            "institution_name",
            "capacity",
        ]
        if column in data.columns
    ]

    data = data[
        REQUIRED_COLUMNS + optional_columns
    ].copy()

    # --------------------------------------------------
    # 5. CONVERT REQUIRED VALUES TO NUMERIC
    # --------------------------------------------------

    data["year"] = pd.to_numeric(
        data["year"],
        errors="coerce"
    )

    data["enrolment"] = pd.to_numeric(
        data["enrolment"],
        errors="coerce"
    )

    # --------------------------------------------------
    # 6. CHECK INVALID YEARS
    # --------------------------------------------------

    invalid_year_count = int(
        data["year"]
        .isna()
        .sum()
    )

    if invalid_year_count > 0:
        errors.append(
            f"{invalid_year_count} row(s) contain "
            "missing or invalid year values."
        )

    # --------------------------------------------------
    # 7. CHECK INVALID ENROLMENT
    # --------------------------------------------------

    invalid_enrolment_count = int(
        data["enrolment"]
        .isna()
        .sum()
    )

    if invalid_enrolment_count > 0:
        errors.append(
            f"{invalid_enrolment_count} row(s) contain "
            "missing or invalid enrolment values."
        )

    # Stop before later checks if required numeric
    # values are unusable.
    if errors:
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "cleaned_data": None,
            "summary": None,
        }

    # --------------------------------------------------
    # 8. CHECK THAT YEARS ARE WHOLE NUMBERS
    # --------------------------------------------------

    non_integer_years = (
        data["year"] % 1 != 0
    )

    if non_integer_years.any():
        errors.append(
            "Year values must be whole numbers."
        )
    else:
        data["year"] = (
            data["year"]
            .astype(int)
        )

    # --------------------------------------------------
    # 9. CHECK NEGATIVE ENROLMENT
    # --------------------------------------------------

    negative_mask = (
        data["enrolment"] < 0
    )

    if negative_mask.any():
        count = int(
            negative_mask.sum()
        )

        errors.append(
            f"{count} row(s) contain negative "
            "enrolment values."
        )

    # --------------------------------------------------
    # 10. CHECK DUPLICATE YEARS
    # --------------------------------------------------

    duplicate_mask = (
        data["year"]
        .duplicated(
            keep=False
        )
    )

    if duplicate_mask.any():

        duplicate_years = sorted(
            data.loc[
                duplicate_mask,
                "year"
            ]
            .unique()
            .tolist()
        )

        errors.append(
            "Duplicate year(s) found: "
            + ", ".join(
                map(str, duplicate_years)
            )
        )

    # --------------------------------------------------
    # 11. SORT CHRONOLOGICALLY
    # --------------------------------------------------

    data = (
        data
        .sort_values("year")
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # 12. CHECK MINIMUM HISTORY
    # --------------------------------------------------

    number_of_years = int(
        data["year"]
        .nunique()
    )

    if number_of_years < MINIMUM_YEARS:
        errors.append(
            f"At least {MINIMUM_YEARS} years of "
            "historical enrolment data are required. "
            f"Only {number_of_years} year(s) were provided."
        )

    # --------------------------------------------------
    # 13. IDENTIFY MISSING YEARS
    # --------------------------------------------------

    if len(data) > 1:

        expected_years = set(
            range(
                int(data["year"].min()),
                int(data["year"].max()) + 1,
            )
        )

        provided_years = set(
            data["year"].tolist()
        )

        missing_years = sorted(
            expected_years
            - provided_years
        )

        if missing_years:
            warnings.append(
                "Missing year(s) in the historical series: "
                + ", ".join(
                    map(str, missing_years)
                )
            )

    # --------------------------------------------------
    # 14. CHECK ZERO ENROLMENT
    # --------------------------------------------------

    zero_count = int(
        (
            data["enrolment"] == 0
        )
        .sum()
    )

    if zero_count > 0:
        warnings.append(
            f"{zero_count} year(s) contain zero enrolment. "
            "The value may be genuine, but it should be "
            "reviewed before forecasting."
        )

    # --------------------------------------------------
    # 15. OPTIONAL CAPACITY VALIDATION
    # --------------------------------------------------

    if "capacity" in data.columns:

        original_capacity = (
            data["capacity"]
            .copy()
        )

        data["capacity"] = pd.to_numeric(
            data["capacity"],
            errors="coerce"
        )

        invalid_capacity = (
            original_capacity.notna()
            & data["capacity"].isna()
        )

        if invalid_capacity.any():
            warnings.append(
                "Some capacity values could not be "
                "interpreted as numbers."
            )

        negative_capacity = (
            data["capacity"] < 0
        )

        if negative_capacity.any():
            errors.append(
                "Capacity values cannot be negative."
            )

    # --------------------------------------------------
    # 16. FINAL VALIDITY
    # --------------------------------------------------

    is_valid = (
        len(errors) == 0
    )

    # --------------------------------------------------
    # 17. CREATE SUMMARY
    # --------------------------------------------------

    summary = None

    if is_valid:
        summary = {
            "number_of_years":
                int(data["year"].nunique()),

            "first_year":
                int(data["year"].min()),

            "last_year":
                int(data["year"].max()),

            "minimum_enrolment":
                float(
                    data["enrolment"].min()
                ),

            "maximum_enrolment":
                float(
                    data["enrolment"].max()
                ),

            "average_enrolment":
                float(
                    data["enrolment"].mean()
                ),

            "latest_enrolment":
                float(
                    data.iloc[-1]["enrolment"]
                ),
        }

    # --------------------------------------------------
    # 18. RETURN STRUCTURED VALIDATION RESULT
    # --------------------------------------------------

    return {
        "valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "cleaned_data":
            data if is_valid else None,
        "summary": summary,
    }