import pandas as pd

from app.services.data_validation import (
    validate_enrolment_data
)


# --------------------------------------------------
# TEST 1: VALID DATA
# --------------------------------------------------

def test_valid_data():

    df = pd.DataFrame({
        "year": [
            2018,
            2019,
            2020,
            2021,
            2022,
            2023
        ],
        "enrolment": [
            1200,
            1250,
            1300,
            1280,
            1350,
            1400
        ]
    })

    result = validate_enrolment_data(df)

    assert result["valid"] is True
    assert result["errors"] == []
    assert result["cleaned_data"] is not None
    assert result["summary"]["number_of_years"] == 6


# --------------------------------------------------
# TEST 2: MISSING REQUIRED COLUMN
# --------------------------------------------------

def test_missing_enrolment_column():

    df = pd.DataFrame({
        "year": [
            2018,
            2019,
            2020,
            2021,
            2022,
            2023
        ]
    })

    result = validate_enrolment_data(df)

    assert result["valid"] is False

    assert any(
        "Missing required column"
        in error
        for error in result["errors"]
    )


# --------------------------------------------------
# TEST 3: NEGATIVE ENROLMENT
# --------------------------------------------------

def test_negative_enrolment():

    df = pd.DataFrame({
        "year": [
            2018,
            2019,
            2020,
            2021,
            2022,
            2023
        ],
        "enrolment": [
            1200,
            1250,
            -50,
            1280,
            1350,
            1400
        ]
    })

    result = validate_enrolment_data(df)

    assert result["valid"] is False

    assert any(
        "negative enrolment"
        in error.lower()
        for error in result["errors"]
    )


# --------------------------------------------------
# TEST 4: DUPLICATE YEAR
# --------------------------------------------------

def test_duplicate_year():

    df = pd.DataFrame({
        "year": [
            2018,
            2019,
            2020,
            2020,
            2022,
            2023
        ],
        "enrolment": [
            1200,
            1250,
            1300,
            1310,
            1350,
            1400
        ]
    })

    result = validate_enrolment_data(df)

    assert result["valid"] is False

    assert any(
        "Duplicate year"
        in error
        for error in result["errors"]
    )


# --------------------------------------------------
# TEST 5: TOO FEW YEARS
# --------------------------------------------------

def test_too_few_years():

    df = pd.DataFrame({
        "year": [
            2020,
            2021,
            2022,
            2023
        ],
        "enrolment": [
            1000,
            1100,
            1150,
            1200
        ]
    })

    result = validate_enrolment_data(df)

    assert result["valid"] is False

    assert any(
        "At least 6 years"
        in error
        for error in result["errors"]
    )


# --------------------------------------------------
# TEST 6: MISSING YEAR IN SERIES
# --------------------------------------------------

def test_missing_year_warning():

    df = pd.DataFrame({
        "year": [
            2017,
            2018,
            2019,
            2021,
            2022,
            2023
        ],
        "enrolment": [
            1000,
            1050,
            1080,
            1160,
            1210,
            1260
        ]
    })

    result = validate_enrolment_data(df)

    assert result["valid"] is True

    assert any(
        "Missing year"
        in warning
        for warning in result["warnings"]
    )


# --------------------------------------------------
# TEST 7: INVALID ENROLMENT TEXT
# --------------------------------------------------

def test_invalid_enrolment_text():

    df = pd.DataFrame({
        "year": [
            2018,
            2019,
            2020,
            2021,
            2022,
            2023
        ],
        "enrolment": [
            1000,
            1100,
            "unknown",
            1200,
            1250,
            1300
        ]
    })

    result = validate_enrolment_data(df)

    assert result["valid"] is False

    assert any(
        "invalid enrolment"
        in error.lower()
        for error in result["errors"]
    )


# --------------------------------------------------
# TEST 8: ZERO ENROLMENT WARNING
# --------------------------------------------------

def test_zero_enrolment_warning():

    df = pd.DataFrame({
        "year": [
            2018,
            2019,
            2020,
            2021,
            2022,
            2023
        ],
        "enrolment": [
            1000,
            1100,
            0,
            1200,
            1250,
            1300
        ]
    })

    result = validate_enrolment_data(df)

    assert result["valid"] is True

    assert any(
        "zero enrolment"
        in warning.lower()
        for warning in result["warnings"]
    )