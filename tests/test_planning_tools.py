
import pytest
from pydantic import ValidationError

from app.services.planning_tools import analyse_capacity


def test_forecast_over_capacity():

    result = analyse_capacity(1900, 1800)

    assert result["capacity_gap"] == 100
    assert result["utilisation_percent"] == 105.56
    assert result["students_over_capacity"] == 100
    assert result["remaining_places"] == 0
    assert result["status"] == "over_capacity"


def test_forecast_within_capacity():

    result = analyse_capacity(1500, 1800)

    assert result["capacity_gap"] == -300
    assert result["utilisation_percent"] == 83.33
    assert result["remaining_places"] == 300
    assert result["students_over_capacity"] == 0
    assert result["status"] == "within_capacity"


def test_forecast_exactly_at_capacity():

    result = analyse_capacity(1800, 1800)

    assert result["capacity_gap"] == 0
    assert result["utilisation_percent"] == 100.0
    assert result["remaining_places"] == 0
    assert result["status"] == "at_capacity"


def test_negative_forecast_rejected():

    with pytest.raises(ValidationError):
        analyse_capacity(-100, 1800)


def test_zero_capacity_rejected():

    with pytest.raises(ValidationError):
        analyse_capacity(1900, 0)


def test_invalid_data_type_rejected():

    with pytest.raises(ValidationError):
        analyse_capacity("1900", 1800)
