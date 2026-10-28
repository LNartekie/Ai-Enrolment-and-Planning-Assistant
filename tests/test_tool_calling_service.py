
import pytest

from pydantic import ValidationError

from app.services.tool_calling_service import (
    explain_capacity_with_tool,
)


class FakeResponse:

    def __init__(self, message):
        self.message = message

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": self.message}


TOOL_MESSAGE = {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {
            "function": {
                "name": "analyse_capacity",
                "arguments": {
                    "forecast_enrolment": 1900,
                    "approved_capacity": 1800,
                },
            }
        }
    ],
}


FINAL_MESSAGE = {
    "role": "assistant",
    "content": (
        "The forecast is 1900 students against "
        "capacity of 1800, exceeding it by 100."
    ),
}


def test_successful_tool_call(monkeypatch):

    calls = []

    def fake_post(url, json, timeout):

        calls.append(json)

        if len(calls) == 1:
            return FakeResponse(TOOL_MESSAGE)

        return FakeResponse(FINAL_MESSAGE)

    monkeypatch.setattr(
        "app.services.tool_calling_service.requests.post",
        fake_post,
    )

    result = explain_capacity_with_tool(
        1900, 1800
    )

    assert result["tool_called"] == "analyse_capacity"

    assert result["tool_result"]["capacity_gap"] == 100

    assert (
        result["tool_result"]["utilisation_percent"]
        == 105.56
    )

    assert len(calls) == 2

    assert "tools" in calls[0]

    assert any(
        message["role"] == "tool"
        for message in calls[1]["messages"]
    )


def test_missing_tool_call_rejected(monkeypatch):

    monkeypatch.setattr(
        "app.services.tool_calling_service.requests.post",
        lambda *args, **kwargs: FakeResponse({
            "role": "assistant",
            "content": "The gap is 100.",
        }),
    )

    with pytest.raises(RuntimeError):
        explain_capacity_with_tool(1900, 1800)


def test_unknown_tool_rejected(monkeypatch):

    monkeypatch.setattr(
        "app.services.tool_calling_service.requests.post",
        lambda *args, **kwargs: FakeResponse({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "delete_database",
                        "arguments": {},
                    }
                }
            ],
        }),
    )

    with pytest.raises(RuntimeError):
        explain_capacity_with_tool(1900, 1800)


def test_unexpected_arguments_rejected(monkeypatch):

    monkeypatch.setattr(
        "app.services.tool_calling_service.requests.post",
        lambda *args, **kwargs: FakeResponse({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "analyse_capacity",
                        "arguments": {
                            "approved_capacity": 9999,
                        },
                    }
                }
            ],
        }),
    )

    with pytest.raises(RuntimeError):
        explain_capacity_with_tool(1900, 1800)


def test_invalid_capacity_rejected():

    with pytest.raises(ValidationError):
        explain_capacity_with_tool(1900, 0)


def test_negative_forecast_rejected():

    with pytest.raises(ValidationError):
        explain_capacity_with_tool(-10, 1800)


def test_empty_final_response_rejected(monkeypatch):

    calls = []

    def fake_post(url, json, timeout):

        calls.append(json)

        if len(calls) == 1:
            return FakeResponse(TOOL_MESSAGE)

        return FakeResponse({
            "role": "assistant",
            "content": "",
        })

    monkeypatch.setattr(
        "app.services.tool_calling_service.requests.post",
        fake_post,
    )

    with pytest.raises(RuntimeError):
        explain_capacity_with_tool(1900, 1800)

    assert len(calls) == 2
