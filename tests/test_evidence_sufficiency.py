
import json

import pytest

from app.services.evidence_sufficiency import (
    assess_evidence_sufficiency,
)


class FakeResponse:

    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {
            "message": {
                "content": self.content
            }
        }


EVIDENCE = [
    {
        "citation": "S1",
        "text": (
            "A variance review is required when a verified "
            "headcount exceeds approved capacity."
        ),
    }
]


def test_supported_evidence(monkeypatch):

    fake_content = json.dumps({
        "supported": True,
        "citation": "S1",
        "verbatim_quote": "A variance review is required",
    })

    monkeypatch.setattr(
        "app.services.evidence_sufficiency.requests.post",
        lambda *args, **kwargs: FakeResponse(fake_content),
    )

    result = assess_evidence_sufficiency(
        "What happens when capacity is exceeded?",
        EVIDENCE,
    )

    assert result["supported"] is True
    assert result["citation"] == "S1"


def test_unsupported_policy(monkeypatch):

    fake_content = json.dumps({
        "supported": False,
        "citation": None,
        "verbatim_quote": None,
    })

    monkeypatch.setattr(
        "app.services.evidence_sufficiency.requests.post",
        lambda *args, **kwargs: FakeResponse(fake_content),
    )

    result = assess_evidence_sufficiency(
        "What is the airport construction policy?",
        EVIDENCE,
    )

    assert result["supported"] is False


def test_invented_quotation_rejected(monkeypatch):

    fake_content = json.dumps({
        "supported": True,
        "citation": "S1",
        "verbatim_quote": (
            "The institution has approved airport construction."
        ),
    })

    monkeypatch.setattr(
        "app.services.evidence_sufficiency.requests.post",
        lambda *args, **kwargs: FakeResponse(fake_content),
    )

    with pytest.raises(RuntimeError):
        assess_evidence_sufficiency(
            "What is the airport construction policy?",
            EVIDENCE,
        )
