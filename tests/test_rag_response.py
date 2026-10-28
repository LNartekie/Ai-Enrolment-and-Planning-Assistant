import pytest
from pydantic import ValidationError

from app.schemas.rag_response import RAGDecision


def test_valid_answered_response():

    response = RAGDecision(
        status="answered",
        answer="The approved capacity is 1800 students.",
        citations=["S1"],
    )

    assert response.status == "answered"
    assert response.citations == ["S1"]


def test_valid_insufficient_evidence():

    response = RAGDecision(
        status="insufficient_evidence",
        answer=None,
        citations=[],
    )

    assert response.answer is None


def test_answer_without_citations_rejected():

    with pytest.raises(ValidationError):

        RAGDecision(
            status="answered",
            answer="Capacity is 1800.",
            citations=[],
        )


def test_insufficient_evidence_with_answer_rejected():

    with pytest.raises(ValidationError):

        RAGDecision(
            status="insufficient_evidence",
            answer="Capacity is 1800.",
            citations=[],
        )


def test_invalid_citation_rejected():

    with pytest.raises(ValidationError):

        RAGDecision(
            status="answered",
            answer="Capacity is 1800.",
            citations=["SOURCE99"],
        )


def test_contradictory_response_rejected():

    with pytest.raises(ValidationError):

        RAGDecision(
            status="answered",
            answer=(
                "Capacity is 1800. "
                "The uploaded documents do not "
                "establish an answer to this question."
            ),
            citations=["S1"],
        )


def test_unexpected_field_rejected():

    with pytest.raises(ValidationError):

        RAGDecision.model_validate({
            "status": "answered",
            "answer": "Capacity is 1800.",
            "citations": ["S1"],
            "invented_field": "extra content",
        })

def test_answer_admitting_missing_policy_rejected():

    with pytest.raises(ValidationError):

        RAGDecision(
            status="answered",
            answer=(
                "The requested policy is not "
                "explicitly detailed in the "
                "provided evidence."
            ),
            citations=["S1"],
        )


def test_unrelated_policy_abstention_accepted():

    response = RAGDecision(
        status="insufficient_evidence",
        answer=None,
        citations=[],
    )

    assert response.status == "insufficient_evidence"
    assert response.answer is None
    assert response.citations == []        