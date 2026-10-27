
import pytest
import requests

from app.services.rag_service import (
    answer_document_question,
    NO_EVIDENCE_MESSAGE,
)


class FakeIndex:

    def __init__(self, matches):
        self.matches = matches

    def search(self, question, top_k, min_similarity):
        return self.matches


class FakeResponse:

    def __init__(self, answer):
        self.answer = answer

    def raise_for_status(self):
        pass

    def json(self):
        return {
            "message": {
                "content": self.answer
            }
        }


SAMPLE_MATCH = {
    "chunk_id": "doc-c0001",
    "source_name": "institution_policy.pdf",
    "page_number": 4,
    "text": (
        "The approved capacity for 2026 is "
        "1800 students."
    ),
    "similarity_score": 0.8,
}


def test_no_evidence_skips_llm(monkeypatch):

    def should_not_call_ollama(*args, **kwargs):
        raise AssertionError(
            "Ollama should not be called without evidence."
        )

    monkeypatch.setattr(
        "app.services.rag_service.requests.post",
        should_not_call_ollama,
    )

    result = answer_document_question(
        question="What is the airport policy?",
        index=FakeIndex([]),
    )

    assert result["status"] == "insufficient_evidence"
    assert result["sources"] == []
    assert result["answer"] == NO_EVIDENCE_MESSAGE


def test_grounded_answer_with_citation(monkeypatch):

    def fake_post(url, json, timeout):

        assert json["messages"][0]["role"] == "system"
        assert "institution_policy.pdf" in (
            json["messages"][1]["content"]
        )

        return FakeResponse(
            "The approved capacity is "
            "1800 students. [S1]"
        )

    monkeypatch.setattr(
        "app.services.rag_service.requests.post",
        fake_post,
    )

    result = answer_document_question(
        question="What is the approved capacity?",
        index=FakeIndex([SAMPLE_MATCH]),
    )

    assert result["status"] == "answered"
    assert result["sources"][0]["citation"] == "S1"
    assert result["sources"][0]["page_number"] == 4


def test_unknown_citation_rejected(monkeypatch):

    monkeypatch.setattr(
        "app.services.rag_service.requests.post",
        lambda *args, **kwargs: FakeResponse(
            "Capacity is 1800 students. [S99]"
        ),
    )

    with pytest.raises(RuntimeError):
        answer_document_question(
            question="What is the capacity?",
            index=FakeIndex([SAMPLE_MATCH]),
        )


def test_answer_without_citation_rejected(monkeypatch):

    monkeypatch.setattr(
        "app.services.rag_service.requests.post",
        lambda *args, **kwargs: FakeResponse(
            "Capacity is 1800 students."
        ),
    )

    with pytest.raises(RuntimeError):
        answer_document_question(
            question="What is the capacity?",
            index=FakeIndex([SAMPLE_MATCH]),
        )


def test_model_can_abstain(monkeypatch):

    monkeypatch.setattr(
        "app.services.rag_service.requests.post",
        lambda *args, **kwargs: FakeResponse(
            NO_EVIDENCE_MESSAGE
        ),
    )

    result = answer_document_question(
        question="Does this document approve an airport?",
        index=FakeIndex([SAMPLE_MATCH]),
    )

    assert result["status"] == "insufficient_evidence"
    assert result["sources"] == []


def test_empty_question_rejected():

    with pytest.raises(ValueError):
        answer_document_question(
            question=" ",
            index=FakeIndex([SAMPLE_MATCH]),
        )


def test_ollama_failure(monkeypatch):

    def fake_post(*args, **kwargs):
        raise requests.ConnectionError(
            "Ollama unavailable"
        )

    monkeypatch.setattr(
        "app.services.rag_service.requests.post",
        fake_post,
    )

    with pytest.raises(RuntimeError):
        answer_document_question(
            question="What is the approved capacity?",
            index=FakeIndex([SAMPLE_MATCH]),
        )


# --------------------------------------------------
# TEST CONTRADICTORY RAG ANSWER
# --------------------------------------------------

def test_contradictory_rag_answer_rejected(monkeypatch):

    contradictory_answer = (
        "The capacity increase was not approved. [S1] "
        "The uploaded documents do not establish "
        "an answer to this question."
    )

    monkeypatch.setattr(
        "app.services.rag_service.requests.post",
        lambda *args, **kwargs: FakeResponse(
            contradictory_answer
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="contradictory",
    ):
        answer_document_question(
            question="Was the capacity increase approved?",
            index=FakeIndex([SAMPLE_MATCH]),
        )


# --------------------------------------------------
# TEST ABSTENTION WITH DIFFERENT FORMATTING
# --------------------------------------------------

def test_abstention_with_whitespace_accepted(monkeypatch):

    monkeypatch.setattr(
        "app.services.rag_service.requests.post",
        lambda *args, **kwargs: FakeResponse(
            "The uploaded documents do not establish\n"
            "an answer to this question"
        ),
    )

    result = answer_document_question(
        question="What is the airport policy?",
        index=FakeIndex([SAMPLE_MATCH]),
    )

    assert result["status"] == "insufficient_evidence"
    assert result["sources"] == []