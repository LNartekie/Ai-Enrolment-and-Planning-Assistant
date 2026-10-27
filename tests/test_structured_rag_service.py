
import json

import pytest
import requests

from app.services.structured_rag_service import (
    answer_document_question_structured,
)


class FakeIndex:

    def __init__(self, matches):
        self.matches = matches

    def search(self, question, top_k, min_similarity):
        return self.matches


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


SAMPLE_MATCH = {
    "chunk_id": "test-c0001",
    "source_name": "planning_policy.pdf",
    "page_number": 5,
    "text": (
        "A variance review is required when "
        "enrolment exceeds capacity."
    ),
    "similarity_score": 0.79,
}


def test_valid_structured_answer(monkeypatch):

    response_content = json.dumps({
        "status": "answered",
        "answer": "A variance review is required.",
        "citations": ["S1"],
    })

    monkeypatch.setattr(
        "app.services.structured_rag_service.requests.post",
        lambda *args, **kwargs: FakeResponse(
            response_content
        ),
    )

    result = answer_document_question_structured(
        question="What happens when capacity is exceeded?",
        index=FakeIndex([SAMPLE_MATCH]),
    )

    assert result["status"] == "answered"
    assert result["sources"][0]["page_number"] == 5


def test_empty_retrieval_skips_llm(monkeypatch):

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "Ollama should not be called."
        )

    monkeypatch.setattr(
        "app.services.structured_rag_service.requests.post",
        fail_if_called,
    )

    result = answer_document_question_structured(
        question="What is the airport policy?",
        index=FakeIndex([]),
    )

    assert result["status"] == "insufficient_evidence"
    assert result["sources"] == []


def test_structured_abstention(monkeypatch):

    response_content = json.dumps({
        "status": "insufficient_evidence",
        "answer": None,
        "citations": [],
    })

    monkeypatch.setattr(
        "app.services.structured_rag_service.requests.post",
        lambda *args, **kwargs: FakeResponse(
            response_content
        ),
    )

    result = answer_document_question_structured(
        question="What is the airport policy?",
        index=FakeIndex([SAMPLE_MATCH]),
    )

    assert result["status"] == "insufficient_evidence"
    assert result["sources"] == []


def test_invalid_json_corrected_on_retry(monkeypatch):

    calls = []

    valid_response = json.dumps({
        "status": "answered",
        "answer": "A variance review is required.",
        "citations": ["S1"],
    })

    def fake_post(*args, **kwargs):
        calls.append(1)

        if len(calls) == 1:
            return FakeResponse("invalid json")

        return FakeResponse(valid_response)

    monkeypatch.setattr(
        "app.services.structured_rag_service.requests.post",
        fake_post,
    )

    result = answer_document_question_structured(
        question="What happens when capacity is exceeded?",
        index=FakeIndex([SAMPLE_MATCH]),
    )

    assert len(calls) == 2
    assert result["status"] == "answered"


def test_unknown_citation_rejected_after_retry(monkeypatch):

    response_content = json.dumps({
        "status": "answered",
        "answer": "A variance review is required.",
        "citations": ["S99"],
    })

    calls = []

    def fake_post(*args, **kwargs):
        calls.append(1)
        return FakeResponse(response_content)

    monkeypatch.setattr(
        "app.services.structured_rag_service.requests.post",
        fake_post,
    )

    with pytest.raises(RuntimeError):
        answer_document_question_structured(
            question="What happens when capacity is exceeded?",
            index=FakeIndex([SAMPLE_MATCH]),
        )

    assert len(calls) == 2


def test_empty_question_rejected():

    with pytest.raises(ValueError):
        answer_document_question_structured(
            question=" ",
            index=FakeIndex([SAMPLE_MATCH]),
        )


def test_ollama_connection_failure(monkeypatch):

    def fake_post(*args, **kwargs):
        raise requests.ConnectionError(
            "Ollama unavailable"
        )

    monkeypatch.setattr(
        "app.services.structured_rag_service.requests.post",
        fake_post,
    )

    with pytest.raises(RuntimeError):
        answer_document_question_structured(
            question="What is the capacity procedure?",
            index=FakeIndex([SAMPLE_MATCH]),
        )
