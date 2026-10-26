
import pytest

from app.services.retrieval_service import (
    SessionVectorIndex,
)


SAMPLE_CHUNKS = [
    {
        "chunk_id": "doc1-c0001",
        "source_name": "capacity_policy.txt",
        "page_number": None,
        "text": "The institution has capacity for 1800 students.",
    },
    {
        "chunk_id": "doc2-c0001",
        "source_name": "admission_policy.txt",
        "page_number": None,
        "text": "Admission decisions require management approval.",
    },
]


def fake_document_embeddings(texts, model_name=None):

    assert all(
        text.startswith("search_document: ")
        for text in texts
    )

    return [
        [1.0, 0.0],
        [0.0, 1.0],
    ]


def fake_query_embedding(question, model_name=None):

    assert question.startswith("search_query: ")

    return [1.0, 0.0]


def test_build_index(monkeypatch):

    monkeypatch.setattr(
        "app.services.retrieval_service.generate_embeddings",
        fake_document_embeddings,
    )

    index = SessionVectorIndex()

    count = index.build(SAMPLE_CHUNKS)

    assert count == 2
    assert index.document_count == 2


def test_retrieve_capacity_policy(monkeypatch):

    monkeypatch.setattr(
        "app.services.retrieval_service.generate_embeddings",
        fake_document_embeddings,
    )

    monkeypatch.setattr(
        "app.services.retrieval_service.embed_query",
        fake_query_embedding,
    )

    index = SessionVectorIndex()

    index.build(SAMPLE_CHUNKS)

    results = index.search(
        "What is the student capacity?",
        top_k=1,
    )

    assert len(results) == 1

    assert (
        results[0]["source_name"]
        == "capacity_policy.txt"
    )

    assert "1800" in results[0]["text"]


def test_empty_index_rejected():

    index = SessionVectorIndex()

    with pytest.raises(ValueError):
        index.build([])


def test_duplicate_chunk_ids_rejected():

    index = SessionVectorIndex()

    duplicate_chunks = [
        SAMPLE_CHUNKS[0],
        dict(SAMPLE_CHUNKS[0]),
    ]

    with pytest.raises(ValueError):
        index.build(duplicate_chunks)


def test_search_without_documents():

    index = SessionVectorIndex()

    assert index.search(
        "What is our capacity?"
    ) == []


def test_no_match_above_threshold(monkeypatch):

    monkeypatch.setattr(
        "app.services.retrieval_service.generate_embeddings",
        fake_document_embeddings,
    )

    monkeypatch.setattr(
        "app.services.retrieval_service.embed_query",
        lambda question, model_name=None: [-1.0, -1.0],
    )

    index = SessionVectorIndex()

    index.build(SAMPLE_CHUNKS)

    results = index.search(
        "Unrelated question",
        min_similarity=0.25,
    )

    assert results == []


def test_clear_index(monkeypatch):

    monkeypatch.setattr(
        "app.services.retrieval_service.generate_embeddings",
        fake_document_embeddings,
    )

    index = SessionVectorIndex()

    index.build(SAMPLE_CHUNKS)

    index.clear()

    assert index.document_count == 0
    assert index.search("What is the capacity?") == []
