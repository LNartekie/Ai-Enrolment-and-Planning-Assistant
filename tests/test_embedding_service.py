
import pytest

from app.services.embedding_service import (
    generate_embeddings,
    embed_query,
)


# --------------------------------------------------
# MOCK OLLAMA RESPONSE
# --------------------------------------------------

class FakeResponse:

    def __init__(self, embeddings):
        self.embeddings = embeddings

    def raise_for_status(self):
        pass

    def json(self):
        return {
            "embeddings": self.embeddings
        }


# --------------------------------------------------
# TEST MULTIPLE EMBEDDINGS
# --------------------------------------------------

def test_generate_embeddings(monkeypatch):

    def fake_post(url, json, timeout):
        assert url.endswith("/api/embed")
        assert json["model"] == "nomic-embed-text"
        assert len(json["input"]) == 2

        return FakeResponse([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ])

    monkeypatch.setattr(
        "app.services.embedding_service.requests.post",
        fake_post,
    )

    result = generate_embeddings(
        ["Enrolment planning", "Admission policy"],
        model_name="nomic-embed-text",
    )

    assert len(result) == 2
    assert len(result[0]) == 3


# --------------------------------------------------
# TEST EMPTY INPUT
# --------------------------------------------------

def test_empty_text_list():

    with pytest.raises(ValueError):
        generate_embeddings([])


# --------------------------------------------------
# TEST BLANK TEXT
# --------------------------------------------------

def test_blank_text_rejected():

    with pytest.raises(ValueError):
        generate_embeddings(["   "])


# --------------------------------------------------
# TEST INCORRECT EMBEDDING COUNT
# --------------------------------------------------

def test_embedding_count_mismatch(monkeypatch):

    def fake_post(url, json, timeout):
        return FakeResponse([
            [0.1, 0.2, 0.3]
        ])

    monkeypatch.setattr(
        "app.services.embedding_service.requests.post",
        fake_post,
    )

    with pytest.raises(RuntimeError):
        generate_embeddings(["First", "Second"])


# --------------------------------------------------
# TEST QUERY EMBEDDING
# --------------------------------------------------

def test_embed_query(monkeypatch):

    def fake_post(url, json, timeout):
        return FakeResponse([
            [0.7, 0.8, 0.9]
        ])

    monkeypatch.setattr(
        "app.services.embedding_service.requests.post",
        fake_post,
    )

    result = embed_query(
        "What is the capacity policy?"
    )

    assert result == [0.7, 0.8, 0.9]

