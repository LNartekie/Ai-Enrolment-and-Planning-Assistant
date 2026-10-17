from app.services.llm_service import (
    check_ollama_connection
)


def test_ollama_connection():

    assert (
        check_ollama_connection()
        is True
    )