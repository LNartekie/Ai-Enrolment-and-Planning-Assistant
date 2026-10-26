
import math
import os

import requests
from dotenv import load_dotenv


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
).rstrip("/")

EMBEDDING_MODEL = os.getenv(
    "OLLAMA_EMBEDDING_MODEL",
    "nomic-embed-text"
)


# --------------------------------------------------
# GENERATE EMBEDDINGS
# --------------------------------------------------

def generate_embeddings(
    texts: list[str],
    model_name: str | None = None,
) -> list[list[float]]:
    """
    Convert a list of texts into numerical embeddings
    using Ollama.

    Every input text must receive one valid embedding.
    """

    if not isinstance(texts, list) or not texts:
        raise ValueError(
            "Provide a non-empty list of texts."
        )

    if any(
        not isinstance(text, str) or not text.strip()
        for text in texts
    ):
        raise ValueError(
            "Every text must be a non-empty string."
        )

    selected_model = model_name or EMBEDDING_MODEL

    payload = {
        "model": selected_model,
        "input": texts,
        "truncate": False,
    }

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

    except requests.RequestException as error:
        raise RuntimeError(
            f"Ollama embedding request failed: {error}"
        ) from error

    try:
        response_data = response.json()
        embeddings = response_data["embeddings"]

    except (ValueError, KeyError, TypeError) as error:
        raise RuntimeError(
            "Ollama returned an invalid embedding response."
        ) from error

    if (
        not isinstance(embeddings, list)
        or len(embeddings) != len(texts)
    ):
        raise RuntimeError(
            "The number of embeddings does not match "
            "the number of input texts."
        )

    dimensions = None

    for vector in embeddings:
        if not isinstance(vector, list) or not vector:
            raise RuntimeError(
                "Ollama returned an empty or invalid vector."
            )

        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in vector
        ):
            raise RuntimeError(
                "Ollama returned invalid numerical values."
            )

        if dimensions is None:
            dimensions = len(vector)

        elif len(vector) != dimensions:
            raise RuntimeError(
                "Embedding dimensions are inconsistent."
            )

    return embeddings


# --------------------------------------------------
# EMBED ONE SEARCH QUERY
# --------------------------------------------------

def embed_query(
    question: str,
    model_name: str | None = None,
) -> list[float]:
    """
    Convert one user question into an embedding.
    """

    return generate_embeddings(
        texts=[question],
        model_name=model_name,
    )[0]
