
import numpy as np

from app.services.embedding_service import (
    EMBEDDING_MODEL,
    generate_embeddings,
    embed_query,
)


class SessionVectorIndex:
    """
    A temporary vector index for one institution's
    uploaded knowledge documents.
    """

    def __init__(self, model_name: str | None = None):

        self.model_name = model_name or EMBEDDING_MODEL

        self._chunks = []
        self._vectors = None


    @property
    def document_count(self):
        return len(self._chunks)


    # ----------------------------------------------
    # BUILD THE INDEX
    # ----------------------------------------------

    def build(self, chunks: list[dict]) -> int:

        if not isinstance(chunks, list) or not chunks:
            raise ValueError(
                "Cannot build an index without document chunks."
            )

        seen_ids = set()

        for chunk in chunks:

            if not isinstance(chunk, dict):
                raise ValueError(
                    "Each document chunk must be a dictionary."
                )

            for field in (
                "chunk_id",
                "source_name",
                "text",
            ):
                if (
                    not isinstance(chunk.get(field), str)
                    or not chunk[field].strip()
                ):
                    raise ValueError(
                        f"Document chunk is missing a valid {field}."
                    )

            if chunk["chunk_id"] in seen_ids:
                raise ValueError(
                    "Duplicate document chunk ID detected."
                )

            seen_ids.add(chunk["chunk_id"])

        # Tell the model these inputs are documents.
        texts = [
            "search_document: " + chunk["text"]
            for chunk in chunks
        ]

        embeddings = generate_embeddings(
            texts=texts,
            model_name=self.model_name,
        )

        vectors = np.asarray(
            embeddings,
            dtype=np.float64,
        )

        if (
            vectors.ndim != 2
            or vectors.shape[0] != len(chunks)
        ):
            raise RuntimeError(
                "Invalid document embedding dimensions."
            )

        if not np.all(np.isfinite(vectors)):
            raise RuntimeError(
                "Document vectors contain invalid values."
            )

        norms = np.linalg.norm(
            vectors,
            axis=1,
        )

        if np.any(norms == 0):
            raise RuntimeError(
                "Document embeddings cannot be zero vectors."
            )

        # Normalise vectors for cosine similarity.
        normalised_vectors = vectors / norms[:, None]

        # Only replace the active index after validation.
        self._vectors = normalised_vectors
        self._chunks = [
            dict(chunk)
            for chunk in chunks
        ]

        return len(self._chunks)


    # ----------------------------------------------
    # SEARCH THE INDEX
    # ----------------------------------------------

    def search(
        self,
        question: str,
        top_k: int = 3,
        min_similarity: float = 0.25,
    ) -> list[dict]:

        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                "The search question cannot be empty."
            )

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k < 1
        ):
            raise ValueError(
                "top_k must be a positive integer."
            )

        if not isinstance(min_similarity, (int, float)):
            raise ValueError(
                "min_similarity must be a number."
            )

        if not 0 <= min_similarity <= 1:
            raise ValueError(
                "min_similarity must be between 0 and 1."
            )

        if self._vectors is None:
            return []

        # Tell the model this input is a search query.
        query_vector = np.asarray(
            embed_query(
                "search_query: " + question,
                model_name=self.model_name,
            ),
            dtype=np.float64,
        )

        if (
            query_vector.ndim != 1
            or len(query_vector) != self._vectors.shape[1]
            or not np.all(np.isfinite(query_vector))
        ):
            raise RuntimeError(
                "Invalid query embedding dimensions or values."
            )

        query_norm = np.linalg.norm(query_vector)

        if query_norm == 0:
            raise RuntimeError(
                "Query embedding cannot be a zero vector."
            )

        normalised_query = query_vector / query_norm

        # Compute cosine similarity against all chunks.
        similarities = (
            self._vectors @ normalised_query
        )

        ranked_indices = np.argsort(
            -similarities,
            kind="stable",
        )

        results = []

        for index in ranked_indices:

            score = float(similarities[index])

            if score < min_similarity:
                continue

            match = dict(self._chunks[index])
            match["similarity_score"] = score

            results.append(match)

            if len(results) >= top_k:
                break

        return results


    # ----------------------------------------------
    # CLEAR INSTITUTIONAL KNOWLEDGE
    # ----------------------------------------------

    def clear(self):

        self._chunks = []
        self._vectors = None
