
import sys
from pathlib import Path


# Allow imports from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.services.document_processing import (
    process_document,
)

from app.services.retrieval_service import (
    SessionVectorIndex,
)


# --------------------------------------------------
# SELECT DOCUMENT
# --------------------------------------------------

if len(sys.argv) < 2:
    raise SystemExit(
        "Usage: python evaluation/04_retrieval_test.py "
        '"path/to/institutional_document.pdf"'
    )


document_path = Path(sys.argv[1])


if not document_path.is_file():
    raise SystemExit(
        f"Document not found: {document_path}"
    )


# --------------------------------------------------
# PROCESS DOCUMENT
# --------------------------------------------------

print("\n" + "=" * 50)
print("STEP 1: DOCUMENT PROCESSING")
print("=" * 50)


result = process_document(
    file_bytes=document_path.read_bytes(),
    filename=document_path.name,
)


if not result["success"]:

    for error in result["errors"]:
        print("ERROR:", error)

    raise SystemExit(1)


chunks = result["chunks"]


print("Document:", result["source_name"])
print("Total chunks:", len(chunks))
print("Source ID:", result["source_id"])


for warning in result["warnings"]:
    print("WARNING:", warning)


# --------------------------------------------------
# BUILD VECTOR INDEX
# --------------------------------------------------

print("\n" + "=" * 50)
print("STEP 2: VECTOR INDEXING")
print("=" * 50)


index = SessionVectorIndex()

indexed_count = index.build(chunks)


print("Indexed chunks:", indexed_count)
print("Embedding model:", index.model_name)


# --------------------------------------------------
# SAMPLE RETRIEVAL QUESTIONS
# --------------------------------------------------

questions = [
    "What is the approved enrolment capacity for 2026?",
    "Was the proposed capacity increase to 2000 students approved?",
    "Who has authority to approve capacity expansion?",
    "What is the provisional enrolment target for 2026?",
    "What procedure applies when enrolment exceeds capacity?",
    "What is the policy on constructing a new airport?",
]


# --------------------------------------------------
# SEARCH DOCUMENT
# --------------------------------------------------

print("\n" + "=" * 50)
print("STEP 3: SEMANTIC RETRIEVAL")
print("=" * 50)


for question in questions:

    print("\n" + "-" * 50)
    print("QUESTION:", question)
    print("-" * 50)

    matches = index.search(
        question=question,
        top_k=3,
        min_similarity=0.65,
    )

    if not matches:

        print(
            "No passage met the similarity threshold."
        )
        continue


    for rank, match in enumerate(
        matches,
        start=1,
    ):

        print(f"\nMATCH {rank}")

        print(
            "Similarity:",
            round(match["similarity_score"], 4),
        )

        print(
            "Source:",
            match["source_name"],
        )

        print(
            "Page:",
            match["page_number"],
        )

        print(
            "Chunk ID:",
            match["chunk_id"],
        )

        print("\nRetrieved text:")

        print(
            match["text"]
        )

        print()


# --------------------------------------------------
# CLEAN UP
# --------------------------------------------------

index.clear()

print("\n" + "=" * 50)
print("RETRIEVAL EVALUATION COMPLETED")
print("=" * 50)

