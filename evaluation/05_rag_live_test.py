
import sys
from pathlib import Path

# --------------------------------------------------
# SET PROJECT ROOT
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------
# IMPORT PROJECT SERVICES
# --------------------------------------------------

from app.services.document_processing import (
    process_document,
)

from app.services.retrieval_service import (
    SessionVectorIndex,
)

from app.services.rag_service import (
    answer_document_question,
)


# --------------------------------------------------
# DOCUMENT PATH
# --------------------------------------------------

DOCUMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "demo"
    / "institutional_enrolment_capacity_policy_2020_2026_REVISED.pdf"
)


# --------------------------------------------------
# EVALUATION QUESTIONS
# --------------------------------------------------

questions = [
    {
        "question": (
            "What is the approved undergraduate "
            "enrolment capacity for 2026?"
        ),
        "expected": "1800 students",
    },
    {
        "question": (
            "Has the proposed capacity increase "
            "to 2000 students been approved?"
        ),
        "expected": "No, the proposal remains unapproved",
    },
    {
        "question": (
            "Who has the authority to approve "
            "capacity expansion?"
        ),
        "expected": "Institutional Management Committee",
    },
    {
        "question": (
            "What is the provisional enrolment "
            "target for 2026?"
        ),
        "expected": "1900 students, not formally adopted",
    },
    {
        "question": (
            "What happens when student enrolment "
            "exceeds approved capacity?"
        ),
        "expected": "Variance review and escalation",
    },
    {
        "question": (
            "What is the institutional policy "
            "on constructing a new airport?"
        ),
        "expected": "Insufficient evidence",
    },
]


# --------------------------------------------------
# STEP 1: LOAD AND PROCESS DOCUMENT
# --------------------------------------------------

print("\n" + "=" * 60)
print("STEP 1: DOCUMENT PROCESSING")
print("=" * 60)

if not DOCUMENT_PATH.exists():
    raise SystemExit(
        f"Document not found: {DOCUMENT_PATH}"
    )

document_result = process_document(
    file_bytes=DOCUMENT_PATH.read_bytes(),
    filename=DOCUMENT_PATH.name,
)

if not document_result["success"]:
    raise RuntimeError(
        f"Document processing failed: "
        f"{document_result['errors']}"
    )

chunks = document_result["chunks"]

print("Document:", document_result["source_name"])
print("Chunks:", len(chunks))


# --------------------------------------------------
# STEP 2: BUILD VECTOR INDEX
# --------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2: BUILD VECTOR INDEX")
print("=" * 60)

index = SessionVectorIndex()

indexed_count = index.build(chunks)

print("Indexed chunks:", indexed_count)
print("Embedding model:", index.model_name)


# --------------------------------------------------
# STEP 3: GENERATE GROUNDED ANSWERS
# --------------------------------------------------

print("\n" + "=" * 60)
print("STEP 3: LIVE RAG EVALUATION")
print("=" * 60)

for number, item in enumerate(questions, start=1):

    print("\n" + "-" * 60)
    print(f"QUESTION {number}")
    print("-" * 60)

    print("Question:", item["question"])
    print("Expected:", item["expected"])

    try:
        result = answer_document_question(
            question=item["question"],
            index=index,
            top_k=3,
            min_similarity=0.65,
        )

        print("\nSTATUS:", result["status"])

        print("\nQWEN ANSWER:")
        print(result["answer"])

        print("\nRETRIEVED COUNT:")
        print(result["retrieved_count"])

        print("\nSOURCES:")

        if not result["sources"]:
            print("No supporting sources returned.")

        else:
            for source in result["sources"]:
                print(
                    f"[{source['citation']}] "
                    f"{source['source_name']} "
                    f"(page {source['page_number']}, "
                    f"chunk {source['chunk_id']})"
                )

    except Exception as error:

        print("\nERROR:")
        print(type(error).__name__)
        print(str(error))


# --------------------------------------------------
# STEP 4: CLEAN UP
# --------------------------------------------------

index.clear()

print("\n" + "=" * 60)
print("LIVE RAG EVALUATION COMPLETED")
print("=" * 60)
