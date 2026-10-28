
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.document_processing import process_document
from app.services.retrieval_service import SessionVectorIndex
from app.services.evidence_sufficiency import (
    assess_evidence_sufficiency,
)


DOCUMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "demo"
    / "institutional_enrolment_capacity_policy_2020_2026_REVISED.pdf"
)

questions = [
    {
        "question": (
            "What happens when student enrolment "
            "exceeds approved capacity?"
        ),
        "expected": True,
    },
    {
        "question": (
            "What is the institutional policy "
            "on constructing a new airport?"
        ),
        "expected": False,
    },
]


print("\nSTEP 1: PROCESS DOCUMENT")

result = process_document(
    file_bytes=DOCUMENT_PATH.read_bytes(),
    filename=DOCUMENT_PATH.name,
)

if not result["success"]:
    raise RuntimeError(result["errors"])

index = SessionVectorIndex()

try:
    print("\nSTEP 2: BUILD VECTOR INDEX")

    index.build(result["chunks"])

    print("Indexed chunks:", len(result["chunks"]))

    print("\nSTEP 3: EVIDENCE-SUFFICIENCY TEST")

    for number, item in enumerate(questions, start=1):

        print("\n" + "=" * 50)
        print(f"QUESTION {number}")
        print("=" * 50)

        question = item["question"]

        matches = index.search(
            question=question,
            top_k=5,
            min_similarity=0.65,
        )

        evidence = [
            {
                "citation": f"S{i}",
                "text": match["text"],
            }
            for i, match in enumerate(matches, start=1)
        ]

        assessment = assess_evidence_sufficiency(
            question=question,
            evidence=evidence,
        )

        print("Question:", question)
        print("Expected supported:", item["expected"])
        print("Actual supported:", assessment["supported"])
        print("Citation:", assessment["citation"])
        print("Quote:", assessment["verbatim_quote"])

        passed = (
            assessment["supported"] == item["expected"]
        )

        print("RESULT:", "PASS" if passed else "FAIL")

finally:
    index.clear()
