import pytest

from app.services.document_processing import (
    clean_document_text,
    chunk_document_text,
    process_document,
)


# --------------------------------------------------
# TEST TEXT CLEANING
# --------------------------------------------------

def test_clean_document_text():

    raw_text = (
        "Enrolment   planning\tpolicy\n\n"
        "Capacity    target"
    )


    cleaned = clean_document_text(
        raw_text
    )


    assert (
        "Enrolment planning policy"
        in cleaned
    )

    assert (
        "Capacity target"
        in cleaned
    )


# --------------------------------------------------
# TEST VALID TXT DOCUMENT
# --------------------------------------------------

def test_process_valid_txt_document():

    file_bytes = (
        b"The university plans to increase "
        b"student enrolment gradually while "
        b"monitoring teaching capacity."
    )


    result = process_document(
        file_bytes=file_bytes,
        filename="planning_policy.txt",
    )


    assert result["success"] is True

    assert len(
        result["chunks"]
    ) >= 1

    assert (
        result["source_name"]
        == "planning_policy.txt"
    )


# --------------------------------------------------
# TEST METADATA
# --------------------------------------------------

def test_document_chunk_contains_metadata():

    file_bytes = (
        b"Undergraduate enrolment planning "
        b"requires annual capacity review."
    )


    result = process_document(
        file_bytes=file_bytes,
        filename="capacity_policy.txt",
    )


    chunk = result[
        "chunks"
    ][0]


    assert "chunk_id" in chunk

    assert "source_id" in chunk

    assert "source_name" in chunk

    assert "file_type" in chunk

    assert "page_number" in chunk

    assert "chunk_index" in chunk

    assert "word_count" in chunk

    assert "text" in chunk


# --------------------------------------------------
# TEST MULTIPLE CHUNKS
# --------------------------------------------------

def test_long_document_creates_multiple_chunks():

    text = " ".join(
        ["enrolment"] * 500
    )


    result = process_document(
        file_bytes=text.encode(
            "utf-8"
        ),
        filename="long_policy.txt",
        chunk_size_words=100,
        overlap_words=20,
    )


    assert result["success"] is True

    assert len(
        result["chunks"]
    ) > 1


# --------------------------------------------------
# TEST EMPTY DOCUMENT
# --------------------------------------------------

def test_empty_document_is_rejected():

    result = process_document(
        file_bytes=b"",
        filename="empty.txt",
    )


    assert result["success"] is False

    assert len(
        result["errors"]
    ) >= 1


# --------------------------------------------------
# TEST UNSUPPORTED FILE
# --------------------------------------------------

def test_unsupported_document_type():

    result = process_document(
        file_bytes=b"test content",
        filename="document.csv",
    )


    assert result["success"] is False

    assert (
        "Unsupported document type"
        in result["errors"][0]
    )


# --------------------------------------------------
# TEST INVALID CHUNK SETTINGS
# --------------------------------------------------

def test_overlap_must_be_smaller_than_chunk():

    with pytest.raises(
        ValueError
    ):

        chunk_document_text(
            text="test words here",
            chunk_size_words=10,
            overlap_words=10,
        )