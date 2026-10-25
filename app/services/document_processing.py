import hashlib
import io
from pathlib import Path


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
}


MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


DEFAULT_CHUNK_SIZE_WORDS = 180

DEFAULT_CHUNK_OVERLAP_WORDS = 30


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_document_text(
    text: str,
) -> str:
    """
    Clean extracted document text while preserving
    meaningful paragraph boundaries.

    The function is deterministic and does not use
    an LLM.
    """

    if not text:
        return ""

    # Replace non-breaking spaces with normal spaces.
    text = text.replace(
        "\u00a0",
        " ",
    )

    # Normalise different newline styles.
    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )


    cleaned_lines = []

    for line in text.split("\n"):

        # Collapse repeated spaces and tabs.
        cleaned_line = " ".join(
            line.split()
        )

        if cleaned_line:
            cleaned_lines.append(
                cleaned_line
            )


    return "\n".join(
        cleaned_lines
    ).strip()


# --------------------------------------------------
# TEXT CHUNKING
# --------------------------------------------------

def chunk_document_text(
    text: str,
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """
    Split text into overlapping word-based chunks.

    The overlap helps preserve context between
    neighbouring chunks.
    """

    if chunk_size_words <= 0:

        raise ValueError(
            "chunk_size_words must be greater than 0."
        )


    if overlap_words < 0:

        raise ValueError(
            "overlap_words cannot be negative."
        )


    if overlap_words >= chunk_size_words:

        raise ValueError(
            "overlap_words must be smaller than "
            "chunk_size_words."
        )


    words = text.split()


    if not words:
        return []


    chunks = []

    start = 0


    while start < len(words):

        end = min(
            start + chunk_size_words,
            len(words),
        )


        chunk_words = words[
            start:end
        ]


        chunk_text = " ".join(
            chunk_words
        ).strip()


        if chunk_text:

            chunks.append(
                chunk_text
            )


        if end >= len(words):
            break


        start = (
            end
            - overlap_words
        )


    return chunks


# --------------------------------------------------
# PDF EXTRACTION
# --------------------------------------------------

def _extract_pdf(
    file_bytes: bytes,
) -> tuple[list[dict], list[str]]:
    """
    Extract text page-by-page from a PDF.

    Page numbers are preserved for provenance.
    """

    from pypdf import PdfReader


    reader = PdfReader(
        io.BytesIO(
            file_bytes
        )
    )


    if reader.is_encrypted:

        raise ValueError(
            "Encrypted PDF documents are not supported."
        )


    units = []

    warnings = []


    for page_index, page in enumerate(
        reader.pages,
        start=1,
    ):

        raw_text = (
            page.extract_text()
            or ""
        )


        cleaned_text = (
            clean_document_text(
                raw_text
            )
        )


        if not cleaned_text:

            warnings.append(
                f"Page {page_index} contained "
                "no extractable text."
            )

            continue


        units.append(
            {
                "text": cleaned_text,
                "page_number": page_index,
            }
        )


    return units, warnings


# --------------------------------------------------
# DOCX EXTRACTION
# --------------------------------------------------

def _extract_docx(
    file_bytes: bytes,
) -> tuple[list[dict], list[str]]:
    """
    Extract paragraph text from a DOCX document.
    """

    from docx import Document


    document = Document(
        io.BytesIO(
            file_bytes
        )
    )


    paragraphs = []


    for paragraph in document.paragraphs:

        cleaned_text = (
            clean_document_text(
                paragraph.text
            )
        )


        if cleaned_text:

            paragraphs.append(
                cleaned_text
            )


    combined_text = "\n".join(
        paragraphs
    )


    units = []


    if combined_text:

        units.append(
            {
                "text": combined_text,
                "page_number": None,
            }
        )


    return units, []


# --------------------------------------------------
# TXT EXTRACTION
# --------------------------------------------------

def _extract_txt(
    file_bytes: bytes,
) -> tuple[list[dict], list[str]]:
    """
    Decode and clean a plain-text document.
    """

    try:

        raw_text = file_bytes.decode(
            "utf-8"
        )

    except UnicodeDecodeError:

        try:

            raw_text = file_bytes.decode(
                "utf-8-sig"
            )

        except UnicodeDecodeError as error:

            raise ValueError(
                "The text file could not be decoded "
                "as UTF-8."
            ) from error


    cleaned_text = (
        clean_document_text(
            raw_text
        )
    )


    units = []


    if cleaned_text:

        units.append(
            {
                "text": cleaned_text,
                "page_number": None,
            }
        )


    return units, []


# --------------------------------------------------
# MAIN DOCUMENT PROCESSOR
# --------------------------------------------------

def process_document(
    file_bytes: bytes,
    filename: str,
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> dict:
    """
    Process one institutional knowledge document.

    Returns cleaned chunks plus metadata that can
    later be used by the RAG pipeline.
    """

    result = {
        "success": False,
        "source_name": filename,
        "source_id": None,
        "file_type": None,
        "chunks": [],
        "errors": [],
        "warnings": [],
        "summary": {},
    }


    # --------------------------------------------------
    # VALIDATE FILE NAME
    # --------------------------------------------------

    if not filename:

        result["errors"].append(
            "A document filename is required."
        )

        return result


    extension = (
        Path(filename)
        .suffix
        .lower()
    )


    result["file_type"] = extension


    if extension not in SUPPORTED_EXTENSIONS:

        result["errors"].append(
            "Unsupported document type. "
            "Supported types are PDF, DOCX and TXT."
        )

        return result


    # --------------------------------------------------
    # VALIDATE FILE CONTENT
    # --------------------------------------------------

    if not file_bytes:

        result["errors"].append(
            "The uploaded document is empty."
        )

        return result


    if len(file_bytes) > MAX_FILE_SIZE_BYTES:

        result["errors"].append(
            "The uploaded document exceeds the "
            "10 MB size limit."
        )

        return result


    # --------------------------------------------------
    # CREATE DETERMINISTIC SOURCE ID
    # --------------------------------------------------

    source_id = (
        hashlib.sha256(
            file_bytes
        )
        .hexdigest()[:12]
    )


    result["source_id"] = source_id


    # --------------------------------------------------
    # EXTRACT DOCUMENT TEXT
    # --------------------------------------------------

    try:

        if extension == ".pdf":

            document_units, warnings = (
                _extract_pdf(
                    file_bytes
                )
            )


        elif extension == ".docx":

            document_units, warnings = (
                _extract_docx(
                    file_bytes
                )
            )


        else:

            document_units, warnings = (
                _extract_txt(
                    file_bytes
                )
            )


    except Exception as error:

        result["errors"].append(
            f"Document extraction failed: {error}"
        )

        return result


    result["warnings"].extend(
        warnings
    )


    # --------------------------------------------------
    # HANDLE DOCUMENTS WITH NO TEXT
    # --------------------------------------------------

    if not document_units:

        result["errors"].append(
            "No usable text could be extracted "
            "from the document."
        )

        return result


    # --------------------------------------------------
    # CHUNK EXTRACTED TEXT
    # --------------------------------------------------

    all_chunks = []

    global_chunk_index = 0


    for unit in document_units:

        text = unit["text"]

        page_number = (
            unit["page_number"]
        )


        try:

            text_chunks = (
                chunk_document_text(
                    text=text,
                    chunk_size_words=chunk_size_words,
                    overlap_words=overlap_words,
                )
            )

        except ValueError as error:

            result["errors"].append(
                str(error)
            )

            return result


        for text_chunk in text_chunks:

            global_chunk_index += 1


            chunk_metadata = {
                "chunk_id": (
                    f"{source_id}-"
                    f"c{global_chunk_index:04d}"
                ),

                "source_id": source_id,

                "source_name": filename,

                "file_type": extension,

                "page_number": page_number,

                "chunk_index": (
                    global_chunk_index
                ),

                "word_count": len(
                    text_chunk.split()
                ),

                "text": text_chunk,
            }


            all_chunks.append(
                chunk_metadata
            )


    # --------------------------------------------------
    # FINAL SAFETY CHECK
    # --------------------------------------------------

    if not all_chunks:

        result["errors"].append(
            "The document contained no usable "
            "chunks after preprocessing."
        )

        return result


    # --------------------------------------------------
    # PROCESSING SUMMARY
    # --------------------------------------------------

    result["chunks"] = all_chunks


    result["summary"] = {
        "total_chunks": len(
            all_chunks
        ),

        "total_words": sum(
            chunk["word_count"]
            for chunk in all_chunks
        ),

        "units_with_text": len(
            document_units
        ),

        "file_size_bytes": len(
            file_bytes
        ),

        "chunk_size_words": (
            chunk_size_words
        ),

        "overlap_words": (
            overlap_words
        ),
    }


    result["success"] = True


    return result


# --------------------------------------------------
# STREAMLIT UPLOAD WRAPPER
# --------------------------------------------------

def process_uploaded_document(
    uploaded_file,
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> dict:
    """
    Convenience wrapper for Streamlit UploadedFile.
    """

    if uploaded_file is None:

        return {
            "success": False,
            "source_name": None,
            "source_id": None,
            "file_type": None,
            "chunks": [],
            "errors": [
                "No document was uploaded."
            ],
            "warnings": [],
            "summary": {},
        }


    return process_document(
        file_bytes=uploaded_file.getvalue(),
        filename=uploaded_file.name,
        chunk_size_words=chunk_size_words,
        overlap_words=overlap_words,
    )