import hashlib
from pathlib import Path

from file_upload.exceptions import UnsupportedFileTypeError

PDF_CONTENT_TYPES = frozenset(
    {
        "application/pdf",
        "application/x-pdf",
        "application/octet-stream",
        "binary/octet-stream",
    }
)


def calculate_file_hash(file_contents: bytes) -> str:
    """Return the SHA-256 digest used as the document's stable ID."""
    return hashlib.sha256(file_contents).hexdigest()


def validate_uploaded_file(*, filename: str | None, content_type: str | None) -> None:
    """Accept a file when its type is supported. Add a new case to allow another type."""
    file_extension = Path(filename).suffix.lower() if filename else ""
    normalized_content_type = (
        content_type.lower().split(";", maxsplit=1)[0].strip() if content_type else None
    )

    match file_extension:
        case ".pdf":
            if (
                normalized_content_type is not None
                and normalized_content_type not in PDF_CONTENT_TYPES
            ):
                raise UnsupportedFileTypeError("The uploaded PDF has an unsupported content type.")
        case _:
            raise UnsupportedFileTypeError(
                "Only PDF files are supported. Please upload a valid .pdf document."
            )
