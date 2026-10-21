import logging
from pathlib import Path
from typing import Literal

import pymupdf
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from core.logs import safe_before_sleep_log

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(2),
    retry=retry_if_exception_type(pymupdf.FileDataError),
    before_sleep=safe_before_sleep_log(logger, "vector.document.load.retry"),
)
def load_pdf(
    file_path: str,
    *,
    document_id: str | None = None,
    page_mode: Literal["page", "single"] = "page",
) -> list[Document]:
    if not Path(file_path).is_file():
        raise FileNotFoundError("The PDF file could not be found. Please try uploading again.")

    try:
        loader = PyMuPDFLoader(file_path, mode=page_mode)
        documents = loader.load()

        # Log details about extracted content
        total_chars = sum(len(doc.page_content) for doc in documents)
        non_empty_pages = sum(1 for doc in documents if doc.page_content.strip())
        logger.info(
            "vector.document.load.completed document_id=%s pages=%s non_empty_pages=%s total_chars=%s",
            document_id,
            len(documents),
            non_empty_pages,
            total_chars,
        )

        if total_chars == 0:
            logger.warning(
                "vector.document.load.no_text document_id=%s - PDF has %s pages but no extractable text (may be scanned/image-only)",
                document_id,
                len(documents),
            )

        return documents
    except ValueError:
        loader = PyMuPDFLoader(file_path, mode="page")
        documents = loader.load()

        total_chars = sum(len(doc.page_content) for doc in documents)
        non_empty_pages = sum(1 for doc in documents if doc.page_content.strip())
        logger.info(
            "vector.document.load.completed document_id=%s pages=%s non_empty_pages=%s total_chars=%s mode=fallback",
            document_id,
            len(documents),
            non_empty_pages,
            total_chars,
        )

        return documents
    except pymupdf.EmptyFileError as empty_file_error:
        raise pymupdf.EmptyFileError(
            "The file uploaded is empty. ",
            """Try uploading a new file.""",
        ) from empty_file_error
    except pymupdf.FileDataError as file_data_error:
        raise pymupdf.FileDataError(
            """The file uploaded is either corrupted or not valid.Check for your file"""
            """before uploading and try again."""
        ) from file_data_error
    except Exception as error:
        raise RuntimeError(f"Error Loading Document: {error}") from error
