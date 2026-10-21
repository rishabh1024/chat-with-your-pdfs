import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

MIN_PAGE_WORD_COUNT = 10
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def clean_pages(pages: list[Document], *,
                document_id: str | None = None) -> list[str]:
    if not pages:
        logger.warning(
            "vector.document.clean_pages.empty_input "
            "document_id=%s - Could no load any pages from PDF file",
            document_id,
        )
        return []

    cleaned_documents = [
        page.page_content for page in pages if len(page.page_content.split()) > MIN_PAGE_WORD_COUNT
    ]

    total_filtered_pages = len(pages) - len(cleaned_documents)
    if total_filtered_pages > 0:
        logger.info(
            "vector.document.clean.filtered document_id=%s total_pages=%s filtered_out=%s remaining=%s min_words=%s",
            document_id,
            len(pages),
            total_filtered_pages,
            len(cleaned_documents),
            MIN_PAGE_WORD_COUNT,
        )

    if not cleaned_documents:
        logger.warning(
            "vector.document.clean.all_filtered document_id=%s - all %s pages had ≤%s words",
            document_id,
            len(pages),
            MIN_PAGE_WORD_COUNT,
        )

    return cleaned_documents


def split_pages(clean_document_list: list[str], *,
                document_id: str | None = None) -> list[Document]:
    if not clean_document_list:
        raise ValueError(
            f"No content available to process for document_id={document_id}. "
            f"The PDF may be empty, contain only images/scanned content, "
            f"or all pages have fewer than {MIN_PAGE_WORD_COUNT} words."
        )

    document_to_chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunked_documents = document_to_chunk_splitter.split_documents(
        [Document(page_content=document) for document in clean_document_list]
    )
    logger.debug(
        "vector.document.chunk.completed document_id=%s count=%s",
        document_id,
        len(chunked_documents),
    )
    return chunked_documents
