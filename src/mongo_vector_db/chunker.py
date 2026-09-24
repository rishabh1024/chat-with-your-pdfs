import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

MIN_PAGE_WORD_COUNT = 10
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def clean_pages(pages: list[Document], *, document_id: str | None = None) -> list[str]:
    cleaned_documents = [
        page.page_content for page in pages if len(page.page_content.split()) > MIN_PAGE_WORD_COUNT
    ]
    logger.debug(
        "vector.document.clean.completed document_id=%s count=%s",
        document_id,
        len(cleaned_documents),
    )
    return cleaned_documents


def split_pages(texts: list[str], *, document_id: str | None = None) -> list[Document]:
    if not texts:
        raise ValueError("No cleaned documents to chunk. Call clean_pages() first.")

    document_to_chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunked_documents = document_to_chunk_splitter.split_documents(
        [Document(page_content=document) for document in texts]
    )
    logger.debug(
        "vector.document.chunk.completed document_id=%s count=%s",
        document_id,
        len(chunked_documents),
    )
    return chunked_documents
