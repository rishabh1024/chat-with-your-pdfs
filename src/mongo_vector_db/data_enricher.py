import logging
import re

from langchain_core.documents import Document
from openrouter.errors.toomanyrequestsresponse_error import TooManyRequestsResponseError
from openrouter.errors.unauthorizedresponse_error import UnauthorizedResponseError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
    wait_random,
)

from core.logs import safe_before_sleep_log

logger = logging.getLogger(__name__)


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


@retry(
    retry=retry_if_exception_type(ValueError) | retry_if_exception_type(UnauthorizedResponseError),
    stop=stop_after_attempt(3),
    wait=wait_fixed(2) + wait_random(0, 3),
    before_sleep=safe_before_sleep_log(logger, "vector.metadata.generate.retry"),
    reraise=True,
)
async def _generate_metadata(structured_llm_instance, text: str) -> dict:
    try:
        metadata = await structured_llm_instance.ainvoke(text)
        if not isinstance(metadata, dict):
            raise ValueError("Metadata generated is not in the expected format.")
        return metadata
    except UnauthorizedResponseError as openrouter_error:
        raise RuntimeError("Metadata provider authorization failed.") from openrouter_error


async def add_metadata(
    chunks: list[Document],
    structured_llm_instance,
    document_id: str | None = None,
) -> list[Document]:
    if not chunks:
        raise ValueError(
            "No document chunks are available for metadata generation. Call split_pages() first."
        )

    try:
        for document_chunk in chunks:
            if document_chunk.metadata.get("document_id"):
                continue
            document_metadata = await _generate_metadata(
                structured_llm_instance,
                document_chunk.page_content,
            )
            if document_metadata:
                if isinstance(document_metadata, str):
                    document_metadata = strip_thinking(document_metadata)
                else:
                    document_chunk.metadata = dict(document_metadata)
            else:
                logger.debug(
                    "vector.metadata.generate.empty document_id=%s",
                    document_id,
                )
            if document_id:
                document_chunk.metadata["document_id"] = document_id
        logger.debug(
            "vector.metadata.generate.completed document_id=%s count=%s",
            document_id,
            len(chunks),
        )
        return chunks
    except TypeError as type_error:
        raise type_error
    except TooManyRequestsResponseError as too_many_requests_error:
        raise too_many_requests_error
    except Exception as error:
        raise RuntimeError(f"Failed to add metadata to document. Error Reason: {error}") from error
