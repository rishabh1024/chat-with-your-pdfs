import datetime
import logging
from typing import Any, Literal

from langchain_core.documents import Document
from openrouter.errors.toomanyrequestsresponse_error import TooManyRequestsResponseError
from pydantic import ValidationError
from pymongo.errors import ServerSelectionTimeoutError
from tenacity import (
    TryAgain,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
)

from core.logs import safe_before_sleep_log
from mongo_vector_db.chunker import clean_pages, split_pages
from mongo_vector_db.client import get_mongo_vector_db
from mongo_vector_db.data_enricher import add_metadata
from mongo_vector_db.loader import load_pdf
from mongo_vector_db.vector_store import create_embeddings

logger = logging.getLogger(__name__)


def get_document_index_status(document_id: str) -> dict[str, Any] | None:
    if not document_id:
        raise ValueError("document_id is required to store indexing status.")
    status_collection = get_mongo_vector_db().db["document_index_status"]
    return status_collection.find_one({"document_id": document_id})


def set_document_index_status(
    document_id: str,
    status: Literal["pending", "success", "failed"],
    message: str,
) -> None:
    if not document_id:
        raise ValueError("document_id is required to store indexing status.")
    now = datetime.datetime.now(datetime.UTC)
    status_collection = get_mongo_vector_db().db["document_index_status"]
    status_collection.update_one(
        {"document_id": document_id},
        {
            "$set": {
                "status": status,
                "message": message,
                "updated_at": now,
            },
            "$setOnInsert": {
                "created_at": now,
            },
        },
        upsert=True,
    )


class DocumentIndexer:
    def __init__(
        self,
        structured_llm_instance,
        file_path: str,
        document_id: str | None = None,
    ) -> None:
        self.file_path = file_path
        self.document_id = document_id
        self.chunked_documents: list[Document] = []
        self.structured_llm_instance = structured_llm_instance

    def _prepare_document_for_embedding_creation(self) -> None:
        pages = load_pdf(self.file_path, document_id=self.document_id)
        cleaned_documents = clean_pages(pages, document_id=self.document_id)
        self.chunked_documents = split_pages(cleaned_documents, document_id=self.document_id)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(TryAgain)
        | retry_if_exception_type(TooManyRequestsResponseError)
        | retry_if_exception_type(ServerSelectionTimeoutError),
        before_sleep=safe_before_sleep_log(logger, "vector.indexing.retry"),
    )
    async def convert_document_to_vector(self) -> dict[str, str]:
        logger.debug("vector.indexing.started document_id=%s", self.document_id)

        if not self.chunked_documents:
            self._prepare_document_for_embedding_creation()

        try:
            self.chunked_documents = await add_metadata(
                self.chunked_documents,
                self.structured_llm_instance,
                document_id=self.document_id,
            )
            embeddings_status = create_embeddings(
                self.chunked_documents,
                document_id=self.document_id,
            )
            if embeddings_status["status"] == "Success":
                logger.debug(
                    "vector.indexing.completed document_id=%s",
                    self.document_id,
                )
                return embeddings_status
            raise TryAgain
        except (ValidationError, TypeError) as error:
            raise error
        except TooManyRequestsResponseError as error:
            raise error
        except ServerSelectionTimeoutError as error:
            raise error
        except Exception as error:
            raise error
