import logging
import tempfile
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi import BackgroundTasks, UploadFile
from langchain_openrouter import ChatOpenRouter
from pymongo.errors import PyMongoError
from storage3.exceptions import StorageApiError
from supabase import SupabaseException

from auth.models import AuthenticatedUser
from core.settings import load_environment_variables
from file_upload.exceptions import IndexingStatusUnavailableError, StorageUnavailableError
from file_upload.helpers import calculate_file_hash, validate_uploaded_file
from file_upload.models import FileUploadResponse, StorageUploadResponse
from file_upload.protocols import StorageClient
from mongo_vector_db.document_indexing import (
    DocumentIndexer,
    get_document_index_status,
    set_document_index_status,
)
from mongo_vector_db.indexing_tracker import IndexingStatusTracker

settings = load_environment_variables()

logger = logging.getLogger(__name__)

DOCUMENT_METADATA_SCHEMA = {
    "title": "DocumentMetadata",
    "properties": {
        "title": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "hasCode": {"type": "boolean"},
    },
    "required": ["title", "keywords", "hasCode"],
}


class FileUploadService:
    def __init__(
        self,
        storage_client: StorageClient,
        indexing_tracker: IndexingStatusTracker,
    ) -> None:
        self._storage_client = storage_client
        self._indexing_tracker = indexing_tracker

    def _create_structured_llm(
        self,
        model: str = "qwen/qwen3-30b-a3b-instruct-2507",
        temperature: float = 0.3,
    ):
        return ChatOpenRouter(
            model=model,
            api_key=settings.openrouter.openrouter_api_key,
            model_kwargs={
                "models": [
                    "qwen/qwen-2.5-7b-instruct",
                    "openai/gpt-oss-20b:free",
                    "meta-llama/llama-3.2-3b-instruct:free",
                ]
            },
            verbose=True,
            temperature=temperature,
        ).with_structured_output(DOCUMENT_METADATA_SCHEMA, method="json_schema")

    def _create_document_indexer(self, temp_file_path: str, document_id: str) -> DocumentIndexer:
        return DocumentIndexer(
            structured_llm_instance=self._create_structured_llm(),
            file_path=temp_file_path,
            document_id=document_id,
        )

    @staticmethod
    def _write_to_temp_file(file_contents: bytes) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_contents)
            return temp_file.name

    async def _upload_to_storage(
        self,
        uploaded_file: UploadFile,
        file_contents: bytes,
        file_hash: str,
    ) -> StorageUploadResponse:
        original_filename = uploaded_file.filename
        if not original_filename:
            original_filename = f"{uuid4().int}.pdf"

        return await self._storage_client.upload(
            file_contents=file_contents,
            original_filename=original_filename,
            file_hash=file_hash,
        )

    async def upload_document(
        self,
        uploaded_file: UploadFile,
        current_user: AuthenticatedUser,
        background_tasks: BackgroundTasks,
    ) -> FileUploadResponse:
        file_contents = await uploaded_file.read()
        validate_uploaded_file(
            filename=uploaded_file.filename,
            content_type=uploaded_file.content_type,
        )
        file_hash = calculate_file_hash(file_contents)
        document_id = file_hash
        temp_file_path = self._write_to_temp_file(file_contents)
        document_indexer = self._create_document_indexer(temp_file_path, document_id)

        try:
            storage_response = await self._upload_to_storage(
                uploaded_file=uploaded_file,
                file_contents=file_contents,
                file_hash=file_hash,
            )
        except (httpx.HTTPError, StorageApiError, SupabaseException) as error:
            Path(temp_file_path).unlink(missing_ok=True)
            logger.error(
                "upload.storage.failed document_id=%s user_id=%s error_type=%s",
                document_id,
                current_user.user_id,
                type(error).__name__,
            )
            raise StorageUnavailableError() from error

        if storage_response.upload_status == "Failed":
            Path(temp_file_path).unlink(missing_ok=True)
            raise StorageUnavailableError()

        storage_event_level = (
            logging.ERROR if storage_response.upload_status == "Failed" else logging.INFO
        )
        logger.log(
            storage_event_level,
            "upload.storage.completed document_id=%s user_id=%s status=%s",
            document_id,
            current_user.user_id,
            storage_response.upload_status.lower().replace(" ", "_"),
        )

        storage_completed = storage_response.upload_status in ("Success", "Already Exists")

        try:
            document_indexing_status = get_document_index_status(document_id=document_id)
        except PyMongoError as error:
            Path(temp_file_path).unlink(missing_ok=True)
            logger.error(
                "upload.indexing_status.retrieve.failed document_id=%s user_id=%s error_type=%s",
                document_id,
                current_user.user_id,
                type(error).__name__,
            )
            raise IndexingStatusUnavailableError.from_pymongo_error(
                error,
                index_failure_state="status_check_failed",
                storage_completed=storage_completed,
            ) from error

        # Schedule indexing when upload succeeded, or when the file already exists
        # but indexing never completed / previously failed.
        should_schedule_indexing = storage_response.upload_status == "Success" or (
            storage_response.upload_status == "Already Exists"
            and (document_indexing_status is None or document_indexing_status["status"] == "failed")
        )

        if should_schedule_indexing:
            try:
                set_document_index_status(
                    document_id=document_id,
                    status="pending",
                    message=f"Indexing is in progress for document: {document_id}",
                )
            except PyMongoError as error:
                Path(temp_file_path).unlink(missing_ok=True)
                logger.error(
                    "upload.indexing_status.schedule.failed "
                    "document_id=%s user_id=%s error_type=%s",
                    document_id,
                    current_user.user_id,
                    type(error).__name__,
                )
                raise IndexingStatusUnavailableError.from_pymongo_error(
                    error,
                    index_failure_state="schedule_failed",
                    storage_completed=storage_completed,
                ) from error

            self._indexing_tracker.register(document_id)
            background_tasks.add_task(
                run_document_indexing_task,
                document_indexer,
                document_id,
                self._indexing_tracker,
            )
            logger.info(
                "upload.indexing.scheduled document_id=%s user_id=%s",
                document_id,
                current_user.user_id,
            )

            indexing_status = {
                "status": "Pending",
                "message": (
                    "The file has been uploaded. Document Indexing is in process. "
                    f"Connect to /index_status/{document_id}/events for real-time updates."
                ),
            }
        elif (
            storage_response.upload_status == "Already Exists"
            and document_indexing_status is not None
            and document_indexing_status["status"] == "success"
        ):
            Path(temp_file_path).unlink(missing_ok=True)
            logger.debug(
                "upload.indexing.skipped document_id=%s user_id=%s reason=already_indexed",
                document_id,
                current_user.user_id,
            )
            indexing_status = {
                "status": "Success",
                "message": "Document Indexing is already indexed",
            }
        elif (
            storage_response.upload_status == "Already Exists"
            and document_indexing_status is not None
            and document_indexing_status["status"] == "pending"
        ):
            Path(temp_file_path).unlink(missing_ok=True)
            logger.debug(
                "upload.indexing.skipped document_id=%s user_id=%s reason=already_pending",
                document_id,
                current_user.user_id,
            )
            indexing_status = {
                "status": "Pending",
                "message": "Document indexing is already in progress.",
            }
        else:
            Path(temp_file_path).unlink(missing_ok=True)
            logger.warning(
                "upload.indexing.skipped document_id=%s user_id=%s reason=storage_failed",
                document_id,
                current_user.user_id,
            )
            indexing_status = {
                "status": "Failed",
                "message": (
                    "File Upload failed. The file has not been indexed in the vector database."
                ),
            }

        return FileUploadResponse(
            document_id=storage_response.document_id,
            file_hash=storage_response.file_hash,
            upload_status=storage_response.upload_status,
            upload_error=storage_response.upload_error,
            document_indexing_status=indexing_status,
        )


async def run_document_indexing_task(
    document_indexer: DocumentIndexer,
    document_id: str,
    indexing_tracker: IndexingStatusTracker,
) -> None:
    temp_file_path = document_indexer.file_path
    try:
        logger.info("upload.indexing.started document_id=%s", document_id)
        await document_indexer.convert_document_to_vector()

        set_document_index_status(
            document_id=document_id,
            status="success",
            message=f"Document indexed successfully. Document: {document_id}",
        )

        indexing_tracker.mark_complete(
            document_id, status="success", message="Document indexed successfully."
        )
        logger.info("upload.indexing.completed document_id=%s", document_id)

    except Exception as error:
        logger.error(
            "upload.indexing.failed document_id=%s error_type=%s",
            document_id,
            type(error).__name__,
        )

        set_document_index_status(
            document_id=document_id,
            status="failed",
            message=f"Background indexing failed for document: {document_id}",
        )

        indexing_tracker.mark_complete(
            document_id, status="failed", message="Document indexing failed."
        )
    finally:
        Path(temp_file_path).unlink(missing_ok=True)
