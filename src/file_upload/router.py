import logging
import tempfile
from collections.abc import AsyncGenerator
from typing import Annotated, cast
from uuid import uuid4

import httpx
import pymupdf
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from langchain_openrouter import ChatOpenRouter
from pymongo.errors import PyMongoError
from src.auth.dependencies import get_current_authenticated_user
from src.auth.models import AuthenticatedUser
from src.core.settings import load_environment_variables
from src.file_upload.file_upload_client import FileUploadClient
from src.mongo_vector_db.data_wrangler import DocumentIndexer
from src.mongo_vector_db.indexing_tracker import IndexingStatusTracker
from storage3.exceptions import StorageApiError
from supabase import SupabaseException

from .models import FileUploadResponse, StorageUploadResponse

settings = load_environment_variables()

REQUIRED_FILE = File(...)

router = APIRouter(prefix="/upload", tags=["file upload"])

logger = logging.getLogger(__name__)

indexing_tracker = IndexingStatusTracker()

schema = {
    "title": "DocumentMetadata",
    "properties": {
        "title": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "hasCode": {"type": "boolean"},
    },
    "required": ["title", "keywords", "hasCode"],
}


def get_file_uploader(request: Request) -> FileUploadClient:
    return request.app.state.file_upload_client


def create_openrouter_chat(
    *,
    api_key: str,
    model: str,
    temperature: float = 0.3,
    **kwargs,
) -> ChatOpenRouter:
    return ChatOpenRouter(
        model=model,
        api_key=settings.openrouter.openrouter_api_key,
        temperature=temperature,
        model_kwargs={
            "models": [
                "qwen/qwen3-next-80b-a3b-instruct:free",
                "poolside/laguna-xs-2.1:free",
                "meta-llama/llama-3.2-3b-instruct:free",
            ]
        },
        **kwargs,
    )


def get_llm_client_for_structured_output(
    *,
    model: str = "qwen/qwen3-30b-a3b-instruct-2507",
    temperature: float = 0.3,
):
    llm_client = ChatOpenRouter(
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
    ).with_structured_output(schema, method="json_schema")
    return llm_client


@router.post("/", response_model=FileUploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    authenticated_user: Annotated[AuthenticatedUser, Depends(get_current_authenticated_user)],
    file_uploader: Annotated[FileUploadClient, Depends(get_file_uploader)],
    input_file: UploadFile = REQUIRED_FILE,
) -> FileUploadResponse:
    file_contents_of_uploaded_file = await input_file.read()

    hash_of_file_contents = FileUploadClient.calculate_file_hash(file_contents_of_uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
        temp.write(file_contents_of_uploaded_file)
        temp_file_path = temp.name

    structured_llm = cast(ChatOpenRouter, get_llm_client_for_structured_output())

    document_indexer = DocumentIndexer(
        structured_llm_instance=structured_llm,
        file_path=temp_file_path,
        document_id=hash_of_file_contents,
    )

    try:
        document_indexer.prepare_document_for_embedding_creation()
    except FileNotFoundError as e:
        logger.warning(
            "upload.preparation.rejected document_id=%s user_id=%s reason=file_not_found",
            hash_of_file_contents,
            authenticated_user.user_id,
        )
        raise HTTPException(
            status_code=404,
            detail="The uploaded document could not be found.",
        ) from e
    except (pymupdf.EmptyFileError, ValueError) as e:
        logger.warning(
            "upload.preparation.rejected document_id=%s user_id=%s reason=invalid_pdf",
            hash_of_file_contents,
            authenticated_user.user_id,
        )
        raise HTTPException(
            status_code=422,
            detail="The uploaded file is empty or invalid.",
        ) from e
    except pymupdf.FileDataError as e:
        logger.warning(
            "upload.preparation.rejected document_id=%s user_id=%s reason=corrupt_pdf",
            hash_of_file_contents,
            authenticated_user.user_id,
        )
        raise HTTPException(
            status_code=422,
            detail="The uploaded PDF is corrupted or invalid.",
        ) from e
    except RuntimeError as e:
        logger.error(
            "upload.preparation.failed document_id=%s user_id=%s error_type=%s",
            hash_of_file_contents,
            authenticated_user.user_id,
            type(e).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="The document could not be prepared for indexing.",
        ) from e

    logger.info(
        "upload.preparation.completed document_id=%s user_id=%s",
        hash_of_file_contents,
        authenticated_user.user_id,
    )

    try:
        file_upload_response = await upload_pdf_file(
            input_file=input_file,
            file_contents=file_contents_of_uploaded_file,
            file_uploader=file_uploader,
            file_hash=hash_of_file_contents,
        )
    except (httpx.HTTPError, StorageApiError, SupabaseException) as error:
        logger.error(
            "upload.storage.failed document_id=%s user_id=%s error_type=%s",
            hash_of_file_contents,
            authenticated_user.user_id,
            type(error).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail="File storage is temporarily unavailable.",
        ) from error

    if file_upload_response.upload_status == "Failed":
        raise HTTPException(
            status_code=503,
            detail="File Storage is temporarily unavailable.",
        )
    storage_event_level = (
        logging.ERROR if file_upload_response.upload_status == "Failed" else logging.INFO
    )
    logger.log(
        storage_event_level,
        "upload.storage.completed document_id=%s user_id=%s status=%s",
        hash_of_file_contents,
        authenticated_user.user_id,
        file_upload_response.upload_status.lower().replace(" ", "_"),
    )

    try:
        document_indexing_status = DocumentIndexer.get_status_for_document_indexing(
            document_id=hash_of_file_contents
        )
    except PyMongoError as error:
        logger.error(
            "upload.indexing_status.retrieve.failed document_id=%s user_id=%s error_type=%s",
            hash_of_file_contents,
            authenticated_user.user_id,
            type(error).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail="Failed to get document index status. Service is temporarily unavailable.",
        ) from error

    """
    The file upload was successful and it has not been indexed yet.
    The file was already uploaded but the indexing failed due to some reason.
    """
    should_schedule_indexing = file_upload_response.upload_status == "Success" or (
        file_upload_response.upload_status == "Already Exists"
        and (document_indexing_status is None or document_indexing_status["status"] == "failed")
    )

    if should_schedule_indexing:
        try:
            DocumentIndexer.set_status_for_document_indexing(
                document_id=hash_of_file_contents,
                status="pending",
                message=f"Indexing is in progress for document: {hash_of_file_contents}",
            )
        except PyMongoError as error:
            logger.error(
                "upload.indexing_status.schedule.failed document_id=%s user_id=%s error_type=%s",
                hash_of_file_contents,
                authenticated_user.user_id,
                type(error).__name__,
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "Document could not be scheduled for indexing. "
                    "Could not connect to the database service."
                ),
            ) from error

        indexing_tracker.register(hash_of_file_contents)
        background_tasks.add_task(
            _run_document_indexing_task, document_indexer, hash_of_file_contents
        )
        logger.info(
            "upload.indexing.scheduled document_id=%s user_id=%s",
            hash_of_file_contents,
            authenticated_user.user_id,
        )

        document_to_vector_status = {
            "status": "Pending",
            "message": "The file has been uploaded. Document Indexing is in process. "
            f"Connect to /index_status/{document_indexer.document_id}/events"
            "for real-time updates.",
        }
    elif (
        file_upload_response.upload_status == "Already Exists"
        and document_indexing_status is not None
        and document_indexing_status["status"] == "success"
    ):
        logger.debug(
            "upload.indexing.skipped document_id=%s user_id=%s reason=already_indexed",
            hash_of_file_contents,
            authenticated_user.user_id,
        )
        document_to_vector_status = {
            "status": "Success",
            "message": "Document Indexing is already indexed",
        }
    elif (
        file_upload_response.upload_status == "Already Exists"
        and document_indexing_status is not None
        and document_indexing_status["status"] == "pending"
    ):
        logger.debug(
            "upload.indexing.skipped document_id=%s user_id=%s reason=already_pending",
            hash_of_file_contents,
            authenticated_user.user_id,
        )
        document_to_vector_status = {
            "status": "Pending",
            "message": "Document indexing is already in progress.",
        }
    else:
        logger.warning(
            "upload.indexing.skipped document_id=%s user_id=%s reason=storage_failed",
            hash_of_file_contents,
            authenticated_user.user_id,
        )
        document_to_vector_status = {
            "status": "Failed",
            "message": "File Upload failed. The file has not been indexed in the vector database.",
        }

    return FileUploadResponse(
        document_id=file_upload_response.document_id,
        file_hash=file_upload_response.file_hash,
        upload_status=file_upload_response.upload_status,
        upload_error=file_upload_response.upload_error,
        document_indexing_status=document_to_vector_status,
    )


async def _run_document_indexing_task(document_indexer: DocumentIndexer, document_id: str) -> None:
    try:
        logger.info("upload.indexing.started document_id=%s", document_id)
        await document_indexer.convert_document_to_vector()

        DocumentIndexer.set_status_for_document_indexing(
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

        DocumentIndexer.set_status_for_document_indexing(
            document_id=document_id,
            status="failed",
            message=f"Background indexing failed for document: {document_id}",
        )

        indexing_tracker.mark_complete(
            document_id, status="failed", message="Document indexing failed."
        )


@router.get("/index_status/{document_id}/events")
async def index_status_sse(
    document_id: str,
    current_user: Annotated[AuthenticatedUser, Depends(get_current_authenticated_user)],
) -> StreamingResponse:
    if not indexing_tracker.is_registered(document_id):
        raise HTTPException(
            status_code=404,
            detail="No indexing task was found for this document.",
        )

    async def event_stream() -> AsyncGenerator[str, None]:
        yield "event: connected\ndata: Waiting for indexing result...\n\n"
        result = await indexing_tracker.wait_for_result(document_id)
        yield f"event: indexing_complete\ndata: {result}\n\n"
        indexing_tracker.cleanup(document_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def upload_pdf_file(
    input_file: UploadFile, file_contents: bytes, file_uploader: FileUploadClient, file_hash
) -> StorageUploadResponse:

    original_file_name_string = input_file.filename

    if not original_file_name_string:
        original_file_name_string = f"{uuid4().int}.pdf"

    return await file_uploader.upload_to_file_storage(
        uploaded_file_content_in_bytes=file_contents,
        original_filename=original_file_name_string,
        file_hash=file_hash,
    )
