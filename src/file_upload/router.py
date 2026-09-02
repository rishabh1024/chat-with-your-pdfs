import asyncio
import logging
import tempfile
from collections.abc import AsyncGenerator
from typing import Annotated, cast
from uuid import uuid4

import pymupdf
from dotenv import load_dotenv
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from langchain_openrouter import ChatOpenRouter
from pydantic import SecretStr
from src.auth.dependencies import get_current_authenticated_user
from src.auth.models import AuthenticatedUser
from src.file_upload.file_upload_client import FileUploadClient
from src.mongo_vector_db.data_wrangler import DocumentIndexer
from src.mongo_vector_db.indexing_tracker import IndexingStatusTracker
from src.mongo_vector_db.main import MongoVectorDB

from .models import FileUploadResponse, StorageUploadResponse

load_dotenv()


REQUIRED_FILE = File(...)

router = APIRouter(
    prefix="/upload",
    tags=["file upload"])

logger = logging.getLogger(__name__)

indexing_tracker = IndexingStatusTracker()

def get_file_uploader(request: Request) -> FileUploadClient:
    return request.app.state.file_upload_client


def get_mongodb_client(request: Request) -> MongoVectorDB:
    return request.app.state.mongodb_client

def get_llm_client_for_structured_response(request: Request) -> ChatOpenRouter:
    return request.app.state.structured_llm_client

def create_openrouter_chat(
    *,
    api_key: str,
    model: str,
    temperature: float = 0.3,
    **kwargs,
) -> ChatOpenRouter:
    return ChatOpenRouter(
        model=model,
        api_key=SecretStr(api_key),
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


@router.post("/", response_model=FileUploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    authenticated_user: Annotated[AuthenticatedUser, Depends(get_current_authenticated_user)],
    file_uploader: Annotated[FileUploadClient, Depends(get_file_uploader)],
    structured_llm: Annotated[ChatOpenRouter, Depends(get_llm_client_for_structured_response)],
    input_file: UploadFile = REQUIRED_FILE,
) -> FileUploadResponse:

    print(f"User has uploaded a file named {input_file.filename}")

    file_contents_of_uploaded_file = await input_file.read()

    hash_of_file_contents = FileUploadClient.calculate_file_hash(file_contents_of_uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
        temp.write(file_contents_of_uploaded_file)
        temp_file_path = temp.name

    document_indexer = DocumentIndexer(
        structured_llm_instance=structured_llm,
        file_path=temp_file_path,
        document_id=hash_of_file_contents,
    )

    try:
        document_indexer.prepare_document_for_embedding_creation()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except (pymupdf.EmptyFileError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except pymupdf.FileDataError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    file_upload_response = await upload_pdf_file(
        input_file=input_file,
        file_contents=file_contents_of_uploaded_file,
        file_uploader=file_uploader,
        file_hash=hash_of_file_contents
    )

    document_indexing_status = DocumentIndexer.get_status_for_document_indexing(
        document_id=hash_of_file_contents
    )

    """
    The file upload was successful and it has not been indexed yet.
    The file was already uploaded but the indexing failed due to some reason.
    """
    should_schedule_indexing = file_upload_response.upload_status == "Success" or (
        file_upload_response.upload_status == "Already Exists"
        and (document_indexing_status is None or document_indexing_status["status"] == "failed")
    )

    if should_schedule_indexing:
        DocumentIndexer.set_status_for_document_indexing(
            document_id=hash_of_file_contents,
            status="pending",
            message=f"Indexing is in progress for document: {hash_of_file_contents}",
        )

        indexing_tracker.register(hash_of_file_contents)
        background_tasks.add_task(
            _run_document_indexing_task, document_indexer, hash_of_file_contents
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
        document_to_vector_status = {
            "status": "Success",
            "message": "Document Indexing is already indexed",
        }
    elif (
        file_upload_response.upload_status == "Already Exists"
        and document_indexing_status is not None
        and document_indexing_status["status"] == "pending"
    ):
        document_to_vector_status = {
            "status": "Pending",
            "message": "Document indexing is already in progress.",
        }
    else:
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


async def _run_document_indexing_task(document_indexer: DocumentIndexer,
                                      document_id: str) -> None:
    try:
        await document_indexer.convert_document_to_vector()

        DocumentIndexer.set_status_for_document_indexing(
            document_id=document_id,
            status="success",
            message=f"Document indexed successfully. Document: {document_id}",
        )

        indexing_tracker.mark_complete(
            document_id, status="success", message="Document indexed successfully."
        )

    except Exception as e:
        logger.exception("Background indexing failed for document %s", document_id)

        DocumentIndexer.set_status_for_document_indexing(
            document_id=document_id,
            status="failed",
            message=f"Background indexing failed for document: {document_id}",
        )

        indexing_tracker.mark_complete(
            document_id, status="failed", message=f"Indexing failed: {e}"
        )


@router.get("/index_status/{document_id}/events")
async def index_status_sse(
    document_id: str,
    current_user: Annotated[AuthenticatedUser, Depends(get_current_authenticated_user)],
) -> StreamingResponse:

    print("Current User: ", current_user.user_id, current_user.claims)

    async def event_stream() -> AsyncGenerator[str, None]:
        yield "event: connected\ndata: Waiting for indexing result...\n\n"
        result = await indexing_tracker.wait_for_result(document_id)
        yield f"event: indexing_complete\ndata: {result}\n\n"
        indexing_tracker.cleanup(document_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},)


async def upload_pdf_file(
    input_file: UploadFile, file_contents: bytes,
    file_uploader: FileUploadClient, file_hash) -> StorageUploadResponse:

    original_file_name_string = input_file.filename

    if not original_file_name_string:
        original_file_name_string = f"{uuid4().int}.pdf"

    file_uploading_task = asyncio.create_task(
        file_uploader.upload_to_file_storage(
            uploaded_file_content_in_bytes=file_contents,
            original_filename=original_file_name_string,
            file_hash=file_hash,
        )
    )

    file_uploading_status = await asyncio.gather(file_uploading_task)
    file_uploading_status = cast(StorageUploadResponse, await file_uploading_task)

    return file_uploading_status
