import logging
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from starlette.formparsers import MultiPartParser

from auth.dependencies import get_current_authenticated_user
from auth.models import AuthenticatedUser
from file_upload.dependencies import get_file_upload_service, get_indexing_status_tracker
from file_upload.models import FileUploadResponse
from file_upload.service import FileUploadService
from mongo_vector_db.indexing_tracker import IndexingStatusTracker

# By changing this parameter, we can tell the upload file method to store file above
# max_part_size in the disk and not in memory
MultiPartParser.max_part_size = 1024 * 1024 * 2

REQUIRED_FILE = File(...)

router = APIRouter(prefix="/upload", tags=["file upload"])

logger = logging.getLogger(__name__)

FileUploadSvc = Annotated[FileUploadService, Depends(get_file_upload_service)]
IndexingTracker = Annotated[IndexingStatusTracker, Depends(get_indexing_status_tracker)]


@router.post("/", response_model=FileUploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    service: FileUploadSvc,
    current_user: Annotated[AuthenticatedUser, Depends(get_current_authenticated_user)],
    input_file: UploadFile = REQUIRED_FILE,
) -> FileUploadResponse:
    return await service.upload_document(
        uploaded_file=input_file,
        current_user=current_user,
        background_tasks=background_tasks,
    )


@router.get("/index_status/{document_id}/events")
async def index_status_sse(
    document_id: str,
    current_user: Annotated[AuthenticatedUser, Depends(get_current_authenticated_user)],
    indexing_tracker: IndexingTracker,
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
