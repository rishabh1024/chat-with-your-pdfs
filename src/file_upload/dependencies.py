from typing import Annotated

from fastapi import Depends, Request

from file_upload.protocols import StorageClient
from file_upload.service import FileUploadService
from mongo_vector_db.indexing_tracker import IndexingStatusTracker

_indexing_tracker = IndexingStatusTracker()


def get_storage_client(request: Request) -> StorageClient:
    return request.app.state.file_upload_client


def get_indexing_status_tracker() -> IndexingStatusTracker:
    return _indexing_tracker


def get_file_upload_service(
    storage_client: Annotated[StorageClient, Depends(get_storage_client)],
    indexing_tracker: Annotated[IndexingStatusTracker, Depends(get_indexing_status_tracker)],
) -> FileUploadService:
    return FileUploadService(storage_client, indexing_tracker)
