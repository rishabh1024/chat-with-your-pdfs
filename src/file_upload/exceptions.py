from typing import Literal

from fastapi import status
from fastapi.responses import JSONResponse
from pymongo.errors import ConnectionFailure, PyMongoError, ServerSelectionTimeoutError

IndexFailureState = Literal["status_check_failed", "schedule_failed"]


class StorageUnavailableError(Exception):
    """Raised when file storage cannot be reached or the upload fails."""

    def __init__(self, error_msg: str = "File storage is temporarily unavailable.") -> None:
        self.error = error_msg
        super().__init__(self.error)


class IndexingStatusUnavailableError(Exception):
    """Raised when document indexing status cannot be read or written."""

    def __init__(
        self,
        error_msg: str,
        *,
        reason_code: str = "INDEXING_DATABASE_ERROR",
        storage_completed: bool = False,
        index_failure_state: IndexFailureState = "status_check_failed",
    ) -> None:
        self.error = error_msg
        self.reason_code = reason_code
        self.storage_completed = storage_completed
        self.index_failure_state = index_failure_state
        super().__init__(self.error)

    @classmethod
    def from_pymongo_error(
        cls,
        error: PyMongoError,
        *,
        index_failure_state: IndexFailureState,
        storage_completed: bool,
    ) -> "IndexingStatusUnavailableError":
        reason_code, detail = _consumer_indexing_message(
            error,
            index_failure_state=index_failure_state,
            storage_completed=storage_completed,
        )
        return cls(
            detail,
            reason_code=reason_code,
            storage_completed=storage_completed,
            index_failure_state=index_failure_state,
        )


class UnsupportedFileTypeError(Exception):
    """Raised when the uploaded file type is not accepted."""

    def __init__(
        self,
        error_msg: str = "This file type is not supported.",
    ) -> None:
        self.error = error_msg
        super().__init__(self.error)


def _consumer_indexing_message(
    error: PyMongoError,
    *,
    index_failure_state: IndexFailureState,
    storage_completed: bool,
) -> tuple[str, str]:
    if isinstance(error, (ServerSelectionTimeoutError, ConnectionFailure)):
        reason_code = "INDEXING_DATABASE_UNREACHABLE"
        if storage_completed and index_failure_state == "status_check_failed":
            detail = (
                "Your file was stored successfully, but indexing status could not be checked "
                "because the document database is temporarily unreachable. "
                "Please try again shortly."
            )
        elif storage_completed:
            detail = (
                "Your file was stored successfully, but indexing could not be started "
                "because the document database is temporarily unreachable. "
                "Please try again shortly."
            )
        else:
            detail = "The document database is temporarily unreachable. Please try again shortly."
        return reason_code, detail

    reason_code = "INDEXING_DATABASE_ERROR"
    if storage_completed and index_failure_state == "status_check_failed":
        detail = (
            "Your file was stored successfully, but indexing status could not be checked. "
            "Please try again shortly."
        )
    elif storage_completed:
        detail = (
            "Your file was stored successfully, but indexing could not be started. "
            "Please try again shortly."
        )
    else:
        detail = "Document indexing is temporarily unavailable. Please try again shortly."
    return reason_code, detail


async def storage_unavailable_error_handler(request, exc: StorageUnavailableError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "File Storage Unavailable",
            "detail": exc.error,
        },
    )


async def indexing_status_unavailable_error_handler(
    request,
    exc: IndexingStatusUnavailableError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "Indexing Database Unavailable",
            "detail": exc.error,
            "reason_code": exc.reason_code,
            "storage_completed": exc.storage_completed,
            "index_failure_state": exc.index_failure_state,
        },
    )


async def unsupported_file_type_error_handler(
    request,
    exc: UnsupportedFileTypeError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Unsupported File Type",
            "detail": exc.error,
        },
    )
