from uuid import UUID

from fastapi import status
from fastapi.responses import JSONResponse


class ConversationNotFoundError(Exception):
    def __init__(self, conversation_id: UUID, error_msg: str) -> None:
        self.conversation_id = conversation_id
        self.error = error_msg or f"Conversation {conversation_id} Not Found"
        super().__init__(self.error)


class ConversationAccessDeniedError(Exception):
    def __init__(self, conversation_id: UUID, error_msg: str) -> None:
        self.conversation_id = conversation_id
        self.error = error_msg or f"Access Denied for Conversation {conversation_id}"
        super().__init__(self.error)


class DatabaseUnavailableError(Exception):
    """Raised when conversation data cannot be read or written from the database."""

    def __init__(self, error_msg: str = "Conversation data is temporarily unavailable.") -> None:
        self.error = error_msg
        super().__init__(self.error)



async def conversation_not_found_exception_handler(request, exc) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "error": "Conversation Not Found",
            "detail": f"{exc.error}",
        },
    )

async def conversation_access_denied_error_handler(request, exc)  -> JSONResponse:

    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={
            "error": "User cannot access this conversation.",
            "detail": f"{exc.error}",
        },
    )


async def database_unavailable_error_handler(request, exc) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "Database is Not Reachable.",
            "detail": exc.error,
        },
    )