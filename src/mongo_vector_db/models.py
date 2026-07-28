from typing import Literal

from pydantic import UUID4, BaseModel


class StorageUploadResponse(BaseModel):
    document_id: str
    file_hash: str
    storage_path: str
    original_filename: str
    upload_status: Literal["Success", "Failed"]
    upload_error: str | None


class FileUploadResponse(BaseModel):
    document_id: str
    file_hash: str
    upload_status: Literal["Success", "Failed"]
    upload_error: str | None
    document_indexing_status: dict[str, str]

class ChatResponseModel(BaseModel):
    
    chat_id: UUID4
    ai_message: str
    chat_history_messages: list[str]