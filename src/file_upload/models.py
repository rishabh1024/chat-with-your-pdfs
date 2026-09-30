from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, field_serializer

IST = ZoneInfo("Asia/Kolkata")


class StorageUploadResponse(BaseModel):
    document_id: str
    file_hash: str
    storage_path: str
    original_filename: str
    upload_status: Literal["Success", "Failed", "Already Exists"]
    upload_error: str | None


class FileUploadResponse(BaseModel):
    document_id: str
    file_hash: str
    upload_status: Literal["Success", "Failed", "Already Exists"]
    upload_error: str | None
    document_indexing_status: dict[str, str]

class UsersDocuments(BaseModel):
    document_id: str
    document_name: str
    created_at: datetime

    @field_serializer("created_at")
    def serialize_in_ist(self, value: datetime) -> datetime:
        return value.astimezone(IST)

