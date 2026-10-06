from typing import Protocol

from file_upload.models import StorageUploadResponse


class StorageClient(Protocol):
    async def upload(
        self,
        file_contents: bytes,
        original_filename: str,
        file_hash: str,
        user_id: str,
    ) -> StorageUploadResponse: ...

    async def file_exists(self, storage_path: str) -> bool: ...
