import hashlib
import logging
from typing import cast

import httpx
from src.core.settings import load_environment_variables
from src.file_upload.models import StorageUploadResponse
from storage3.exceptions import StorageApiError
from storage3.types import FileOptions
from supabase import SupabaseException
from supabase.client import AsyncClient, create_async_client

settings = load_environment_variables()
logger = logging.getLogger(__name__)


class FileUploadClient:
    def __init__(self, supabase_client: AsyncClient) -> None:
        self.supabase_client = supabase_client
        self.bucket_name = "pdf-file-storage"

    @classmethod
    async def create(cls) -> "FileUploadClient":
        client = cls(await cls.get_supabase_client())
        logger.info("storage.client.initialize.completed")
        return client

    @staticmethod
    async def get_supabase_client() -> AsyncClient:
        supabase_url = settings.supabase.url.unicode_string()
        supabase_key = settings.supabase.key.get_secret_value()
        if not supabase_url:
            raise ValueError("SUPABASE_URL must be set")
        if not supabase_key:
            raise ValueError("SUPABASE_URL  SUPABASE_KEY must be set")
        try:
            return await create_async_client(supabase_url, supabase_key)
        except SupabaseException as error:
            logger.error(
                "storage.client.initialize.failed error_type=%s",
                type(error).__name__,
            )
            raise SupabaseException("Unable to initialize storage client.") from error

    async def create_supabase_storage_bucket(self, bucket_id: str):
        if await self.bucket_exists(bucket_id):
            logger.debug("storage.bucket.create.skipped reason=already_exists")
            return

        try:
            await self.supabase_client.storage.create_bucket(
                id=bucket_id,
                options={
                    "public": False,
                    "allowed_mime_types": ["application/pdf"],
                },
            )
        except Exception as error:
            logger.error(
                "storage.bucket.create.failed error_type=%s",
                type(error).__name__,
            )
            raise

        logger.info("storage.bucket.create.completed")

    async def bucket_exists(self, bucket_id) -> bool:
        logger.debug("storage.bucket.lookup.started")
        try:
            await self.supabase_client.storage.get_bucket(bucket_id)
            return True
        except StorageApiError as error:
            logger.warning(
                "storage.bucket.lookup.failed error_type=%s",
                type(error).__name__,
            )
            return False

    def is_file_valid(self, file_name: str) -> bool: ...

    @staticmethod
    def calculate_file_hash(file_content: bytes) -> str:
        """Return the SHA-256 digest used as the document's stable ID."""
        return hashlib.sha256(file_content).hexdigest()

    # Check the file's mime type and throw an error for invalid file type
    async def upload_to_file_storage(
        self, uploaded_file_content_in_bytes: bytes, original_filename: str, file_hash: str
    ) -> StorageUploadResponse:

        storage_path = f"{self.bucket_name}/pdf-files/{file_hash}.pdf"

        bucket_storage_client = self.supabase_client.storage.from_(self.bucket_name)

        # Skip upload when the hash-based object already exists; upsert handles races.
        try:
            logger.debug("storage.upload.dispatch document_id=%s", file_hash)
            if not await bucket_storage_client.exists(storage_path):
                await bucket_storage_client.upload(
                    path=storage_path,
                    file=uploaded_file_content_in_bytes,
                    file_options=cast(
                        FileOptions,
                        {
                            "content-type": "application/pdf",
                            "upsert": "True",
                            "metadata": {"file_hash": file_hash},
                        },
                    ),
                )
            else:
                logger.debug(
                    "storage.upload.skipped document_id=%s reason=already_exists",
                    file_hash,
                )
                return StorageUploadResponse(
                    document_id=file_hash,
                    file_hash=file_hash,
                    storage_path=storage_path,
                    original_filename=original_filename,
                    upload_status="Already Exists",
                    upload_error="File already exists in storage",
                )
        except httpx.ConnectError as error:
            logger.error(
                "storage.upload.failed document_id=%s error_type=%s",
                file_hash,
                type(error).__name__,
            )
            return StorageUploadResponse(
                document_id=file_hash,
                file_hash=file_hash,
                storage_path=storage_path,
                original_filename=original_filename,
                upload_status="Failed",
                upload_error="File storage is currently unavailable.",
            )

        return StorageUploadResponse(
            document_id=file_hash,
            file_hash=file_hash,
            storage_path=storage_path,
            original_filename=original_filename,
            upload_status="Success",
            upload_error=None,
        )
