import hashlib
import os
from typing import cast

import httpx
from dotenv import load_dotenv
from fastapi import UploadFile
from storage3.exceptions import StorageApiError
from storage3.types import FileOptions
from supabase import SupabaseException
from supabase.client import AsyncClient, create_async_client

from mongo_vector_db.models import StorageUploadResponse

load_dotenv()

class FileUpload:

    def __init__(self, supabase_client: AsyncClient) -> None:
        self.supabase_client = supabase_client
        self.bucket_name = "pdf-file-storage"
    
    @classmethod
    async def create(cls) -> "FileUpload":
        return cls(await cls.get_supabase_client())

    @staticmethod
    async def get_supabase_client() -> AsyncClient:
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        if not supabase_url:
            raise ValueError("SUPABASE_URL must be set")
        if not supabase_key:
            raise ValueError("SUPABASE_URL  SUPABASE_KEY must be set")
        try:
            return await create_async_client(supabase_url, supabase_key)
        except SupabaseException as e:
            raise SupabaseException("Exception raised by Supabase client.") from e
    
    async def create_supabase_storage_bucket(self, bucket_id: str):
      if not await self.bucket_exists(bucket_id):
        response = await self.supabase_client.storage.create_bucket(
            id=bucket_id,
            options={
                "public": False,
                "allowed_mime_types": ["application/pdf"],
            },
        )

        print("Response from Buckert Creation: ", response)
      print("Bucket already exists. Bucket ID: ", bucket_id)
    
    async def bucket_exists(self, bucket_id) -> bool:
      try:
        await self.supabase_client.storage.get_bucket(bucket_id)
        return True
      except StorageApiError:
        return False

    def is_file_valid(self, file_name: str) -> bool: ...

    @staticmethod
    def calculate_file_hash(file_content: bytes) -> str:
      """Return the SHA-256 digest used as the document's stable ID."""
      return hashlib.sha256(file_content).hexdigest()

    async def upload_to_file_storage(
      self, uploaded_file_content_in_bytes: bytes,
      original_filename: str,
      file_hash: str
    ) -> StorageUploadResponse:

        print("Uploading file the Supabase Object Storage")
        storage_path = f"{self.bucket_name}/pdf-files/{file_hash}.pdf"

        bucket_storage_client = self.supabase_client.storage.from_(self.bucket_name)

        # Skip upload when the hash-based object already exists; upsert handles races.
        try:
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
                return StorageUploadResponse(
                    document_id=file_hash,
                    file_hash=file_hash,
                    storage_path=storage_path,
                    original_filename=original_filename,
                    upload_status="Already Exists",
                    upload_error="File already exists in storage",
                )
        except httpx.ConnectError as exc:
            return StorageUploadResponse(
                document_id=file_hash,
                file_hash=file_hash,
                storage_path=storage_path,
                original_filename=original_filename,
                upload_status="Failed",
                upload_error=f"File Storage is Currently Unavailable. Error: {str(exc)}",
            )

        return StorageUploadResponse(
            document_id=file_hash,
            file_hash=file_hash,
            storage_path=storage_path,
            original_filename=original_filename,
            upload_status="Success",
            upload_error=None,
        )
      
    def read_pdf_file(self, file_to_read: UploadFile):
      print(type(file_to_read), dir(file_to_read))