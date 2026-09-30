"""Per-user document ownership records.

`document_id` (the SHA-256 of the file bytes) is deduped globally, so the
same document can be shared by many users. `document_index_status` tracks
indexing progress for the document itself (one row per document_id), while
this module tracks *who uploaded what* (one row per (user_id, document_id)
pair), which is what powers "list my documents".
"""

import datetime
import logging
from typing import Any
from uuid import UUID

from pymongo import ASCENDING

from mongo_vector_db.client import get_mongo_vector_db

logger = logging.getLogger(__name__)

USER_DOCUMENTS_COLLECTION = "user_documents"

_indexes_ensured = False


def _user_documents_collection():
    global _indexes_ensured
    collection = get_mongo_vector_db().db[USER_DOCUMENTS_COLLECTION]
    if not _indexes_ensured:
        collection.create_index(
            [("user_id", ASCENDING), ("document_id", ASCENDING)],
            unique=True,
            name="uq_user_document",
        )
        _indexes_ensured = True
    return collection


def insert_document_upload_record(
    user_id: UUID,
    document_id: str,
    original_filename: str | None,
    storage_path: str,
) -> None:
    """Record that a document has been uploaded by a user.

    Idempotent: safe to call again for the same (user_id, document_id) pair,
    e.g. on re-upload of an already-deduped file.
    """
    if not document_id:
        raise ValueError("document_id is required to record a document.")

    now = datetime.datetime.now(datetime.UTC)
    collection = _user_documents_collection()
    collection.update_one(
        {"user_id": str(user_id), "document_id": document_id},
        {
            "$set": {
                "original_filename": original_filename,
                "storage_path": storage_path,
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )


def get_documents_for_user(user_id: UUID) -> list[dict[str, Any]]:
    collection = _user_documents_collection()
    return list(
        collection.find({"user_id": str(user_id)}).sort("created_at", -1)
    )
