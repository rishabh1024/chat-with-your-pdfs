import asyncio
import logging
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import jwt
import pytest


def record_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records]


def test_invalid_authentication_logs_reason_without_token(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from auth import token_validator
    from auth.exceptions import InvalidTokenError

    token = "sensitive-token-value"

    class InvalidJwksClient:
        def get_signing_key_from_jwt(self, supplied_token: str) -> None:
            assert supplied_token == token
            raise jwt.InvalidTokenError("provider included sensitive-token-value")

    monkeypatch.setattr(
        token_validator,
        "get_jwks_client",
        lambda: InvalidJwksClient(),
    )
    caplog.set_level(logging.WARNING, logger=token_validator.__name__)

    with pytest.raises(InvalidTokenError):
        token_validator.validate_jw_token(token)

    messages = record_messages(caplog)
    assert any(
        "auth.token.rejected reason=invalid or malformed error_type=InvalidTokenError" in message
        for message in messages
    )
    assert all(token not in message for message in messages)


def test_conversation_mutation_failure_logs_safe_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from conversations import service
    from conversations.repository import ConversationRepository
    from conversations.service import ConversationService

    user_id = uuid4()

    class FailingSession:
        def add(self, instance: object) -> None:
            self.instance = instance

        async def commit(self) -> None:
            raise RuntimeError("provider leaked a sensitive prompt")

        async def rollback(self) -> None:
            return None

    class FakeChatService:
        def send_message(self, chat_id, user_message):
            return "", []

    conversation_service = ConversationService(
        repository=ConversationRepository(FailingSession()),  # type: ignore[arg-type]
        chat_agent=FakeChatService(),  # type: ignore[arg-type]
    )

    caplog.set_level(logging.ERROR, logger=service.__name__)

    with pytest.raises(RuntimeError):
        asyncio.run(conversation_service.create_conversation(user_id, None))

    messages = record_messages(caplog)
    assert messages == [f"conversation.create.failed user_id={user_id} error_type=RuntimeError"]
    assert "sensitive prompt" not in messages[0]


def test_storage_outage_logs_no_external_exception_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from file_upload.file_upload_client import FileUploadClient

    document_id = "document-123"

    class UnavailableBucket:
        async def exists(self, path: str) -> bool:
            raise httpx.ConnectError(
                "storage-key-sensitive",
                request=httpx.Request("GET", "https://storage.invalid"),
            )

    class Storage:
        def from_(self, bucket_name: str) -> UnavailableBucket:
            return UnavailableBucket()

    class SupabaseClient:
        storage = Storage()

    client = FileUploadClient(SupabaseClient())
    caplog.set_level(logging.DEBUG, logger="file_upload.file_upload_client")

    result = asyncio.run(
        client.upload(
            file_contents=b"pdf",
            original_filename="private-filename.pdf",
            file_hash=document_id,
        )
    )

    messages = record_messages(caplog)
    assert result.upload_status == "Failed"
    assert any(
        message == f"storage.upload.failed document_id={document_id} error_type=ConnectError"
        for message in messages
    )
    assert all("storage-key-sensitive" not in message for message in messages)
    assert all("private-filename.pdf" not in message for message in messages)


def test_indexing_timeout_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    from mongo_vector_db.indexing_tracker import IndexingStatusTracker

    document_id = "document-timeout"
    tracker = IndexingStatusTracker()
    tracker.register(document_id)
    caplog.set_level(logging.WARNING, logger="mongo_vector_db.indexing_tracker")

    result = asyncio.run(tracker.wait_for_result(document_id, timeout=0))

    assert result["status"] == "timeout"
    assert (
        f"indexing.tracker.wait.failed document_id={document_id} reason=timeout"
        in record_messages(caplog)
    )


def test_application_lifecycle_logs_completed_events(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import main as main_module

    class FakeConnectionPool:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def open(self) -> None:
            return None

        def close(self) -> None:
            return None

    class FakePostgresSaver:
        def __init__(self, pool) -> None:
            self.pool = pool

        def setup(self) -> None:
            return None

    monkeypatch.setattr(main_module, "ConnectionPool", FakeConnectionPool)
    monkeypatch.setattr(main_module, "PostgresSaver", FakePostgresSaver)
    monkeypatch.setattr(main_module, "MongoDocumentSearch", lambda: object())
    monkeypatch.setattr(
        main_module,
        "RAGAgent",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(main_module, "ChatService", lambda rag_agent=None: object())
    monkeypatch.setattr(main_module, "configure_logger", lambda settings: None)
    monkeypatch.setattr(main_module, "init_database", AsyncMock())
    monkeypatch.setattr(main_module, "close_database", AsyncMock())
    monkeypatch.setattr(main_module.FileUploadClient, "create", AsyncMock(return_value=object()))
    caplog.set_level(logging.INFO, logger=main_module.__name__)

    app = main_module.create_app()

    async def run_lifespan() -> None:
        async with main_module.lifespan(app):
            pass

    asyncio.run(run_lifespan())

    messages = record_messages(caplog)
    assert "app.startup.completed" in messages
    assert "app.shutdown.completed" in messages
