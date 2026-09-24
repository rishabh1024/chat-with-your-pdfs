import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool

from conversations.agent import RAGAgent
from conversations.chat import ChatService
from conversations.exceptions import (
    ConversationAccessDeniedError,
    ConversationNotFoundError,
    DatabaseUnavailableError,
    conversation_access_denied_error_handler,
    conversation_not_found_exception_handler,
    database_unavailable_error_handler,
)
from conversations.router import router as conversation_router
from conversations.schemas import LLMConfiguration
from conversations.tools import MongoDocumentSearch
from core.exception import unexpected_error_handler
from core.logs import configure_logger
from core.settings import load_environment_variables
from database.configuration import close_database, init_database
from file_upload.exceptions import (
    IndexingStatusUnavailableError,
    StorageUnavailableError,
    UnsupportedFileTypeError,
    indexing_status_unavailable_error_handler,
    storage_unavailable_error_handler,
    unsupported_file_type_error_handler,
)
from file_upload.file_upload_client import FileUploadClient
from file_upload.router import router as upload_router

logger = logging.getLogger(__name__)

def enable_langsmith_tracing(settings) -> None:
    """Push validated Langsmith settings into the process environment
    so the LangSmith/LangChain SDK (which reads os.environ directly)
    picks them up."""
    os.environ["LANGSMITH_TRACING"] = str(settings.tracing).lower()

    if settings.api_key is not None:
        os.environ["LANGSMITH_API_KEY"] = settings.api_key.get_secret_value()

    if settings.project is not None:
        os.environ["LANGSMITH_PROJECT"] = settings.project


@asynccontextmanager
async def lifespan(app: FastAPI):
    environment_variables = load_environment_variables()
    configure_logger(environment_variables.logger)
    enable_langsmith_tracing(environment_variables.langsmith)
    checkpointer_memory_pool: ConnectionPool[Connection[DictRow]] | None = None
    try:
        checkpointer_memory_pool = ConnectionPool(
            environment_variables.supabase.db_url,
            min_size=1,
            max_size=environment_variables.database.pool_size,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            open=False,
        )
        checkpointer_memory_pool.open()
        app.state.checkpointer = PostgresSaver(checkpointer_memory_pool)
        app.state.checkpointer.setup()
        app.state.model_configuration = LLMConfiguration()
        app.state.document_search = MongoDocumentSearch()
        app.state.rag_agent = RAGAgent(
            model_configuration=app.state.model_configuration,
            checkpointer=app.state.checkpointer,
            document_search=app.state.document_search,
        )
        app.state.chat_service = ChatService(rag_agent=app.state.rag_agent)
        app.state.file_upload_client = await FileUploadClient.create()
        await init_database()
        logger.info("app.startup.completed")
        yield
    except Exception as error:
        logger.error(
            "app.startup.failed error_type=%s",
            type(error).__name__,
        )
        raise
    finally:
        if checkpointer_memory_pool is not None:
            checkpointer_memory_pool.close()
        await close_database()
        logger.info("app.shutdown.completed")


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(
        ConversationNotFoundError,
        conversation_not_found_exception_handler,
    )
    app.add_exception_handler(
        ConversationAccessDeniedError,
        conversation_access_denied_error_handler,
    )
    app.add_exception_handler(
        DatabaseUnavailableError,
        database_unavailable_error_handler,
    )
    app.add_exception_handler(
        StorageUnavailableError,
        storage_unavailable_error_handler,
    )
    app.add_exception_handler(
        IndexingStatusUnavailableError,
        indexing_status_unavailable_error_handler,
    )
    app.add_exception_handler(
        UnsupportedFileTypeError,
        unsupported_file_type_error_handler,
    )
    app.add_exception_handler(Exception, unexpected_error_handler)


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    settings = load_environment_variables()

    # Add CORS first so it wraps the app outermost and always handles preflight.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors.allowed_origins,
        allow_credentials=settings.cors.allow_credentials,
        allow_methods=settings.cors.allow_methods,
        allow_headers=settings.cors.allow_headers,
    )

    register_error_handlers(app)

    app.include_router(conversation_router)
    app.include_router(router=upload_router)

    return app
