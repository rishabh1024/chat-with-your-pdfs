import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.conversations.router import router as conversation_router
from src.core.logs import configure_logger
from src.core.settings import load_environment_variables
from src.database.configuration import (
    close_database,
    init_database,
    initialize_app_checkpointer,
)
from src.file_upload.file_upload_client import FileUploadClient
from src.file_upload.router import router as upload_router

environment_variables = load_environment_variables()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logger(environment_variables.logger)
    try:
        initialize_app_checkpointer()
        app.state.file_upload_client = await FileUploadClient.create()
        await init_database()
    except Exception as error:
        logger.error(
            "app.startup.failed error_type=%s",
            type(error).__name__,
        )
        raise

    logger.info("app.startup.completed")
    yield

    try:
        await close_database()
    except Exception as error:
        logger.error(
            "app.shutdown.failed error_type=%s",
            type(error).__name__,
        )
        raise

    logger.info("app.shutdown.completed")


app = FastAPI(lifespan=lifespan)
app.include_router(conversation_router)
app.include_router(router=upload_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
