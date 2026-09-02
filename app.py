import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Load env before importing modules that may read os.environ at import time.
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_openrouter import ChatOpenRouter
from pydantic import SecretStr
from src.conversations.router import router as conversation_router
from src.database.configuration import (
    close_database,
    init_database,
    initialize_app_checkpointer,
)
from src.file_upload.file_upload_client import FileUploadClient
from src.file_upload.router import (
    create_openrouter_chat,
)
from src.file_upload.router import (
    router as upload_router,
)


logger = logging.getLogger(__name__)

schema = {
    "title": "DocumentMetadata",
    "properties": {
        "title": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "hasCode": {"type": "boolean"},
    },
    "required": ["title", "keywords", "hasCode"],
}

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await FileUploadClient.create()
    app.state.llm_client = create_openrouter_chat(
        model="qwen/qwen3-30b-a3b-instruct-2507",
        api_key=OPENROUTER_API_KEY,
        temperature=0.3
    )

    app.state.structured_llm_client = ChatOpenRouter(
        model="qwen/qwen3-30b-a3b-instruct-2507",
        api_key=SecretStr(OPENROUTER_API_KEY or ""),
        model_kwargs={
            "models": [
                "qwen/qwen-2.5-7b-instruct",
                "openai/gpt-oss-20b:free",
                "meta-llama/llama-3.2-3b-instruct:free",
            ]
        },
        verbose=True,
        temperature=0.4,
    ).with_structured_output(schema, method="json_schema")

    await init_database()
    yield
    await close_database()


initialize_app_checkpointer()

app = FastAPI(lifespan=lifespan)
app.include_router(conversation_router)
app.include_router(router=upload_router)

app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)