import logging

from langchain_core.documents import Document
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr
from pymongo.errors import ServerSelectionTimeoutError
from tenacity import retry, stop_after_attempt, wait_random_exponential

from core.logs import safe_before_sleep_log
from core.settings import load_environment_variables
from mongo_vector_db.client import get_mongo_vector_db

logger = logging.getLogger(__name__)

settings = load_environment_variables()

OPENROUTER_API_KEY = settings.openrouter.openrouter_api_key.get_secret_value()

VECTOR_NAMESPACE = "sample_mflix.pdf_embeddings"
VECTOR_INDEX_NAME = "document_embeddings"
EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"

_vector_store: MongoDBAtlasVectorSearch | None = None


def get_embedding_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        base_url="https://openrouter.ai/api/v1",
        model=EMBEDDING_MODEL,
        api_key=SecretStr(OPENROUTER_API_KEY or ""),
        embedding_ctx_length=4096,
        check_embedding_ctx_length=False,
        model_kwargs={"encoding_format": "float"},
    )


def get_vector_store_instance() -> MongoDBAtlasVectorSearch:
    global _vector_store
    if _vector_store is None:
        _vector_store = MongoDBAtlasVectorSearch(
            collection=get_mongo_vector_db().collection,
            embedding=get_embedding_model(),
            index_name=VECTOR_INDEX_NAME,
        )
    return _vector_store


@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(min=2),
    before_sleep=safe_before_sleep_log(logger, "vector.embedding.create.retry"),
)
def create_embeddings(
    chunks: list[Document],
    *,
    document_id: str | None = None,
) -> dict[str, str]:
    try:
        MongoDBAtlasVectorSearch.from_documents(
            documents=chunks,
            embedding=get_embedding_model(),
            collection=get_mongo_vector_db().collection,
            index_name=VECTOR_INDEX_NAME,
        )
        logger.debug(
            "vector.embedding.create.completed document_id=%s count=%s",
            document_id,
            len(chunks),
        )
        return {"status": "Success", "message": "Document Embeddings Created Successfully"}
    except ServerSelectionTimeoutError as server_selection_error:
        raise server_selection_error
    except Exception as error:
        raise error
