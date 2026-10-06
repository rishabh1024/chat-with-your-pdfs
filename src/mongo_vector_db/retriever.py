import logging

from langchain_core.documents import Document
from pymongo.errors import ServerSelectionTimeoutError

from mongo_vector_db.vector_store import get_vector_store_instance

logger = logging.getLogger(__name__)


def search_similar_documents(user_query: str, user_id: str) -> list[Document] | dict:
    logger.info(f"vector.retrieval search_similar_documents called with search query: {user_query}")
    try:
        vector_store_instance = get_vector_store_instance()
        vector_store_retriever = vector_store_instance.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": 0.10,
                "pre_filter": {"user_id": {"$eq": user_id}},
            },
        )

        documents = vector_store_retriever.invoke(user_query)
        logger.debug(
            "vector.retrieval.completed count=%s",
            len(documents),
        )
        return documents
    except ServerSelectionTimeoutError as error:
        logger.warning(
            msg="vector.retrieval.failed error_type=%s",
            extra={"error": type(error).__name__,
                   "reason": f"Server Selection Timed Out.: {error}"},
        )
        return {
            "status": "Failed",
            "message": "Failed to retrieve any relevant documents.",
            "error": str(error),
        }
    except Exception as error:
        logger.error(
            "vector.retrieval.failed error_type=%s",
            extra={"error": type(error).__name__,
                   "reason": f"Error Occured in Search Documents: {error}"},
        )
        return {
            "status": "Failed",
            "message": "Failed to retrieve any relevant documents.",
            "error": str(error),
        }
