from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from conversations.protocols import DocumentVectorStore
from mongo_vector_db.retriever import search_similar_documents


def build_search_documents_tool(document_search_store: DocumentVectorStore):
    @tool(
        description="This tool searches for relevant documents from the"
        "vector database using the provided search query"
    )
    def search_documents(search_query: str, config: RunnableConfig) -> str:
        """Search the user's uploaded documents for information relevant to the query."""
        # user_id is injected by ChatService from the authenticated request,
        # never from the model, so it cannot be spoofed via tool arguments.
        user_id = config.get("configurable", {}).get("user_id")
        if not user_id:
            return "No relevant documents were found. User could not be identified."
        return document_search_store.search(query=search_query, user_id=user_id)

    return search_documents


class MongoDocumentSearch:
    def search(self, query: str, user_id: str) -> str:
        documents = search_similar_documents(user_query=query, user_id=user_id)
        if isinstance(documents, list):
            return "\n\n---\n\n".join(d.page_content for d in documents)
        return f"No relevant documents were found. {documents['message']}"
