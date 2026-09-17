from langchain_core.tools import tool
from src.conversations.protocols import DocumentVectorStore
from src.mongo_vector_db.data_wrangler import DocumentIndexer


def build_search_documents_tool(document_search_store: DocumentVectorStore):
    @tool(
        description="This tool searches for relevant documents from the"
        "vector database using the provided search query"
    )
    def search_documents(search_query: str) -> str:
        """Search the user's uploaded documents for information relevant to the query."""
        return document_search_store.search(query=search_query)

    return search_documents

class MongoDocumentSearch:
    def search(self, query: str) -> str:
        documents = DocumentIndexer.get_similar_documents_from_database(user_query=query)
        if isinstance(documents, list):
            return "\n\n---\n\n".join(d.page_content for d in documents)
        return f"No relevant documents were found. {documents['message']}"