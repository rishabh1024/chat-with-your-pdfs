import os
from uuid import UUID

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter
from pydantic import SecretStr

from mongo_vector_db.data_wrangler import DocumentIndexer

load_dotenv(override=True)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

BASE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's questions clearly and concisely. "
    "If the question is about the Rishabh or you need more information to answer"
    "your questions, use the search_documents tool to get information from relevant documents."
    "Otherwise answer directly. Do not fabricate information when asked about Rishabh"
)


@tool(description="This tool searches for relevant data in the vectorDB based on the search query")
def search_documents(query: str) -> str:
    """Search the user's uploaded documents for information relevant to the query."""
    print(f"Tool has been called with query: {query}")

    documents = DocumentIndexer.get_similar_documents_from_database(user_query=query)
    if isinstance(documents, list):
        return "\n\n---\n\n".join(document.page_content for document in documents)
    return f"No relevant document's were found. {documents["message"]}"


class ChatService:
    def __init__(self) -> None:
        self.llm = ChatOpenRouter(
            model="qwen/qwen3-30b-a3b-instruct-2507",
            api_key=SecretStr(OPENROUTER_API_KEY or ""),
            model_kwargs={
                "models": [
                    "qwen/qwen3-next-80b-a3b-instruct:free",
                    "poolside/laguna-xs-2.1:free",
                    "meta-llama/llama-3.2-3b-instruct:free",
                ]
            },
        )
        self.llm_with_tools = self.llm.bind_tools([search_documents])
        # In-memory history per conversation. Lost on restart -- intentionally basic.
        self.chat_histories: dict[UUID, list[BaseMessage]] = {}

    def _get_history(self, chat_id: UUID) -> list[BaseMessage]:
        return self.chat_histories.setdefault(chat_id, [])

    def send_message(self, chat_id: UUID, user_message: str) -> tuple[str, list[str]]:
        history = self._get_history(chat_id)
        history.append(HumanMessage(content=user_message))

        messages: list[BaseMessage] = [SystemMessage(content=BASE_SYSTEM_PROMPT), *history]
        response = self.llm_with_tools.invoke(messages)

        # The model decided it needs context from the vector database.
        if isinstance(response, AIMessage) and response.tool_calls:
            messages.append(response)
            for tool_call in response.tool_calls:
                tool_result = search_documents.invoke(tool_call["args"])
                messages.append(
                    ToolMessage(content=tool_result, tool_call_id=tool_call["id"] or "")
                )
            response = self.llm_with_tools.invoke(messages)

        # Only the clean user/AI exchange goes into history, not tool traffic.
        history.append(AIMessage(content=response.content))

        chat_history_messages = [f"{message.type}: {message.content}" for message in history]
        return str(response.content), chat_history_messages
