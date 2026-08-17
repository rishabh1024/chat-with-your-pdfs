import os
from uuid import UUID

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import RunnableConfig
from pydantic import SecretStr

from mongo_vector_db.data_wrangler import DocumentIndexer

load_dotenv(override=True)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's questions clearly and concisely. "
    "If the question is about the Rishabh or you need more information to answer "
    "your questions, use the search_documents tool to get information from relevant documents. "
    "The user might sometimes ask you to refer to the documents again, follow his instructions"
    "Otherwise answer directly. Do not fabricate information when asked about Rishabh."
)


@tool(description="This tool searches for relevant data in the vectorDB based on the search query")
def search_documents(query: str) -> str:
    """Search the user's uploaded documents for information relevant to the query."""
    print(f"Tool has been called with query: {query}")

    documents = DocumentIndexer.get_similar_documents_from_database(user_query=query)
    if isinstance(documents, list):
        return "\n\n---\n\n".join(document.page_content for document in documents)
    return f"No relevant document's were found. {documents['message']}"


class ChatService:
    def __init__(self) -> None:
        self.openrouter_language_model = ChatOpenRouter(
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
        memory_checkpointer = InMemorySaver()
        self.rag_agent_with_tools = create_agent(
            model=self.openrouter_language_model,
            tools=[search_documents],
            system_prompt=SYSTEM_PROMPT,
            checkpointer=memory_checkpointer,
            middleware=[
                ToolRetryMiddleware(backoff_factor=2.0, initial_delay=1.0),
                ModelRetryMiddleware(backoff_factor=2.0, initial_delay=1.0),
            ],
        )

    def send_message(self, chat_id: UUID, user_message: str) -> tuple[str, list[str]]:
        thread_configuration = RunnableConfig({"configurable": {"thread_id": str(chat_id)}})
        agent_response= self.rag_agent_with_tools.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config=thread_configuration,
        )
        conversation_messages: list[BaseMessage] = agent_response["messages"]
        last_ai_message = self.get_last_ai_message_from_response(conversation_messages)
        user_visible_chat_history = self._format_user_visible_chat_history(conversation_messages)
        return last_ai_message, user_visible_chat_history

    @staticmethod
    def get_last_ai_message_from_response(conversation_messages: list[BaseMessage]) -> str:
        ai_messages = [
            message for message in conversation_messages if isinstance(message, AIMessage)
        ]
        if not ai_messages:
            return ""
        return str(ai_messages[-1].content)

    @staticmethod
    def _format_user_visible_chat_history(conversation_messages: list[BaseMessage]) -> list[str]:
        user_visible_chat_history: list[str] = []
        for message in conversation_messages:
            if isinstance(message, HumanMessage) or (
                isinstance(message, AIMessage) and message.content
            ):
                user_visible_chat_history.append(f"{message.type}: {message.content}")
        return user_visible_chat_history
