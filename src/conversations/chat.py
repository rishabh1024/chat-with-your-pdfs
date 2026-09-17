import logging
from typing import Any
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph, RunnableConfig
from src.conversations.agent import RAGAgent
from src.conversations.schemas import LLMConfiguration

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, rag_agent: RAGAgent) -> None:
        self.rag_agent: CompiledStateGraph[Any, LLMConfiguration, Any, Any] = (
            rag_agent.agent_graph
        )

    def send_message(
        self,
        conversation_id: UUID,
        user_message: str,
        ai_model: LLMConfiguration | None = None,
    ) -> tuple[str, list[str]]:
        thread_configuration = RunnableConfig(
            {"configurable": {"thread_id": str(conversation_id)}}
        )
        agent_context = self.get_agent_context(ai_model)
        agent_response = self.rag_agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config=thread_configuration,
            context=agent_context,
        )
        conversation_messages: list[BaseMessage] = agent_response["messages"]
        last_ai_message = self.get_last_ai_message_from_response(conversation_messages)
        user_visible_chat_history = self._format_user_visible_chat_history(conversation_messages)
        return last_ai_message, user_visible_chat_history

    @staticmethod
    def get_agent_context(ai_model: LLMConfiguration | None) -> LLMConfiguration | None:
        if ai_model is None or ai_model == LLMConfiguration():
            return None
        return ai_model

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
