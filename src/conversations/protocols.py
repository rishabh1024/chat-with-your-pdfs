from typing import Protocol
from uuid import UUID

from conversations.schemas import LLMConfiguration


class DocumentVectorStore(Protocol):
    def search(self, query: str) -> str: ...


class ChatAgent(Protocol):
    def send_message(
        self,
        conversation_id: UUID,
        user_message: str,
        ai_model: LLMConfiguration | None = None,
    ) -> tuple[str, list[str]]: ...
