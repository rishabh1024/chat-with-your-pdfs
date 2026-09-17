import logging
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.database.models import Conversation, Message
from src.database.retries import retry_on_transient_db_error

logger = logging.getLogger(__name__)

DEFAULT_CONVERSATION_TITLE = "New conversation"
DEFAULT_MESSAGE_LIMIT = 20


class ConversationRepository:

    def __init__(self, session: AsyncSession) -> None:
        self._session = session


    @retry_on_transient_db_error(logger, "conversation.create.retry")
    async def create(
        self,
        user_id: UUID,
        title: str | None,
    ) -> Conversation:
        new_conversation = Conversation(
            user_id=user_id,
            title=title or DEFAULT_CONVERSATION_TITLE,
        )
        try:
            self._session.add(instance=new_conversation)
            await self._session.commit()
            await self._session.refresh(new_conversation)
            return new_conversation
        except Exception:
            await self._session.rollback()
            raise


    @retry_on_transient_db_error(logger, "conversation.list.retry")
    async def get_all_conversations_for_user(
        self, user_id: UUID
    ) -> tuple[int, Sequence[Conversation]]:
        try:
            list_of_all_conversations = (
                (
                    await self._session.execute(
                        select(Conversation)
                        .where(Conversation.user_id == user_id)
                        .order_by(Conversation.updated_at.desc())
                    )
                )
                .scalars()
                .all()
            )

            return len(list_of_all_conversations), list_of_all_conversations
        except Exception:
            await self._session.rollback()
            raise


    @retry_on_transient_db_error(logger, "conversation.delete.retry")
    async def delete(
        self,
        conversation: Conversation,
    ) -> None:
        try:
            await self._session.delete(conversation)
            await self._session.commit()
        except Exception:
            await self._session.rollback()
            raise

    @retry_on_transient_db_error(logger, "conversation.add.message.retry")
    async def add_message(
        self, role: str, message_content: str, conversation_id: UUID
    ) -> None:

        message = Message(conversation_id=conversation_id, role=role, content=message_content)
        try:
            self._session.add(message)
            await self._session.commit()
        except Exception:
            await self._session.rollback()
            raise


    @retry_on_transient_db_error(logger, "conversation.load.retry")
    async def load_conversation(self,
        conversation: Conversation,
        limit: int = DEFAULT_MESSAGE_LIMIT
    ) -> list[Message]:
        try:
            result = await self._session.execute(
                select(Message)
                .where(Message.conversation_id == conversation.id)
                .order_by(Message.created_at)
                .limit(limit)
            )

            return list(result.scalars().all())
        except Exception:
            await self._session.rollback()
            raise