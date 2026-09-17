import logging
from collections.abc import Sequence
from time import perf_counter
from uuid import UUID

from psycopg import Error as PsycopgError
from sqlalchemy.exc import SQLAlchemyError
from src.database.models import Conversation, Message

from .chat import ChatService
from .exceptions import DatabaseUnavailableError
from .repository import DEFAULT_MESSAGE_LIMIT, ConversationRepository
from .schemas import LLMConfiguration

logger = logging.getLogger(__name__)


class ConversationService:
    def __init__(
        self,
        repository: ConversationRepository,
        chat_agent: ChatService,
    ) -> None:
        self._repository = repository
        self._chat_agent = chat_agent

    async def create_conversation(
        self,
        user_id: UUID,
        title: str | None,
    ) -> Conversation:
        try:
            conversation = await self._repository.create(user_id, title)
        except SQLAlchemyError as error:
            logger.error(
                "conversation.create.failed user_id=%s error_type=%s",
                user_id,
                type(error).__name__,
            )
            raise DatabaseUnavailableError() from error
        except Exception as error:
            logger.error(
                "conversation.create.failed user_id=%s error_type=%s",
                user_id,
                type(error).__name__,
            )
            raise

        logger.info(
            "conversation.created conversation_id=%s user_id=%s",
            conversation.id,
            user_id,
        )
        return conversation

    async def list_conversations(
        self,
        user_id: UUID,
    ) -> tuple[int, Sequence[Conversation]]:
        try:
            return await self._repository.get_all_conversations_for_user(user_id)
        except SQLAlchemyError as error:
            logger.error(
                "conversation.list.failed user_id=%s error_type=%s",
                user_id,
                type(error).__name__,
            )
            raise DatabaseUnavailableError() from error

    async def delete_conversation(self, conversation: Conversation) -> None:
        conversation_id = conversation.id
        user_id = conversation.user_id
        try:
            await self._repository.delete(conversation)
        except SQLAlchemyError as error:
            logger.error(
                "conversation.delete.failed conversation_id=%s user_id=%s error_type=%s",
                conversation_id,
                user_id,
                type(error).__name__,
            )
            raise DatabaseUnavailableError() from error
        except Exception as error:
            logger.error(
                "conversation.delete.failed conversation_id=%s user_id=%s error_type=%s",
                conversation_id,
                user_id,
                type(error).__name__,
            )
            raise

        logger.info(
            "conversation.deleted conversation_id=%s user_id=%s",
            conversation_id,
            user_id,
        )

    async def load_messages(
        self,
        conversation: Conversation,
        limit: int = DEFAULT_MESSAGE_LIMIT,
    ) -> list[Message]:
        try:
            return await self._repository.load_conversation(conversation, limit)
        except SQLAlchemyError as error:
            logger.error(
                "conversation.messages.load.failed conversation_id=%s user_id=%s error_type=%s",
                conversation.id,
                conversation.user_id,
                type(error).__name__,
            )
            raise DatabaseUnavailableError() from error

    async def send_user_message(
        self,
        conversation_id: UUID,
        user_message_content: str,
        ai_model: LLMConfiguration | None = None,
    ) -> str:
        started_at = perf_counter()
        logger.info("conversation.chat.started conversation_id=%s", conversation_id)
        try:
            await self._repository.add_message(
                "user",
                message_content=user_message_content,
                conversation_id=conversation_id,
            )
            ai_message_response, _ = self._chat_agent.send_message(
                conversation_id=conversation_id,
                user_message=user_message_content,
                ai_model=ai_model,
            )
            await self._repository.add_message(
                "ai",
                message_content=ai_message_response,
                conversation_id=conversation_id,
            )
        except (SQLAlchemyError, PsycopgError) as error:
            duration_ms = round((perf_counter() - started_at) * 1000)
            logger.error(
                "conversation.chat.failed conversation_id=%s duration_ms=%s error_type=%s",
                conversation_id,
                duration_ms,
                type(error).__name__,
            )
            raise DatabaseUnavailableError() from error
        except Exception as error:
            duration_ms = round((perf_counter() - started_at) * 1000)
            logger.error(
                "conversation.chat.failed conversation_id=%s duration_ms=%s error_type=%s",
                conversation_id,
                duration_ms,
                type(error).__name__,
            )
            raise

        duration_ms = round((perf_counter() - started_at) * 1000)
        logger.info(
            "conversation.chat.completed conversation_id=%s duration_ms=%s",
            conversation_id,
            duration_ms,
        )
        return ai_message_response
