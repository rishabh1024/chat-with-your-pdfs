import logging
from typing import Annotated
from uuid import UUID

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.auth.dependencies import get_current_authenticated_user
from src.auth.models import AuthenticatedUser
from src.database.configuration import get_database_session
from src.database.models import Conversation

from .exceptions import ConversationAccessDeniedError, ConversationNotFoundError

logger = logging.getLogger(__name__)


async def validate_conversationid_and_user_authorization(
    conversation_id: UUID,
    current_user: Annotated[AuthenticatedUser, Depends(get_current_authenticated_user)],
    db_session: Annotated[AsyncSession, Depends(get_database_session)],
) -> Conversation:
    conversation = await db_session.get(Conversation, conversation_id)

    if conversation is None:
        logger.warning(
            "conversation.access.rejected reason=not_found conversation_id=%s user_id=%s",
            conversation_id,
            current_user.user_id,
        )
        raise ConversationNotFoundError(
            conversation_id,
            error_msg="Conversation not found.",
        )
    if conversation.user_id != current_user.user_id:
        logger.warning(
            "conversation.access.rejected reason=access_denied conversation_id=%s user_id=%s",
            conversation_id,
            current_user.user_id,
        )
        raise ConversationAccessDeniedError(
            conversation_id,
            error_msg="Access denied.",
        )

    return conversation
