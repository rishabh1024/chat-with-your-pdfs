import os
from collections.abc import Sequence
from uuid import UUID

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool
from sqlalchemy import select
from sqlalchemy.exc import (
    DBAPIError,
    DisconnectionError,
    InterfaceError,
    OperationalError,
    TimeoutError,
)
from sqlalchemy.ext.asyncio import AsyncSession
from src.database.models import Conversation, Message
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from .chat import ChatService
from .exceptions import ConversationAccessDeniedError, ConversationNotFoundError
from .models import LLMConfiguration

DEFAULT_CONVERSATION_TITLE = "New conversation"
DEFAULT_MESSAGE_LIMIT = 20

@retry(
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(OperationalError)
    | retry_if_exception_type(TimeoutError)
    | retry_if_exception_type(DisconnectionError)
    | retry_if_exception_type(InterfaceError),
    wait=wait_exponential_jitter(),
    reraise=True,
)
async def create_a_new_conversation(
    db_session: AsyncSession,
    user_id: UUID,
    title: str | None,
) -> Conversation:
    new_conversation = Conversation(
        user_id=user_id,
        title=title or DEFAULT_CONVERSATION_TITLE,
    )
    try:
        db_session.add(instance=new_conversation)
        await db_session.commit()
        await db_session.refresh(new_conversation)
        return new_conversation
    except Exception:
        await db_session.rollback()
        raise


@retry(
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(OperationalError)
    | retry_if_exception_type(DBAPIError)
    | retry_if_exception_type(TimeoutError)
    | retry_if_exception_type(DisconnectionError)
    | retry_if_exception_type(InterfaceError),
    wait=wait_exponential_jitter(),
    reraise=True,
)
async def get_list_all_conversation_for_user(db_session: AsyncSession,
                                            user_id: UUID) -> tuple[int, Sequence[Conversation]]:
    try:
        list_of_all_conversations = ((
            await db_session.execute(
                    select(Conversation)
                    .where(Conversation.user_id == user_id)
                    .order_by(Conversation.updated_at.desc())
                    ))
            .scalars()
            .all())

        return len(list_of_all_conversations), list_of_all_conversations
    except Exception:
        await db_session.rollback()
        raise


@retry(
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(OperationalError)
    | retry_if_exception_type(DBAPIError)
    | retry_if_exception_type(TimeoutError)
    | retry_if_exception_type(DisconnectionError)
    | retry_if_exception_type(InterfaceError),
    wait=wait_exponential_jitter(),
    reraise=True,
)
async def delete_conversation_thread(
    user_id: UUID,
    conversation_id: UUID,
    db_session: AsyncSession,
):

  conversation = await db_session.get(Conversation, conversation_id)

  if conversation is None:
      raise ConversationNotFoundError(
          conversation_id,
          error_msg="Conversation not found.",
      )
  if conversation.user_id != user_id:
      raise ConversationAccessDeniedError(
          conversation_id,
          error_msg="Access denied.",
      )
  try:
    await db_session.delete(conversation)
    await db_session.commit()
  except Exception:
    await db_session.rollback()
    raise



async def add_message_to_db(
        role: str,
        message_content: str,
        db_session: AsyncSession,
        conversation_id: UUID):

  user_message = Message(conversation_id=conversation_id, role= role, content=message_content)
  try:
    db_session.add(user_message)
    await db_session.commit()
  except Exception:
    await db_session.rollback()
    raise

async def process_user_query_with_rag(thread_id, user_message_content, db_session: AsyncSession, llm_configuration: LLMConfiguration) -> str:
    pool: ConnectionPool[Connection[DictRow]] = ConnectionPool(
        conninfo=os.environ["SUPABASE_DB_URL"],
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
    )
    pool.open()
    checkpointer = PostgresSaver(pool)
    chat_service = ChatService(checkpointer=checkpointer)

    await add_message_to_db(
        "user",
        message_content=user_message_content,
        db_session=db_session,
        conversation_id=thread_id,
    )
    ai_message_response, _ = chat_service.send_message(
        chat_id=thread_id, user_message=user_message_content
    )
    await add_message_to_db(
        "ai",
        message_content=ai_message_response,
        db_session=db_session,
        conversation_id=thread_id,
    )

    return ai_message_response


@retry(
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(OperationalError) |
    retry_if_exception_type(DBAPIError) |
    retry_if_exception_type(TimeoutError) |
    retry_if_exception_type(DisconnectionError) |
    retry_if_exception_type(InterfaceError),
    wait= wait_exponential_jitter(),
    reraise=True
)
async def load_conversation_from_database(
    user_id: UUID,
    conversation_id: UUID,
    db_session: AsyncSession,
    limit: int = DEFAULT_MESSAGE_LIMIT
) -> list[Message]:

    conversation = await db_session.get(Conversation, conversation_id)

    if conversation is None:
        raise ConversationNotFoundError(conversation_id, error_msg="Failed to load Conversation.")
    if conversation.user_id != user_id:
        raise ConversationAccessDeniedError(
            conversation_id, error_msg="Failed to load Conversation. Access Denied."
        )
    try:
        result = await db_session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
            .limit(limit)
        )

        return list(result.scalars().all())
    except Exception:
        await db_session.rollback()
        raise