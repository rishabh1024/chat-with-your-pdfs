from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from src.auth.dependencies import get_current_authenticated_user
from src.auth.models import AuthenticatedUser
from src.conversations.exceptions import (
    ConversationAccessDeniedError,
    ConversationNotFoundError,
)
from src.conversations.models import ConversationListResponse
from src.database.configuration import get_database_session

from .models import (
    AIMessageResponse,
    ChatMessage,
    ConversationMessagesResponse,
    ConversationResponse,
    CreateConversationRequest,
    LLMConfiguration,
    UserMessageRequest,
)
from .service import (
    create_a_new_conversation,
    delete_conversation_thread,
    get_list_all_conversation_for_user,
    load_conversation_from_database,
    process_user_query_with_rag,
)

router = APIRouter(
    prefix="/conversations",
    tags=["conversations"],
    dependencies=[Depends(get_database_session), Depends(get_current_authenticated_user)],
)

DatabaseSession = Annotated[AsyncSession, Depends(get_database_session)]
User = Annotated[AuthenticatedUser, Depends(get_current_authenticated_user)]

"""

"""
@router.post("", response_model=ConversationResponse,
             status_code=status.HTTP_201_CREATED)
async def create_new_conversation(
    request_body: CreateConversationRequest,
    current_user: User,
    database_session: DatabaseSession,
) -> ConversationResponse:
    new_conversation = await create_a_new_conversation(database_session,
                                                    current_user.user_id,
                                                    title=request_body.title)
    return ConversationResponse.model_validate(new_conversation)

@router.get("", response_model=ConversationListResponse)
async def list_all_conversations_for_the_user(
    current_user: User,
    database_session: DatabaseSession,
) -> ConversationListResponse:

    total_threads, list_of_threads = await get_list_all_conversation_for_user(
        user_id=current_user.user_id, db_session=database_session
    )

    return ConversationListResponse(
        count_conversations=total_threads,
        conversations=[ConversationResponse.model_validate(thread) for thread in list_of_threads]
    )

@router.delete(path="/{conversation_id}",
               status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    current_user: User,
    conversation_id: UUID,
    database_session: DatabaseSession,
) -> None:
    try:
        await delete_conversation_thread(
            user_id=current_user.user_id,
            db_session=database_session,
            conversation_id=conversation_id)

    except ConversationNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={"conversation_id": str(e.conversation_id), "error_message": e.error},
        ) from e
    except ConversationAccessDeniedError as e:
        raise HTTPException(
            status_code=403,
            detail={"conversation_id": str(e.conversation_id), "error_message": e.error},
        ) from e


@router.post("/{conversation_id}/messages",
             response_model=AIMessageResponse,
             status_code=status.HTTP_200_OK)
async def send_message_to_agent(
    conversation_id,
    user_message: UserMessageRequest,
    database_session: DatabaseSession,
    model_configuration: LLMConfiguration
):
    agent_response = await process_user_query_with_rag(
        conversation_id,
        user_message.message_content,
        db_session=database_session,
        llm_configuration=model_configuration,
    )

    return AIMessageResponse(
        conversation_id=conversation_id,
        content=agent_response
    )


"""
GET Method : 'conversations/{conversation_id}/messages'
Fetch all messages in a conversation.

1. Method takes in conversation id, current user, and database_session
2.

"""
@router.get(
    path="/{conversation_id}/messages/",
    status_code=status.HTTP_200_OK,
    description="Load all messages for a given Conversation")
async def load_all_chat_messages(
    current_user: User,
    conversation_id: UUID,
    database_session: DatabaseSession):

    try:
        list_of_all_messages = await load_conversation_from_database(
            current_user.user_id, conversation_id, database_session
        )
        print("Messages", list_of_all_messages)
        return ConversationMessagesResponse(
            conversation_id=conversation_id,
            messages=[ChatMessage.model_validate(m) for m in list_of_all_messages],
        )

    except ConversationAccessDeniedError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail={"conversation_id":str( e.conversation_id),
                                    "error_message": e.error}) from e
    except ConversationNotFoundError as e:

        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail={"conversation_id": str(e.conversation_id),
                                    "error_message": e.error}) from e