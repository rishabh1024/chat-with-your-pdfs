from typing import Annotated

from fastapi import APIRouter, Depends, status

from auth.dependencies import get_current_authenticated_user
from auth.models import AuthenticatedUser
from conversations.dependencies import (
    get_conversation_service,
    validate_conversationid_and_user_authorization,
)
from conversations.service import ConversationService
from database.models import Conversation

from .schemas import (
    AIMessageResponse,
    ChatMessage,
    ConversationListResponse,
    ConversationMessagesResponse,
    ConversationResponse,
    CreateConversationRequest,
    UserMessageRequest,
)

router = APIRouter(
    prefix="/conversations",
    tags=["conversations"],
    dependencies=[Depends(get_current_authenticated_user)],
)

User = Annotated[AuthenticatedUser, Depends(get_current_authenticated_user)]
ValidatedConversation = Annotated[
    Conversation, Depends(validate_conversationid_and_user_authorization)
]
ConversationSvc = Annotated[ConversationService, Depends(get_conversation_service)]


@router.post(path="", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_new_conversation(
    request_body: CreateConversationRequest,
    current_user: User,
    service: ConversationSvc,
) -> ConversationResponse:
    new_conversation = await service.create_conversation(
        current_user.user_id, title=request_body.title
    )
    return ConversationResponse.model_validate(new_conversation)


@router.get("", response_model=ConversationListResponse)
async def list_all_conversations_for_the_user(
    current_user: User,
    service: ConversationSvc,
) -> ConversationListResponse:
    total_threads, list_of_threads = await service.list_conversations(user_id=current_user.user_id)
    return ConversationListResponse(
        count_conversations=total_threads,
        conversations=[ConversationResponse.model_validate(thread) for thread in list_of_threads],
    )


@router.delete(path="/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    valid_conversation: ValidatedConversation,
    service: ConversationSvc,
) -> None:
    await service.delete_conversation(conversation=valid_conversation)


@router.post(
    "/{conversation_id}/messages",
    response_model=AIMessageResponse,
    status_code=status.HTTP_200_OK,
)
async def send_message_to_agent(
    user_message: UserMessageRequest,
    conversation: ValidatedConversation,
    service: ConversationSvc,
) -> AIMessageResponse:
    agent_response = await service.send_user_message(
        conversation_id=conversation.id,
        user_message_content=user_message.message_content,
        ai_model=user_message.ai_model,
    )
    return AIMessageResponse(conversation_id=conversation.id, content=agent_response)


@router.get(
    path="/{conversation_id}/messages",
    response_model=ConversationMessagesResponse,
    status_code=status.HTTP_200_OK,
    description="Load all messages for a given Conversation",
)
async def load_all_chat_messages(
    conversation: ValidatedConversation,
    service: ConversationSvc,
) -> ConversationMessagesResponse:
    list_of_all_messages = await service.load_messages(conversation=conversation)
    return ConversationMessagesResponse(
        conversation_id=conversation.id,
        messages=[ChatMessage.model_validate(m) for m in list_of_all_messages],
    )
