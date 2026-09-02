from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, field_serializer
from pydantic.types import UUID4

IST = ZoneInfo("Asia/Kolkata")

class CreateConversationRequest(BaseModel):
  title: str | None = "New Conversation"


class ConversationResponse(BaseModel):
  model_config = ConfigDict(from_attributes=True)

  id: UUID4
  user_id: UUID4
  title: str | None
  created_at: datetime
  updated_at: datetime

  @field_serializer("created_at", "updated_at")
  def serialize_in_ist(self, value: datetime) -> datetime:
    return value.astimezone(IST)


class ConversationListResponse(BaseModel):
  count_conversations: int
  conversations: list[ConversationResponse]


class ChatMessage(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID4
    conversation_id: UUID4
    role: Literal["user", "ai"]
    content: str
    created_at: datetime

    @field_serializer("created_at")
    def serialize_in_ist(self, value: datetime) -> datetime:
        return value.astimezone(IST)


class ConversationMessagesResponse(BaseModel):
    conversation_id: UUID4
    messages: list[ChatMessage]


class UserMessageRequest(BaseModel):
    message_content: str


class AIMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    conversation_id: UUID4
    content: str

class ChatResponseModel(BaseModel):
    chat_id: UUID4
    ai_message: str
    chat_history_messages: list[str]

class LLMConfiguration(BaseModel): ...