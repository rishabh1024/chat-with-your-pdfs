from uuid import UUID


class ConversationNotFoundError(Exception):
    def __init__(self, conversation_id: UUID, error_msg: str) -> None:
        self.conversation_id = conversation_id
        self.error = error_msg or f"Conversation {conversation_id} Not Found"
        super().__init__(self.error)


class ConversationAccessDeniedError(Exception):
    def __init__(self, conversation_id: UUID, error_msg: str) -> None:
        self.conversation_id = conversation_id
        self.error = error_msg or f"Access Denied for Conversation {conversation_id}"
        super().__init__(self.error)