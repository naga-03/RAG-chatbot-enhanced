from typing import Dict, List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class InMemoryChatMessageHistory(BaseChatMessageHistory):
    """
    Simple in-memory chat history implementation compatible with
    RunnableWithMessageHistory, without relying on langchain_community.
    """

    def __init__(self) -> None:
        self._messages: List[BaseMessage] = []

    @property
    def messages(self) -> List[BaseMessage]:
        return self._messages

    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)

    def add_user_message(self, content: str) -> None:
        self.add_message(HumanMessage(content=content))

    def add_ai_message(self, content: str) -> None:
        self.add_message(AIMessage(content=content))

    def get_messages(self) -> List[BaseMessage]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()


# In-memory store for session-based chat histories
session_histories: Dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Get or create a BaseChatMessageHistory for the given session_id.
    """
    if session_id not in session_histories:
        session_histories[session_id] = InMemoryChatMessageHistory()
    return session_histories[session_id]


class SessionMemory:
    """
    Lightweight, framework-agnostic session memory that stores the last N chat turns.
    Each turn includes user message, assistant message, and optional context summary.
    Automatically trims old turns when max_turns is exceeded.
    """

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self._turns: List[Dict[str, Optional[str]]] = []

    def add_turn(self, user_message: str, assistant_message: str, context_summary: Optional[str] = None):
        """
        Add a new turn to the session memory.
        :param user_message: The user's message.
        :param assistant_message: The assistant's message.
        :param context_summary: Optional context summary.
        """
        turn = {
            'user': user_message,
            'assistant': assistant_message,
            'context_summary': context_summary
        }
        self._turns.append(turn)
        if len(self._turns) > self.max_turns:
            self._turns.pop(0)  # Remove the oldest turn

    def get_formatted_history(self) -> str:
        """
        Return the chat history formatted as:
        User: <user message>
        Assistant: <assistant message>
        ...
        """
        formatted_lines = []
        for turn in self._turns:
            formatted_lines.append(f"User: {turn['user']}")
            formatted_lines.append(f"Assistant: {turn['assistant']}")
        return "\n".join(formatted_lines)
