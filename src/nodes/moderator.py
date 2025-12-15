"""Moderator node for input safety.

A simple input guard that blocks inappropriate content before routing.
Runs before the supervisor to ensure safe conversations.
"""

from typing import Literal
import re

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from src.state import SupportState


# Simple patterns for content that should be blocked
# In production, you'd use a proper moderation API (OpenAI, Perspective, etc.)
BLOCKED_PATTERNS = [
    r'\b(hack|exploit|crack|steal|password)\b.*\b(account|database|system)\b',
    r'\b(sql|script)\s*injection\b',
    r'\b(bypass|circumvent)\s*(security|verification|auth)\b',
]

# Compile patterns for efficiency
BLOCKED_REGEX = [re.compile(p, re.IGNORECASE) for p in BLOCKED_PATTERNS]


def _is_blocked(text: str) -> bool:
    """Check if text matches any blocked patterns."""
    for pattern in BLOCKED_REGEX:
        if pattern.search(text):
            return True
    return False


def _get_last_user_message(state: SupportState) -> str:
    """Get the content of the last human message."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content.strip()
    return ""


def moderator_node(
    state: SupportState
) -> Command[Literal["supervisor", "__end__"]]:
    """Check user input for safety.
    
    A simple guard that:
    - Allows normal messages through to supervisor
    - Blocks suspicious/malicious patterns
    
    In production, you'd use a proper moderation service.
    """
    last_message = _get_last_user_message(state)
    
    if _is_blocked(last_message):
        return Command(
            update={
                "messages": [AIMessage(content="I'm sorry, but I can't help with that request. If you have questions about our music catalog or your account, I'd be happy to assist!")]
            },
            goto="__end__"
        )
    
    # Message is safe - proceed to supervisor
    return Command(goto="supervisor")
