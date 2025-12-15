"""Moderator node for input safety.

A simple input guard that blocks inappropriate content before routing.
Uses OpenAI moderation API for general safety + LLM for business rules.
"""

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from openai import OpenAI
from pydantic import BaseModel, Field

from src.state import SupportState


# Initialize OpenAI client for moderation API
client = OpenAI()


class ModerationCheck(BaseModel):
    """Result of business rules moderation check."""
    
    is_allowed: bool = Field(
        description="True if the request is allowed, False if it violates business rules"
    )
    reason: str = Field(
        default="",
        description="Brief reason if blocked (empty if allowed)"
    )


BUSINESS_RULES_PROMPT = """You are a security moderator for a music store customer support bot.

Your job is to check if the user's message violates any of these business rules:

## BLOCKED REQUESTS:
If a user wants to access someone else's (different id or username) data, make changes to someone else's account (different id or username)

ALLOWED REQUESTS:
BUT REMEMBER: They can view their OWN account information, and make changes to their OWN account, view their OWN purchase history


Analyze the user's message and determine if it's allowed or blocked.
Be reasonable - normal customer requests are fine. Only block clear attempts to access unauthorized data."""


def _is_blocked_by_openai(text: str) -> bool:
    """Check if text is flagged by OpenAI's moderation API (hate, violence, etc.)."""
    try:
        response = client.moderations.create(input=text)
        return response.results[0].flagged
    except Exception:
        return False


def _check_business_rules(text: str) -> ModerationCheck:
    """Check if text violates business rules using LLM."""
    try:
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        checker = model.with_structured_output(ModerationCheck)
        
        result = checker.invoke([
            SystemMessage(content=BUSINESS_RULES_PROMPT),
            HumanMessage(content=f"User message to check:\n\n{text}")
        ])
        return result
    except Exception:
        # If check fails, allow (don't block on errors)
        return ModerationCheck(is_allowed=True, reason="")


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
    
    Two-layer check:
    1. OpenAI Moderation API - catches hate speech, violence, etc.
    2. Business rules LLM check - catches unauthorized data access attempts
    """
    last_message = _get_last_user_message(state)
    
    if not last_message:
        return Command(goto="supervisor")
    
    # Layer 1: OpenAI moderation for general safety
    if _is_blocked_by_openai(last_message):
        return Command(
            update={
                "messages": [AIMessage(content="I'm sorry, but I can't help with that request. If you have questions about our music catalog or your account, I'd be happy to assist!")]
            },
            goto="__end__"
        )
    
    # Layer 2: Business rules check
    check = _check_business_rules(last_message)
    if not check.is_allowed:
        return Command(
            update={
                "messages": [AIMessage(content="I can only help you with your own account and purchases. I'm not able to access other customers' information or perform actions on their behalf. Is there something I can help you with for your account?")]
            },
            goto="__end__"
        )
    
    return Command(goto="supervisor")
