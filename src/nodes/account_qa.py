"""Account QA node for customer account information.

Handles everything account-related:
- Profile information
- Invoices and purchase history
- Email changes (hands off to email_change subgraph)

Uses EXPLICIT keyword detection for email change intent (deterministic, not LLM-based).
"""

from typing import Literal

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from src.state import SupportState
from src.tools.account import (
    get_my_profile,
    get_my_invoices,
    get_my_invoice_lines,
)


ACCOUNT_SYSTEM_PROMPT = """You are a helpful customer service assistant for a music store.

## CONTEXT AWARENESS:
- Check the conversation history for information you've already shown the user
- If they ask "what was my phone?" or "show me that again", look at previous messages first
- Don't make them repeat requests - recall what you've already shared when possible

You can help customers with their account:
- View their profile information (name, email, phone, address)
- See their purchase history (invoices)
- View details of specific invoices (what tracks they bought)

You have secure access to their account - you don't need to ask for their customer ID.

## TOOLS AVAILABLE:
- get_my_profile: Get customer's profile info
- get_my_invoices: Get customer's invoice history
- get_my_invoice_lines: Get details of a specific invoice

Be helpful and protect their privacy - don't share sensitive info unless they ask.

NOTE: Email changes are handled separately - just answer profile/invoice questions here."""


ACCOUNT_TOOLS = [
    get_my_profile,
    get_my_invoices,
    get_my_invoice_lines,
]


def _get_last_user_message(state: SupportState) -> str:
    """Get the content of the last human message."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content.strip()
    return ""


def _detect_email_change_intent(state: SupportState) -> bool:
    """Detect if user EXPLICITLY wants to change their email.
    
    Uses deterministic keyword detection instead of LLM interpretation.
    This prevents false positives when conversation history is full of
    email change context but user is asking about something else.
    
    Only triggers for EXPLICIT email change requests in the CURRENT message.
    """
    last_message = _get_last_user_message(state).lower()
    
    # Must have both email-related word AND change-related word
    email_words = ["email", "e-mail", "mail address"]
    change_words = ["change", "update", "modify", "edit", "new", "different", "switch"]
    
    has_email = any(word in last_message for word in email_words)
    has_change = any(word in last_message for word in change_words)
    
    # Explicit phrases that always trigger
    explicit_phrases = [
        "change my email",
        "update my email",
        "new email",
        "different email",
        "change email",
        "update email",
        "modify my email",
        "edit my email",
    ]
    
    has_explicit = any(phrase in last_message for phrase in explicit_phrases)
    
    return has_explicit or (has_email and has_change)


def account_qa_node(
    state: SupportState
) -> Command[Literal["account_tools", "email_change", "__end__"]]:
    """Handle account-related questions.
    
    Routes:
    - email_change: When EXPLICIT email change intent detected (deterministic)
    - account_tools: When LLM wants to call tools
    - __end__: When response is complete
    
    Email change detection is EXPLICIT (keyword-based) not LLM-based.
    This prevents false positives from polluted conversation context.
    """
    
    # FIRST: Check for explicit email change intent (deterministic, not LLM)
    # This must happen BEFORE any LLM calls to avoid context pollution
    if _detect_email_change_intent(state):
        return Command(
            update={
                "messages": [AIMessage(content="I'd be happy to help you update your email address. We'll need to verify your identity first - I'll send a verification code to your phone on file.")]
            },
            goto="email_change"
        )
    
    # Not email change - handle as normal account query
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Let LLM use tools if needed
    model_with_tools = model.bind_tools(ACCOUNT_TOOLS)
    messages = [SystemMessage(content=ACCOUNT_SYSTEM_PROMPT)] + state["messages"]
    
    # Pass customer_id through config for the tools
    customer_id = state.get("customer_id", 1)
    config = {"configurable": {"customer_id": customer_id}}
    
    response = model_with_tools.invoke(messages, config=config)
    
    # If the model wants to call tools, route to tool node
    if response.tool_calls:
        return Command(
            update={"messages": [response]},
            goto="account_tools"
        )
    
    # No tool calls - return response directly
    # NOTE: We removed the structured output check for wants_email_change
    # because explicit keyword detection above handles it deterministically
    return Command(
        update={"messages": [response]},
        goto="__end__"
    )
