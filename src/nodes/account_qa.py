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
- If in the conversation history they made changes to their email, make sure to reference the newest version!

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


def _email_recently_discussed(state: SupportState) -> bool:
    """Check if email was discussed in recent conversation.
    
    Looks at the last few messages to see if email was mentioned,
    either by user asking about it or system showing it.
    """
    messages = state.get("messages", [])
    # Check last 6 messages for email context
    recent_messages = messages[-6:] if len(messages) > 6 else messages
    
    for msg in recent_messages:
        content = ""
        if isinstance(msg, HumanMessage):
            content = msg.content.lower()
        elif isinstance(msg, AIMessage):
            content = msg.content.lower() if msg.content else ""
        
        # Check if email was discussed
        if any(word in content for word in ["email", "e-mail", "@"]):
            return True
    
    return False


def _detect_email_change_intent(state: SupportState) -> bool:
    """Detect if user wants to change their email.
    
    Uses deterministic keyword detection with context awareness:
    1. VIEW intent takes priority - "show me my email" should NOT trigger change
    2. Explicit phrases trigger ("change my email")
    3. Email + change words trigger ("update email")
    4. "change it" triggers if email was recently discussed
    
    This balances reliability with conversational flexibility.
    """
    last_message = _get_last_user_message(state).lower()
    
    # VIEW INTENT - If user wants to VIEW their email, don't trigger change flow
    # This takes priority over everything else
    view_words = ["show", "what is", "what's", "display", "see", "view", "tell me"]
    has_view_intent = any(word in last_message for word in view_words)
    
    # If viewing AND mentions email but NOT explicit change action, skip
    if has_view_intent and "change" not in last_message and "update" not in last_message:
        return False
    
    # Must have both email-related word AND change-related word
    email_words = ["email", "e-mail", "mail address"]
    # NOTE: "new" removed - too ambiguous ("show my new email" vs "I want a new email")
    change_words = ["change", "update", "modify", "edit", "different", "switch"]
    
    has_email = any(word in last_message for word in email_words)
    has_change = any(word in last_message for word in change_words)
    
    # Explicit phrases that always trigger - must be ACTION-oriented
    explicit_phrases = [
        "change my email",
        "update my email",
        "different email",
        "change email",
        "update email",
        "modify my email",
        "edit my email",
        "i want a new email",
        "i need a new email",
        "set a new email",
        "give me a new email",
    ]
    
    has_explicit = any(phrase in last_message for phrase in explicit_phrases)
    
    # Contextual reference: "change it", "update it", "can I change it"
    # Only triggers if email was recently discussed
    contextual_phrases = [
        "change it",
        "update it",
        "modify it",
        "edit it",
        "can i change",
        "want to change",
        "like to change",
        "need to change",
    ]
    
    has_contextual = any(phrase in last_message for phrase in contextual_phrases)
    email_in_context = _email_recently_discussed(state) if has_contextual else False
    
    return has_explicit or (has_email and has_change) or (has_contextual and email_in_context)


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
