"""Account QA node for customer account information.

Handles everything account-related:
- Profile information
- Invoices and purchase history
- Email changes (hands off to email_change subgraph)

Uses structured output for detecting email change intent instead of magic strings.
"""

from typing import Literal

from langchain_core.messages import SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.state import SupportState
from src.tools.account import (
    get_my_profile,
    get_my_invoices,
    get_my_invoice_lines,
)


class AccountResponse(BaseModel):
    """Structured response from the account agent."""
    
    response: str = Field(
        description="The response message to show the user"
    )
    
    wants_email_change: bool = Field(
        default=False,
        description="True if the user wants to change/update their email address"
    )


ACCOUNT_SYSTEM_PROMPT = """You are a helpful customer service assistant for a music store.

You can help customers with their account:
- View their profile information (name, email, phone, address)
- See their purchase history (invoices)
- View details of specific invoices (what tracks they bought)

You have secure access to their account - you don't need to ask for their customer ID.

## TOOLS AVAILABLE:
- get_my_profile: Get customer's profile info
- get_my_invoices: Get customer's invoice history
- get_my_invoice_lines: Get details of a specific invoice

## EMAIL CHANGES:
If a customer wants to UPDATE their email address, we need to verify their identity first.
Set wants_email_change to True in your response to start the verification process.

Example responses when they want to change email:
- "I'd be happy to help you update your email. We'll need to verify your identity first - I'll send a code to your phone on file."

Be helpful and protect their privacy - don't share sensitive info unless they ask."""


ACCOUNT_TOOLS = [
    get_my_profile,
    get_my_invoices,
    get_my_invoice_lines,
]


def account_qa_node(
    state: SupportState
) -> Command[Literal["account_tools", "email_change", "__end__"]]:
    """Handle account-related questions.
    
    Routes:
    - account_tools: When LLM wants to call tools
    - email_change: When email change intent detected (direct handoff to subgraph)
    - __end__: When response is complete
    """
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # First pass: let LLM use tools if needed
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
    
    # No tool calls - check for email change intent
    try:
        structured_model = model.with_structured_output(AccountResponse)
        structured_response = structured_model.invoke(messages + [response])
        
        if structured_response.wants_email_change:
            return Command(
                update={"messages": [AIMessage(content=structured_response.response)]},
                goto="email_change"  # Direct handoff to email_change subgraph
            )
        
        # Normal response
        return Command(
            update={"messages": [AIMessage(content=structured_response.response)]},
            goto="__end__"
        )
        
    except Exception:
        # Fallback to unstructured response
        return Command(
            update={"messages": [response]},
            goto="__end__"
        )
