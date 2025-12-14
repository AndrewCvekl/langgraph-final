"""Account QA node for customer account information.

Handles questions about profile, invoices, and purchase history.
Can detect email change intent for handoff.

Follows LangGraph "Thinking in LangGraph" design principles:
- Returns Command with explicit goto destination
- Type hints declare all possible destinations
- Routes to its own tool node (account_tools)
"""

from typing import Literal

from langchain_core.messages import SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from src.state import SupportState
from src.tools.account import (
    get_my_profile,
    get_my_invoices,
    get_my_invoice_lines,
)


ACCOUNT_SYSTEM_PROMPT = """You are a helpful customer service assistant for a music store.

You can help customers with their account:
- View their profile information (name, email, phone, address)
- See their purchase history (invoices)
- View details of specific invoices (what tracks they bought)

You have access to their account securely - you don't need to ask for their customer ID.

If a customer wants to UPDATE their email address, let them know you'll transfer them 
to our email verification process. Include "[EMAIL_CHANGE_INTENT]" at the end of your 
response so the system can route them appropriately.

Be helpful and protect their privacy - don't share sensitive info unless they ask for it.
If they have concerns about security, reassure them that we verify email changes with a 
code sent to their phone on file."""


ACCOUNT_TOOLS = [
    get_my_profile,
    get_my_invoices,
    get_my_invoice_lines,
]


def account_qa_node(
    state: SupportState
) -> Command[Literal["account_tools", "router", "__end__"]]:
    """Handle account-related questions.
    
    Uses customer-scoped tools and may detect email change intent.
    
    Returns Command with explicit routing:
    - account_tools: When LLM wants to call tools
    - router: When email change intent detected (router sends to email_change)
    - __end__: When response is complete
    """
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    model_with_tools = model.bind_tools(ACCOUNT_TOOLS)
    
    messages = [SystemMessage(content=ACCOUNT_SYSTEM_PROMPT)] + state["messages"]
    
    # Pass customer_id through config for the tools
    customer_id = state.get("customer_id", 1)  # Default to demo customer
    config = {"configurable": {"customer_id": customer_id}}
    
    response = model_with_tools.invoke(messages, config=config)
    
    # If the model wants to call tools, route to our dedicated tool node
    if response.tool_calls:
        return Command(
            update={"messages": [response]},
            goto="account_tools"
        )
    
    # Check for email change intent
    content = response.content
    
    if "[EMAIL_CHANGE_INTENT]" in content:
        # Clean up the message and route to email_change via router
        clean_content = content.replace("[EMAIL_CHANGE_INTENT]", "").strip()
        return Command(
            update={
                "messages": [AIMessage(content=clean_content)],
                "route": "email_change",
            },
            goto="router"
        )
    
    # Normal response - end this turn
    return Command(
        update={"messages": [response]},
        goto="__end__"
    )

