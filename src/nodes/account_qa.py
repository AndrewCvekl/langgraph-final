"""Account QA node for customer account information.

Handles questions about profile, invoices, and purchase history.
Can detect email change intent for handoff.
"""

from langchain_core.messages import SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

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


def account_qa_node(state: SupportState) -> dict:
    """Handle account-related questions.
    
    Uses customer-scoped tools and may detect email change intent.
    """
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    model_with_tools = model.bind_tools(ACCOUNT_TOOLS)
    
    messages = [SystemMessage(content=ACCOUNT_SYSTEM_PROMPT)] + state["messages"]
    
    # Pass customer_id through config for the tools
    customer_id = state.get("customer_id", 1)  # Default to demo customer
    config = {"configurable": {"customer_id": customer_id}}
    
    response = model_with_tools.invoke(messages, config=config)
    
    # Check if the model wants to call tools
    if response.tool_calls:
        return {"messages": [response]}
    
    # Check for email change intent
    content = response.content
    result = {"messages": [response]}
    
    if "[EMAIL_CHANGE_INTENT]" in content:
        result["route"] = "email_change"
        # Clean up the message
        clean_content = content.replace("[EMAIL_CHANGE_INTENT]", "").strip()
        result["messages"] = [AIMessage(content=clean_content)]
    
    return result

