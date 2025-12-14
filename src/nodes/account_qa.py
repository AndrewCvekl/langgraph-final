"""Account QA node for customer account information.

Handles questions about profile, invoices, and purchase history.
Can detect email change intent for handoff.
"""

import json
import re

from langchain_core.messages import SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.state import SupportState
from src.tools.account import (
    get_my_profile,
    get_my_invoices,
    get_my_invoice_lines,
)


class AccountResponse(BaseModel):
    """Structured response for the account lane (control signals separated)."""

    message: str = Field(..., description="User-facing assistant response")
    email_change: bool = Field(
        default=False,
        description="If true, triggers email_change workflow",
    )


def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_structured_account_response(raw: str) -> AccountResponse | None:
    try:
        cleaned = _strip_code_fences(raw)
        data = json.loads(cleaned)
        return AccountResponse.model_validate(data)
    except Exception:
        return None


ACCOUNT_SYSTEM_PROMPT = """You are a helpful customer service assistant for a music store.

You can help customers with their account:
- View their profile information (name, email, phone, address)
- See their purchase history (invoices)
- View details of specific invoices (what tracks they bought)

You have access to their account securely - you don't need to ask for their customer ID.

If a customer wants to UPDATE their email address, let them know you'll transfer them 
to our email verification process.

Be helpful and protect their privacy - don't share sensitive info unless they ask for it.
If they have concerns about security, reassure them that we verify email changes with a 
code sent to their phone on file."""


# NOTE: We keep the legacy [EMAIL_CHANGE_INTENT] convention for backwards
# compatibility, but we now prefer structured control signals. When you are
# ready to respond to the user (no tool calls), output ONLY valid JSON:
#
# {
#   "message": "<user-facing text>",
#   "email_change": true | false
# }
#
# - Set email_change=true ONLY when the user explicitly wants to update/change
#   their email address now.
ACCOUNT_SYSTEM_PROMPT += """

## Output Format (Preferred)
When you are ready to respond to the user (no tool calls), output ONLY valid JSON:
{
  "message": "string",
  "email_change": true or false
}
Do not wrap in markdown fences. Do not include any extra keys.
"""


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
    
    # Preferred: structured JSON control signal
    structured = _parse_structured_account_response(response.content)
    if structured is not None:
        result: dict = {"messages": [AIMessage(content=structured.message)]}
        if structured.email_change:
            result["route"] = "email_change"
        return result

    # Fallback: legacy tag parsing
    content = response.content
    result = {"messages": [response]}
    if "[EMAIL_CHANGE_INTENT]" in content:
        result["route"] = "email_change"
        clean_content = content.replace("[EMAIL_CHANGE_INTENT]", "").strip()
        result["messages"] = [AIMessage(content=clean_content)]

    return result

