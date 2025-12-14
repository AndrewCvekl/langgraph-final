"""Guardrail node for security + policy enforcement.

This node is intended to run BEFORE routing/agent behavior.

Policies enforced:
1) A customer must not be able to view/update/tamper with any other customer's data.
2) A customer must not have access to employer/employee (internal staff) information.

Design:
- Fast deterministic checks catch obvious violations cheaply.
- A structured LLM check handles ambiguous phrasing.
- If blocked, we return a refusal message and end the turn cleanly.
"""

from __future__ import annotations

import re
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.state import SupportState


class GuardrailDecision(BaseModel):
    """Structured output for guardrail decisions."""

    allow: bool = Field(description="True if the user's request is allowed.")
    category: Optional[str] = Field(
        default=None,
        description="If blocked, one of: other_customer_data, employer_info, other",
    )
    reason: str = Field(description="Short explanation for the decision.")


GUARDRAIL_SYSTEM_PROMPT = """You are a strict security guardrail for a digital music store assistant.

Your job: decide whether the user's LATEST message is allowed.

## Policy (must enforce)

1) Customer data isolation
- The customer may only view, update, or act on THEIR OWN customer data.
- They must NOT be able to view, change, update, delete, or tamper with any other customer's data.
- If the user asks for "another customer", "someone else's", "all customers", or provides a different CustomerId, it is NOT allowed.

2) Employer/Employee (internal) information
- The customer must NOT have access to employer/employee/staff/internal information.
- This includes employee records, staff emails/phones, salaries, HR data, internal IDs, etc.

## How to decide
- Focus only on the user's latest message (treat it as a fresh request).
- If the request is clearly disallowed by either policy, set allow=false and choose an appropriate category:
  - other_customer_data
  - employer_info
  - other
- If the request is about their own account, invoices, profile, purchasing music, or catalog browsing, it is allowed.

Return ONLY the structured output."""


# -----------------------------
# Fast-path heuristics (cheap)
# -----------------------------

_OTHER_CUSTOMER_PATTERNS = [
    re.compile(r"\ball customers\b", re.IGNORECASE),
    re.compile(r"\blist (all )?customers\b", re.IGNORECASE),
    re.compile(r"\bother customer'?s\b", re.IGNORECASE),
    re.compile(r"\bsomeone else'?s\b", re.IGNORECASE),
    re.compile(r"\banother customer\b", re.IGNORECASE),
    re.compile(r"\bview (a|another|someone else's) (profile|invoice|invoices|account)\b", re.IGNORECASE),
    re.compile(r"\bchange (a|another|someone else's) (email|address|phone|profile)\b", re.IGNORECASE),
    re.compile(r"\bCustomerId\b", re.IGNORECASE),
]

_EMPLOYER_PATTERNS = [
    re.compile(r"\bemployer\b", re.IGNORECASE),
    re.compile(r"\bemployee\b", re.IGNORECASE),
    re.compile(r"\bstaff\b", re.IGNORECASE),
    re.compile(r"\bmanager\b", re.IGNORECASE),
    re.compile(r"\bHR\b", re.IGNORECASE),
    re.compile(r"\bpayroll\b", re.IGNORECASE),
    re.compile(r"\bsalary\b", re.IGNORECASE),
    re.compile(r"\bEmployeeId\b", re.IGNORECASE),
]

_CUSTOMER_ID_MENTION = re.compile(r"\bcustomer\s*id\s*[:#]?\s*(\d+)\b", re.IGNORECASE)


def _get_last_user_message(state: SupportState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return (msg.content or "").strip()
    return ""


def _refusal_message(category: str) -> str:
    if category == "other_customer_data":
        return (
            "I can’t help with accessing or changing another customer’s data. "
            "I can help with **your** account (your profile, invoices, and purchases)."
        )
    if category == "employer_info":
        return (
            "I can’t help with employer/employee (internal staff) information. "
            "I can help with the music catalog or your account."
        )
    return (
        "I can’t help with that request. "
        "I can help with the music catalog or your account."
    )


def guardrails_node(state: SupportState) -> dict:
    """Enforce basic security guardrails before routing."""

    last_user_msg = _get_last_user_message(state)
    if not last_user_msg:
        return {"guardrail_blocked": False, "guardrail_reason": None}

    customer_id = int(state.get("customer_id", 1))

    # Fast path: internal staff/employer info
    for pat in _EMPLOYER_PATTERNS:
        if pat.search(last_user_msg):
            return {
                "guardrail_blocked": True,
                "guardrail_reason": "Request attempts to access employer/employee (internal) information.",
                "route": "final",
                "messages": [AIMessage(content=_refusal_message("employer_info"))],
            }

    # Fast path: other customer data intent
    for pat in _OTHER_CUSTOMER_PATTERNS:
        if pat.search(last_user_msg):
            return {
                "guardrail_blocked": True,
                "guardrail_reason": "Request attempts to access another customer's data.",
                "route": "final",
                "messages": [AIMessage(content=_refusal_message("other_customer_data"))],
            }

    # Fast path: explicit customer id mismatch
    m = _CUSTOMER_ID_MENTION.search(last_user_msg)
    if m:
        requested_id = int(m.group(1))
        if requested_id != customer_id:
            return {
                "guardrail_blocked": True,
                "guardrail_reason": f"User requested CustomerId={requested_id} (authenticated customer_id={customer_id}).",
                "route": "final",
                "messages": [AIMessage(content=_refusal_message("other_customer_data"))],
            }

    # LLM backstop for ambiguous phrasing
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    structured = model.with_structured_output(GuardrailDecision)

    decision: GuardrailDecision = structured.invoke(
        [
            SystemMessage(content=GUARDRAIL_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Authenticated customer_id: {customer_id}\n"
                    f"User message: {last_user_msg}"
                )
            ),
        ]
    )

    if decision.allow:
        return {"guardrail_blocked": False, "guardrail_reason": None}

    category = decision.category or "other"
    return {
        "guardrail_blocked": True,
        "guardrail_reason": decision.reason,
        "route": "final",
        "messages": [AIMessage(content=_refusal_message(category))],
    }

