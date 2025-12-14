"""Purchase flow node with HITL confirmation.

Handles track purchases with explicit user confirmation.
Uses LangGraph's interrupt() for human-in-the-loop confirmation.
"""

from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.types import interrupt, Command

from src.state import SupportState
from src.tools.purchase import create_invoice_for_track
from src.tools.account import check_if_already_purchased


def purchase_flow_node(
    state: SupportState
) -> Command[Literal["__end__"]]:
    """Handle track purchase with confirmation.
    
    Requires pending_track_id in state. Shows track details and
    asks for confirmation before creating the invoice.
    
    IMPORTANT: This node should only be entered when pending_track_id is set.
    The router ensures this by routing to catalog_qa first if no track is pending.
    After completion (confirm or cancel), this node goes to END to cleanly
    finish the graph invocation and allow the next user message to start fresh.
    """
    track_id = state.get("pending_track_id")
    track_name = state.get("pending_track_name", "Unknown Track")
    track_price = state.get("pending_track_price", 0.99)
    customer_id = state.get("customer_id", 1)  # Default to demo customer
    
    # Safety check: if no track_id, ask user to specify (shouldn't happen with proper routing)
    if not track_id:
        return Command(
            update={
                "messages": [AIMessage(content="I'd be happy to help you make a purchase! Could you please tell me which track you'd like to buy? You can search for a track by name or browse our catalog.")],
                "pending_track_id": None,
                "pending_track_name": None,
                "pending_track_price": None,
            },
            goto="__end__"
        )
    
    # Check if the customer already owns this track
    config = {"configurable": {"customer_id": customer_id}}
    ownership_check = check_if_already_purchased.invoke({"track_id": track_id}, config=config)
    if "Yes" in ownership_check:
        return Command(
            update={
                "messages": [AIMessage(content=f"Great news! You already own **{track_name}** - it's in your library! Is there anything else I can help you with?")],
                "pending_track_id": None,
                "pending_track_name": None,
                "pending_track_price": None,
            },
            goto="__end__"
        )
    
    # HITL: Confirm the purchase
    confirm = interrupt({
        "type": "confirm",
        "title": "Confirm Purchase",
        "message": f"""Please confirm your purchase:

**Track:** {track_name}
**Track ID:** {track_id}
**Price:** ${track_price:.2f}

This will charge your account and add the track to your library.""",
        "options": ["confirm", "cancel"]
    })
    
    if confirm.lower() != "confirm":
        # Cancel the purchase: clear all pending state and end the turn cleanly.
        # Going to END allows the next user message to start fresh from router.
        return Command(
            update={
                "messages": [AIMessage(content="No problem! The purchase has been cancelled. The track is still available if you change your mind. Is there anything else I can help you with?")],
                "pending_track_id": None,
                "pending_track_name": None,
                "pending_track_price": None,
            },
            goto="__end__"
        )
    
    # Execute the purchase
    result = create_invoice_for_track.invoke({"track_id": track_id}, config=config)
    
    # Clear purchase state and end the turn cleanly.
    # Going to END allows the next user message to start fresh from router.
    return Command(
        update={
            "messages": [AIMessage(content=result + "\n\nIs there anything else I can help you with?")],
            "pending_track_id": None,
            "pending_track_name": None,
            "pending_track_price": None,
        },
        goto="__end__"
    )

