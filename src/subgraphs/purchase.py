"""Purchase subgraph with HITL confirmation.

A proper multi-node workflow for handling track purchases.
Following "Thinking in LangGraph" - each step is its own node.

Flow:
1. check_ownership: Verify customer doesn't already own the track
2. confirm_purchase: HITL interrupt for purchase confirmation
3. execute_purchase: Create invoice and complete the purchase
"""

from typing import Annotated, Optional, Literal

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from typing_extensions import TypedDict

from src.tools.purchase import create_invoice_for_track
from src.tools.account import check_if_already_purchased


class PurchaseState(TypedDict):
    """State for the purchase subgraph.
    
    Inherits context from parent state.
    """
    # Inherited from parent (input)
    messages: Annotated[list[BaseMessage], add_messages]
    customer_id: int
    pending_track_id: Optional[int]
    pending_track_name: Optional[str]
    pending_track_price: Optional[float]


def check_ownership_node(
    state: PurchaseState
) -> Command[Literal["confirm_purchase", "__end__"]]:
    """Step 1: Check if customer already owns the track.
    
    Routes to:
    - confirm_purchase: If track exists and not owned
    - __end__: If no track specified or already owned
    """
    track_id = state.get("pending_track_id")
    track_name = state.get("pending_track_name", "Unknown Track")
    customer_id = state.get("customer_id", 1)
    
    # Safety check: if no track_id, ask user to specify
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
    
    # Track exists and not owned - proceed to confirmation
    return Command(goto="confirm_purchase")


def confirm_purchase_node(
    state: PurchaseState
) -> Command[Literal["execute_purchase", "__end__"]]:
    """Step 2: HITL confirmation before purchase.
    
    Uses interrupt() to pause and wait for user confirmation.
    
    Routes to:
    - execute_purchase: If user confirms
    - __end__: If user cancels
    """
    track_id = state.get("pending_track_id")
    track_name = state.get("pending_track_name", "Unknown Track")
    track_price = state.get("pending_track_price", 0.99)
    
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
        # User cancelled
        return Command(
            update={
                "messages": [AIMessage(content="No problem! The purchase has been cancelled. The track is still available if you change your mind. Is there anything else I can help you with?")],
                "pending_track_id": None,
                "pending_track_name": None,
                "pending_track_price": None,
            },
            goto="__end__"
        )
    
    # User confirmed - proceed to execute
    return Command(goto="execute_purchase")


def execute_purchase_node(state: PurchaseState) -> dict:
    """Step 3: Execute the purchase.
    
    Creates the invoice in the database.
    Always returns to END after completion.
    """
    track_id = state.get("pending_track_id")
    customer_id = state.get("customer_id", 1)
    
    # Execute the purchase
    config = {"configurable": {"customer_id": customer_id}}
    result = create_invoice_for_track.invoke({"track_id": track_id}, config=config)
    
    return {
        "messages": [AIMessage(content=result + "\n\nIs there anything else I can help you with?")],
        "pending_track_id": None,
        "pending_track_name": None,
        "pending_track_price": None,
    }


def build_purchase_subgraph() -> StateGraph:
    """Build the purchase subgraph.
    
    Graph structure:
        START → check_ownership → confirm_purchase → execute_purchase → END
                     │                   │
                     ▼                   ▼
                    END                 END
               (already owned)      (cancelled)
    """
    builder = StateGraph(PurchaseState)
    
    # Add nodes
    builder.add_node("check_ownership", check_ownership_node)
    builder.add_node("confirm_purchase", confirm_purchase_node)
    builder.add_node("execute_purchase", execute_purchase_node)
    
    # Add edges
    builder.add_edge(START, "check_ownership")
    # check_ownership and confirm_purchase use Command for routing
    builder.add_edge("execute_purchase", END)
    
    return builder


# Compiled subgraph for import
purchase_subgraph = build_purchase_subgraph().compile()
