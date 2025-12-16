"""State model for the customer support bot.

Simplified state following the principle of minimal core state.
Flow-specific state is handled by subgraphs.
"""

from typing import Annotated, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class SupportState(TypedDict):
    """Core state for the customer support bot graph.
    
    Follows the principle of minimal state:
    - Core fields used everywhere: messages, customer_id
    - Handoff context for subgraphs (set before entering, cleared after)
    
    Flow-specific state (email verification attempts, lyrics identification 
    progress, etc.) is managed internally by subgraphs, not polluting the main state.
    """
    
    # Core state - always present
    messages: Annotated[list[BaseMessage], add_messages]
    customer_id: int
    
    # Purchase handoff context (set by catalog_qa or lyrics subgraph before entering purchase subgraph)
    # Cleared when purchase completes
    pending_track_id: Optional[int]
    pending_track_name: Optional[str]
    pending_track_price: Optional[float]
    
    # Lyrics workflow - minimal handoff context
    # lyrics_query: The original lyrics text for identification
    # lyrics_purchase_confirmed: Set by lyrics subgraph when user confirms purchase
    lyrics_query: Optional[str]
    lyrics_purchase_confirmed: Optional[bool]
    
    # Current user message for subgraph handoff (prevents message accumulation)
    # Set by parent nodes before routing to subgraphs that use Input/Output schemas
    current_user_message: Optional[str]
    
    # Last identified track - PERSISTS across workflow completion
    # Used for "the song from before" type references
    # NOT cleared when user declines purchase - only overwritten when new track identified
    last_identified_track_id: Optional[int]
    last_identified_track_name: Optional[str]
    last_identified_track_artist: Optional[str]


def get_initial_state(customer_id: int = 1) -> dict:
    """Create initial state for a new conversation.
    
    Args:
        customer_id: The authenticated customer's ID.
        
    Returns:
        Initial state dictionary with defaults.
    """
    return {
        "messages": [],
        "customer_id": customer_id,
        "pending_track_id": None,
        "pending_track_name": None,
        "pending_track_price": None,
        "lyrics_query": None,
        "lyrics_purchase_confirmed": None,
        "current_user_message": None,
        "last_identified_track_id": None,
        "last_identified_track_name": None,
        "last_identified_track_artist": None,
    }
