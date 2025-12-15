"""State model for the customer support bot.

Simplified state following the principle of minimal core state.
Flow-specific state is handled by subgraphs.
"""

from typing import Annotated, Optional, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class SupportState(TypedDict):
    """Core state for the customer support bot graph.
    
    Follows the principle of minimal state:
    - Core fields used everywhere: messages, customer_id
    - Handoff context for subgraphs (set before entering, cleared after)
    
    Flow-specific state (email verification attempts, etc.) is managed
    internally by subgraphs, not polluting the main state.
    """
    
    # Core state - always present
    messages: Annotated[list[BaseMessage], add_messages]
    customer_id: int
    
    # Purchase handoff context (set by catalog_qa or lyrics_qa before entering purchase subgraph)
    # Cleared when purchase completes
    pending_track_id: Optional[int]
    pending_track_name: Optional[str]
    pending_track_price: Optional[float]
    
    # Lyrics workflow context - conversational flow (NOT HITL)
    # After lyrics identification, we ask user yes/no and wait for their NEXT message
    lyrics_query: Optional[str]  # The lyrics the user provided
    lyrics_awaiting_response: Optional[bool]  # True = waiting for user yes/no response
    lyrics_song_in_catalog: Optional[bool]  # Was the identified song in our catalog?
    lyrics_identified_song: Optional[str]  # Song name from identification
    lyrics_identified_artist: Optional[str]  # Artist name from identification
    lyrics_purchase_confirmed: Optional[bool]  # Set by lyrics subgraph when user confirms purchase
    
    # Current user message for subgraph handoff (prevents message accumulation)
    # Set by parent nodes before routing to subgraphs that use Input/Output schemas
    current_user_message: Optional[str]


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
        "lyrics_awaiting_response": None,
        "lyrics_song_in_catalog": None,
        "lyrics_identified_song": None,
        "lyrics_identified_artist": None,
        "lyrics_purchase_confirmed": None,
        "current_user_message": None,
    }
