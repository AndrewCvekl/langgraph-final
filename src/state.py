"""State model for the customer support bot.

Defines the typed state that flows through all nodes in the graph.
"""

from typing import Annotated, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class SupportState(TypedDict):
    """State for the customer support bot graph.
    
    Attributes:
        messages: Conversation history with add_messages reducer.
        customer_id: Authenticated customer ID (never user-supplied).
        route: Current route/lane for the conversation.
        
        # Email verification flow
        pending_email: New email address awaiting verification.
        verified: Whether email verification succeeded.
        verification_attempts: Number of code entry attempts.
        masked_phone: Masked phone number for display (e.g., "***-5555").
        verification_code: Generated verification code (mock).
        
        # Purchase flow
        pending_track_id: TrackId for pending purchase.
        pending_track_name: Track name for display.
        pending_track_price: Track price for confirmation.
        
        # Lyrics flow
        pending_genius_title: Song title from Genius lookup.
        pending_genius_artist: Artist name from Genius lookup.
        in_catalog: Whether the identified song is in our catalog.
    """
    
    # Core state - always present
    messages: Annotated[list[BaseMessage], add_messages]
    customer_id: int
    
    # Routing
    route: Optional[str]
    
    # Email verification flow
    pending_email: Optional[str]
    verified: bool
    verification_attempts: int
    masked_phone: str
    verification_code: Optional[str]  # For mock mode
    verification_id: Optional[str]  # Twilio verification ID for real SMS
    phone: Optional[str]  # Full phone number for Twilio
    
    # Purchase flow
    pending_track_id: Optional[int]
    pending_track_name: Optional[str]
    pending_track_price: Optional[float]
    
    # Lyrics flow
    pending_genius_title: Optional[str]
    pending_genius_artist: Optional[str]
    in_catalog: Optional[bool]


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
        "route": None,
        # Email verification
        "pending_email": None,
        "verified": False,
        "verification_attempts": 0,
        "masked_phone": "",
        "verification_code": None,
        "verification_id": None,
        "phone": None,
        # Purchase flow
        "pending_track_id": None,
        "pending_track_name": None,
        "pending_track_price": None,
        # Lyrics flow
        "pending_genius_title": None,
        "pending_genius_artist": None,
        "in_catalog": None,
    }

