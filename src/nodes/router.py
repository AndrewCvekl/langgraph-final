"""Router node for intent classification.

Uses structured output to classify user intent into one of the defined routes.
"""

from typing import Literal
import re

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.state import SupportState


# Simple patterns that should NOT trigger email_change flow
GREETING_PATTERNS = re.compile(
    r'^(hi|hello|hey|yo|sup|greetings|good\s*(morning|afternoon|evening)|thanks|thank you|ok|okay|cool|great|awesome|nice|perfect|sounds good)[\s!?.]*$',
    re.IGNORECASE
)

# Patterns that indicate purchase confirmation (when we have a pending track)
PURCHASE_CONFIRM_PATTERNS = re.compile(
    r'^(yes|yep|yeah|yup|sure|ok|okay|buy\s*(it)?|purchase|confirm|i\'?ll?\s*take\s*it|get\s*it|i\s*want\s*(it|to buy))[\s!?.]*$',
    re.IGNORECASE
)

# Patterns that indicate declining a purchase offer (when we have a pending track)
PURCHASE_DECLINE_PATTERNS = re.compile(
    r'^(no|nope|nah|no thanks|not now|not today|maybe later|i\'?ll?\s*pass|pass|never\s*mind|nevermind|not interested|not really|i\'?m\s*(good|ok|okay))[\s!?.]*$',
    re.IGNORECASE
)

# Simple affirmative/negative responses that should NOT trigger lyrics_flow
# These are conversational responses, not lyrics!
# Also includes purchase-related phrases that shouldn't be treated as lyrics
SIMPLE_RESPONSE_PATTERNS = re.compile(
    r'^(yes|no|yep|yeah|yup|nope|nah|sure|ok|okay|please|thanks|thank you|cool|great|awesome|nice|perfect|sounds good|definitely|absolutely|of course|no thanks|not really|maybe|i guess|can i buy it|i want to buy it|buy it|purchase it|i\'ll take it|get it for me)[\s!?.]*$',
    re.IGNORECASE
)


class RouteDecision(BaseModel):
    """Structured output for routing decisions."""
    
    route: Literal[
        "catalog_qa",
        "account_qa",
        "email_change",
        "lyrics_flow",
        "purchase_flow",
        "final"
    ] = Field(
        description="""The route to send the user to:
        - catalog_qa: Questions about music, artists, albums, genres, tracks
        - account_qa: Questions about their account, profile, invoices, purchase history
        - email_change: User wants to update their email address
        - lyrics_flow: User has lyrics and wants to identify a song
        - purchase_flow: User wants to buy a specific track (must have TrackId)
        - final: Conversation is complete or user said goodbye
        """
    )
    
    reasoning: str = Field(
        description="Brief explanation of why this route was chosen"
    )


ROUTER_SYSTEM_PROMPT = """You are a customer service router for a digital music store.

Your job is to analyze the customer's LATEST message and route them to the appropriate specialist.

IMPORTANT: Focus on what the user is asking NOW, not what they asked before. Each new message is a fresh request.

Routes:
1. **catalog_qa**: For browsing music - genres, artists, albums, tracks, searching for songs. Also use for general greetings like "hi", "hello", "hey" when the user is starting a new topic.
2. **account_qa**: For account questions - viewing profile, invoices, purchase history
3. **email_change**: ONLY when the user EXPLICITLY wants to update/change their email address in their CURRENT message (e.g., "change my email", "update my email address")
4. **lyrics_flow**: When the user provides song lyrics and wants to identify the song
5. **purchase_flow**: When the user wants to buy a specific track (they should mention a track name or have a TrackId)
6. **final**: When the conversation is complete or user says goodbye (bye, goodbye, thanks that's all, etc.)

CRITICAL Guidelines:
- Route based on the user's LATEST message, not the conversation history
- Simple greetings ("hi", "hello", "hey") should go to catalog_qa
- If a user just completed a task and says "hi" or asks a new question, treat it as a NEW request
- Only route to email_change if the user EXPLICITLY asks to change/update their email in their current message
- Don't continue previous flows unless the user explicitly asks to

Be decisive - pick the most appropriate single route based on the CURRENT user message."""


def _get_last_user_message(state: SupportState) -> str:
    """Get the content of the last human message."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content.strip()
    return ""


def router_node(state: SupportState) -> dict:
    """Classify user intent and set the route.
    
    Uses structured output to ensure we get a valid route.
    Implements proper state-aware routing for flows with pending state.
    
    Key routing logic:
    - If user confirms purchase AND we have pending_track_id -> purchase_flow
    - If purchase intent but no pending track -> catalog_qa to find it first
    - Greetings -> catalog_qa (not email_change)
    """
    # Get the last user message for safety checks
    last_user_msg = _get_last_user_message(state)
    has_pending_track = state.get("pending_track_id") is not None
    
    # Build state updates
    state_updates = {}
    
    # =========================================================================
    # FAST PATH: If user confirms purchase and we have a pending track,
    # route directly to purchase_flow without calling the LLM.
    # This is more reliable and faster than relying on LLM routing.
    # =========================================================================
    if has_pending_track and PURCHASE_CONFIRM_PATTERNS.match(last_user_msg):
        state_updates["route"] = "purchase_flow"
        return state_updates
    
    # =========================================================================
    # FAST PATH: If user DECLINES purchase and we have a pending track,
    # route to catalog_qa (not "final") so they get an acknowledgment.
    # Also clear the pending track state.
    # =========================================================================
    if has_pending_track and PURCHASE_DECLINE_PATTERNS.match(last_user_msg):
        state_updates["route"] = "catalog_qa"
        state_updates.update({
            "pending_track_id": None,
            "pending_track_name": None,
            "pending_track_price": None,
        })
        return state_updates
    
    # =========================================================================
    # STANDARD PATH: Use LLM to classify intent
    # =========================================================================
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_model = model.with_structured_output(RouteDecision)
    
    messages = [SystemMessage(content=ROUTER_SYSTEM_PROMPT)] + state["messages"]
    
    decision: RouteDecision = structured_model.invoke(messages)
    route = decision.route
    state_updates["route"] = route
    
    # SAFETY: If the LLM routes to email_change but the last message is just a greeting,
    # redirect to catalog_qa. This prevents the LLM from being confused by conversation history.
    if route == "email_change" and GREETING_PATTERNS.match(last_user_msg):
        route = "catalog_qa"
        state_updates["route"] = route
    
    # SAFETY: If the LLM routes to lyrics_flow but the last message is just a simple response
    # (yes/no/thanks/etc), redirect to catalog_qa. These are conversational responses, not lyrics!
    # This prevents "yes" from being interpreted as lyrics and matching songs like "Yesterday".
    if route == "lyrics_flow" and SIMPLE_RESPONSE_PATTERNS.match(last_user_msg):
        route = "catalog_qa"
        state_updates["route"] = route
    
    # IMPORTANT: Only route to purchase_flow if we have a pending track ready for confirmation.
    # If the user expresses purchase intent but we don't have track details yet,
    # route to catalog_qa first to find and identify the track.
    if route == "purchase_flow" and not has_pending_track:
        route = "catalog_qa"
        state_updates["route"] = route
    
    # SAFETY: If the LLM routes to "final" but the user just said a simple response
    # (like "no" declining a suggested purchase), redirect to catalog_qa so they get
    # an acknowledgment instead of silence.
    if route == "final" and SIMPLE_RESPONSE_PATTERNS.match(last_user_msg):
        route = "catalog_qa"
        state_updates["route"] = route
    
    # ALWAYS clear email state if not routing to email_change.
    # This ensures stale state from completed flows doesn't persist.
    if route != "email_change":
        state_updates.update({
            "pending_email": None,
            "verification_code": None,
            "verification_attempts": 0,
            "verified": False,
            "masked_phone": "",
        })
    
    # If routing away from lyrics_flow and there's stale lyrics state, clear it.
    if route != "lyrics_flow":
        if state.get("pending_genius_title") or state.get("pending_genius_artist"):
            state_updates.update({
                "pending_genius_title": None,
                "pending_genius_artist": None,
                "in_catalog": None,
            })
    
    # If NOT routing to purchase_flow, clear pending track state.
    # This prevents stale purchase state from persisting.
    if route != "purchase_flow":
        state_updates.update({
            "pending_track_id": None,
            "pending_track_name": None,
            "pending_track_price": None,
        })
    
    return state_updates

