"""Catalog QA node for music browsing and purchasing.

Handles everything music-related:
- Browsing genres, artists, albums
- Searching for tracks
- Detecting lyrics identification requests (routes to lyrics subgraph)
- Initiating purchases (hands off to purchase subgraph)

catalog_qa is the "music brain" - it handles browsing AND detects lyrics intent.
"""

from typing import Literal, Optional

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.state import SupportState
from src.tools.catalog import (
    list_genres,
    artists_in_genre,
    albums_by_artist,
    tracks_in_album,
    find_track,
)


class CatalogResponse(BaseModel):
    """Structured response from the catalog agent."""
    
    response: str = Field(
        description="The response to the user's query. If they reference something from earlier in the conversation (like 'the song from before'), use that context."
    )
    
    purchase_track_id: Optional[int] = Field(
        default=None,
        description="TrackId if the user wants to purchase a track (only set after confirming track exists)"
    )
    
    purchase_track_name: Optional[str] = Field(
        default=None,
        description="Track name for purchase confirmation"
    )
    
    purchase_track_price: Optional[float] = Field(
        default=None,
        description="Track price for purchase confirmation"
    )


CATALOG_SYSTEM_PROMPT = """You are a helpful music store assistant specializing in our catalog.

## CONTEXT AWARENESS:
- Always check the conversation history for context
- When users say "the song from before", "that track", "it", etc., look back at the chat to find what they're referring to
- If a song was previously identified (via lyrics or search), remember it for follow-up questions

## You can help customers:
- Browse genres available in the store
- Find artists in a specific genre
- View albums by an artist
- See tracks in an album
- Search for specific tracks by name or TrackId

## TOOLS AVAILABLE:
- list_genres: Show all music genres
- artists_in_genre: Find artists in a genre
- albums_by_artist: Get albums by an artist
- tracks_in_album: See tracks in an album
- find_track: Search for tracks by name or TrackId

## IMPORTANT: When showing tracks, ALWAYS include:
- TrackId (customers need this to purchase)
- Price
- Artist and album info

## PURCHASE FLOW:
If a customer wants to buy a track:
1. First check if they're referring to a song from earlier in the conversation
2. If it's a NEW track request, use find_track to look it up
3. ONLY AFTER confirming the track exists, set the purchase fields in your response
4. Include the TrackId, track name, and price

Examples:
- "I want to buy the song from before" → Check conversation history, find the previously mentioned track
- "I want to buy Kashmir" → Use find_track("Kashmir"), then set purchase fields
- "Show me rock artists" → Just respond with the list, no purchase fields

Be helpful and conversational. If you can't find what they're looking for, suggest alternatives."""


CATALOG_TOOLS = [
    list_genres,
    artists_in_genre,
    albums_by_artist,
    tracks_in_album,
    find_track,
]


def _get_last_user_message(state: SupportState) -> str:
    """Get the content of the last human message."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content.strip()
    return ""


def _detect_lyrics_intent(state: SupportState) -> bool:
    """Detect if the user is asking about lyrics identification.
    
    Simple keyword check - only triggers lyrics flow if the user
    explicitly mentions lyrics-related terms. This prevents false positives
    when users are just browsing tracks/albums with song names.
    """
    last_message = _get_last_user_message(state).lower()
    
    lyrics_indicators = [
        "lyrics",
        "lyric", 
        "what song goes",
        "what song has",
        "what song is",
        "identify the song",
        "identify this song",
        "what's the song",
        "name that song",
        "which song goes",
    ]
    
    return any(indicator in last_message for indicator in lyrics_indicators)


def _detect_previous_track_reference(state: SupportState) -> bool:
    """Detect if the user is referring to a previously identified track.
    
    Catches phrases like:
    - "the song from before"
    - "that track"
    - "buy it"
    - "purchase the song we talked about"
    - "the one you found"
    """
    last_message = _get_last_user_message(state).lower()
    
    previous_track_indicators = [
        "song from before",
        "track from before",
        "the song we",
        "the track we",
        "that song",
        "that track",
        "the one you found",
        "the one we found",
        "the one you mentioned",
        "from earlier",
        "we just talked about",
        "we were just",
        "you just found",
        "you just identified",
    ]
    
    # Check for purchase intent combined with reference
    purchase_with_reference = [
        "buy it",
        "purchase it",
        "get it",
        "want it",
    ]
    
    has_previous_ref = any(indicator in last_message for indicator in previous_track_indicators)
    has_purchase_ref = any(ref in last_message for ref in purchase_with_reference)
    
    # Also check for simple "buy" + no specific track name (implies reference)
    has_buy_intent = any(word in last_message for word in ["buy", "purchase"])
    
    return has_previous_ref or (has_purchase_ref and has_buy_intent)


def catalog_qa_node(
    state: SupportState
) -> Command[Literal["catalog_tools", "lyrics", "purchase", "__end__"]]:
    """Handle catalog-related questions, lyrics identification, and purchases.
    
    As the "music brain", catalog_qa handles:
    - Browsing and searching the catalog
    - Detecting lyrics intent and routing to lyrics subgraph
    - Initiating purchases (including references to previously identified tracks)
    
    Routes:
    - lyrics: When user provides lyrics to identify
    - catalog_tools: When LLM wants to call catalog tools
    - purchase: When purchase intent detected (direct handoff to subgraph)
    - __end__: When response is complete
    """
    
    # Get the current user message for subgraph handoff
    last_message = _get_last_user_message(state)
    
    # Check if this is a lyrics identification request
    if _detect_lyrics_intent(state):
        # Route to lyrics subgraph - it handles identification with HITL
        return Command(
            update={
                "lyrics_query": last_message,
                "current_user_message": last_message,
            },
            goto="lyrics"
        )
    
    # Check if user is referring to a previously identified track (e.g., "buy the song from before")
    # This uses STATE for reliability instead of relying on LLM memory
    if _detect_previous_track_reference(state):
        last_track_id = state.get("last_identified_track_id")
        last_track_name = state.get("last_identified_track_name")
        last_track_artist = state.get("last_identified_track_artist")
        
        if last_track_id and last_track_name:
            # We have a previously identified track - use it directly!
            return Command(
                update={
                    "messages": [AIMessage(content=f"Sure! Let me set up your purchase for **{last_track_name}** by **{last_track_artist}**...")],
                    "pending_track_id": last_track_id,
                    "pending_track_name": last_track_name,
                    "pending_track_price": 0.99,  # Default price, purchase subgraph will verify
                },
                goto="purchase"
            )
        elif last_track_name:
            # We have a track name but no ID (wasn't in catalog) - inform user
            return Command(
                update={
                    "messages": [AIMessage(content=f"I remember we found **{last_track_name}** by **{last_track_artist}**, but unfortunately that track isn't available in our catalog for purchase. Would you like me to help you find something similar?")]
                },
                goto="__end__"
            )
        # No previous track in state - fall through to LLM handling
    
    # Not lyrics and not a previous track reference - handle as normal catalog query
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Let LLM use tools if needed
    model_with_tools = model.bind_tools(CATALOG_TOOLS)
    messages = [SystemMessage(content=CATALOG_SYSTEM_PROMPT)] + state["messages"]
    
    response = model_with_tools.invoke(messages)
    
    # If the model wants to call tools, route to tool node
    if response.tool_calls:
        return Command(
            update={"messages": [response]},
            goto="catalog_tools"
        )
    
    # No tool calls - try to get structured response for purchase detection
    try:
        structured_model = model.with_structured_output(CatalogResponse)
        structured_response = structured_model.invoke(messages + [response])
        
        # Check for purchase intent
        if structured_response.purchase_track_id:
            return Command(
                update={
                    "messages": [AIMessage(content=structured_response.response)],
                    "pending_track_id": structured_response.purchase_track_id,
                    "pending_track_name": structured_response.purchase_track_name,
                    "pending_track_price": structured_response.purchase_track_price or 0.99,
                },
                goto="purchase"  # Direct handoff to purchase subgraph
            )
        
        # Normal response
        return Command(
            update={"messages": [AIMessage(content=structured_response.response)]},
            goto="__end__"
        )
        
    except Exception:
        # Fallback to unstructured response
        return Command(
            update={"messages": [response]},
            goto="__end__"
        )
