"""Catalog QA node for music browsing and purchasing.

Handles everything music-related:
- Browsing genres, artists, albums
- Searching for tracks
- Detecting lyrics identification requests (routes to lyrics subgraph)
- Initiating purchases (hands off to purchase subgraph)

catalog_qa is the "music brain" - it handles browsing AND detects lyrics intent.
"""

from typing import Literal, Optional
import json
import ast

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from src.state import SupportState
from src.tools.catalog import (
    list_genres,
    artists_in_genre,
    albums_by_artist,
    tracks_in_album,
    find_track,
)
from src.tools.account import check_if_already_purchased


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

## PURCHASE GUIDANCE:
When showing track details, tell the user they can say "buy it" or "purchase it" to start the purchase flow.
Do NOT ask "Would you like to proceed with the purchase?" or similar yes/no questions.
Instead, say something like: "If you'd like to purchase this track, just say 'buy it'!"

Examples of good responses after showing track info:
- "Here's the track info! Say 'buy it' if you want to purchase."
- "Found it! Let me know if you want to buy it."

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


def _extract_track_from_tool_results(state: SupportState) -> Optional[dict]:
    """Extract track info from recent tool results.
    
    When user browses tracks (via find_track or tracks_in_album), we should
    remember the track(s) shown so "buy it" can work.
    
    Returns dict with track_id, track_name, artist, price if found, else None.
    Only returns info if there's a SINGLE track or a clearly focused track.
    """
    messages = state.get("messages", [])
    last_msg = _get_last_user_message(state).lower().strip()
    
    # Look at recent messages for tool results - use larger window for longer conversations
    # We search backwards to find the most recent relevant tool result
    for msg in reversed(messages[-20:]):
        if isinstance(msg, ToolMessage):
            try:
                # Parse the tool result
                content = msg.content
                data = None
                
                if isinstance(content, str):
                    # Try JSON first (standard format)
                    try:
                        data = json.loads(content)
                    except json.JSONDecodeError:
                        # Try Python literal syntax (single quotes, etc.)
                        try:
                            data = ast.literal_eval(content)
                        except (ValueError, SyntaxError):
                            continue
                elif isinstance(content, list):
                    # Already a Python list
                    data = content
                else:
                    continue
                
                # Check if it's a list that contains tracks (has TrackId)
                if isinstance(data, list) and len(data) > 0:
                    first_item = data[0]
                    if not isinstance(first_item, dict) or "TrackId" not in first_item:
                        # Not a track list (might be albums, genres, etc.)
                        continue
                    
                    # If single track, use it
                    if len(data) == 1:
                        track = data[0]
                        return {
                            "track_id": track.get("TrackId"),
                            "track_name": track.get("TrackName", "Unknown Track"),
                            "artist": track.get("ArtistName", "Unknown Artist"),
                            "price": track.get("UnitPrice", 0.99)
                        }
                    
                    # Multiple tracks - check if user asked about a specific one
                    for track in data:
                        if isinstance(track, dict) and "TrackName" in track:
                            track_name = track.get("TrackName", "")
                            track_name_lower = track_name.lower()
                            
                            # Match if track name is in user message OR user message matches track name
                            # This handles both "Letterbomb" and "play letterbomb" cases
                            if track_name_lower and (
                                track_name_lower in last_msg or 
                                last_msg in track_name_lower or
                                track_name_lower == last_msg
                            ):
                                return {
                                    "track_id": track.get("TrackId"),
                                    "track_name": track_name,
                                    "artist": track.get("ArtistName", "Unknown Artist"),
                                    "price": track.get("UnitPrice", 0.99)
                                }
            except Exception:
                continue
    
    return None


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


def _detect_previous_track_purchase_intent(state: SupportState) -> bool:
    """Detect if user wants to PURCHASE a previously identified track.
    
    IMPORTANT: This requires BOTH:
    1. Purchase intent (buy, purchase, get it, want it)
    2. Reference to a previous track (from before, that song, it, etc.)
    
    Does NOT trigger for simple recall questions like "what was that song?"
    """
    last_message = _get_last_user_message(state).lower()
    
    # Words indicating purchase intent - REQUIRED
    purchase_words = ["buy", "purchase", "get", "want"]
    has_purchase_intent = any(word in last_message for word in purchase_words)
    
    # If no purchase intent, don't trigger (fixes "what was the song from earlier?" bug)
    if not has_purchase_intent:
        return False
    
    # Reference to a previous track
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
        "you just found",
        "you just identified",
        " it",  # "buy it", "get it" - space prefix to avoid matching "with"
    ]
    
    has_previous_ref = any(indicator in last_message for indicator in previous_track_indicators)
    
    return has_previous_ref


def _detect_purchase_confirmation(state: SupportState) -> bool:
    """Detect if user is confirming a purchase with simple 'yes' when track info is in state.
    
    This handles the case where:
    1. User views a track (state has last_identified_track_id)
    2. Bot asks if they want to buy (shouldn't happen with new prompt, but might)
    3. User says "yes" or "sure" or similar
    
    Only triggers for SHORT confirmatory responses when track info exists.
    """
    last_message = _get_last_user_message(state).lower().strip()
    
    # Only trigger for short messages (avoids false positives)
    if len(last_message) > 30:
        return False
    
    # Must have track info in state
    if not state.get("last_identified_track_id"):
        return False
    
    # Simple confirmation phrases
    confirmation_phrases = [
        "yes",
        "yeah",
        "yep",
        "yup",
        "sure",
        "ok",
        "okay",
        "confirm",
        "yes please",
        "yes i do",
        "i do",
        "definitely",
        "absolutely",
        "let's do it",
        "go ahead",
        "proceed",
    ]
    
    return any(last_message == phrase or last_message.startswith(phrase + " ") for phrase in confirmation_phrases)


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
    
    # Extract track info from recent tool results and store in state
    # This enables "buy it" to work after browsing tracks
    track_info = _extract_track_from_tool_results(state)
    state_updates = {}
    if track_info:
        state_updates["last_identified_track_id"] = track_info["track_id"]
        state_updates["last_identified_track_name"] = track_info["track_name"]
        state_updates["last_identified_track_artist"] = track_info["artist"]
    
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
    
    # Check if user wants to PURCHASE a previously identified track
    # Handles: "buy the song from before", "buy it", "get it", or simple "yes" confirmation
    # This uses STATE for reliability instead of relying on LLM memory
    if _detect_previous_track_purchase_intent(state) or _detect_purchase_confirmation(state):
        last_track_id = state.get("last_identified_track_id")
        last_track_name = state.get("last_identified_track_name")
        last_track_artist = state.get("last_identified_track_artist")
        customer_id = state.get("customer_id", 1)
        
        if last_track_id and last_track_name:
            # Check ownership FIRST before promising to set up purchase
            config = {"configurable": {"customer_id": customer_id}}
            ownership_check = check_if_already_purchased.invoke(
                {"track_id": last_track_id}, 
                config=config
            )
            
            if "Yes" in ownership_check:
                # Already owned - inform user directly, don't route to purchase
                return Command(
                    update={
                        "messages": [AIMessage(content=f"Great news! You already own **{last_track_name}** by **{last_track_artist}** - it's in your library! Is there anything else I can help you with?")]
                    },
                    goto="__end__"
                )
            
            # Not owned - proceed to purchase
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
    # Still include state_updates in case we extracted track info before the LLM call
    if response.tool_calls:
        return Command(
            update={"messages": [response], **state_updates},
            goto="catalog_tools"
        )
    
    # No tool calls - return response directly
    # Purchase detection is handled by:
    # 1. _detect_previous_track_purchase_intent() - catches "buy the song from before"
    # 2. User explicitly naming a track - LLM calls find_track, then user says "buy it"
    # 
    # NOTE: We removed the structured output double-call pattern here.
    # It was causing context pollution bugs (e.g., after lyrics identification,
    # asking "show me genres" would return info about the previous song).
    # See account_qa.py for the same fix.
    return Command(
        update={"messages": [response], **state_updates},
        goto="__end__"
    )
