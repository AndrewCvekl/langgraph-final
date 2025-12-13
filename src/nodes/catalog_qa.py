"""Catalog QA node for music browsing.

Handles questions about genres, artists, albums, and tracks.
Can detect purchase intent and extract TrackId for handoff.
"""

from langchain_core.messages import SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

from src.state import SupportState
from src.tools.catalog import (
    list_genres,
    artists_in_genre,
    albums_by_artist,
    tracks_in_album,
    find_track,
)


CATALOG_SYSTEM_PROMPT = """You are a helpful music store assistant specializing in our catalog.

You can help customers:
- Browse genres available in the store
- Find artists in a specific genre
- View albums by an artist
- See tracks in an album
- Search for specific tracks by name or TrackId

When showing tracks, always include the TrackId and price - customers need the TrackId to make a purchase.

## IMPORTANT: Handling Purchase Intent

If a customer expresses interest in purchasing a track:
1. **ALWAYS use the find_track tool first** to look up the track by name if they mention a track name
2. If they reference a track from a previous listing (e.g., "buy track 3" or "yes" after you showed a track), use the TrackId from that listing
3. If they mention a number, look for the track in the most recent list you showed
4. **Only after you have confirmed the track exists**, include "[PURCHASE_INTENT: TrackId=X, Name=Y, Price=Z]" at the end of your response

Examples:
- "I want to buy Kashmir" → First use find_track("Kashmir"), then include [PURCHASE_INTENT: ...] with the results
- "yes" (after showing a track) → Include [PURCHASE_INTENT: ...] with the previously shown track
- "buy track 2969" → Use find_track to verify, then include [PURCHASE_INTENT: TrackId=2969, Name=..., Price=...]

Be helpful and conversational. If you can't find what they're looking for, suggest alternatives.

CRITICAL: When a customer wants to purchase a track by name (like "I want to buy Kashmir"):
1. You MUST use find_track tool to search for it first
2. Show them the track details
3. Include [PURCHASE_INTENT: TrackId=X, Name=Y, Price=Z] to trigger the purchase flow

Never include [PURCHASE_INTENT:...] without first verifying the track exists in our catalog!"""


CATALOG_TOOLS = [
    list_genres,
    artists_in_genre,
    albums_by_artist,
    tracks_in_album,
    find_track,
]


def catalog_qa_node(state: SupportState) -> dict:
    """Handle catalog-related questions.
    
    Uses tools to query the database and may detect purchase intent.
    """
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    model_with_tools = model.bind_tools(CATALOG_TOOLS)
    
    messages = [SystemMessage(content=CATALOG_SYSTEM_PROMPT)] + state["messages"]
    
    response = model_with_tools.invoke(messages)
    
    # Check if the model wants to call tools
    if response.tool_calls:
        return {"messages": [response]}
    
    # Check for purchase intent in the response
    content = response.content
    result = {"messages": [response]}
    
    # Parse purchase intent if present
    if "[PURCHASE_INTENT:" in content:
        import re
        match = re.search(
            r'\[PURCHASE_INTENT:\s*TrackId=(\d+),\s*Name=([^,]+),\s*Price=([^\]]+)\]',
            content
        )
        if match:
            result["pending_track_id"] = int(match.group(1))
            result["pending_track_name"] = match.group(2).strip()
            try:
                result["pending_track_price"] = float(match.group(3).strip().replace("$", ""))
            except ValueError:
                result["pending_track_price"] = 0.99
            result["route"] = "purchase_flow"
            
            # Don't add any message - purchase_flow will handle all messaging
            # This prevents duplicate/confusing messages
            result["messages"] = []
    
    return result

