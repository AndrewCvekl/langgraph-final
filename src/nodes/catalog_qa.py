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


class IntentClassification(BaseModel):
    """Classify the user's intent."""
    
    intent: Literal["lyrics", "browse", "purchase", "other"] = Field(
        description="""The user's intent:
        - lyrics: User provided song lyrics and wants to identify the song
        - browse: User wants to browse/search the catalog (genres, artists, albums, tracks)
        - purchase: User wants to buy a specific track they already know
        - other: General conversation or unclear intent
        """
    )
    
    reasoning: str = Field(
        description="Brief explanation of why this intent was detected"
    )


class CatalogResponse(BaseModel):
    """Structured response from the catalog agent."""
    
    response: str = Field(
        description="The response to the user's CURRENT/LATEST query. Focus only on what they just asked, not previous conversation topics."
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

## CRITICAL: Always respond to the user's LATEST message
- Focus on what the user is asking NOW, not previous topics
- If there was a previous purchase discussion that ended, move on
- Each new query deserves a fresh, focused response

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
1. FIRST use find_track to look it up and confirm it exists
2. ONLY AFTER confirming, set the purchase fields in your response
3. Include the TrackId, track name, and price

Examples:
- "I want to buy Kashmir" → Use find_track("Kashmir"), then set purchase_track_id, name, price
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
    """Use LLM to detect if the user is asking about lyrics."""
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    classifier = model.with_structured_output(IntentClassification)
    
    last_message = _get_last_user_message(state)
    
    classification = classifier.invoke([
        SystemMessage(content="""Classify the user's intent for a music store chatbot.

LYRICS intent means the user provided SONG LYRICS and wants to identify what song it is.
Examples of LYRICS intent:
- "What song goes 'we will rock you'?"
- "Is this the real life, is this just fantasy"
- "What song has the lyrics 'hello from the other side'?"

BROWSE intent means they want to explore the catalog.
PURCHASE intent means they want to buy a specific track.
OTHER is for greetings, thanks, or unclear queries."""),
        HumanMessage(content=last_message)
    ])
    
    return classification.intent == "lyrics"


def catalog_qa_node(
    state: SupportState
) -> Command[Literal["catalog_tools", "lyrics", "purchase", "__end__"]]:
    """Handle catalog-related questions, lyrics identification, and purchases.
    
    As the "music brain", catalog_qa owns all music-related context including:
    - Browsing and searching the catalog
    - Detecting lyrics intent and routing to lyrics subgraph
    - Continuing lyrics conversations (when lyrics_awaiting_response is True)
    - Initiating purchases
    
    Routes:
    - lyrics: When user provides lyrics OR when continuing a lyrics conversation
    - catalog_tools: When LLM wants to call catalog tools
    - purchase: When purchase intent detected (direct handoff to subgraph)
    - __end__: When response is complete
    """
    
    # First, check if we're in the middle of a lyrics conversation
    # (user is responding to "Would you like to purchase?" or "Want us to add it?")
    if state.get("lyrics_awaiting_response"):
        # Continue the lyrics conversation - subgraph's router will handle the response
        return Command(goto="lyrics")
    
    # Check if this is a NEW lyrics identification request
    if _detect_lyrics_intent(state):
        # Route to lyrics subgraph - it will handle the identification workflow
        last_message = _get_last_user_message(state)
        return Command(
            update={"lyrics_query": last_message},
            goto="lyrics"
        )
    
    # Not lyrics - handle as normal catalog query
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
