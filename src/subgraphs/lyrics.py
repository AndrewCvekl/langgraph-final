"""Lyrics identification subgraph.

A self-contained workflow for identifying songs from lyrics AND handling user responses.
Following "Thinking in LangGraph" - each step is its own node.

Flow (two modes):

MODE 1 - New lyrics query (lyrics_awaiting_response = False):
    router → identify_song → check_catalog → get_youtube → present_options → END
    
MODE 2 - Handling user response (lyrics_awaiting_response = True):
    router → handle_response → END (or → purchase via Command)

The router at START checks which mode we're in and routes appropriately.
This keeps all lyrics-related logic self-contained in one subgraph.
"""

from typing import Annotated, Optional, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from src.tools.mocks import genius_search, youtube_lookup, check_song_in_catalog
from src.tools.account import check_if_already_purchased
from src.db import get_db


class ExtractedLyrics(BaseModel):
    """Extracted lyrics from user message."""
    
    lyrics: str = Field(
        description="The actual song lyrics/words the user is trying to identify. Extract ONLY the lyrics themselves, not the surrounding question."
    )
    
    has_lyrics: bool = Field(
        description="True if the user provided actual lyrics to search for, False if they just asked about lyrics in general without providing any."
    )


class UserResponse(BaseModel):
    """Classify user's yes/no response."""
    
    response: Literal["yes", "no", "unclear"] = Field(
        description="""Classify the user's response:
        - yes: User wants to proceed (buy the track, or request it be added)
        - no: User declines / doesn't want to proceed
        - unclear: Can't determine intent (ask for clarification)
        """
    )


class LyricsState(TypedDict):
    """State for the lyrics identification subgraph."""
    # Inherited from parent
    messages: Annotated[list[BaseMessage], add_messages]
    customer_id: int
    
    # Lyrics workflow state
    lyrics_query: Optional[str]  # The lyrics the user provided
    identified_song: Optional[str]
    identified_artist: Optional[str]
    song_in_catalog: Optional[bool]
    track_id: Optional[int]
    track_price: Optional[float]
    already_owned: Optional[bool]
    youtube_url: Optional[str]
    
    # Conversational state - for handling yes/no on next turn
    lyrics_awaiting_response: Optional[bool]
    lyrics_song_in_catalog: Optional[bool]
    lyrics_identified_song: Optional[str]
    lyrics_identified_artist: Optional[str]
    
    # Output flag - set by handle_response when user confirms purchase
    # Parent graph uses this to route to purchase subgraph
    lyrics_purchase_confirmed: Optional[bool]
    
    # These are set for handoff to purchase subgraph
    pending_track_id: Optional[int]
    pending_track_name: Optional[str]
    pending_track_price: Optional[float]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_last_user_message(state: LyricsState) -> str:
    """Get the content of the last human message."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content.strip()
    return ""


def _extract_lyrics_from_messages(state: LyricsState) -> str:
    """Extract just the lyrics from the user's message using LLM.
    
    Users often wrap lyrics in context like:
    - "what song goes like [lyrics]"
    - "I heard this song, it went like [lyrics]"
    - "here are the lyrics: [lyrics]"
    - "do you know the song that goes [lyrics]"
    
    We use an LLM to extract just the lyrics portion for cleaner Genius search.
    """
    user_message = _get_last_user_message(state)
    
    if not user_message or len(user_message) < 5:
        return state.get("lyrics_query", "")
    
    # Use LLM to extract just the lyrics
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    extractor = model.with_structured_output(ExtractedLyrics)
    
    result = extractor.invoke([
        SystemMessage(content="""Extract the actual song lyrics from the user's message.

Users ask about songs in many ways:
- "what song goes like [lyrics]" → extract just the lyrics
- "I heard this song it went like [lyrics]" → extract just the lyrics  
- "here are the lyrics: [lyrics]" → extract just the lyrics
- "do you know [lyrics]" → extract just the lyrics
- "[just lyrics directly]" → extract the lyrics

Extract ONLY the lyrics/words they're trying to identify, NOT the question framing.

Examples:
- Input: "what song goes I got my mind set on you" → lyrics: "I got my mind set on you"
- Input: "do you know the song that goes we will rock you" → lyrics: "we will rock you"
- Input: "I heard a song today, the chorus went something like 'don't stop believing'" → lyrics: "don't stop believing"
- Input: "is this the real life is this just fantasy" → lyrics: "is this the real life is this just fantasy"

If the user is asking about lyrics in general without providing any actual lyrics, set has_lyrics to False."""),
        HumanMessage(content=user_message)
    ])
    
    if result.has_lyrics and result.lyrics:
        return result.lyrics.strip()
    
    # Fallback to the whole message if extraction failed
    return user_message


def _classify_response(user_message: str) -> str:
    """Classify the user's yes/no response using LLM."""
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    classifier = model.with_structured_output(UserResponse)
    
    result = classifier.invoke([
        SystemMessage(content="""Classify whether the user is saying YES or NO.

YES means they want to proceed - examples:
- "yes", "yeah", "sure", "ok", "okay", "please", "yep", "yup"
- "I'd like that", "sounds good", "let's do it", "go ahead"
- "definitely", "absolutely", "I want it"

NO means they decline - examples:
- "no", "nah", "nope", "no thanks", "not right now"
- "maybe later", "I'll pass", "nevermind"
- "I'm good", "that's okay" (declining)

UNCLEAR if you genuinely can't tell - but be generous in interpretation."""),
        HumanMessage(content=user_message)
    ])
    
    return result.response


# =============================================================================
# ROUTER NODE - Entry point that decides which flow to use
# =============================================================================

def router_node(
    state: LyricsState
) -> Command[Literal["identify_song", "handle_response"]]:
    """Route based on whether we're starting fresh or handling a response.
    
    This is the entry point for the lyrics subgraph.
    - If lyrics_awaiting_response is True → handle the user's yes/no
    - Otherwise → start the song identification flow
    """
    if state.get("lyrics_awaiting_response"):
        return Command(goto="handle_response")
    else:
        return Command(goto="identify_song")


# =============================================================================
# IDENTIFICATION FLOW NODES
# =============================================================================

def identify_song_node(state: LyricsState) -> Command[Literal["check_catalog", "__end__"]]:
    """Step 1: Identify the song from lyrics using Genius API."""
    lyrics = _extract_lyrics_from_messages(state)
    
    if not lyrics:
        return Command(
            update={
                "messages": [AIMessage(content="I'd be happy to help identify a song! Could you share some lyrics?")],
                "identified_song": None,
            },
            goto="__end__"
        )
    
    # Call Genius to identify the song
    result = genius_search.invoke({"lyrics_snippet": lyrics})
    
    # Parse the result to extract song info
    if "Could not identify" in result:
        return Command(
            update={
                "messages": [AIMessage(content=f"I couldn't identify a song from those lyrics. Could you try providing more of the lyrics or the song title directly?")],
                "identified_song": None,
            },
            goto="__end__"
        )
    
    # Extract song and artist from result
    song_title = None
    artist = None
    
    for line in result.split("\n"):
        if "Title:" in line:
            song_title = line.split("Title:")[-1].strip()
        elif "Artist:" in line:
            artist = line.split("Artist:")[-1].strip()
    
    if not song_title:
        return Command(
            update={
                "messages": [AIMessage(content="I had trouble parsing the song identification. Could you try again?")],
            },
            goto="__end__"
        )
    
    return Command(
        update={
            "identified_song": song_title,
            "identified_artist": artist or "Unknown Artist",
        },
        goto="check_catalog"
    )


def check_catalog_node(state: LyricsState) -> Command[Literal["get_youtube"]]:
    """Step 2: Check if the song is in our catalog."""
    song_title = state.get("identified_song", "")
    artist = state.get("identified_artist", "")
    customer_id = state.get("customer_id", 1)
    
    # Check if song is in catalog
    result = check_song_in_catalog.invoke({
        "song_title": song_title,
        "artist": artist
    })
    
    in_catalog = "Found in catalog" in result
    track_id = None
    track_price = 0.99
    already_owned = False
    
    if in_catalog:
        # Parse track ID and price
        import re
        track_id_match = re.search(r"TrackId:\s*(\d+)", result)
        price_match = re.search(r"Price:\s*\$?([\d.]+)", result)
        
        if track_id_match:
            track_id = int(track_id_match.group(1))
        if price_match:
            track_price = float(price_match.group(1))
        
        # Check if already owned
        if track_id:
            config = {"configurable": {"customer_id": customer_id}}
            ownership_result = check_if_already_purchased.invoke(
                {"track_id": track_id}, 
                config=config
            )
            already_owned = "Yes" in ownership_result
    
    return Command(
        update={
            "song_in_catalog": in_catalog,
            "track_id": track_id,
            "track_price": track_price,
            "already_owned": already_owned,
        },
        goto="get_youtube"
    )


def get_youtube_node(state: LyricsState) -> Command[Literal["present_options"]]:
    """Step 3: Get a YouTube video link."""
    song_title = state.get("identified_song", "")
    artist = state.get("identified_artist", "")
    
    # Get YouTube link
    result = youtube_lookup.invoke({
        "song_title": song_title,
        "artist": artist
    })
    
    # Extract URL from result
    youtube_url = result.strip() if result else None
    
    return Command(
        update={
            "youtube_url": youtube_url,
        },
        goto="present_options"
    )


def present_options_node(state: LyricsState) -> dict:
    """Step 4: Present the info and ask the user - CONVERSATIONAL (not HITL).
    
    Sets lyrics_awaiting_response = True so the NEXT user message
    will be routed back here via router → handle_response.
    """
    song_title = state.get("identified_song", "Unknown")
    artist = state.get("identified_artist", "Unknown Artist")
    in_catalog = state.get("song_in_catalog", False)
    track_price = state.get("track_price", 0.99)
    already_owned = state.get("already_owned", False)
    youtube_url = state.get("youtube_url", "")
    track_id = state.get("track_id")
    
    # Build the response message
    if already_owned:
        # They already own it - no need to ask anything
        msg = f"🎵 I found it! That's **{song_title}** by **{artist}**!\n\n"
        msg += f"Great news! You already own this track - it's in your library!"
        if youtube_url:
            msg += f"\n\nHere's the video if you want to take a listen: {youtube_url}"
        msg += "\n\nIs there anything else I can help you with?"
        
        return {
            "messages": [AIMessage(content=msg)],
            # Clear awaiting state - no action needed
            "lyrics_awaiting_response": False,
            "lyrics_song_in_catalog": None,
            "lyrics_identified_song": None,
            "lyrics_identified_artist": None,
        }
    
    if in_catalog:
        # Song is available for purchase - ask if they want to buy
        msg = f"🎵 I found it! That's **{song_title}** by **{artist}**!"
        if youtube_url:
            msg += f"\n\nHere's a video so you can take a listen: {youtube_url}"
        msg += f"\n\nThis track is available in our store for **${track_price:.2f}**."
        msg += "\n\n**Would you like to purchase it?**"
        
        return {
            "messages": [AIMessage(content=msg)],
            # Set awaiting state for next turn
            "lyrics_awaiting_response": True,
            "lyrics_song_in_catalog": True,
            "lyrics_identified_song": song_title,
            "lyrics_identified_artist": artist,
            # Set pending track info for potential purchase
            "pending_track_id": track_id,
            "pending_track_name": song_title,
            "pending_track_price": track_price,
        }
    
    else:
        # Song is NOT in catalog - ask if they'd like to see it added
        msg = f"🎵 I found it! That's **{song_title}** by **{artist}**!"
        if youtube_url:
            msg += f"\n\nHere's a video so you can take a listen: {youtube_url}"
        msg += "\n\nUnfortunately, this track isn't currently in our catalogue."
        msg += "\n\n**Is this the sort of song you'd like to see added?**"
        
        return {
            "messages": [AIMessage(content=msg)],
            # Set awaiting state for next turn
            "lyrics_awaiting_response": True,
            "lyrics_song_in_catalog": False,
            "lyrics_identified_song": song_title,
            "lyrics_identified_artist": artist,
            # No pending track since not in catalog
            "pending_track_id": None,
            "pending_track_name": None,
            "pending_track_price": None,
        }


# =============================================================================
# RESPONSE HANDLING NODE
# =============================================================================

def handle_response_node(state: LyricsState) -> dict:
    """Handle user's yes/no response after song identification.
    
    Sets lyrics_purchase_confirmed=True if user wants to buy,
    which the parent graph uses to route to purchase subgraph.
    """
    user_message = _get_last_user_message(state)
    song_in_catalog = state.get("lyrics_song_in_catalog", False)
    song_name = state.get("lyrics_identified_song", "the song")
    artist = state.get("lyrics_identified_artist", "")
    
    # Classify the user's response
    response = _classify_response(user_message)
    
    # Clear the awaiting state regardless of outcome
    clear_state = {
        "lyrics_awaiting_response": False,
        "lyrics_song_in_catalog": None,
        "lyrics_identified_song": None,
        "lyrics_identified_artist": None,
    }
    
    if response == "yes":
        if song_in_catalog:
            # User wants to buy - set flag for parent graph to route to purchase
            # pending_track_id, name, price were already set by present_options
            return {
                **clear_state,
                "lyrics_purchase_confirmed": True,  # Parent graph will route to purchase
                "messages": [AIMessage(content=f"Great choice! Let me set up your purchase of **{song_name}**.")],
            }
        else:
            # User wants song added - log feedback and thank them
            return {
                **clear_state,
                "lyrics_purchase_confirmed": False,
                "pending_track_id": None,
                "pending_track_name": None,
                "pending_track_price": None,
                "messages": [AIMessage(content=f"Thanks for your feedback! 🎵 We've noted your interest in **{song_name}** by **{artist}**. We'll consider adding it to our catalogue.\n\nIs there anything else I can help you with?")],
            }
    
    elif response == "no":
        # User declined
        return {
            **clear_state,
            "lyrics_purchase_confirmed": False,
            "pending_track_id": None,
            "pending_track_name": None,
            "pending_track_price": None,
            "messages": [AIMessage(content="No problem! Let me know if there's anything else I can help you with.")],
        }
    
    else:
        # Unclear - ask for clarification
        if song_in_catalog:
            clarify_msg = f"I wasn't sure if you wanted to purchase **{song_name}**. Would you like to buy it? Just say yes or no."
        else:
            clarify_msg = f"I wasn't sure about your response. Would you like us to consider adding **{song_name}** to our catalogue? Just say yes or no."
        
        # Keep awaiting_response True so we'll come back here
        return {
            "lyrics_purchase_confirmed": False,
            "messages": [AIMessage(content=clarify_msg)],
        }


# =============================================================================
# BUILD THE SUBGRAPH
# =============================================================================

def build_lyrics_subgraph() -> StateGraph:
    """Build the lyrics subgraph with router for two modes.
    
    Graph structure:
    
        START → router ─┬─→ identify_song → check_catalog → get_youtube → present_options → END
                        │
                        └─→ handle_response ─┬─→ END
                                             │
                                             └─→ purchase (via Command to parent)
    
    The router checks lyrics_awaiting_response to decide which flow to use.
    """
    builder = StateGraph(LyricsState)
    
    # Add nodes
    builder.add_node("router", router_node)
    builder.add_node("identify_song", identify_song_node)
    builder.add_node("check_catalog", check_catalog_node)
    builder.add_node("get_youtube", get_youtube_node)
    builder.add_node("present_options", present_options_node)
    builder.add_node("handle_response", handle_response_node)
    
    # Add edges
    builder.add_edge(START, "router")
    # router uses Command to go to identify_song or handle_response
    # identify_song uses Command to route to check_catalog or END
    # check_catalog and get_youtube always proceed to next step
    builder.add_edge("present_options", END)
    builder.add_edge("handle_response", END)
    
    return builder


# Compiled subgraph for import
lyrics_subgraph = build_lyrics_subgraph().compile()
