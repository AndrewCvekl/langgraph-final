"""Lyrics QA node for song identification.

AGENTIC DESIGN with proper LangGraph tool patterns!

This node uses an LLM with bound tools to:
1. Identify songs from lyrics (Genius API)
2. Check if the song is in our catalog
3. Get YouTube video links
4. Build helpful responses

Follows LangGraph "Thinking in LangGraph" design principles:
- Returns Command with explicit goto destination
- Type hints declare all possible destinations
- Routes to its own tool node (lyrics_tools)
"""

from typing import Literal
import re

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from src.state import SupportState
from src.tools.mocks import genius_search, youtube_lookup, check_song_in_catalog
from src.tools.account import check_if_already_purchased


LYRICS_SYSTEM_PROMPT = """You are a helpful music assistant specializing in identifying songs from lyrics.

You have access to these tools:
1. **genius_search**: Identify a song from lyrics the user provides
2. **check_song_in_catalog**: Check if an identified song is available in our store
3. **check_if_already_purchased**: Check if the customer already owns a track (use after finding a track in catalog)
4. **youtube_lookup**: Get a YouTube video link for a song

## Your Workflow:

When a user provides lyrics or asks about a song:
1. First, use genius_search to identify the song
2. If identified, use check_song_in_catalog to see if it's available for purchase
3. If in catalog, use check_if_already_purchased with the TrackId to see if they already own it
4. Always use youtube_lookup to provide a video link (users love this!)
5. Give a comprehensive response with all the information

## Response Guidelines:

- If they ALREADY OWN the track: Let them know it's in their library! No purchase prompt needed.
- If the song IS in our catalog (and they don't own it): Show the track details and ask if they'd like to purchase
  - Include [PURCHASE_READY: TrackId=X, Name=Y, Price=Z] at the end so the system can set up the purchase
- If the song is NOT in our catalog: Show the YouTube link and ask if they'd like us to note their interest
- When including YouTube: output the **raw YouTube URL on its own line** (e.g. https://www.youtube.com/watch?v=VIDEO_ID). Do NOT use markdown link syntax like [Watch on YouTube](...).
- Always be helpful and conversational

## Example Flow:
User: "What song goes like back in black I hit the sack"
1. Call genius_search("back in black I hit the sack")
2. Get result: "Back in Black" by AC/DC
3. Call check_song_in_catalog("Back in Black", "AC/DC")
4. If found (e.g. TrackId=5), call check_if_already_purchased(track_id=5)
5. Call youtube_lookup("Back in Black", "AC/DC")
6. Respond based on whether they own it or not!

Be enthusiastic about helping users discover and enjoy music!"""


# Tools available to the lyrics QA node
LYRICS_TOOLS = [
    genius_search,
    check_song_in_catalog,
    check_if_already_purchased,
    youtube_lookup,
]


def lyrics_qa_node(
    state: SupportState
) -> Command[Literal["lyrics_tools", "router", "__end__"]]:
    """Handle lyrics-based song identification using LLM + tools.
    
    The LLM decides which tools to call based on the user's query.
    
    Returns Command with explicit routing:
    - lyrics_tools: When LLM wants to call tools
    - router: When purchase intent detected (router sends to purchase_flow)
    - __end__: When response is complete
    """
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    model_with_tools = model.bind_tools(LYRICS_TOOLS)
    
    messages = [SystemMessage(content=LYRICS_SYSTEM_PROMPT)] + state["messages"]
    
    response = model_with_tools.invoke(messages)
    
    # If the model wants to call tools, route to our dedicated tool node
    if response.tool_calls:
        return Command(
            update={"messages": [response]},
            goto="lyrics_tools"
        )
    
    # Check for purchase intent in the response
    content = response.content
    
    # Parse purchase ready tag if present
    if "[PURCHASE_READY:" in content:
        match = re.search(
            r'\[PURCHASE_READY:\s*TrackId=(\d+),\s*Name=([^,]+),\s*Price=([^\]]+)\]',
            content
        )
        if match:
            try:
                price = float(match.group(3).strip().replace("$", ""))
            except ValueError:
                price = 0.99
            
            # Route to router which will send to purchase_flow
            return Command(
                update={
                    "messages": [response],
                    "pending_track_id": int(match.group(1)),
                    "pending_track_name": match.group(2).strip(),
                    "pending_track_price": price,
                    "route": "purchase_flow",
                },
                goto="router"
            )
    
    # Normal response - end this turn
    return Command(
        update={"messages": [response]},
        goto="__end__"
    )

