"""Lyrics QA node for song identification.

AGENTIC DESIGN with proper LangGraph tool patterns!

This node uses an LLM with bound tools to:
1. Identify songs from lyrics (Genius API)
2. Check if the song is in our catalog
3. Get YouTube video links
4. Build helpful responses

The LLM decides which tools to call based on the user's query,
making this a true agentic node that demonstrates LangGraph best practices.

Design Justification:
- Shows proper LangGraph tool binding and ToolNode patterns
- LLM orchestrates the flow based on user input
- Tools are visible in output for observability
- Consistent with catalog_qa and account_qa patterns
"""

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from src.state import SupportState
from src.tools.mocks import genius_search, youtube_lookup, check_song_in_catalog


LYRICS_SYSTEM_PROMPT = """You are a helpful music assistant specializing in identifying songs from lyrics.

You have access to these tools:
1. **genius_search**: Identify a song from lyrics the user provides
2. **check_song_in_catalog**: Check if an identified song is available in our store
3. **youtube_lookup**: Get a YouTube video link for a song

## Your Workflow:

When a user provides lyrics or asks about a song:
1. First, use genius_search to identify the song
2. If identified, use check_song_in_catalog to see if it's available for purchase
3. Always use youtube_lookup to provide a video link (users love this!)
4. Give a comprehensive response with all the information

## Response Guidelines:

- If the song IS in our catalog: Show the track details and ask if they'd like to purchase
  - Include [PURCHASE_READY: TrackId=X, Name=Y, Price=Z] at the end so the system can set up the purchase
- If the song is NOT in our catalog: Show the YouTube link and ask if they'd like us to note their interest
- Always be helpful and conversational

## Example Flow:
User: "What song goes like back in black I hit the sack"
1. Call genius_search("back in black I hit the sack")
2. Get result: "Back in Black" by AC/DC
3. Call check_song_in_catalog("Back in Black", "AC/DC")
4. Call youtube_lookup("Back in Black", "AC/DC")
5. Respond with all the info!

Be enthusiastic about helping users discover and enjoy music!"""


# Tools available to the lyrics QA node
LYRICS_TOOLS = [
    genius_search,
    check_song_in_catalog,
    youtube_lookup,
]


def lyrics_qa_node(state: SupportState) -> dict:
    """Handle lyrics-based song identification using LLM + tools.
    
    The LLM decides which tools to call based on the user's query.
    This demonstrates proper LangGraph agentic patterns.
    """
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    model_with_tools = model.bind_tools(LYRICS_TOOLS)
    
    messages = [SystemMessage(content=LYRICS_SYSTEM_PROMPT)] + state["messages"]
    
    response = model_with_tools.invoke(messages)
    
    # Check if the model wants to call tools
    if response.tool_calls:
        return {"messages": [response]}
    
    # Check for purchase intent in the response
    content = response.content
    result = {"messages": [response]}
    
    # Parse purchase ready tag if present
    if "[PURCHASE_READY:" in content:
        import re
        match = re.search(
            r'\[PURCHASE_READY:\s*TrackId=(\d+),\s*Name=([^,]+),\s*Price=([^\]]+)\]',
            content
        )
        if match:
            result["pending_track_id"] = int(match.group(1))
            result["pending_track_name"] = match.group(2).strip()
            try:
                result["pending_track_price"] = float(match.group(3).strip().replace("$", ""))
            except ValueError:
                result["pending_track_price"] = 0.99
    
    return result

