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

import json
import re

from langchain_core.messages import SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.state import SupportState
from src.tools.mocks import genius_search, youtube_lookup, check_song_in_catalog
from src.tools.account import check_if_already_purchased


class PurchaseHandoff(BaseModel):
    """Structured purchase handoff payload (control signal)."""

    track_id: int = Field(..., description="TrackId to purchase")
    name: str = Field(..., description="Track name for display")
    price: float = Field(..., description="Unit price in dollars")


class LyricsResponse(BaseModel):
    """Structured response for the lyrics lane."""

    message: str = Field(..., description="User-facing assistant response")
    purchase: PurchaseHandoff | None = Field(
        default=None,
        description="If set, populates pending_track_* for follow-on purchase confirmation.",
    )


def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_structured_lyrics_response(raw: str) -> LyricsResponse | None:
    try:
        cleaned = _strip_code_fences(raw)
        data = json.loads(cleaned)
        return LyricsResponse.model_validate(data)
    except Exception:
        return None


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
  - If they want to buy it, include a purchase handoff in the JSON output ("purchase" with track_id/name/price)
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


# NOTE: We keep the legacy [PURCHASE_READY: ...] convention for backwards
# compatibility, but we now prefer structured control signals. When you are
# ready to respond to the user (no tool calls), output ONLY valid JSON:
#
# {
#   "message": "<user-facing text (may include raw YouTube URL)>",
#   "purchase": null | {"track_id": <int>, "name": "<str>", "price": <float>}
# }
LYRICS_SYSTEM_PROMPT += """

## Output Format (Preferred)
When you are ready to respond to the user (no tool calls), output ONLY valid JSON:
{
  "message": "string",
  "purchase": null or { "track_id": 123, "name": "Track Name", "price": 0.99 }
}
Do not wrap in markdown fences. Do not include any extra keys.
"""


# Tools available to the lyrics QA node
LYRICS_TOOLS = [
    genius_search,
    check_song_in_catalog,
    check_if_already_purchased,
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
    
    # Preferred: structured JSON control signal
    structured = _parse_structured_lyrics_response(response.content)
    if structured is not None:
        result: dict = {"messages": [AIMessage(content=structured.message)]}
        if structured.purchase is not None:
            result["pending_track_id"] = structured.purchase.track_id
            result["pending_track_name"] = structured.purchase.name.strip()
            result["pending_track_price"] = float(structured.purchase.price)
        return result

    # Fallback: legacy tag parsing
    content = response.content
    result = {"messages": [response]}
    if "[PURCHASE_READY:" in content:
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

