"""Catalog QA node for music browsing.

Handles questions about genres, artists, albums, and tracks.
Can detect purchase intent and extract TrackId for handoff.
"""

import json
import re

from langchain_core.messages import SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.state import SupportState
from src.tools.catalog import (
    list_genres,
    artists_in_genre,
    albums_by_artist,
    tracks_in_album,
    find_track,
)


class PurchaseHandoff(BaseModel):
    """Structured purchase handoff payload (control signal)."""

    track_id: int = Field(..., description="TrackId to purchase")
    name: str = Field(..., description="Track name for display")
    price: float = Field(..., description="Unit price in dollars")


class CatalogResponse(BaseModel):
    """Structured response for the catalog lane.

    This separates user-facing text from control signals, avoiding brittle
    parsing of tags inside assistant content.
    """

    message: str = Field(..., description="User-facing assistant response")
    purchase: PurchaseHandoff | None = Field(
        default=None,
        description="If set, triggers purchase_flow by populating pending_track_*.",
    )


def _strip_code_fences(text: str) -> str:
    """Strip common ```json fences if the model includes them."""
    text = (text or "").strip()
    if text.startswith("```"):
        # Remove leading ```lang and trailing ```
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_structured_catalog_response(raw: str) -> CatalogResponse | None:
    """Parse a CatalogResponse from model output, if possible."""
    try:
        cleaned = _strip_code_fences(raw)
        data = json.loads(cleaned)
        return CatalogResponse.model_validate(data)
    except Exception:
        return None


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
4. **Only after you have confirmed the track exists**, prepare a purchase handoff in the JSON output (set "purchase" with track_id/name/price)

Examples:
- "I want to buy Kashmir" → First use find_track("Kashmir"), then set "purchase" in the JSON output
- "yes" (after showing a track) → Set "purchase" in the JSON output using the previously shown track
- "buy track 2969" → Use find_track to verify, then set "purchase" in the JSON output

Be helpful and conversational. If you can't find what they're looking for, suggest alternatives.

CRITICAL: When a customer wants to purchase a track by name (like "I want to buy Kashmir"):
1. You MUST use find_track tool to search for it first
2. Show them the track details
3. If they want to buy it, set "purchase" in the JSON output to trigger the purchase flow

Never set "purchase" without first verifying the track exists in our catalog!

IMPORTANT: If the conversation history indicates the customer ALREADY OWNS a track, do NOT set "purchase". Just let them know they already own it and it's in their library."""


# NOTE: We keep the legacy tag instructions above for backwards compatibility,
# but we now prefer structured control signals. If you are ready to respond
# to the user (i.e., you do NOT need to call any tools), output a JSON object
# matching this schema and NOTHING else:
#
# {
#   "message": "<user-facing text>",
#   "purchase": null | {"track_id": <int>, "name": "<str>", "price": <float>}
# }
#
# - Set "purchase" ONLY when the user wants to buy a specific track AND you have
#   verified the track exists (via tools or a recent tool result).
CATALOG_SYSTEM_PROMPT += """

## Output Format (Preferred)
When you are ready to respond to the user (no tool calls), output ONLY valid JSON:
{
  "message": "string",
  "purchase": null or { "track_id": 123, "name": "Track Name", "price": 0.99 }
}
Do not wrap in markdown fences. Do not include any extra keys.
"""


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
    
    # Preferred: structured JSON control signal (message + optional purchase handoff)
    structured = _parse_structured_catalog_response(response.content)
    if structured is not None:
        result: dict = {"messages": [AIMessage(content=structured.message)]}
        if structured.purchase is not None:
            result["pending_track_id"] = structured.purchase.track_id
            result["pending_track_name"] = structured.purchase.name.strip()
            result["pending_track_price"] = float(structured.purchase.price)
            result["route"] = "purchase_flow"
        return result

    # Fallback: legacy tag parsing (kept to avoid breaking existing behavior)
    content = response.content
    result = {"messages": [response]}
    if "[PURCHASE_INTENT:" in content:
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

    return result

