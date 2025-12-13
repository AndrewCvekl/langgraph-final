"""Tool definitions for the support bot."""

from src.tools.catalog import (
    albums_by_artist,
    artists_in_genre,
    find_track,
    list_genres,
    tracks_in_album,
)
from src.tools.account import (
    get_my_invoices,
    get_my_invoice_lines,
    get_my_profile,
    update_my_email,
)
from src.tools.purchase import create_invoice_for_track
from src.tools.mocks import (
    genius_search,
    twilio_send_code,
    youtube_lookup,
)

# Catalog tools (read-only, public)
CATALOG_TOOLS = [
    list_genres,
    artists_in_genre,
    albums_by_artist,
    tracks_in_album,
    find_track,
]

# Account tools (customer-scoped)
ACCOUNT_TOOLS = [
    get_my_profile,
    get_my_invoices,
    get_my_invoice_lines,
    update_my_email,
]

# Purchase tools (requires HITL)
PURCHASE_TOOLS = [
    create_invoice_for_track,
]

# External API tools (real APIs with mock fallback)
API_TOOLS = [
    genius_search,
    youtube_lookup,
    twilio_send_code,
]

# All tools combined
ALL_TOOLS = CATALOG_TOOLS + ACCOUNT_TOOLS + PURCHASE_TOOLS + API_TOOLS

__all__ = [
    # Catalog
    "list_genres",
    "artists_in_genre", 
    "albums_by_artist",
    "tracks_in_album",
    "find_track",
    # Account
    "get_my_profile",
    "get_my_invoices",
    "get_my_invoice_lines",
    "update_my_email",
    # Purchase
    "create_invoice_for_track",
    # External APIs
    "genius_search",
    "youtube_lookup",
    "twilio_send_code",
    # Tool groups
    "CATALOG_TOOLS",
    "ACCOUNT_TOOLS",
    "PURCHASE_TOOLS",
    "API_TOOLS",
    "ALL_TOOLS",
]

