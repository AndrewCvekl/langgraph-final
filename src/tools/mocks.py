"""External API tools for Genius, YouTube, and Twilio.

These tools use real APIs when credentials are configured (via .env):
- GENIUS_ACCESS_TOKEN: Genius API for song identification from lyrics
- YOUTUBE_API_KEY: YouTube Data API for music video lookup
- TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_VERIFY_SERVICE_SID: SMS verification

Falls back to mock mode automatically if credentials are not set.
"""

import random
import logging

from langchain_core.tools import tool

from src.tools.services import get_genius_service, get_youtube_service, get_twilio_service

logger = logging.getLogger(__name__)


# Mock database for fallback lyrics matching
MOCK_LYRICS_DB = [
    {"title": "Love Me Do", "artist": "The Beatles", "lyrics_snippet": "love me do", "genius_id": "mock_1"},
    {"title": "Hey Jude", "artist": "The Beatles", "lyrics_snippet": "hey jude", "genius_id": "mock_2"},
    {"title": "Yesterday", "artist": "The Beatles", "lyrics_snippet": "yesterday", "genius_id": "mock_3"},
    {"title": "Bohemian Rhapsody", "artist": "Queen", "lyrics_snippet": "bohemian rhapsody is this the real life", "genius_id": "mock_4"},
    {"title": "We Will Rock You", "artist": "Queen", "lyrics_snippet": "we will rock you", "genius_id": "mock_5"},
    {"title": "We Are the Champions", "artist": "Queen", "lyrics_snippet": "we are the champions", "genius_id": "mock_6"},
    {"title": "Stairway to Heaven", "artist": "Led Zeppelin", "lyrics_snippet": "stairway to heaven", "genius_id": "mock_7"},
    {"title": "Whole Lotta Love", "artist": "Led Zeppelin", "lyrics_snippet": "whole lotta love", "genius_id": "mock_8"},
    {"title": "Smoke On The Water", "artist": "Deep Purple", "lyrics_snippet": "smoke on the water", "genius_id": "mock_9"},
    {"title": "Highway to Hell", "artist": "AC/DC", "lyrics_snippet": "highway to hell", "genius_id": "mock_10"},
    {"title": "Back in Black", "artist": "AC/DC", "lyrics_snippet": "back in black i hit the sack", "genius_id": "mock_11"},
    {"title": "Sweet Child O' Mine", "artist": "Guns N' Roses", "lyrics_snippet": "sweet child o mine", "genius_id": "mock_12"},
    {"title": "November Rain", "artist": "Guns N' Roses", "lyrics_snippet": "november rain", "genius_id": "mock_13"},
    {"title": "Smells Like Teen Spirit", "artist": "Nirvana", "lyrics_snippet": "smells like teen spirit", "genius_id": "mock_14"},
    {"title": "Come As You Are", "artist": "Nirvana", "lyrics_snippet": "come as you are", "genius_id": "mock_15"},
    {"title": "Enter Sandman", "artist": "Metallica", "lyrics_snippet": "enter sandman", "genius_id": "mock_16"},
    {"title": "Nothing Else Matters", "artist": "Metallica", "lyrics_snippet": "nothing else matters", "genius_id": "mock_17"},
    {"title": "One", "artist": "Metallica", "lyrics_snippet": "one", "genius_id": "mock_18"},
    {"title": "Thunderstruck", "artist": "AC/DC", "lyrics_snippet": "thunderstruck", "genius_id": "mock_19"},
    {"title": "Black Dog", "artist": "Led Zeppelin", "lyrics_snippet": "black dog", "genius_id": "mock_20"},
    {"title": "Let There Be Rock", "artist": "AC/DC", "lyrics_snippet": "woowoo let there be rock", "genius_id": "mock_21"},
]


@tool
def genius_search(lyrics_snippet: str) -> str:
    """Search for a song using lyrics (Genius API).
    
    Uses the real Genius API when GENIUS_ACCESS_TOKEN is configured.
    Falls back to mock mode for demo/testing purposes.
    
    Args:
        lyrics_snippet: A snippet of lyrics to search for.
        
    Returns:
        The identified song title and artist, or a "not found" message.
    """
    if not lyrics_snippet or len(lyrics_snippet.strip()) < 4:
        return """
Could not identify the song from those lyrics.
Try providing more distinctive lyrics or the song title directly.
"""
    
    # Get the Genius service (uses real API if configured)
    genius = get_genius_service()
    mode = "LIVE API" if genius.is_live else "MOCK"
    logger.info(f"[genius_search] Mode: {mode}, Query: '{lyrics_snippet[:40]}...'")
    
    # If not using live API, inject mock songs for fallback
    if not genius.is_live:
        genius.songs = MOCK_LYRICS_DB
    
    results = genius.search_by_lyrics(lyrics_snippet)
    
    if not results:
        logger.info(f"[genius_search] No results found")
        return """
Could not identify the song from those lyrics.
Try providing more distinctive lyrics or the song title directly.
"""
    
    # Return the best match
    best = results[0]
    source = "Genius" if genius.is_live else "Genius (mock)"
    logger.info(f"[genius_search] Found: '{best['title']}' by {best['artist']} (score: {best['score']})")
    
    return f"""
Song identified!
- Title: {best['title']}
- Artist: {best['artist']}
- Confidence: {int(best['score'] * 100)}%
- Source: {source}
"""


@tool
def youtube_lookup(song_title: str, artist: str) -> str:
    """Look up a song on YouTube.
    
    Uses the real YouTube Data API when YOUTUBE_API_KEY is configured.
    Falls back to mock mode for demo/testing purposes.
    
    Args:
        song_title: The title of the song.
        artist: The artist name.
        
    Returns:
        YouTube video info with a YouTube URL that the frontend can auto-embed.
    """
    youtube = get_youtube_service()
    mode = "LIVE API" if youtube.is_live else "MOCK"
    
    # Build search query
    query = f"{song_title} {artist} official audio"
    logger.info(f"[youtube_lookup] Mode: {mode}, Query: '{query}'")
    
    result = youtube.search_video(query)
    video_id = result['video_id']
    logger.info(f"[youtube_lookup] Found: '{result['title']}' ({video_id}) on {result['channel']}")
    
    source = "" if youtube.is_live else "\n\n(Demo mode - using sample video)"
    
    # Return only the raw URL (no label, no markdown) so the frontend can
    # reliably render a thumbnail card in the chat bubble.
    return f"https://www.youtube.com/watch?v={video_id}{source}"


@tool
def twilio_send_code(phone_number: str) -> str:
    """Send a verification code via SMS (Twilio).
    
    Uses the real Twilio Verify API when credentials are configured.
    Falls back to mock mode for demo/testing purposes.
    
    Args:
        phone_number: The phone number to send the code to.
        
    Returns:
        Confirmation that the code was sent, and verification ID.
    """
    twilio = get_twilio_service()
    mode = "LIVE API" if twilio.is_live else "MOCK"
    logger.info(f"[twilio_send_code] Mode: {mode}, Phone: {phone_number[-4:]}...")
    
    # Send the verification code
    verification_id = twilio.send_code(phone_number)
    logger.info(f"[twilio_send_code] Verification ID: {verification_id[:20]}...")
    
    # Mask the phone number for display
    if len(phone_number) >= 4:
        masked = "*" * (len(phone_number) - 4) + phone_number[-4:]
    else:
        masked = phone_number
    
    if twilio.is_live:
        return f"""
Verification code sent!
- Phone: {masked}
- Verification ID: {verification_id}
- Expires: 10 minutes

The code has been sent via SMS to your phone.
"""
    else:
        # In mock mode, show the code for testing
        code = twilio.get_pending_code(verification_id)
        return f"""
Verification code sent!
- Phone: {masked}
- Code: {code}
- Verification ID: {verification_id}
- Expires: 10 minutes

(Demo mode: Code is shown for testing. In production, this would be sent via SMS.)
"""


@tool
def check_song_in_catalog(song_title: str, artist: str) -> str:
    """Check if a song is available in our music store catalog.
    
    Use this after identifying a song to see if it's available for purchase.
    
    Args:
        song_title: The title of the song to check.
        artist: The artist name.
        
    Returns:
        Catalog status including TrackId and price if available.
    """
    from src.db import get_db
    import re
    
    db = get_db()
    
    # Search for the track
    result = db.run(
        f"""
        SELECT 
            Track.TrackId,
            Track.Name as TrackName,
            Artist.Name as ArtistName,
            Track.UnitPrice
        FROM Track
        JOIN Album ON Track.AlbumId = Album.AlbumId
        JOIN Artist ON Album.ArtistId = Artist.ArtistId
        WHERE Track.Name LIKE '%{song_title}%'
        AND Artist.Name LIKE '%{artist}%'
        LIMIT 1;
        """,
        include_columns=True
    )
    
    if result and "TrackId" in result:
        # Parse the result
        track_id_match = re.search(r"TrackId['\"]?:\s*(\d+)", result)
        price_match = re.search(r"UnitPrice['\"]?:\s*([\d.]+)", result)
        name_match = re.search(r"TrackName['\"]?:\s*['\"]?([^'\"]+)['\"]?", result)
        
        if track_id_match:
            track_id = track_id_match.group(1)
            price = price_match.group(1) if price_match else "0.99"
            name = name_match.group(1).strip() if name_match else song_title
            
            return f"""
Found in catalog!
- TrackId: {track_id}
- Track Name: {name}
- Artist: {artist}
- Price: ${price}

IMPORTANT: Call check_if_already_purchased(track_id={track_id}) to see if they already own this track before offering to sell it.
If they don't own it, include [PURCHASE_READY: TrackId={track_id}, Name={name}, Price={price}] in your response.
"""
    
    return f"""
Not in catalog.
'{song_title}' by {artist} is not currently available in our store.
Ask the customer if they'd like us to note their interest in this track.
"""


def generate_verification_code() -> str:
    """Generate a random 6-digit verification code."""
    return str(random.randint(100000, 999999))


def mask_phone_number(phone: str) -> str:
    """Mask a phone number, showing only the last 4 digits."""
    if not phone:
        return "No phone on file"
    if len(phone) >= 4:
        return "*" * (len(phone) - 4) + phone[-4:]
    return phone
