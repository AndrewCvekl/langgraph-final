"""Mock external API tools.

These simulate external services for demo purposes:
- Genius: Song identification from lyrics
- YouTube: Music video lookup
- Twilio: SMS verification codes
"""

import random
import hashlib

from langchain_core.tools import tool


# Mock database of songs that can be identified from lyrics
MOCK_LYRICS_DB = {
    "love me do": ("Love Me Do", "The Beatles"),
    "hey jude": ("Hey Jude", "The Beatles"),
    "yesterday": ("Yesterday", "The Beatles"),
    "bohemian rhapsody": ("Bohemian Rhapsody", "Queen"),
    "we will rock you": ("We Will Rock You", "Queen"),
    "we are the champions": ("We Are the Champions", "Queen"),
    "stairway to heaven": ("Stairway to Heaven", "Led Zeppelin"),
    "whole lotta love": ("Whole Lotta Love", "Led Zeppelin"),
    "smoke on the water": ("Smoke On The Water", "Deep Purple"),
    "highway to hell": ("Highway to Hell", "AC/DC"),
    "back in black": ("Back in Black", "AC/DC"),
    "sweet child o mine": ("Sweet Child O' Mine", "Guns N' Roses"),
    "november rain": ("November Rain", "Guns N' Roses"),
    "smells like teen spirit": ("Smells Like Teen Spirit", "Nirvana"),
    "come as you are": ("Come As You Are", "Nirvana"),
    "enter sandman": ("Enter Sandman", "Metallica"),
    "nothing else matters": ("Nothing Else Matters", "Metallica"),
    "one": ("One", "Metallica"),
    "thunderstruck": ("Thunderstruck", "AC/DC"),
    "black dog": ("Black Dog", "Led Zeppelin"),
    "woowoo": ("Let There Be Rock", "AC/DC"),
}


@tool
def mock_genius_search(lyrics_snippet: str) -> str:
    """Search for a song using lyrics (Genius API mock).
    
    This simulates the Genius API for identifying songs from lyrics.
    In production, this would call the real Genius API.
    
    Args:
        lyrics_snippet: A snippet of lyrics to search for.
        
    Returns:
        The identified song title and artist, or a "not found" message.
    """
    # Normalize the query
    query = lyrics_snippet.lower().strip()
    
    # Require a minimum length to avoid matching simple words like "yes" to "yesterday"
    if len(query) < 4:
        return f"""
Could not identify the song from those lyrics.
Try providing more distinctive lyrics or the song title directly.
"""
    
    # Check for partial matches
    for key, (title, artist) in MOCK_LYRICS_DB.items():
        # Only match if the query contains the key OR the key contains the query
        # AND the match is significant (at least 50% of the shorter string)
        if key in query:
            return f"""
Song identified!
- Title: {title}
- Artist: {artist}
- Source: Genius (mock)
"""
        # For query in key, require query to be at least half the key length
        # This prevents "yes" from matching "yesterday" (3 chars vs 9 chars = 33%)
        if query in key and len(query) >= len(key) * 0.5:
            return f"""
Song identified!
- Title: {title}
- Artist: {artist}
- Source: Genius (mock)
"""
    
    # Generate a plausible "not found" response
    return f"""
Could not identify the song from those lyrics.
Try providing more distinctive lyrics or the song title directly.
"""


@tool
def mock_youtube_lookup(song_title: str, artist: str) -> str:
    """Look up a song on YouTube (mock).
    
    This simulates finding a YouTube music video.
    In production, this would call the YouTube Data API.
    
    Args:
        song_title: The title of the song.
        artist: The artist name.
        
    Returns:
        A mock YouTube link and video info.
    """
    # Generate a deterministic but fake video ID
    seed = f"{song_title.lower()}{artist.lower()}"
    video_id = hashlib.md5(seed.encode()).hexdigest()[:11]
    
    return f"""
Found on YouTube!
- Title: {song_title} - {artist} (Official Music Video)
- URL: https://www.youtube.com/watch?v={video_id}
- Duration: 4:32
- Views: 125M

Note: This is a mock link for demo purposes.
"""


@tool
def mock_twilio_send_code(phone_number: str) -> str:
    """Send a verification code via SMS (Twilio mock).
    
    This simulates sending an SMS verification code.
    In production, this would call the Twilio API.
    
    Args:
        phone_number: The phone number to send the code to.
        
    Returns:
        Confirmation that the code was sent.
    """
    # Generate a 6-digit code
    code = str(random.randint(100000, 999999))
    
    # In a real implementation, we'd actually send the SMS
    # For demo, we'll return the code (in production, never do this!)
    
    # Mask the phone number for display
    if len(phone_number) >= 4:
        masked = "*" * (len(phone_number) - 4) + phone_number[-4:]
    else:
        masked = phone_number
    
    return f"""
Verification code sent!
- Phone: {masked}
- Code: {code}
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

The customer can purchase this track. Include [PURCHASE_READY: TrackId={track_id}, Name={name}, Price={price}] in your response if they want to buy.
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

