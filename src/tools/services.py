"""External API services for Genius, Twilio, and YouTube.

Each service uses the real API when credentials are configured,
and falls back to mock mode for testing/demo purposes.
"""

import os
import re
import uuid
import random
import hashlib
import logging
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Optional

logger = logging.getLogger(__name__)


def _similarity(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# =============================================================================
# Genius API Service
# =============================================================================

class GeniusService:
    """Genius API service for lyrics search.
    
    Uses the Genius API to search for songs by lyrics.
    Falls back to mock mode with fuzzy matching if API token is not configured.
    """
    
    def __init__(self, access_token: Optional[str] = None, songs: Optional[list[dict]] = None):
        """Initialize the Genius service.
        
        Args:
            access_token: Genius API access token (or set GENIUS_ACCESS_TOKEN env var).
            songs: Optional list of song dictionaries for mock mode.
        """
        self.access_token = access_token or os.getenv("GENIUS_ACCESS_TOKEN")
        self.songs = songs or []
        
        if self.access_token:
            logger.info("[Genius] Initialized with real API")
        else:
            logger.info("[Genius] No API token configured, using mock mode")
    
    @property
    def is_live(self) -> bool:
        """Check if using real API or mock mode."""
        return self.access_token is not None
    
    def search_by_lyrics(self, lyrics: str) -> list[dict]:
        """Search for songs by lyrics snippet.
        
        Args:
            lyrics: Lyrics snippet to search for.
            
        Returns:
            List of matching songs sorted by score (best first).
            Each dict has: title, artist, score, genius_id
        """
        if not lyrics or not lyrics.strip():
            return []
        
        if self.is_live:
            return self._search_real(lyrics)
        else:
            return self._search_mock(lyrics)
    
    def _search_real(self, lyrics: str) -> list[dict]:
        """Search using the real Genius API."""
        try:
            import requests
            
            url = "https://api.genius.com/search"
            params = {
                "access_token": self.access_token,
                "q": lyrics
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            hits = data.get('response', {}).get('hits', [])
            
            if not hits:
                logger.info(f"[Genius] No results for: {lyrics[:30]}...")
                return []
            
            results = []
            for i, hit in enumerate(hits[:5]):  # Top 5 results
                if hit.get('type') == 'song':
                    result = hit.get('result', {})
                    artist_info = result.get('primary_artist', {})
                    
                    title = result.get('title', 'Unknown').strip()
                    artist = artist_info.get('name', 'Unknown').strip()
                    genius_id = str(result.get('id', f'genius_{i}'))
                    
                    # Score based on position (first result is best)
                    score = round(1.0 - (i * 0.15), 3)
                    
                    results.append({
                        "title": title,
                        "artist": artist,
                        "score": score,
                        "genius_id": genius_id,
                    })
            
            logger.info(f"[Genius] Search for '{lyrics[:30]}...' found {len(results)} matches")
            return results
            
        except ImportError:
            logger.warning("[Genius] requests package not installed, falling back to mock")
            return self._search_mock(lyrics)
        except Exception as e:
            logger.error(f"[Genius] API error: {e}")
            return self._search_mock(lyrics)
    
    def _search_mock(self, lyrics: str) -> list[dict]:
        """Search using fuzzy matching against sample database."""
        if not self.songs:
            return []
            
        results = []
        lyrics_lower = lyrics.lower()
        
        for song in self.songs:
            snippet = song.get("lyrics_snippet", "").lower()
            
            # Calculate similarity score
            score = _similarity(lyrics_lower, snippet)
            
            # Boost if lyrics is a substring
            if lyrics_lower in snippet:
                score = max(score, 0.8)
            
            # Only include if there's some match
            if score > 0.2:
                results.append({
                    "title": song["title"],
                    "artist": song["artist"],
                    "score": round(score, 3),
                    "genius_id": song["genius_id"],
                })
        
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        
        logger.info(f"[Genius/Mock] Search for '{lyrics[:30]}...' found {len(results)} matches")
        return results


# =============================================================================
# Twilio Verification Service
# =============================================================================

class TwilioService:
    """Twilio verification service.
    
    Uses the Twilio Verify API for SMS verification.
    Falls back to mock mode if credentials are not configured.
    """
    
    def __init__(
        self,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
        verify_service_sid: Optional[str] = None,
        use_random_codes: bool = False
    ):
        """Initialize the Twilio service.
        
        Args:
            account_sid: Twilio Account SID (or set TWILIO_ACCOUNT_SID env var).
            auth_token: Twilio Auth Token (or set TWILIO_AUTH_TOKEN env var).
            verify_service_sid: Twilio Verify Service SID (or set TWILIO_VERIFY_SERVICE_SID env var).
            use_random_codes: If True and in mock mode, generate random codes.
                            If False, always use "123456" for easy testing.
        """
        self.use_random_codes = use_random_codes
        
        # Twilio credentials
        self.account_sid = account_sid or os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = auth_token or os.getenv('TWILIO_AUTH_TOKEN')
        self.verify_service_sid = verify_service_sid or os.getenv('TWILIO_VERIFY_SERVICE_SID')
        
        # Check if Twilio is fully configured
        self._client = None
        self.twilio_enabled = all([self.account_sid, self.auth_token, self.verify_service_sid])
        
        if self.twilio_enabled:
            try:
                from twilio.rest import Client
                self._client = Client(self.account_sid, self.auth_token)
                logger.info("[Twilio] Initialized with real API")
            except ImportError:
                logger.warning("[Twilio] twilio package not installed, using mock mode")
                self.twilio_enabled = False
            except Exception as e:
                logger.warning(f"[Twilio] Failed to initialize client: {e}, using mock mode")
                self.twilio_enabled = False
        else:
            logger.info("[Twilio] Credentials not configured, using mock mode")
        
        # Store for mock verifications and phone tracking
        self._verifications: dict[str, dict] = {}
    
    @property
    def is_live(self) -> bool:
        """Check if using real API or mock mode."""
        return self.twilio_enabled and self._client is not None
    
    def _format_phone_number(self, phone: str) -> str:
        """Format phone number to E.164 format required by Twilio."""
        digits = re.sub(r'\D', '', phone)
        
        if phone.startswith('+'):
            return '+' + digits
        if len(digits) == 10:
            return f'+1{digits}'
        if len(digits) == 11 and digits.startswith('1'):
            return f'+{digits}'
        if len(digits) > 10:
            return f'+{digits}'
        return f'+1{digits}'
    
    def _mask_phone(self, phone: str) -> str:
        """Mask phone number for display."""
        return phone[:-4] + '****' if len(phone) > 4 else '****'
    
    def send_code(self, phone: str) -> str:
        """Send a verification code to a phone number.
        
        Args:
            phone: Phone number to send code to.
            
        Returns:
            Verification ID to use when checking the code.
        """
        formatted_phone = self._format_phone_number(phone)
        
        if self.is_live:
            return self._send_code_real(formatted_phone)
        else:
            return self._send_code_mock(formatted_phone)
    
    def _send_code_real(self, phone: str) -> str:
        """Send verification code using Twilio Verify API."""
        try:
            verification = self._client.verify.v2.services(
                self.verify_service_sid
            ).verifications.create(
                to=phone,
                channel='sms'
            )
            
            # Store verification info (using SID as verification_id)
            verification_id = verification.sid
            self._verifications[verification_id] = {
                'phone': phone,
                'status': 'pending',
                'created_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(minutes=10),
            }
            
            masked = self._mask_phone(phone)
            logger.info(f"[Twilio] Sent verification to {masked}, status: {verification.status}")
            
            return verification_id
            
        except Exception as e:
            logger.error(f"[Twilio] Failed to send code: {e}")
            # Fall back to mock on error
            return self._send_code_mock(phone)
    
    def _send_code_mock(self, phone: str) -> str:
        """Send mock verification code."""
        verification_id = str(uuid.uuid4())
        
        if self.use_random_codes:
            code = str(random.randint(100000, 999999))
        else:
            code = "123456"
        
        self._verifications[verification_id] = {
            'phone': phone,
            'code': code,
            'status': 'pending',
            'created_at': datetime.now(),
        }
        
        masked = self._mask_phone(phone)
        logger.info(f"[Twilio/Mock] Sent code {code} to {masked}")
        
        return verification_id
    
    def check_code(self, verification_id: str, code: str) -> bool:
        """Check if a verification code is correct.
        
        Args:
            verification_id: ID returned from send_code.
            code: Code entered by the user.
            
        Returns:
            True if the code matches, False otherwise.
        """
        verification = self._verifications.get(verification_id)
        
        if not verification:
            logger.warning(f"[Twilio] Unknown verification ID: {verification_id}")
            return False
        
        if self.is_live and 'code' not in verification:
            # Real Twilio verification - check via API
            return self._check_code_real(verification['phone'], code, verification_id)
        else:
            # Mock verification - check stored code
            return self._check_code_mock(verification_id, code)
    
    def _check_code_real(self, phone: str, code: str, verification_id: str) -> bool:
        """Verify code using Twilio Verify API."""
        try:
            verification_check = self._client.verify.v2.services(
                self.verify_service_sid
            ).verification_checks.create(
                to=phone,
                code=code.strip()
            )
            
            is_valid = verification_check.status == 'approved'
            
            if is_valid:
                self._verifications[verification_id]['status'] = 'verified'
                logger.info("[Twilio] Code verified successfully")
            else:
                logger.info(f"[Twilio] Invalid code, status: {verification_check.status}")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"[Twilio] Verification check failed: {e}")
            return False
    
    def _check_code_mock(self, verification_id: str, code: str) -> bool:
        """Check mock verification code."""
        verification = self._verifications.get(verification_id)
        expected = verification.get('code') if verification else None
        
        if expected is None:
            return False
        
        is_valid = code.strip() == expected
        
        if is_valid:
            del self._verifications[verification_id]
            logger.info("[Twilio/Mock] Code verified successfully")
        else:
            logger.info(f"[Twilio/Mock] Invalid code: expected {expected}, got {code}")
        
        return is_valid
    
    def get_pending_code(self, verification_id: str) -> Optional[str]:
        """Get the pending code for a verification (mock mode only, for testing).
        
        Args:
            verification_id: Verification ID.
            
        Returns:
            The code (if mock mode), or None.
        """
        verification = self._verifications.get(verification_id)
        return verification.get('code') if verification else None


# =============================================================================
# YouTube API Service
# =============================================================================

class YouTubeService:
    """YouTube API service for video search.
    
    Uses the YouTube Data API v3 to search for videos.
    Falls back to deterministic mock responses if API key is not configured.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the YouTube service.
        
        Args:
            api_key: YouTube Data API v3 key (or set YOUTUBE_API_KEY env var).
        """
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY")
        self._client = None
        
        if self.api_key:
            try:
                from googleapiclient.discovery import build
                self._client = build('youtube', 'v3', developerKey=self.api_key)
                logger.info("[YouTube] Initialized with real API")
            except ImportError:
                logger.warning("[YouTube] google-api-python-client not installed, using mock mode")
            except Exception as e:
                logger.warning(f"[YouTube] Failed to initialize API client: {e}, using mock mode")
        else:
            logger.info("[YouTube] No API key configured, using mock mode")
    
    @property
    def is_live(self) -> bool:
        """Check if using real API or mock mode."""
        return self._client is not None
    
    def search_video(self, query: str) -> dict:
        """Search for a YouTube video.
        
        Args:
            query: Search query (e.g., "Song Title Artist official audio").
            
        Returns:
            Dict with video_id, title, url, and channel.
        """
        if not query or not query.strip():
            return self._empty_result()
        
        if self._client:
            return self._search_real(query)
        else:
            return self._search_mock(query)
    
    def _search_real(self, query: str) -> dict:
        """Search using the real YouTube API."""
        try:
            search_response = self._client.search().list(
                q=query,
                part='id,snippet',
                maxResults=1,
                type='video',
                videoEmbeddable='true'
            ).execute()
            
            items = search_response.get('items', [])
            if not items:
                logger.info(f"[YouTube] No videos found for: {query[:50]}")
                return self._empty_result()
            
            video = items[0]
            video_id = video['id']['videoId']
            title = video['snippet']['title']
            channel = video['snippet']['channelTitle']
            url = f"https://www.youtube.com/watch?v={video_id}"
            
            logger.info(f"[YouTube] Found: {title} ({video_id})")
            
            return {
                "video_id": video_id,
                "title": title,
                "url": url,
                "channel": channel,
            }
            
        except Exception as e:
            logger.error(f"[YouTube] API error: {e}")
            # Fall back to mock on error
            return self._search_mock(query)
    
    def _search_mock(self, query: str) -> dict:
        """Generate a mock response for testing."""
        video_id = self._generate_video_id(query)
        title = self._format_title(query)
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        logger.info(f"[YouTube/Mock] Generated: {title} ({video_id})")
        
        return {
            "video_id": video_id,
            "title": title,
            "url": url,
            "channel": "Mock Channel",
        }
    
    def _empty_result(self) -> dict:
        """Return an empty/default result."""
        return {
            "video_id": "dQw4w9WgXcQ",
            "title": "Unknown Video",
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "channel": "Unknown",
        }
    
    def _generate_video_id(self, query: str) -> str:
        """Generate a deterministic 11-character video ID from query."""
        hash_bytes = hashlib.md5(query.encode()).hexdigest()[:11]
        return hash_bytes.replace('a', 'A').replace('e', 'E')[:11]
    
    def _format_title(self, query: str) -> str:
        """Format the search query as a video title."""
        for suffix in [" official audio", " official video", " music video", " lyrics"]:
            query = query.lower().replace(suffix, "")
        return query.strip().title()
    
    def get_embed_html(self, video_id: str, autoplay: bool = True) -> str:
        """Generate an HTML embed iframe for a video.
        
        Args:
            video_id: YouTube video ID.
            autoplay: Whether to autoplay the video.
            
        Returns:
            HTML iframe string.
        """
        autoplay_param = "1" if autoplay else "0"
        return (
            f'<iframe width="560" height="315" '
            f'src="https://www.youtube.com/embed/{video_id}?autoplay={autoplay_param}" '
            f'frameborder="0" allow="accelerometer; autoplay; clipboard-write; '
            f'encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'
        )


# =============================================================================
# Singleton instances (initialized on first import)
# =============================================================================

# These are initialized lazily to allow .env loading first
_genius_service: Optional[GeniusService] = None
_twilio_service: Optional[TwilioService] = None
_youtube_service: Optional[YouTubeService] = None


def get_genius_service() -> GeniusService:
    """Get or create the Genius service singleton."""
    global _genius_service
    if _genius_service is None:
        _genius_service = GeniusService()
    return _genius_service


def get_twilio_service() -> TwilioService:
    """Get or create the Twilio service singleton."""
    global _twilio_service
    if _twilio_service is None:
        _twilio_service = TwilioService()
    return _twilio_service


def get_youtube_service() -> YouTubeService:
    """Get or create the YouTube service singleton."""
    global _youtube_service
    if _youtube_service is None:
        _youtube_service = YouTubeService()
    return _youtube_service

