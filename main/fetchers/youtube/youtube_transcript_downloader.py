"""
YouTube Transcript Downloader - Downloads transcripts using youtube-transcript-api.

This module provides functionality to:
1. Download transcripts for YouTube videos
2. Format transcripts with timestamps
3. Handle videos without available transcripts
4. Use browser cookies to avoid rate limiting
"""

import logging
from typing import Optional, Dict, List
from http.cookiejar import CookieJar
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

from .retry_utils import execute_with_exponential_backoff


def create_http_client_with_cookies() -> Optional[requests.Session]:
    """
    Create a requests Session with browser cookies for YouTube authentication.

    This function attempts to extract YouTube cookies from installed browsers
    and creates a requests Session configured with those cookies.

    Returns:
        requests.Session with browser cookies, or None if loading failed

    Example:
        >>> session = create_http_client_with_cookies()
        >>> if session:
        ...     print("Successfully created session with browser cookies")
    """
    try:
        import browser_cookie3

        cookies = None
        browser_name = None

        # Try Chrome first (most common)
        try:
            cookies = browser_cookie3.chrome(domain_name='youtube.com')
            cookie_count = len(list(cookies))
            if cookie_count > 0:
                browser_name = "Chrome"
                logging.info(f"Loaded {cookie_count} cookies from Chrome browser")
        except Exception as e:
            logging.debug(f"Failed to load Chrome cookies: {e}")

        # Try Firefox if Chrome failed
        if not cookies:
            try:
                cookies = browser_cookie3.firefox(domain_name='youtube.com')
                cookie_count = len(list(cookies))
                if cookie_count > 0:
                    browser_name = "Firefox"
                    logging.info(f"Loaded {cookie_count} cookies from Firefox browser")
            except Exception as e:
                logging.debug(f"Failed to load Firefox cookies: {e}")

        # Try Safari (macOS) if both failed
        if not cookies:
            try:
                cookies = browser_cookie3.safari(domain_name='youtube.com')
                cookie_count = len(list(cookies))
                if cookie_count > 0:
                    browser_name = "Safari"
                    logging.info(f"Loaded {cookie_count} cookies from Safari browser")
            except Exception as e:
                logging.debug(f"Failed to load Safari cookies: {e}")

        if not cookies or not browser_name:
            logging.warning(
                "Could not load cookies from any browser. "
                "Make sure you are logged into YouTube in Chrome, Firefox, or Safari."
            )
            return None

        # Create requests session and add cookies
        session = requests.Session()
        session.cookies = cookies
        logging.info(f"✓ Created HTTP session with {browser_name} cookies")
        return session

    except ImportError:
        logging.error("browser-cookie3 not installed. Run: uv add browser-cookie3")
        return None
    except Exception as e:
        logging.warning(f"Unexpected error creating HTTP session with cookies: {e}")
        return None


class YouTubeTranscriptDownloader:
    """Downloads video transcripts using youtube-transcript-api."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        prefer_languages: List[str] = None,
        use_browser_cookies: bool = False,
    ):
        """
        Initialize the transcript downloader.

        Args:
            max_retries: Maximum number of retry attempts for failed requests
            base_delay: Base delay for exponential backoff (seconds)
            prefer_languages: List of preferred language codes (default: ["en"])
            use_browser_cookies: Whether to use browser cookies for authentication (default: False)
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.prefer_languages = prefer_languages or ["en"]
        self.use_browser_cookies = use_browser_cookies

        # Load browser cookies if requested and create HTTP client
        self.http_client = None
        if use_browser_cookies:
            logging.info("Attempting to load browser cookies for YouTube authentication...")
            self.http_client = create_http_client_with_cookies()
            if self.http_client:
                logging.info("✓ Browser cookies loaded successfully - requests will be authenticated")
            else:
                logging.warning("✗ Failed to load browser cookies - requests will be unauthenticated")

        # Initialize the API with HTTP client (if cookies were loaded)
        if self.http_client:
            self.api = YouTubeTranscriptApi(http_client=self.http_client)
        else:
            self.api = YouTubeTranscriptApi()

    def download_transcript(self, video_id: str) -> Optional[Dict]:
        """
        Download transcript for a video.

        Args:
            video_id: YouTube video ID

        Returns:
            Dictionary containing transcript data, or None if unavailable
            {
                "available": True,
                "language": "en",
                "segments": [
                    {"start": 0.0, "duration": 2.5, "text": "Hello world"},
                    ...
                ]
            }

        Example:
            >>> downloader = YouTubeTranscriptDownloader()
            >>> transcript = downloader.download_transcript("dQw4w9WgXcQ")
            >>> if transcript:
            ...     print(f"Downloaded {len(transcript['segments'])} segments")
        """

        def fetch_transcript():
            logging.debug(f"Fetching transcript for video: {video_id}")

            try:
                # Fetch transcript (cookies are automatically used if http_client was configured)
                transcript_segments = self.api.fetch(video_id, self.prefer_languages)

                if not transcript_segments:
                    logging.warning(f"No transcript segments returned for video: {video_id}")
                    return None

                # Convert to our format
                segments = []
                for segment in transcript_segments:
                    segments.append(
                        {
                            "start": segment.start,
                            "duration": segment.duration,
                            "text": segment.text,
                        }
                    )

                # Detect language (assume first preferred language for now)
                language = self.prefer_languages[0] if self.prefer_languages else "en"

                result = {
                    "available": True,
                    "language": language,
                    "segments": segments,
                }

                logging.info(
                    f"Successfully downloaded transcript for {video_id}: "
                    f"{len(segments)} segments, language: {language}"
                )

                return result

            except TranscriptsDisabled:
                logging.warning(f"Transcripts disabled for video: {video_id}")
                return None

            except NoTranscriptFound:
                logging.warning(
                    f"No transcript found in preferred languages {self.prefer_languages} "
                    f"for video: {video_id}"
                )
                return None

            except VideoUnavailable:
                logging.warning(f"Video unavailable: {video_id}")
                return None

            except Exception as e:
                # Log unexpected errors but let retry logic handle it
                logging.warning(f"Unexpected error fetching transcript for {video_id}: {e}")
                raise

        try:
            return execute_with_exponential_backoff(
                fetch_transcript,
                f"Downloading transcript for video {video_id}",
                max_retries=self.max_retries,
                base_delay=self.base_delay,
            )
        except Exception as e:
            # All retries failed
            logging.error(f"Failed to download transcript for {video_id} after retries: {e}")
            return None

    def format_transcript_with_timestamps(self, segments: List[Dict]) -> str:
        """
        Format transcript segments with [MM:SS] timestamps.

        Args:
            segments: List of transcript segments

        Returns:
            Formatted transcript string

        Example:
            >>> segments = [
            ...     {"start": 0.0, "duration": 2.5, "text": "Hello world"},
            ...     {"start": 2.5, "duration": 3.0, "text": "This is a test"}
            ... ]
            >>> formatted = downloader.format_transcript_with_timestamps(segments)
            >>> print(formatted)
            [00:00] Hello world
            [00:02] This is a test
        """
        if not segments:
            return ""

        lines = []
        for segment in segments:
            timestamp = self._format_timestamp(segment["start"])
            text = segment["text"]
            lines.append(f"[{timestamp}] {text}")

        return "\n".join(lines)

    def format_transcript_plain(self, segments: List[Dict]) -> str:
        """
        Format transcript segments as plain text (no timestamps).

        Args:
            segments: List of transcript segments

        Returns:
            Plain text transcript

        Example:
            >>> segments = [
            ...     {"start": 0.0, "duration": 2.5, "text": "Hello world"},
            ...     {"start": 2.5, "duration": 3.0, "text": "This is a test"}
            ... ]
            >>> formatted = downloader.format_transcript_plain(segments)
            >>> print(formatted)
            Hello world This is a test
        """
        if not segments:
            return ""

        texts = [segment["text"] for segment in segments]
        return " ".join(texts)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """
        Format seconds into [MM:SS] or [HH:MM:SS] format.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted timestamp string

        Example:
            >>> YouTubeTranscriptDownloader._format_timestamp(65.5)
            '01:05'
            >>> YouTubeTranscriptDownloader._format_timestamp(3665.5)
            '01:01:05'
        """
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

    def __str__(self) -> str:
        """String representation."""
        cookie_status = "with cookies" if self.http_client else "without cookies"
        return f"YouTubeTranscriptDownloader(languages={self.prefer_languages}, {cookie_status})"
