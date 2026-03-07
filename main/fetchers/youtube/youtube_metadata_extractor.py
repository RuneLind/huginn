"""
YouTube Metadata Extractor - Extracts video metadata using yt-dlp.

This module provides functionality to:
1. Discover all video IDs from a YouTube channel
2. Extract detailed metadata for individual videos
"""

import yt_dlp
import logging
from typing import List, Dict, Optional
from datetime import datetime

from .retry_utils import execute_with_exponential_backoff


class YouTubeMetadataExtractor:
    """Extracts video metadata from YouTube channels using yt-dlp."""

    def __init__(
        self,
        channel_url: str,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ):
        """
        Initialize the metadata extractor.

        Args:
            channel_url: YouTube channel URL (e.g., https://www.youtube.com/@EmmaHubbard/videos)
            max_retries: Maximum number of retry attempts for failed requests
            base_delay: Base delay for exponential backoff (seconds)
        """
        self.channel_url = channel_url
        self.max_retries = max_retries
        self.base_delay = base_delay

        # yt-dlp options for channel discovery
        self.channel_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,  # Don't download, just get video list
            "skip_download": True,
        }

        # yt-dlp options for video metadata
        self.video_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "extract_flat": False,  # Get full metadata
        }

    def get_all_video_ids(self) -> List[str]:
        """
        Get list of all video IDs from the channel.

        Returns:
            List of video IDs

        Raises:
            Exception: If unable to fetch video list after retries

        Example:
            >>> extractor = YouTubeMetadataExtractor("https://www.youtube.com/@EmmaHubbard/videos")
            >>> video_ids = extractor.get_all_video_ids()
            >>> print(f"Found {len(video_ids)} videos")
        """

        def fetch_video_list():
            logging.info(f"Fetching video list from channel: {self.channel_url}")

            with yt_dlp.YoutubeDL(self.channel_opts) as ydl:
                # Extract channel info
                result = ydl.extract_info(self.channel_url, download=False)

                if not result:
                    raise Exception("Failed to extract channel information")

                # Get entries (videos)
                entries = result.get("entries", [])

                if not entries:
                    logging.warning("No videos found in channel")
                    return []

                # Extract video IDs
                video_ids = []
                for entry in entries:
                    if entry and "id" in entry:
                        video_ids.append(entry["id"])

                logging.info(f"Found {len(video_ids)} videos in channel")
                return video_ids

        return execute_with_exponential_backoff(
            fetch_video_list,
            f"Fetching video list from {self.channel_url}",
            max_retries=self.max_retries,
            base_delay=self.base_delay,
        )

    def get_video_metadata(self, video_id: str) -> Optional[Dict]:
        """
        Get detailed metadata for a single video.

        Args:
            video_id: YouTube video ID

        Returns:
            Dictionary containing video metadata, or None if video unavailable

        Metadata includes:
            - video_id: YouTube video ID
            - title: Video title
            - url: Full YouTube URL
            - upload_date: Upload date (YYYY-MM-DD format)
            - duration: Duration in seconds
            - view_count: Number of views
            - description: Video description
            - channel: Channel name
            - channel_id: Channel ID

        Example:
            >>> extractor = YouTubeMetadataExtractor("https://www.youtube.com/@EmmaHubbard/videos")
            >>> metadata = extractor.get_video_metadata("abc123xyz")
            >>> print(metadata["title"])
        """

        def fetch_metadata():
            video_url = f"https://www.youtube.com/watch?v={video_id}"

            logging.debug(f"Fetching metadata for video: {video_id}")

            try:
                with yt_dlp.YoutubeDL(self.video_opts) as ydl:
                    info = ydl.extract_info(video_url, download=False)

                    if not info:
                        logging.warning(f"No metadata returned for video: {video_id}")
                        return None

                    # Extract relevant metadata
                    metadata = {
                        "video_id": video_id,
                        "title": info.get("title", "Untitled"),
                        "url": video_url,
                        "upload_date": self._format_upload_date(
                            info.get("upload_date")
                        ),
                        "duration": info.get("duration", 0),
                        "view_count": info.get("view_count", 0),
                        "description": info.get("description", ""),
                        "channel": info.get("uploader", "Unknown"),
                        "channel_id": info.get("channel_id", ""),
                        "thumbnail": info.get("thumbnail", ""),
                    }

                    logging.debug(f"Successfully fetched metadata for: {metadata['title']}")
                    return metadata

            except yt_dlp.utils.DownloadError as e:
                # Video might be unavailable, private, or deleted
                logging.warning(f"Video {video_id} unavailable: {e}")
                return None

        return execute_with_exponential_backoff(
            fetch_metadata,
            f"Fetching metadata for video {video_id}",
            max_retries=self.max_retries,
            base_delay=self.base_delay,
        )

    def get_all_videos_metadata(self) -> List[Dict]:
        """
        Get metadata for all videos in the channel.

        This is a convenience method that combines get_all_video_ids()
        and get_video_metadata() for each video.

        Returns:
            List of metadata dictionaries (excludes unavailable videos)

        Example:
            >>> extractor = YouTubeMetadataExtractor("https://www.youtube.com/@EmmaHubbard/videos")
            >>> all_metadata = extractor.get_all_videos_metadata()
            >>> for meta in all_metadata:
            ...     print(f"{meta['title']} - {meta['upload_date']}")
        """
        video_ids = self.get_all_video_ids()

        all_metadata = []
        for video_id in video_ids:
            metadata = self.get_video_metadata(video_id)
            if metadata:
                all_metadata.append(metadata)

        logging.info(
            f"Successfully fetched metadata for {len(all_metadata)}/{len(video_ids)} videos"
        )

        return all_metadata

    @staticmethod
    def _format_upload_date(upload_date_str: Optional[str]) -> str:
        """
        Format upload date from yt-dlp format (YYYYMMDD) to YYYY-MM-DD.

        Args:
            upload_date_str: Upload date in YYYYMMDD format

        Returns:
            Formatted date string (YYYY-MM-DD) or empty string if invalid
        """
        if not upload_date_str:
            return ""

        try:
            # Parse YYYYMMDD format
            date_obj = datetime.strptime(str(upload_date_str), "%Y%m%d")
            # Return in YYYY-MM-DD format
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            logging.warning(f"Invalid upload date format: {upload_date_str}")
            return ""

    def __str__(self) -> str:
        """String representation."""
        return f"YouTubeMetadataExtractor(channel_url={self.channel_url})"
