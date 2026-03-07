"""
YouTube State Manager - Tracks processed videos for resumability.

This module manages the state file that tracks which videos have been processed,
enabling the fetcher to skip already-processed videos on subsequent runs.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Set, Optional
import logging


class YouTubeStateManager:
    """Manages state tracking for YouTube video fetching."""

    def __init__(self, channel_url: str, output_base_path: str = "data/sources/youtube-transcripts"):
        """
        Initialize the state manager.

        Args:
            channel_url: The YouTube channel URL
            output_base_path: Base path for output (default: data/sources/youtube-transcripts)
        """
        self.channel_url = channel_url
        self.output_base_path = output_base_path
        self.channel_name = self._extract_channel_name(channel_url)

        # State directory: data/sources/youtube-transcripts/.state/
        self.state_dir = Path(output_base_path) / ".state"
        self.state_file = self.state_dir / f"{self.channel_name}.json"

        # Ensure state directory exists
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Load existing state or initialize new
        self.state = self._load_state()

    def _extract_channel_name(self, channel_url: str) -> str:
        """
        Extract channel name from URL.

        Args:
            channel_url: YouTube channel URL (e.g., https://www.youtube.com/@EmmaHubbard/videos)

        Returns:
            Channel name (e.g., EmmaHubbard)
        """
        # Extract from @ChannelName format
        if "@" in channel_url:
            parts = channel_url.split("@")
            if len(parts) > 1:
                name = parts[1].split("/")[0]
                return name

        # Fallback: use last part of URL
        return channel_url.rstrip("/").split("/")[-1]

    def _load_state(self) -> dict:
        """
        Load state from file or create new state.

        Returns:
            State dictionary
        """
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    logging.info(f"Loaded state for channel: {self.channel_name}")
                    return state
            except Exception as e:
                logging.warning(f"Failed to load state file, creating new state: {e}")

        # Create new state
        return self._create_new_state()

    def _create_new_state(self) -> dict:
        """
        Create a new state dictionary.

        Returns:
            New state dictionary
        """
        return {
            "channel_name": self.channel_name,
            "channel_url": self.channel_url,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_fetch_time": None,
            "processed_videos": {},
            "statistics": {
                "total_videos": 0,
                "successful": 0,
                "failed": 0,
                "skipped": 0,
            },
        }

    def save_state(self):
        """Save current state to file."""
        try:
            self.state["last_fetch_time"] = datetime.now(timezone.utc).isoformat()

            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)

            logging.debug(f"State saved to {self.state_file}")
        except Exception as e:
            logging.error(f"Failed to save state: {e}")
            raise

    def is_video_processed(self, video_id: str) -> bool:
        """
        Check if a video has been processed successfully.

        Args:
            video_id: YouTube video ID

        Returns:
            True if video was successfully processed, False otherwise
        """
        video_state = self.state["processed_videos"].get(video_id)
        if video_state:
            return video_state.get("status") == "success"
        return False

    def mark_video_processed(
        self,
        video_id: str,
        status: str,
        metadata: Optional[Dict] = None,
        error: Optional[str] = None,
    ):
        """
        Mark a video as processed with given status.

        Args:
            video_id: YouTube video ID
            status: Status - "success", "failed", or "skipped"
            metadata: Optional video metadata (title, etc.)
            error: Optional error message if failed
        """
        video_record = {
            "video_id": video_id,
            "status": status,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }

        if metadata:
            video_record["title"] = metadata.get("title")
            video_record["upload_date"] = metadata.get("upload_date")

        if error:
            video_record["error"] = error

        self.state["processed_videos"][video_id] = video_record

        # Update statistics
        if status == "success":
            self.state["statistics"]["successful"] += 1
        elif status == "failed":
            self.state["statistics"]["failed"] += 1
        elif status == "skipped":
            self.state["statistics"]["skipped"] += 1

        self.state["statistics"]["total_videos"] = len(self.state["processed_videos"])

    def get_processed_video_ids(self) -> Set[str]:
        """
        Get set of all successfully processed video IDs.

        Returns:
            Set of video IDs that were successfully processed
        """
        return {
            video_id
            for video_id, record in self.state["processed_videos"].items()
            if record.get("status") == "success"
        }

    def get_failed_video_ids(self) -> Set[str]:
        """
        Get set of all failed video IDs.

        Returns:
            Set of video IDs that failed processing
        """
        return {
            video_id
            for video_id, record in self.state["processed_videos"].items()
            if record.get("status") == "failed"
        }

    def get_statistics(self) -> dict:
        """
        Get processing statistics.

        Returns:
            Statistics dictionary
        """
        return self.state["statistics"].copy()

    def get_video_record(self, video_id: str) -> Optional[dict]:
        """
        Get processing record for a specific video.

        Args:
            video_id: YouTube video ID

        Returns:
            Video record dictionary or None if not found
        """
        return self.state["processed_videos"].get(video_id)

    def reset_failed_videos(self):
        """Reset status of failed videos to allow re-processing."""
        failed_ids = self.get_failed_video_ids()
        for video_id in failed_ids:
            del self.state["processed_videos"][video_id]

        self.state["statistics"]["failed"] = 0
        self.state["statistics"]["total_videos"] = len(self.state["processed_videos"])

        logging.info(f"Reset {len(failed_ids)} failed videos for re-processing")

    def __str__(self) -> str:
        """String representation of state."""
        stats = self.get_statistics()
        return (
            f"YouTubeStateManager(channel={self.channel_name}, "
            f"total={stats['total_videos']}, "
            f"successful={stats['successful']}, "
            f"failed={stats['failed']}, "
            f"skipped={stats['skipped']})"
        )
