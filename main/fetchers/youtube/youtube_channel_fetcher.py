"""
YouTube Channel Fetcher - Main orchestrator for fetching videos from a channel.

This module orchestrates the entire fetching process:
1. Discover videos from channel
2. Extract metadata
3. Download transcripts
4. Save dual format (JSON + Markdown)
5. Track state for resumability
"""

import json
import logging
import time
import random
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional, List

from .youtube_metadata_extractor import YouTubeMetadataExtractor
from .youtube_transcript_downloader import YouTubeTranscriptDownloader
from .youtube_state_manager import YouTubeStateManager
from .filename_utils import create_safe_filename


class YouTubeChannelFetcher:
    """Main orchestrator for fetching videos from a YouTube channel."""

    def __init__(
        self,
        channel_url: str,
        output_base_path: str = "data/sources/youtube-transcripts",
        channel_name: Optional[str] = None,
        skip_existing: bool = True,
        include_timestamps: bool = True,
        max_retries: int = 3,
        prefer_languages: List[str] = None,
        delay_between_videos: float = 0,
        use_browser_cookies: bool = False,
    ):
        """
        Initialize the channel fetcher.

        Args:
            channel_url: YouTube channel URL (e.g., https://www.youtube.com/@EmmaHubbard/videos)
            output_base_path: Base output directory (default: data/sources/youtube-transcripts)
            channel_name: Override channel name (derived from URL if not provided)
            skip_existing: Skip videos already processed (default: True)
            include_timestamps: Include timestamps in markdown (default: True)
            max_retries: Max retries for failed requests (default: 3)
            prefer_languages: Preferred transcript languages (default: ["en"])
            delay_between_videos: Delay in seconds between processing videos (default: 0)
            use_browser_cookies: Use browser cookies for authentication to avoid rate limiting (default: False)
        """
        self.channel_url = channel_url
        self.output_base_path = Path(output_base_path)
        self.skip_existing = skip_existing
        self.include_timestamps = include_timestamps
        self.max_retries = max_retries
        self.delay_between_videos = delay_between_videos
        self.use_browser_cookies = use_browser_cookies

        # Initialize components
        self.metadata_extractor = YouTubeMetadataExtractor(
            channel_url=channel_url,
            max_retries=max_retries,
        )

        self.transcript_downloader = YouTubeTranscriptDownloader(
            max_retries=max_retries,
            prefer_languages=prefer_languages or ["en"],
            use_browser_cookies=use_browser_cookies,
        )

        self.state_manager = YouTubeStateManager(
            channel_url=channel_url,
            output_base_path=str(output_base_path),
        )

        # Override channel name if provided
        if channel_name:
            self.state_manager.channel_name = channel_name

        self.channel_name = self.state_manager.channel_name

        # Setup output directories
        self._setup_directories()

    def _setup_directories(self):
        """Create output directory structure."""
        self.json_dir = self.output_base_path / "json" / self.channel_name
        self.markdown_dir = self.output_base_path / "markdown" / self.channel_name

        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Output directories created:")
        logging.info(f"  JSON: {self.json_dir}")
        logging.info(f"  Markdown: {self.markdown_dir}")

    def fetch_all(self) -> Dict:
        """
        Fetch all videos from the channel.

        Returns:
            Statistics dictionary with results

        Example:
            >>> fetcher = YouTubeChannelFetcher("https://www.youtube.com/@EmmaHubbard/videos")
            >>> stats = fetcher.fetch_all()
            >>> print(f"Successfully fetched {stats['successful']} videos")
        """
        logging.info(f"Starting fetch for channel: {self.channel_name}")
        logging.info(f"Channel URL: {self.channel_url}")
        logging.info(f"Skip existing: {self.skip_existing}")

        # Get all video IDs from channel
        logging.info("Discovering videos from channel...")
        video_ids = self.metadata_extractor.get_all_video_ids()
        logging.info(f"Found {len(video_ids)} videos in channel")

        if not video_ids:
            logging.warning("No videos found in channel")
            return self.state_manager.get_statistics()

        # Filter out already processed videos if skip_existing is True
        if self.skip_existing:
            processed_ids = self.state_manager.get_processed_video_ids()
            video_ids_to_fetch = [vid for vid in video_ids if vid not in processed_ids]

            skipped_count = len(video_ids) - len(video_ids_to_fetch)
            if skipped_count > 0:
                logging.info(
                    f"Skipping {skipped_count} already processed videos"
                )

            video_ids = video_ids_to_fetch

        if not video_ids:
            logging.info("All videos already processed")
            return self.state_manager.get_statistics()

        logging.info(f"Fetching {len(video_ids)} videos...")

        # Process each video
        for i, video_id in enumerate(video_ids, 1):
            logging.info(f"\nProcessing video {i}/{len(video_ids)}: {video_id}")

            try:
                success = self.fetch_video(video_id)

                if success:
                    logging.info(f"✓ Successfully processed video {video_id}")
                else:
                    logging.warning(f"○ Skipped video {video_id} (no transcript or unavailable)")

            except Exception as e:
                logging.error(f"✗ Failed to process video {video_id}: {e}")
                self.state_manager.mark_video_processed(
                    video_id, "failed", error=str(e)
                )

            # Save state after each video
            self.state_manager.save_state()

            # Add delay between videos if configured (to avoid rate limiting)
            if self.delay_between_videos > 0 and i < len(video_ids):
                # Add random jitter (0-2 seconds) to appear more human-like
                jitter = random.uniform(0, 2)
                total_delay = self.delay_between_videos + jitter
                logging.info(f"Waiting {total_delay:.1f}s before next video (base: {self.delay_between_videos}s + jitter: {jitter:.1f}s)...")
                time.sleep(total_delay)

        # Final statistics
        stats = self.state_manager.get_statistics()
        logging.info(f"\n{'=' * 60}")
        logging.info(f"Fetch Complete for {self.channel_name}")
        logging.info(f"{'=' * 60}")
        logging.info(f"Total videos processed: {stats['total_videos']}")
        logging.info(f"Successful: {stats['successful']}")
        logging.info(f"Failed: {stats['failed']}")
        logging.info(f"Skipped: {stats['skipped']}")
        logging.info(f"{'=' * 60}")

        return stats

    def fetch_video(self, video_id: str) -> bool:
        """
        Fetch a single video (metadata + transcript).

        Args:
            video_id: YouTube video ID

        Returns:
            True if successful, False if skipped (no transcript)

        Raises:
            Exception: If an error occurs during processing
        """
        # Step 1: Get metadata
        logging.info(f"  Fetching metadata...")
        metadata = self.metadata_extractor.get_video_metadata(video_id)

        if not metadata:
            logging.warning(f"  Video unavailable: {video_id}")
            self.state_manager.mark_video_processed(
                video_id, "skipped", error="Video unavailable"
            )
            return False

        logging.info(f"  Title: {metadata['title']}")

        # Step 2: Download transcript
        logging.info(f"  Downloading transcript...")
        transcript_data = self.transcript_downloader.download_transcript(video_id)

        if not transcript_data or not transcript_data.get("available"):
            logging.warning(f"  No transcript available for: {metadata['title']}")
            self.state_manager.mark_video_processed(
                video_id, "skipped", metadata=metadata, error="No transcript available"
            )
            return False

        logging.info(f"  Downloaded {len(transcript_data['segments'])} transcript segments")

        # Step 3: Save dual format
        logging.info(f"  Saving files...")
        self._save_dual_format(video_id, metadata, transcript_data)

        # Step 4: Update state
        self.state_manager.mark_video_processed(
            video_id, "success", metadata=metadata
        )

        return True

    def _save_dual_format(
        self, video_id: str, metadata: Dict, transcript_data: Dict
    ):
        """
        Save video data in both JSON and Markdown formats.

        Args:
            video_id: YouTube video ID
            metadata: Video metadata dictionary
            transcript_data: Transcript data dictionary
        """
        # Create safe filename
        filename = create_safe_filename(video_id, metadata["title"], "")

        # Save JSON
        json_file = self.json_dir / f"{filename}.json"
        self._save_json(json_file, metadata, transcript_data)
        logging.info(f"    JSON: {json_file.name}")

        # Save Markdown
        markdown_file = self.markdown_dir / f"{filename}.md"
        self._save_markdown(markdown_file, metadata, transcript_data)
        logging.info(f"    Markdown: {markdown_file.name}")

    def _save_json(self, file_path: Path, metadata: Dict, transcript_data: Dict):
        """Save raw data as JSON."""
        data = {
            **metadata,
            "transcript": transcript_data,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_markdown(self, file_path: Path, metadata: Dict, transcript_data: Dict):
        """Save formatted content as Markdown with frontmatter."""
        content = self._create_markdown_content(metadata, transcript_data)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    def _create_markdown_content(self, metadata: Dict, transcript_data: Dict) -> str:
        """
        Create markdown content with frontmatter.

        Args:
            metadata: Video metadata
            transcript_data: Transcript data

        Returns:
            Formatted markdown string
        """
        # Build frontmatter
        frontmatter_data = {
            "title": metadata["title"],
            "video_id": metadata["video_id"],
            "channel": metadata["channel"],
            "url": metadata["url"],
            "upload_date": metadata["upload_date"],
            "duration": metadata["duration"],
            "view_count": metadata["view_count"],
            "description": metadata["description"][:500] if metadata["description"] else "",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

        # Build markdown
        lines = ["---"]

        for key, value in frontmatter_data.items():
            # Escape special characters in values
            if isinstance(value, str) and (":" in value or "\n" in value):
                # Use block scalar for multi-line or strings with colons
                value = value.replace("\n", " ").replace('"', '\\"')
                lines.append(f'{key}: "{value}"')
            else:
                lines.append(f"{key}: {value}")

        lines.append("---")
        lines.append("")

        # Add title
        lines.append(f"# {metadata['title']}")
        lines.append("")

        # Add metadata section
        lines.append(f"**Channel**: {metadata['channel']}")
        lines.append(f"**Published**: {metadata['upload_date']}")

        # Format duration
        duration_min = metadata['duration'] // 60
        duration_sec = metadata['duration'] % 60
        lines.append(f"**Duration**: {duration_min}:{duration_sec:02d}")

        lines.append(f"**Views**: {metadata['view_count']:,}")
        lines.append(f"**URL**: {metadata['url']}")
        lines.append("")

        # Add description if present
        if metadata.get("description"):
            lines.append("## Description")
            lines.append("")
            lines.append(metadata["description"])
            lines.append("")

        # Add transcript
        lines.append("## Transcript")
        lines.append("")

        if self.include_timestamps:
            transcript_text = self.transcript_downloader.format_transcript_with_timestamps(
                transcript_data["segments"]
            )
        else:
            transcript_text = self.transcript_downloader.format_transcript_plain(
                transcript_data["segments"]
            )

        lines.append(transcript_text)

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation."""
        return (
            f"YouTubeChannelFetcher(channel={self.channel_name}, "
            f"skip_existing={self.skip_existing})"
        )
