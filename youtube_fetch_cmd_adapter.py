#!/usr/bin/env python3
"""
YouTube Fetch Command Adapter - CLI for fetching YouTube video transcripts.

This script fetches all videos from a YouTube channel, downloads their transcripts,
and saves them in dual format (JSON + Markdown) ready for indexing.

Usage:
    # Fetch all videos from a channel
    uv run youtube_fetch_cmd_adapter.py \
      --channelUrl "https://www.youtube.com/@EmmaHubbard/videos" \
      --channelName "EmmaHubbard"

    # Fetch with custom output path
    uv run youtube_fetch_cmd_adapter.py \
      --channelUrl "https://www.youtube.com/@EmmaHubbard/videos" \
      --channelName "EmmaHubbard" \
      --outputPath "./my_transcripts"

    # Fetch without skipping existing (re-fetch all)
    uv run youtube_fetch_cmd_adapter.py \
      --channelUrl "https://www.youtube.com/@EmmaHubbard/videos" \
      --skipExisting False

After fetching, index the transcripts using:
    uv run files_collection_create_cmd_adapter.py \
      --basePath "./data/sources/youtube-transcripts/markdown/EmmaHubbard" \
      --collection "emma-hubbard-transcripts"
"""

import argparse
import sys
from pathlib import Path

from main.utils.logger import setup_root_logger
from main.fetchers.youtube.youtube_channel_fetcher import YouTubeChannelFetcher

setup_root_logger()


def parse_boolean(value):
    """Parse boolean arguments."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    """Main entry point for YouTube fetch CLI."""
    parser = argparse.ArgumentParser(
        description="Fetch YouTube video transcripts from a channel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - fetch all videos from a channel
  %(prog)s --channelUrl "https://www.youtube.com/@EmmaHubbard/videos"

  # With delay to avoid rate limiting (recommended for large channels)
  %(prog)s --channelUrl "https://www.youtube.com/@EmmaHubbard/videos" --delayBetweenVideos 3

  # With browser cookies for authentication (best for avoiding rate limits)
  %(prog)s --channelUrl "https://www.youtube.com/@EmmaHubbard/videos" --useBrowserCookies True

  # Combined: cookies + delay for maximum reliability
  %(prog)s --channelUrl "https://www.youtube.com/@EmmaHubbard/videos" --useBrowserCookies True --delayBetweenVideos 3

  # Specify channel name explicitly
  %(prog)s --channelUrl "https://www.youtube.com/@EmmaHubbard/videos" --channelName "EmmaHubbard"

  # Custom output path
  %(prog)s --channelUrl "https://www.youtube.com/@EmmaHubbard/videos" --outputPath "./my_transcripts"

  # Re-fetch all videos (don't skip existing)
  %(prog)s --channelUrl "https://www.youtube.com/@EmmaHubbard/videos" --skipExisting False

After fetching, index with:
  uv run files_collection_create_cmd_adapter.py \\
    --basePath "./data/sources/youtube-transcripts/markdown/EmmaHubbard" \\
    --collection "emma-hubbard-transcripts"
        """
    )

    # Required arguments
    parser.add_argument(
        "-channelUrl",
        "--channelUrl",
        required=True,
        help="YouTube channel URL (e.g., https://www.youtube.com/@EmmaHubbard/videos)",
    )

    # Optional arguments
    parser.add_argument(
        "-channelName",
        "--channelName",
        required=False,
        default=None,
        help="Channel name for folder structure (derived from URL if not provided)",
    )

    parser.add_argument(
        "-outputPath",
        "--outputPath",
        required=False,
        default="data/sources/youtube-transcripts",
        help="Base output path (default: ./data/sources/youtube-transcripts)",
    )

    parser.add_argument(
        "-skipExisting",
        "--skipExisting",
        type=parse_boolean,
        required=False,
        default=True,
        help="Skip already processed videos (default: True)",
    )

    parser.add_argument(
        "-includeTimestamps",
        "--includeTimestamps",
        type=parse_boolean,
        required=False,
        default=True,
        help="Include timestamps in markdown transcripts (default: True)",
    )

    parser.add_argument(
        "-maxRetries",
        "--maxRetries",
        type=int,
        required=False,
        default=3,
        help="Maximum retry attempts for failed requests (default: 3)",
    )

    parser.add_argument(
        "-languages",
        "--languages",
        required=False,
        default=["en"],
        nargs='+',
        help="Preferred transcript languages (default: en)",
    )

    parser.add_argument(
        "-delayBetweenVideos",
        "--delayBetweenVideos",
        type=float,
        required=False,
        default=0,
        help="Delay in seconds between processing videos to avoid rate limiting (default: 0). Recommended: 3-5 seconds. Random jitter (0-2s) will be added automatically.",
    )

    parser.add_argument(
        "-useBrowserCookies",
        "--useBrowserCookies",
        type=parse_boolean,
        required=False,
        default=False,
        help="Use browser cookies for authentication to avoid rate limiting (default: False). Requires being logged into YouTube in Chrome, Firefox, or Safari.",
    )

    args = parser.parse_args()

    # Display configuration
    print("=" * 80)
    print("YouTube Transcript Fetcher")
    print("=" * 80)
    print(f"Channel URL:        {args.channelUrl}")
    print(f"Channel Name:       {args.channelName or '(auto-detected)'}")
    print(f"Output Path:        {args.outputPath}")
    print(f"Skip Existing:      {args.skipExisting}")
    print(f"Include Timestamps: {args.includeTimestamps}")
    print(f"Max Retries:        {args.maxRetries}")
    print(f"Languages:          {', '.join(args.languages)}")
    print(f"Delay Between Vids: {args.delayBetweenVideos}s (+ random jitter 0-2s)")
    print(f"Browser Cookies:    {args.useBrowserCookies}")
    print("=" * 80)
    print()

    try:
        # Initialize fetcher
        fetcher = YouTubeChannelFetcher(
            channel_url=args.channelUrl,
            output_base_path=args.outputPath,
            channel_name=args.channelName,
            skip_existing=args.skipExisting,
            include_timestamps=args.includeTimestamps,
            max_retries=args.maxRetries,
            prefer_languages=args.languages,
            delay_between_videos=args.delayBetweenVideos,
            use_browser_cookies=args.useBrowserCookies,
        )

        # Run fetch
        stats = fetcher.fetch_all()

        # Display results
        print()
        print("=" * 80)
        print("Fetch Complete!")
        print("=" * 80)
        print(f"Total videos:  {stats['total_videos']}")
        print(f"Successful:    {stats['successful']}")
        print(f"Failed:        {stats['failed']}")
        print(f"Skipped:       {stats['skipped']}")
        print("=" * 80)
        print()

        # Show output paths
        output_path = Path(args.outputPath)
        json_dir = output_path / "json" / fetcher.channel_name
        markdown_dir = output_path / "markdown" / fetcher.channel_name

        print("Output directories:")
        print(f"  JSON:     {json_dir}")
        print(f"  Markdown: {markdown_dir}")
        print()

        # Show next steps
        if stats['successful'] > 0:
            print("Next steps:")
            print("  1. Review the markdown files to ensure quality")
            print("  2. Index the transcripts:")
            print(f"     uv run files_collection_create_cmd_adapter.py \\")
            print(f"       --basePath \"{markdown_dir}\" \\")
            print(f"       --collection \"{fetcher.channel_name.lower()}-transcripts\"")
            print()
            print("  3. Search the indexed transcripts:")
            print(f"     uv run collection_search_cmd_adapter.py \\")
            print(f"       --collection \"{fetcher.channel_name.lower()}-transcripts\" \\")
            print(f"       --query \"your search query\"")
            print()

        # Exit with appropriate code
        if stats['failed'] > 0:
            print(f"Warning: {stats['failed']} videos failed. Check logs for details.")
            return 1
        else:
            return 0

    except KeyboardInterrupt:
        print("\n\nFetch interrupted by user.")
        return 130
    except Exception as e:
        print(f"\n\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
