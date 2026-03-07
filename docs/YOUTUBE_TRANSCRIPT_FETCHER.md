# YouTube Transcript Fetcher

Fetch YouTube video transcripts from any channel and save them in dual format (JSON + Markdown) ready for indexing into your vector database.

## Overview

This tool downloads all videos from a YouTube channel, extracts their transcripts, and saves them in a format optimized for vector search and RAG applications. It integrates seamlessly with the existing `documents-vector-search` infrastructure.

**Key Features**:
- ✅ Fetch all videos from any YouTube channel
- ✅ Download transcripts with timestamps
- ✅ Save dual format: JSON (raw data) + Markdown (indexable content)
- ✅ Resumability - skip already processed videos
- ✅ State tracking for incremental updates
- ✅ Automatic retry with exponential backoff
- ✅ Handles rate limiting gracefully
- ✅ Ready for vector database indexing

## Installation

Dependencies are already included in the project:
- `youtube-transcript-api` - For downloading transcripts
- `yt-dlp` - For video metadata extraction

No additional setup required!

## Quick Start

### 1. Fetch Transcripts from a Channel

```bash
uv run youtube_fetch_cmd_adapter.py \
  --channelUrl "https://www.youtube.com/@EmmaHubbard/videos" \
  --channelName "EmmaHubbard"
```

This will:
1. Discover all videos in the channel
2. Download metadata and transcripts
3. Save to `./data/sources/youtube-transcripts/markdown/EmmaHubbard/`
4. Track progress in `.state` file

### 2. Index the Transcripts

```bash
uv run files_collection_create_cmd_adapter.py \
  --basePath "./data/sources/youtube-transcripts/markdown/EmmaHubbard" \
  --collection "emmahubbard-transcripts"
```

### 3. Search the Transcripts

```bash
uv run collection_search_cmd_adapter.py \
  --collection "emmahubbard-transcripts" \
  --query "baby sleep tips"
```

## Usage

### Basic Usage

```bash
# Minimal command - channel URL only
uv run youtube_fetch_cmd_adapter.py \
  --channelUrl "https://www.youtube.com/@ChannelName/videos"
```

### Advanced Options

```bash
uv run youtube_fetch_cmd_adapter.py \
  --channelUrl "https://www.youtube.com/@EmmaHubbard/videos" \
  --channelName "EmmaHubbard" \
  --outputPath "./my_transcripts" \
  --skipExisting True \
  --includeTimestamps True \
  --maxRetries 3 \
  --languages en
```

**Options**:
- `--channelUrl` (required): YouTube channel URL
- `--channelName` (optional): Override channel name (auto-detected if not provided)
- `--outputPath` (optional): Output directory (default: `./data/sources/youtube-transcripts`)
- `--skipExisting` (optional): Skip already processed videos (default: `True`)
- `--includeTimestamps` (optional): Include `[MM:SS]` timestamps in markdown (default: `True`)
- `--maxRetries` (optional): Max retry attempts for failed requests (default: `3`)
- `--languages` (optional): Preferred transcript languages (default: `en`)

### Check Video Count Before Fetching

```bash
uv run scripts/check_channel_video_count.py \
  --channelUrl "https://www.youtube.com/@EmmaHubbard/videos"
```

This shows how many videos are in the channel before starting the fetch.

## Output Format

### Directory Structure

```
data/sources/youtube-transcripts/
├── json/
│   └── EmmaHubbard/
│       ├── abc123_video_title.json
│       └── def456_another_video.json
├── markdown/
│   └── EmmaHubbard/
│       ├── abc123_video_title.md
│       └── def456_another_video.md
└── .state/
    └── EmmaHubbard.json
```

### JSON Format

Complete raw data for each video:

```json
{
  "video_id": "abc123xyz",
  "title": "Video Title",
  "url": "https://www.youtube.com/watch?v=abc123xyz",
  "upload_date": "2024-01-15",
  "duration": 1234,
  "view_count": 50000,
  "description": "Video description...",
  "channel": "Emma Hubbard",
  "channel_id": "UCxxxxxxxxx",
  "thumbnail": "https://...",
  "transcript": {
    "available": true,
    "language": "en",
    "segments": [
      {
        "start": 0.0,
        "duration": 2.5,
        "text": "Hello everyone"
      }
    ]
  },
  "fetched_at": "2025-11-10T21:25:00Z"
}
```

### Markdown Format

Optimized for vector search with YAML frontmatter:

```markdown
---
title: Video Title
video_id: abc123xyz
channel: Emma Hubbard
url: "https://www.youtube.com/watch?v=abc123xyz"
upload_date: 2024-01-15
duration: 1234
view_count: 50000
description: "Video description..."
fetched_at: "2025-11-10T21:25:00Z"
---

# Video Title

**Channel**: Emma Hubbard
**Published**: 2024-01-15
**Duration**: 20:34
**Views**: 50,000
**URL**: https://www.youtube.com/watch?v=abc123xyz

## Description

Video description text...

## Transcript

[00:00] Hello everyone
[00:02] Today we're talking about...
[00:05] This is the main topic
```

## State Management & Resumability

The fetcher automatically tracks which videos have been processed in `.state/{ChannelName}.json`:

```json
{
  "channel_name": "EmmaHubbard",
  "channel_url": "https://www.youtube.com/@EmmaHubbard/videos",
  "created_at": "2025-11-10T21:25:23Z",
  "last_fetch_time": "2025-11-10T21:48:00Z",
  "processed_videos": {
    "abc123": {
      "video_id": "abc123",
      "status": "success",
      "processed_at": "2025-11-10T21:25:27Z",
      "title": "Video Title",
      "upload_date": "2024-01-15"
    }
  },
  "statistics": {
    "total_videos": 181,
    "successful": 21,
    "failed": 0,
    "skipped": 164
  }
}
```

**Benefits**:
- Re-run the same command to fetch new videos
- Automatically skips already processed videos
- Resume after interruptions (rate limiting, errors, manual stop)
- Track success/failure/skip statistics

## Common Scenarios

### Incremental Updates

Fetch new videos from a channel you've already processed:

```bash
# First run - fetches all videos
uv run youtube_fetch_cmd_adapter.py \
  --channelUrl "https://www.youtube.com/@EmmaHubbard/videos"

# Later run - only fetches new videos
uv run youtube_fetch_cmd_adapter.py \
  --channelUrl "https://www.youtube.com/@EmmaHubbard/videos"
```

The second run automatically skips videos already fetched.

### Re-fetch All Videos

Force re-downloading all videos (ignoring state):

```bash
uv run youtube_fetch_cmd_adapter.py \
  --channelUrl "https://www.youtube.com/@EmmaHubbard/videos" \
  --skipExisting False
```

### Multiple Channels

Fetch from multiple channels (each gets its own folder):

```bash
# Channel 1
uv run youtube_fetch_cmd_adapter.py \
  --channelUrl "https://www.youtube.com/@EmmaHubbard/videos"

# Channel 2
uv run youtube_fetch_cmd_adapter.py \
  --channelUrl "https://www.youtube.com/@AnotherChannel/videos"

# Results in:
# data/sources/youtube-transcripts/markdown/EmmaHubbard/
# data/sources/youtube-transcripts/markdown/AnotherChannel/
```

## Integration with Vector Database

### Create Collection

After fetching transcripts, index them:

```bash
uv run files_collection_create_cmd_adapter.py \
  --basePath "./data/sources/youtube-transcripts/markdown/EmmaHubbard" \
  --collection "emmahubbard-transcripts"
```

### Update Existing Collection

After fetching new videos, update the collection:

```bash
uv run collection_update_cmd_adapter.py \
  --collection "emmahubbard-transcripts"
```

### Search Collection

Search for content across all transcripts:

```bash
uv run collection_search_cmd_adapter.py \
  --collection "emmahubbard-transcripts" \
  --query "baby sleep tips"
```

### MCP Integration

Once indexed, the transcripts are automatically available via MCP for AI agents to search.

## Troubleshooting

### Rate Limiting

**Problem**: YouTube blocks requests after fetching many videos quickly.

**Solution**: This is normal. The fetcher will:
1. Retry with exponential backoff
2. Save all progress to state file
3. Allow you to resume later

Simply wait a few hours and re-run the same command. Already-fetched videos will be skipped.

### No Transcript Available

**Problem**: Some videos show as "skipped" with "No transcript available".

**Reason**: Not all YouTube videos have transcripts. This is normal.

**Action**: None needed. The skipped videos are tracked in the state file.

### Invalid Channel URL

**Problem**: Error "Unable to download API page: HTTP Error 404".

**Solution**: Verify the channel URL format:
- ✅ Correct: `https://www.youtube.com/@ChannelName/videos`
- ✅ Correct: `https://www.youtube.com/c/ChannelName/videos`
- ❌ Wrong: `https://www.youtube.com/@ChannelName` (missing `/videos`)

### Out of Memory

**Problem**: Process crashes with memory errors on very large channels.

**Solution**: Fetch in batches by manually stopping and restarting. State tracking ensures no duplicate work.

## Performance

**Typical Performance**:
- ~2-3 seconds per video (metadata + transcript)
- 100 videos ≈ 5-10 minutes
- 1000 videos ≈ 1-2 hours

**Rate Limiting**: YouTube may temporarily block after ~20-50 videos. Wait 2-4 hours and resume.

## Examples

### Example 1: Fetch & Index Emma Hubbard's Channel

```bash
# Step 1: Fetch transcripts
uv run youtube_fetch_cmd_adapter.py \
  --channelUrl "https://www.youtube.com/@EmmaHubbard/videos" \
  --channelName "EmmaHubbard"

# Step 2: Index
uv run files_collection_create_cmd_adapter.py \
  --basePath "./data/sources/youtube-transcripts/markdown/EmmaHubbard" \
  --collection "emmahubbard-transcripts"

# Step 3: Search
uv run collection_search_cmd_adapter.py \
  --collection "emmahubbard-transcripts" \
  --query "how to handle tantrums"
```

### Example 2: Weekly Updates

Set up a cron job or scheduled task:

```bash
#!/bin/bash
# weekly_youtube_update.sh

# Fetch new videos
uv run youtube_fetch_cmd_adapter.py \
  --channelUrl "https://www.youtube.com/@EmmaHubbard/videos" \
  --channelName "EmmaHubbard"

# Update the collection
uv run collection_update_cmd_adapter.py \
  --collection "emmahubbard-transcripts"
```

## Architecture

### Components

1. **YouTubeMetadataExtractor** (`youtube_metadata_extractor.py`)
   - Uses `yt-dlp` to discover videos and extract metadata
   - Handles channel video discovery
   - Formats upload dates

2. **YouTubeTranscriptDownloader** (`youtube_transcript_downloader.py`)
   - Uses `youtube-transcript-api` to download transcripts
   - Formats timestamps `[MM:SS]` or `[HH:MM:SS]`
   - Supports multiple languages

3. **YouTubeStateManager** (`youtube_state_manager.py`)
   - Tracks processed videos in JSON state file
   - Enables resumability and skip-existing logic
   - Maintains statistics

4. **YouTubeChannelFetcher** (`youtube_channel_fetcher.py`)
   - Main orchestrator coordinating all components
   - Handles dual format saving
   - Progress reporting

5. **Retry Utilities** (`retry_utils.py`)
   - Exponential backoff for rate limiting
   - Configurable retry attempts
   - Error handling

6. **Filename Utilities** (`filename_utils.py`)
   - Safe cross-platform filenames
   - Unicode character normalization
   - Length limits

### Data Flow

```
Channel URL
    ↓
[Metadata Extractor] → Video IDs + Metadata
    ↓
[State Manager] → Filter already processed
    ↓
[Transcript Downloader] → Transcripts
    ↓
[Channel Fetcher] → Save JSON + Markdown
    ↓
[State Manager] → Update state
    ↓
Output: data/sources/youtube-transcripts/{json,markdown}/ChannelName/
```

## Testing

Run the test suite:

```bash
# Test dependencies
uv run scripts/test_youtube_dependencies.py

# Test Phase 1 components (state, filenames)
uv run scripts/test_phase1_components.py

# Test Phase 2 components (metadata, transcripts)
uv run scripts/test_phase2_components.py

# Test single video
uv run scripts/test_single_video.py --videoId dQw4w9WgXcQ

# Test channel fetcher
uv run scripts/test_channel_fetcher.py --videoId dQw4w9WgXcQ
```

## Contributing

The YouTube transcript fetcher follows the established patterns in this codebase:
- Reader-Converter pattern (like Jira/Confluence sources)
- Retry logic with exponential backoff
- State management for resumability
- Dual format storage (raw + indexable)

## License

Same as the parent project.

## Credits

Built using:
- [`youtube-transcript-api`](https://github.com/jdepoix/youtube-transcript-api) by jdepoix
- [`yt-dlp`](https://github.com/yt-dlp/yt-dlp) - YouTube metadata extraction
