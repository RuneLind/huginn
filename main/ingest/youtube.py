"""YouTube transcript ingest: fetch, summarize via Claude, save as categorized markdown."""
import json
import logging
import os
import re
import subprocess
import urllib.request
import datetime as dt
from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel

from main.fetchers.youtube.youtube_transcript_downloader import YouTubeTranscriptDownloader
from main.ingest.categories import CATEGORIES
from main.ingest._summary_ingest import write_summary
from main.utils.claude_cli import call_claude

logger = logging.getLogger(__name__)


class YouTubeIngestRequest(BaseModel):
    title: str
    url: str
    video_id: Optional[str] = None
    transcript: Optional[str] = None  # if provided, skip fetching
    summary: Optional[str] = None  # if provided, skip Claude summarization
    date: Optional[str] = None
    category: Optional[str] = None  # auto-detected if not provided


_GENERIC_TITLES = {"youtube", "youtube.com", "(1) youtube", "(2) youtube", "(3) youtube", ""}

SUMMARIZE_PROMPT = """Summarize this YouTube video transcript into structured key insights.

Format rules:
- Use ### for section headers that group related points
- Use numbered lists or bullet points with emoji prefixes for each insight
- Use **bold** for key terms, concepts, and important data points
- Keep it concise but capture all important points and actionable takeaways
- Each point should be self-contained and informative

Also pick the single best category from this list: {categories}

Video title: {title}

Transcript:
{transcript}

Respond in this exact format:
CATEGORY: <one category from the list>

SUMMARY:
<your markdown summary>"""


def _extract_video_id(url_or_id: str) -> str:
    """Extract video ID from YouTube URL or return as-is."""
    match = re.search(r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})', url_or_id)
    if match:
        return match.group(1)
    if len(url_or_id) == 11 and re.match(r'^[a-zA-Z0-9_-]+$', url_or_id):
        return url_or_id
    raise HTTPException(status_code=400, detail=f"Invalid YouTube URL or video ID: {url_or_id}")


def _fetch_youtube_title(video_id: str) -> Optional[str]:
    """Fetch the real video title from YouTube's oembed API."""
    try:
        url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("title")
    except Exception as e:
        logger.warning(f"Failed to fetch YouTube title for {video_id}: {e}")
        return None


def fetch_transcript(video_id_or_url: str) -> str:
    """Fetch YouTube transcript server-side using YouTubeTranscriptDownloader."""
    video_id = _extract_video_id(video_id_or_url)
    downloader = YouTubeTranscriptDownloader(max_retries=3, prefer_languages=["en"])

    transcript_data = downloader.download_transcript(video_id)
    if not transcript_data or not transcript_data.get("available"):
        raise HTTPException(status_code=422, detail=f"No transcript available for video {video_id}")

    text = downloader.format_transcript_plain(transcript_data["segments"])
    if not text.strip():
        raise HTTPException(status_code=422, detail="Transcript is empty")

    logger.info(f"Fetched transcript for {video_id}: {len(text)} chars")
    return text


def _call_claude_headless(prompt: str, model: Optional[str] = None) -> str:
    """Call Claude headless and translate transport errors into FastAPI ``HTTPException``."""
    model = model or os.environ.get("CLAUDE_MODEL", "sonnet")
    try:
        return call_claude(prompt, model=model, timeout=180)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Claude CLI not found. Install Claude Code first.")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Claude CLI timed out after 180s")
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)[:500])


def _parse_claude_response(text: str) -> tuple[str, str]:
    """Parse CATEGORY and SUMMARY from Claude's response."""
    category = "ai/general"
    summary = text

    if "CATEGORY:" in text and "SUMMARY:" in text:
        parts = text.split("SUMMARY:", 1)
        cat_line = parts[0].strip()
        summary = parts[1].strip()
        for line in cat_line.split("\n"):
            if line.startswith("CATEGORY:"):
                cat = line.replace("CATEGORY:", "").strip()
                if cat in CATEGORIES:
                    category = cat
                break

    return category, summary


def ingest_youtube(req: YouTubeIngestRequest, *, transcripts_path: str) -> dict:
    """Ingest a YouTube transcript: resolve title, fetch transcript, summarize via Claude, save markdown.

    Returns: {file_path, category, summary, title, url}.
    """
    date = req.date or dt.date.today().isoformat()

    # Chrome extension sometimes sends "YouTube" or the URL before page loads
    title = req.title
    title_lower = title.lower().strip()
    is_generic = title_lower in _GENERIC_TITLES
    is_url = "youtube.com/" in title_lower or "youtu.be/" in title_lower
    if is_generic or is_url:
        video_id = _extract_video_id(req.video_id or req.url)
        real_title = _fetch_youtube_title(video_id)
        if real_title:
            title = real_title
            logger.info(f"Replaced generic title '{req.title}' with '{real_title}'")
        else:
            raise HTTPException(status_code=400, detail=f"Title '{req.title}' is too generic and couldn't fetch real title from YouTube")

    # Pre-made summary (e.g. from javrvis streaming) skips transcript fetch + Claude
    if req.summary:
        summary = req.summary
        category = req.category or "ai/general"
    else:
        transcript = req.transcript
        if not transcript:
            transcript = fetch_transcript(req.video_id or req.url)

        prompt = SUMMARIZE_PROMPT.format(
            categories=", ".join(CATEGORIES),
            title=title,
            transcript=transcript[:100000],
        )
        claude_response = _call_claude_headless(prompt)
        auto_category, summary = _parse_claude_response(claude_response)
        category = req.category or auto_category

    result = write_summary(
        root=transcripts_path,
        title=title,
        url=req.url,
        summary=summary,
        category=category,
        date=date,
    )
    logger.info(f"YouTube ingest: saved {result['file_path']} (category: {result['category']})")

    result["title"] = title
    result["url"] = req.url
    return result


def list_categories(transcripts_path: str) -> list[dict]:
    """List YouTube transcript categories (subdirectories with .md files)."""
    categories = []
    for root, dirs, files in os.walk(transcripts_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "project-notes"]
        rel = os.path.relpath(root, transcripts_path)
        md_count = sum(1 for f in files if f.endswith(".md"))
        if md_count > 0 and rel != ".":
            categories.append({"name": rel, "count": md_count})
    categories.sort(key=lambda c: c["name"])
    return categories
