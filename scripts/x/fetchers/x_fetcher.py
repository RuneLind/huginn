#!/usr/bin/env python3
"""
X/Twitter Timeline Fetcher — Playwright + GraphQL interception.

Loads auth cookies, navigates to x.com/home, intercepts GraphQL
timeline responses, and outputs structured tweet JSON to stdout.

Usage:
    # Fetch latest ~50 tweets from home timeline
    uv run scripts/x/fetchers/x_fetcher.py

    # Fetch more (scroll N times)
    uv run scripts/x/fetchers/x_fetcher.py --scrolls 5

    # Output to file instead of stdout
    uv run scripts/x/fetchers/x_fetcher.py --output data/x/timeline.json

    # Save as markdown files (for indexing into huginn collections)
    uv run scripts/x/fetchers/x_fetcher.py --saveMd data/sources/x-timeline
"""

import asyncio
import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from playwright.async_api import async_playwright, Response

logger = logging.getLogger(__name__)

AUTH_FILE = Path(__file__).resolve().parent.parent / "auth" / "x_auth.json"

# GraphQL endpoints that contain timeline data
TIMELINE_ENDPOINTS = ("HomeTimeline", "HomeLatestTimeline")


# ---------------------------------------------------------------------------
# Tweet extraction from GraphQL response
# ---------------------------------------------------------------------------

def extract_tweets(graphql_data: dict) -> list[dict]:
    """Walk a GraphQL timeline response and extract tweet objects."""
    tweets = []
    instructions = _dig(graphql_data, "data", "home", "home_timeline_urt", "instructions")
    if not instructions:
        # Try alternative response shapes
        instructions = _dig(graphql_data, "data", "timeline_by_id", "timeline", "instructions")
    if not instructions:
        return tweets

    for instruction in instructions:
        entries = instruction.get("entries", [])
        for entry in entries:
            tweet = _extract_tweet_from_entry(entry)
            if tweet:
                tweets.append(tweet)

    return tweets


def _extract_tweet_from_entry(entry: dict) -> dict | None:
    """Extract a single tweet dict from a timeline entry."""
    content = entry.get("content", {})

    # Regular tweet entries
    item_content = content.get("itemContent") or _dig(content, "items", 0, "item", "itemContent")
    if not item_content:
        return None

    tweet_results = item_content.get("tweet_results", {})
    result = tweet_results.get("result", {})

    # Handle "TweetWithVisibilityResults" wrapper
    if result.get("__typename") == "TweetWithVisibilityResults":
        result = result.get("tweet", {})

    if result.get("__typename") not in ("Tweet", None):
        if result.get("__typename") == "TweetTombstone":
            return None

    return _parse_tweet_result(result)


def _parse_tweet_result(result: dict) -> dict | None:
    """Parse a tweet result object into our output format."""
    core = result.get("core", {})
    user_results = _dig(core, "user_results", "result") or {}
    legacy_user = user_results.get("legacy", {})
    legacy_tweet = result.get("legacy", {})

    tweet_id = legacy_tweet.get("id_str") or result.get("rest_id")
    if not tweet_id:
        return None

    full_text = legacy_tweet.get("full_text", "")
    screen_name = legacy_user.get("screen_name", "")

    # Expand t.co URLs in text
    full_text = _expand_urls(full_text, legacy_tweet.get("entities", {}))

    # Check if this is a retweet
    is_retweet = False
    retweeted_status = legacy_tweet.get("retweeted_status_result", {}).get("result")
    if retweeted_status:
        is_retweet = True
        # For retweets, parse the original tweet for the text
        original = _parse_tweet_result(retweeted_status)
        if original:
            full_text = f"RT @{original['handle']}: {original['text']}"

    # Quoted tweet
    quoted_tweet = None
    quoted_result = result.get("quoted_status_result", {}).get("result")
    if quoted_result:
        quoted_tweet = _parse_tweet_result(quoted_result)

    # Media
    media = []
    for m in legacy_tweet.get("entities", {}).get("media", []):
        media.append({
            "type": m.get("type", "photo"),
            "url": m.get("media_url_https") or m.get("url", ""),
        })
    # Extended media (videos)
    for m in legacy_tweet.get("extended_entities", {}).get("media", []):
        if m.get("type") == "video":
            variants = m.get("video_info", {}).get("variants", [])
            best = max(
                (v for v in variants if v.get("content_type") == "video/mp4"),
                key=lambda v: v.get("bitrate", 0),
                default=None,
            )
            if best:
                media.append({"type": "video", "url": best["url"]})

    return {
        "id": tweet_id,
        "author": legacy_user.get("name", ""),
        "handle": screen_name,
        "text": full_text,
        "created_at": legacy_tweet.get("created_at", ""),
        "url": f"https://x.com/{screen_name}/status/{tweet_id}",
        "likes": legacy_tweet.get("favorite_count", 0),
        "retweets": legacy_tweet.get("retweet_count", 0),
        "replies": legacy_tweet.get("reply_count", 0),
        "is_retweet": is_retweet,
        "quoted_tweet": quoted_tweet,
        "media": media if media else None,
    }


def _expand_urls(text: str, entities: dict) -> str:
    """Replace t.co URLs with their expanded form."""
    for url_entity in entities.get("urls", []):
        short = url_entity.get("url", "")
        expanded = url_entity.get("expanded_url", short)
        if short:
            text = text.replace(short, expanded)
    return text


def _dig(data: Any, *keys: Any) -> Any:
    """Safely traverse nested dicts/lists."""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key)
        elif isinstance(data, (list, tuple)) and isinstance(key, int) and key < len(data):
            data = data[key]
        else:
            return None
    return data


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def save_tweets_as_markdown(tweets: list[dict], output_dir: str):
    """Save each tweet as a markdown file."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)

    for tweet in tweets:
        handle = tweet["handle"]
        tweet_id = tweet["id"]
        filename = f"{handle}_{tweet_id}.md"
        filepath = base / filename

        lines = [
            f"# @{handle} — {tweet['author']}",
            "",
            tweet["text"],
            "",
            "---",
            "",
            f"- **Date:** {tweet['created_at']}",
            f"- **Likes:** {tweet['likes']}  **Retweets:** {tweet['retweets']}  **Replies:** {tweet['replies']}",
            f"- **URL:** {tweet['url']}",
        ]

        if tweet.get("is_retweet"):
            lines.append("- **Type:** Retweet")

        if tweet.get("quoted_tweet"):
            qt = tweet["quoted_tweet"]
            lines.extend([
                "",
                f"> **Quoted @{qt['handle']}:** {qt['text']}",
            ])

        if tweet.get("media"):
            lines.append("")
            for m in tweet["media"]:
                lines.append(f"- [{m['type']}]({m['url']})")

        filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved {len(tweets)} tweets to {base}/", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main fetcher
# ---------------------------------------------------------------------------

async def fetch_timeline(scrolls: int = 2) -> list[dict]:
    """Launch browser, intercept GraphQL responses, return tweets."""
    if not AUTH_FILE.exists():
        print(
            f"Error: Auth file not found at {AUTH_FILE}\n"
            "Run auth setup first: uv run scripts/x/auth_setup.py",
            file=sys.stderr,
        )
        sys.exit(1)

    seen_ids: set[str] = set()
    tweets: list[dict] = []

    async def handle_response(response: Response):
        url = response.url
        if not any(ep in url for ep in TIMELINE_ENDPOINTS):
            return
        try:
            data = await response.json()
            for tweet in extract_tweets(data):
                if tweet["id"] not in seen_ids:
                    seen_ids.add(tweet["id"])
                    tweets.append(tweet)
        except Exception:
            pass  # Non-JSON or broken response, skip

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            storage_state=str(AUTH_FILE),
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
        )

        page = await context.new_page()
        page.on("response", handle_response)

        print("Navigating to x.com/home ...", file=sys.stderr)
        await page.goto("https://x.com/home", wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(5000)  # Let initial timeline load

        # Check if we got redirected to login
        if "login" in page.url.lower():
            print(
                "Error: Session expired — redirected to login.\n"
                "Re-run auth setup: uv run scripts/x/auth_setup.py",
                file=sys.stderr,
            )
            await browser.close()
            sys.exit(1)

        # Scroll to load more tweets
        for i in range(scrolls):
            print(f"Scrolling ({i + 1}/{scrolls}) — {len(tweets)} tweets so far ...", file=sys.stderr)
            await page.keyboard.press("End")
            await page.wait_for_timeout(2000)

        # Save auth state (session refresh)
        await context.storage_state(path=str(AUTH_FILE))

        await browser.close()

    print(f"Fetched {len(tweets)} tweets total.", file=sys.stderr)
    return tweets


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch X/Twitter home timeline")
    parser.add_argument(
        "--scrolls", type=int, default=2,
        help="Number of scroll actions to load more tweets (default: 2)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Write JSON output to this file instead of stdout",
    )
    parser.add_argument(
        "--saveMd", type=str, default=None,
        help="Save tweets as markdown files to this directory",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    tweets = await fetch_timeline(scrolls=args.scrolls)

    if args.saveMd:
        save_tweets_as_markdown(tweets, args.saveMd)

    json_output = json.dumps(tweets, indent=2, ensure_ascii=False)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_output, encoding="utf-8")
        print(f"Wrote JSON to {out_path}", file=sys.stderr)
    else:
        print(json_output)


if __name__ == "__main__":
    asyncio.run(main())
