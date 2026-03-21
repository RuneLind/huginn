#!/usr/bin/env python3
"""
X/Twitter Timeline Fetcher — direct HTTP with cookie auth.

Uses cookies extracted from your real browser to call X's GraphQL
API directly. No Playwright / browser automation — just HTTP requests.

Usage:
    # Fetch home timeline (~20 tweets per page)
    uv run scripts/x/fetchers/x_fetcher.py

    # Fetch more (multiple pages)
    uv run scripts/x/fetchers/x_fetcher.py --pages 3

    # Output to file instead of stdout
    uv run scripts/x/fetchers/x_fetcher.py --output data/x/timeline.json

    # Save as markdown files (for indexing into huginn collections)
    uv run scripts/x/fetchers/x_fetcher.py --saveMd data/sources/x-timeline
"""

import asyncio
import argparse
import json
import os
import re
import sys
from datetime import datetime as _dt
from pathlib import Path
from typing import Any

import httpx

AUTH_FILE = Path(__file__).resolve().parent.parent / "auth" / "x_auth.json"

# X's public web-app bearer token (embedded in their JS, same for all users)
BEARER_TOKEN = (
    "AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs"
    "%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA"
)

# GraphQL features required by the HomeTimeline query.
# These change occasionally when X ships new features.
TIMELINE_FEATURES = {
    "rweb_tipjar_consumption_enabled": True,
    "responsive_web_graphql_exclude_directive_enabled": True,
    "verified_phone_label_enabled": False,
    "creator_subscriptions_tweet_preview_api_enabled": True,
    "responsive_web_graphql_timeline_navigation_enabled": True,
    "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
    "communities_web_enable_tweet_community_results_fetch": True,
    "c9s_tweet_anatomy_moderator_badge_enabled": True,
    "articles_preview_enabled": True,
    "responsive_web_edit_tweet_api_enabled": True,
    "graphql_is_translatable_rweb_tweet_is_translatable_enabled": True,
    "view_counts_everywhere_api_enabled": True,
    "longform_notetweets_consumption_enabled": True,
    "responsive_web_twitter_article_tweet_consumption_enabled": True,
    "tweet_awards_web_tipping_enabled": False,
    "creator_subscriptions_quote_tweet_preview_enabled": False,
    "freedom_of_speech_not_reach_fetch_enabled": True,
    "standardized_nudges_misinfo": True,
    "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": True,
    "rweb_video_timestamps_enabled": True,
    "longform_notetweets_rich_text_read_enabled": True,
    "longform_notetweets_inline_media_enabled": True,
    "responsive_web_enhance_cards_enabled": False,
}


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def load_auth() -> tuple[str, str]:
    """Load auth_token and ct0 from the auth file."""
    if not AUTH_FILE.exists():
        print(
            f"Error: Auth file not found at {AUTH_FILE}\n"
            "Run auth setup first: uv run scripts/x/auth_setup.py",
            file=sys.stderr,
        )
        sys.exit(1)

    data = json.loads(AUTH_FILE.read_text())
    auth_token = data.get("auth_token")
    ct0 = data.get("ct0")

    if not auth_token or not ct0:
        print("Error: auth_token or ct0 missing from auth file.", file=sys.stderr)
        sys.exit(1)

    return auth_token, ct0


def build_headers(auth_token: str, ct0: str) -> dict[str, str]:
    """Build request headers mimicking X's web client."""
    return {
        "authorization": f"Bearer {BEARER_TOKEN}",
        "x-csrf-token": ct0,
        "cookie": f"auth_token={auth_token}; ct0={ct0}",
        "user-agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "x-twitter-auth-type": "OAuth2Session",
        "x-twitter-active-user": "yes",
        "x-twitter-client-language": "en",
    }


# ---------------------------------------------------------------------------
# GraphQL query ID discovery
# ---------------------------------------------------------------------------

async def discover_query_id(operation: str = "HomeTimeline") -> str:
    """Extract a GraphQL query ID from X's JavaScript bundles.

    Uses a separate unauthenticated client — JS bundles are public
    and sending auth headers to x.com's main page causes a 401.
    """
    public_headers = {
        "user-agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
    }

    async with httpx.AsyncClient(headers=public_headers, timeout=30, follow_redirects=True) as pub:
        resp = await pub.get("https://x.com")
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to fetch x.com: HTTP {resp.status_code}")

        # Find JS bundle URLs — X serves different bundle names over time
        js_urls: list[str] = re.findall(
            r'src="(https://abs\.twimg\.com/responsive-web/client-web[^/]*/main\.[a-f0-9]+\.js)"',
            resp.text,
        )
        if not js_urls:
            js_urls = re.findall(
                r'src="(https://abs\.twimg\.com/responsive-web/client-web[^"]*\.js)"',
                resp.text,
            )

        # Prioritize api/endpoints bundles that are more likely to contain query IDs
        api_urls = re.findall(
            r'src="(https://abs\.twimg\.com/responsive-web/client-web[^"]*(?:api|endpoints)[^"]*\.js)"',
            resp.text,
        )
        js_urls = api_urls + js_urls

        for js_url in js_urls[:10]:
            js_resp = await pub.get(js_url)
            if js_resp.status_code != 200:
                continue

            # Pattern: {queryId:"ABC123",operationName:"HomeTimeline",operationType:"query"}
            pattern = rf'queryId:"([^"]+)",operationName:"{re.escape(operation)}"'
            match = re.search(pattern, js_resp.text)
            if match:
                return match.group(1)

            pattern2 = rf"queryId:'([^']+)',operationName:'{re.escape(operation)}'"
            match2 = re.search(pattern2, js_resp.text)
            if match2:
                return match2.group(1)

    raise RuntimeError(
        f"Could not find queryId for {operation} in X's JS bundles. "
        "X may have changed their JS structure."
    )


# ---------------------------------------------------------------------------
# Tweet extraction from GraphQL response
# ---------------------------------------------------------------------------

def extract_tweets(graphql_data: dict) -> list[dict]:
    """Walk a GraphQL timeline response and extract tweet objects."""
    tweets = []
    instructions = _dig(graphql_data, "data", "home", "home_timeline_urt", "instructions")
    if not instructions:
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


def extract_cursor(graphql_data: dict, cursor_type: str = "Bottom") -> str | None:
    """Find the pagination cursor from a timeline response."""
    instructions = _dig(graphql_data, "data", "home", "home_timeline_urt", "instructions")
    if not instructions:
        instructions = _dig(graphql_data, "data", "timeline_by_id", "timeline", "instructions")
    if not instructions:
        return None

    for instruction in instructions:
        for entry in instruction.get("entries", []):
            content = entry.get("content", {})
            if content.get("cursorType") == cursor_type:
                return content.get("value")
            # Nested cursor format
            entry_type = content.get("entryType") or content.get("__typename")
            if entry_type == "TimelineTimelineCursor" and content.get("cursorType") == cursor_type:
                return content.get("value")

    return None


def _extract_tweet_from_entry(entry: dict) -> dict | None:
    """Extract a single tweet dict from a timeline entry."""
    content = entry.get("content", {})

    item_content = content.get("itemContent") or _dig(content, "items", 0, "item", "itemContent")
    if not item_content:
        return None

    tweet_results = item_content.get("tweet_results", {})
    result = tweet_results.get("result", {})

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
    # X moved name/screen_name from legacy into user.core
    user_core = user_results.get("core", {})
    legacy_user = user_results.get("legacy", {})
    legacy_tweet = result.get("legacy", {})

    tweet_id = legacy_tweet.get("id_str") or result.get("rest_id")
    if not tweet_id:
        return None

    full_text = legacy_tweet.get("full_text", "")
    screen_name = user_core.get("screen_name") or legacy_user.get("screen_name", "")

    # Long-form note tweets override legacy full_text
    tweet_type = "tweet"
    note_tweet = result.get("note_tweet", {})
    if note_tweet:
        note_text = _dig(note_tweet, "note_tweet_results", "result", "text")
        if note_text:
            full_text = note_text
            tweet_type = "note"

    full_text = _expand_urls(full_text, legacy_tweet.get("entities", {}))

    # View count (top-level, not in legacy)
    views = 0
    views_raw = _dig(result, "views", "count")
    if views_raw:
        try:
            views = int(views_raw)
        except (ValueError, TypeError):
            pass

    # Retweet
    is_retweet = False
    retweeted_status = legacy_tweet.get("retweeted_status_result", {}).get("result")
    if retweeted_status:
        is_retweet = True
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
        "author": user_core.get("name") or legacy_user.get("name", ""),
        "handle": screen_name,
        "text": full_text,
        "created_at": legacy_tweet.get("created_at", ""),
        "url": f"https://x.com/{screen_name}/status/{tweet_id}",
        "likes": legacy_tweet.get("favorite_count", 0),
        "retweets": legacy_tweet.get("retweet_count", 0),
        "replies": legacy_tweet.get("reply_count", 0),
        "views": views,
        "bookmarks": legacy_tweet.get("bookmark_count", 0),
        "tweet_type": tweet_type,
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
# Markdown helpers
# ---------------------------------------------------------------------------

def _parse_x_date(created_at: str) -> str:
    """Parse X's created_at into YYYY-MM-DD. Returns '' on failure."""
    try:
        dt = _dt.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return ""


def _yaml_value(value: str) -> str:
    """Quote a YAML value if it contains special characters."""
    if not value:
        return '""'
    if any(c in value for c in (':', '#', '{', '}', '[', ']', ',', '&', '*', '?', '|', '-', '<', '>', '=', '!', '%', '@', '`', '"', "'", '\\')):
        escaped = value.replace('\\', '\\\\').replace('"', '\\"')
        return f'"{escaped}"'
    return value


def _build_tweet_frontmatter(tweet: dict) -> str:
    """Build YAML frontmatter with fields the huginn indexer preserves."""
    handle = tweet["handle"]
    text_preview = tweet["text"][:100].replace("\n", " ")
    date_str = _parse_x_date(tweet["created_at"])

    lines = ["---"]
    lines.append(f"title: {_yaml_value(f'@{handle} — {text_preview}')}")
    lines.append(f"url: {tweet['url']}")
    if date_str:
        lines.append(f"date: {date_str}")
    lines.append("category: x-feed")
    lines.append("---\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def save_tweets_as_markdown(
    tweets: list[dict],
    output_dir: str,
    skip_existing: bool = False,
    include_retweets: bool = False,
):
    """Save each tweet as a markdown file with YAML frontmatter.

    Filenames: {date}_{handle}_{tweet_id}.md
    Skips retweets by default. With skip_existing, scans output_dir
    for existing tweet IDs in filenames and skips those.
    """
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)

    # Build set of existing tweet IDs from filenames
    existing_ids: set[str] = set()
    if skip_existing:
        for f in base.glob("*.md"):
            # tweet_id is always the last underscore-separated segment
            parts = f.stem.rsplit("_", 1)
            if len(parts) == 2:
                existing_ids.add(parts[-1])

    saved = 0
    skipped_existing = 0
    skipped_retweet = 0

    for tweet in tweets:
        tweet_id = tweet["id"]

        if tweet.get("is_retweet") and not include_retweets:
            skipped_retweet += 1
            continue

        if skip_existing and tweet_id in existing_ids:
            skipped_existing += 1
            continue

        handle = tweet["handle"]
        date_str = _parse_x_date(tweet["created_at"])
        prefix = f"{date_str}_" if date_str else ""
        filename = f"{prefix}{handle}_{tweet_id}.md"
        filepath = base / filename

        frontmatter = _build_tweet_frontmatter(tweet)

        body_lines = [
            f"# @{handle} — {tweet['author']}",
            "",
            tweet["text"],
        ]

        if tweet.get("quoted_tweet"):
            qt = tweet["quoted_tweet"]
            body_lines.extend([
                "",
                f"> **Quoted @{qt['handle']}:** {qt['text']}",
            ])

        if tweet.get("media"):
            body_lines.append("")
            for m in tweet["media"]:
                body_lines.append(f"- [{m['type']}]({m['url']})")

        body_lines.extend([
            "",
            "---",
            "",
            f"- **Engagement:** {tweet['likes']:,} likes · {tweet['retweets']:,} retweets · {tweet.get('views', 0):,} views · {tweet.get('bookmarks', 0):,} bookmarks",
            f"- **Date:** {tweet['created_at']}",
            f"- **Type:** {tweet.get('tweet_type', 'tweet')}",
            f"- **Link:** {tweet['url']}",
        ])

        content = frontmatter + "\n".join(body_lines) + "\n"
        filepath.write_text(content, encoding="utf-8")

        # Set file mtime to tweet timestamp for incremental update support
        if date_str:
            try:
                ts = _dt.strptime(tweet["created_at"], "%a %b %d %H:%M:%S %z %Y").timestamp()
                os.utime(filepath, (ts, ts))
            except (ValueError, OSError):
                pass

        saved += 1

    print(f"Saved {saved} tweets to {base}/", file=sys.stderr)
    if skipped_existing:
        print(f"  Skipped {skipped_existing} (already existing)", file=sys.stderr)
    if skipped_retweet:
        print(f"  Skipped {skipped_retweet} retweets", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main fetcher
# ---------------------------------------------------------------------------

async def fetch_timeline(pages: int = 1) -> list[dict]:
    """Fetch home timeline via direct HTTP requests."""
    auth_token, ct0 = load_auth()
    headers = build_headers(auth_token, ct0)

    seen_ids: set[str] = set()
    all_tweets: list[dict] = []
    cursor: str | None = None

    async with httpx.AsyncClient(headers=headers, timeout=30) as client:
        # Discover the current GraphQL query ID
        print("Discovering GraphQL query ID ...", file=sys.stderr)
        query_id = await discover_query_id("HomeTimeline")
        print(f"Found query ID: {query_id}", file=sys.stderr)

        for page_num in range(pages):
            variables: dict[str, Any] = {
                "count": 20,
                "includePromotedContent": True,
                "latestControlAvailable": True,
                "requestContext": "launch",
                "withCommunity": True,
            }
            if cursor:
                variables["cursor"] = cursor

            params = {
                "variables": json.dumps(variables, separators=(",", ":")),
                "features": json.dumps(TIMELINE_FEATURES, separators=(",", ":")),
            }

            url = f"https://x.com/i/api/graphql/{query_id}/HomeTimeline"
            print(f"Fetching page {page_num + 1}/{pages} ...", file=sys.stderr)

            resp = await client.get(url, params=params)

            if resp.status_code == 401:
                print(
                    "Error: Unauthorized (401) — session expired.\n"
                    "Re-run auth setup: uv run scripts/x/auth_setup.py",
                    file=sys.stderr,
                )
                sys.exit(1)

            if resp.status_code == 429:
                print("Error: Rate limited (429). Try again later.", file=sys.stderr)
                sys.exit(1)

            if resp.status_code != 200:
                print(
                    f"Error: HTTP {resp.status_code}\n{resp.text[:500]}",
                    file=sys.stderr,
                )
                sys.exit(1)

            data = resp.json()
            page_tweets = extract_tweets(data)

            new_count = 0
            for tweet in page_tweets:
                if tweet["id"] not in seen_ids:
                    seen_ids.add(tweet["id"])
                    all_tweets.append(tweet)
                    new_count += 1

            print(f"  Got {new_count} new tweets (total: {len(all_tweets)})", file=sys.stderr)

            if new_count == 0:
                print("  No new tweets, stopping pagination.", file=sys.stderr)
                break

            # Get cursor for next page
            cursor = extract_cursor(data)
            if not cursor:
                print("  No cursor found, stopping pagination.", file=sys.stderr)
                break

    return all_tweets


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch X/Twitter home timeline")
    parser.add_argument(
        "--pages", type=int, default=1,
        help="Number of pages to fetch (default: 1, ~20 tweets each)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Write JSON output to this file instead of stdout",
    )
    parser.add_argument(
        "--saveMd", type=str, default=None,
        help="Save tweets as markdown files to this directory",
    )
    parser.add_argument(
        "--skipExisting", action="store_true", default=False,
        help="Skip tweets already saved as markdown (by tweet ID in filename)",
    )
    parser.add_argument(
        "--includeRetweets", action="store_true", default=False,
        help="Include retweets in markdown output (skipped by default)",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    tweets = await fetch_timeline(pages=args.pages)

    if args.saveMd:
        save_tweets_as_markdown(
            tweets, args.saveMd,
            skip_existing=args.skipExisting,
            include_retweets=args.includeRetweets,
        )

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
