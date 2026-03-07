#!/usr/bin/env python3
"""
Check for updates in a Confluence space since the last fetch.

This script reads the fetch_metadata.json from a previous fetch and queries
Confluence to see if any pages have been modified since then.

Usage:
    uv run confluence_check_updates.py --space MYSPACE
    uv run confluence_check_updates.py --space MYSPACE --show-pages
"""

import asyncio
import json
import argparse
from datetime import datetime, timezone
from playwright.async_api import async_playwright
from pathlib import Path
from typing import Optional


def get_default_output_dir() -> str:
    """Get the default output directory relative to project root"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    return str(project_root / "data" / "downloaded" / "confluence_hierarchical")


def load_fetch_metadata(output_dir: str) -> Optional[dict]:
    """Load the fetch metadata from a previous fetch"""
    metadata_file = Path(output_dir) / "fetch_metadata.json"
    if not metadata_file.exists():
        return None
    return json.loads(metadata_file.read_text(encoding='utf-8'))


def format_datetime(iso_string: str) -> str:
    """Format ISO datetime string for display"""
    dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")


def format_confluence_date(iso_string: str) -> str:
    """Format ISO datetime for Confluence CQL query (YYYY-MM-DD)"""
    dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
    return dt.strftime("%Y-%m-%d")


async def check_for_updates(
    base_url: str,
    space_key: str,
    since_date: str,
    show_pages: bool = False
) -> dict:
    """Query Confluence for pages modified since the given date"""

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)

        storage_state = None
        auth_file = Path(__file__).parent.parent / "auth" / "confluence_auth.json"
        if auth_file.exists():
            storage_state = str(auth_file)

        context = await browser.new_context(storage_state=storage_state)
        page = await context.new_page()

        # Navigate to space to trigger auth
        space_url = f"{base_url}/spaces/{space_key}/overview"
        await page.goto(space_url)

        try:
            await page.wait_for_url("**/spaces/**", timeout=120000)
            print("✅ Authenticated successfully")
        except:
            print("❌ Authentication timeout")
            await browser.close()
            return {"error": "Authentication failed"}

        # Save auth state
        auth_file.parent.mkdir(parents=True, exist_ok=True)
        await context.storage_state(path=str(auth_file))

        # Query for modified pages
        cql = f"space = '{space_key}' AND type = page AND lastModified >= '{since_date}'"

        api_url = f"{base_url}/rest/api/content/search"

        modified_pages = []
        start = 0
        limit = 50

        while True:
            try:
                response = await page.request.get(
                    api_url,
                    params={
                        'cql': cql,
                        'limit': limit,
                        'start': start,
                        'expand': 'version'
                    }
                )

                if response.status != 200:
                    print(f"❌ API returned status {response.status}")
                    break

                data = json.loads(await response.text())
                results = data.get('results', [])

                if not results:
                    break

                modified_pages.extend(results)

                if len(results) < limit:
                    break

                start += limit

            except Exception as e:
                print(f"❌ Error querying Confluence: {e}")
                break

        await browser.close()

        # Also get total page count
        total_pages = await get_total_page_count(base_url, space_key)

        return {
            "modified_pages": modified_pages,
            "total_in_space": total_pages,
            "since_date": since_date
        }


async def get_total_page_count(base_url: str, space_key: str) -> int:
    """Get the total number of pages in the space"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        storage_state = None
        auth_file = Path(__file__).parent.parent / "auth" / "confluence_auth.json"
        if auth_file.exists():
            storage_state = str(auth_file)

        context = await browser.new_context(storage_state=storage_state)
        page = await context.new_page()

        api_url = f"{base_url}/rest/api/content/search"

        try:
            response = await page.request.get(
                api_url,
                params={
                    'cql': f"space = '{space_key}' AND type = page",
                    'limit': 1
                }
            )

            if response.status == 200:
                data = json.loads(await response.text())
                await browser.close()
                return data.get('totalSize', 0)

        except Exception:
            pass

        await browser.close()
        return 0


async def main():
    parser = argparse.ArgumentParser(
        description="Check for Confluence updates since last fetch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check for updates in MYSPACE space
  uv run confluence_check_updates.py --space MYSPACE

  # Show which pages were modified
  uv run confluence_check_updates.py --space MYSPACE --show-pages

  # Check custom output directory
  uv run confluence_check_updates.py --space MYSPACE --output ./my_confluence
        """
    )

    parser.add_argument(
        "--space", "-s",
        default="MYSPACE",
        help="Confluence space key (default: MYSPACE)"
    )

    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory containing fetch_metadata.json"
    )

    parser.add_argument(
        "--base-url", "-u",
        required=True,
        help="Confluence base URL (e.g. https://confluence.example.com)"
    )

    parser.add_argument(
        "--show-pages", "-p",
        action="store_true",
        help="Show list of modified pages"
    )

    args = parser.parse_args()

    output_dir = args.output if args.output else get_default_output_dir()

    print(f"🔍 Confluence Update Checker")
    print(f"   Space: {args.space}")
    print(f"   Output dir: {output_dir}")
    print()

    # Load previous fetch metadata
    metadata = load_fetch_metadata(output_dir)

    if not metadata:
        print("❌ No previous fetch found (fetch_metadata.json not found)")
        print(f"   Expected location: {output_dir}/fetch_metadata.json")
        print()
        print("   Run the fetcher first to create initial data:")
        print(f"   uv run confluence_fetcher_hierarchical.py --space {args.space}")
        return

    fetch_time = metadata.get("fetch_time")
    previous_count = metadata.get("total_pages", 0)

    print(f"📋 Previous fetch:")
    print(f"   Time: {format_datetime(fetch_time)}")
    print(f"   Pages: {previous_count}")
    print()

    # Query for updates
    since_date = format_confluence_date(fetch_time)
    print(f"🔎 Checking for updates since {since_date}...")
    print()

    result = await check_for_updates(
        base_url=args.base_url,
        space_key=args.space,
        since_date=since_date,
        show_pages=args.show_pages
    )

    if "error" in result:
        print(f"❌ {result['error']}")
        return

    modified_pages = result["modified_pages"]
    total_in_space = result["total_in_space"]

    print()
    print("=" * 50)

    if not modified_pages:
        print("✅ No updates found - your local copy is up to date!")
    else:
        print(f"📊 Updates found:")
        print(f"   Modified pages: {len(modified_pages)}")
        print(f"   Total in space: {total_in_space}")

        if total_in_space != previous_count:
            diff = total_in_space - previous_count
            sign = "+" if diff > 0 else ""
            print(f"   Page count change: {sign}{diff} (was {previous_count})")

        if args.show_pages:
            print()
            print("📄 Modified pages:")
            for page in sorted(modified_pages, key=lambda p: p.get('version', {}).get('when', ''), reverse=True):
                title = page.get('title', 'Unknown')
                version = page.get('version', {})
                when = version.get('when', 'Unknown')
                by = version.get('by', {}).get('displayName', 'Unknown')
                print(f"   - {title}")
                print(f"     Modified: {when} by {by}")

        print()
        print("💡 To update, run:")
        print(f"   uv run confluence_fetcher_hierarchical.py --space {args.space}")

    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
