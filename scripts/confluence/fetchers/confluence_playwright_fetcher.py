import argparse
import asyncio
import json
from playwright.async_api import async_playwright
from pathlib import Path

async def fetch_confluence_pages(space_key: str, output_dir: str, base_url: str):
    """Fetch Confluence pages using Playwright with SSO authentication"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        # Launch browser (use chromium with persistent context to save session)
        browser = await p.chromium.launch(headless=False)  # Set to True after first login
        context = await browser.new_context(
            storage_state="confluence_auth.json" if Path("confluence_auth.json").exists() else None
        )
        page = await context.new_page()

        # Navigate to Confluence space
        space_url = f"{base_url}/spaces/{space_key}/overview"
        await page.goto(space_url)

        # Wait for user to login if needed (first time only)
        print("Please login if prompted...")
        await page.wait_for_url("**/spaces/**", timeout=120000)  # Wait up to 2 minutes

        # Save authentication state
        await context.storage_state(path="confluence_auth.json")
        print("Authentication saved")

        # Now fetch pages via REST API using the authenticated session
        # Get cookies from browser context
        cookies = await context.cookies()
        cookie_str = "; ".join([f"{c['name']}={c['value']}" for c in cookies])

        # Use these cookies for API calls
        api_url = f"{base_url}/rest/api/content/search"

        # Navigate to API endpoint to get JSON
        response = await page.request.get(
            api_url,
            params={
                'cql': f"space = '{space_key}' AND type = page",
                'limit': 100,
                'expand': 'body.storage,version'
            }
        )

        data = await response.json()

        # Save pages
        for result in data.get('results', []):
            page_id = result['id']
            title = result['title']
            content = result['body']['storage']['value']

            page_file = output_path / f"{page_id}.json"
            page_file.write_text(json.dumps({
                'id': page_id,
                'title': title,
                'content': content,
                'url': f"{base_url}/spaces/{space_key}/pages/{page_id}"
            }, indent=2))

            print(f"Saved: {title}")

        await browser.close()

        return len(data.get('results', []))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Confluence pages using Playwright")
    parser.add_argument("--space", "-s", required=True, help="Confluence space key")
    parser.add_argument("--output", "-o", default="./confluence_pages", help="Output directory")
    parser.add_argument("--base-url", "-u", required=True,
                        help="Confluence base URL (e.g. https://confluence.example.com)")
    args = parser.parse_args()
    asyncio.run(fetch_confluence_pages(args.space, args.output, args.base_url))
