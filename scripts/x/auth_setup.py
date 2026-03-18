#!/usr/bin/env python3
"""
X/Twitter Auth Setup — interactive browser login + cookie save.

Opens a Chromium browser window so you can log in to X manually
(handles 2FA, captcha, whatever X throws). After login, saves the
browser session state to scripts/x/auth/x_auth.json for reuse by
the fetcher.

Usage:
    uv run scripts/x/auth_setup.py
"""

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

AUTH_FILE = Path(__file__).parent / "auth" / "x_auth.json"


async def main():
    print("=" * 60)
    print("  X/Twitter Auth Setup")
    print("=" * 60)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)

        # Load existing auth if available (allows session refresh)
        storage_state = str(AUTH_FILE) if AUTH_FILE.exists() else None
        context = await browser.new_context(storage_state=storage_state)

        page = await context.new_page()
        await page.goto("https://x.com/login", timeout=60000)

        # Check if already logged in (redirected to home)
        await asyncio.sleep(3)
        if "/home" in page.url:
            print("\nAlready logged in!")
        else:
            print("\n1. Complete the login process in the browser")
            print("2. Handle 2FA / captcha if prompted")
            print("3. Wait until you see your home timeline")
            print("=" * 60)

            # Wait up to 5 minutes for user to complete login
            await page.wait_for_url(
                lambda url: "/home" in url,
                timeout=300000,
            )
            print("\nLogin successful!")

        # Save session state
        AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
        await context.storage_state(path=str(AUTH_FILE))
        print(f"Session saved to {AUTH_FILE}")

        await browser.close()

    print("\nDone. You can now run the fetcher:")
    print("  uv run scripts/x/fetchers/x_fetcher.py")


if __name__ == "__main__":
    asyncio.run(main())
