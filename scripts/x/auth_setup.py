#!/usr/bin/env python3
"""
X/Twitter Auth Setup — extract cookies from your real browser.

Guides you through copying auth_token and ct0 cookies from Chrome
DevTools and saves them for the fetcher. No Playwright needed.

Usage:
    uv run scripts/x/auth_setup.py
"""

import json
import sys
from pathlib import Path

import httpx

AUTH_FILE = Path(__file__).parent / "auth" / "x_auth.json"

# X's public web-app bearer token (embedded in their JavaScript, same for all users)
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA"


def verify_cookies(auth_token: str, ct0: str) -> bool:
    """Make a lightweight API call to verify the cookies work."""
    headers = {
        "authorization": f"Bearer {BEARER_TOKEN}",
        "x-csrf-token": ct0,
        "cookie": f"auth_token={auth_token}; ct0={ct0}",
    }
    try:
        resp = httpx.get(
            "https://api.x.com/1.1/account/verify_credentials.json",
            headers=headers,
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            print(f"\nVerified: logged in as @{data.get('screen_name', '?')}")
            return True
        print(f"\nVerification failed (HTTP {resp.status_code}): {resp.text[:200]}")
        return False
    except Exception as e:
        print(f"\nVerification request failed: {e}")
        return False


def main():
    print("=" * 60)
    print("  X/Twitter Cookie Setup")
    print("=" * 60)
    print()
    print("1. Open Chrome and go to x.com (make sure you're logged in)")
    print("2. Open DevTools: Cmd+Option+I (Mac) or F12 (Windows/Linux)")
    print("3. Go to Application tab -> Cookies -> https://x.com")
    print("4. Find and copy these two cookie values:")
    print()

    auth_token = input("   auth_token: ").strip()
    if not auth_token:
        print("Error: auth_token is required", file=sys.stderr)
        sys.exit(1)

    ct0 = input("   ct0:        ").strip()
    if not ct0:
        print("Error: ct0 is required", file=sys.stderr)
        sys.exit(1)

    # Verify cookies work
    if not verify_cookies(auth_token, ct0):
        print("\nCookies didn't work. Double-check you copied them correctly.")
        print("Make sure you're copying the Value column, not the Name column.")
        sys.exit(1)

    # Save
    AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    AUTH_FILE.write_text(json.dumps({
        "auth_token": auth_token,
        "ct0": ct0,
    }, indent=2))
    print(f"\nSaved to {AUTH_FILE}")
    print("\nYou can now run the fetcher:")
    print("  uv run scripts/x/fetchers/x_fetcher.py")


if __name__ == "__main__":
    main()
