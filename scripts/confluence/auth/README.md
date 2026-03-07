# Confluence Authentication

This directory contains authentication files for Confluence access.

## Required File

**confluence_auth.json** - Browser session state from Playwright
- Generated automatically on first run
- Contains cookies and authentication state
- **DO NOT commit to git** (already in .gitignore)

## Setup

1. Run a confluence fetcher script for the first time
2. Log in via the browser window that opens
3. The session will be saved to `confluence_auth.json`
4. Subsequent runs will reuse this session
