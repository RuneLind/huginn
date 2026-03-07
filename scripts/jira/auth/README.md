# Jira Authentication

This directory contains authentication files for Jira access.

## Required File

**jira_auth.json** - Browser session state from Playwright
- Generated automatically on first run
- Contains cookies and authentication state
- **DO NOT commit to git** (already in .gitignore)

## Setup

1. Run the jira fetcher script for the first time
2. Log in via the browser window that opens
3. The session will be saved to `jira_auth.json`
4. Subsequent runs will reuse this session
