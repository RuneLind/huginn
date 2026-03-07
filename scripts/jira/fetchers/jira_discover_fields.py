#!/usr/bin/env python3
"""
Discover available Jira fields including Epic Link
"""

import asyncio
import json
import os
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

JIRA_BASE_URL = os.environ.get("JIRA_BASE_URL", "https://jira.example.com")
PROJECT_KEY = "MYPROJECT"
AUTH_FILE = "jira_auth.json"


async def discover_fields():
    """Fetch one issue with all fields to discover Epic Link field"""

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)

        # Load saved auth
        storage_state = None
        if os.path.exists(AUTH_FILE):
            print(f"Loading authentication from {AUTH_FILE}")
            with open(AUTH_FILE, 'r') as f:
                storage_state = json.load(f)
        else:
            print("ERROR: No jira_auth.json found. Run jira_fetcher.py first!")
            return

        context = await browser.new_context(storage_state=storage_state)
        page = await context.new_page()

        # Fetch ONE issue with ALL fields
        api_url = (
            f"{JIRA_BASE_URL}/rest/api/2/search?"
            f"jql=project={PROJECT_KEY}&"
            f"maxResults=1&"
            f"fields=*all"
        )

        print("Fetching sample issue with all fields...")
        await page.goto(api_url, wait_until="networkidle", timeout=60000)

        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')
        pre_tag = soup.find('pre')

        if pre_tag:
            json_text = pre_tag.get_text()
        else:
            json_text = content

        data = json.loads(json_text)

        if not data.get('issues'):
            print("No issues found!")
            await browser.close()
            return

        issue = data['issues'][0]
        fields = issue.get('fields', {})

        print("\n" + "=" * 80)
        print(f"SAMPLE ISSUE: {issue['key']}")
        print("=" * 80)

        # Look for Epic-related fields
        epic_fields = []
        parent_fields = []

        print("\nSearching for Epic and Parent fields...\n")

        for field_key, field_value in fields.items():
            field_lower = field_key.lower()

            # Check for Epic-related fields
            if 'epic' in field_lower:
                epic_fields.append((field_key, field_value))
                print(f"EPIC FIELD FOUND: {field_key}")
                if isinstance(field_value, dict):
                    print(f"  Value: {json.dumps(field_value, indent=2)}")
                else:
                    print(f"  Value: {field_value}")
                print()

            # Check for parent fields (for sub-tasks/stories under epics)
            if 'parent' in field_lower:
                parent_fields.append((field_key, field_value))
                print(f"PARENT FIELD FOUND: {field_key}")
                if isinstance(field_value, dict):
                    print(f"  Value: {json.dumps(field_value, indent=2)}")
                else:
                    print(f"  Value: {field_value}")
                print()

        # Show all custom fields for reference
        print("\n" + "=" * 80)
        print("ALL CUSTOM FIELDS (customfield_*):")
        print("=" * 80)

        custom_fields = {k: v for k, v in fields.items() if k.startswith('customfield_')}

        for field_key in sorted(custom_fields.keys()):
            field_value = custom_fields[field_key]

            # Skip empty fields
            if field_value is None or field_value == '' or field_value == []:
                continue

            print(f"\n{field_key}:")
            if isinstance(field_value, (dict, list)):
                print(f"  {json.dumps(field_value, indent=2)[:200]}...")
            else:
                print(f"  {str(field_value)[:200]}")

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY:")
        print("=" * 80)

        if epic_fields:
            print("\nEpic-related fields found:")
            for field_key, _ in epic_fields:
                print(f"  - {field_key}")
        else:
            print("\n⚠️  No Epic fields found with 'epic' in name")
            print("    Check custom fields above - Epic Link is often customfield_10014 or similar")

        if parent_fields:
            print("\nParent-related fields found:")
            for field_key, _ in parent_fields:
                print(f"  - {field_key}")

        print("\nLook for fields containing epic names or links above.")
        print("Common Epic Link field names: customfield_10014, customfield_10008, customfield_10100")

        # Check if customfield_13510 is the Epic Link
        if 'customfield_13510' in fields and fields['customfield_13510']:
            epic_key = fields['customfield_13510']
            print("\n" + "=" * 80)
            print(f"CHECKING customfield_13510: {epic_key}")
            print("=" * 80)

            # Fetch the referenced issue
            epic_api_url = f"{JIRA_BASE_URL}/rest/api/2/issue/{epic_key}?fields=issuetype,summary"

            epic_page = await context.new_page()
            await epic_page.goto(epic_api_url, wait_until="networkidle", timeout=60000)

            epic_content = await epic_page.content()
            epic_soup = BeautifulSoup(epic_content, 'html.parser')
            epic_pre = epic_soup.find('pre')

            if epic_pre:
                epic_json_text = epic_pre.get_text()
            else:
                epic_json_text = epic_content

            epic_data = json.loads(epic_json_text)

            issue_type = epic_data.get('fields', {}).get('issuetype', {}).get('name', '')
            summary = epic_data.get('fields', {}).get('summary', '')

            print(f"\nIssue Type: {issue_type}")
            print(f"Summary: {summary}")

            if issue_type.lower() == 'epic':
                print("\n✓ CONFIRMED: customfield_13510 is the Epic Link field!")
            else:
                print(f"\n⚠️  Not an Epic, it's a {issue_type}")

            await epic_page.close()

        await browser.close()


if __name__ == "__main__":
    asyncio.run(discover_fields())