#!/usr/bin/env python3
"""
Jira Fetcher with Playwright Authentication.

Fetches Jira issues via REST API using Playwright browser context for auth.
Supports incremental updates, curated markdown output, and exclude manifests.

Works with both Atlassian Cloud (*.atlassian.net) and self-hosted Server/Data
Center, auto-detected from --base-url. On Cloud it uses the enhanced
/rest/api/3/search/jql endpoint (token pagination), reads ADF rich-text, and
takes the Epic from each issue's parent; on Server it uses the classic
offset-paginated /search with the legacy Epic Link custom field.

Usage:
    # Full download from Atlassian Cloud (e.g. NAV's MELOSYS project)
    uv run scripts/jira/fetchers/jira_fetcher.py \\
        --base-url https://nav.atlassian.net --project MELOSYS \\
        --saveMd ./data/sources/jira-issues

    # Incremental: only issues updated since cutoff
    uv run scripts/jira/fetchers/jira_fetcher.py --saveMd ./data/sources/jira-issues \\
        --startFromTime "2026-03-01T00:00:00"

    # Skip already-downloaded issues (for interrupted bulk downloads)
    uv run scripts/jira/fetchers/jira_fetcher.py --saveMd ./data/sources/jira-issues --skipExisting

    # Structured output (json + markdown in subdirs)
    uv run scripts/jira/fetchers/jira_fetcher.py --output ./data/downloaded/jira_data

    # Custom JQL override
    uv run scripts/jira/fetchers/jira_fetcher.py --saveMd ./data/sources/jira-issues \\
        --jql "project = MYPROJECT AND status != Rejected ORDER BY updated DESC"
"""

import asyncio
import json
import logging
import os
import re
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

# Ensure project root is on sys.path so `scripts.*` imports work
_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

from scripts.jira.sanitizers.pii_sanitizer import PiiSanitizer
# Dependency-free ADF flattener shared with the indexing converter; importing
# this leaf module does not pull in the indexing layer's heavier deps.
from main.sources.jira.adf_text import adf_to_text

logger = logging.getLogger(__name__)


class JiraFetcher:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        # Atlassian Cloud (e.g. https://nav.atlassian.net) vs self-hosted
        # Server/Data Center. The two differ in search endpoint, rich-text
        # format (ADF vs HTML/wiki), and how the Epic link is expressed.
        self.is_cloud = self.base_url.endswith(".atlassian.net")

    def _search_fields(self) -> str:
        """Fields to request per issue.

        `parent` carries the Epic on Cloud; the legacy Epic Link custom field
        (customfield_13510) only exists on the old Server instance and would
        trigger a 400 ("field does not exist") if requested on Cloud.
        """
        fields = ['summary', 'description', 'status', 'issuetype', 'priority',
                  'created', 'updated', 'assignee', 'reporter', 'labels',
                  'comment', 'parent']
        if not self.is_cloud:
            fields.append('customfield_13510')
        return ','.join(fields)

    def _render_field(self, value) -> str:
        """Render a Jira rich-text field (description / comment body) to text.

        Cloud returns ADF (a dict); Server returns HTML/wiki markup (a string).
        Dispatch on type so both ingest paths share one call site.
        """
        if not value:
            return ""
        if isinstance(value, dict):
            return adf_to_text(value)
        return self.html_to_markdown(value)

    def _extract_epic(self, fields: dict, epic_info: Dict[str, str]) -> tuple:
        """Return (epic_key, epic_summary) for an issue.

        Cloud: the Epic is the issue's `parent` (when that parent is an Epic),
        with the summary available inline in the search response. Server/DC: the
        legacy Epic Link custom field (customfield_13510), whose summary is
        resolved from the separately-fetched epic_info map.
        """
        parent = fields.get('parent') or {}
        parent_fields = parent.get('fields', {})
        parent_type = parent_fields.get('issuetype', {}).get('name', '')
        if parent.get('key') and parent_type.lower() == 'epic':
            return parent['key'], parent_fields.get('summary', '')

        legacy_key = fields.get('customfield_13510') or ''
        if legacy_key:
            return legacy_key, epic_info.get(legacy_key, '')
        return '', ''

    @staticmethod
    def sanitize_filename(text: str, max_length: int = 100) -> str:
        """Create safe filename from text."""
        text = re.sub(r'[<>:"/\\|?*]', '', text)
        text = re.sub(r'[-\s]+', '_', text)
        return text[:max_length].strip('_')

    @staticmethod
    def html_to_markdown(html_content: str) -> str:
        """Convert Jira HTML to Markdown."""
        if not html_content:
            return ""
        soup = BeautifulSoup(html_content, 'html.parser')
        for element in soup(['script', 'style']):
            element.decompose()
        text = soup.get_text(separator='\n', strip=True)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def _yaml_value(value: str) -> str:
        """Quote a YAML value if it contains special characters."""
        if not value:
            return '""'
        if any(c in value for c in (':', '#', '{', '}', '[', ']', ',', '&', '*', '?', '|', '-', '<', '>', '=', '!', '%', '@', '`', '"', "'", '\\')):
            escaped = value.replace('\\', '\\\\').replace('"', '\\"')
            return f'"{escaped}"'
        return value

    def _build_frontmatter(self, issue: dict, epic_link: str = "", epic_summary: str = "") -> str:
        """Build proper YAML frontmatter for a Jira issue."""
        fields = issue['fields']
        issue_key = issue['key']
        summary = fields.get('summary', '')
        status = fields.get('status', {}).get('name', '')
        issue_type = fields.get('issuetype', {}).get('name', '')
        priority = fields.get('priority', {}).get('name', '') if fields.get('priority') else ''
        created = fields.get('created', '')
        updated = fields.get('updated', '')
        assignee = fields.get('assignee', {}).get('displayName', 'Unassigned') if fields.get('assignee') else 'Unassigned'
        reporter = fields.get('reporter', {}).get('displayName', '') if fields.get('reporter') else ''
        labels = fields.get('labels', [])
        project = issue_key.split('-')[0] if '-' in issue_key else ''
        url = f"{self.base_url}/browse/{issue_key}"

        lines = ["---"]
        lines.append(f"title: {self._yaml_value(summary)}")
        lines.append(f"issue_key: {issue_key}")
        lines.append(f"issue_id: {self._yaml_value(str(issue['id']))}")
        lines.append(f"summary: {self._yaml_value(summary)}")
        lines.append(f"status: {self._yaml_value(status)}")
        lines.append(f"issue_type: {self._yaml_value(issue_type)}")
        lines.append(f"priority: {self._yaml_value(priority)}")
        lines.append(f"created: {self._yaml_value(created)}")
        lines.append(f"updated: {self._yaml_value(updated)}")
        lines.append(f"modifiedTime: {self._yaml_value(updated)}")
        lines.append(f"assignee: {self._yaml_value(assignee)}")
        lines.append(f"reporter: {self._yaml_value(reporter)}")

        labels_str = ", ".join(labels) if labels else ""
        lines.append(f"labels: {self._yaml_value(labels_str)}")

        lines.append(f"epic_link: {self._yaml_value(epic_link)}")
        lines.append(f"epic_summary: {self._yaml_value(epic_summary)}")

        parent = fields.get('parent') or {}
        parent_key = parent.get('key', '')
        parent_fields = parent.get('fields', {})
        # When the parent IS the Epic (Cloud company-managed projects link a
        # Story to its Epic via `parent`), that relationship is already captured
        # by epic_link. Emitting it again as `parent` makes the graph extractor
        # add a spurious er_subtask_av edge, so keep `parent` for genuine
        # non-epic parents only (e.g. a sub-task's parent story).
        if parent_key and parent_fields.get('issuetype', {}).get('name', '').lower() == 'epic':
            parent_key = ''
            parent_summary = ''
        else:
            parent_summary = parent_fields.get('summary', '') if parent_key else ''
        lines.append(f"parent: {self._yaml_value(parent_key)}")
        lines.append(f"parent_summary: {self._yaml_value(parent_summary)}")

        lines.append(f"project: {project}")
        lines.append(f"url: {self._yaml_value(url)}")
        lines.append("---\n")

        return "\n".join(lines)

    def issue_to_markdown(self, issue: dict, epic_link: str = "", epic_summary: str = "") -> str:
        """Convert Jira issue to Markdown with frontmatter."""
        fields = issue['fields']
        frontmatter = self._build_frontmatter(issue, epic_link, epic_summary)

        md_lines = [frontmatter]
        md_lines.append(f"# {issue['key']}: {fields.get('summary', '')}\n")

        if epic_link and epic_summary:
            md_lines.append(f"**Epic:** [{epic_link}]({self.base_url}/browse/{epic_link}) - {epic_summary}\n")

        description = fields.get('description')
        if description:
            md_lines.append("## Description\n")
            md_lines.append(self._render_field(description) + "\n")

        comments = fields.get('comment', {}).get('comments', [])
        if comments:
            md_lines.append("## Comments\n")
            for comment in comments:
                author = comment.get('author', {}).get('displayName', 'Unknown')
                created = comment.get('created', '')
                body = self._render_field(comment.get('body', ''))
                md_lines.append(f"### {author} - {created}\n")
                md_lines.append(f"{body}\n")

        return "\n".join(md_lines)

    @staticmethod
    def _set_file_mtime(file_path: Path, iso_time: str):
        """Set file mtime to match the Jira issue's updated time.

        Critical for incremental update cutoff calculations in
        files_document_reader.
        """
        if not iso_time:
            return
        try:
            ts = datetime.fromisoformat(iso_time).timestamp()
            os.utime(file_path, (ts, ts))
        except (ValueError, OSError):
            pass

    @staticmethod
    def scan_existing_issue_keys(md_dir: str) -> Set[str]:
        """Scan .md files and return set of issue_keys from frontmatter."""
        keys = set()
        md_path = Path(md_dir)
        if not md_path.exists():
            return keys

        for md_file in md_path.rglob("*.md"):
            if ".excluded" in md_file.parts:
                continue
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    in_fm = False
                    for line in f:
                        if line.strip() == "---" and not in_fm:
                            in_fm = True
                            continue
                        if line.strip() == "---" and in_fm:
                            break
                        if in_fm and line.startswith("issue_key:"):
                            key = line.partition(":")[2].strip().strip('"')
                            if key:
                                keys.add(key)
                            break
            except Exception:
                pass

        return keys

    # Exclusion reasons that are STABLE — an issue excluded for one of these
    # won't stop being noise just because it was edited, so it's safe to skip it
    # permanently on re-fetch. Content-based reasons (low_word_count,
    # empty_stub, minimal_content) and age (too_old) are DYNAMIC: a stub can
    # grow into a real issue and an old issue can be revived by an update. If we
    # skipped those permanently, such issues would be trapped in .excluded/
    # forever — never re-fetched, so never re-evaluated by cleanup. So we only
    # treat noise_* reasons as permanent; dynamic exclusions fall through and
    # get re-fetched (the incremental JQL only re-pulls the ones that actually
    # changed, and cleanup re-excludes any that are still noise that same run).
    _PERMANENT_EXCLUDE_PREFIX = "noise_"

    @classmethod
    def load_exclude_manifest(cls, manifest_path: str) -> Set[str]:
        """Load issue_keys to permanently skip on re-fetch.

        Only returns keys excluded for a STABLE reason (see
        _PERMANENT_EXCLUDE_PREFIX). Content/age exclusions are intentionally
        omitted so a stub that later grew, or an old issue that was revived,
        gets re-fetched and re-classified instead of being trapped forever.
        """
        path = Path(manifest_path)
        if not path.exists():
            return set()
        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        return {
            e["issue_key"]
            for e in entries
            if e.get("issue_key")
            and str(e.get("reason", "")).startswith(cls._PERMANENT_EXCLUDE_PREFIX)
        }

    async def fetch_issues(self, context, jql: str) -> List[Dict]:
        """Fetch all Jira issues matching JQL query using Playwright API context.

        Cloud uses the enhanced search endpoint (/rest/api/3/search/jql), which
        paginates with an opaque nextPageToken and returns no total — the old
        /search endpoint (startAt/total) was removed from Jira Cloud in 2025.
        Server/Data Center still uses the classic offset-paginated /search.
        """
        print(f"Fetching issues with JQL: {jql}")

        fields = self._search_fields()
        all_issues: List[Dict] = []
        next_page_token = None
        start_at = 0
        max_results = 100

        page = await context.new_page()

        try:
            while True:
                if self.is_cloud:
                    params = {
                        'jql': jql,
                        'maxResults': str(max_results),
                        'fields': fields,
                    }
                    if next_page_token:
                        params['nextPageToken'] = next_page_token
                    url = f"{self.base_url}/rest/api/3/search/jql"
                else:
                    params = {
                        'jql': jql,
                        'startAt': str(start_at),
                        'maxResults': str(max_results),
                        'expand': 'renderedFields',
                        'fields': fields,
                    }
                    url = f"{self.base_url}/rest/api/2/search"

                # Retry transient errors a few times, then RAISE rather than
                # break: silently returning a partial page set would let an
                # incremental run index a truncated result and advance its cutoff
                # past the gap, permanently losing the un-fetched issues.
                for attempt in range(3):
                    response = await page.request.get(url, params=params)
                    if response.status == 200:
                        break
                    if response.status in (429, 502, 503, 504) and attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    break

                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"Jira API returned status {response.status}: {error_text[:300]}"
                    )

                data = json.loads(await response.text())
                issues = data.get('issues', [])
                all_issues.extend(issues)

                if self.is_cloud:
                    print(f"  Fetched {len(all_issues)} issues...")
                    next_page_token = data.get('nextPageToken')
                    # `not issues` also terminates an empty page that still
                    # echoes a token (a known Cloud quirk) — without it the loop
                    # would spin forever re-requesting the same page.
                    if data.get('isLast') or not next_page_token or not issues:
                        break
                else:
                    total = data.get('total', 0)
                    print(f"  Fetched {len(all_issues)}/{total} issues")
                    if not issues or len(all_issues) >= total:
                        break
                    # Advance by the actual count returned, not the requested
                    # page size, in case the instance caps maxResults below 100.
                    start_at += len(issues)

                await asyncio.sleep(0.5)
        finally:
            await page.close()

        print(f"\nTotal issues fetched: {len(all_issues)}")
        return all_issues

    async def fetch_epic_info(self, context, epic_key: str) -> Optional[str]:
        """Fetch Epic summary for a given Epic key."""
        if not epic_key:
            return None

        page = await context.new_page()
        try:
            response = await page.request.get(
                f"{self.base_url}/rest/api/2/issue/{epic_key}",
                params={'fields': 'summary'}
            )
            if response.status != 200:
                return None
            data = json.loads(await response.text())
            return data.get('fields', {}).get('summary', '')
        except Exception as e:
            print(f"Warning: Could not fetch Epic {epic_key}: {e}")
            return None
        finally:
            await page.close()

    async def is_api_authenticated(self, context) -> bool:
        """Return True if the current session can call the Jira REST API.

        Checked against /myself rather than by sniffing the page URL: on
        Atlassian Cloud an unauthenticated /browse/<key> does not reliably
        redirect to a URL containing "login"/"auth", and an unauthenticated
        search returns 200 with zero issues — so URL-sniffing produced a false
        "Already authenticated" and silently empty fetches.
        """
        api_version = "3" if self.is_cloud else "2"
        try:
            # context.request issues a background HTTP call without opening a
            # page/tab, so polling this while the user logs in doesn't steal
            # focus or churn the visible login window.
            resp = await context.request.get(f"{self.base_url}/rest/api/{api_version}/myself")
            return resp.status == 200
        except Exception:
            return False

    async def authenticate(self, context, project_key: str):
        """Ensure the session is authenticated to the REST API; prompt if not."""
        if await self.is_api_authenticated(context):
            print("Already authenticated!")
            return

        page = await context.new_page()
        await page.goto(f"{self.base_url}/browse/{project_key}", timeout=60000)

        print("\n" + "=" * 60)
        print("PLEASE LOG IN TO JIRA")
        print("=" * 60)
        print("1. Complete the login process in the browser")
        print("2. Use your ID and authenticator")
        print(f"3. Wait until you see the {project_key} project page")
        print("=" * 60)

        # Poll the API until the session becomes valid (up to 5 minutes), so we
        # only proceed once /myself actually returns 200 — no false positives.
        authenticated = False
        for _ in range(150):
            await asyncio.sleep(2)
            if await self.is_api_authenticated(context):
                authenticated = True
                break

        await page.close()

        if not authenticated:
            raise RuntimeError(
                "Timed out waiting for Jira authentication (no valid API session after 5 min)."
            )

        print("\nAuthentication successful!")
        auth_file = Path(__file__).parent.parent / "auth" / "jira_auth.json"
        auth_file.parent.mkdir(parents=True, exist_ok=True)
        await context.storage_state(path=str(auth_file))
        print(f"Saved authentication to {auth_file}")

    def save_issues_as_markdown(self, issues: List[Dict], save_md_path: str,
                                 epic_info: Dict[str, str],
                                 skip_keys: Optional[Set[str]] = None):
        """Save issues as flat markdown files to curated directory.

        PII (personnummer, emails, passwords) is automatically redacted
        before writing to disk.
        """
        md_base = Path(save_md_path)
        md_base.mkdir(parents=True, exist_ok=True)

        # Map existing files by issue_key (filename is "<key>_<title>.md"). A
        # renamed issue would otherwise leave its previous-title file on disk;
        # the files indexer keys documents by path, so that orphan becomes a
        # duplicate document for the same issue. Built once to avoid re-globbing
        # the directory per issue.
        existing_by_key: Dict[str, List[Path]] = {}
        for existing in md_base.glob("*.md"):
            key = existing.name.split("_", 1)[0]
            existing_by_key.setdefault(key, []).append(existing)

        sanitizer = PiiSanitizer()
        saved = 0
        skipped = 0
        pii_files = 0
        pii_total = 0

        for i, issue in enumerate(issues, 1):
            issue_key = issue['key']

            if skip_keys and issue_key in skip_keys:
                skipped += 1
                continue

            fields = issue['fields']
            summary = fields.get('summary', 'no-title')
            updated = fields.get('updated', '')
            epic_link, epic_summary = self._extract_epic(fields, epic_info)

            safe_title = self.sanitize_filename(summary)
            filename = f"{issue_key}_{safe_title}.md"

            # Drop any stale file for this issue under a previous title so the
            # renamed issue doesn't end up indexed twice.
            for stale in existing_by_key.get(issue_key, []):
                if stale.name != filename:
                    stale.unlink(missing_ok=True)

            md_content = self.issue_to_markdown(issue, epic_link or '', epic_summary)

            # Sanitize PII before writing
            result = sanitizer.sanitize(md_content)
            if result.has_pii:
                pii_files += 1
                pii_total += len(result.findings)
                cats = {}
                for f in result.findings:
                    cats[f.category] = cats.get(f.category, 0) + 1
                cat_str = ", ".join(f"{c}:{n}" for c, n in cats.items())
                logger.info(f"PII redacted in {issue_key}: {cat_str}")
                md_content = result.sanitized_text

            md_file = md_base / filename
            md_file.write_text(md_content, encoding='utf-8')
            self._set_file_mtime(md_file, updated)
            saved += 1

            if saved % 50 == 0:
                print(f"  Saved {saved} issues so far...")

        print(f"\nSaved {saved} issues to {md_base}")
        if skipped:
            print(f"  Skipped {skipped} (already existing or excluded)")
        if pii_total > 0:
            print(f"  PII redacted: {pii_total} findings in {pii_files} files")

    def save_issues_structured(self, issues: List[Dict], output_dir: str,
                                epic_info: Dict[str, str]):
        """Save issues organized by Epic in json/ + markdown/ subdirs.

        PII is redacted from markdown output. JSON files are saved as-is
        (raw API responses are not published to the index).
        """
        output = Path(output_dir)
        json_dir = output / "json"
        md_dir = output / "markdown"
        json_dir.mkdir(parents=True, exist_ok=True)
        md_dir.mkdir(parents=True, exist_ok=True)

        sanitizer = PiiSanitizer()
        epic_stats = {}
        pii_total = 0

        for i, issue in enumerate(issues, 1):
            issue_key = issue['key']
            fields = issue['fields']
            summary = fields.get('summary', 'no-title')
            updated = fields.get('updated', '')
            epic_link, epic_summary = self._extract_epic(fields, epic_info)

            if epic_link:
                epic_name = self.sanitize_filename(f"{epic_link}_{epic_summary}" if epic_summary else epic_link)
            else:
                epic_name = "No_Epic"
            epic_stats[epic_name] = epic_stats.get(epic_name, 0) + 1

            epic_json = json_dir / epic_name
            epic_md = md_dir / epic_name
            epic_json.mkdir(exist_ok=True)
            epic_md.mkdir(exist_ok=True)

            safe_title = self.sanitize_filename(summary)
            filename = f"{issue_key}_{safe_title}"

            # Save JSON
            json_path = epic_json / f"{filename}.json"
            json_path.write_text(
                json.dumps(issue, indent=2, ensure_ascii=False),
                encoding='utf-8'
            )

            # Save Markdown (sanitized)
            md_content = self.issue_to_markdown(issue, epic_link or '', epic_summary)
            result = sanitizer.sanitize(md_content)
            if result.has_pii:
                pii_total += len(result.findings)
                logger.info(f"PII redacted in {issue_key}: {len(result.findings)} findings")
                md_content = result.sanitized_text

            md_path = epic_md / f"{filename}.md"
            md_path.write_text(md_content, encoding='utf-8')
            self._set_file_mtime(md_path, updated)

            if (i) % 100 == 0:
                print(f"  Processed {i}/{len(issues)} issues...")

        print(f"\nSaved {len(issues)} issues")
        print(f"  JSON: {json_dir}")
        print(f"  Markdown: {md_dir}")
        if pii_total > 0:
            print(f"  PII redacted: {pii_total} findings")

        print("\nIssues per Epic:")
        for epic_name, count in sorted(epic_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {epic_name}: {count}")


def build_jql(project: str, jql_override: Optional[str], start_from_time: Optional[str]) -> str:
    """Build JQL query from arguments."""
    if jql_override:
        jql = jql_override
        if start_from_time:
            # Jira expects "YYYY-MM-DD HH:mm" or "YYYY/MM/DD HH:mm" format
            jira_date = start_from_time[:10].replace("-", "/")
            if len(start_from_time) > 10:
                jira_date += " " + start_from_time[11:16]
            jql += f' AND updated >= "{jira_date}"'
        return jql

    jql = f"project = {project}"
    if start_from_time:
        jira_date = start_from_time[:10].replace("-", "/")
        if len(start_from_time) > 10:
            jira_date += " " + start_from_time[11:16]
        jql += f' AND updated >= "{jira_date}"'
    jql += " ORDER BY updated DESC"
    return jql


async def main():
    parser = argparse.ArgumentParser(
        description="Fetch Jira issues with Playwright authentication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full download to curated directory
  uv run scripts/jira/fetchers/jira_fetcher.py --saveMd ./data/sources/jira-issues

  # Incremental: only issues updated since cutoff
  uv run scripts/jira/fetchers/jira_fetcher.py --saveMd ./data/sources/jira-issues \\
      --startFromTime "2026-03-01T00:00:00"

  # Resume interrupted download (skip existing)
  uv run scripts/jira/fetchers/jira_fetcher.py --saveMd ./data/sources/jira-issues --skipExisting

  # Structured output with Epic folders
  uv run scripts/jira/fetchers/jira_fetcher.py --output ./data/downloaded/jira_data

  # Custom JQL
  uv run scripts/jira/fetchers/jira_fetcher.py --saveMd ./data/sources/jira-issues \\
      --jql "project = MYPROJECT AND status != Rejected"
        """
    )

    parser.add_argument("--project", "-p", default="MYPROJECT",
                        help="Jira project key (default: MYPROJECT)")
    parser.add_argument("--base-url", "-u", required=True,
                        help="Jira base URL. Cloud: https://your-domain.atlassian.net "
                             "(e.g. https://nav.atlassian.net). Server/DC: https://jira.example.com")
    parser.add_argument("--jql", default=None,
                        help="Override JQL query entirely (--startFromTime still appended)")
    parser.add_argument("--saveMd", default=None,
                        help="Save markdown to curated directory (flat files, no subdirs). "
                             "When set, --output is ignored.")
    parser.add_argument("--output", "-o", default=None,
                        help="Structured output dir with json/ + markdown/ subdirs")
    parser.add_argument("--skipExisting", action="store_true", default=False,
                        help="Skip issue_keys already on disk (for interrupted bulk downloads)")
    parser.add_argument("--startFromTime", default=None,
                        help="Only fetch issues updated on/after this ISO datetime")
    parser.add_argument("--excludeManifest", default=None,
                        help="Path to excluded_manifest.json — skip issue_keys listed there")

    args = parser.parse_args()

    save_md = args.saveMd
    output_dir = args.output or str(
        Path(__file__).parent.parent.parent.parent / "data" / "downloaded" / "jira_data"
    )

    print("=" * 60)
    print("JIRA FETCHER")
    print("=" * 60)
    print(f"  Project: {args.project}")
    print(f"  Base URL: {args.base_url}")
    if save_md:
        print(f"  Save markdown to: {save_md}")
    else:
        print(f"  Output: {output_dir}")
    if args.startFromTime:
        print(f"  Start from: {args.startFromTime}")
    if args.skipExisting:
        print(f"  Skip existing: yes")
    if args.excludeManifest:
        print(f"  Exclude manifest: {args.excludeManifest}")
    print()

    jql = build_jql(args.project, args.jql, args.startFromTime)

    fetcher = JiraFetcher(base_url=args.base_url)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)

        # Load auth state
        auth_file = Path(__file__).parent.parent / "auth" / "jira_auth.json"
        storage_state = str(auth_file) if auth_file.exists() else None
        context = await browser.new_context(storage_state=storage_state)

        # Authenticate
        await fetcher.authenticate(context, args.project)

        # Save auth state for future use
        auth_file.parent.mkdir(parents=True, exist_ok=True)
        await context.storage_state(path=str(auth_file))

        # Fetch issues
        issues = await fetcher.fetch_issues(context, jql)

        if not issues:
            print("No issues were fetched")
            await browser.close()
            return

        # Collect Epic info. On Cloud the Epic is each issue's parent and its
        # summary is already inline in the search response, so no extra requests
        # are needed. On Server the Epic Link is a custom field id, so we resolve
        # each unique Epic's summary with a follow-up request.
        print("\nCollecting Epic information...")
        epic_info: Dict[str, str] = {}
        if fetcher.is_cloud:
            # Reuse the same parent->epic detection as the per-issue path so the
            # two never drift apart.
            for issue in issues:
                epic_key, epic_summary = fetcher._extract_epic(issue['fields'], {})
                if epic_key:
                    epic_info[epic_key] = epic_summary
        else:
            epic_keys = set()
            for issue in issues:
                epic_link = issue['fields'].get('customfield_13510')
                if epic_link:
                    epic_keys.add(epic_link)
            for epic_key in epic_keys:
                summary = await fetcher.fetch_epic_info(context, epic_key)
                if summary:
                    epic_info[epic_key] = summary
                    print(f"  {epic_key}: {summary}")

        print(f"Found {len(epic_info)} unique Epics")

        await browser.close()

    # Save results
    if save_md:
        skip_keys: Set[str] = set()
        if args.skipExisting:
            existing = JiraFetcher.scan_existing_issue_keys(save_md)
            skip_keys.update(existing)
            print(f"Found {len(existing)} existing issues on disk")

        if args.excludeManifest:
            excluded = JiraFetcher.load_exclude_manifest(args.excludeManifest)
            skip_keys.update(excluded)
            print(f"Loaded {len(excluded)} permanently-excluded issue_keys from manifest "
                  f"(noise only; content/age exclusions are re-fetched and re-evaluated)")

        fetcher.save_issues_as_markdown(issues, save_md, epic_info, skip_keys)
        print(f"\nDone! {len(issues)} issues fetched from Jira")
    else:
        fetcher.save_issues_structured(issues, output_dir, epic_info)
        print(f"\nDone! {len(issues)} issues processed")

    # Default the suggested collection name to the destination folder, NOT a
    # hardcoded "jira-issues" — blindly running the old hint would re-index a
    # sample dir on top of the production collection and clobber it.
    suggested_collection = Path(save_md).name if save_md else "my-jira"

    print("\nNEXT STEPS:")
    if save_md:
        print(f"  # Clean up noise:")
        print(f"  uv run jira_cleanup_md.py --saveMd {save_md} --dryRun")
        print(f"  # Index into vector database (collection defaults to the folder name —")
        print(f"  # change it deliberately if you mean to update an existing collection):")
        print(f"  uv run files_collection_create_cmd_adapter.py \\")
        print(f"    --basePath {save_md} --collection {suggested_collection} \\")
        print(f"    --excludePatterns \"^\\.excluded/.*\"")
    else:
        md_path = Path(output_dir) / "markdown"
        print(f"  # Index into vector database:")
        print(f"  uv run files_collection_create_cmd_adapter.py \\")
        print(f"    --basePath \"{md_path}\" --collection my-jira")


if __name__ == "__main__":
    asyncio.run(main())
