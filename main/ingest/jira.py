"""Jira issue ingest: convert DOM-scraped content to markdown with PII sanitization."""
import logging
import os
import re
import datetime as dt
from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel

from scripts.jira.sanitizers.pii_sanitizer import PiiSanitizer

logger = logging.getLogger(__name__)

_pii_sanitizer = PiiSanitizer()


class JiraIngestComment(BaseModel):
    author: str = "Unknown"
    date: str = ""
    body: str = ""


class JiraIngestRequest(BaseModel):
    """Jira issue content scraped from the page DOM by the Chrome extension."""
    issueKey: str  # e.g., "PROJECT-1234"
    url: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    status: Optional[str] = None
    type: Optional[str] = None
    priority: Optional[str] = None
    assignee: Optional[str] = None
    reporter: Optional[str] = None
    labels: Optional[list[str]] = None
    description: Optional[str] = None
    comments: Optional[list[JiraIngestComment]] = None
    created: Optional[str] = None
    updated: Optional[str] = None
    epicLink: Optional[str] = None


def _read_existing_frontmatter(filepath: str) -> dict:
    """Read YAML frontmatter from an existing markdown file into a dict.

    Handles multi-line YAML lists (e.g. labels) by collecting list items.
    Returns {} on read/parse failure (logged at warning level).
    """
    metadata = {}
    current_list_key = None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            in_fm = False
            for line in f:
                if line.strip() == "---" and not in_fm:
                    in_fm = True
                    continue
                if line.strip() == "---" and in_fm:
                    break
                if not in_fm:
                    continue
                stripped = line.strip()
                if stripped.startswith("- ") and current_list_key:
                    item = stripped[2:].strip()
                    existing = metadata.get(current_list_key, "")
                    metadata[current_list_key] = (existing + "," + item) if existing else item
                    continue
                current_list_key = None
                if ":" in line:
                    key, _, value = line.partition(":")
                    key = key.strip()
                    value = value.strip().strip('"')
                    if key and value:
                        metadata[key] = value
                    elif key and not value:
                        # Key with no value — likely start of a YAML list
                        current_list_key = key
    except (OSError, UnicodeDecodeError) as e:
        logger.warning(f"Could not read frontmatter from {filepath}: {e}")
    return metadata


def _find_existing_jira_file(jira_path: str, issue_key: str) -> tuple[Optional[str], dict]:
    """Find an existing markdown file for `issue_key`. Returns (filepath, frontmatter) or (None, {}).

    Scans files whose names start with the issue key (handles underscore and
    space filename conventions) and verifies the parsed `issue_key` matches.
    """
    if not os.path.isdir(jira_path):
        return None, {}
    prefix = issue_key
    for filename in os.listdir(jira_path):
        if not filename.endswith(".md"):
            continue
        # Match "PROJECT-1234_..." or "PROJECT-1234 ..." or "PROJECT-1234.md"
        if filename.startswith(prefix + "_") or filename.startswith(prefix + " ") or filename == prefix + ".md":
            filepath = os.path.join(jira_path, filename)
            metadata = _read_existing_frontmatter(filepath)
            if metadata.get("issue_key") == issue_key:
                return filepath, metadata
    return None, {}


def _jira_content_to_markdown(req: JiraIngestRequest, existing_metadata: Optional[dict] = None) -> str:
    """Convert DOM-scraped Jira issue content to markdown with frontmatter.

    If existing_metadata is provided, preserves fields the Chrome extension
    doesn't capture (epic_summary, project, modifiedTime).
    """
    key = req.issueKey
    summary = req.summary or req.title or ""
    existing = existing_metadata or {}

    def _yaml_escape(val: str) -> str:
        """Wrap value in quotes and escape internal quotes for safe YAML."""
        return '"' + val.replace('\\', '\\\\').replace('"', '\\"') + '"'

    lines = ["---"]
    lines.append(f"title: {_yaml_escape(summary)}")
    lines.append(f"issue_key: {key}")
    lines.append(f"summary: {_yaml_escape(summary)}")
    lines.append(f"status: {_yaml_escape(req.status or existing.get('status', ''))}")
    lines.append(f"issue_type: {_yaml_escape(req.type or existing.get('issue_type', ''))}")
    lines.append(f"priority: {_yaml_escape(req.priority or existing.get('priority', ''))}")
    lines.append(f"created: {_yaml_escape(req.created or existing.get('created', ''))}")
    updated = req.updated or existing.get('updated', '')
    lines.append(f"updated: {_yaml_escape(updated)}")
    lines.append(f"modifiedTime: {_yaml_escape(updated)}")
    lines.append(f"assignee: {_yaml_escape(req.assignee or existing.get('assignee', ''))}")
    lines.append(f"reporter: {_yaml_escape(req.reporter or existing.get('reporter', ''))}")

    # Labels — write as comma-separated string (not YAML list) for parser compatibility
    if req.labels:
        labels_str = ", ".join(req.labels)
        lines.append(f"labels: {_yaml_escape(labels_str)}")
    elif existing.get('labels'):
        lines.append(f"labels: {_yaml_escape(existing['labels'])}")
    else:
        lines.append(f"labels: {_yaml_escape('')}")

    # Epic — preserve existing epic_summary if extension doesn't provide it
    epic_link = req.epicLink or existing.get('epic_link', '')
    epic_summary = existing.get('epic_summary', '')
    lines.append(f"epic_link: {_yaml_escape(epic_link)}")
    lines.append(f"epic_summary: {_yaml_escape(epic_summary)}")

    # Project — extension doesn't provide, preserve from existing
    project = existing.get('project', key.split('-')[0] if '-' in key else '')
    lines.append(f"project: {_yaml_escape(project)}")

    if req.url:
        lines.append(f"url: {_yaml_escape(req.url)}")
    elif existing.get('url'):
        lines.append(f"url: {_yaml_escape(existing['url'])}")
    lines.append("---\n")

    lines.append(f"# {key}: {summary}\n")

    # Epic context in body (if we have it)
    if epic_link and epic_summary:
        base_url = existing.get('url', '').rsplit('/browse/', 1)[0] if existing.get('url') else ''
        if base_url:
            lines.append(f"**Epic:** [{epic_link}]({base_url}/browse/{epic_link}) - {epic_summary}\n")
        else:
            lines.append(f"**Epic:** {epic_link} - {epic_summary}\n")

    if req.description:
        lines.append("## Description\n")
        lines.append(req.description + "\n")

    if req.comments:
        lines.append("## Comments\n")
        for comment in req.comments:
            lines.append(f"### {comment.author} ({comment.date})\n")
            lines.append(comment.body + "\n")

    return "\n".join(lines)


def ingest_jira(req: JiraIngestRequest, *, sources_path: str) -> dict:
    """Save a Jira issue as PII-sanitized markdown. Merges metadata if a file already exists.

    Returns: {file_path, issue_key, summary}.
    """
    if not re.match(r"^[A-Z][A-Z0-9]+-\d+$", req.issueKey):
        raise HTTPException(status_code=400, detail=f"Invalid Jira issue key: {req.issueKey}")

    summary_text = req.summary or req.title or "untitled"

    os.makedirs(sources_path, exist_ok=True)
    existing_filepath, existing_metadata = _find_existing_jira_file(sources_path, req.issueKey)

    if existing_metadata:
        logger.info(f"Jira ingest: found existing file for {req.issueKey}, merging metadata")

    md_content = _jira_content_to_markdown(req, existing_metadata)

    sanitize_result = _pii_sanitizer.sanitize(md_content)
    if sanitize_result.has_pii:
        cats = {}
        for f in sanitize_result.findings:
            cats[f.category] = cats.get(f.category, 0) + 1
        logger.info(f"Jira ingest PII redacted in {req.issueKey}: "
                    + ", ".join(f"{c}:{n}" for c, n in cats.items()))
        md_content = sanitize_result.sanitized_text

    # Use existing filename if found, otherwise create new one using underscore convention
    if existing_filepath:
        filepath = existing_filepath
        filename = os.path.basename(filepath)
    else:
        safe_title = re.sub(r'[<>:"/\\|?*]', '', summary_text)
        safe_title = re.sub(r'[-\s]+', '_', safe_title)[:100].strip('_')
        filename = f"{req.issueKey}_{safe_title}.md"
        filepath = os.path.join(sources_path, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Set file mtime to issue updated time for correct incremental updates
    updated = req.updated or existing_metadata.get('updated', '')
    if updated:
        try:
            ts = dt.datetime.fromisoformat(updated).timestamp()
            os.utime(filepath, (ts, ts))
        except (ValueError, OSError):
            pass

    logger.info(f"Jira ingest: saved {filename}")

    return {
        "file_path": filename,
        "issue_key": req.issueKey,
        "summary": summary_text,
    }
