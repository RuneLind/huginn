"""Convert Claude Code session transcripts to markdown files for vector indexing.

Reads JSONL session files from ~/.claude/projects/, extracts user/assistant
conversation turns, and writes one markdown file per session with YAML frontmatter.

Usage:
    uv run scripts/claude_sessions/claude_sessions_to_markdown.py --saveMd ./data/sources/claude-sessions
    uv run scripts/claude_sessions/claude_sessions_to_markdown.py --saveMd ./data/sources/claude-sessions --skipExisting
    uv run scripts/claude_sessions/claude_sessions_to_markdown.py --saveMd ./data/sources/claude-sessions --startFromTime 2026-02-01T00:00:00
    uv run scripts/claude_sessions/claude_sessions_to_markdown.py --saveMd ./data/sources/claude-sessions --projects my-project huginn
"""

import argparse
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"

# Message content patterns to skip entirely (internal plumbing)
_SKIP_CONTENT_PATTERNS = [
    re.compile(r"^<local-command-"),
    re.compile(r"^<command-name>/clear"),
    re.compile(r"^<command-name>/compact"),
]

# Regex to clean XML command tags from user messages (e.g. slash commands)
_COMMAND_TAG_RE = re.compile(r"<command-(?:name|message)>.*?</command-(?:name|message)>\s*", re.DOTALL)
_COMMAND_ARGS_RE = re.compile(r"<command-args>(.*?)</command-args>", re.DOTALL)
_SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)

# Max chars to include from a single thinking block
_MAX_THINKING_CHARS = 2000

# Max chars for a tool_use summary line
_MAX_TOOL_SUMMARY_CHARS = 200


def _parse_sessions_index(project_dir: Path) -> dict:
    """Parse sessions-index.json if it exists. Returns {sessionId: entry}."""
    index_path = project_dir / "sessions-index.json"
    if not index_path.exists():
        return {}
    try:
        with open(index_path, encoding="utf-8") as f:
            data = json.load(f)
        return {e["sessionId"]: e for e in data.get("entries", [])}
    except Exception as e:
        log.warning(f"Failed to parse {index_path}: {e}")
        return {}


def _discover_sessions(project_dir: Path) -> list[Path]:
    """Find all session JSONL files (not subagent files)."""
    sessions = []
    for item in project_dir.iterdir():
        if item.suffix == ".jsonl" and item.is_file():
            sessions.append(item)
    return sessions


def _extract_project_name(project_dir_name: str) -> str:
    """Extract a human-readable project name from the dir name.

    e.g. '-Users-rune-source-private-my-project' -> 'my-project'
    """
    # The dir name encodes the full path with - separators
    # Build prefixes dynamically from the user's home directory
    home_encoded = str(Path.home()).replace(os.sep, "-")
    if not home_encoded.startswith("-"):
        home_encoded = "-" + home_encoded
    prefixes = [
        f"{home_encoded}-source-nav-",
        f"{home_encoded}-source-private-",
        f"{home_encoded}-source-work-",
        f"{home_encoded}-source-",
        f"{home_encoded}-",
    ]
    for prefix in prefixes:
        if project_dir_name.startswith(prefix) and len(project_dir_name) > len(prefix):
            return project_dir_name[len(prefix):]
    return project_dir_name.strip("-") or "unknown"


def _extract_text_from_content(content) -> str | None:
    """Extract plain text from a message content field.

    Content can be a string or a list of content blocks.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block["text"])
            elif isinstance(block, str):
                texts.append(block)
        return "\n".join(texts) if texts else None
    return None


def _extract_tool_uses(content) -> list[str]:
    """Extract brief tool_use summaries from content blocks."""
    if not isinstance(content, list):
        return []
    summaries = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            name = block.get("name", "unknown")
            inp = block.get("input", {})
            # Build a brief summary depending on tool type
            if name == "Bash":
                cmd = inp.get("command", "")[:_MAX_TOOL_SUMMARY_CHARS]
                summaries.append(f"[Tool: Bash] `{cmd}`")
            elif name == "Read":
                summaries.append(f"[Tool: Read] {inp.get('file_path', '')}")
            elif name == "Write":
                summaries.append(f"[Tool: Write] {inp.get('file_path', '')}")
            elif name == "Edit":
                summaries.append(f"[Tool: Edit] {inp.get('file_path', '')}")
            elif name == "Grep":
                summaries.append(f"[Tool: Grep] pattern=`{inp.get('pattern', '')}`")
            elif name == "Glob":
                summaries.append(f"[Tool: Glob] {inp.get('pattern', '')}")
            elif name == "Task":
                desc = inp.get("description", inp.get("prompt", ""))[:_MAX_TOOL_SUMMARY_CHARS]
                summaries.append(f"[Tool: Task] {desc}")
            elif name == "WebSearch":
                summaries.append(f"[Tool: WebSearch] `{inp.get('query', '')}`")
            elif name == "WebFetch":
                summaries.append(f"[Tool: WebFetch] {inp.get('url', '')}")
            else:
                brief = str(inp)[:80]
                summaries.append(f"[Tool: {name}] {brief}")
    return summaries


def _extract_thinking(content) -> str | None:
    """Extract thinking text from all thinking blocks in content."""
    if not isinstance(content, list):
        return None
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "thinking":
            thinking = block.get("thinking", "")
            if thinking:
                if len(thinking) > _MAX_THINKING_CHARS:
                    thinking = thinking[:_MAX_THINKING_CHARS] + "..."
                parts.append(thinking)
    return "\n\n".join(parts) if parts else None


def _clean_user_text(text: str) -> str:
    """Clean XML command tags and system reminders from user text."""
    # Extract args from command-args tags before stripping
    args_match = _COMMAND_ARGS_RE.search(text)
    args_text = args_match.group(1).strip() if args_match else ""

    # Extract the command name for context
    cmd_match = re.search(r"<command-name>(/\w+)</command-name>", text)
    cmd_name = cmd_match.group(1) if cmd_match else ""

    # Strip all command tags
    cleaned = _COMMAND_TAG_RE.sub("", text)
    cleaned = _COMMAND_ARGS_RE.sub("", cleaned)
    cleaned = _SYSTEM_REMINDER_RE.sub("", cleaned)
    cleaned = cleaned.strip()

    # If only command tags were present, reconstruct as "/<command> <args>"
    if not cleaned and (cmd_name or args_text):
        parts = [cmd_name, args_text]
        cleaned = " ".join(p for p in parts if p).strip()

    return cleaned


def _should_skip_user_content(content_text: str) -> bool:
    """Check if user message content is internal plumbing that should be skipped."""
    if not content_text:
        return True
    for pattern in _SKIP_CONTENT_PATTERNS:
        if pattern.search(content_text.strip()):
            return True
    return False


def _parse_session(session_path: Path) -> list[dict]:
    """Parse a session JSONL file and return ordered conversation messages.

    Returns list of dicts with keys: role, text, tools, thinking, timestamp
    """
    messages = []
    with open(session_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")

            # Skip non-conversation types
            if msg_type not in ("user", "assistant"):
                continue

            # Skip sidechains (subagent messages)
            if msg.get("isSidechain"):
                continue

            # Skip meta messages
            if msg.get("isMeta"):
                continue

            message_body = msg.get("message", {})
            content = message_body.get("content")
            timestamp = msg.get("timestamp", "")

            if msg_type == "user":
                # Skip tool results (content is array with tool_result)
                if isinstance(content, list):
                    has_tool_result = any(
                        isinstance(b, dict) and b.get("type") == "tool_result"
                        for b in content
                    )
                    if has_tool_result:
                        continue

                text = _extract_text_from_content(content)
                if _should_skip_user_content(text):
                    continue

                text = _clean_user_text(text)
                if not text:
                    continue

                messages.append({
                    "role": "user",
                    "text": text,
                    "tools": [],
                    "thinking": None,
                    "timestamp": timestamp,
                })

            elif msg_type == "assistant":
                text = _extract_text_from_content(content)
                if text:
                    text = _SYSTEM_REMINDER_RE.sub("", text).strip()
                tools = _extract_tool_uses(content)
                thinking = _extract_thinking(content)

                # Skip if there's nothing meaningful
                if not text and not tools and not thinking:
                    continue

                messages.append({
                    "role": "assistant",
                    "text": text,
                    "tools": tools,
                    "thinking": thinking,
                    "timestamp": timestamp,
                })

    return messages


def _session_to_markdown(messages: list[dict], metadata: dict) -> str:
    """Convert parsed session messages to a markdown document."""
    lines = []

    # YAML frontmatter — single-line values only
    lines.append("---")
    for key, value in metadata.items():
        if value is not None:
            # Flatten to single line and escape quotes
            safe_value = str(value).replace("\n", " ").replace("\r", "").replace('"', '\\"')
            if len(safe_value) > 200:
                safe_value = safe_value[:200]
            lines.append(f'{key}: "{safe_value}"')
    lines.append("---")
    lines.append("")

    # Title
    title = metadata.get("summary") or metadata.get("firstPrompt") or "Untitled session"
    lines.append(f"# {title[:120]}")
    lines.append("")

    for msg in messages:
        if msg["role"] == "user":
            lines.append("## User")
            lines.append("")
            if msg["text"]:
                lines.append(msg["text"].strip())
            lines.append("")

        elif msg["role"] == "assistant":
            lines.append("## Assistant")
            lines.append("")
            if msg["thinking"]:
                lines.append("<details><summary>Thinking</summary>")
                lines.append("")
                lines.append(msg["thinking"].strip())
                lines.append("")
                lines.append("</details>")
                lines.append("")
            if msg["text"]:
                lines.append(msg["text"].strip())
                lines.append("")
            if msg["tools"]:
                for tool in msg["tools"]:
                    lines.append(f"- {tool}")
                lines.append("")

    return "\n".join(lines)


def _make_filename(session_id: str, index_entry: dict | None, first_prompt: str | None) -> str:
    """Build a descriptive filename for the session markdown."""
    # Use customTitle or summary from index if available
    title = None
    if index_entry:
        title = index_entry.get("customTitle") or index_entry.get("summary")

    if not title and first_prompt:
        title = first_prompt[:80]

    if not title:
        title = session_id[:12]

    # Sanitize
    title = re.sub(r'[<>:"/\\|?*\n\r]', '_', title)
    title = re.sub(r'[\s_]+', ' ', title).strip()
    if len(title) > 120:
        title = title[:120]

    # Prefix with short session id for uniqueness
    return f"{session_id[:8]}_{title}.md"


def convert_sessions(
    save_md_path: str,
    skip_existing: bool = False,
    start_from_time: str | None = None,
    project_filters: list[str] | None = None,
    min_messages: int = 4,
):
    """Main conversion function."""
    save_dir = Path(save_md_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    cutoff = None
    if start_from_time:
        cutoff = datetime.fromisoformat(start_from_time)

    # Scan existing session IDs if skipping
    existing_session_ids = set()
    if skip_existing:
        for f in save_dir.rglob("*.md"):
            # Session ID is the first 8 chars of filename
            existing_session_ids.add(f.name[:8])
        log.info(f"Found {len(existing_session_ids)} existing session files")

    if not CLAUDE_PROJECTS_DIR.exists():
        log.error(f"Claude projects directory not found: {CLAUDE_PROJECTS_DIR}")
        return

    total_converted = 0
    total_skipped = 0
    total_too_short = 0

    for project_dir in sorted(CLAUDE_PROJECTS_DIR.iterdir()):
        if not project_dir.is_dir():
            continue

        project_name = _extract_project_name(project_dir.name)

        # Apply project filter if specified
        if project_filters:
            if not any(f.lower() in project_name.lower() for f in project_filters):
                continue

        index_entries = _parse_sessions_index(project_dir)
        sessions = _discover_sessions(project_dir)

        if not sessions:
            continue

        log.info(f"Processing project: {project_name} ({len(sessions)} session files)")

        # Create project subdirectory
        project_save_dir = save_dir / project_name
        project_save_dir.mkdir(parents=True, exist_ok=True)

        project_converted = 0
        for session_path in sessions:
            session_id = session_path.stem

            # Skip if already converted
            if skip_existing and session_id[:8] in existing_session_ids:
                total_skipped += 1
                continue

            # Check cutoff time using file mtime
            if cutoff:
                file_mtime = datetime.fromtimestamp(session_path.stat().st_mtime)
                if file_mtime < cutoff:
                    total_skipped += 1
                    continue

            # Parse session
            try:
                messages = _parse_session(session_path)
            except Exception as e:
                log.warning(f"Failed to parse {session_path}: {e}")
                continue

            # Skip very short sessions
            if len(messages) < min_messages:
                total_too_short += 1
                continue

            # Get index metadata
            index_entry = index_entries.get(session_id)

            # Extract first user prompt for metadata
            first_prompt = None
            for m in messages:
                if m["role"] == "user" and m["text"]:
                    first_prompt = m["text"][:200]
                    break

            # Get timestamps
            first_ts = messages[0]["timestamp"] if messages else None
            last_ts = messages[-1]["timestamp"] if messages else None

            # Build metadata
            # Get the original project path for resume context
            project_path = index_entry.get("projectPath") if index_entry else None

            metadata = {
                "session_id": session_id,
                "project": project_name,
                "url": f"claude --resume {session_id}",
                "projectPath": project_path,
                "created": index_entry.get("created", first_ts) if index_entry else first_ts,
                "modified": index_entry.get("modified", last_ts) if index_entry else last_ts,
                "gitBranch": index_entry.get("gitBranch") if index_entry else None,
                "summary": index_entry.get("summary") if index_entry else None,
                "firstPrompt": first_prompt,
                "messageCount": len(messages),
            }

            # Convert to markdown
            markdown = _session_to_markdown(messages, metadata)

            # Write file
            filename = _make_filename(session_id, index_entry, first_prompt)
            output_path = project_save_dir / filename

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown)

            total_converted += 1
            project_converted += 1

        if project_converted > 0:
            log.info(f"  {project_name}: converted {project_converted} sessions")

    log.info(f"Done. Converted: {total_converted}, Skipped: {total_skipped}, Too short (<{min_messages} msgs): {total_too_short}")


def print_stats(project_filters: list[str] | None = None, min_messages: int = 4):
    """Print per-project session stats without converting."""
    if not CLAUDE_PROJECTS_DIR.exists():
        log.error(f"Claude projects directory not found: {CLAUDE_PROJECTS_DIR}")
        return

    stats = []
    total_sessions = 0
    total_qualifying = 0

    for project_dir in sorted(CLAUDE_PROJECTS_DIR.iterdir()):
        if not project_dir.is_dir():
            continue

        project_name = _extract_project_name(project_dir.name)

        if project_filters:
            if not any(f.lower() in project_name.lower() for f in project_filters):
                continue

        sessions = _discover_sessions(project_dir)
        if not sessions:
            continue

        index_entries = _parse_sessions_index(project_dir)

        qualifying = 0
        for session_path in sessions:
            try:
                messages = _parse_session(session_path)
                if len(messages) >= min_messages:
                    qualifying += 1
            except Exception:
                pass

        total_sessions += len(sessions)
        total_qualifying += qualifying
        stats.append((project_name, len(sessions), qualifying))

    # Print table
    print(f"\n{'Project':<45} {'Total':>8} {'Indexable':>10}")
    print("-" * 65)
    for name, total, qual in sorted(stats, key=lambda x: -x[2]):
        print(f"{name:<45} {total:>8} {qual:>10}")
    print("-" * 65)
    print(f"{'TOTAL':<45} {total_sessions:>8} {total_qualifying:>10}")
    print(f"\n(Indexable = sessions with >= {min_messages} conversation messages)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert Claude Code sessions to markdown for vector indexing")
    ap.add_argument("--saveMd", required=False, help="Directory to save markdown files (required unless --stats)")
    ap.add_argument("--skipExisting", action="store_true", help="Skip sessions already converted")
    ap.add_argument("--startFromTime", default=None, help="ISO datetime cutoff — only convert sessions modified after this time")
    ap.add_argument("--projects", nargs="+", default=None, help="Only process projects matching these names (substring match)")
    ap.add_argument("--minMessages", type=int, default=4, help="Skip sessions with fewer conversation messages (default: 4)")
    ap.add_argument("--stats", action="store_true", help="Print per-project session stats without converting")
    args = ap.parse_args()

    if args.stats:
        print_stats(
            project_filters=args.projects,
            min_messages=args.minMessages,
        )
    else:
        if not args.saveMd:
            ap.error("--saveMd is required when not using --stats")
        convert_sessions(
            save_md_path=args.saveMd,
            skip_existing=args.skipExisting,
            start_from_time=args.startFromTime,
            project_filters=args.projects,
            min_messages=args.minMessages,
        )
