"""Shared helpers for calling Claude CLI headless and processing markdown documents."""
import json
import os
import re
import subprocess

FRONTMATTER_RE = re.compile(r'^---\n(.*?)\n---', re.DOTALL)


def call_claude(prompt: str, model: str = "claude-haiku-4-5-20251001",
                timeout: int = 60) -> str:
    """Call claude CLI and return the result text. Kills process on timeout."""
    # Strip CLAUDECODE env vars to allow spawning from within a Claude Code session
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("CLAUDECODE") and k != "CLAUDE_CODE_ENTRYPOINT"}
    proc = subprocess.Popen(
        ["claude", "-p", prompt, "--output-format", "json", "--model", model],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise RuntimeError(f"claude CLI timed out after {timeout}s")

    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI failed (exit {proc.returncode}): {stderr[:200]}")

    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"Bad JSON from claude CLI: {stdout[:200]}")

    if envelope.get("is_error"):
        raise RuntimeError(f"Claude error: {envelope.get('result', 'unknown')}")

    return envelope.get("result", "")


def extract_frontmatter(content: str) -> dict[str, str]:
    """Extract frontmatter fields and return as a dict."""
    match = FRONTMATTER_RE.match(content)
    if not match:
        return {}
    fields = {}
    for line in match.group(1).split('\n'):
        if ':' in line:
            key, _, value = line.partition(':')
            fields[key.strip()] = value.strip().strip('"')
    return fields


def get_content_excerpt(content: str, max_chars: int = 2000) -> str:
    """Get content without frontmatter, truncated to max_chars."""
    stripped = FRONTMATTER_RE.sub('', content).strip()
    if len(stripped) > max_chars:
        return stripped[:max_chars] + "..."
    return stripped


def extract_json_array(text: str) -> list | None:
    """Robustly extract a JSON array from text."""
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    return None
