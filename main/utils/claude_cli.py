"""Headless Claude CLI wrapper used across Huginn ingestion pipelines.

Spawns ``claude -p - --output-format json --model <model>`` and returns the
``result`` field of the JSON envelope. The prompt is fed via stdin so callers
do not have to worry about OS ``ARG_MAX`` limits with long inputs (e.g.
YouTube transcripts). ``CLAUDECODE*`` and ``CLAUDE_CODE_ENTRYPOINT`` env vars
are stripped so the subprocess can be safely spawned from inside an active
Claude Code session.

Errors surface as ``RuntimeError`` (non-zero exit, bad JSON, ``is_error``
envelope, timeout) or ``FileNotFoundError`` (``claude`` not on PATH). Callers
that need typed HTTP errors map these at the call site.
"""
import json
import os
import subprocess


def call_claude(prompt: str, *, model: str, timeout: int = 60) -> str:
    """Run ``claude`` headless and return the ``result`` text from its JSON envelope."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("CLAUDECODE") and k != "CLAUDE_CODE_ENTRYPOINT"}
    cmd = ["claude", "-p", "-", "--output-format", "json", "--model", model]

    proc = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()[:500] or "unknown error"
        raise RuntimeError(f"claude CLI failed (exit {proc.returncode}): {stderr}")

    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"Bad JSON from claude CLI: {proc.stdout[:200]}")

    if envelope.get("is_error"):
        raise RuntimeError(f"Claude error: {envelope.get('result', 'unknown')}")

    return envelope.get("result", "")
