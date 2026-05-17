import json
import logging
import shutil
import subprocess

from main.core.contextual_prefix.backends.ollama_backend import _parse_prefix_array
from main.core.contextual_prefix.prompts import PREFIX_SYSTEM_PROMPT, render_user_prompt


logger = logging.getLogger(__name__)


class ClaudeCodeBackend:
    """Headless Claude Code backend.

    Invokes `claude -p` with the given model. Used to send prefix-generation jobs
    to Claude Haiku (default) via the user's existing Claude Code installation —
    no API key plumbing required.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        cli: str = "claude",
        timeout: int = 180,
    ):
        self.model = model
        self.cli = cli
        self.timeout = timeout

    @property
    def model_id(self) -> str:
        return f"claude-code:{self.model}"

    def generate(self, document_text: str, chunks: list[str]) -> list[str]:
        if not shutil.which(self.cli):
            logger.warning("Claude Code CLI (%s) not on PATH; returning empty prefixes", self.cli)
            return []

        prompt = f"{PREFIX_SYSTEM_PROMPT}\n\n{render_user_prompt(document_text, chunks)}"

        try:
            result = subprocess.run(
                [self.cli, "-p", prompt, "--model", self.model, "--output-format", "json"],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Claude Code call timed out (timeout=%ds, doc chunks=%d)", self.timeout, len(chunks))
            return []

        if result.returncode != 0:
            logger.warning("Claude Code exited %d: %s", result.returncode, result.stderr.strip()[:400])
            return []

        text = _extract_result_text(result.stdout)
        return _parse_prefix_array(text, expected_count=len(chunks))


def _extract_result_text(stdout: str) -> str:
    """Pull the assistant's text out of Claude Code's --output-format json envelope."""
    stdout = stdout.strip()
    if not stdout:
        return ""
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        return stdout

    if isinstance(parsed, dict):
        for key in ("result", "output", "response", "text", "content"):
            value = parsed.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, list) and value and isinstance(value[0], dict):
                inner = value[0].get("text") or value[0].get("content")
                if isinstance(inner, str):
                    return inner
    return stdout
