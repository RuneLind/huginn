"""Headless Ollama chat wrapper — the local-model analogue of ``main.utils.claude_cli``.

Sends a single prompt to Ollama's ``/api/chat`` endpoint and returns the
message content. Mirrors ``call_claude``'s shape (keyword-only ``model`` /
``timeout``, ``RuntimeError`` on failure) so the tagging scripts can swap
backends with a one-line change.

Extracted from ``main.core.contextual_prefix.backends.ollama_backend`` (the
HTTP-chat core) so a JSON-array-returning caller — tagging, prefixing — shares
one transport. ``format:"json"`` + ``think:false`` keep a reasoning-capable
local model (e.g. qwen3) terse and machine-parseable.
"""
import json
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"

DEFAULT_MODEL = "qwen3.6:35b-a3b-nvfp4"


def call_ollama(prompt: str, *, model: str = DEFAULT_MODEL, timeout: int = 120,
                host: str = OLLAMA_URL, temperature: float = 0.2) -> str:
    """Run a single-message Ollama chat and return the assistant message content.

    Raises ``RuntimeError`` on a request/transport error or malformed response,
    matching ``call_claude`` so callers can treat both backends uniformly.
    """
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": temperature},
    }).encode("utf-8")

    req = urllib.request.Request(
        host,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError) as e:
        raise RuntimeError(f"Ollama request failed: {e}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Bad JSON from Ollama: {e}")

    if result.get("error"):
        raise RuntimeError(f"Ollama error: {result['error']}")

    return (result.get("message") or {}).get("content", "")
