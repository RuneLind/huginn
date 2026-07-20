"""Headless Ollama chat wrapper — the local-model analogue of ``main.utils.claude_cli``.

Sends a single prompt to Ollama's ``/api/chat`` endpoint and returns the
message content. Mirrors ``call_claude``'s shape (keyword-only ``model`` /
``timeout``, ``RuntimeError`` on failure) so the tagging scripts can swap
backends with a one-line change.

This is the single ``/api/chat`` transport shared by every local-model caller
under ``main/`` and ``scripts/`` — the contextual-prefix ``OllamaBackend`` and
the knowledge-graph entity extractor both route through it. Optional ``system``
prepends a system message; optional ``options`` composes with ``temperature``
(base ``{"temperature": temperature}`` shallow-merged under the caller's dict,
so ``num_predict`` or a ``temperature`` override rides alongside without
dropping the default). ``format:"json"`` + ``think:false`` keep a
reasoning-capable local model (e.g. qwen3) terse and machine-parseable.

Callers that need to swallow failure instead of propagating must wrap this in
their own try/except: it raises ``RuntimeError`` on transport/JSON failure AND
on a response ``error`` field, and returns a raw string (no parsing).
"""
import json
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"

DEFAULT_MODEL = "qwen3.6:35b-a3b-nvfp4"


def call_ollama(prompt: str, *, model: str = DEFAULT_MODEL, timeout: int = 120,
                host: str = OLLAMA_URL, temperature: float = 0.2,
                system: str | None = None, options: dict | None = None) -> str:
    """Run a single-message Ollama chat and return the assistant message content.

    ``system`` prepends a system message before the user prompt. ``options``
    is shallow-merged over the base ``{"temperature": temperature}`` — so
    ``options={"temperature": 0, "num_predict": 3000}`` overrides the default
    temperature and adds ``num_predict``, while ``temperature`` alone still
    applies when ``options`` is omitted.

    Raises ``RuntimeError`` on a request/transport error, malformed response,
    or a response ``error`` field, matching ``call_claude`` so callers can
    treat both backends uniformly.
    """
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    merged_options = {"temperature": temperature}
    if options:
        merged_options.update(options)

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "format": "json",
        "options": merged_options,
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
