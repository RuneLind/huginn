import json
import logging
import os
import re
import time
import urllib.error
import urllib.request

from main.core.contextual_prefix.prompts import PREFIX_SYSTEM_PROMPT, render_user_prompt


logger = logging.getLogger(__name__)


OLLAMA_URL = "http://localhost:11434/api/chat"
JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
TRAILING_COMMA_RE = re.compile(r",(\s*[\]}])\s*\Z")
PARSE_FAILURE_DUMP_DIR = os.environ.get("CONTEXTUAL_PREFIX_DEBUG_DIR", "./data/contextual_caches/parse_failures")


class OllamaBackend:
    """Ollama backend.

    Default model: qwen3.6:35b-a3b-nvfp4 (MoE, ~3B active params per token via MLX
    on Apple Silicon — fastest large-quality option in the Ollama library).

    Chunks are processed in batches (default 10/call). A doc with N chunks turns into
    ceil(N/10) Ollama calls. Keeps each call's generated-token count bounded — a single
    long doc no longer risks blowing num_predict — and limits blast radius when the
    model hiccups on JSON structure for one batch.
    """

    def __init__(
        self,
        model: str = "qwen3.6:35b-a3b-nvfp4",
        host: str = OLLAMA_URL,
        timeout: int = 600,
        num_predict: int = 4000,
        temperature: float = 0.2,
        chunks_per_call: int = 10,
    ):
        self.model = model
        self.host = host
        self.timeout = timeout
        self.num_predict = num_predict
        self.temperature = temperature
        self.chunks_per_call = max(1, chunks_per_call)

    @property
    def model_id(self) -> str:
        return f"ollama:{self.model}"

    def generate(self, document_text: str, chunks: list[str]) -> list[str]:
        if not chunks:
            return []

        prefixes: list[str] = []
        for batch_start in range(0, len(chunks), self.chunks_per_call):
            batch = chunks[batch_start: batch_start + self.chunks_per_call]
            batch_prefixes = self._generate_batch(document_text, batch)
            if len(batch_prefixes) != len(batch):
                logger.warning(
                    "Batch %d-%d of %d returned %d prefixes; aborting this doc's prefixing",
                    batch_start, batch_start + len(batch), len(chunks), len(batch_prefixes),
                )
                return []
            prefixes.extend(batch_prefixes)
        return prefixes

    def _generate_batch(self, document_text: str, batch_chunks: list[str]) -> list[str]:
        user_content = render_user_prompt(document_text, batch_chunks)
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": PREFIX_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "think": False,
            "format": "json",
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            self.host,
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            logger.warning("Ollama request failed (%s); returning empty prefixes for batch", e)
            return []

        content = (result.get("message") or {}).get("content", "").strip()
        return _parse_prefix_array(content, expected_count=len(batch_chunks))


def _parse_prefix_array(raw: str, expected_count: int) -> list[str]:
    if not raw:
        return []

    cleaned = JSON_FENCE_RE.sub("", raw).strip()
    cleaned = TRAILING_COMMA_RE.sub(r"\1", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        dump_path = _dump_parse_failure(cleaned, exc)
        logger.warning(
            "Could not parse JSON from model output (len=%d, expected_count=%d, err=%s).\n"
            "  first 400: %s\n"
            "  last  400: %s\n"
            "  raw saved to: %s",
            len(cleaned), expected_count, exc,
            cleaned[:400].replace("\n", "\\n"),
            cleaned[-400:].replace("\n", "\\n"),
            dump_path,
        )
        return []

    if isinstance(parsed, dict):
        for key in ("prefixes", "results", "chunks"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break

    if not isinstance(parsed, list):
        logger.warning("Expected JSON array of prefixes, got %s", type(parsed).__name__)
        return []

    prefixes = [str(p).strip() for p in parsed]

    if len(prefixes) != expected_count:
        logger.warning("Got %d prefixes, expected %d", len(prefixes), expected_count)

    return prefixes


def _dump_parse_failure(raw: str, exc: Exception) -> str:
    try:
        os.makedirs(PARSE_FAILURE_DUMP_DIR, exist_ok=True)
        path = os.path.join(PARSE_FAILURE_DUMP_DIR, f"parse-fail-{int(time.time() * 1000)}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# JSONDecodeError: {exc}\n")
            f.write(f"# raw output length: {len(raw)}\n")
            f.write("# -----\n")
            f.write(raw)
        return path
    except OSError as e:
        logger.warning("Failed to dump parse failure: %s", e)
        return "(dump failed)"
