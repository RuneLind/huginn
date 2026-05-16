import json
import logging
import re
import urllib.error
import urllib.request

from main.core.contextual_prefix.prompts import PREFIX_SYSTEM_PROMPT, render_user_prompt


logger = logging.getLogger(__name__)


OLLAMA_URL = "http://localhost:11434/api/chat"
JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class OllamaBackend:
    """Ollama backend.

    Default model: qwen3.6:35b-a3b-nvfp4 (MoE, ~3B active params per token via MLX
    on Apple Silicon — fastest large-quality option in the Ollama library).
    """

    def __init__(
        self,
        model: str = "qwen3.6:35b-a3b-nvfp4",
        host: str = OLLAMA_URL,
        timeout: int = 600,
        num_predict: int = 1500,
        temperature: float = 0.2,
    ):
        self.model = model
        self.host = host
        self.timeout = timeout
        self.num_predict = num_predict
        self.temperature = temperature

    @property
    def model_id(self) -> str:
        return f"ollama:{self.model}"

    def generate(self, document_text: str, chunks: list[str]) -> list[str]:
        user_content = render_user_prompt(document_text, chunks)
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
            logger.warning("Ollama request failed (%s); returning empty prefixes", e)
            return []

        content = (result.get("message") or {}).get("content", "").strip()
        return _parse_prefix_array(content, expected_count=len(chunks))


def _parse_prefix_array(raw: str, expected_count: int) -> list[str]:
    if not raw:
        return []

    cleaned = JSON_FENCE_RE.sub("", raw).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Could not parse JSON from model output (len=%d, first 200 chars): %s",
                       len(cleaned), cleaned[:200])
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
