import json
import logging
import urllib.error
import urllib.request

from main.core.contextual_prefix.parsing import parse_prefix_array
from main.core.contextual_prefix.prompts import PREFIX_SYSTEM_PROMPT, render_user_prompt


logger = logging.getLogger(__name__)


OLLAMA_URL = "http://localhost:11434/api/chat"


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
        return parse_prefix_array(content, expected_count=len(batch_chunks))
