import logging
import os

from anthropic import Anthropic

from main.core.contextual_prefix.parsing import parse_prefix_array
from main.core.contextual_prefix.prompts import PREFIX_SYSTEM_PROMPT, render_user_prompt


logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 4000  # matches OllamaBackend.num_predict; per-doc batches stay bounded
_DEFAULT_TIMEOUT_S = 180


class AnthropicSdkBackend:
    """Direct Anthropic SDK backend.

    Mirrors ClaudeCodeBackend.generate() shape but skips the CLI subprocess + MCP
    catalog injection. ~7x faster wall time on Haiku per measured muninn PR #120
    A/B (same model, same prompt shape).

    Auth resolution (instance-cached client):
      1. ANTHROPIC_API_KEY        -> x-api-key header (production / shared)
      2. ANTHROPIC_AUTH_TOKEN     -> Authorization: Bearer (SDK-native OAuth var)
      3. CLAUDE_CODE_OAUTH_TOKEN  -> Authorization: Bearer (personal Max-subscription
                                    dev via `claude setup-token`)
    Raises clear error if none are set.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        timeout: int = _DEFAULT_TIMEOUT_S,
        client: Anthropic | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client = client or _build_client()

    @property
    def model_id(self) -> str:
        return f"anthropic:{self.model}"

    def generate(self, document_text: str, chunks: list[str]) -> list[str]:
        if not chunks:
            return []
        # Each prefix is ~60-100 output tokens; budget ~80 to leave JSON-structure headroom.
        # Past this, the JSON array would be silently truncated and ChunkPrefixer would
        # drop all prefixes for the doc on the count mismatch. Warn so a future giant doc
        # is visible in logs rather than mysteriously un-prefixed.
        if len(chunks) > self.max_tokens // 80:
            logger.warning(
                "doc has %d chunks; near max_tokens=%d ceiling for the response. "
                "Consider raising max_tokens or batching.",
                len(chunks), self.max_tokens,
            )
        user_prompt = render_user_prompt(document_text, chunks)
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=PREFIX_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                timeout=self.timeout,
            )
        except Exception as e:
            logger.warning("Anthropic SDK call failed (%s); returning empty prefixes", e)
            return []
        text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        usage = getattr(response, "usage", None)
        if usage is not None:
            logger.info(
                "anthropic_sdk usage: input=%s output=%s cache_read=%s cache_creation=%s",
                getattr(usage, "input_tokens", None),
                getattr(usage, "output_tokens", None),
                getattr(usage, "cache_read_input_tokens", None),
                getattr(usage, "cache_creation_input_tokens", None),
            )
        return parse_prefix_array(text, expected_count=len(chunks))


def _build_client() -> Anthropic:
    # CLAUDE_CODE_OAUTH_TOKEN is a Claude-Code convention the SDK doesn't auto-resolve;
    # the explicit reads are what make that token usable here.
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return Anthropic(api_key=api_key)
    auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if auth_token:
        return Anthropic(api_key=None, auth_token=auth_token)
    raise RuntimeError(
        "anthropic backend: none of ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, "
        "or CLAUDE_CODE_OAUTH_TOKEN is set"
    )
