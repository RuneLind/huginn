from typing import Protocol, runtime_checkable


@runtime_checkable
class PrefixGenerator(Protocol):
    """Generates a short context prefix per chunk, anchoring each chunk in its document.

    Implementations should:
    - Process all chunks of a single document in one call (lets the backend reuse
      doc context across chunks via prompt caching / KV reuse).
    - Return prefixes in the same order as the input chunks.
    - Return an empty string for chunks they decline to prefix; upstream skips those.
    """

    model_id: str

    def generate(self, document_text: str, chunks: list[str]) -> list[str]:
        ...
