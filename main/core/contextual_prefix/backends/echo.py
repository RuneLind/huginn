class EchoBackend:
    """Deterministic test backend — returns a fixed prefix per chunk index.

    Use in tests and during Phase 1 plumbing so we can verify end-to-end
    wiring without LLM cost or network.
    """

    def __init__(self, model_id: str = "echo:v1"):
        self.model_id = model_id

    def generate(self, document_text: str, chunks: list[str]) -> list[str]:
        doc_len = len(document_text)
        return [f"[echo prefix for chunk {i + 1} of {len(chunks)}; doc len={doc_len}]" for i in range(len(chunks))]
