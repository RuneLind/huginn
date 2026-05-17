from dataclasses import dataclass

from main.core.contextual_prefix.prefix_generator import PrefixGenerator


@dataclass(frozen=True)
class BackendSpec:
    kind: str   # "none" | "ollama" | "claude-code" | "echo"
    model: str  # e.g. "qwen3.6:35b-a3b-nvfp4" or "claude-haiku-4-5"

    @property
    def model_id(self) -> str:
        return f"{self.kind}:{self.model}" if self.model else self.kind

    @classmethod
    def parse(cls, spec: str) -> "BackendSpec":
        spec = (spec or "").strip()
        if not spec or spec.lower() == "none":
            return cls(kind="none", model="")
        if ":" in spec:
            kind, model = spec.split(":", 1)
            return cls(kind=kind.strip().lower(), model=model.strip())
        return cls(kind=spec.lower(), model="")


def make_backend(spec: BackendSpec | str) -> PrefixGenerator | None:
    if isinstance(spec, str):
        spec = BackendSpec.parse(spec)

    if spec.kind == "none":
        return None
    if spec.kind == "echo":
        from main.core.contextual_prefix.backends.echo import EchoBackend
        return EchoBackend(model_id=spec.model_id)
    if spec.kind == "ollama":
        from main.core.contextual_prefix.backends.ollama_backend import OllamaBackend
        if not spec.model:
            raise ValueError("ollama backend requires a model name, e.g. ollama:qwen3.6:35b-a3b-nvfp4")
        return OllamaBackend(model=spec.model)
    if spec.kind == "claude-code":
        from main.core.contextual_prefix.backends.claude_code_backend import ClaudeCodeBackend
        if not spec.model:
            raise ValueError("claude-code backend requires a model name, e.g. claude-code:claude-haiku-4-5")
        return ClaudeCodeBackend(model=spec.model)

    raise ValueError(f"Unknown contextual-prefix backend: {spec.kind}")
