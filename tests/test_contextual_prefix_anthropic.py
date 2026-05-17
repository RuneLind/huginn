from unittest.mock import MagicMock

import pytest

from main.core.contextual_prefix import ContextualCache, make_backend
from main.core.contextual_prefix.backends.anthropic_sdk_backend import (
    AnthropicSdkBackend,
    _build_client,
)


# ---------- helpers ----------

_AUTH_ENV_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")


def _clear_auth_env(monkeypatch):
    for var in _AUTH_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _patch_anthropic(monkeypatch) -> dict:
    """Install a stub for the `Anthropic` class. Returns a dict that will be
    populated with the kwargs the next `Anthropic(...)` call receives."""
    captured: dict = {}

    def fake_anthropic(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(
        "main.core.contextual_prefix.backends.anthropic_sdk_backend.Anthropic",
        fake_anthropic,
    )
    return captured


def _make_response(text: str, *, input_tokens: int = 10, output_tokens: int = 20):
    """Build a mock Anthropic SDK response whose .content[0].text == `text`."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.usage = MagicMock(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    return response


# ---------- construction / auth resolution ----------

def test_build_client_raises_when_no_env_set(monkeypatch):
    _clear_auth_env(monkeypatch)
    with pytest.raises(RuntimeError) as exc:
        _build_client()
    assert "ANTHROPIC_API_KEY" in str(exc.value)
    assert "ANTHROPIC_AUTH_TOKEN" in str(exc.value)
    assert "CLAUDE_CODE_OAUTH_TOKEN" in str(exc.value)


@pytest.mark.parametrize(
    "env, expected_kwargs",
    [
        (
            {"ANTHROPIC_API_KEY": "sk-test-123"},
            {"api_key": "sk-test-123"},
        ),
        (
            {"ANTHROPIC_AUTH_TOKEN": "oat-anthropic-xyz"},
            {"api_key": None, "auth_token": "oat-anthropic-xyz"},
        ),
        (
            {"CLAUDE_CODE_OAUTH_TOKEN": "oat-claude-code-abc"},
            {"api_key": None, "auth_token": "oat-claude-code-abc"},
        ),
        (
            # Precedence: ANTHROPIC_AUTH_TOKEN wins when both OAuth vars are set.
            {"ANTHROPIC_AUTH_TOKEN": "from-anthropic", "CLAUDE_CODE_OAUTH_TOKEN": "from-claude-code"},
            {"api_key": None, "auth_token": "from-anthropic"},
        ),
    ],
    ids=["api_key", "anthropic_auth_token", "claude_code_oauth_token", "anthropic_auth_token_wins_over_claude_code"],
)
def test_build_client_resolves_auth_in_priority_order(monkeypatch, env, expected_kwargs):
    _clear_auth_env(monkeypatch)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    captured = _patch_anthropic(monkeypatch)
    _build_client()
    assert captured == expected_kwargs


# ---------- model_id ----------

def test_model_id_property():
    client = MagicMock()
    backend = AnthropicSdkBackend(model="claude-haiku-4-5", client=client)
    assert backend.model_id == "anthropic:claude-haiku-4-5"


# ---------- generate() ----------

def test_generate_happy_path_returns_parsed_prefixes():
    client = MagicMock()
    client.messages.create.return_value = _make_response(
        '["prefix one", "prefix two", "prefix three"]'
    )

    backend = AnthropicSdkBackend(model="claude-haiku-4-5", client=client)
    result = backend.generate("doc text", ["c1", "c2", "c3"])

    assert result == ["prefix one", "prefix two", "prefix three"]
    assert client.messages.create.call_count == 1
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5"
    assert kwargs["max_tokens"] == 4000
    assert kwargs["timeout"] == 180
    assert isinstance(kwargs["system"], str) and kwargs["system"]
    assert kwargs["messages"][0]["role"] == "user"


def test_generate_empty_chunks_returns_empty_and_does_not_call_sdk():
    client = MagicMock()
    backend = AnthropicSdkBackend(model="claude-haiku-4-5", client=client)
    result = backend.generate("doc text", [])
    assert result == []
    client.messages.create.assert_not_called()


def test_generate_returns_empty_when_sdk_raises():
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("API blew up")
    backend = AnthropicSdkBackend(model="claude-haiku-4-5", client=client)
    # Must not propagate.
    result = backend.generate("doc text", ["c1", "c2"])
    assert result == []


def test_generate_returns_empty_when_response_is_malformed_json():
    client = MagicMock()
    client.messages.create.return_value = _make_response("definitely not json at all")
    backend = AnthropicSdkBackend(model="claude-haiku-4-5", client=client)
    result = backend.generate("doc text", ["c1", "c2"])
    assert result == []


# ---------- wiring through make_backend / BackendSpec ----------

def test_make_backend_returns_anthropic_sdk_backend(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-xyz")
    _patch_anthropic(monkeypatch)
    backend = make_backend("anthropic:claude-haiku-4-5")
    assert isinstance(backend, AnthropicSdkBackend)
    assert backend.model_id == "anthropic:claude-haiku-4-5"


def test_make_backend_requires_model_for_anthropic():
    with pytest.raises(ValueError):
        make_backend("anthropic")


# ---------- cache key non-collision across backends ----------

def test_cache_keys_do_not_collide_across_backends(tmp_path):
    """A prefix cached under claude-code:claude-haiku-4-5 must not satisfy
    a lookup keyed on anthropic:claude-haiku-4-5 (same model name, different backend)."""
    cache = ContextualCache(str(tmp_path / "cache.json"))

    doc_id = "doc-1"
    chunk_text = "chunk body text for testing"

    # Pretend the claude-code path wrote a prefix earlier.
    cache.put(doc_id, chunk_text, "claude-code:claude-haiku-4-5", "old-cli-prefix")
    assert cache.get(doc_id, chunk_text, "claude-code:claude-haiku-4-5") == "old-cli-prefix"

    # A lookup with the new SDK backend's model_id must miss — even though
    # the underlying Haiku model name is identical.
    sdk_model_id = "anthropic:claude-haiku-4-5"
    assert cache.get(doc_id, chunk_text, sdk_model_id) is None
