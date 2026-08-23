"""Tests for main.utils.ollama_cli — the headless Ollama chat wrapper."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from main.utils.ollama_cli import DEFAULT_MODEL, call_ollama

REPO_ROOT = Path(__file__).resolve().parent.parent


class _FakeResp:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _resp(payload: dict) -> _FakeResp:
    return _FakeResp(json.dumps(payload).encode("utf-8"))


class TestCallOllama:
    def test_returns_message_content(self):
        body = _resp({"message": {"content": '["a", "b"]'}})
        with patch("main.utils.ollama_cli.urllib.request.urlopen", return_value=body):
            assert call_ollama("hi", model="m") == '["a", "b"]'

    def test_request_payload_shape(self):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["data"] = json.loads(req.data)
            captured["timeout"] = timeout
            return _resp({"message": {"content": "ok"}})

        with patch("main.utils.ollama_cli.urllib.request.urlopen", side_effect=fake_urlopen):
            call_ollama("prompt-text", model="qwen", timeout=90)

        data = captured["data"]
        assert data["model"] == "qwen"
        assert data["stream"] is False
        assert data["think"] is False
        assert data["format"] == "json"
        assert data["messages"] == [{"role": "user", "content": "prompt-text"}]
        assert captured["timeout"] == 90

    def test_system_prepends_system_message(self):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["data"] = json.loads(req.data)
            return _resp({"message": {"content": "ok"}})

        with patch("main.utils.ollama_cli.urllib.request.urlopen", side_effect=fake_urlopen):
            call_ollama("user-text", model="m", system="sys-text")

        assert captured["data"]["messages"] == [
            {"role": "system", "content": "sys-text"},
            {"role": "user", "content": "user-text"},
        ]

    def test_no_system_omits_system_message(self):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["data"] = json.loads(req.data)
            return _resp({"message": {"content": "ok"}})

        with patch("main.utils.ollama_cli.urllib.request.urlopen", side_effect=fake_urlopen):
            call_ollama("user-text", model="m")

        assert captured["data"]["messages"] == [{"role": "user", "content": "user-text"}]

    def test_options_shallow_merge_over_temperature(self):
        """options composes with temperature: base {temperature} with the
        caller's options merged on top (so an explicit temperature in options
        wins and extra keys like num_predict ride alongside)."""
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["data"] = json.loads(req.data)
            return _resp({"message": {"content": "ok"}})

        with patch("main.utils.ollama_cli.urllib.request.urlopen", side_effect=fake_urlopen):
            call_ollama("hi", model="m", temperature=0.2,
                        options={"temperature": 0, "num_predict": 3000})

        assert captured["data"]["options"] == {"temperature": 0, "num_predict": 3000}

    def test_options_none_keeps_temperature_only(self):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["data"] = json.loads(req.data)
            return _resp({"message": {"content": "ok"}})

        with patch("main.utils.ollama_cli.urllib.request.urlopen", side_effect=fake_urlopen):
            call_ollama("hi", model="m", temperature=0.7)

        assert captured["data"]["options"] == {"temperature": 0.7}

    def test_options_adds_key_without_dropping_temperature(self):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["data"] = json.loads(req.data)
            return _resp({"message": {"content": "ok"}})

        with patch("main.utils.ollama_cli.urllib.request.urlopen", side_effect=fake_urlopen):
            call_ollama("hi", model="m", temperature=0.2, options={"num_predict": 4000})

        assert captured["data"]["options"] == {"temperature": 0.2, "num_predict": 4000}

    def test_default_model(self):
        def fake_urlopen(req, timeout=None):
            assert json.loads(req.data)["model"] == DEFAULT_MODEL
            return _resp({"message": {"content": "ok"}})

        with patch("main.utils.ollama_cli.urllib.request.urlopen", side_effect=fake_urlopen):
            call_ollama("hi")

    def test_missing_message_returns_empty(self):
        with patch("main.utils.ollama_cli.urllib.request.urlopen", return_value=_resp({})):
            assert call_ollama("hi", model="m") == ""

    def test_error_field_raises(self):
        with patch("main.utils.ollama_cli.urllib.request.urlopen",
                   return_value=_resp({"error": "model not found"})):
            with pytest.raises(RuntimeError, match="model not found"):
                call_ollama("hi", model="m")

    def test_url_error_raises_runtime_error(self):
        import urllib.error

        with patch("main.utils.ollama_cli.urllib.request.urlopen",
                   side_effect=urllib.error.URLError("connection refused")):
            with pytest.raises(RuntimeError, match="Ollama request failed"):
                call_ollama("hi", model="m")

    def test_bad_json_raises_runtime_error(self):
        with patch("main.utils.ollama_cli.urllib.request.urlopen",
                   return_value=_FakeResp(b"not json")):
            with pytest.raises(RuntimeError, match="Bad JSON"):
                call_ollama("hi", model="m")


class TestDefaultModelPropagation:
    """Who inherits DEFAULT_MODEL and who deliberately does not.

    The constant is the general-purpose local model. The tagging scripts take it
    so a model swap is a one-line change; the knowledge-graph extractor and the
    contextual-prefix backend pin their own, because their caches are keyed by
    document id rather than by model — swapping under them silently produces a
    hybrid graph mixing two models' output (docs/knowledge-graph-when-to-use-
    what.md).

    The propagation assertions are deliberately RELATIONAL (`== DEFAULT_MODEL`),
    so they keep holding across a model bump. That leaves the constant's own
    value unguarded — every one of them passes if it silently reverts — so
    `test_the_default_model_is_the_one_the_campaign_chose` pins it separately.
    A bump then costs one line here, which is the right price: choosing the
    machine's general-purpose model is a decision worth restating out loud.
    """

    def test_the_default_model_is_the_one_the_campaign_chose(self):
        """The one assertion that is allowed to name a model.

        Without it the constant can be reverted with the whole suite green: the
        propagation tests compare things TO DEFAULT_MODEL, and the extractor
        test only requires it to DIFFER, so all four hold at any value. Pinned
        because moving it was a deliberate deliverable (the A/B is in the PR
        that introduced this line), not an incidental default.
        """
        assert DEFAULT_MODEL == "qwen3.8:27b-mlx"

    def test_the_tagging_scripts_inherit_it(self):
        from scripts.tagging import discover_tags, tag_documents
        assert tag_documents.DEFAULT_OLLAMA_MODEL == DEFAULT_MODEL
        assert discover_tags.DEFAULT_OLLAMA_MODEL == DEFAULT_MODEL

    def test_the_tagging_argparse_defaults_are_the_same_constant(self):
        """The import is not the contract — the flag default is what a run uses."""
        import re

        for path in ("scripts/tagging/tag_documents.py", "scripts/tagging/discover_tags.py"):
            source = (REPO_ROOT / path).read_text(encoding="utf-8")
            assert re.search(r'--ollama-model"[^)]*default=DEFAULT_OLLAMA_MODEL', source), path

    def test_the_graph_extractor_pins_its_own(self):
        """Its `--model` default is a LITERAL, and this is which one.

        `!= DEFAULT_MODEL` was the whole assertion, which holds for any literal
        at all — including one someone changes to a third model by accident. The
        extraction cache is keyed by document id rather than by model, so a
        changed pin silently produces a graph mixing two models' output; the
        value is the thing worth guarding, and the A/B that would justify moving
        it has not been run at a size that decides.
        """
        import re

        source = (REPO_ROOT / "scripts" / "knowledge_graph"
                  / "extract_entities_llm.py").read_text(encoding="utf-8")
        match = re.search(r'"--model",\s*default="([^"]+)"', source)
        assert match, "the extractor no longer pins a literal --model default"
        assert match.group(1) == "qwen3.6:35b-a3b-coding-nvfp4"

    def test_the_contextual_prefix_ollama_backend_pins_its_own(self):
        """Same argument, same cache shape: the contextual prefixes stored on a
        collection carry no model id, so re-running under another model leaves a
        collection whose chunks were prefixed by two."""
        from main.core.contextual_prefix.backends.ollama_backend import OllamaBackend

        import inspect

        pinned = inspect.signature(OllamaBackend.__init__).parameters["model"].default
        assert pinned == "qwen3.6:35b-a3b-nvfp4" != DEFAULT_MODEL

    def test_the_sweep_inherits_it(self):
        from scripts.audit import sensitivity_sweep
        assert sensitivity_sweep.build_parser().get_default("model") == DEFAULT_MODEL
