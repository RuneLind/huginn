"""Tests for main.utils.ollama_cli — the headless Ollama chat wrapper."""
import json
from unittest.mock import MagicMock, patch

import pytest

from main.utils.ollama_cli import DEFAULT_MODEL, call_ollama


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
