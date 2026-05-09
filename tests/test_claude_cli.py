"""Tests for main.utils.claude_cli — the headless Claude CLI wrapper."""
import json
import os
import subprocess
from unittest.mock import patch

import pytest

from main.utils.claude_cli import call_claude


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class TestCallClaude:
    def test_returns_envelope_result(self):
        envelope = json.dumps({"result": "the summary"})
        with patch("main.utils.claude_cli.subprocess.run", return_value=_completed(stdout=envelope)) as run:
            assert call_claude("hi", model="sonnet") == "the summary"
        # cmd uses stdin (-p -) and selected model
        args, kwargs = run.call_args
        assert args[0] == ["claude", "-p", "-", "--output-format", "json", "--model", "sonnet"]
        assert kwargs["input"] == "hi"

    def test_strips_claudecode_env_vars(self):
        with patch.dict(os.environ, {"CLAUDECODE": "1", "CLAUDE_CODE_ENTRYPOINT": "cli", "PATH": "/usr/bin"}, clear=True):
            with patch("main.utils.claude_cli.subprocess.run", return_value=_completed(stdout='{"result": "x"}')) as run:
                call_claude("hi", model="sonnet")
        env = run.call_args.kwargs["env"]
        assert "CLAUDECODE" not in env
        assert "CLAUDE_CODE_ENTRYPOINT" not in env
        assert env["PATH"] == "/usr/bin"

    def test_passes_timeout_through(self):
        with patch("main.utils.claude_cli.subprocess.run", return_value=_completed(stdout='{"result": ""}')) as run:
            call_claude("hi", model="sonnet", timeout=42)
        assert run.call_args.kwargs["timeout"] == 42

    def test_default_timeout_is_60(self):
        with patch("main.utils.claude_cli.subprocess.run", return_value=_completed(stdout='{"result": ""}')) as run:
            call_claude("hi", model="sonnet")
        assert run.call_args.kwargs["timeout"] == 60

    def test_nonzero_exit_raises_runtime_error(self):
        proc = _completed(stdout="", stderr="boom", returncode=2)
        with patch("main.utils.claude_cli.subprocess.run", return_value=proc):
            with pytest.raises(RuntimeError, match="exit 2.*boom"):
                call_claude("hi", model="sonnet")

    def test_nonzero_exit_with_empty_stderr_uses_unknown(self):
        proc = _completed(stdout="", stderr="", returncode=1)
        with patch("main.utils.claude_cli.subprocess.run", return_value=proc):
            with pytest.raises(RuntimeError, match="unknown error"):
                call_claude("hi", model="sonnet")

    def test_bad_json_raises_runtime_error(self):
        with patch("main.utils.claude_cli.subprocess.run", return_value=_completed(stdout="not json")):
            with pytest.raises(RuntimeError, match="Bad JSON"):
                call_claude("hi", model="sonnet")

    def test_is_error_envelope_raises_runtime_error(self):
        envelope = json.dumps({"is_error": True, "result": "rate limited"})
        with patch("main.utils.claude_cli.subprocess.run", return_value=_completed(stdout=envelope)):
            with pytest.raises(RuntimeError, match="rate limited"):
                call_claude("hi", model="sonnet")

    def test_missing_result_field_returns_empty_string(self):
        envelope = json.dumps({"some_other_field": "x"})
        with patch("main.utils.claude_cli.subprocess.run", return_value=_completed(stdout=envelope)):
            assert call_claude("hi", model="sonnet") == ""

    def test_timeout_propagates(self):
        exc = subprocess.TimeoutExpired(cmd=["claude"], timeout=5)
        with patch("main.utils.claude_cli.subprocess.run", side_effect=exc):
            with pytest.raises(subprocess.TimeoutExpired):
                call_claude("hi", model="sonnet", timeout=5)

    def test_file_not_found_propagates(self):
        with patch("main.utils.claude_cli.subprocess.run", side_effect=FileNotFoundError("claude")):
            with pytest.raises(FileNotFoundError):
                call_claude("hi", model="sonnet")
