"""CLI wiring for ``scripts/traces/extract_query_doc_pairs.py``.

``--output`` lost its default because that default was a private sub-repo path,
which must not appear in this public repo. Making it ``required=True`` overshot:
the path is only read to write the file and to pre-load trace IDs under
``--append``, so it gated ``--dry-run`` — the one mode that never touches it.
"""

import importlib.util
import os
import re

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A far-future --since matches no session, so main() returns without scanning.
_NO_SESSIONS = ["--since", "2999-01-01"]


@pytest.fixture(scope="module")
def script():
    """Import the script by path — ``scripts/`` is not an importable package."""
    path = os.path.join(_REPO_ROOT, "scripts", "traces", "extract_query_doc_pairs.py")
    spec = importlib.util.spec_from_file_location("extract_query_doc_pairs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_run_does_not_require_output(script, capsys):
    script.main(["--dry-run", *_NO_SESSIONS])
    assert "not writing output" in capsys.readouterr().out


def test_missing_output_is_an_error(script):
    with pytest.raises(SystemExit) as exc:
        script.main(_NO_SESSIONS)
    assert exc.value.code == 2


def test_append_requires_output_even_when_dry_run(script):
    # --append reads the existing file before the --dry-run early return.
    with pytest.raises(SystemExit) as exc:
        script.main(["--dry-run", "--append", *_NO_SESSIONS])
    assert exc.value.code == 2


def test_help_names_no_private_sub_repo_and_no_shell_metacharacters(script, capsys):
    with pytest.raises(SystemExit):
        script.build_parser().parse_args(["--help"])
    help_text = capsys.readouterr().out
    assert "<domain>" not in help_text
    # Only the glob placeholder may name a sub-repo; any concrete huginn-<name> is a leak.
    assert re.search(r"huginn-(?!\*)[A-Za-z]", help_text) is None
    assert "huginn-*/scripts/benchmarks" in help_text
