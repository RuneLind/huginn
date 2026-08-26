"""Node and edge order must not depend on filesystem enumeration order.

``extract_jira_graph.py`` walks its source tree twice — once for the active
issues, once for the ``.excluded/`` stub subtasks that enrich the graph — and
both walks feed ``issues``/``nodes`` in the order they enumerate. Raw
``Path.rglob`` order is a filesystem detail, so without ``sorted()`` the graph's
node order and its non-cross-ref edges are whatever the volume happened to hand
back.

The trap this file exists to avoid: on the APFS volume this was written on,
enumeration order is a pure function of the file-name set, so two consecutive
runs agree *whether or not the fix is present*. A rerun test would pass on the
unfixed extractor and prove nothing. So the check forces a pathological order
instead — reversed enumeration stands in for "a corpus rebuilt on another
filesystem", which is the case that would otherwise diff every node at once.

``sitecustomize`` is imported at interpreter startup for anything on
``PYTHONPATH``, which is how the subprocess gets its ``rglob`` patched without
the extractor knowing it is under test. The gate test's ``_run`` cannot be
reused for it: that helper pins ``PYTHONPATH`` to the repo root *after* merging
``extra_env``, deliberately, so a caller cannot redirect the privacy root. So
this file drives the subprocess itself and keeps the same pinning for
``HUGINN_PRIVACY_ROOT``.
"""
import json
import os
import subprocess
import sys

import pytest

from tests.test_jira_graph_gate import (  # noqa: E402
    EXTRACTOR,
    IN_SCOPE_SOURCE,
    REPO_ROOT,
    _issue,
    _map,
)

REVERSING_SITECUSTOMIZE = """
import pathlib

_real = pathlib.Path.rglob


def _reversed_rglob(self, pattern):
    return iter(sorted(_real(self, pattern), reverse=True))


pathlib.Path.rglob = _reversed_rglob
"""


def _run(root, source, output, *, reverse_enumeration=False):
    """Drive the extractor, optionally under reversed enumeration order.

    ``HUGINN_PRIVACY_ROOT`` is pinned to ``root`` here for the same reason the
    gate test pins it: without it a test discovers the operator's real map.
    """
    path_entries = [str(REPO_ROOT)]
    if reverse_enumeration:
        (root / "sitecustomize.py").write_text(REVERSING_SITECUSTOMIZE, encoding="utf-8")
        # Prepended, so the interpreter imports THIS sitecustomize at startup.
        path_entries.insert(0, str(root))
    env = {**os.environ,
           "PYTHONPATH": os.pathsep.join(path_entries),
           "HUGINN_PRIVACY_ROOT": str(root)}
    result = subprocess.run(
        [sys.executable, str(EXTRACTOR), "--source", str(source), "--output", str(output)],
        cwd=root, env=env, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return output.read_text(encoding="utf-8")


@pytest.fixture
def corpus(tmp_path):
    """Issues across nested directories, plus ``.excluded/`` stubs.

    Both walks have to be exercised: sorting only the active scan would leave
    the enrichment loop appending stub nodes in enumeration order.
    """
    source = tmp_path / IN_SCOPE_SOURCE
    for i in range(1, 7):
        # Spread across subdirectories so the walk has directories to order,
        # not just filenames within one.
        directory = source / f"batch-{i % 3}"
        _issue(directory, f"MELOSYS-{i}", f"Sak nummer {i}",
               epic_link="MELOSYS-100", epic_summary="Samleepos")

    excluded = source / ".excluded"
    for i in range(1, 4):
        _issue(excluded / f"stub-{i % 2}", f"MELOSYS-90{i}", f"Stubb {i}",
               parent=f"MELOSYS-{i}")

    (tmp_path / "huginn-fixture" / "privacy").mkdir(parents=True)
    (tmp_path / "huginn-fixture" / "privacy" / "aliases.json").write_text(
        json.dumps(_map()), encoding="utf-8")
    return tmp_path, source


def test_reversed_enumeration_produces_a_byte_identical_graph(corpus):
    """The guarantee: enumeration order is not an input to the output."""
    root, source = corpus
    forward = _run(root, source, root / "forward.json")
    backward = _run(root, source, root / "reversed.json", reverse_enumeration=True)
    assert forward == backward


def test_the_reversing_patch_actually_takes_effect(corpus):
    """Guard the guard.

    If ``sitecustomize`` silently failed to load, the test above would compare
    two identical forward runs and pass no matter what the extractor did — which
    is exactly what happened on the first draft of this file.
    """
    root, source = corpus
    probe = root / "probe.txt"
    (root / "sitecustomize.py").write_text(
        "import pathlib\n"
        "_real = pathlib.Path.rglob\n"
        "def _probe(self, pattern):\n"
        f"    open({str(probe)!r}, 'a').write('called\\n')\n"
        "    return iter(sorted(_real(self, pattern), reverse=True))\n"
        "pathlib.Path.rglob = _probe\n",
        encoding="utf-8")
    env = {**os.environ,
           "PYTHONPATH": os.pathsep.join([str(root), str(REPO_ROOT)]),
           "HUGINN_PRIVACY_ROOT": str(root)}
    result = subprocess.run(
        [sys.executable, str(EXTRACTOR), "--source", str(source),
         "--output", str(root / "probed.json")],
        cwd=root, env=env, capture_output=True, text=True, timeout=300)

    assert result.returncode == 0, result.stderr
    assert probe.exists(), "sitecustomize never ran — the order test would be vacuous"


def test_node_and_edge_order_is_stable_across_runs(corpus):
    """The weaker same-filesystem property, kept because it is what CI sees."""
    root, source = corpus
    assert _run(root, source, root / "one.json") == _run(root, source, root / "two.json")
