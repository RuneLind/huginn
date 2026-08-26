"""Node and edge order must not depend on filesystem enumeration order.

``extract_jira_graph.py`` walks its source tree twice — once for the active
issues, once for the ``.excluded/`` stub subtasks that enrich the graph — and
both walks feed ``issues``/``nodes`` in the order they enumerate. Raw
``Path.rglob`` order is a filesystem detail, so without ``sorted()`` the graph's
node order and all of its edges are whatever the volume happened to hand back —
cross-reference edges included, since huginn #123 sorted only the ``cross_refs``
set *inside* each issue and left the walk over the issues themselves unsorted.

The trap this file exists to avoid: on the APFS volume this was written on,
enumeration order is a pure function of the file-name set, so two consecutive
runs agree *whether or not the fix is present*. A rerun test would pass on the
unfixed extractor and prove nothing. So the check forces a pathological order
instead — reversed enumeration stands in for "a corpus rebuilt on another
filesystem", which is the case that would otherwise diff every node at once.

``sitecustomize`` is imported at interpreter startup for anything on
``PYTHONPATH``, which is how the subprocess gets its ``rglob`` patched without
the extractor knowing it is under test. Two things that mechanism has already
got wrong once each, both now structural rather than remembered:

- The gate test's ``_run`` cannot be reused: it pins ``PYTHONPATH`` to the repo
  root *after* merging ``extra_env``, deliberately, so a caller cannot redirect
  the privacy root. Injecting through it silently did nothing. So this file
  drives the subprocess itself and keeps the same ``HUGINN_PRIVACY_ROOT`` pin.
- A patch that fails to load turns the comparison into two identical forward
  runs, which passes on a broken extractor. So the patch drops a marker file and
  ``_run`` asserts it on every reversed invocation — the assertion lives on the
  path the real test takes, instead of in a second test that could drift away
  from the constant it is supposed to be guarding.
"""
import json
import os
import subprocess
import sys

import pytest

from tests.test_jira_graph_gate import (
    EXTRACTOR,
    IN_SCOPE_SOURCE,
    REPO_ROOT,
    _issue,
    _map,
)

MARKER = "rglob-patch-loaded"

REVERSING_SITECUSTOMIZE = """
import pathlib

_real = pathlib.Path.rglob


def _reversed_rglob(self, pattern):
    open({marker!r}, "a").write("called\\n")
    return iter(sorted(_real(self, pattern), reverse=True))


pathlib.Path.rglob = _reversed_rglob
"""


def _run(root, source, output, *, reverse_enumeration=False):
    """Drive the extractor, optionally under reversed enumeration order.

    ``HUGINN_PRIVACY_ROOT`` is pinned to ``root`` for the same reason the gate
    test pins it: without it a test discovers the operator's real map.
    """
    path_entries = [str(REPO_ROOT)]
    marker = root / MARKER
    if reverse_enumeration:
        (root / "sitecustomize.py").write_text(
            REVERSING_SITECUSTOMIZE.format(marker=str(marker)), encoding="utf-8")
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
    if reverse_enumeration:
        # Guard the guard, on the path the real test actually takes: without
        # this, a sitecustomize that silently failed to load would make the
        # comparison below two forward runs and pass on a broken extractor.
        assert marker.exists(), "the rglob patch never loaded — the comparison would be vacuous"
    return output.read_text(encoding="utf-8")


@pytest.fixture
def corpus(tmp_path):
    """Issues across nested directories, plus ``.excluded/`` stubs.

    Both walks have to be exercised: sorting only the active scan would leave
    the enrichment loop appending stub nodes in enumeration order. The bodies
    carry real cross-references so the walk that emits ``refererer_til`` blocks
    is covered too, not just the node scan.
    """
    source = tmp_path / IN_SCOPE_SOURCE
    keys = [f"MELOSYS-{i}" for i in range(1, 7)]
    for i, key in enumerate(keys, 1):
        # Spread across subdirectories so the walk has directories to order,
        # not just filenames within one.
        directory = source / f"batch-{i % 3}"
        _issue(directory, key, f"Sak nummer {i}",
               epic_link="MELOSYS-100", epic_summary="Samleepos")
        # Reference every other issue, so each issue emits a multi-edge
        # cross-reference block whose position depends on the outer walk.
        others = " ".join(k for k in keys if k != key)
        path = directory / f"{key}.md"
        path.write_text(path.read_text(encoding="utf-8") + f"\nSe også {others}.\n",
                        encoding="utf-8")

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
