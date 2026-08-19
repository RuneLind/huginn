"""Public-repo hygiene: no tracked file names a private ``huginn-*`` sub-repo.

The sub-repos are gitignored and their names carry employer / customer /
personal context (see CLAUDE.md, "Private sub-repos"). Code discovers them by
glob (``graph_loader._discover_auto_glob_dirs``, ``routes.graph
.find_author_scores_path``, ``indexing_schedule.ROUTING_GLOBS``), docs use a
``<private-sub-repo>`` placeholder. This test is the deterministic backstop for
the manual sweeps that landed #111 and #113.

The names are NOT spelled out here — that would publish them in the very file
meant to keep them out. They are read from the ``huginn-*/`` directories
present on disk (gitignored, so only the operator's checkout has them), which
also covers any sub-repo added later. A clone without sub-repos has nothing to
check and skips. The scan runs against the git index so an untracked local
file never trips it.
"""
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
# Self-exempt only; sibling guards must assert structurally (no literal names).
ALLOWLIST = {"tests/test_public_hygiene.py"}


def _private_subrepo_names():
    return sorted(p.name for p in REPO_ROOT.glob("huginn-*") if (p / ".git").exists())


def _tracked_files():
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, check=True,
    ).stdout
    return [p for p in out.decode().split("\0") if p]


def test_no_tracked_file_names_a_private_subrepo():
    names = _private_subrepo_names()
    if not names:
        pytest.skip("no private huginn-*/ sub-repos checked out here")
    try:
        files = _tracked_files()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("not a git checkout")
    pattern = re.compile("(?:" + "|".join(re.escape(n) for n in names) + r")\b")
    hits = []
    for rel in files:
        if rel in ALLOWLIST:
            continue
        if pattern.search(rel):
            hits.append(f"{rel}: (filename)")
        path = REPO_ROOT / rel
        if not path.is_file():
            continue  # deleted-but-staged, or a gitlink
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary
        for lineno, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{rel}:{lineno}: {line.strip()[:100]}")
    assert hits == [], "private sub-repo names in tracked files:\n" + "\n".join(hits)
