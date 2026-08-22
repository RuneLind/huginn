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

The second guard below does the same for real *people*: no tracked file may
contain a literal from the private alias map. Same construction, same reason.
"""
import json
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
    pattern = re.compile("(?:" + "|".join(re.escape(n) for n in names) + r")(?![A-Za-z0-9])", re.IGNORECASE)
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


# --- no real person from the private alias map in a tracked file --------------

MIN_NAME_LENGTH = 5  # shorter literals collide with ordinary words and code
# The MIT copyright line names the repo owner, who is also a mapped person
# because they author documents in the indexed corpora. Public by intent.
PERSON_ALLOWLIST = {"LICENSE"}


def _mapped_person_literals():
    """Every real-person literal the alias map knows, from the gitignored maps.

    Same shape as the sub-repo guard above: the names are never spelled out
    here, they are read at runtime from `huginn-*/privacy/aliases.json`, and a
    clone without private sub-repos skips. This is the backstop for the one
    mistake this campaign already made once — a real name in a docstring, which
    had to be removed by rewriting history.
    """
    literals = set()
    for path in sorted(REPO_ROOT.glob("huginn-*/privacy/aliases.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for entry in data.get("entries", []):
            literals.update(entry.get("variants", []))
        for variants in data.get("unmapped_people_variants", {}).values():
            literals.update(variants)
        literals.update(data.get("bare_given_name_residual", {}))
    return sorted({literal for literal in literals
                   if isinstance(literal, str) and len(literal) >= MIN_NAME_LENGTH})


def test_no_tracked_file_names_a_mapped_person():
    literals = _mapped_person_literals()
    if not literals:
        pytest.skip("no private alias map checked out here")
    try:
        files = _tracked_files()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("not a git checkout")
    pattern = re.compile("|".join(r"(?<!\w)" + re.escape(n) + r"(?!\w)" for n in literals), re.I)
    hits = []
    for rel in files:
        if rel in PERSON_ALLOWLIST:
            continue
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                # The offending text is NOT echoed: this assertion message ends
                # up in CI logs and pasted into PRs.
                hits.append(f"{rel}:{lineno}")
    assert hits == [], "a mapped person's name appears in these tracked files:\n" + "\n".join(hits)
