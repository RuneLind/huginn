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

# The public given-name gazetteer the distribution gate's bigram detector reads.
# It is a list of common given names compiled from public sources, and a
# Norwegian corpus's colleagues necessarily share given names with it — a list
# of common Norwegian names with holes punched where this machine's colleagues
# happen to be is both useless as a gazetteer and a *reverse* fingerprint. So
# single-token literals (bare given names, and the map's single-token mononyms)
# are not checked against this ONE file. Everything else still is: a full name,
# a `first.last` slug or any multi-token variant in the gazetteer is a leak, and
# `test_given_names_file_holds_only_single_tokens` makes it structurally
# impossible for one to be added.
GIVEN_NAMES_FILE = "main/privacy/given_names.txt"


def _mapped_person_literals(single_tokens: bool = True):
    """Every real-person literal the alias map knows, from the gitignored maps.

    Same shape as the sub-repo guard above: the names are never spelled out
    here, they are read at runtime from `huginn-*/privacy/aliases.json`, and a
    clone without private sub-repos skips. This is the backstop for the one
    mistake this campaign already made once — a real name in a docstring, which
    had to be removed by rewriting history.

    `single_tokens=False` drops the bare given names and mononyms, for the one
    file allowed to contain those (see GIVEN_NAMES_FILE).
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
                   if isinstance(literal, str) and len(literal) >= MIN_NAME_LENGTH
                   and (single_tokens or not literal.isalpha())})


def _pattern_for(literals):
    return re.compile("|".join(r"(?<!\w)" + re.escape(n) + r"(?!\w)" for n in literals), re.I)


def test_no_tracked_file_names_a_mapped_person():
    literals = _mapped_person_literals()
    if not literals:
        pytest.skip("no private alias map checked out here")
    try:
        files = _tracked_files()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("not a git checkout")
    pattern = _pattern_for(literals)
    multi_token = _pattern_for(_mapped_person_literals(single_tokens=False))
    hits = []
    for rel in files:
        if rel in PERSON_ALLOWLIST:
            continue
        if rel == GIVEN_NAMES_FILE:
            # Given names only; a full name or a slug is still a leak here.
            pattern_for_file = multi_token
        else:
            pattern_for_file = pattern
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if pattern_for_file.search(line):
                # The offending text is NOT echoed: this assertion message ends
                # up in CI logs and pasted into PRs.
                hits.append(f"{rel}:{lineno}")
    assert hits == [], "a mapped person's name appears in these tracked files:\n" + "\n".join(hits)


def test_given_names_file_holds_only_single_tokens():
    """The structural half of the carve-out above.

    The gazetteer is exempt from the bare-given-name check, so what keeps a full
    name out of it is this: every entry must be ONE alphabetic token. No space,
    no dot, no underscore, no comma, no digit — which makes `First Last`,
    `first.last` and `Last, First` all impossible to add, whatever the intent.
    """
    path = REPO_ROOT / GIVEN_NAMES_FILE
    assert path.exists(), f"{GIVEN_NAMES_FILE} is missing; the bigram detector needs it"
    bad = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if not entry.isalpha():
            bad.append(f"line {lineno}")
    assert bad == [], (f"{GIVEN_NAMES_FILE} must contain one alphabetic token per line; "
                       f"offending lines: {bad}")
