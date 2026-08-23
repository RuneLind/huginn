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
import unicodedata
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
# happen to be is both useless as a gazetteer and a *reverse* fingerprint.
#
# The carve-out is therefore NOT "single-token literals are fine here". That
# exempted every mononym and every bare surname the map knows, so a colleague
# known by one name could be added to a PUBLIC file and no test would notice.
# What is exempt is the map's GIVEN-NAME set — the population this file is
# legitimately made of — plus a small reviewed set of given-name/surname
# overlaps derived from the map at runtime and capped below.
GIVEN_NAMES_FILE = "main/privacy/given_names.txt"

# Patronymics and given-name-derived family names are the same words in most
# European locales, so a handful of the map's surnames are inevitably also
# common given names. The cap is what makes it a reviewed set rather than an
# open door: a NEW overlap has to be looked at by a person.
MAX_GIVEN_NAME_SURNAME_OVERLAPS = 8

_TOKEN_SPLIT_RE = re.compile(r"[\s.,_]+")


def _alias_maps():
    for path in sorted(REPO_ROOT.glob("huginn-*/privacy/aliases.json")):
        yield json.loads(path.read_text(encoding="utf-8"))


def _tokens(literal: str):
    return [t.lower() for t in _TOKEN_SPLIT_RE.split(literal) if t]


def _split_literal(literal: str):
    """(given-name tokens, surname tokens) of one map literal.

    `Last, First` inverts the order, and `first.last` slugs are one token to
    `str.split` and two names to a reader — both have to be understood, or a
    surname is filed as a given name and silently exempted.
    """
    if "," in literal:
        head, _, tail = literal.partition(",")
        return _tokens(tail), _tokens(head)
    tokens = _tokens(literal)
    return tokens[:1], tokens[1:]


def _person_literals(data: dict):
    for entry in data.get("entries", []):
        name = entry.get("name")
        if isinstance(name, str):
            yield name
        yield from (v for v in entry.get("variants", []) if isinstance(v, str))
    for label, variants in data.get("unmapped_people_variants", {}).items():
        yield label
        yield from (v for v in variants if isinstance(v, str))


def _map_name_tokens():
    """(given-name set, surname-only set) over every private map on this machine.

    A mononym — a person the map knows by a single token — counts as a given
    name: it is the only form of their name there is, and the gazetteer is
    allowed to contain common given names.
    """
    given, surnames = set(), set()
    for data in _alias_maps():
        for literal in _person_literals(data):
            literal = literal.strip()
            if not literal:
                continue
            heads, tails = _split_literal(literal)
            if len(_tokens(literal)) == 1:
                given.update(heads)          # a mononym
            else:
                given.update(heads)
                surnames.update(tails)
        given |= {n.lower() for n in data.get("bare_given_name_residual", {})}
    return given, surnames - given


def _gazetteer_lines():
    path = REPO_ROOT / GIVEN_NAMES_FILE
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")]


def _reviewed_overlaps(gazetteer: set) -> set:
    """Surname-only map literals that are also common public given names."""
    _, surnames = _map_name_tokens()
    return surnames & gazetteer


def _mapped_person_literals():
    """Every real-person literal the alias map knows, from the gitignored maps.

    Same shape as the sub-repo guard above: the names are never spelled out
    here, they are read at runtime from `huginn-*/privacy/aliases.json`, and a
    clone without private sub-repos skips. This is the backstop for the one
    mistake this campaign already made once — a real name in a docstring, which
    had to be removed by rewriting history.
    """
    literals = set()
    for data in _alias_maps():
        literals.update(_person_literals(data))
        literals.update(data.get("bare_given_name_residual", {}))
    return sorted({literal for literal in literals
                   if isinstance(literal, str) and len(literal) >= MIN_NAME_LENGTH})


def _pattern_for(literals):
    if not literals:
        # `"|".join([])` is the empty pattern, which matches at every position:
        # the guard would then report every line of every tracked file.
        return None
    return re.compile("|".join(r"(?<!\w)" + re.escape(n) + r"(?!\w)" for n in literals), re.I)


def test_no_tracked_file_names_a_mapped_person():
    literals = _mapped_person_literals()
    if not literals:
        pytest.skip("no private alias map checked out here")
    try:
        files = _tracked_files()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("not a git checkout")
    gazetteer = {line.lower() for line in _gazetteer_lines()}
    given, _ = _map_name_tokens()
    exempt_here = given | _reviewed_overlaps(gazetteer)
    pattern = _pattern_for(literals)
    # The gazetteer file may contain the map's GIVEN names and the reviewed
    # overlaps — nothing else. A full name, a `first.last` slug, a mononym the
    # map knows or an unreviewed surname is still a leak in it.
    gazetteer_pattern = _pattern_for(
        [literal for literal in literals if literal.lower() not in exempt_here])
    hits = []
    for rel in files:
        if rel in PERSON_ALLOWLIST:
            continue
        pattern_for_file = gazetteer_pattern if rel == GIVEN_NAMES_FILE else pattern
        if pattern_for_file is None:
            continue
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


def test_the_gazetteer_surname_carve_out_stays_small_and_reviewed():
    """The carve-out is a reviewed list, not a rule that grows by itself."""
    if not list(_alias_maps()):
        pytest.skip("no private alias map checked out here")
    overlaps = _reviewed_overlaps({line.lower() for line in _gazetteer_lines()})
    assert len(overlaps) <= MAX_GIVEN_NAME_SURNAME_OVERLAPS, (
        f"{len(overlaps)} gazetteer lines are surnames in the private map, above the "
        f"reviewed cap of {MAX_GIVEN_NAME_SURNAME_OVERLAPS}. Review the new one and "
        f"either drop it from the gazetteer or raise the cap deliberately.")


def test_no_two_gazetteer_lines_concatenate_into_a_mapped_full_name():
    """The single-token rule keeps a full name out of any ONE line. It does not
    keep the two halves of one out of two lines — and `First` on line 40 plus
    `Last` on line 700 is the same disclosure, one grep away.

    Only where the second half is a SURNAME. A two-token map literal is not
    automatically `First Last`: a Norwegian double given name is two given
    names and no surname at all, so reassembling one from two gazetteer
    lines discloses nothing that a list of common given names does not already
    say. Treating every two-token literal as a full name forced ten ordinary
    given names out of the public file — of which only three are in the map's
    own runtime given-name set, so the other seven made `<given> <Surname>`
    pairs *undetectable* by check 9. A hole punched in a common-names list to
    hide a name that is not there is a cost with no return.

    "Surname" is decided by the MAP, never by the gazetteer: `_map_name_tokens`
    files a two-token literal's second token as a surname unless the map itself
    also knows it as a given name. So adding a line to `given_names.txt` can
    never exempt a pair from this test — the exemption has to already be true
    of the map.
    """
    if not list(_alias_maps()):
        pytest.skip("no private alias map checked out here")
    gazetteer = {line.lower() for line in _gazetteer_lines()}
    given, _ = _map_name_tokens()
    hits = set()
    for data in _alias_maps():
        for literal in _person_literals(data):
            tokens = _tokens(literal or "")
            if (len(tokens) == 2 and all(token in gazetteer for token in tokens)
                    and tokens[1] not in given):
                # Reported as a SHAPE: this message reaches CI logs.
                hits.add(re.sub(r"[^\W\d_]", "x", " ".join(tokens)))
    assert not hits, (f"{len(hits)} mapped full names can be reassembled from two gazetteer "
                      f"lines (shapes {sorted(hits)[:5]}); drop one half of each")


def test_given_names_file_holds_only_single_tokens():
    """The structural half of the carve-out above.

    Every entry must be ONE alphabetic token. No space, no dot, no underscore,
    no comma, no digit — which makes `First Last`, `first.last` and
    `Last, First` all impossible to add, whatever the intent.
    """
    path = REPO_ROOT / GIVEN_NAMES_FILE
    assert path.exists(), f"{GIVEN_NAMES_FILE} is missing; the bigram detector needs it"
    bad = [f"line {lineno}" for lineno, entry in enumerate(_gazetteer_lines(), 1)
           if not entry.isalpha()]
    assert bad == [], (f"{GIVEN_NAMES_FILE} must contain one alphabetic token per line; "
                       f"offending lines: {bad}")


def _sort_key(name: str) -> str:
    """Casefold, with the Norwegian letters written out. `Ådne` sorts as `Aadne`,
    `Bjørn` as `Bjoern` — the order a Norwegian reader scans the list in."""
    folded = name.casefold().replace("å", "aa").replace("æ", "ae").replace("ø", "oe")
    return unicodedata.normalize("NFKD", folded).encode("ascii", "ignore").decode()


def test_given_names_file_is_sorted_and_duplicate_free():
    """Both properties are how a person adds a name without adding it twice, and
    how a reviewer sees an insertion as a one-line diff instead of a move."""
    lines = _gazetteer_lines()
    lowered = [line.lower() for line in lines]
    duplicates = sorted({line for line in lowered if lowered.count(line) > 1})
    assert duplicates == [], f"duplicate gazetteer lines: {duplicates}"
    expected = sorted(lines, key=lambda name: (_sort_key(name), name))
    misplaced = [line for line, want in zip(lines, expected) if line != want]
    assert misplaced == [], (f"{len(misplaced)} gazetteer lines are out of order; "
                             f"re-sort with the key in _sort_key")


def test_the_gazetteer_comment_rule_matches_the_scanner():
    """The scanner and this file must agree on what a comment is, or a line one
    of them skips becomes a name to the other."""
    from main.privacy.index_scan import load_public_given_names
    assert load_public_given_names(REPO_ROOT / GIVEN_NAMES_FILE) == {
        line.lower() for line in _gazetteer_lines()}
