"""Build-time people aliasing for indexed collections.

The point of this module is that a *built* collection under data/collections/
carries no real person name, so it can be copied to a teammate's machine. The
substitution therefore happens inside the document converter — before chunking,
before contextual prefixing, before anything is written to disk — not at query
time. The reverse map (``huginn-*/privacy/aliases.json``) never leaves this
machine; it lives in a gitignored private sub-repo.

Six substitution classes, resolved in one left-to-right pass plus a handle pass
and a home-path pass:

  (a) entry variants          -> the entry's alias        ("dev-06")
  (b) unmapped_people_variants-> person_redaction_token   ("[~ukjent-person]")
  (c) non_person_labels       -> themselves (EXEMPT)      role nouns, test users,
                                                          countries. Redacting one
                                                          corrupts the corpus.
  (d) NAV idents              -> "[~person]"              "[~Q000124]" and bare
                                                          "Q000124"; the wrapper is
                                                          consumed with the token.
  (e) dotted handles and      -> "@person" / "person@"    only what (a)/(b) did not
      dotted email locals                                 already claim.
  (f) `/Users/<name>/`        -> "/Users/<user>/"         the account name in a
                                                          pasted stack trace or
                                                          shell transcript.

(a)-(d) share ONE compiled alternation sorted longest-first, which is what makes
(c) win over any shorter overlapping person variant: at a given position the
regex engine takes the first alternative that matches, and the longer exempt
label is listed before the shorter person variant. Doing it as separate passes
would let a person variant eat half a role phrase.

(e) runs afterwards, over the already-substituted text, which is exactly what
"not matched by a mapped variant" means: a mapped handle has become "@dev-06" by
then and no longer matches the handle shape (no dot, and digits/hyphens are
outside its character class). (f) runs last, for the same reason.

NEVER substituted: the document `id` and `url`. They are the join keys between
the index mapping, the derived JSON and the source file; rewriting them would
break document lookup and deletion. Measured on the three in-scope collections:
no source path contains a mapped person variant, so keeping them intact does not
leak (see the PR body).
"""

import glob
import json
import logging
import os
import re
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Bump when the *substitution behaviour* changes in a way that makes previously
# built artifacts or cached derivations (contextual prefixes, LLM graph
# extractions) stale. Stamped into the manifest and mixed into the graph
# extractor's cache key.
POLICY_VERSION = 1

# Marker the converter puts on a document whose text the registry actually
# changed. The pipeline pops it (so it is never persisted) and uses it to
# invalidate exactly that document's contextual-prefix cache entries.
ALIAS_CHANGED_KEY = "_aliasChanged"

IDENT_TOKEN = "[~person]"
HANDLE_TOKEN = "@person"
EMAIL_LOCAL_TOKEN = "person"

# What may separate the tokens of a multi-token variant. A hard-wrapped comment
# ("First\nLast"), a double space, a non-breaking space and the percent-encoded
# form inside URLs are all the same name; a single literal space was not.
#
# At most ONE newline, and only with horizontal space around it. `\s+` also
# spanned a blank line, which is a paragraph break: it welded the last token of
# one sentence to the first of the next and aliased the pair as a name.
VARIANT_SEPARATOR = r"(?:[^\S\n]+|[^\S\n]*\n[^\S\n]*|%20)"
_SEPARATOR_RUN = re.compile(r"(?:\s|%20)+", re.IGNORECASE)

# Boundaries for a multi-token variant. `\w` alone is wrong inside a URL query
# string: in `?f=%2CAda%20Example%40nav.no` the character before the name is the
# `C` of `%2C` and the one after is the `%` of `%40`, so `(?<!\w)`/`(?!\w)`
# refused a full name sitting in the clear. Exported because
# main/privacy/index_scan.py must build its needles with the same boundary, or
# the sweep certifies a form the substituter never removed.
#
# Any percent-escape, not the three that happened to show up first: `%2C`/`%20`/
# `%40` were enumerated from one corpus sample, and a name fenced by `%3B`, `%2F`
# or `%3A` (query separators every bit as common) fell straight through. The
# generalised form is also symmetric — an enumerated left and a differently
# enumerated right boundary is a bug waiting for the next escape.
VARIANT_LEFT_BOUNDARY = r"(?:(?<!\w)|(?<=%[0-9A-Fa-f]{2}))"
VARIANT_RIGHT_BOUNDARY = r"(?:(?!\w)|(?=%[0-9A-Fa-f]{2}))"

# The repo root, derived from this file rather than from the process CWD.
# EVERY private-file glob and every relative scope path resolves against it.
# CWD-relative discovery meant the whole privacy layer silently found nothing
# from any other working directory — and `tag_documents.py`'s external-backend
# guard, whose entire job is to refuse, therefore no-opped and called the hosted
# backend when run from /tmp. Module-level so a test can point it at a tmp dir.
#
# ``HUGINN_PRIVACY_ROOT`` relocates it, for tests that drive an entry point as a
# subprocess. It fails CLOSED: the public scope file is read from ``__file__``
# and is unaffected, so pointing the root at an empty directory still puts the
# public collections in scope and then finds no map — PrivacyMapMissing, not a
# silent unaliased build.
REPO_ROOT = os.environ.get("HUGINN_PRIVACY_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SCOPE_FILE = os.path.join(os.path.dirname(__file__), "scope.json")
_PRIVATE_SCOPE_GLOB = "huginn-*/privacy/scope.json"
_PRIVATE_MAP_GLOB = "huginn-*/privacy/aliases.json"
_PRIVATE_IDENT_EXCEPTIONS_GLOB = "huginn-*/privacy/ident_exceptions.json"
_COLLECTIONS_DIR = os.path.join("data", "collections")


def _private_glob(pattern: str) -> list:
    """Sorted matches of a private-sub-repo glob, resolved against REPO_ROOT."""
    return sorted(glob.glob(os.path.join(REPO_ROOT, pattern)))


def _repo_path(path: str) -> str:
    """Absolutise a possibly-relative configured path against REPO_ROOT."""
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)

# A NAV ident is one letter and six digits. Either bare or inside the Jira
# mention wrapper `[~Q000124]`, which is consumed together with the token so the
# result reads `[~person]` rather than `[~[~person]]`.
_IDENT_ALTERNATIVE = r"\[~\s*[A-Za-z]\d{6}\s*\]|(?<![\w])[A-Za-z]\d{6}(?![\w])"
_IDENT_INNER = re.compile(r"[A-Za-z]\d{6}")

# A bare NAV ident, as a standalone scanner. The substituter uses
# _IDENT_ALTERNATIVE above (which also consumes the `[~…]` wrapper); this is the
# form main/privacy/index_scan.py and main/privacy/sensitivity_scanner.py need,
# and it lives here so there is exactly one copy of the shape.
IDENT_RE = re.compile(r"(?i)(?<!\w)[a-z]\d{6}(?!\w)")

# `(?<![\w.])` is what keeps an email's own domain out: in "ola@firma.internal"
# the `@` is preceded by a word character, so the domain never enters this pass
# at all and no TLD allow-list has to know "internal". The local part is handled
# separately by _EMAIL_LOCAL_RE, which fires only on a *dotted* local part.
# Public for the same reason as IDENT_RE: the gate must count exactly what the
# substituter would have redacted.
HANDLE_RE = re.compile(r"(?<![\w.])@[a-zæøå0-9]+(?:\.[a-zæøå0-9]+)+", re.IGNORECASE)
_HANDLE_RE = HANDLE_RE

# A dotted local part only counts as an email local part when what follows the
# `@` is a real domain: at least one dot and an alphabetic last label. Without
# that, `@` between two dotted expressions — `np.eye@vec`, `A.T@B`, `self.w@x`,
# i.e. Python's matmul operator, which the code-heavy corpora are full of — was
# read as an address and the left operand was rewritten to `person`.
_EMAIL_LOCAL_RE = re.compile(
    r"(?<![\w.])[a-zæøå0-9]+(?:\.[a-zæøå0-9]+)+(?=@[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}\b)",
    re.IGNORECASE)

# Last labels that make a dotted handle a host rather than a person. The
# non-public suffixes (`internal`, `local`, `test`, `dev`, `localhost`) are here
# because internal service hostnames are what the corpora actually contain.
_HANDLE_TLDS = frozenset({"no", "com", "org", "net", "io", "eu", "co", "uk",
                          "se", "dk", "fi", "de", "fr", "nl", "info", "dev",
                          "ai", "internal", "local", "test", "localhost"})
_HANDLE_PACKAGE_ROOTS = frozenset({"org", "com", "no", "io", "java", "jakarta",
                                   "kotlin", "android", "net"})

# Class precedence inside the alternation for equal-length literals. Exempt
# beats person beats redaction: exempting a role noun is safe, redacting one
# corrupts the corpus.
_EXEMPT, _PERSON, _REDACT = 0, 1, 2

# Distinct from None, which is the stored value meaning "matched, leave as is".
_UNRESOLVED = object()

# A bare given name: one token, letters only. `Ada` substitutes half the corpus,
# so it is refused. Anything with a separator, a digit, `_` or a hyphen is a full
# name or a slug — `Nord-Hansen` is a compound surname, and rejecting it made the
# whole map unloadable over a legitimate entry.
_BARE_GIVEN_NAME_RE = re.compile(r"[^\W\d_]+")

# A macOS home directory. The account name in it is a person — and the one
# person the alias map is least likely to list, because they are the operator
# rather than a document author. It arrives in the corpora inside pasted stack
# traces, shell transcripts and tool output, which is why the distribution
# gate's check 10 found the shape and nothing removed it.
#
# Requires the trailing `/`: `/Users/` at the end of a line has no segment to
# bound the account name against, and `s3://bucket/Users/x` is not a home
# directory at all — hence the `(?<![\w])` in front. Idempotent, because
# `<user>` is itself a `[^/\s]+` segment and re-substituting it changes nothing.
# Check 10 keeps failing on the forms this does NOT rewrite; that is the point.
USER_PATH_RE = re.compile(r"(?<![\w])/Users/[^/\s]+/")
USER_PATH_TOKEN = "/Users/<user>/"


def _lookup_key(literal: str) -> str:
    """Table key: separator runs collapsed, casefolded.

    casefold() rather than lower() because IGNORECASE folds `ſ` onto `s` while
    lower() does not, and a missed lookup used to redact an exempt role noun.
    """
    return _SEPARATOR_RUN.sub(" ", literal).casefold()


def shape(literal: str) -> str:
    """Letters -> x, digits -> 9. Map literals are real names; anything this
    module raises or logs has to stay safe to paste into a public issue.

    The ONE masking helper for the whole privacy package — index_scan and
    sensitivity_scanner import it rather than keeping their own. The gate's copy
    used to mask letters only, so a shape printed from it still carried the six
    birth-date digits of a fødselsnummer.
    """
    return re.sub(r"\d", "9", re.sub(r"[^\W\d_]", "x", literal))


_shape = shape


def _case_key(literal: str) -> str:
    """Weaker fold, used only to tell "the same literal in another case" from
    "a different literal". `.lower()` leaves ß alone where casefold() maps it to
    `ss`, which is exactly the pair the collision check has to separate."""
    return _SEPARATOR_RUN.sub(" ", literal).lower()


class PrivacyMapMissing(RuntimeError):
    """An in-scope collection was built without a loadable alias map.

    Raised *before* the collection folder is removed or anything is written, so
    a missing private sub-repo fails the build instead of silently producing an
    index full of real names.
    """


class PrivacyMapInvalid(PrivacyMapMissing):
    """A map that loaded but would substitute wrongly. Same fail-closed effect."""


def boundaried(literal: str) -> str:
    r"""Escape `literal` and wrap it in the boundaries its shape needs.

    A *slug* (one token with `.`/`_`) must not match inside a longer dotted path,
    but must match after a word character — a dotted name appears
    percent-encoded in URLs (`…%2CAda.Example%40nav.no`). A *bare token* blocks a
    preceding `.` and word character, so a mononym cannot be welded onto a
    surviving path segment; a preceding `-` does NOT block, because a hyphen in
    front of a name is punctuation (a list bullet, a diff marker) far more often
    than it is a path. A *multi-token* variant uses the percent-aware boundaries,
    since a name in a URL query string is fenced by `%2C`/`%20`/`%40` rather than
    by non-word characters. A trailing full stop never blocks: it ends a
    sentence. A boundary applies only on a side whose edge character is a word
    character (`Ada Example [X]` ends in `]`).

    Public because main/privacy/index_scan.py builds its needles with it. Before
    that, the sweep wrapped EVERY needle in the multi-token boundary, which is
    the loosest of the three: a mononym needle sitting after a dot
    (`some.path.Zylphia`) was reported as a leak the substituter deliberately
    does not remove, and a slug needle matched inside a longer dotted path.
    """
    tokens = literal.split()
    body = VARIANT_SEPARATOR.join(re.escape(t) for t in tokens) if len(tokens) > 1 \
        else re.escape(literal)
    slug = ("." in literal or "_" in literal) and len(tokens) == 1
    if slug:
        left, right = r"(?<![.\-])", r"(?![\w\-])(?!\.\w)"
    elif len(tokens) == 1:
        left, right = r"(?<![.\w])", r"(?!\w)"
    else:
        left, right = VARIANT_LEFT_BOUNDARY, VARIANT_RIGHT_BOUNDARY
    prefix = left if literal[:1].isalnum() or literal[:1] == "_" else ""
    suffix = right if literal[-1:].isalnum() or literal[-1:] == "_" else ""
    return f"{prefix}{body}{suffix}"


def is_person_handle(handle: str) -> bool:
    """True for an `@first.last` shape; False for versions, code and domains.

    Three exclusions, in order, and then everything left is a person:

    * a purely numeric segment — `@v1.2.3` is a version, not a name;
    * a **code annotation or package path**: a known package root
      (`@org.junit.Test`), or a lowercase root followed by a CamelCase segment
      (`@lombok.Setter`, `@mockito.Captor`, `@dagger.Provides`). The root list
      alone cannot keep up — the annotation vocabulary is open-ended — and the
      CamelCase tail is what an annotation has and a name does not;
    * a **domain**, decided by an explicit last-label list only.

    There used to be a fourth test: "last segment of at most 4 alphabetic
    characters" also counted as a TLD shape. It is gone. Norwegian surnames are
    short — `@ola.berg`, `@kari.moe`, `@per.aas` all read as domains under it and
    survived into the built collection. An explicit list is the only version of
    this rule that fails towards redaction.

    Shared with main/privacy/index_scan.py so the acceptance
    sweep counts exactly what the substituter would have redacted.
    """
    segments = handle.lstrip("@").split(".")
    if any(segment.isdigit() for segment in segments):
        return False                                  # @v1.2.3 — a version
    lowered = [segment.lower() for segment in segments]
    if lowered[0] in _HANDLE_PACKAGE_ROOTS:
        return False                                  # @org.junit.Test, @jakarta.persistence.x
    if (len(segments) >= 2 and segments[0].islower()
            and any(s[:1].isupper() for s in segments[1:])):
        return False                                  # @lombok.Setter — an annotation
    if len(segments) >= 2 and lowered[-1] in _HANDLE_TLDS:
        return False                                  # a host: the last label is a known TLD
    return True


def _substitute_email_local(match: re.Match) -> str:
    local = match.group(0)
    segments = local.split(".")
    if segments[0].lower() in _HANDLE_PACKAGE_ROOTS or any(s.isdigit() for s in segments):
        return local
    return EMAIL_LOCAL_TOKEN


def _substitute_handle(match: re.Match) -> str:
    handle = match.group(0)
    return HANDLE_TOKEN if is_person_handle(handle) else handle


def _validate(map_data: dict) -> None:
    """Refuse a map that would substitute wrongly, before anything is compiled.

    The private map has its own lint; this is the runtime backstop for the ways
    a hand-edited or truncated map fails *silently* rather than loudly.
    """
    entries = map_data.get("entries")
    if not isinstance(entries, list) or not entries:
        raise PrivacyMapInvalid("alias map has no entries; refusing to build a stamped index")
    unmapped = map_data.get("unmapped_people_variants", {})
    if not isinstance(unmapped, dict):
        raise PrivacyMapInvalid("'unmapped_people_variants' must be an object keyed by name")
    labels = map_data.get("non_person_labels", [])
    if not isinstance(labels, list):
        raise PrivacyMapInvalid("'non_person_labels' must be a list")

    for entry in entries:
        if (not isinstance(entry, dict) or not entry.get("alias")
                or not isinstance(entry.get("variants"), list)):
            raise PrivacyMapInvalid("every entry needs an 'alias' and a 'variants' list")
        # `name` is not optional even though the substituter never reads it: the
        # distribution gate builds its token-permutation needles from it, and a
        # map without it raised KeyError halfway through a scan instead of being
        # refused up front. It is also the only field guaranteed to spell the
        # person `First Last` — `variants` may only carry `Last, First`.
        if not isinstance(entry.get("name"), str) or not entry["name"].strip():
            raise PrivacyMapInvalid("every entry needs a non-empty 'name'; the gate builds "
                                    "its permutation needles from it")
        for variant in entry["variants"]:
            _require_literal(variant)
            # `.strip()` first: `" Ada "` is the same bare given name as `"Ada"`,
            # but fullmatch() against the padded form fails, so a variant that
            # picked up whitespace in hand-editing walked past this guard and
            # then compiled into a pattern that substitutes a given name.
            if _BARE_GIVEN_NAME_RE.fullmatch(variant.strip()):
                raise PrivacyMapInvalid("an entry variant is a bare given name (a single "
                                        "all-alphabetic token); those are never substituted")
    for variants in unmapped.values():
        if not isinstance(variants, list):
            raise PrivacyMapInvalid("each 'unmapped_people_variants' value must be a list")
        for variant in variants:
            _require_literal(variant)
    for label in labels:
        _require_literal(label)


def _require_literal(value) -> None:
    # A blank alternative matches at every position and swallows the corpus.
    if not isinstance(value, str) or not value.strip():
        raise PrivacyMapInvalid("blank or non-string literal in the alias map")


class AliasRegistry:
    """Compiled substituter for one alias map."""

    def __init__(self, map_data: dict, ident_exceptions=()):
        _validate(map_data)
        self.map_version = map_data.get("version")
        self.redaction_token = map_data.get("person_redaction_token") or "[~ukjent-person]"
        self._ident_exceptions = {t.lower() for t in ident_exceptions}
        self._warned_unresolved = False

        # alias -> the entry's canonical `name`, i.e. the inverse of (a). Used by
        # exactly one caller (`query_privacy.name_space_twin`) for the one text
        # that has to travel BACK into name space: the corrective-rescue query
        # handed to an out-of-scope collection, whose index has never contained
        # an alias token. Nothing built from it is logged, traced or echoed.
        self._alias_names = {}
        for entry in map_data.get("entries", []):
            alias, name = entry.get("alias"), entry.get("name")
            if not (alias and name):
                continue
            if alias in self._alias_names and self._alias_names[alias] != name:
                # An inverted duplicate would send an out-of-scope collection the
                # WRONG person's name; the key must be injective to be a key.
                raise PrivacyMapInvalid(f"alias {alias!r} is used by two entries")
            self._alias_names[alias] = name
        # `(?<![\w-])`/`(?![\w-])` rather than `\b`: an alias is `dev-06`, so a
        # word boundary would happily fire inside `dev-061` or `pre-dev-06`.
        alternation = "|".join(sorted((re.escape(alias) for alias in self._alias_names),
                                      key=len, reverse=True))
        self._name_pattern = re.compile(
            rf"(?<![\w-])(?:{alternation})(?![\w-])") if alternation else None

        # The first token of every entry `name` and every unmapped label. A bare
        # given name is deliberately NEVER substituted (`Ada` would alias half
        # the corpus), so one standing beside its own alias in an expansion term
        # re-pairs the two; query_privacy drops those.
        self.given_names = frozenset(
            literal.split()[0].strip(",").lower()
            for literal in [entry.get("name") for entry in map_data.get("entries", [])]
            + list(map_data.get("unmapped_people_variants", {}))
            if isinstance(literal, str) and literal.strip()
        )

        # normalized literal -> replacement, or None meaning "leave as is"
        self._replacements: dict[str, str | None] = {}
        ranked: list[tuple[int, int, str, str]] = []

        for label in map_data.get("non_person_labels", []):
            ranked.append((-len(label), _EXEMPT, label, label))
        for entry in map_data.get("entries", []):
            for variant in entry["variants"]:
                ranked.append((-len(variant), _PERSON, variant, entry["alias"]))
        for variants in map_data.get("unmapped_people_variants", {}).values():
            for variant in variants:
                ranked.append((-len(variant), _REDACT, variant, self.redaction_token))

        ranked.sort(key=lambda r: (r[0], r[1], r[2]))

        alternatives: list[str] = []
        seen_rank: dict[str, int] = {}
        seen_literal: dict[str, str] = {}
        seen_replacement: dict[str, str | None] = {}
        for _, rank, literal, replacement in ranked:
            key = _lookup_key(literal)
            stored = None if rank == _EXEMPT else replacement
            if key in seen_rank:
                if (seen_literal[key] != _case_key(literal)
                        and seen_replacement[key] != stored):
                    # Two DIFFERENT literals collapsing onto one table key
                    # ("Weiss" and "Weiß" both casefold to "weiss"). The table is
                    # keyed on the casefold, so the second literal's replacement
                    # is silently dropped and both spellings get the first one's.
                    # Nothing downstream can see that; refuse to compile.
                    # The key is a real name; report its SHAPE, since this
                    # message ends up in build logs and tracebacks.
                    #
                    # Only when the replacements actually differ. Two spellings
                    # of one person's name (`Weiss`/`Weiß` under the same alias)
                    # collapse onto the same key and get the same alias either
                    # way — there is nothing to silently drop, and refusing made
                    # the map unloadable over an entry that was simply thorough.
                    raise PrivacyMapInvalid(
                        f"two distinct literals share a casefold key (shape "
                        f"{_shape(key)!r}); one would silently take the other's replacement")
                if seen_literal[key] != _case_key(literal):
                    # A different spelling that folds onto the same key and wants
                    # the same replacement. The TABLE already has the answer (the
                    # lookup casefolds too), but the PATTERN does not: IGNORECASE
                    # folds case, not ß onto ss, so without its own alternative
                    # this spelling is never matched at all.
                    alternatives.append(boundaried(literal))
                    continue
                # Same literal twice. Within a class this is just case variation
                # in the map ("Saksbehandler"/"saksbehandler"); across classes it
                # means the map's own lints let a real conflict through, so say so.
                if seen_rank[key] != rank:
                    logger.warning("Alias map: a literal (shape %r) appears in two classes; "
                                   "keeping the higher-precedence one", _shape(literal))
                continue
            seen_rank[key] = rank
            seen_literal[key] = _case_key(literal)
            seen_replacement[key] = stored
            self._replacements[key] = stored
            alternatives.append(boundaried(literal))

        alternatives.append(_IDENT_ALTERNATIVE)
        self._pattern = re.compile("|".join(alternatives), re.IGNORECASE)

    # --- loading -----------------------------------------------------------

    @classmethod
    def load(cls, path: str, ident_exceptions_path: str | None = None) -> "AliasRegistry":
        with open(path, "r", encoding="utf-8") as f:
            map_data = json.load(f)
        return cls(map_data, ident_exceptions=_load_ident_exceptions(ident_exceptions_path))

    # --- substitution ------------------------------------------------------

    def _substitute(self, match: re.Match) -> str:
        matched = match.group(0)
        replacement = self._replacements.get(_lookup_key(matched), _UNRESOLVED)
        if replacement is not _UNRESOLVED:
            return matched if replacement is None else replacement
        inner = _IDENT_INNER.search(matched)
        if inner is None:
            # Needs a character whose casefold disagrees with the regex's
            # IGNORECASE fold (dotless ı and friends). Pass it through: the map
            # says nothing about this text, and redacting it would invent a
            # person claim about an ordinary word.
            if not self._warned_unresolved:
                self._warned_unresolved = True
                logger.warning("Alias map: a match did not resolve to a literal or an ident; "
                               "left unchanged")
            return matched
        if inner.group(0).lower() in self._ident_exceptions:
            return matched
        return IDENT_TOKEN

    def apply(self, text: str) -> str:
        if not text:
            return text
        substituted = self._pattern.sub(self._substitute, text)
        substituted = _EMAIL_LOCAL_RE.sub(_substitute_email_local, substituted)
        substituted = _HANDLE_RE.sub(_substitute_handle, substituted)
        # Last, so a mapped person's own home directory has already become
        # `/Users/dev-06/` and this only has to deal with the unmapped rest.
        return USER_PATH_RE.sub(USER_PATH_TOKEN, substituted)

    def to_names(self, text: str) -> str:
        """The inverse of the alias substitution: `dev-06` -> the entry's name.

        Lossy on purpose — one canonical name per alias, and the redaction
        tokens have no inverse at all. It exists for the single text that has to
        be turned back into name space (see ``query_privacy.name_space_twin``),
        never for anything that is stored or returned.
        """
        if not text or self._name_pattern is None:
            return text
        return self._name_pattern.sub(lambda m: self._alias_names[m.group(0)], text)

    def _apply_value(self, value):
        """Alias any JSON value; returns (value, changed). Dict KEYS are left alone.

        Metadata is arbitrary JSON — the Jira reader puts lists of comment dicts
        there — so the walk has to be recursive rather than str / list-of-str.
        """
        if isinstance(value, str):
            new_value = self.apply(value)
            return new_value, new_value != value
        if isinstance(value, list):
            changed = False
            out = []
            for item in value:
                new_item, item_changed = self._apply_value(item)
                out.append(new_item)
                changed |= item_changed
            return (out if changed else value), changed
        if isinstance(value, dict):
            changed = False
            for key, item in value.items():
                new_item, item_changed = self._apply_value(item)
                if item_changed:
                    value[key] = new_item
                    changed = True
            return value, changed
        return value, False

    def _apply_metadata(self, metadata) -> bool:
        if not isinstance(metadata, dict):
            return False
        return self._apply_value(metadata)[1]

    def apply_document(self, converted_document: dict) -> bool:
        """Alias a converted document in place. Returns True if anything changed.

        Covers `text`, every chunk's `indexedData` and `heading`, and the string /
        list-of-string values of both document and chunk `metadata`. `id`, `url`
        and `modifiedTime` are deliberately left alone (see module docstring).
        """
        changed = False

        text = converted_document.get("text")
        if isinstance(text, str):
            new_text = self.apply(text)
            if new_text != text:
                converted_document["text"] = new_text
                changed = True

        for chunk in converted_document.get("chunks", []):
            for field in ("indexedData", "heading"):
                value = chunk.get(field)
                if isinstance(value, str):
                    new_value = self.apply(value)
                    if new_value != value:
                        chunk[field] = new_value
                        changed = True
            changed |= self._apply_metadata(chunk.get("metadata"))

        changed |= self._apply_metadata(converted_document.get("metadata"))
        return changed

    def manifest_stamp(self, aliased_at=None) -> dict:
        return {
            "policy_version": POLICY_VERSION,
            "map_version": self.map_version,
            "aliasedAt": (aliased_at or datetime.now(timezone.utc)).isoformat(),
        }


# --- scoping ---------------------------------------------------------------

def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Privacy: could not read %s: %s", path, e)
        return None


def _load_ident_exceptions(path: str | None = None) -> set:
    """Literal ident-shaped tokens that are NOT people (git SHAs, document type
    ids, test users). Missing file => no exceptions, i.e. redact everything.

    Public-by-convention despite the underscore: ``scripts/audit/scan_index.py``
    calls it rather than keeping a second copy of the glob and the schema.
    """
    candidates = [path] if path else _private_glob(_PRIVATE_IDENT_EXCEPTIONS_GLOB)
    tokens = set()
    for candidate in candidates:
        if not candidate or not os.path.exists(candidate):
            continue
        data = _load_json(candidate)
        if not isinstance(data, dict):
            continue
        tokens.update(t for t in data.get("tokens", []) if isinstance(t, str) and t)
    return tokens


def load_scope() -> tuple[set, set]:
    """Public scope plus any private extension, as (collection names, realpaths).

    The public file names only collections and source dirs that are already
    public in CLAUDE.md. Private sub-repos extend it through
    `huginn-*/privacy/scope.json`, mirroring how graph_routing.json is
    discovered.

    Relative basePaths resolve against **REPO_ROOT**, not the process CWD. They
    are written relative to the repo (`./data/sources/…`) and the callers are
    not all servers started from it: resolving them against the CWD made the
    whole scope evaporate for anything run from elsewhere, which turns every
    fail-closed guard built on this function into a no-op exactly where it
    matters.
    """
    collections, base_paths = set(), set()
    for path in [_SCOPE_FILE, *_private_glob(_PRIVATE_SCOPE_GLOB)]:
        if not os.path.exists(path):
            continue
        data = _load_json(path)
        if not isinstance(data, dict):
            continue
        collections.update(c for c in data.get("collections", []) if isinstance(c, str) and c)
        for base_path in data.get("basePaths", []):
            if isinstance(base_path, str) and base_path:
                base_paths.add(os.path.realpath(_repo_path(base_path)))
    return collections, base_paths


def scoped_base_paths() -> set:
    """Every source tree in scope: the declared basePaths PLUS the reader
    basePath of every in-scope collection's built manifest.

    The two lists say the same thing in different places and they drift: a
    collection is put in scope by NAME, its `basePaths` entry is forgotten, and
    the source tree is then out of scope for every path-based guard while the
    built index is aliased. The manifests under ``data/collections/`` are the
    record of what each in-scope collection actually reads, so they are read
    here rather than trusted to be mirrored by hand.
    """
    collections, base_paths = load_scope()
    root = os.path.join(REPO_ROOT, _COLLECTIONS_DIR)
    for name in collections:
        path = os.path.join(root, name, "manifest.json")
        # Quietly: a collection in scope that is simply not built on this machine
        # is the normal case, not a warning worth one log line per call.
        if not os.path.exists(path):
            continue
        manifest = _load_json(path)
        if not isinstance(manifest, dict):
            continue
        base_path = (manifest.get("reader") or {}).get("basePath")
        if isinstance(base_path, str) and base_path:
            base_paths.add(os.path.realpath(_repo_path(base_path)))
    return base_paths


def path_in_scope(path: str) -> bool:
    """True when `path` is an in-scope source tree, or lives inside one.

    Containment, not equality: a collection's reader points at the whole tree,
    but a caller — `scripts/tagging/tag_documents.py --source <subdir>` — may
    name any directory under it, and the documents there are the same documents.
    Realpath on both sides, so a symlink pointing into an in-scope tree is in
    scope and `../` cannot walk out of one.
    """
    if not path:
        return False
    resolved = os.path.realpath(path)
    return any(resolved == base or resolved.startswith(base + os.sep)
               for base in scoped_base_paths())


def discover_map_path(explicit: str | None = None) -> str:
    """The one alias map on this machine, or a PrivacyMapMissing saying why not.

    Shared with ``scripts/audit/scan_index.py`` so the gate verifies with the
    same map the build substituted with, discovered by the same rule. Two
    matches are ambiguous rather than first-wins: which map applied would
    otherwise depend on directory order.
    """
    if explicit:
        return explicit
    candidates = _private_glob(_PRIVATE_MAP_GLOB)
    if len(candidates) != 1:
        raise PrivacyMapMissing(
            f"expected exactly one alias map ({_PRIVATE_MAP_GLOB} under {REPO_ROOT}), "
            f"found {len(candidates)}")
    return candidates[0]


def resolve_registry(collection_name, base_path, armed_by_manifest: bool = False,
                     map_path: str | None = None) -> AliasRegistry | None:
    """The single scoping decision both construction sites make.

    Returns a registry for an in-scope collection, or None for an out-of-scope
    one — in which case the converter behaves byte-identically to before this
    module existed. An in-scope collection whose map is missing or unloadable
    raises PrivacyMapMissing rather than quietly building an index full of real
    names; the caller must let that propagate before the collection folder is
    removed.

    `armed_by_manifest` keeps an already-aliased collection aliased on update
    even if the scope files no longer say so — un-aliasing half a collection on
    the next nightly run would be worse than over-applying.
    """
    in_scope = armed_by_manifest
    if not in_scope:
        collections, _ = load_scope()
        in_scope = collection_name in collections or path_in_scope(base_path)
    if not in_scope:
        return None

    candidates = [map_path] if map_path else _private_glob(_PRIVATE_MAP_GLOB)
    if len(candidates) > 1:
        # Picking the first would make which map applies depend on directory
        # order — a silently different substitution, which is the one failure
        # mode this module cannot afford.
        raise PrivacyMapMissing(
            f"Collection {collection_name!r} is in privacy scope but the alias map is "
            f"ambiguous: {len(candidates)} files match {_PRIVATE_MAP_GLOB}. Pass an explicit "
            f"map, or keep exactly one.")
    for candidate in candidates:
        if not candidate or not os.path.exists(candidate):
            continue
        try:
            registry = AliasRegistry.load(candidate)
        except (OSError, ValueError, json.JSONDecodeError, KeyError) as e:
            raise PrivacyMapMissing(
                f"Collection {collection_name!r} is in privacy scope but its alias map "
                f"{candidate} could not be loaded: {e}"
            ) from e
        logger.info("Privacy: aliasing %s from %s (map v%s, policy v%s)",
                    collection_name, candidate, registry.map_version, POLICY_VERSION)
        return registry

    raise PrivacyMapMissing(
        f"Collection {collection_name!r} is in privacy scope but no alias map was found "
        f"(looked for {map_path or _PRIVATE_MAP_GLOB}). Refusing to build it with real "
        f"names in the index."
    )


def resolve_registry_for_manifest(manifest, collection_name=None,
                                  map_path: str | None = None) -> AliasRegistry | None:
    """`resolve_registry` with the three arguments read out of a built manifest.

    The collection name, the reader `basePath` and the `privacy` stamp are the
    scoping decision's whole input, and every caller that has a manifest was
    unpacking them by hand — the serving store and the update factory, which
    MUST agree or a collection is updated aliased and served unaliased.
    """
    manifest = manifest or {}
    return resolve_registry(
        collection_name or manifest.get("collectionName"),
        (manifest.get("reader") or {}).get("basePath"),
        armed_by_manifest=bool(manifest.get("privacy")),
        map_path=map_path,
    )
