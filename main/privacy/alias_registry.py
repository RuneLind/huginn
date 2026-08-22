"""Build-time people aliasing for indexed collections.

The point of this module is that a *built* collection under data/collections/
carries no real person name, so it can be copied to a teammate's machine. The
substitution therefore happens inside the document converter — before chunking,
before contextual prefixing, before anything is written to disk — not at query
time. The reverse map (``huginn-*/privacy/aliases.json``) never leaves this
machine; it lives in a gitignored private sub-repo.

Five substitution classes, resolved in one left-to-right pass plus a second
handle pass:

  (a) entry variants          -> the entry's alias        ("dev-06")
  (b) unmapped_people_variants-> person_redaction_token   ("[~ukjent-person]")
  (c) non_person_labels       -> themselves (EXEMPT)      role nouns, test users,
                                                          countries. Redacting one
                                                          corrupts the corpus.
  (d) NAV idents              -> "[~person]"              "[~Q000124]" and bare
                                                          "Q000124"; the wrapper is
                                                          consumed with the token.
  (e) dotted handles          -> "@person"                only what (a)/(b) did not
                                                          already claim.

(a)-(d) share ONE compiled alternation sorted longest-first, which is what makes
(c) win over any shorter overlapping person variant: at a given position the
regex engine takes the first alternative that matches, and the longer exempt
label is listed before the shorter person variant. Doing it as separate passes
would let a person variant eat half a role phrase.

(e) runs afterwards, over the already-substituted text, which is exactly what
"not matched by a mapped variant" means: a mapped handle has become "@dev-06" by
then and no longer matches the handle shape (no dot, and digits/hyphens are
outside its character class).

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

_SCOPE_FILE = os.path.join(os.path.dirname(__file__), "scope.json")
_PRIVATE_SCOPE_GLOB = "huginn-*/privacy/scope.json"
_PRIVATE_MAP_GLOB = "huginn-*/privacy/aliases.json"
_PRIVATE_IDENT_EXCEPTIONS_GLOB = "huginn-*/privacy/ident_exceptions.json"

# A NAV ident is one letter and six digits. Either bare or inside the Jira
# mention wrapper `[~Q000124]`, which is consumed together with the token so the
# result reads `[~person]` rather than `[~[~person]]`.
_IDENT_ALTERNATIVE = r"\[~\s*[A-Za-z]\d{6}\s*\]|(?<![\w])[A-Za-z]\d{6}(?![\w])"
_IDENT_INNER = re.compile(r"[A-Za-z]\d{6}")

# Lowercase-only on purpose: it makes the "annotation path" exclusion free
# (`@Abac.Attr` never matches at all) and every mapped slug variant is lowercase
# too, so a capitalised handle is still caught case-insensitively by pass 1.
_HANDLE_RE = re.compile(r"@[a-zæøå]+(?:\.[a-zæøå]+)+")
_HANDLE_TLDS = frozenset({"no", "com", "org", "net", "io", "eu", "co", "uk",
                          "se", "dk", "de", "nl", "info", "dev", "ai", "local"})
_HANDLE_PACKAGE_ROOTS = frozenset({"org", "com", "no", "io", "java"})

# Class precedence inside the alternation for equal-length literals. Exempt
# beats person beats redaction: exempting a role noun is safe, redacting one
# corrupts the corpus.
_EXEMPT, _PERSON, _REDACT = 0, 1, 2


class PrivacyMapMissing(RuntimeError):
    """An in-scope collection was built without a loadable alias map.

    Raised *before* the collection folder is removed or anything is written, so
    a missing private sub-repo fails the build instead of silently producing an
    index full of real names.
    """


def _boundaried(literal: str) -> str:
    r"""Escape `literal` and wrap it in the right word boundaries.

    Two regimes, because the variants are two different shapes:

    * A *slug* — a single token containing `.` or `_` ("ada.example",
      "ada_example") — must not match a *longer* dotted path: `ada.example` is
      not the person in `ada.example.no` or `no.nav.ada.example`, and matching
      there would weld a surviving surname onto an alias. So a `.` followed by
      another path segment blocks on the right, and a preceding `.` blocks on
      the left. A `.` that ends a sentence ("Kommentar fra ada.example.") must
      NOT block — that one left a real slug unaliased.
      On its left a slug does not demand a non-word character either: a dotted
      full name shows up percent-encoded inside URLs
      ("…%2CAda.Example%40nav.no"), where the preceding character is the `C` of
      `%2C`, and a `\w` lookbehind there left a real full name in the clear.
      Several dotted name-tokens cannot plausibly be the tail of another word.
    * Everything else — anything with whitespace ("Ada Example", "Example, Ada",
      "Ada Example [X]", "Ada K. Example") and every bare single token
      ("Zylphia", "case-owner", "srvtestbruker") — gets plain `\w` boundaries.
      A trailing full stop must NOT block those: it is a sentence end, and
      blocking it left an unmapped mononym unredacted at the end of a sentence.

    The boundary is only applied on a side whose edge character is a word
    character; `Ada Example [X]` ends in `]`, and demanding a non-word character
    after it would fail on `[X]and` and leave a real name in the clear.
    """
    body = re.escape(literal)
    slug = ("." in literal or "_" in literal) and not any(c.isspace() for c in literal)
    left = r"(?<![.\-])" if slug else r"(?<!\w)"
    right = r"(?![\w\-])(?!\.\w)" if slug else r"(?!\w)"
    prefix = left if literal[:1].isalnum() or literal[:1] == "_" else ""
    suffix = right if literal[-1:].isalnum() or literal[-1:] == "_" else ""
    return f"{prefix}{body}{suffix}"


class AliasRegistry:
    """Compiled substituter for one alias map."""

    def __init__(self, map_data: dict, ident_exceptions=(), source_path: str | None = None):
        self.policy_version = POLICY_VERSION
        self.map_version = map_data.get("version")
        self.source_path = source_path
        self.redaction_token = map_data.get("person_redaction_token") or "[~ukjent-person]"
        self._ident_exceptions = {t.lower() for t in ident_exceptions}

        # literal (lowercased) -> replacement, or None meaning "leave as is"
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
        for _, rank, literal, replacement in ranked:
            key = literal.lower()
            if key in seen_rank:
                # Same literal twice. Within a class this is just case variation
                # in the map ("Saksbehandler"/"saksbehandler"); across classes it
                # means the map's own lints let a real conflict through, so say so.
                if seen_rank[key] != rank:
                    logger.warning("Alias map: literal %r appears in two classes; keeping the "
                                   "higher-precedence one", literal)
                continue
            seen_rank[key] = rank
            self._replacements[key] = None if rank == _EXEMPT else replacement
            alternatives.append(_boundaried(literal))

        alternatives.append(_IDENT_ALTERNATIVE)
        self._pattern = re.compile("|".join(alternatives), re.IGNORECASE)

    # --- loading -----------------------------------------------------------

    @classmethod
    def load(cls, path: str, ident_exceptions_path: str | None = None) -> "AliasRegistry":
        with open(path, "r", encoding="utf-8") as f:
            map_data = json.load(f)
        if not isinstance(map_data.get("entries"), list):
            raise ValueError(f"Alias map {path} has no 'entries' list")
        return cls(map_data, ident_exceptions=_load_ident_exceptions(ident_exceptions_path), source_path=path)

    # --- substitution ------------------------------------------------------

    def _substitute(self, match: re.Match) -> str:
        matched = match.group(0)
        key = matched.lower()
        if key in self._replacements:
            replacement = self._replacements[key]
            return matched if replacement is None else replacement
        # Not a literal => the ident alternative fired.
        inner = _IDENT_INNER.search(matched)
        if inner is None:
            # Unreachable for this corpus: it needs a character whose str.lower()
            # disagrees with the regex's IGNORECASE fold (dotted I and friends).
            # Redact rather than pass it through — a leak is worse than a scar.
            logger.warning("Alias map: matched text did not resolve to a literal or an ident")
            return self.redaction_token
        if inner.group(0).lower() in self._ident_exceptions:
            return matched
        return IDENT_TOKEN

    def _substitute_handle(self, match: re.Match) -> str:
        segments = match.group(0)[1:].split(".")
        if segments[-1] in _HANDLE_TLDS:
            return match.group(0)          # an email domain, not a handle
        if len(segments) >= 3 and segments[0] in _HANDLE_PACKAGE_ROOTS:
            return match.group(0)          # org.springframework.… and friends
        return HANDLE_TOKEN

    def apply(self, text: str) -> str:
        if not text:
            return text
        substituted = self._pattern.sub(self._substitute, text)
        return _HANDLE_RE.sub(self._substitute_handle, substituted)

    def _apply_metadata(self, metadata) -> bool:
        """Alias string and list-of-string metadata values in place."""
        if not isinstance(metadata, dict):
            return False
        changed = False
        for key, value in metadata.items():
            if isinstance(value, str):
                new_value = self.apply(value)
                if new_value != value:
                    metadata[key] = new_value
                    changed = True
            elif isinstance(value, list) and all(isinstance(v, str) for v in value):
                new_value = [self.apply(v) for v in value]
                if new_value != value:
                    metadata[key] = new_value
                    changed = True
        return changed

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
            "policy_version": self.policy_version,
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


def _load_ident_exceptions(path: str | None) -> set:
    """Literal ident-shaped tokens that are NOT people (git SHAs, document type
    ids, test users). Missing file => no exceptions, i.e. redact everything."""
    candidates = [path] if path else sorted(glob.glob(_PRIVATE_IDENT_EXCEPTIONS_GLOB))
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
    discovered. Relative basePaths resolve against the process CWD, the same way
    FilesDocumentReader and the update factory read them.
    """
    collections, base_paths = set(), set()
    for path in [_SCOPE_FILE, *sorted(glob.glob(_PRIVATE_SCOPE_GLOB))]:
        if not os.path.exists(path):
            continue
        data = _load_json(path)
        if not isinstance(data, dict):
            continue
        collections.update(c for c in data.get("collections", []) if isinstance(c, str) and c)
        for base_path in data.get("basePaths", []):
            if isinstance(base_path, str) and base_path:
                base_paths.add(os.path.realpath(base_path))
    return collections, base_paths


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
        collections, base_paths = load_scope()
        in_scope = collection_name in collections or (
            bool(base_path) and os.path.realpath(base_path) in base_paths
        )
    if not in_scope:
        return None

    candidates = [map_path] if map_path else sorted(glob.glob(_PRIVATE_MAP_GLOB))
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
                    collection_name, candidate, registry.map_version, registry.policy_version)
        return registry

    raise PrivacyMapMissing(
        f"Collection {collection_name!r} is in privacy scope but no alias map was found "
        f"(looked for {map_path or _PRIVATE_MAP_GLOB}). Refusing to build it with real "
        f"names in the index."
    )
