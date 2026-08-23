"""Query-side half of build-time people aliasing.

The index is aliased at build time (``alias_registry``), so a served in-scope
collection contains ``dev-06`` where the source said a person's name. That
leaves two query-time problems, and this module is the one place both are
solved:

1. **A real name typed into the search box.** It must never reach the parts of
   the request that persist or echo: the trace's ``query.raw``, the query log,
   ``retryHints`` (``detectedEntities`` / ``narrowerQuery`` / ``broaderQuery``),
   ``corrective.queriesTried``. Running the query through the *same* substituter
   the build used turns it into the alias the index actually holds, which fixes
   retrieval and the echo in one move. ``apply`` is idempotent — an already
   aliased query is unchanged — so the seam is safe to apply more than once.

2. **An alias token typed into the search box.** ``dev-06`` is an opaque string
   to a cross-encoder: measured on the live index, ``BAAI/bge-reranker-v2-m3``
   scores every candidate for such a query as uniform noise (0.00095), and
   ``search_policy.apply_confidence_filtering`` then drops all of them — BM25
   retrieved the right documents and the reranker threw them away. An exact
   alias-token hit in a chunk is lexical evidence the cross-encoder structurally
   cannot weigh, so the searcher pins those chunks ahead of the CE ranking. The
   pattern is derived from the registry's own alias vocabulary, never hardcoded.

Only the ``<prefix>-<number>`` alias vocabulary is pinnable. ``[~person]``,
``@person`` and ``[~ukjent-person]`` are redaction tokens, not identities:
they appear in thousands of chunks and mean "someone", so pinning on them would
promote noise.
"""
import re

# One alias: a letters-only role prefix, a hyphen, digits ("dev-06", "fag-01").
# Anything else in the map's alias column simply contributes no prefix.
_ALIAS_SHAPE = re.compile(r"^([^\W\d_]+)-\d+$")

# Cached compiled pattern, hung on the registry: the alias set is fixed for a
# registry's lifetime, and the searcher would otherwise recompile per search.
_PATTERN_ATTR = "_query_alias_pattern"


def dealias_query(query, registry):
    """The query as the aliased index spells it.

    ``registry`` is ``None`` for an out-of-scope collection, in which case the
    query is returned unchanged and nothing about that collection's search
    changes. Idempotent for the same reason ``AliasRegistry.apply`` is.
    """
    if registry is None or not query:
        return query
    return registry.apply(query)


def dealias_value(value, registry):
    """``dealias_query`` over arbitrary JSON, returning a NEW structure.

    For the knowledge graph, whose nodes were extracted from the corpus
    *before* it was aliased and therefore still spell people by name: an
    expansion term, an entity label or an entity context reaching the response
    is as much a leak as the query echo. Non-mutating on purpose — these
    structures are the loaded graph's own, shared by every request.
    """
    if registry is None:
        return value
    if isinstance(value, str):
        return registry.apply(value)
    if isinstance(value, list):
        return [dealias_value(item, registry) for item in value]
    if isinstance(value, dict):
        return {key: dealias_value(item, registry) for key, item in value.items()}
    return value


def first_armed_registry(registries):
    """The registry to use for the request's *shared* text (trace, log, hints).

    The alias map is global — one map per machine — so any armed collection's
    registry substitutes identically. Taking the first one keeps the shared text
    single-valued while the per-collection search text stays per-collection.
    """
    for registry in (registries or {}).values():
        if registry is not None:
            return registry
    return None


def unaliased_expansion_twin(raw_query, public_query, expanded_public_query):
    """The graph-expanded query as it would read in *name* space.

    Graph augmentation runs once, on the de-aliased text — running it on the raw
    text too would put the typed name into the trace's detected-entity spans,
    which is the leak this whole seam exists to close. It appends its expansion
    terms as a suffix (``augment_query``: ``q + " " + terms``), so the raw twin
    is that same suffix on the raw query.

    Returns ``None`` when nothing was de-aliased (the caller then has no twin to
    pass and every collection uses the one text), and falls back to the raw
    query if the expansion ever stops being a pure suffix.
    """
    if raw_query == public_query:
        return None
    if expanded_public_query and expanded_public_query.startswith(public_query):
        return raw_query + expanded_public_query[len(public_query):]
    return raw_query


def alias_token_pattern(registry):
    """A ``\\b(?:dev|fag|pers)-\\d+\\b`` matcher built from the registry's aliases.

    ``None`` when the registry is ``None`` or its aliases yield no usable
    prefix, which the searcher reads as "no pin possible".
    """
    if registry is None:
        return None
    cached = getattr(registry, _PATTERN_ATTR, False)
    if cached is not False:
        return cached
    prefixes = set()
    for alias in getattr(registry, "aliases", ()) or ():
        match = _ALIAS_SHAPE.match(alias)
        if match:
            prefixes.add(match.group(1).lower())
    pattern = None
    if prefixes:
        pattern = re.compile(
            r"\b(?:" + "|".join(re.escape(p) for p in sorted(prefixes)) + r")-\d+\b",
            re.IGNORECASE,
        )
    try:
        setattr(registry, _PATTERN_ATTR, pattern)
    except AttributeError:  # a stand-in object that refuses attributes
        pass
    return pattern


def alias_tokens_in(text, pattern):
    """Lower-cased alias tokens present in ``text`` (empty set when none)."""
    if not pattern or not text:
        return set()
    return {token.lower() for token in pattern.findall(text)}


def text_contains_any_token(text, tokens, pattern):
    """True when ``text`` carries one of ``tokens`` as a whole alias token.

    Matched through the same alias pattern rather than a substring test, so
    ``dev-1`` never matches inside ``dev-10`` and a chunk mentioning some other
    person's alias is not counted.
    """
    if not tokens or not text:
        return False
    return bool(alias_tokens_in(text, pattern) & tokens)
