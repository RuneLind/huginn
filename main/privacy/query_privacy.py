"""Query-side half of build-time people aliasing.

The index is aliased at build time (``alias_registry``), so a served in-scope
collection contains ``dev-06`` where the source said a person's name. A real
name typed into the search box must therefore never reach the parts of the
request that persist or echo: the trace's ``query.raw``, the query log,
``retryHints`` (``detectedEntities`` / ``narrowerQuery`` / ``broaderQuery``),
``corrective.queriesTried``. Running the query through the *same* substituter
the build used turns it into the alias the index actually holds, which fixes
retrieval and the echo in one move. ``apply`` is idempotent — an already
aliased query is unchanged — so the seam is safe to apply more than once.

Retrieval on an alias token itself needs nothing extra here: measured on the
repaired live indexes, an alias query reranks and scores normally (``fag-01``:
5 results, best 0.958). An earlier revision of this module pinned alias-token
hits past the cross-encoder because every candidate scored as uniform noise —
that was a symptom of unreadable chunk texts after a rebuild swap, not of the
cross-encoder. See ``scripts/audit/rebuild_aliased.py`` and check 12 in
``main/privacy/index_scan.py``, which is what catches it now.
"""


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
