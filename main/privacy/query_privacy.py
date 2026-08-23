"""Query-side half of build-time people aliasing.

The index is aliased at build time, so a served in-scope collection contains
``dev-06`` where the source said a person's name. A typed real name must
therefore never reach what persists or echoes — the trace's ``query.raw``, the
query log, ``retryHints``, ``corrective.queriesTried`` — and running it through
the *same* substituter the build used fixes retrieval and the echo in one move.
``AliasRegistry.apply`` is idempotent, so the seam is safe to apply twice.

The mirror image is the out-of-scope collections a mixed request also targets:
their indexes still spell people by name, so every text handed to one is a
*twin* built here. A twin never reaches the log, the trace or the response.

See CLAUDE.md's "Build-time people aliasing (privacy)" section for the whole
design, including the rebuild-swap incident this seam was once blamed for.
"""
from dataclasses import dataclass

from main.privacy.alias_registry import HANDLE_TOKEN, IDENT_TOKEN


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

    For the knowledge graph, whose nodes predate aliasing and still spell people
    by name: an expansion term, an entity id, a label or a context reaching the
    response is as much a leak as the query echo. Non-mutating on purpose —
    these structures are the loaded graph's own, shared by every request.

    Tuples are walked as tuples (one used to pass straight through with the name
    still in it); dict KEYS are left alone, being field names, not corpus text.
    """
    if registry is None:
        return value
    if isinstance(value, str):
        return registry.apply(value)
    if isinstance(value, tuple):
        return tuple(dealias_value(item, registry) for item in value)
    if isinstance(value, list):
        return [dealias_value(item, registry) for item in value]
    if isinstance(value, dict):
        return {key: dealias_value(item, registry) for key, item in value.items()}
    return value


def keep_expansion_term(term, registry):
    """False for a de-aliased term that must not be offered back to the user.

    Two categories the substituter creates or leaves behind: the **redaction
    tokens** (what an unmapped person became — noise as a "related term"), and a
    **bare given name** of a mapped entry. The latter is never substituted (one
    all-alphabetic token would alias half the corpus — ``_BARE_GIVEN_NAME_RE``),
    so a first name surfacing beside its own alias re-pairs the two.

    Without a registry nothing is filtered.
    """
    if registry is None or not term:
        return True
    if term in (IDENT_TOKEN, HANDLE_TOKEN, registry.redaction_token):
        return False
    return term.lower() not in registry.given_names


def first_armed_registry(registries):
    """The registry to use for the request's *shared* text (trace, log, hints).

    The alias map is global — one map per machine — so any armed collection's
    registry substitutes identically. Taking the first one keeps the shared text
    single-valued while the per-collection search text stays per-collection.

    Callers pass **every served** registry, not the targeted subset: a request
    scoped to an out-of-scope collection is still a request that typed a real
    name, and its query log line and trace must not carry it.
    """
    for registry in (registries or {}).values():
        if registry is not None:
            return registry
    return None


def unaliased_expansion_twin(raw_query, public_query, expanded_public_query,
                             raw_expansion_terms=()):
    """The graph-expanded query as it would read in *name* space.

    Graph augmentation runs once, on the de-aliased text — running it on the raw
    text too would put the typed name into the trace's detected-entity spans. It
    appends its terms as a suffix (``q + " " + terms``), so the twin is the raw
    query plus those terms *before* they were de-aliased. Which is why the twin
    is not conditional on the query having changed: the graph is a pre-alias
    artifact, so even a clean query picks up a suffix no out-of-scope index can
    match.

    ``None`` when there is nothing to carry over (every collection then uses the
    one text); the raw query if the expansion ever stops being a pure suffix.
    """
    if raw_expansion_terms:
        return raw_query + " " + " ".join(raw_expansion_terms)
    if raw_query == public_query:
        return None
    if expanded_public_query and expanded_public_query.startswith(public_query):
        return raw_query + expanded_public_query[len(public_query):]
    return raw_query


def name_space_twin(text, registry):
    """``text`` with its alias tokens turned back into names, or ``None``.

    Handing an alias token to an out-of-scope collection searches a name-space
    index for something it has never contained. Two texts need it: the
    corrective rescue query (built from the retry hints, so alias space with no
    typed twin to carry over) and a query the *user* typed as an alias.
    ``None`` when nothing is armed or nothing changed: the caller then uses the
    one text, exactly as before.
    """
    if registry is None or not text:
        return None
    twin = registry.to_names(text)
    return twin if twin != text else None


@dataclass(frozen=True)
class AliasedRequest:
    """One request in both spaces, shared by both transports.

    ``public_query`` is alias space: what the armed collections search and the
    only form anything records or echoes. ``name_query`` is name space: the
    typed query with any alias the *user* typed mapped back to its name, which
    is what the out-of-scope collections search and what every collection
    matches titles with. ``armed`` is None when no served collection is aliased,
    in which case both twins are None and the request behaves exactly as it did
    before this module existed.
    """

    raw_query: str
    public_query: str
    name_query: str
    armed: object

    def expansion_twin(self, expanded_public_query, raw_expansion_terms=()):
        if self.armed is None:
            return None
        return unaliased_expansion_twin(self.name_query, self.public_query,
                                        expanded_public_query, raw_expansion_terms)

    def title_boost_text(self):
        """The text to match document TITLES with — name space, always.

        Filenames are never aliased, so an alias token cannot legitimately match
        one; what it can do is match by accident. ``dev-01`` tokenises to
        ``{dev, 01}`` and boosts every ``…-01-…`` filename — measured on a live
        collection, an alias query boosted 8 candidates and surfaced three
        documents that the same person's *name* did not, which is also what made
        the two spellings return different result sets.
        """
        return None if self.armed is None else self.name_query


def prepare_aliased_request(query, registries) -> AliasedRequest:
    """Pick the shared registry and put the query in both spaces, once.

    ``registries`` is every **served** collection's registry (see
    ``first_armed_registry``). Both the HTTP route and the MCP tool ran this
    same preamble; they now share it, so the two seams cannot drift.
    """
    armed = first_armed_registry(registries)
    return AliasedRequest(
        raw_query=query,
        public_query=dealias_query(query, armed),
        # A typed alias is not name space, however raw it is.
        name_query=name_space_twin(query, armed) or query,
        armed=armed,
    )
