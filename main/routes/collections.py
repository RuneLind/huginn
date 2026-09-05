"""Collection-level routes — listing, tags, document lookup, manual update."""
import json
import logging
import math
import os
import shutil

from datetime import datetime, timedelta, timezone
from statistics import median

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request

from main.runtime.indexing_run_ledger import (
    INCOMPLETE_AFTER_SECONDS,
    MAX_RECORD_BYTES,
    VALID_TRIGGERS,
    IndexingRunLedger,
    InvalidCollectionName,
)
from main.runtime.indexing_schedule import load_schedules
from main.runtime.knowledge_store import (
    KnowledgeStore,
    get_store,
    maybe_enqueue_reindex,
    run_collection_update,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _reader_patterns(manifest: dict) -> tuple[list, list]:
    """The reader's EFFECTIVE include/exclude patterns for a localFiles collection.

    Mirrors ``_build_local_files`` in the update-collection factory: a localFiles
    reader that omits ``includePatterns`` defaults to ``[".*"]`` (index-all) and an
    omitted ``excludePatterns`` to ``[]``. Returning the effective patterns (not the
    literal manifest fields) matters for consumers that partition on-disk files by
    these rules (muninn's wiki index-coverage card) — an empty include would
    otherwise read as "index nothing" and mislabel every page. Non-localFiles
    readers (jira/confluence/notion) have no such concept ⇒ empty arrays.
    """
    reader = manifest.get("reader") or {}
    if reader.get("type") == "localFiles":
        include = reader.get("includePatterns")
        exclude = reader.get("excludePatterns")
        return (
            include if include is not None else [".*"],
            exclude if exclude is not None else [],
        )
    return [], []


@router.get("/api/collections")
def list_collections(store: KnowledgeStore = Depends(get_store)):
    result = []
    for name, searcher in store.get_searchers().items():
        try:
            manifest_text = store.disk_persister.read_text_file(f"{name}/manifest.json")
            manifest = json.loads(manifest_text)
        except FileNotFoundError:
            manifest = {}
        include_patterns, exclude_patterns = _reader_patterns(manifest)
        result.append({
            "name": name,
            "document_count": manifest.get("numberOfDocuments", 0),
            "chunk_count": manifest.get("numberOfChunks", 0),
            "embedding_count": searcher.indexer.get_size(),
            "updatedTime": manifest.get("updatedTime"),
            # Reader file-selection rules, exposed so callers can tell a deliberately
            # excluded/out-of-scope file (meta denylist, scoped include) from a real
            # indexing gap. Empty arrays for readers without file patterns.
            "includePatterns": include_patterns,
            "excludePatterns": exclude_patterns,
        })
    return {"collections": result}


@router.get("/api/tags")
def list_tags(
    collection: str = Query(None, description="Collection name (all if omitted)"),
    store: KnowledgeStore = Depends(get_store),
):
    """Return tag distribution for a collection (or all collections). Cached at startup."""
    target_names = [collection] if collection else store.collection_names()
    result = {}
    for name in target_names:
        if not store.has_collection(name):
            raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
        tags = store.get_tag_counts([name]).get(name, {})
        result[name] = {
            "unique_tags": len(tags),
            "tags": tags,
        }
    return result


def _resolve_doc_date(doc: dict) -> str | None:
    """Best-effort 'added' date for a document.

    Prefers the frontmatter ``date`` (day-precision, set at ingest) and falls
    back to ``modifiedTime`` (file mtime, which can be reset by bulk reindexing).
    """
    metadata = doc.get("metadata") or {}
    return metadata.get("date") or doc.get("modifiedTime")


#: Ranking scores attached by ``include_scores``. ``combined_score`` is the one
#: callers rank on; the two inputs ride along for debuggability.
SCORE_FIELDS = ("combined_score", "relevance_score", "engagement_score")


def _resolve_doc_scores(doc: dict) -> dict[str, float]:
    """Coerce a document's frontmatter ranking scores to floats.

    ``read_frontmatter`` serves numerics as STRINGS (e.g. ``"0.604"``), which sort
    lexicographically if a caller uses them raw ("0.9" < "0.1234"). Coercing here
    means every consumer gets a real number. A field that is absent, non-numeric,
    or non-finite is *omitted* rather than emitted as a string or a NaN — callers
    can then treat "key missing" as the single "no score" signal.
    """
    metadata = doc.get("metadata") or {}
    scores: dict[str, float] = {}
    for field in SCORE_FIELDS:
        raw = metadata.get(field)
        if raw is None or isinstance(raw, bool):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        scores[field] = value
    return scores


def _read_doc(store: KnowledgeStore, doc_path: str) -> dict | None:
    """Read and parse a single document JSON, or return ``None`` on error.

    A missing/unreadable file or malformed JSON yields ``None`` (logged), so one bad
    document doesn't fail the whole listing — but genuinely unexpected errors are
    left to propagate rather than silently swallowed.
    """
    if not doc_path:
        return None
    try:
        return json.loads(store.disk_persister.read_text_file(doc_path))
    except (OSError, ValueError) as e:
        logger.warning("Could not read document %s: %s", doc_path, e)
        return None


@router.get("/api/collection/{name}/documents")
def list_collection_documents(
    name: str,
    include_dates: bool = Query(
        False,
        description="Attach each document's added date. Slower — reads every document file.",
    ),
    include_scores: bool = Query(
        False,
        description="Attach each document's ranking scores as floats. Slower — reads every document file.",
    ),
    include_thumbnails: bool = Query(
        False,
        description="Attach each document's frontmatter thumbnail_url when it has one. Slower — reads every document file.",
    ),
    store: KnowledgeStore = Depends(get_store),
):
    """List all documents in a collection with their IDs and URLs.

    When ``include_dates`` is set, each entry also carries a ``date`` field
    (frontmatter date, falling back to file mtime) so callers can sort/group by
    recency, plus a ``modifiedTime`` field (full-precision ingest timestamp,
    when the document has one) so callers can break intra-day ties.

    When ``include_scores`` is set, each entry carries whichever of
    ``combined_score`` / ``relevance_score`` / ``engagement_score`` the document
    has, coerced to floats (see ``_resolve_doc_scores``) so callers can rank the
    listing before fetching bodies. Absent/unparseable scores are omitted.

    When ``include_thumbnails`` is set, each entry that has a string
    ``thumbnail_url`` in its frontmatter carries it (the Vimeo ingest writes
    one); absent or non-string is omitted, so "key missing" stays the one
    no-thumbnail signal.

    All three flags read every document file, so they are opt-in to keep the
    default listing (used by hot paths like duplicate checks) cheap. Setting
    several still reads each file only once.
    """
    if not store.has_collection(name):
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    try:
        mapping_text = store.disk_persister.read_text_file(
            f"{name}/indexes/index_document_mapping.json"
        )
        mapping = json.loads(mapping_text)
    except Exception:
        return {"documents": []}

    seen_ids = set()
    documents = []
    for entry in mapping.values():
        doc_id = entry.get("documentId", "")
        doc_url = entry.get("documentUrl", "")
        if doc_id in seen_ids or not doc_url:
            continue
        seen_ids.add(doc_id)
        doc = {"id": doc_id, "url": doc_url}
        if include_dates or include_scores or include_thumbnails:
            parsed = _read_doc(store, entry.get("documentPath", ""))
            # A document JSON that parses to a list/string is still "unreadable"
            # for our purposes — the resolvers below call ``.get``, so anything
            # that isn't a dict has to collapse to the empty-dict no-op.
            raw = parsed if isinstance(parsed, dict) else {}
            if include_dates:
                doc["date"] = _resolve_doc_date(raw)
                modified_time = raw.get("modifiedTime")
                if modified_time:
                    doc["modifiedTime"] = modified_time
            if include_scores:
                doc.update(_resolve_doc_scores(raw))
            if include_thumbnails:
                thumbnail = (raw.get("metadata") or {}).get("thumbnail_url")
                if isinstance(thumbnail, str) and thumbnail:
                    doc["thumbnail_url"] = thumbnail
        documents.append(doc)

    return {"documents": documents}


def _is_inside(base_dir: str, resolved: str) -> bool:
    """Is ``resolved`` a path strictly INSIDE ``base_dir``? (both already realpath'd)

    The one containment rule every path-taking route here shares. Deliberately
    strict (``base_dir`` itself is not "inside" it) and separator-anchored, so a
    sibling whose name merely shares a prefix — ``/data/sources/x-articles-old``
    against ``/data/sources/x-articles`` — is correctly a different tree.
    """
    return resolved.startswith(base_dir + os.sep)


@router.get("/api/document/{collection}/{doc_id:path}")
def get_document(collection: str, doc_id: str, store: KnowledgeStore = Depends(get_store)):
    if not store.has_collection(collection):
        raise HTTPException(status_code=404, detail=f"Collection '{collection}' not found")

    if doc_id.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid document ID")

    doc_path = f"{collection}/documents/{doc_id}"
    if not doc_id.endswith(".json"):
        doc_path += ".json"

    base_dir = os.path.realpath(store.disk_persister.base_path)
    try:
        resolved = os.path.realpath(os.path.join(base_dir, doc_path))
    except ValueError:
        # Embedded NUL (``%00`` in the URL) — realpath raises rather than
        # returning something we could containment-check. Same 400 as traversal.
        raise HTTPException(status_code=400, detail="Invalid document ID")
    if not _is_inside(base_dir, resolved):
        raise HTTPException(status_code=400, detail="Invalid document ID")

    try:
        doc_text = store.disk_persister.read_text_file(doc_path)
        return json.loads(doc_text)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")


#: Where soft-deleted source files are parked. Deliberately CWD-relative by
#: default, exactly like a manifest's relative ``reader.basePath`` — the server's
#: working directory is the one anchor both already share. ``data/`` is
#: gitignored, so nothing here is ever committed. Overridable (tests, and an
#: operator who wants the trash on another volume) via ``HUGINN_DELETED_DIR``.
DELETED_DIR_ENV = "HUGINN_DELETED_DIR"
DEFAULT_DELETED_DIR = "./data/deleted"


def _deleted_root() -> str:
    return os.environ.get(DELETED_DIR_ENV) or DEFAULT_DELETED_DIR


def _localfiles_base_path(store: KnowledgeStore, collection: str) -> str:
    """Resolved, existing ``reader.basePath`` for a localFiles collection.

    400 (not 404/500) for every "this collection cannot be deleted from" case, so
    the caller learns the request was wrong rather than getting a half-done
    delete: a non-localFiles reader has no enumerable source tree, so the orphan
    pruning this endpoint depends on never runs for it (see
    ``__prune_orphaned_documents`` — it is skipped for readers without
    ``get_all_document_ids``). A relative basePath resolves against this process's
    CWD, matching how ``FilesDocumentReader`` and the update factory
    (``DiskPersister(base_path="./data/collections")``) already read it.
    """
    try:
        manifest = json.loads(
            store.disk_persister.read_text_file(f"{collection}/manifest.json")
        )
    except (OSError, ValueError) as e:
        raise HTTPException(
            status_code=500, detail=f"Could not read manifest for '{collection}': {e}"
        )

    reader = manifest.get("reader") or {}
    if reader.get("type") != "localFiles":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Collection '{collection}' has reader type "
                f"'{reader.get('type')}'; deletion is only supported for "
                f"'localFiles' collections"
            ),
        )

    base_path = reader.get("basePath")
    if not base_path:
        raise HTTPException(
            status_code=400,
            detail=f"Collection '{collection}' has no reader.basePath",
        )

    resolved = os.path.realpath(base_path)
    if not os.path.isdir(resolved):
        raise HTTPException(
            status_code=400,
            detail=(
                f"reader.basePath '{base_path}' for collection '{collection}' "
                f"does not resolve to an existing directory (resolved: {resolved})"
            ),
        )
    return resolved


def _resolve_source_file(base_dir: str, doc_id: str) -> str:
    """Absolute path of the source file backing ``doc_id``, containment-guarded.

    A document id is the source file's path relative to basePath (see
    ``FilesDocumentReader.get_all_document_ids``), so it maps back by simple
    join — but it arrives from an untrusted URL path segment, so the REALPATH of
    the result must still land strictly inside the realpath of basePath.
    Comparing resolved paths (not the raw string) is what catches both ``../``
    traversal and a symlink pointing out of the tree.

    A doc id that is ITSELF a symlink is refused outright, even when its target
    is inside basePath: realpath would resolve ``alias.md`` to ``target.md`` and
    the move would then delete a DIFFERENT document while leaving ``alias.md``
    dangling. Symlinked source files are exotic; refusing beats guessing.
    """
    if not doc_id or doc_id.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid document ID")

    try:
        lexical = os.path.join(base_dir, doc_id)
        if os.path.islink(lexical):
            raise HTTPException(status_code=400, detail="Invalid document ID")
        resolved = os.path.realpath(lexical)
    except ValueError:
        # Embedded NUL (``DELETE .../a%00b``) — os.lstat/realpath raise instead
        # of returning a path. Same "this id is not a path" 400 as traversal.
        raise HTTPException(status_code=400, detail="Invalid document ID")

    if not _is_inside(base_dir, resolved):
        raise HTTPException(status_code=400, detail="Invalid document ID")
    return resolved


def _free_destination(path: str) -> str:
    """``path``, or the first free ``name.N.ext`` beside it.

    Two junk docs can share a basename across ingests (or the same one can be
    re-ingested and re-deleted), and the trash is the only undo story for a
    delete — overwriting a previous soft-delete would destroy it silently.
    """
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    counter = 1
    while os.path.exists(f"{stem}.{counter}{ext}"):
        counter += 1
    return f"{stem}.{counter}{ext}"


def _free_directory(path: str) -> str:
    """``path`` if it is free or already a directory, else the first free ``path.N``.

    The directory-shaped twin of ``_free_destination``. Trash paths mirror source
    ids, and a doc id is only ever a file — so deleting extensionless ``x`` and
    later ``x/y.md`` needs ``x`` to be a FILE and then a DIRECTORY at the same
    trash path. Without side-stepping, ``makedirs`` raises and that second delete
    500s forever. Nothing existing is overwritten either way.
    """
    if not os.path.exists(path) or os.path.isdir(path):
        return path
    stem, ext = os.path.splitext(path)
    counter = 1
    while True:
        candidate = f"{stem}.{counter}{ext}"
        if not os.path.exists(candidate) or os.path.isdir(candidate):
            return candidate
        counter += 1


def _prepare_destination(root_dir: str, rel_path: str) -> str:
    """Create the trash directory chain for ``rel_path`` and return a free leaf.

    Walks the chain one segment at a time so a collision at ANY level (not just
    the leaf) is side-stepped rather than fatal, and returns the path the move
    should target.
    """
    parts = [p for p in rel_path.split(os.sep) if p]
    current = root_dir
    for part in parts[:-1]:
        current = _free_directory(os.path.join(current, part))
    os.makedirs(current, exist_ok=True)
    return _free_destination(os.path.join(current, parts[-1]))


def _collections_sharing_source(store: KnowledgeStore, source_path: str,
                                named_collection: str) -> list[str]:
    """Every served localFiles collection whose basePath contains ``source_path``.

    Several collections deliberately share one basePath (``wiki`` + ``wiki-life``
    over the jarvis wiki; the ``nav-wiki*`` family; ``jira-issues`` + its
    baseline). Reindexing only the named one would leave the siblings
    serving the deleted document from a dangling index entry, so all of them get
    the same reconciliation. The named collection comes first.
    """
    result = [named_collection]
    for name in store.collection_names():
        if name == named_collection:
            continue
        try:
            manifest = json.loads(
                store.disk_persister.read_text_file(f"{name}/manifest.json")
            )
        except (OSError, ValueError):
            continue
        reader = manifest.get("reader") or {}
        if reader.get("type") != "localFiles" or not reader.get("basePath"):
            continue
        try:
            resolved = os.path.realpath(reader["basePath"])
        except ValueError:
            continue
        if os.path.isdir(resolved) and _is_inside(resolved, source_path):
            result.append(name)
    return result


def _require_indexed_document(store: KnowledgeStore, collection: str, doc_id: str) -> None:
    """404 unless ``doc_id`` is an actual indexed document of ``collection``.

    basePath is not the collection — for several wikis it is a live git repo root
    whose reader excludes most of what lives there (``CLAUDE.md``, ``index.md``,
    dot-dirs), and ``.git/`` is skipped by the reader's walk outright. Without
    this check the endpoint happily moves ``.git/config`` or an excluded page out
    of a real repository: a destructive edit to something the collection never
    owned, and one no reindex would ever "complete", since pruning only ever
    reconciles against documents that ARE in the index.

    Checked against the persisted index mapping rather than a fresh reader
    enumeration: it is a single small JSON read, and it is exactly the set the
    subsequent orphan pruning reconciles against.
    """
    path = f"{collection}/indexes/reverse_index_document_mapping.json"
    try:
        mapping = json.loads(store.disk_persister.read_text_file(path))
    except (OSError, ValueError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read the index mapping for '{collection}': {e}",
        )
    if doc_id not in mapping:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Document '{doc_id}' is not indexed in collection '{collection}' "
                f"(the file may exist under reader.basePath but be excluded from "
                f"the collection); refusing to move it"
            ),
        )


def _relative_to_cwd(path: str) -> str:
    """``path`` relative to the server's CWD when it is under it, else absolute."""
    cwd = os.path.realpath(os.getcwd())
    if path == cwd or path.startswith(cwd + os.sep):
        return os.path.relpath(path, cwd)
    return path


@router.delete("/api/document/{collection}/{doc_id:path}")
def delete_document(
    collection: str,
    doc_id: str,
    background_tasks: BackgroundTasks,
    store: KnowledgeStore = Depends(get_store),
):
    """Soft-delete a document by moving its SOURCE file out of the collection.

    Deleting the derived JSON under ``<collection>/documents/`` would achieve
    nothing durable: it is regenerated from the source markdown under
    ``reader.basePath``, and the index entry would survive pointing at a path
    that no longer exists. Removing the SOURCE is the operation that sticks —
    the next incremental update's orphan pruning
    (``DocumentCollectionCreator.__prune_orphaned_documents``) reconciles the
    index against the reader's full id enumeration and drops both the index
    entries and the derived JSON.

    The source is MOVED to ``data/deleted/<collection>/<doc_id>`` rather than
    unlinked, and deliberately to a location OUTSIDE basePath. A ``.excluded/``
    folder inside basePath would only work for collections that declare a
    matching ``excludePatterns`` — none of the summary collections do, and with
    ``includePatterns: [".*"]`` the moved file would simply be re-indexed under a
    new id. Moving out of the tree needs no manifest edit and, since ``data/`` is
    gitignored (so a hard unlink would have no undo), doubles as the undo story.

    Deletion is NOT synchronous. The move is immediate; the document leaves
    search and the document listing only once the background update finishes.
    ``reindex`` is a per-collection map, because several collections share one
    basePath (``wiki`` + ``wiki-life``, the ``nav-wiki*`` family, …) and every one
    of them would otherwise keep serving the deleted document from a dangling
    index entry — the named collection is reported first, its siblings after.
    ``pollUrls`` carries an update-status URL for the ``started`` collections
    ONLY: a ``skipped_already_running`` collection's status belongs to the update
    that was already in flight, which will report ``succeeded`` without ever
    having seen this delete. Those need a manual POST
    ``/api/collections/{name}/update`` afterwards. The delete is never queued
    behind the running update, because that update may already have passed its
    own enumeration step and would then leave the index stale with no signal.
    """
    if not store.has_collection(collection):
        raise HTTPException(status_code=404, detail=f"Collection '{collection}' not found")

    # ``DELETE .../deep.md/`` reaches the handler with the trailing slash intact;
    # normalize so the echoed doc_id is the real, usable id.
    doc_id = doc_id.rstrip("/")

    base_dir = _localfiles_base_path(store, collection)
    source_path = _resolve_source_file(base_dir, doc_id)

    if not os.path.isfile(source_path):
        raise HTTPException(
            status_code=404,
            detail=f"Source file for document '{doc_id}' not found in collection '{collection}'",
        )
    _require_indexed_document(store, collection, doc_id)

    source_rel = os.path.relpath(source_path, base_dir)
    trash_root = _deleted_root()
    destination = os.path.realpath(os.path.join(trash_root, collection, source_rel))
    # The whole design rests on the destination being outside the reader's tree.
    # A pathological config (a deleted-dir nested under basePath) would otherwise
    # leave the file indexed under a new id instead of deleting it.
    if destination == base_dir or _is_inside(base_dir, destination):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Deleted-documents directory resolves inside reader.basePath for "
                f"collection '{collection}'; refusing to move (the file would be "
                f"re-indexed). Set {DELETED_DIR_ENV} to a path outside {base_dir}."
            ),
        )

    try:
        destination = os.path.realpath(
            _prepare_destination(trash_root, os.path.join(collection, source_rel))
        )
        shutil.move(source_path, destination)
    except OSError as e:
        logger.warning("Could not move source file for %s/%s", collection, doc_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not move source file: {e}")

    logger.info("Soft-deleted %s from %s -> %s", doc_id, collection, destination)

    # Same enqueue contract the ingest routes use (try_begin_update IS the
    # per-collection mutex), applied to every collection reading this file.
    reindex = {
        name: maybe_enqueue_reindex(
            store, background_tasks, name, trigger="manual", variant="incremental"
        )
        for name in _collections_sharing_source(store, source_path, collection)
    }

    return {
        "status": "deleted",
        "collection": collection,
        "doc_id": doc_id,
        "movedTo": _relative_to_cwd(destination),
        "reindex": reindex,
        "pollUrls": {
            name: f"/api/collections/{name}/update-status"
            for name, status in reindex.items()
            if status == "started"
        },
    }


async def _optional_correlation(request: Request) -> dict:
    """Parse the optional {runId, job, trigger, variant} body of POST /update.

    Existing callers (the launchd shell scripts today) send no body and no
    Content-Type at all, so anything unparseable is treated as "no correlation
    supplied" rather than a 400 — backward compatibility of this endpoint is
    mandatory in both directions.
    """
    try:
        raw = await request.body()
    except Exception:
        return {}
    if not raw:
        return {}
    try:
        body = json.loads(raw)
    except ValueError:
        return {}
    if not isinstance(body, dict):
        return {}
    trigger = body.get("trigger")
    return {
        "run_id": body.get("runId") or None,
        "job": body.get("job") or None,
        "trigger": trigger if trigger in VALID_TRIGGERS else None,
        "variant": body.get("variant") or None,
    }


@router.post("/api/collections/{name}/update")
async def update_collection(
    name: str,
    request: Request,
    background_tasks: BackgroundTasks,
    store: KnowledgeStore = Depends(get_store),
):
    if not store.has_collection(name):
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    correlation = await _optional_correlation(request)
    if not store.try_begin_update(
        name,
        run_id=correlation.get("run_id"),
        job=correlation.get("job"),
        trigger=correlation.get("trigger"),
        variant=correlation.get("variant") or "incremental",
    ):
        raise HTTPException(
            status_code=409, detail=f"An update for collection '{name}' is already in progress"
        )

    background_tasks.add_task(run_collection_update, name, store)
    return {"status": "update_started", "collection": name}


@router.post("/api/collections/{name}/reload")
def reload_collection(name: str, store: KnowledgeStore = Depends(get_store)):
    """Swap a served collection's in-memory searcher for the one on disk.

    A rebuild done out-of-band (the x-feed watch job builds a fresh index under a
    temp name and renames it into place) leaves this process serving its stale
    in-memory searcher until someone reloads it. This endpoint does exactly that
    reload without any rebuild of its own.

    Gated on ``has_collection``: ``reload_collection`` unconditionally inserts into
    ``self.searchers``, so an ungated route would let a caller load an arbitrary new
    collection this server was never configured to serve. Unknown collection ⇒ 404.
    """
    if not store.has_collection(name):
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    # reload_collection builds the new searcher before swapping it in, so a
    # failure here (a missing/broken on-disk dir at reload time) leaves the old
    # in-memory searcher untouched and still serving. Surface a clean 500 saying
    # so, instead of a bare traceback.
    try:
        store.reload_collection(name)
    except Exception as e:
        logger.warning("Could not reload collection %s", name, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Could not reload collection '{name}': {e}; previous index still serving",
        )
    return {"reloaded": name}


@router.get("/api/collections/{name}/update-status")
def collection_update_status(name: str, store: KnowledgeStore = Depends(get_store)):
    """Report the outcome of the most recent (or in-flight) update for a collection.

    status is one of idle / running / succeeded / failed; a failed update carries
    its error so a stale collection surfaces instead of hiding behind an earlier 200.
    """
    if not store.has_collection(name):
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    return store.get_update_status(name)


def _parse_iso(value: str | None) -> datetime | None:
    """Aware datetime from an ISO string, or None. Naive values are taken as UTC.

    Accepts both timestamp dialects this endpoint meets: the ledger's fixed-width
    ``...Z`` form and the in-memory update state's ``+00:00`` isoformat.
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _elapsed_seconds(started_at: str | None) -> int | None:
    start = _parse_iso(started_at)
    if start is None:
        return None
    return max(0, int((datetime.now(timezone.utc) - start).total_seconds()))


# The folded ledger record is writer-defined: any key any writer ever appended
# survives folding, and POST /api/indexing/runs accepts extra keys by design.
# ``lastRun`` is the endpoint's contract with consumers (the muninn dashboard),
# so it exposes this FIXED projection — every key always present, None when the
# run doesn't carry it — instead of the raw record. Internal bookkeeping
# (``source``, ``stage``, ``unclosedSources``, ``sourceLog``, arbitrary extras)
# stays out; a writer adding a field makes a deliberate decision to publish it
# by adding it here, covered by the response-shape test.
LAST_RUN_FIELDS = (
    "runId", "startedAt", "finishedAt", "durationSeconds", "status", "variant",
    "job", "trigger", "documentCount", "chunkCount", "phases", "error",
)

# Fixed window for medianDurationSeconds, decoupled from the ``history`` query
# param — two dashboard widgets asking for different history depths must not
# disagree about what a collection's "median" is.
MEDIAN_WINDOW_RUNS = 50


def _project_run(run: dict | None) -> dict | None:
    if run is None:
        return None
    return {field: run.get(field) for field in LAST_RUN_FIELDS}


def _next_run_at(schedule, last_run: dict | None, now: datetime | None = None) -> str | None:
    """Next scheduled fire as a UTC ``...Z`` timestamp, or None when unknowable.

    launchd calendar entries are machine-local wall-clock while every timestamp
    this endpoint emits is UTC; computing "next run" server-side is what spares
    consumers mixing the two (a 2h error in Oslo summer). Wall-clock arithmetic
    is done in naive local time and converted at the end, so the answer matches
    what launchd will actually do on this machine. launchd weekday numbering:
    0 and 7 are both Sunday. ``interval`` schedules fire relative to load time,
    which this process cannot see — approximated as lastRun.finishedAt + seconds,
    None when there is no finished run.
    """
    if not isinstance(schedule, dict):
        return None
    now = now or datetime.now(timezone.utc)
    local = now.astimezone().replace(tzinfo=None)
    kind = schedule.get("kind")
    if kind == "hourly" and isinstance(schedule.get("minute"), int):
        candidate = local.replace(minute=schedule["minute"], second=0, microsecond=0)
        if candidate <= local:
            candidate += timedelta(hours=1)
    elif kind == "calendar" and isinstance(schedule.get("hour"), int):
        minute = schedule.get("minute") if isinstance(schedule.get("minute"), int) else 0
        candidate = local.replace(hour=schedule["hour"], minute=minute,
                                  second=0, microsecond=0)
        if candidate <= local:
            candidate += timedelta(days=1)
        weekday = schedule.get("weekday")
        if isinstance(weekday, int):
            target = (weekday % 7 + 6) % 7  # launchd Sunday=0/7 -> Python Monday=0
            while candidate.weekday() != target:
                candidate += timedelta(days=1)
    elif kind == "interval" and isinstance(schedule.get("seconds"), int):
        finished = _parse_iso((last_run or {}).get("finishedAt"))
        if finished is None:
            return None
        return (finished + timedelta(seconds=schedule["seconds"])) \
            .astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        return None
    return candidate.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _current_running(state: dict | None, last_run: dict | None) -> dict | None:
    """The single "is anything running right now" channel for a collection.

    Merges the two sources that can each see work the other cannot: the
    in-memory update state (a reindex THIS process is executing) and the folded
    ledger (a script-side run — fetch/tag phases, or any run on a collection
    this server does not serve). ``source`` says which side(s) reported:
    ``reindex`` / ``script`` / ``both``. When both report, ``startedAt`` is the
    earlier of the two — the script wraps the reindex, so its start is the
    whole-run start and the elapsed the dashboard should show.
    """
    sources = []
    started = []
    if state and state.get("status") == "running":
        sources.append("reindex")
        started.append(_parse_iso(state.get("startedAt")))
    if last_run and last_run.get("status") == "running":
        sources.append("script")
        started.append(_parse_iso(last_run.get("startedAt")))
    if not sources:
        return None
    known = [s for s in started if s is not None]
    started_at = min(known).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") \
        if known else None
    return {
        "status": "running",
        "source": "both" if len(sources) == 2 else sources[0],
        "startedAt": started_at,
        "elapsedSeconds": _elapsed_seconds(started_at),
    }


def _median_by_variant(runs: list[dict]) -> dict:
    """Median duration per variant. Incremental and rebuild runs differ by an
    order of magnitude, so a single pooled median would track the mix rather than
    any real drift in either."""
    buckets: dict[str, list] = {}
    for run in runs:
        duration = run.get("durationSeconds")
        # Only runs that actually completed carry a meaningful duration; a
        # failed, incomplete or in-flight run would drag the median toward
        # whatever fraction of the job happened to be recorded.
        if duration is None or run.get("status") not in ("succeeded", "degraded"):
            continue
        buckets.setdefault(run.get("variant") or "incremental", []).append(duration)
    return {variant: int(median(values)) for variant, values in buckets.items() if values}


# Ceiling on the unauthenticated POST /api/indexing/runs body. MAX_RECORD_BYTES
# (64 KiB) is what a record is truncated TO at write time, but a body may arrive
# larger — pre-truncation phase detail payloads — so allow headroom over it while
# still bounding the read: json.loads(await request.body()) would otherwise buffer
# an arbitrarily large body into memory, an OOM vector on an open endpoint.
MAX_REQUEST_BODY_BYTES = 4 * MAX_RECORD_BYTES


@router.post("/api/indexing/runs")
async def append_indexing_run(request: Request):
    """Append a script-reported run record to the ledger.

    The shell helper posts here so the tagging phase huginn cannot observe lands
    in the same run as the reindex it can. Both sides only ever APPEND their own
    partial sharing a ``runId``; folding happens at read time, which is what
    makes arrival order irrelevant — for an API-triggered reindex huginn's record
    lands first, but on the 409 and API-down paths the script's does.

    Deliberately not gated on ``store.has_collection``: the CLI-fallback and
    rebuild paths report runs for collections this process may not serve, and
    dropping those is exactly the blind spot the ledger exists to remove.
    """
    # Content-Length catches the common (buffered) case cheaply — the shell client
    # posts via `curl --data-binary @-`, which sets it — but a chunked body carries
    # none, so the streamed read below is the real guard.
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > MAX_REQUEST_BODY_BYTES:
                raise HTTPException(status_code=413, detail="Request body too large")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length")

    body = b""
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_REQUEST_BODY_BYTES:
            raise HTTPException(status_code=413, detail="Request body too large")

    try:
        record = json.loads(body)
    except ValueError:
        raise HTTPException(status_code=400, detail="Body must be JSON")
    if not isinstance(record, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")

    try:
        written = IndexingRunLedger().append(record)
    except InvalidCollectionName as e:
        raise HTTPException(status_code=400, detail=str(e))
    except OSError as e:
        logger.warning("Could not append indexing run record", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not write ledger: {e}")

    return {"status": "recorded", "runId": written["runId"],
            "collection": written["collection"]}


# Floor for the cadence-derived incomplete threshold. Comfortably above the
# slowest observed job (~76 min) and POLL_TIMEOUT (3600s), so a genuinely
# in-flight run is never mislabelled `incomplete` even on a short cadence.
_INCOMPLETE_FLOOR_SECONDS = 2 * 3600


def _incomplete_after_for_schedule(schedule):
    """Seconds past which an unclosed run of this cadence folds to ``incomplete``.

    ``max(2 × cadence, floor)`` from the launchd schedule, so a dead hourly run
    stops reading as ``running`` for six subsequent runs the way a flat 6h let it.
    Cadence mapping: hourly → 3600; interval → its seconds; a calendar entry →
    daily (86400), or weekly (604800) when it pins a Weekday. An unknown or absent
    schedule keeps the flat ``INCOMPLETE_AFTER_SECONDS`` — the ledger's own default.
    """
    if not isinstance(schedule, dict):
        return INCOMPLETE_AFTER_SECONDS
    kind = schedule.get("kind")
    if kind == "hourly":
        cadence = 3600
    elif kind == "interval" and isinstance(schedule.get("seconds"), int):
        cadence = schedule["seconds"]
    elif kind == "calendar":
        cadence = 604800 if schedule.get("weekday") is not None else 86400
    else:
        return INCOMPLETE_AFTER_SECONDS
    return max(2 * cadence, _INCOMPLETE_FLOOR_SECONDS)


@router.get("/api/indexing/jobs")
def indexing_jobs(
    history: int = Query(20, ge=0, le=500, description="History entries per collection"),
    store: KnowledgeStore = Depends(get_store),
):
    """Per-collection indexing run overview: live status, last run, history, schedule.

    Rows are the UNION of collections with a ledger file and collections this
    server currently serves. Iterating only loaded collections would hide every
    collection this process does not happen to serve (the whole Jira / Confluence
    / Notion backfill); iterating only ledger files would advertise collections
    huginn cannot answer searches for. Rows the server does not serve are marked
    ``loaded: false`` instead of being dropped.

    Response contract (what the muninn dashboard couples to):
    - ``lastRun`` is the fixed ``LAST_RUN_FIELDS`` projection, never the raw
      folded record; ``history`` entries are the smaller 5-field projection.
    - ``current`` is the ONE running channel, merging the in-memory reindex
      state and ledger-side script runs (``source``: reindex/script/both).
    - ``nextRunAt`` is UTC; the raw ``schedule`` dict keeps launchd's
      machine-local wall-clock fields and is tagged ``timezone: "local"``.
    - ``medianDurationSeconds`` is computed over a fixed window
      (``MEDIAN_WINDOW_RUNS``), independent of the ``history`` param.
    """
    ledger = IndexingRunLedger()
    try:
        ledger_collections = set(ledger.collections())
    except OSError:
        ledger_collections = set()
    loaded = set(store.collection_names())
    try:
        schedules = load_schedules()
    except Exception:
        # A missing/unreadable LaunchAgents dir costs the "schedule" field, not
        # the endpoint. The run history is the part that matters here.
        logger.warning("Could not read launchd schedules", exc_info=True)
        schedules = {}

    jobs = []
    for name in sorted(ledger_collections | loaded):
        schedule_entry = schedules.get(name) or {}
        incomplete_after = _incomplete_after_for_schedule(schedule_entry.get("schedule"))
        try:
            runs = ledger.recent(name, limit=max(history, MEDIAN_WINDOW_RUNS),
                                 incomplete_after=incomplete_after)
        except Exception:
            logger.warning("Could not read run ledger for %s", name, exc_info=True)
            runs = []

        state = store.get_update_status(name) if name in loaded else None
        last_run = runs[-1] if runs else None
        schedule = schedule_entry.get("schedule")

        jobs.append({
            "collection": name,
            "loaded": name in loaded,
            "job": schedule_entry.get("job"),
            # Copy, both to tag it and because load_schedules returns its shared
            # cached dict — mutating that would corrupt the cache for every
            # later caller.
            "schedule": {**schedule, "timezone": "local"} if schedule else None,
            "nextRunAt": _next_run_at(schedule, last_run),
            "current": _current_running(state, last_run),
            "lastRun": _project_run(last_run),
            "history": [
                {
                    "runId": run.get("runId"),
                    "startedAt": run.get("startedAt"),
                    "durationSeconds": run.get("durationSeconds"),
                    "status": run.get("status"),
                    "variant": run.get("variant"),
                }
                for run in runs[-history:]
            ] if history else [],
            "medianDurationSeconds": _median_by_variant(runs[-MEDIAN_WINDOW_RUNS:]),
        })
    return {"jobs": jobs}
