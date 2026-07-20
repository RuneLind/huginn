"""Collection-level routes — listing, tags, document lookup, manual update."""
import json
import logging
import os

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
from main.runtime.knowledge_store import KnowledgeStore, get_store, run_collection_update

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


def _read_doc_dates(store: KnowledgeStore, doc_path: str) -> tuple[str | None, str | None]:
    """Read a single document JSON and return ``(date, modifiedTime)``, or Nones on error.

    A missing/unreadable file or malformed JSON yields Nones (logged), so one bad
    document doesn't fail the whole listing — but genuinely unexpected errors are
    left to propagate rather than silently swallowed.
    """
    if not doc_path:
        return None, None
    try:
        doc = json.loads(store.disk_persister.read_text_file(doc_path))
    except (OSError, ValueError) as e:
        logger.warning("Could not read date for document %s: %s", doc_path, e)
        return None, None
    return _resolve_doc_date(doc), doc.get("modifiedTime")


@router.get("/api/collection/{name}/documents")
def list_collection_documents(
    name: str,
    include_dates: bool = Query(
        False,
        description="Attach each document's added date. Slower — reads every document file.",
    ),
    store: KnowledgeStore = Depends(get_store),
):
    """List all documents in a collection with their IDs and URLs.

    When ``include_dates`` is set, each entry also carries a ``date`` field
    (frontmatter date, falling back to file mtime) so callers can sort/group by
    recency, plus a ``modifiedTime`` field (full-precision ingest timestamp,
    when the document has one) so callers can break intra-day ties. This reads
    every document file, so it is opt-in to keep the default listing (used by
    hot paths like duplicate checks) cheap.
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
        if include_dates:
            date, modified_time = _read_doc_dates(store, entry.get("documentPath", ""))
            doc["date"] = date
            if modified_time:
                doc["modifiedTime"] = modified_time
        documents.append(doc)

    return {"documents": documents}


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
    resolved = os.path.realpath(os.path.join(base_dir, doc_path))
    if not resolved.startswith(base_dir + os.sep):
        raise HTTPException(status_code=400, detail="Invalid document ID")

    try:
        doc_text = store.disk_persister.read_text_file(doc_path)
        return json.loads(doc_text)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")


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
    makes arrival order irrelevant — for mimir huginn's record lands first, but on
    the 409 and API-down paths the script's does.

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
# slowest observed job (mimir, ~76 min) and POLL_TIMEOUT (3600s), so a genuinely
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
