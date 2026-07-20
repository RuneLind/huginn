import argparse
import json
import logging
import os

from main.utils.logger import setup_root_logger
from main.factories.update_collection_factory import create_collection_updater
from main.runtime.indexing_run_ledger import IndexingRunLedger, mint_run_id, now_iso

setup_root_logger()

ap = argparse.ArgumentParser()
ap.add_argument("-collection", "--collection", required=True, help="Collection name (will be used to determine root folder and manifest file)")
ap.add_argument("--contextual-model", required=False, default=None,
                help="Override contextual-prefix backend spec (e.g. 'ollama:qwen3.6:35b-a3b-nvfp4', 'claude-code:claude-haiku-4-5', 'anthropic:claude-haiku-4-5', 'none'). "
                     "If omitted, uses manifest['contextualPrefix']['model'] — letting scheduled updates inherit the create-time choice.")
ap.add_argument("--contextual-cache", required=False, default=None,
                help="Path to the contextual-prefix cache JSON (defaults to data/contextual_caches/<name>.json — outside the collection folder so it survives re-creates).")
ap.add_argument("--contextual-workers", required=False, type=int, default=1,
                help="How many documents to prefix concurrently (default 1 = sequential).")
ap.add_argument("--run-id", required=False, default=None,
                help="Correlation id shared with a wrapping script's own ledger record, so both sides fold into one run. Empty or omitted mints a new one.")
ap.add_argument("--job", required=False, default=None,
                help="launchd job label to record on the ledger entry (e.g. com.huginn.mimir-index).")
ap.add_argument("--trigger", required=False, default="cli",
                choices=["scheduled", "manual", "cli", "unknown"],
                help="How this run was triggered (default: cli).")
args = vars(ap.parse_args())

collection = args["collection"]
started_at = now_iso()
run_id = args["run_id"] or mint_run_id(collection, started_at)


def _manifest_counts():
    """Counts from the manifest, empty when it is absent. A failed run never
    rewrote it, and creation removes the folder outright on a zero-document read,
    so a missing manifest is normal rather than exceptional."""
    try:
        with open(os.path.join("./data/collections", collection, "manifest.json"),
                  encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _record(status, error, duration_source_start):
    """Emit the ledger entry for this run.

    This adapter bypasses the HTTP server entirely, so nothing else writes a
    record for it — and it is the fallback path used when the API is unhealthy,
    which makes it the likeliest place for a failure to happen unobserved. Hence
    the try/finally around .run(): appending only on the success path would give a
    ledger where failures simply do not appear.
    """
    finished_at = now_iso()
    manifest = _manifest_counts() if status != "failed" else {}
    try:
        IndexingRunLedger().append({
            "runId": run_id,
            "collection": collection,
            "job": args["job"],
            "trigger": args["trigger"],
            "variant": "incremental",
            "startedAt": duration_source_start,
            "finishedAt": finished_at,
            "status": status,
            "phases": [{
                "name": "reindex",
                "status": status,
                # Stamp the phase start so the fold orders by time, not arrival.
                # This is the API-down fallback the x-feed script hits, so omitting
                # it leaves `reindex` mis-sorted before `fetch` on exactly that path.
                "startedAt": duration_source_start,
                "fatal": True,
            }],
            "documentCount": manifest.get("numberOfDocuments"),
            "chunkCount": manifest.get("numberOfChunks"),
            "error": error,
            "source": "huginn",
        })
    except Exception:
        logging.warning("Could not write indexing run ledger record for %s",
                        collection, exc_info=True)


updater = create_collection_updater(collection,
                                    contextual_backend_spec=args['contextual_model'],
                                    contextual_cache_path=args['contextual_cache'],
                                    contextual_workers=args['contextual_workers'])

try:
    updater.run()
except BaseException as e:
    _record("failed", str(e), started_at)
    raise
else:
    _record("succeeded", None, started_at)
