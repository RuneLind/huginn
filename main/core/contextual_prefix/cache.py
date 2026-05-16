import hashlib
import json
import os
import threading
from datetime import datetime, timezone


CACHE_VERSION = 1


def chunk_fingerprint(doc_id: str, chunk_text: str) -> str:
    h = hashlib.sha256()
    h.update(doc_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(chunk_text.encode("utf-8"))
    return h.hexdigest()


class ContextualCache:
    """Disk-backed cache of (doc_id, chunk_hash, model_id) -> prefix.

    Survives re-indexes. New/changed chunks miss; unchanged chunks hit.
    A model change misses (because model_id is in the key) — old prefixes
    stay around for traceability but don't pollute the augmented text.
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._entries: dict = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            self._entries = {}
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version") != CACHE_VERSION:
                self._entries = {}
                return
            self._entries = data.get("entries", {})
        except (OSError, json.JSONDecodeError):
            self._entries = {}

    def _key(self, doc_id: str, chunk_hash: str, model_id: str) -> str:
        return f"{doc_id}::{chunk_hash}::{model_id}"

    def get(self, doc_id: str, chunk_text: str, model_id: str) -> str | None:
        chunk_hash = chunk_fingerprint(doc_id, chunk_text)
        with self._lock:
            entry = self._entries.get(self._key(doc_id, chunk_hash, model_id))
        return entry["prefix"] if entry else None

    def put(self, doc_id: str, chunk_text: str, model_id: str, prefix: str, doc_path: str | None = None) -> None:
        chunk_hash = chunk_fingerprint(doc_id, chunk_text)
        with self._lock:
            self._entries[self._key(doc_id, chunk_hash, model_id)] = {
                "prefix": prefix,
                "created": datetime.now(timezone.utc).isoformat(),
                "docPath": doc_path,
            }
            self._dirty = True

    def flush(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"version": CACHE_VERSION, "entries": self._entries}, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
            self._dirty = False

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
