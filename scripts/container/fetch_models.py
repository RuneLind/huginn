#!/usr/bin/env python3
"""Pinned model files for the serve image.

``fetch LOCK`` runs inside the image build: it downloads exactly the files the
lock lists, at the lock's revision, into ``HF_HOME``. A download by commit sha
writes no ``refs/main``, and huginn loads both models by name without a
revision (``sentence_embeder.py``, ``cross_encoder_reranker.py``), so under
``HF_HUB_OFFLINE=1`` the load would fail. This step therefore writes
``refs/main`` containing the pinned sha, and removes ``.locks/``.

``lock LOCK`` runs on the build machine: it fills each listed file's git blob
id and LFS sha256 from the Hub, which ``image_check.py`` verifies the image's
``hf-cache`` against.
"""
import json
import os
import shutil
import sys


def fetch(lock_path):
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE

    with open(lock_path) as fh:
        lock = json.load(fh)
    for model in lock["models"]:
        snapshot_download(
            model["repo"],
            revision=model["revision"],
            allow_patterns=[f["path"] for f in model["files"]],
        )
        refs = os.path.join(HF_HUB_CACHE, "models--" + model["repo"].replace("/", "--"), "refs")
        os.makedirs(refs, exist_ok=True)
        with open(os.path.join(refs, "main"), "w") as fh:
            fh.write(model["revision"])
    shutil.rmtree(os.path.join(HF_HUB_CACHE, ".locks"), ignore_errors=True)


def refresh_lock(lock_path):
    from huggingface_hub import HfApi

    api = HfApi()
    with open(lock_path) as fh:
        lock = json.load(fh)
    for model in lock["models"]:
        wanted = {f["path"] for f in model["files"]}
        hub = {
            f.path: f
            for f in api.list_repo_tree(model["repo"], revision=model["revision"], recursive=True)
            if hasattr(f, "blob_id")
        }
        missing = wanted - hub.keys()
        if missing:
            raise SystemExit(f"{model['repo']}@{model['revision']} has no {sorted(missing)}")
        model["files"] = [
            {"path": p, "blobId": hub[p].blob_id, "lfsSha256": hub[p].lfs.sha256 if hub[p].lfs else None}
            for p in sorted(wanted)
        ]
    with open(lock_path, "w") as fh:
        json.dump(lock, fh, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("fetch", "lock"):
        raise SystemExit("usage: fetch_models.py fetch|lock LOCK")
    (fetch if sys.argv[1] == "fetch" else refresh_lock)(sys.argv[2])
