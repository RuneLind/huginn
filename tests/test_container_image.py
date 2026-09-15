"""The serve image's staging verification and image check, on synthetic files only."""
import hashlib
import io
import json
import tarfile

import pytest

from scripts.container import image_build
from scripts.container.image_check import check_layers, name_refusal
from scripts.container.provenance import (
    Refused,
    extract_package,
    git_blob_id,
    read_package,
    stamp_refusals,
    verify_staging,
)

PASSING_CHECKS = {"1": {"passed": True, "ran": True}, "5": {"passed": False, "ran": False}}


def _stamp(**overrides):
    stamp = {"collection": "demo", "numberOfDocuments": 1,
             "sensitivitySweep": {"status": "pass"}, "scanChecks": PASSING_CHECKS}
    stamp.update(overrides)
    return stamp


def _add(tar, name, data=None, *, symlink=None, directory=False, mode=None):
    info = tarfile.TarInfo(name)
    if mode is not None:
        info.mode = mode
    if directory:
        info.type = tarfile.DIRTYPE
        tar.addfile(info)
    elif symlink is not None:
        info.type = tarfile.SYMTYPE
        info.linkname = symlink
        tar.addfile(info)
    else:
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))


def _package(tmp_path, *, stamp=None, extra=(), manifest_docs=1, name="demo.tar.gz",
             document=b'{"text": "synthetic"}'):
    path = tmp_path / name
    with tarfile.open(path, "w:gz") as tar:
        _add(tar, "PACKAGE-STAMP.json", json.dumps(stamp or _stamp()).encode())
        _add(tar, "data/collections/demo/manifest.json",
             json.dumps({"numberOfDocuments": manifest_docs}).encode())
        _add(tar, "data/collections/demo/documents/doc.json", document)
        for entry in extra:
            _add(tar, *entry[:2], **(entry[2] if len(entry) > 2 else {}))
    return path


class TestStampRule:
    def test_pass_ignores_checks_that_did_not_run(self):
        assert stamp_refusals(_stamp()) == []

    def test_sweep_warn_is_refused(self):
        assert stamp_refusals(_stamp(sensitivitySweep={"status": "warn"}))

    def test_a_check_that_ran_and_failed_is_refused(self):
        checks = {**PASSING_CHECKS, "9": {"passed": False, "ran": True}}
        assert stamp_refusals(_stamp(scanChecks=checks)) == ["check 9 ran and did not pass"]

    def test_missing_checks_are_refused(self):
        assert stamp_refusals(_stamp(scanChecks={}))


class TestReadPackage:
    def test_members_carry_the_stamp_at_its_relocated_path(self, tmp_path):
        package = read_package(_package(tmp_path))
        assert package.collection == "demo"
        assert set(package.members) == {
            "data/collections/demo/manifest.json",
            "data/collections/demo/documents/doc.json",
            "data/collections/demo/PACKAGE-STAMP.json",
        }

    def test_member_outside_the_collection_prefix_is_refused(self, tmp_path):
        with pytest.raises(Refused, match="outside"):
            read_package(_package(tmp_path, extra=[("data/sources/raw.md", b"x")]))

    def test_symlink_member_is_refused(self, tmp_path):
        with pytest.raises(Refused, match="not a regular file"):
            read_package(_package(tmp_path, extra=[("data/collections/demo/l", None, {"symlink": "/etc"})]))

    def test_failing_stamp_is_refused(self, tmp_path):
        with pytest.raises(Refused, match="stamp refused"):
            read_package(_package(tmp_path, stamp=_stamp(sensitivitySweep={"status": "warn"})))

    def test_manifest_document_count_mismatch_is_refused(self, tmp_path):
        with pytest.raises(Refused, match="numberOfDocuments"):
            read_package(_package(tmp_path, manifest_docs=2))

    def test_refusal_never_names_a_document(self, tmp_path):
        with pytest.raises(Refused) as exc:
            read_package(_package(tmp_path, extra=[("data/collections/demo/documents/x", None,
                                                    {"symlink": "y"})]))
        assert "documents/<document>" in str(exc.value) and "/x" not in str(exc.value)


class TestVerifyStaging:
    def _staged(self, tmp_path):
        package = read_package(_package(tmp_path))
        staging = tmp_path / "staging"
        (staging / "src" / "main").mkdir(parents=True)
        (staging / "src" / "main" / "app.py").write_bytes(b"print(1)\n")
        extract_package(package, staging / "collections")
        tree = {"main/app.py": ("100644", git_blob_id(b"print(1)\n"))}
        return staging, tree, package

    def test_clean_staging_passes(self, tmp_path):
        staging, tree, package = self._staged(tmp_path)
        assert (staging / "collections/data/collections/demo/PACKAGE-STAMP.json").is_file()
        assert verify_staging(staging, tree, [package]) == []

    def test_extra_file_is_refused(self, tmp_path):
        staging, tree, package = self._staged(tmp_path)
        (staging / "src" / "main" / "extra.py").write_text("x")
        assert verify_staging(staging, tree, [package]) == ["src/main/extra.py: not in the pinned commit"]

    def test_changed_tracked_file_is_refused(self, tmp_path):
        staging, tree, package = self._staged(tmp_path)
        (staging / "src" / "main" / "app.py").write_bytes(b"print(2)\n")
        assert verify_staging(staging, tree, [package]) == ["src/main/app.py: content differs from the commit"]

    def test_changed_collection_file_is_refused_without_its_name(self, tmp_path):
        staging, tree, package = self._staged(tmp_path)
        (staging / "collections/data/collections/demo/documents/doc.json").write_text("{}")
        assert verify_staging(staging, tree, [package]) == [
            "collections/data/collections/demo/documents/<document>: content differs from the package member"]

    def test_file_outside_src_and_collections_is_refused(self, tmp_path):
        staging, tree, package = self._staged(tmp_path)
        (staging / "aliases.json").write_text("{}")
        assert verify_staging(staging, tree, [package]) == ["aliases.json: neither src/ nor collections/"]


# --- the image check over a synthetic `docker save` ---------------------------

APP_SOURCE = b"print('app')\n"
CONFIG = b"{}"
WEIGHTS = b"W" * 10
REVISION = "a" * 40
CONFIG_ID = git_blob_id(CONFIG)
WEIGHTS_SHA = hashlib.sha256(WEIGHTS).hexdigest()
MODEL = "app/hf-cache/hub/models--org--model"
LOCK = {"models": [{"repo": "org/model", "revision": REVISION, "files": [
    {"path": "config.json", "blobId": CONFIG_ID, "lfsSha256": None},
    {"path": "w.bin", "blobId": "0" * 40, "lfsSha256": WEIGHTS_SHA},
]}]}
TREE = {"main/app.py": ("100644", git_blob_id(APP_SOURCE))}


def _good_layers(package_path):
    stamp = None
    with tarfile.open(package_path) as tar:
        collection = [(m.name, tar.extractfile(m).read()) for m in tar if m.isfile()]
    layers = [
        [("usr/local/lib/python3.12/site-packages/lib.py", b"x = 1\n")],
        [(f"{MODEL}/blobs", None, {"directory": True}),
         (f"{MODEL}/blobs/{CONFIG_ID}", CONFIG),
         (f"{MODEL}/blobs/{WEIGHTS_SHA}", WEIGHTS),
         (f"{MODEL}/snapshots/{REVISION}/config.json", None, {"symlink": f"../../blobs/{CONFIG_ID}"}),
         (f"{MODEL}/snapshots/{REVISION}/w.bin", None, {"symlink": f"../../blobs/{WEIGHTS_SHA}"}),
         (f"{MODEL}/refs/main", REVISION.encode())],
        [("app/main/app.py", APP_SOURCE)],
    ]
    collections_layer = []
    for name, data in collection:
        if name == "PACKAGE-STAMP.json":
            stamp = data
        else:
            collections_layer.append((f"app/{name}", data))
    collections_layer.append(("app/data/collections/demo/PACKAGE-STAMP.json", stamp))
    layers.append(collections_layer)
    return layers


def _save(tmp_path, layers):
    path = tmp_path / "image.tar"
    manifest = [{"Config": "blobs/sha256/config", "Layers": []}]
    with tarfile.open(path, "w") as outer:
        for index, entries in enumerate(layers):
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w") as layer:
                for entry in entries:
                    _add(layer, *entry[:2], **(entry[2] if len(entry) > 2 else {}))
            name = f"blobs/sha256/layer{index}"
            manifest[0]["Layers"].append(name)
            _add(outer, name, buffer.getvalue())
        _add(outer, "manifest.json", json.dumps(manifest).encode())
    return path


def _check(tmp_path, mutate=None):
    package_path = _package(tmp_path)
    layers = _good_layers(package_path)
    if mutate:
        mutate(layers)
    report, _ = check_layers(_save(tmp_path, layers), TREE, [read_package(package_path)], LOCK)
    return report.refusals


class TestImageCheck:
    def test_clean_image_passes(self, tmp_path):
        assert _check(tmp_path) == []

    def test_whiteout_is_refused(self, tmp_path):
        refusals = _check(tmp_path, lambda l: l.append([("tmp/.wh.scratch", b"")]))
        assert [c for c, _ in refusals] == ["whiteout"]

    def test_alias_map_name_in_site_packages_is_refused(self, tmp_path):
        refusals = _check(tmp_path, lambda l: l[0].append(
            ("usr/local/lib/python3.12/site-packages/aliases.json", b"{}")))
        assert refusals == [("name", "layer 0: usr/local/lib/python3.12/site-packages/aliases.json "
                                     "(an alias map file name)")]

    def test_untracked_app_file_is_refused(self, tmp_path):
        refusals = _check(tmp_path, lambda l: l[2].append(("app/main/extra.py", b"x")))
        assert refusals == [("provenance", "layer 2: app/main/extra.py (not in the pinned commit)")]

    def test_graph_file_under_app_fails_name_and_provenance(self, tmp_path):
        refusals = _check(tmp_path, lambda l: l[2].append(
            ("app/scripts/knowledge_graph/demo_llm_graph.json", b"{}")))
        assert sorted(c for c, _ in refusals) == ["name", "provenance"]

    def test_extra_file_in_hf_cache_is_refused(self, tmp_path):
        refusals = _check(tmp_path, lambda l: l[1].append((f"{MODEL}/snapshots/{REVISION}/notes.txt", b"x")))
        assert [c for c, _ in refusals] == ["provenance"]

    def test_refs_main_naming_another_revision_is_refused(self, tmp_path):
        def mutate(layers):
            layers[1][-1] = (f"{MODEL}/refs/main", b"b" * 40)
        assert _check(tmp_path, mutate) == [
            ("provenance", f"layer 1: {MODEL}/refs/main (refs/main names another revision)")]

    def test_blob_with_wrong_content_is_refused(self, tmp_path):
        def mutate(layers):
            layers[1][2] = (f"{MODEL}/blobs/{WEIGHTS_SHA}", b"X" * 10)
        assert [c for c, _ in _check(tmp_path, mutate)] == ["provenance"]

    def test_snapshot_symlink_to_another_blob_is_refused(self, tmp_path):
        def mutate(layers):
            layers[1][3] = (f"{MODEL}/snapshots/{REVISION}/config.json", None,
                            {"symlink": f"../../blobs/{WEIGHTS_SHA}"})
        assert [c for c, _ in _check(tmp_path, mutate)] == ["provenance"]

    def test_changed_tracked_app_file_is_refused_even_if_a_later_layer_restores_it(self, tmp_path):
        def mutate(layers):
            layers[2] = [("app/main/app.py", b"print('changed')\n")]
            layers.append([("app/main/app.py", APP_SOURCE)])
        assert _check(tmp_path, mutate) == [
            ("provenance", "layer 2: app/main/app.py (content differs from the commit)")]

    def test_changed_collection_file_is_refused_without_its_name(self, tmp_path):
        def mutate(layers):
            layers[3] = [(n, b"{}" if n.endswith("doc.json") else d) for n, d in layers[3]]
        assert _check(tmp_path, mutate) == [("provenance", "layer 3: app/data/collections/demo/documents/"
                                                           "<document> (differs from the package member)")]

    def test_second_stamp_is_refused(self, tmp_path):
        def mutate(layers):
            layers.append([layers[3][-1]])
        assert ("name", "demo: 2 stamps, expected exactly 1") in _check(tmp_path, mutate)

    def test_missing_collection_is_refused(self, tmp_path):
        def mutate(layers):
            layers.pop(3)
        assert sorted(c for c, _ in _check(tmp_path, mutate)) == ["name", "provenance"]


def test_large_collection_file_is_hashed_and_extracted_in_one_read(tmp_path, monkeypatch):
    import scripts.container.image_check as image_check

    # Above the limit: the document and the stamp. Below it: manifest and refs/main.
    monkeypatch.setattr(image_check, "_TEXT_REPORT_MAX_BYTES", 60)
    document = b'{"text": "' + b"s" * 88 + b'"}'
    package_path = _package(tmp_path, document=document)
    layers = _good_layers(package_path)
    report, _ = check_layers(_save(tmp_path, layers), TREE, [read_package(package_path)], LOCK,
                             extract_to=tmp_path / "out")
    assert report.refusals == []
    collection = tmp_path / "out/data/collections/demo"
    assert (collection / "documents/doc.json").read_bytes() == document
    assert json.loads((collection / "PACKAGE-STAMP.json").read_bytes())["collection"] == "demo"


class TestNameRule:
    @pytest.mark.parametrize("key, is_dir", [
        ("app/huginn-private/x.txt", False),
        ("opt/huginn-private", True),
        ("app/data/sources/a.md", False),
        ("app/data/prealias", True),
        ("root/data.zip", False),
        ("root/backup.tar.xz", False),
        ("usr/local/lib/python3.12/site-packages/nvidia_cublas_cu12-12.1.dist-info/METADATA", False),
    ])
    def test_refused(self, key, is_dir):
        assert name_refusal(key, is_dir)

    @pytest.mark.parametrize("key", ["app/main/graph.py", "usr/lib/huginn.txt", "app/data/collections/x/manifest.json"])
    def test_allowed(self, key):
        assert name_refusal(key, False) is None


class TestPytorchIndex:
    STEP = "#7 [3/6] RUN --mount=from=ghcr.io/astral-sh/uv:0.8.14@sha256:abc,source=/uv,target=/tmp/uv"

    def test_torch_only(self):
        log = f"{self.STEP}\n#7 1.2 Downloading https://download.pytorch.org/whl/cpu/torch-2.10.0%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl\n"
        assert image_build.pytorch_index_packages(log)[0] == {"torch"}

    def test_wheel_host_is_download_r2(self):
        log = (f"{self.STEP}\n#10 0.131 DEBUG No cache entry for: https://download.pytorch.org/whl/cpu/torch/\n"
               "#10 1.421 DEBUG No cache entry for: https://download-r2.pytorch.org/whl/cpu/"
               "torch-2.10.0%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl\n")
        assert image_build.pytorch_index_packages(log)[0] == {"torch"}

    def test_other_package_is_reported(self):
        log = (f"{self.STEP}\n#7 1.2 https://download.pytorch.org/whl/cpu/torch-2.10.0-cp312.whl\n"
               "#7 1.3 https://download.pytorch.org/whl/numpy-2.5.3-cp312.whl\n")
        assert image_build.pytorch_index_packages(log)[0] == {"torch", "numpy"}

    def test_cached_step_cannot_answer(self):
        assert image_build.pytorch_index_packages(f"{self.STEP}\n#7 CACHED\n")[0] is None


# --- fix round 1: gaps the review found ---------------------------------------


def _save_oci(tmp_path, layers):
    """The OCI layout `docker save` writes when manifest.json is absent."""
    path = tmp_path / "image-oci.tar"
    digests = []
    with tarfile.open(path, "w") as outer:
        for index, entries in enumerate(layers):
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w") as layer:
                for entry in entries:
                    _add(layer, *entry[:2], **(entry[2] if len(entry) > 2 else {}))
            digest = hashlib.sha256(buffer.getvalue()).hexdigest()
            digests.append(digest)
            _add(outer, f"blobs/sha256/{digest}", buffer.getvalue())
        manifest = json.dumps({"layers": [{"digest": f"sha256:{d}"} for d in digests]}).encode()
        manifest_digest = hashlib.sha256(manifest).hexdigest()
        _add(outer, f"blobs/sha256/{manifest_digest}", manifest)
        _add(outer, "index.json", json.dumps({"manifests": [{"digest": f"sha256:{manifest_digest}"}]}).encode())
    return path


def _hardlink(tar, name, target):
    info = tarfile.TarInfo(name)
    info.type = tarfile.LNKTYPE
    info.linkname = target
    tar.addfile(info)


class TestFixRound1Gaps:
    """Each of these passed a bad image before fix round 1."""

    def test_dotdot_entry_landing_in_app_is_refused(self, tmp_path):
        refusals = _check(tmp_path, lambda l: l[2].append(("usr/../app/main/evil.py", b"x")))
        assert refusals and refusals[0][0] == "name"

    def test_dotdot_entry_is_never_written_outside_the_extract_folder(self, tmp_path):
        package_path = _package(tmp_path)
        layers = _good_layers(package_path)
        layers[3].append(("app/data/collections/demo/../../../../escaped.txt", b"x"))
        out = tmp_path / "a" / "b" / "extracted"
        report, _ = check_layers(_save(tmp_path, layers), TREE, [read_package(package_path)], LOCK,
                                 extract_to=out)
        assert report.refusals
        assert not any(p.name == "escaped.txt" for p in tmp_path.rglob("*"))

    def test_folder_under_app_nothing_implies_is_refused(self, tmp_path):
        refusals = _check(tmp_path, lambda l: l[2].append(("app/Synthetic Probe Dir", None, {"directory": True})))
        assert [c for c, _ in refusals] == ["provenance"]

    def test_empty_collection_folder_is_refused(self, tmp_path):
        refusals = _check(tmp_path, lambda l: l[3].append(
            ("app/data/collections/probe-dir", None, {"directory": True})))
        assert refusals

    def test_world_writable_tracked_file_is_refused(self, tmp_path):
        def mutate(layers):
            layers[2] = [("app/main/app.py", APP_SOURCE, {"mode": 0o777})]
        assert [c for c, _ in _check(tmp_path, mutate)] == ["provenance"]

    def test_exec_bit_the_commit_does_not_have_is_refused(self, tmp_path):
        def mutate(layers):
            layers[2] = [("app/main/app.py", APP_SOURCE, {"mode": 0o755})]
        assert [c for c, _ in _check(tmp_path, mutate)] == ["provenance"]

    @pytest.mark.parametrize("key", ["usr/share/Aliases.JSON", "root/data.tar.zst", "root/x.7z",
                                     "root/x.RAR", "root/data.tgz", "root/data.tar.lz4", "root/data.tzst",
                                     "opt/HUGINN-private/x", "srv/Data/Sources/a.md", "a/B_GRAPH.json"])
    def test_name_rule_is_case_insensitive_and_knows_more_archives(self, key):
        assert name_refusal(key, False)

    @pytest.mark.parametrize("key", ["etc/alternatives/awk.1.gz", "var/log/apt/eipp.log.xz",
                                     "usr/local/lib/python3.12/site-packages/pkg/test/data/x.pkl.bz2",
                                     "usr/share/doc/tarfile-notes.txt", "usr/lib/foo.tar_helper.py"])
    def test_single_file_compression_is_not_an_archive(self, key):
        assert name_refusal(key, False) is None

    def test_member_missing_from_the_image_is_refused(self, tmp_path):
        def mutate(layers):
            layers[3] = [e for e in layers[3] if not e[0].endswith("doc.json")]
        refusals = _check(tmp_path, mutate)
        assert ("provenance", "demo: 1 package member(s) missing from the image") in refusals

    def test_empty_package_set_is_refused(self, tmp_path):
        layers = _good_layers(_package(tmp_path))[:3]
        report, _ = check_layers(_save(tmp_path, layers), TREE, [], LOCK)
        assert report.refusals

    def test_unreadable_layer_is_a_refusal_not_a_traceback(self, tmp_path):
        path = tmp_path / "image.tar"
        with tarfile.open(path, "w") as outer:
            _add(outer, "blobs/sha256/layer0", b"\x28\xb5\x2f\xfd" + b"not a tar" * 50)
            _add(outer, "manifest.json", json.dumps([{"Layers": ["blobs/sha256/layer0"]}]).encode())
        with pytest.raises(Refused, match="layer 0"):
            check_layers(path, TREE, [read_package(_package(tmp_path))], LOCK)


class TestFixRound1StampRule:
    def test_no_check_that_ran_is_refused(self):
        assert stamp_refusals(_stamp(scanChecks={"5": {"passed": False, "ran": False}}))

    def test_non_dict_check_entry_is_refused_not_raised(self):
        assert stamp_refusals(_stamp(scanChecks={"1": True}))

    def test_sweep_that_is_not_an_object_is_refused_not_raised(self):
        assert stamp_refusals(_stamp(sensitivitySweep="pass"))

    def test_document_count_missing_on_both_sides_is_refused(self, tmp_path):
        stamp = _stamp()
        del stamp["numberOfDocuments"]
        path = tmp_path / "nocount.tar.gz"
        with tarfile.open(path, "w:gz") as tar:
            _add(tar, "PACKAGE-STAMP.json", json.dumps(stamp).encode())
            _add(tar, "data/collections/demo/manifest.json", b"{}")
        with pytest.raises(Refused, match="numberOfDocuments"):
            read_package(path)


class TestFixRound1BuildInputs:
    def test_duplicate_collection_is_refused(self, tmp_path):
        a = read_package(_package(tmp_path, name="a.tar.gz"))
        b = read_package(_package(tmp_path, name="b.tar.gz"))
        assert image_build.input_refusals([a, b])

    def test_no_package_is_refused(self):
        assert image_build.input_refusals([])

    def test_pytorch_gate(self):
        assert image_build.pytorch_refusal({"torch"}) is None
        assert image_build.pytorch_refusal(set())
        assert image_build.pytorch_refusal({"torch", "numpy"})
        assert image_build.pytorch_refusal(None) is None

    def test_staging_oserror_prints_no_path(self, tmp_path, monkeypatch, capsys):
        package_path = _package(tmp_path)

        def boom(package, root):
            raise OSError(28, "No space left on device", "/x/documents/secret-document-id.json")

        monkeypatch.setattr(image_build, "extract_package", boom)
        status = image_build.main(["--commit", "HEAD", "--package", str(package_path),
                                   "--platform", "linux/arm64", "--tag", "t:t"])
        out = capsys.readouterr().out
        assert status == 2 and "secret-document-id" not in out and "REFUSED" in out


class TestFixRound1Pinning:
    """Branches that existed but no test could fail on (review mutants M1-M10, M16)."""

    def test_hardlink_under_app_is_refused(self, tmp_path):
        package_path = _package(tmp_path)
        layers = _good_layers(package_path)
        path = _save(tmp_path, layers)
        # append a hardlink layer by hand
        with tarfile.open(path) as outer:
            members = {m.name: outer.extractfile(m).read() for m in outer if m.isfile()}
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as layer:
            _hardlink(layer, "app/main/app.py", "app/main/app.py")
        manifest = json.loads(members.pop("manifest.json"))
        manifest[0]["Layers"].append("blobs/sha256/hardlink")
        members["blobs/sha256/hardlink"] = buffer.getvalue()
        members["manifest.json"] = json.dumps(manifest).encode()
        with tarfile.open(path, "w") as outer:
            for name, data in members.items():
                _add(outer, name, data)
        report, _ = check_layers(path, TREE, [read_package(package_path)], LOCK)
        assert ("provenance", "layer 4: app/main/app.py (not a regular file or symlink)") in report.refusals

    def test_refs_main_with_trailing_newline_is_refused(self, tmp_path):
        def mutate(layers):
            layers[1][-1] = (f"{MODEL}/refs/main", REVISION.encode() + b"\n")
        assert [c for c, _ in _check(tmp_path, mutate)] == ["provenance"]

    def test_check_without_ran_key_counts_as_ran(self):
        assert stamp_refusals(_stamp(scanChecks={"1": {"passed": True, "ran": True}, "2": {"passed": False}}))

    def test_foreign_folder_in_hf_cache_is_refused(self, tmp_path):
        refusals = _check(tmp_path, lambda l: l[1].append((f"{MODEL}/.no_exist", None, {"directory": True})))
        assert [c for c, _ in refusals] == ["provenance"]

    def test_duplicate_package_member_is_refused(self, tmp_path):
        with pytest.raises(Refused, match="duplicate"):
            read_package(_package(tmp_path, extra=[("data/collections/demo/manifest.json", b"{}")]))

    @pytest.mark.parametrize("name", ["../escape.json", "/etc/abs.json"])
    def test_escaping_package_member_is_refused(self, tmp_path, name):
        with pytest.raises(Refused, match="escapes"):
            read_package(_package(tmp_path, extra=[(name, b"x")]))

    def test_directory_symlink_in_staging_is_refused(self, tmp_path):
        staging, tree, package = TestVerifyStaging()._staged(tmp_path)
        (staging / "src" / "linked").symlink_to(staging / "src" / "main")
        assert verify_staging(staging, tree, [package]) == ["src/linked: not in the pinned commit"]

    def test_file_where_the_commit_has_a_symlink_is_refused(self, tmp_path):
        staging, tree, package = TestVerifyStaging()._staged(tmp_path)
        tree["main/link"] = ("120000", git_blob_id(b"app.py"))
        (staging / "src" / "main" / "link").write_bytes(b"app.py")
        assert verify_staging(staging, tree, [package]) == ["src/main/link: the commit has a symlink here"]

    def test_oci_layout_is_read(self, tmp_path):
        package_path = _package(tmp_path)
        layers = _good_layers(package_path)
        layers.append([("tmp/.wh.gone", b"")])
        report, _ = check_layers(_save_oci(tmp_path, layers), TREE, [read_package(package_path)], LOCK)
        assert [c for c, _ in report.refusals] == ["whiteout"]

    def test_group_writable_file_without_exec_bit_is_refused(self, tmp_path):
        def mutate(layers):
            layers[2] = [("app/main/app.py", APP_SOURCE, {"mode": 0o664})]
        assert _check(tmp_path, mutate) == [
            ("provenance", "layer 2: app/main/app.py (a group- or world-writable file)")]

    def test_world_writable_implied_folder_is_refused(self, tmp_path):
        refusals = _check(tmp_path, lambda l: l[2].append(("app/main", None, {"directory": True, "mode": 0o777})))
        assert refusals == [("provenance", "layer 2: app/main (a group- or world-writable folder)")]

    def test_snapshot_entry_that_is_a_regular_file_is_refused(self, tmp_path):
        def mutate(layers):
            layers[1][3] = (f"{MODEL}/snapshots/{REVISION}/config.json", CONFIG)
        assert _check(tmp_path, mutate) == [("provenance", f"layer 1: {MODEL}/snapshots/{REVISION}/config.json "
                                                           f"(a snapshot entry that is not a symlink)")]
