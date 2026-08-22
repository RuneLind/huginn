"""Benchmark: the distribution gate's detectors — precision, recall, FP rate.

Two measurements, both cheap enough to run without loading any model:

1. ``bench_sensitivity_detection`` — per-category precision and recall for
   ``main.privacy.sensitivity_scanner`` over labelled synthetic fixtures. The
   negatives are the shapes that actually produced false positives on the real
   corpora before the anchors went in: Confluence ``pageId`` values that pass
   MOD11, lists of eight-digit ids, GitHub run ids.
2. ``bench_bigram_detector`` — for check 9 of the gate:
   * **recall**, measured against the real map's own names, each held out as if
     nobody had ever mapped them and embedded in a synthetic sentence. That is
     the population the check exists to catch, and it is the only sample of it
     that exists. The map is read at runtime from the gitignored private
     sub-repo and only counts leave this function; with no map the measurement
     is skipped rather than faked.
   * **false-positive rate** on the real aliased ``nav-wiki``: how many distinct
     capitalised pairs survive retention *before* any allow-listing, over how
     many pairs the corpus contains. That ratio is what decides whether the gate
     is triageable or noise.

Every number reported here is a count or a rate. No fixture contains a real
name, and nothing read from the private map is returned.
"""

import json
import time
from pathlib import Path

from benchmarks.results import BenchmarkResult
from main.privacy.index_scan import (
    BIGRAM_RE, bigram_candidates, load_allowed_bigrams, load_public_given_names,
    map_given_names,
)
from main.privacy.sensitivity_scanner import ALL_CATEGORIES, SensitivityScanner

REPO_ROOT = Path(__file__).resolve().parents[2]
MAP_GLOB = "huginn-*/privacy/aliases.json"
NAV_WIKI = REPO_ROOT / "data" / "collections" / "nav-wiki"

# Every numeric literal below is synthetic and constructed to satisfy (or
# deliberately fail) the relevant check digit. None of them is anyone's.
CASES = [
    # --- organisasjonsnummer -------------------------------------------------
    ("Org. nr.: 987654325 er registrert", {"organisasjonsnummer"}),
    ("arbeidsgiver med orgnr 888888888", {"organisasjonsnummer"}),
    ("Foretaksnummer 987 654 325 i registeret", {"organisasjonsnummer"}),
    ("987 654 325", {"organisasjonsnummer"}),              # published grouping, no anchor
    ("confluence.example.no/pages/viewpage.action?pageId=987654325", set()),
    ("Org. nr.: 912345670 finnes ikke", set()),            # anchored, MOD11 fails
    ("referanse 123456789 i saken", set()),                # no anchor, no grouping
    # --- bankkonto -----------------------------------------------------------
    ("kontonummer 1234.56.78903", {"bankkonto"}),
    ("Kontonr 12345678903 for utbetaling", {"bankkonto"}),
    ("versjon 1234.56.78903 av skjemaet", {"bankkonto"}),  # grouped form is the signal
    ("id 12345678903 i loggen", set()),                    # 11 digits, no anchor
    # --- telefon -------------------------------------------------------------
    ("ring +47 91234567 ved feil", {"telefon"}),
    ("tlf: 22 33 44 55", {"telefon"}),
    ("Mobilnummer 90 12 34 56", {"telefon"}),
    ("ider: 22334455, 22334466, 22334477", set()),
    ("bygg 20260823 kjørte", set()),
    # --- credential ----------------------------------------------------------
    ("api_key = A1b2C3d4E5f6G7h8J9k0L1m2", {"credential"}),
    ("client_secret: 'zzzzzzzzzzzzzzzzzzzzzzzz'", {"credential"}),
    ("https://svc:hunter2secret@intern.example.no/api", {"credential", "email"}),
    ("token count was 24000 for the run", set()),
    ("secret_key = short", set()),                          # too short to be a key
    # --- personnummer (PiiSanitizer, reused through detect) ------------------
    ("Fødselsnummer: 01010100050", {"personnummer"}),
    ("GitHub run 12345678901", set()),
    # --- email / password (advisory) ----------------------------------------
    ("kontakt drift@intern.example.no", {"email"}),
    ("passord er hemmelig123", {"password"}),
    ("Foo@file.xsl er en referanse", set()),
    # --- nav ident / dotted handle ------------------------------------------
    ("saken ble endret av Q000124", {"nav_ident"}),
    ("skrevet av @ada.example", {"dotted_handle"}),
    ("bruk @org.junit.Test her", set()),
    ("oppgrader til @v1.2.3", set()),
]

# Sentences the held-out names are embedded in for the recall measurement. Each
# puts the name somewhere the detector must still see it: mid-sentence, after a
# list bullet, in a table cell, at the end before punctuation.
RECALL_TEMPLATES = [
    "Saken ble behandlet av {name} i forrige uke.",
    "- {name} tok over ansvaret",
    "| Ansvarlig | {name} | 2026 |",
    "Kontakt {name}.",
]


def _rates(tp: int, fp: int, fn: int) -> dict:
    return {
        "precision": tp / (tp + fp) if (tp + fp) else 1.0,
        "recall": tp / (tp + fn) if (tp + fn) else 1.0,
        "tp": tp, "fp": fp, "fn": fn,
    }


def bench_sensitivity_detection(ctx=None) -> BenchmarkResult:
    """Per-category precision and recall over the labelled fixtures."""
    scanner = SensitivityScanner()
    start = time.monotonic()

    tp = {c: 0 for c in ALL_CATEGORIES}
    fp = {c: 0 for c in ALL_CATEGORIES}
    fn = {c: 0 for c in ALL_CATEGORIES}
    for text, expected in CASES:
        found = {finding.category for finding in scanner.detect(text)}
        for category in ALL_CATEGORIES:
            if category in expected and category in found:
                tp[category] += 1
            elif category in expected:
                fn[category] += 1
            elif category in found:
                fp[category] += 1

    metrics = {"cases": len(CASES)}
    for category in sorted(ALL_CATEGORIES):
        for key, value in _rates(tp[category], fp[category], fn[category]).items():
            metrics[f"{category}_{key}"] = value
    total = _rates(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    metrics.update({f"overall_{k}": v for k, v in total.items()})

    return BenchmarkResult(name="sensitivity_detection", category="pii", metrics=metrics,
                           duration_ms=(time.monotonic() - start) * 1000)


def _allowlist_path():
    found = sorted(REPO_ROOT.glob("huginn-*/privacy/non_person_bigrams.json"))
    return found[0] if len(found) == 1 else None


def _load_map() -> dict | None:
    maps = sorted(REPO_ROOT.glob(MAP_GLOB))
    if len(maps) != 1:
        return None
    return json.loads(maps[0].read_text(encoding="utf-8"))


def bench_bigram_detector(ctx=None) -> BenchmarkResult:
    """Recall against the map's own names, held out; FP rate on the real corpus."""
    start = time.monotonic()
    alias_map = _load_map()
    if alias_map is None:
        return BenchmarkResult(name="bigram_detector", category="pii",
                               metrics={"skipped": 1.0},
                               duration_ms=(time.monotonic() - start) * 1000,
                               metadata={"reason": "no private alias map on this machine"})

    public = load_public_given_names()
    private = map_given_names(alias_map)
    names = [entry["name"] for entry in alias_map["entries"]
             if len(entry.get("name", "").split()) >= 2]

    # Held out: the gazetteer is stripped of everything the 90 ENTRIES contribute
    # to it, which is what "this person was never mapped" actually means. The
    # public file stays (it is not derived from the map at all) and so does
    # `bare_given_name_residual`, which is derived from the corpus rather than
    # from the entries — an unmapped person's given name would be in it for
    # exactly the same reason a mapped person's is.
    residual = {n.lower() for n in alias_map.get("bare_given_name_residual", {})}
    detected_public, detected_union = 0, 0
    for index, name in enumerate(names):
        text = RECALL_TEMPLATES[index % len(RECALL_TEMPLATES)].format(name=name)
        if bigram_candidates([text], public, set(), set())[0]:
            detected_public += 1
        if bigram_candidates([text], public | residual, set(), set())[0]:
            detected_union += 1

    metrics = {
        "map_names": len(names),
        "recall_public_gazetteer": detected_public / len(names) if names else 0.0,
        "recall_public_plus_residual": detected_union / len(names) if names else 0.0,
        "public_gazetteer_size": len(public),
        "private_given_names": len(private),
    }

    if (NAV_WIKI / "documents").exists():
        texts, pairs = [], 0
        for path in sorted((NAV_WIKI / "documents").rglob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            text = payload.get("text") or ""
            texts.append(text)
            pairs += len(BIGRAM_RE.findall(text))
        exempt = {label.lower() for label in alias_map.get("non_person_labels", [])}
        retained, _ = bigram_candidates(texts, public | private, exempt, set())
        allowed = load_allowed_bigrams(_allowlist_path())
        after, _ = bigram_candidates(texts, public | private, exempt, allowed)
        metrics.update({
            "corpus_capitalised_pairs": pairs,
            "corpus_candidates_before_allowlist": len(retained),
            # Of every capitalised pair in the corpus, this fraction lands in
            # front of a reviewer. It is the number that decides whether the
            # check is triageable.
            "corpus_candidate_rate": len(retained) / pairs if pairs else 0.0,
            "corpus_candidates_after_allowlist": len(after),
            # Once the reviewed non-person pairs are subtracted, what fraction of
            # the remaining candidates is a real person the map does not know.
            "corpus_candidate_precision_after_allowlist":
                len(after) / len(retained) if retained else 0.0,
        })

    return BenchmarkResult(name="bigram_detector", category="pii", metrics=metrics,
                           duration_ms=(time.monotonic() - start) * 1000)


if __name__ == "__main__":
    for result in (bench_sensitivity_detection(), bench_bigram_detector()):
        print(f"\n{result.name} ({result.duration_ms:.0f} ms)")
        for key, value in result.metrics.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
