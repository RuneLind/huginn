# Distribution-gate detectors — measured results

Numbers only. No fixture, no table cell and no example below contains a real
name; the corpus measurements are counts taken from collections whose contents
stay on the machine that produced them.

Reproduce with:

```sh
.venv/bin/python -m benchmarks.pii.bench_sensitivity_detection   # both benchmarks, no models
uv run benchmarks/runner.py --suite pii                          # inside the runner
```

The bigram measurements read the private alias map at runtime and are skipped
entirely when no private sub-repo is checked out.

## 1. `SensitivityScanner` — labelled synthetic fixtures

30 cases — 18 labelled positives carrying 19 expected findings across 9
categories (one case is both a credential and an email), plus 12 negatives that
actually produced false positives on the real corpora before the anchors went
in (`pageId=` values that pass MOD11, lists of eight-digit ids, GitHub run ids,
short `secret_key` values, file-like `@` references, package annotations).

| Category | Precision | Recall | TP | FP | FN |
|---|---|---|---|---|---|
| personnummer | 1.00 | 1.00 | 1 | 0 | 0 |
| organisasjonsnummer | 1.00 | 1.00 | 4 | 0 | 0 |
| bankkonto | 1.00 | 1.00 | 3 | 0 | 0 |
| credential | 1.00 | 1.00 | 3 | 0 | 0 |
| nav_ident | 1.00 | 1.00 | 1 | 0 | 0 |
| dotted_handle | 1.00 | 1.00 | 1 | 0 | 0 |
| email | 1.00 | 1.00 | 2 | 0 | 0 |
| password | 1.00 | 1.00 | 1 | 0 | 0 |
| telefon | 1.00 | 1.00 | 3 | 0 | 0 |
| **overall** | **1.00** | **1.00** | 19 | 0 | 0 |

A fixture suite the detector scores 1.00 on proves only that the shapes it was
built for are the shapes it matches. The measurement that carries weight is the
next one.

## 2. The same detectors on the three real in-scope collections

Findings per collection over the documents of each. Three variants of the same
detectors, to isolate what each precision requirement buys:

* **anchored** — what shipped: check digit, plus a keyword anchor within 60
  characters or the published grouping, plus the leading 8/9 for org numbers;
* **no anchor** — check digit and leading digit only;
* **shape** — check digit only, which is where a detector built from the format
  description alone lands.

| Detector | | nav-wiki | melosys-confluence-v3 | jira-issues |
|---|---|---|---|---|
| organisasjonsnummer | anchored | 6 | 18 | 20 |
| | no anchor | 8 | 18 | 38 |
| | shape | 270 | 484 | 72 |
| telefon | anchored | 0 | 0 | 2 |
| | no anchor / shape | 4 | 42 | 317 |
| bankkonto | anchored | 0 | 0 | 0 |
| | no anchor / shape | 0 | 0 | 71 |
| personnummer | | 0 | 0 | 0 |
| credential | | 0 | 0 | 0 |
| email (advisory) | | 0 | 2 | 0 |
| password (advisory) | | 0 | 0 | 0 |

**48 findings to triage, against 1 262 for the shape-only detectors.** Nothing
in the difference was a real organisation number, phone number or account
number: they were Confluence `pageId` values, lists of eight-digit identifiers,
and eleven-digit ids that happen to satisfy MOD11. Anchoring is what makes this
a gate someone will keep running rather than switch off.

## 3. Capitalised-bigram detector — recall against the map's own names

Each of the 90 mapped people's full names is embedded in a synthetic sentence
(mid-sentence, after a list bullet, in a table cell, before a full stop) and the
detector runs with a gazetteer stripped of everything those 90 entries
contribute — the measurement of "would this person be caught if nobody had ever
mapped them".

| Gazetteer | Recall over 90 names |
|---|---|
| public `given_names.txt` only (875 entries) | 0.53 |
| public + `bare_given_name_residual` | 0.72 |

The 28 % the union misses are people whose given name is in neither list. That
is the detector's real ceiling and the reason the private map's own given names
(232 of them) are unioned in at runtime: recall on the *unmapped* population is
bounded by how well the gazetteer covers given names, not by the pattern.

## 4. Capitalised-bigram detector — false positives on the real corpus

Measured on the aliased `nav-wiki` document text, before any allow-listing:

| Measure | Value |
|---|---|
| capitalised pairs in the corpus | 1 072 |
| pairs retained as candidates | 22 |
| candidate rate | 2.1 % |
| candidates left after the reviewed allow-list | 4 |
| share of candidates that were real unmapped people | 18 % |

Across all three collections the first run produced 59 distinct candidates; 44
were reviewed as non-people and allow-listed (place and street names, design-
system components, Norwegian words that are also given names, the national
synthetic test identities) and 15 were real people missing from the alias map.

2 % of capitalised pairs reaching a reviewer, one time, is the cost. 15 people
who would otherwise have been copied to another machine under their own names
is the return.
