# Distribution-gate detectors — measured results

Numbers only. No fixture, no table cell and no example below contains a real
name; the corpus measurements are counts taken from collections whose contents
stay on the machine that produced them.

The three in-scope collections are reported as **A / B / C**, in sorted order of
the names in `main/privacy/scope.json`. The mapping is deliberately not written
here: a per-collection finding count is a statement about a named corpus, and
this file is tracked in a public repo.

Reproduce with:

```sh
.venv/bin/python -m benchmarks.pii.bench_sensitivity_detection   # all three, no models
uv run benchmarks/runner.py --suite pii                          # inside the runner
```

The corpus measurements read the private alias map and the built collections at
runtime, and are skipped entirely when neither is present.

## 1. `SensitivityScanner` — labelled synthetic fixtures

36 cases — 21 labelled positives carrying 22 expected findings across 9
categories (one case is both a credential and an email), plus 15 negatives that
actually produced false positives on the real corpora before the anchors went
in (`pageId=` values that pass MOD11, lists of eight-digit ids, GitHub run ids,
short `secret_key` values, a file path assigned to `private_key`, a nine-digit
id with only ONE group separator, a grouped account number whose check digit
fails, file-like `@` references, package annotations).

| Category | Precision | Recall | TP | FP | FN |
|---|---|---|---|---|---|
| personnummer | 1.00 | 1.00 | 1 | 0 | 0 |
| organisasjonsnummer | 1.00 | 1.00 | 5 | 0 | 0 |
| bankkonto | 1.00 | 1.00 | 3 | 0 | 0 |
| credential | 1.00 | 1.00 | 5 | 0 | 0 |
| nav_ident | 1.00 | 1.00 | 1 | 0 | 0 |
| dotted_handle | 1.00 | 1.00 | 1 | 0 | 0 |
| email | 1.00 | 1.00 | 2 | 0 | 0 |
| password | 1.00 | 1.00 | 1 | 0 | 0 |
| telefon | 1.00 | 1.00 | 3 | 0 | 0 |
| **overall** | **1.00** | **1.00** | 22 | 0 | 0 |

A fixture suite the detector scores 1.00 on proves only that the shapes it was
built for are the shapes it matches. The measurement that carries weight is the
next one.

## 2. The same detectors on the three real in-scope collections

Findings over the document text of each collection, from
`bench_anchoring_on_corpus`. Three variants of the same detectors, to isolate
what each precision requirement buys:

* **anchored** — what shipped: check digit, plus a keyword anchor within 60
  characters or the published grouping *with both separators*, plus the leading
  8/9 for org numbers;
* **no anchor** — check digit and leading digit only;
* **shape** — check digit only, which is where a detector built from the format
  description alone lands.

| Detector | | A | B | C |
|---|---|---|---|---|
| organisasjonsnummer | anchored | 10 | 8 | 3 |
| | no anchor | 17 | 0 | 4 |
| | shape | 33 | 5 | 90 |
| telefon | anchored | 1 | 0 | 0 |
| | no anchor / shape | 153 | 16 | 2 |
| bankkonto | anchored | 0 | 0 | 0 |
| | no anchor / shape | 35 | 0 | 0 |
| personnummer | | 0 | 0 | 0 |
| credential | | 0 | 0 | 0 |
| email (advisory) | | 0 | 1 | 0 |
| password (advisory) | | 0 | 0 | 0 |

**23 findings to triage, against 334 for the shape-only detectors.** Nothing in
the difference was a real organisation number, phone number or account number:
they were Confluence `pageId` values, lists of eight-digit identifiers, and
eleven-digit ids that happen to satisfy MOD11. Anchoring is what makes this a
gate someone will keep running rather than switch off.

Of the categories above only `personnummer`, `bankkonto` and `credential` (plus
the ident and handle checks, which have their own numbers) BLOCK a hand-off.
`organisasjonsnummer` is advisory: a Norwegian organisation number identifies a
company in a public register, it is not personal data, and it is legitimate
content in an issue about an employer. It is also the category that fires most
— blocking on it would make the gate unpassable for the subject matter.

## 3. Capitalised-bigram detector — recall against the map's own names

Each mapped person's full name is embedded in a synthetic sentence
(mid-sentence, after a list bullet, in a table cell, before a full stop) and the
detector runs with a gazetteer stripped of everything the map's entries
contribute — the measurement of "would this person be caught if nobody had ever
mapped them".

| Gazetteer | Recall over the map's names |
|---|---|
| public `given_names.txt` only (871 entries) | 0.72 |
| public + `bare_given_name_residual` | 0.97 |

Both figures were 0.53 / 0.72 before the detector evaluated **every** adjacent
pair of a capitalised run rather than only the pair at its head. The names it
was missing were not exotic: a full name after an acronym, after a heading word,
or after another name is behind a capitalised token, and the head-only probe
never looked past it.

The remainder the union still misses are people whose given name is in neither
list. That is the detector's real ceiling and the reason the private map's own
given names (232 of them) are unioned in at runtime.

## 4. Capitalised-bigram detector — false positives on the real corpus

Measured on collection A's aliased document text, before any allow-listing:

| Measure | Value |
|---|---|
| adjacent capitalised pairs in the corpus | 1 230 |
| pairs retained as candidates | 25 |
| candidate rate | 2.0 % |
| candidates left after the reviewed allow-list | 4 |
| share of candidates that were real unmapped people | 16 % |

Across all three collections the pairwise run produced 90 candidates before the
allow-list; 71 are now reviewed as non-people and allow-listed (design-system
components, systems whose name is also a given name, place and street names,
typeface names, Norwegian function words at a capitalised position, the national
synthetic test identity, and a given name followed by a section heading) and 17
are real people missing from the alias map — two more than the head-only
detector found.

2 % of capitalised pairs reaching a reviewer, one time, is the cost. 17 people
who would otherwise have been copied to another machine under their own names
is the return.

## 5. Gate runtime

The needle alternation is ~1 800 branches, each with its own lookarounds, and
Python's engine tries every branch at every position.

| Step | Before | After |
|---|---|---|
| check 1 over collection A (4.0 M characters) | 177 s | 0.9 s |
| whole `scan_index.py` run over collection A | 83 s | 2.4 s |

Two changes, both of which keep the scan a *superset* of what the single
alternation finds: needles are bucketed by their lowercased first word and a
bucket runs only when that word occurs as a substring of the text, and — because
every needle in a bucket begins with the bucket word — the bucket's alternation
is tried only at the positions where that word occurs, rather than at all four
million. `Pattern.match(text, pos)` still evaluates lookbehinds against the text
before `pos`, which is what keeps the boundaries meaningful.

The predecessor was a token prefilter, which is not a superset: it missed a slug
welded onto a preceding word and a name behind a percent escape whose hex digits
ate its first letters. A miss there is a silent PASS, and
`test_the_bucketed_scan_never_hides_a_hit_the_alternation_finds` fuzzes 500
fenced texts against the full alternation to hold the property.
