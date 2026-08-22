"""Detect-only scan for sensitive tokens in a built collection.

This module NEVER rewrites text. It exists because the distribution gate has to
answer "is this tarball safe to hand to a teammate", and that question is wider
than "did the alias map catch every person": a fødselsnummer, an API token or a
bank account number in an indexed document is just as unshippable as a name.

Two halves:

* the categories ``scripts/jira/sanitizers/pii_sanitizer.PiiSanitizer`` already
  knows (personnummer, email, plaintext password), reused through its
  ``detect()`` — the read-only half of that class;
* the categories added here (NAV idents, dotted handles, organisasjonsnummer,
  Norwegian phone numbers, bank account numbers, credentials).

**The new categories are deliberately NOT wired into ``PiiSanitizer.sanitize``.**
That method is the live Jira ingest write path: a false positive there rewrites
the stored document and the original is gone. A false positive *here* costs one
line of triage in a gate report. The two error costs are orders of magnitude
apart, which is why detection and redaction stay separate classes.

PRECISION IS THE WHOLE DESIGN. A gate that cries wolf is a gate someone
disables. Every numeric category therefore needs more than a shape:

* organisasjonsnummer — 9 digits, MOD11 check digit, **and** a leading 8 or 9
  (Norwegian org numbers are allocated from those series), **and** either a
  keyword anchor nearby or the published `NNN NNN NNN` grouping. Shape alone
  matched 270/466/66 times across the three in-scope collections, essentially
  all of them Confluence `pageId=` values.
* telefon — 8 digits is a Norwegian number and also a line in a list of ids, a
  timestamp fragment, a Jira id. Requires `+47` or a keyword anchor. Shape alone
  matched 319 times on one collection, none of them phone numbers.
* bankkonto — the published `NNNN.NN.NNNNN` grouping, or 11 MOD11-valid digits
  with a keyword anchor. Bare 11-digit MOD11 is the fødselsnummer rule and stays
  that category.

Findings carry a SHAPE, never the matched text: gate output is pasted into PRs.
"""

import re
from dataclasses import dataclass

from scripts.jira.sanitizers.pii_sanitizer import PiiSanitizer

# Categories that make a collection unshippable on their own. The rest are
# reported with counts but do not fail the gate — see the module docstring of
# main/privacy/index_scan.py for the measurement behind each choice.
BLOCKING_CATEGORIES = frozenset({
    "personnummer", "organisasjonsnummer", "bankkonto", "credential",
    "nav_ident", "dotted_handle",
})

ADVISORY_CATEGORIES = frozenset({"email", "password", "telefon"})

ALL_CATEGORIES = BLOCKING_CATEGORIES | ADVISORY_CATEGORIES


@dataclass(frozen=True)
class Finding:
    """One detection. ``shape`` is the match with letters -> x and digits -> 9."""
    category: str
    shape: str
    line_number: int


def shape(text: str) -> str:
    """Letters -> x, digits -> 9. Everything this module reports goes through it.

    PiiSanitizer's own ``matched_text`` for a personnummer keeps the six
    birth-date digits readable, which is fine for its local ingest log and not
    fine for a gate report that gets pasted somewhere.
    """
    return re.sub(r"\d", "9", re.sub(r"[^\W\d_]", "x", text))


# --- new detect-only categories ---------------------------------------------

# 9 digits, first one 8 or 9 (the series Brønnøysund allocates from), optionally
# in the published `NNN NNN NNN` grouping.
_ORGNR_RE = re.compile(r"(?<!\d)([89]\d{2})[  ]?(\d{3})[  ]?(\d{3})(?!\d)")
_ORGNR_ANCHOR = re.compile(
    r"(?i)(?:orgnr|org\.?\s?nr|organisasjonsnr|organisasjonsnummer|orgnummer"
    r"|organisasjoner|foretaksnummer|virksomhetsnummer|bedriftsnummer)")

_PHONE_RE = re.compile(r"(?<![\w+])(\+47[  ]?)?(\d{2}[  ]?\d{2}[  ]?\d{2}[  ]?\d{2}"
                       r"|\d{3}[  ]?\d{2}[  ]?\d{3})(?![\d\w])")
_PHONE_ANCHOR = re.compile(r"(?i)(?:tlf|telefon|mobil(?:nr|nummer)?|phone|tel\.)")

# Published account-number grouping, or eleven bare digits.
_BANK_GROUPED_RE = re.compile(r"(?<!\d)(\d{4})\.(\d{2})\.(\d{5})(?!\d)")
_BANK_BARE_RE = re.compile(r"(?<![\d./])(\d{11})(?![\d./])")
_BANK_ANCHOR = re.compile(r"(?i)(?:kontonr|kontonummer|bankkonto|kontonummeret"
                          r"|account\s?(?:no|number)|iban)")

_IDENT_RE = re.compile(r"(?i)(?<!\w)[a-z]\d{6}(?!\w)")
_HANDLE_RE = re.compile(r"(?<![\w.])@[a-zæøå0-9]+(?:\.[a-zæøå0-9]+)+", re.IGNORECASE)

# Credentials. Three shapes, all of which are unambiguous enough to block on:
# a key=value assignment with a long opaque value, a URL carrying userinfo with
# a password, and the vendor-prefixed token formats plus a JWT.
_CRED_ASSIGN_RE = re.compile(
    r"""(?ix)
    \b(?:api[_-]?key|apikey|access[_-]?token|auth[_-]?token|client[_-]?secret
       |secret[_-]?key|private[_-]?key|bearer)\b
    \s*[:=]\s*
    ["']?(?!<redacted)[A-Za-z0-9_\-./+]{20,}["']?
    """)
_CRED_URL_RE = re.compile(r"(?i)\b[a-z][a-z0-9+.\-]*://[^/\s:@]+:[^/\s@]{4,}@[\w.\-]+")
_CRED_TOKEN_RE = re.compile(
    r"(?:\bsk-[A-Za-z0-9]{20,}"
    r"|\bgh[pousr]_[A-Za-z0-9]{30,}"
    r"|\bxox[baprs]-[A-Za-z0-9-]{10,}"
    r"|\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})")

_ANCHOR_WINDOW = 60


def _mod11(digits: str, weights: list) -> bool:
    """Norwegian MOD11 check digit over `digits[:-1]` with `weights`."""
    total = sum(int(digits[i]) * weights[i] for i in range(len(weights)))
    check = 11 - (total % 11)
    if check == 11:
        check = 0
    return check != 10 and check == int(digits[-1])


def _anchored(text: str, start: int, end: int, anchor: re.Pattern) -> bool:
    window = text[max(0, start - _ANCHOR_WINDOW):min(len(text), end + _ANCHOR_WINDOW)]
    return bool(anchor.search(window))


class SensitivityScanner:
    """Detect-only. Composes PiiSanitizer's detection with the extra categories.

    ``categories`` narrows what is looked for; the default is everything.
    ``ident_exceptions`` are ident-shaped tokens that are not people (git SHAs,
    document type ids), the same set the alias registry takes.
    """

    def __init__(self, categories=None, ident_exceptions=()):
        self.categories = frozenset(categories) if categories else ALL_CATEGORIES
        self._ident_exceptions = {token.lower() for token in ident_exceptions}
        self._pii = PiiSanitizer(
            redact_personnummer="personnummer" in self.categories,
            redact_emails="email" in self.categories,
            redact_passwords="password" in self.categories,
        )

    def detect(self, text: str) -> list:
        if not text:
            return []
        findings = []
        if self.categories & {"personnummer", "email", "password"}:
            for pii in self._pii.detect(text):
                if pii.category in self.categories:
                    findings.append(Finding(pii.category, shape(pii.matched_text),
                                            pii.line_number))
        line_starts = self._line_starts(text)
        for category, matcher in (("organisasjonsnummer", self._organisasjonsnummer),
                                  ("telefon", self._telefon),
                                  ("bankkonto", self._bankkonto),
                                  ("credential", self._credential),
                                  ("nav_ident", self._nav_ident),
                                  ("dotted_handle", self._dotted_handle)):
            if category not in self.categories:
                continue
            for start, matched in matcher(text):
                findings.append(Finding(category, shape(matched),
                                        self._line_of(line_starts, start)))
        return findings

    # --- per-category matchers (each yields (start, matched_text)) ----------

    def _organisasjonsnummer(self, text):
        for match in _ORGNR_RE.finditer(text):
            digits = "".join(match.groups())
            if not _mod11(digits, [3, 2, 7, 6, 5, 4, 3, 2]):
                continue
            grouped = match.group(0) != digits          # printed as NNN NNN NNN
            if grouped or _anchored(text, match.start(), match.end(), _ORGNR_ANCHOR):
                yield match.start(), match.group(0)

    def _telefon(self, text):
        for match in _PHONE_RE.finditer(text):
            if match.group(1) or _anchored(text, match.start(), match.end(), _PHONE_ANCHOR):
                yield match.start(), match.group(0)

    def _bankkonto(self, text):
        for match in _BANK_GROUPED_RE.finditer(text):
            if _mod11("".join(match.groups()), [5, 4, 3, 2, 7, 6, 5, 4, 3, 2]):
                yield match.start(), match.group(0)
        for match in _BANK_BARE_RE.finditer(text):
            if (_mod11(match.group(1), [5, 4, 3, 2, 7, 6, 5, 4, 3, 2])
                    and _anchored(text, match.start(), match.end(), _BANK_ANCHOR)):
                yield match.start(), match.group(0)

    def _credential(self, text):
        for pattern in (_CRED_ASSIGN_RE, _CRED_URL_RE, _CRED_TOKEN_RE):
            for match in pattern.finditer(text):
                yield match.start(), match.group(0)

    def _nav_ident(self, text):
        for match in _IDENT_RE.finditer(text):
            if match.group(0).lower() not in self._ident_exceptions:
                yield match.start(), match.group(0)

    def _dotted_handle(self, text):
        # Imported lazily: alias_registry imports nothing from this module, and
        # keeping the dependency one-way lets the sanitizer be used standalone.
        from main.privacy.alias_registry import is_person_handle
        for match in _HANDLE_RE.finditer(text):
            if is_person_handle(match.group(0)):
                yield match.start(), match.group(0)

    # --- line numbers -------------------------------------------------------

    @staticmethod
    def _line_starts(text: str) -> list:
        starts = [0]
        for index, char in enumerate(text):
            if char == "\n":
                starts.append(index + 1)
        return starts

    @staticmethod
    def _line_of(starts: list, position: int) -> int:
        low, high = 0, len(starts) - 1
        while low < high:
            mid = (low + high + 1) // 2
            if starts[mid] <= position:
                low = mid
            else:
                high = mid - 1
        return low + 1
