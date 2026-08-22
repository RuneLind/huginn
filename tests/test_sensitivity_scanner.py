"""Guards on main/privacy/sensitivity_scanner — the detect-only half of the gate.

Two things this module must never do, and one it must always do:

* it must never rewrite anything, and in particular the new categories must stay
  out of `PiiSanitizer.sanitize`, which is the live Jira ingest WRITE path. A
  false positive there rewrites a stored document and the original is gone; a
  false positive here costs one line of triage;
* it must never put matched text in a finding — gate output gets pasted into
  PRs, and PiiSanitizer's own `matched_text` keeps a fødselsnummer's six
  birth-date digits readable;
* it must stay precise enough to be worth running. The negatives below are the
  shapes that actually produced false positives on the real corpora.

Every numeric literal is synthetic and constructed to satisfy (or deliberately
fail) the relevant check digit.
"""
import pytest

from main.privacy.sensitivity_scanner import (
    ADVISORY_CATEGORIES, ALL_CATEGORIES, BLOCKING_CATEGORIES, SensitivityScanner, shape,
)
from scripts.jira.sanitizers.pii_sanitizer import PiiSanitizer

VALID_ORGNR = "987654325"
INVALID_ORGNR = "912345670"          # leading 9, anchored, MOD11 fails
VALID_BANK = "12345678903"


@pytest.fixture
def scanner():
    return SensitivityScanner()


def categories(scanner, text):
    return {finding.category for finding in scanner.detect(text)}


# --- organisasjonsnummer ----------------------------------------------------

@pytest.mark.parametrize("text", [
    f"Org. nr.: {VALID_ORGNR}",
    f"arbeidsgiver med orgnr {VALID_ORGNR} i registeret",
    f"organisasjonsnummer er {VALID_ORGNR}",
    "Foretaksnummer 987 654 325",
    "987 654 325",                                   # the published grouping IS the anchor
])
def test_an_organisasjonsnummer_is_detected(scanner, text):
    assert "organisasjonsnummer" in categories(scanner, text)


@pytest.mark.parametrize("text", [
    f"viewpage.action?pageId={VALID_ORGNR}",         # 270 of these on one real collection
    f"Org. nr.: {INVALID_ORGNR}",                    # anchored, check digit fails
    f"referanse {VALID_ORGNR} i saken",              # valid shape, no anchor, no grouping
    "123456789 er en id",                            # does not start in the 8/9 series
])
def test_a_number_that_only_looks_like_one_is_not(scanner, text):
    assert "organisasjonsnummer" not in categories(scanner, text)


# --- bankkonto --------------------------------------------------------------

@pytest.mark.parametrize("text", ["kontonummer 1234.56.78903",
                                  f"Kontonr {VALID_BANK} for utbetaling",
                                  "versjon 1234.56.78903 av skjemaet"])
def test_a_bank_account_is_detected(scanner, text):
    assert "bankkonto" in categories(scanner, text)


def test_eleven_bare_digits_without_an_anchor_are_not_an_account(scanner):
    """Bare 11-digit MOD11 is the fødselsnummer rule and stays that category;
    71 of these were ordinary ids on one real collection."""
    assert "bankkonto" not in categories(scanner, f"id {VALID_BANK} i loggen")


# --- telefon ----------------------------------------------------------------

@pytest.mark.parametrize("text", ["ring +47 91234567", "tlf: 22 33 44 55",
                                  "Mobilnummer 90 12 34 56"])
def test_a_phone_number_is_detected(scanner, text):
    assert "telefon" in categories(scanner, text)


def test_a_list_of_eight_digit_ids_is_not_a_phone_number(scanner):
    assert "telefon" not in categories(scanner, "ider: 22334455, 22334466, 22334477")


# --- credentials ------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "api_key = A1b2C3d4E5f6G7h8J9k0L1m2",
    "client_secret: 'zzzzzzzzzzzzzzzzzzzzzzzz'",
    "https://svc:hunter2secret@intern.example.no/api",
    "Authorization: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijkl",
])
def test_a_credential_is_detected(scanner, text):
    assert "credential" in categories(scanner, text)


@pytest.mark.parametrize("text", ["token count was 24000 for the run",
                                  "secret_key = short",
                                  "api_key is documented in the README"])
def test_prose_about_credentials_is_not_a_credential(scanner, text):
    assert "credential" not in categories(scanner, text)


# --- reused PiiSanitizer categories -----------------------------------------

def test_the_pii_categories_are_reused_not_reimplemented(scanner):
    found = categories(scanner, "Fødselsnummer: 01010100050\npassord er hemmelig123")
    assert {"personnummer", "password"} <= found


def test_a_narrowed_scanner_only_looks_for_what_it_was_asked(scanner):
    narrow = SensitivityScanner(categories={"organisasjonsnummer"})
    assert categories(narrow, f"orgnr {VALID_ORGNR} og passord er hemmelig123") == \
        {"organisasjonsnummer"}


def test_ident_exceptions_are_honoured():
    exempt = SensitivityScanner(ident_exceptions={"Q000124"})
    assert "nav_ident" not in categories(exempt, "endret av Q000124")
    assert "nav_ident" in categories(exempt, "endret av Q000999")


# --- the invariants ---------------------------------------------------------

def test_no_finding_carries_the_matched_text(scanner):
    """Shapes only. PiiSanitizer's own `matched_text` keeps the six birth-date
    digits of a fødselsnummer readable, which is fine for its ingest log and not
    fine for something pasted into a PR."""
    text = (f"Fødselsnummer: 01010100050, orgnr {VALID_ORGNR}, "
            f"api_key = A1b2C3d4E5f6G7h8J9k0L1m2")
    for finding in scanner.detect(text):
        assert not any(char.isdigit() and char != "9" for char in finding.shape)
        assert "A1b2C3d4" not in finding.shape


def test_the_new_categories_are_not_wired_into_the_sanitize_write_path():
    """The one invariant that cannot be relaxed later without losing documents.

    PiiSanitizer.sanitize rewrites stored Jira content irreversibly. An
    organisation number, a phone number, an account number or a token going
    through it would be mangled with no original to recover.
    """
    text = (f"orgnr {VALID_ORGNR}, tlf: 22 33 44 55, kontonummer 1234.56.78903, "
            f"api_key = A1b2C3d4E5f6G7h8J9k0L1m2")
    result = PiiSanitizer().sanitize(text)
    assert result.sanitized_text == text
    assert result.findings == []


def test_blocking_and_advisory_categories_are_disjoint_and_complete():
    assert BLOCKING_CATEGORIES & ADVISORY_CATEGORIES == set()
    assert BLOCKING_CATEGORIES | ADVISORY_CATEGORIES == ALL_CATEGORIES
    # The categories measured precise enough to stop a hand-off.
    assert {"personnummer", "organisasjonsnummer", "bankkonto",
            "credential"} <= BLOCKING_CATEGORIES
    # …and the ones that would fail every real collection if they blocked.
    assert {"email", "telefon"} <= ADVISORY_CATEGORIES


def test_findings_carry_the_line_they_were_found_on(scanner):
    findings = scanner.detect(f"linje en\nlinje to\norgnr {VALID_ORGNR}")
    assert [f.line_number for f in findings] == [3]


def test_shape_masks_letters_and_digits():
    assert shape("Ada-99 X") == "xxx-99 x"


def test_empty_text_is_not_scanned(scanner):
    assert scanner.detect("") == []
