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
# The published groupings, MOD11-valid and MOD11-invalid.
GROUPED_BANK = "1234.56.78903"
GROUPED_BANK_BAD_CHECK = "1234.56.00000"
# MOD11-valid, leading 9, but only ONE separator: not the published grouping.
HALF_GROUPED_ORGNR = "974 760002"
NBSP = "\u00a0"


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

@pytest.mark.parametrize("text", [f"kontonummer {GROUPED_BANK}",
                                  f"Kontonr {VALID_BANK} for utbetaling",
                                  f"{GROUPED_BANK} er kontoen"])
def test_a_bank_account_is_detected(scanner, text):
    assert "bankkonto" in categories(scanner, text)


def test_a_grouped_number_that_fails_mod11_is_not_an_account(scanner):
    """The grouping alone is the anchor for this category, so the check digit is
    the ONLY thing separating an account number from `versjon 1234.56.00000`."""
    assert "bankkonto" not in categories(scanner, f"kontonummer {GROUPED_BANK_BAD_CHECK}")


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
    # Spelled out rather than asserted by rule: "no digit other than 9" is
    # satisfied by the empty string, by a shape the masking dropped characters
    # from, and by anything the detector simply stopped finding.
    assert [(f.category, f.shape) for f in scanner.detect(text)] == [
        ("personnummer", "999999*****"),
        ("organisasjonsnummer", "999999999"),
        ("credential", "xxx_xxx = x9x9x9x9x9x9x9x9x9x9x9x9"),
    ]


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
    # The categories measured precise enough to stop a hand-off, and about
    # something that must not leave the machine in the first place.
    assert {"personnummer", "bankkonto", "credential"} <= BLOCKING_CATEGORIES
    # …and the ones that would fail every real collection if they blocked, or
    # are not personal data at all (an organisation number is public register
    # data about a company).
    assert {"email", "telefon", "organisasjonsnummer"} <= ADVISORY_CATEGORIES


def test_findings_carry_the_line_they_were_found_on(scanner):
    findings = scanner.detect(f"linje en\nlinje to\norgnr {VALID_ORGNR}")
    assert [f.line_number for f in findings] == [3]


def test_shape_masks_letters_and_digits():
    assert shape("Ada-99 X") == "xxx-99 x"


def test_empty_text_is_not_scanned(scanner):
    assert scanner.detect("") == []


# --- separators, groupings and the categories that block ---------------------

def test_a_non_breaking_space_separates_an_organisasjonsnummer(scanner):
    """`[  ]` in the source was two ASCII spaces, not space-and-NBSP.

    Copy-paste out of Confluence and Word produces U+00A0 between the groups,
    which is precisely the published `NNN NNN NNN` form the detector treats as
    its own anchor — so the shape it was written for was the one it missed.
    """
    assert "organisasjonsnummer" in categories(scanner, f"987{NBSP}654{NBSP}325")


def test_a_non_breaking_space_separates_a_phone_number(scanner):
    assert "telefon" in categories(scanner, f"tlf: 22{NBSP}33{NBSP}44{NBSP}55")


def test_one_separator_is_not_the_published_grouping(scanner):
    """`974 760002` is a line-wrapped id, not `NNN NNN NNN`. Accepting a single
    separator made the grouping rule — which stands in for a keyword anchor —
    fire on half the digit strings a table cell wraps."""
    assert "organisasjonsnummer" not in categories(scanner, HALF_GROUPED_ORGNR)
    assert "organisasjonsnummer" in categories(scanner, f"orgnr {HALF_GROUPED_ORGNR}")


def test_an_organisasjonsnummer_is_advisory(scanner):
    """A Norwegian organisation number is public register data about a company.

    It is not personal data, it is legitimate content in a Jira issue about an
    employer, and it fired 6/18/20 times across the three in-scope collections.
    Blocking the hand-off on it makes the gate unpassable for the thing the
    corpus is *about*, which is how a gate gets switched off.
    """
    from main.privacy.sensitivity_scanner import ADVISORY_CATEGORIES, BLOCKING_CATEGORIES
    assert "organisasjonsnummer" in ADVISORY_CATEGORIES
    assert "organisasjonsnummer" not in BLOCKING_CATEGORIES


# --- credentials -------------------------------------------------------------

def test_an_authorization_bearer_header_is_a_credential(scanner):
    """The keyword rule wanted `bearer` followed by `:` or `=`; the header puts
    the colon after `Authorization` and the token after the word `Bearer`."""
    assert "credential" in categories(
        scanner, "Authorization: Bearer A1b2C3d4E5f6G7h8J9k0L1m2N3o4")


def test_a_pem_private_key_block_is_a_credential(scanner):
    assert "credential" in categories(
        scanner, "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA\n")
    assert "credential" in categories(scanner, "-----BEGIN PRIVATE KEY-----")


def test_a_long_file_path_assigned_to_a_key_is_not_a_credential(scanner):
    """`private_key: /etc/ssl/private/service-account-signing.pem` is a
    CONFIGURATION line naming where the key lives. Reporting it as the key
    itself is the false positive that makes an operator stop reading check 11.
    """
    assert "credential" not in categories(
        scanner, "private_key: /etc/ssl/private/service-account-signing.pem")
    assert "credential" not in categories(
        scanner, "api_key_file = ./config/secrets/api-key-production.txt")
    # …and a base64-shaped value with slashes in it is still a credential.
    assert "credential" in categories(
        scanner, "access_token=aB3/dE6+gH9jK2mN5pQ8rS1tU4vW7xY0zA3b=")


# Real secret material that the old "contains a slash and no `=`/`+`" rule
# filed as a file path. `/` is 62 of the 64 base64 alphabet's characters away
# from being path-specific — it IS a base64 character — and `=` was never even
# reachable, because the assignment pattern's value class does not capture it.
# Every one of these was therefore silently dropped from a BLOCKING category.
@pytest.mark.parametrize("text", [
    "api_key=abcdefgh/ijklmnopqrstuvwxyz012345",
    'client_secret: "Gq3/8vNwq2Lx0pTz9RmYb4Kd7Jc1Ae6H"',
    "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "Authorization: Bearer abc/def/ghi/jklmnopqrs",
])
def test_a_secret_with_a_slash_in_it_is_still_a_credential(scanner, text):
    assert "credential" in categories(scanner, text)


@pytest.mark.parametrize("text", [
    "api_key = ../../config/some/very/long/relative/path.json",
    "private_key: /Users/x/.ssh/id_rsa_deployment_key_file",
])
def test_a_path_shaped_value_is_still_not_a_credential(scanner, text):
    """The precision the rule above must not cost: a relative path climbing out
    of the working directory, and an absolute path with no extension at all."""
    assert "credential" not in categories(scanner, text)


# --- laziness ---------------------------------------------------------------

def test_line_starts_are_not_computed_for_clean_text(scanner, monkeypatch):
    """`_line_starts` walks the string character by character in Python. It runs
    on every string of every document; on a clean collection every one of those
    walks is wasted."""
    from main.privacy.sensitivity_scanner import SensitivityScanner as S
    calls = []
    monkeypatch.setattr(S, "_line_starts", staticmethod(
        lambda text: calls.append(text) or [0]))
    assert scanner.detect("en helt vanlig setning uten noe som helst") == []
    assert calls == []
    scanner.detect(f"orgnr {VALID_ORGNR}")
    assert len(calls) == 1
