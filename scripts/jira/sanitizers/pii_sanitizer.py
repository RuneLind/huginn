"""
PII Sanitizer for Jira issue content.

Detects and redacts personally identifiable information (PII) from text,
focused on Norwegian data:
  - Fødselsnummer / D-nummer (11-digit Norwegian national IDs)
  - Email addresses
  - Plaintext passwords

Usage:
    from scripts.jira.sanitizers.pii_sanitizer import PiiSanitizer

    sanitizer = PiiSanitizer()
    clean_text, findings = sanitizer.sanitize(text)

    # Or just detect without modifying:
    findings = sanitizer.detect(text)
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PiiFinding:
    """A single PII detection."""
    category: str          # "personnummer", "email", "password"
    matched_text: str      # The original matched text
    redacted_text: str     # What it was replaced with
    line_number: int       # 1-based line number in the text
    context: str           # Surrounding text for log readability


@dataclass
class SanitizeResult:
    """Result of sanitizing a piece of text."""
    original_text: str
    sanitized_text: str
    findings: list[PiiFinding] = field(default_factory=list)

    @property
    def has_pii(self) -> bool:
        return len(self.findings) > 0

    @property
    def changed(self) -> bool:
        return self.original_text != self.sanitized_text


# Norwegian personnummer (fnr/dnr) validation
# Format: DDMMYY + IIIKK (11 digits total)
# D-nummer: first digit +4 (day 41-71)
# Negative lookbehind rejects digits preceded by / or # (URL paths, GitHub run IDs)
_FNR_PATTERN = re.compile(r'(?<![/#])(?<!\d)\b(\d{11})\b(?!\d)')

# Email pattern — exclude file-like references (e.g. Foo@file.xsl)
_EMAIL_PATTERN = re.compile(
    r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.(?!xsl\b|xml\b|xsd\b|json\b|ya?ml\b|css\b|js\b|ts\b|java\b|kt\b|py\b|md\b|txt\b|html?\b|log\b)[a-zA-Z]{2,}\b'
)

# Password patterns: "passord er X", "password: X", "password is X", etc.
# Negative lookahead avoids re-matching already-redacted values.
_PASSWORD_PATTERN = re.compile(
    r'(?:passord|password)\s*(?:er|is|:)\s*(?!<redacted-password>)(\S+)',
    re.IGNORECASE,
)


def _is_plausible_fnr(digits: str) -> bool:
    """Check if 11 digits could be a Norwegian fødselsnummer or D-nummer.

    Uses basic date validation (not full mod-11 checksum) to filter out
    obviously non-personnummer numbers like GitHub run IDs, timestamps, etc.
    """
    if len(digits) != 11:
        return False

    day = int(digits[0:2])
    month = int(digits[2:4])

    # D-nummer: first digit has 4 added (day range 41-71)
    if day > 40:
        day -= 40

    if day < 1 or day > 31:
        return False
    if month < 1 or month > 12:
        return False

    return True


def _redact_fnr(digits: str) -> str:
    """Redact a personnummer, keeping first 6 digits (birthdate) visible."""
    return digits[:6] + "*****"


def _make_context(text: str, start: int, end: int, max_ctx: int = 40) -> str:
    """Extract surrounding context for a match, for logging."""
    ctx_start = max(0, start - max_ctx)
    ctx_end = min(len(text), end + max_ctx)
    before = text[ctx_start:start].replace('\n', ' ')
    after = text[end:ctx_end].replace('\n', ' ')
    matched = text[start:end]
    return f"...{before}[{matched}]{after}..."


class PiiSanitizer:
    """Detects and redacts PII from text content."""

    def __init__(
        self,
        redact_personnummer: bool = True,
        redact_emails: bool = True,
        redact_passwords: bool = True,
        email_allowlist: Optional[set[str]] = None,
        email_domain_allowlist: Optional[set[str]] = None,
    ):
        self.redact_personnummer = redact_personnummer
        self.redact_emails = redact_emails
        self.redact_passwords = redact_passwords
        self.email_allowlist = email_allowlist or set()
        self.email_domain_allowlist = email_domain_allowlist or {
            "example.com", "test.com", "example.org",
        }

    def detect(self, text: str) -> list[PiiFinding]:
        """Detect PII in text without modifying it."""
        return self._scan(text)

    def sanitize(self, text: str) -> SanitizeResult:
        """Detect and redact PII from text. Returns SanitizeResult."""
        findings = self._scan(text)
        if not findings:
            return SanitizeResult(original_text=text, sanitized_text=text)

        # Apply replacements from end to start to preserve offsets
        sanitized = text
        # Sort findings by position (we use line_number but need actual offsets)
        # Re-apply replacements using patterns to avoid offset tracking
        sanitized = self._apply_redactions(sanitized)

        return SanitizeResult(
            original_text=text,
            sanitized_text=sanitized,
            findings=findings,
        )

    def _scan(self, text: str) -> list[PiiFinding]:
        """Scan text for all PII categories."""
        findings: list[PiiFinding] = []

        # Pre-compute line starts for line number lookup
        line_starts = [0]
        for i, ch in enumerate(text):
            if ch == '\n':
                line_starts.append(i + 1)

        def _line_of(pos: int) -> int:
            """1-based line number for a character position."""
            lo, hi = 0, len(line_starts) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if line_starts[mid] <= pos:
                    lo = mid
                else:
                    hi = mid - 1
            return lo + 1

        if self.redact_personnummer:
            for m in _FNR_PATTERN.finditer(text):
                digits = m.group(1)
                if _is_plausible_fnr(digits):
                    findings.append(PiiFinding(
                        category="personnummer",
                        matched_text=digits,
                        redacted_text=_redact_fnr(digits),
                        line_number=_line_of(m.start()),
                        context=_make_context(text, m.start(), m.end()),
                    ))

        if self.redact_emails:
            for m in _EMAIL_PATTERN.finditer(text):
                email = m.group(0)
                domain = email.split('@', 1)[1].lower()
                if email.lower() in self.email_allowlist:
                    continue
                if domain in self.email_domain_allowlist:
                    continue
                findings.append(PiiFinding(
                    category="email",
                    matched_text=email,
                    redacted_text="<redacted-email>",
                    line_number=_line_of(m.start()),
                    context=_make_context(text, m.start(), m.end()),
                ))

        if self.redact_passwords:
            for m in _PASSWORD_PATTERN.finditer(text):
                password_value = m.group(1)
                findings.append(PiiFinding(
                    category="password",
                    matched_text=m.group(0),
                    redacted_text=m.group(0).replace(password_value, "<redacted-password>"),
                    line_number=_line_of(m.start()),
                    context=_make_context(text, m.start(), m.end()),
                ))

        return findings

    def _apply_redactions(self, text: str) -> str:
        """Apply all redactions to text."""
        if self.redact_passwords:
            def _replace_password(m: re.Match) -> str:
                password_value = m.group(1)
                return m.group(0).replace(password_value, "<redacted-password>")
            text = _PASSWORD_PATTERN.sub(_replace_password, text)

        if self.redact_personnummer:
            def _replace_fnr(m: re.Match) -> str:
                digits = m.group(1)
                if _is_plausible_fnr(digits):
                    return _redact_fnr(digits)
                return digits
            text = _FNR_PATTERN.sub(_replace_fnr, text)

        if self.redact_emails:
            def _replace_email(m: re.Match) -> str:
                email = m.group(0)
                domain = email.split('@', 1)[1].lower()
                if email.lower() in self.email_allowlist:
                    return email
                if domain in self.email_domain_allowlist:
                    return email
                return "<redacted-email>"
            text = _EMAIL_PATTERN.sub(_replace_email, text)

        return text
