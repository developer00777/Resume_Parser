"""
Coercion of parsed resume values into shapes Salesforce will accept.

Everything the LLM emits is a string. Salesforce rejects a string where the
field wants a Date, a Number, or a restricted picklist value — and it rejects
the *whole record*, so one unparseable date loses every other field with it.
Values are therefore coerced here, before they reach the org, and anything that
cannot be coerced becomes None rather than a guess.

Field lengths and picklist value sets are org-specific: they come from a
Describe of SCSCHAMPS__Candidate__c, not from this file. FIELD_SPECS is where
that mapping lands, populated by load_specs_from_describe(). Until it has run,
text passes through untruncated and picklist values pass through unchanged, and
write-back refuses to execute at all.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date

logger = logging.getLogger(__name__)

# Resume dates outside this range are data errors, not biographies: a 1900 date
# of birth or a 2099 graduation is an extraction artefact. Salesforce would
# happily store either.
_MIN_YEAR = 1920
_MAX_YEAR = date.today().year + 10

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}


@dataclass
class FieldSpec:
    """
    What the org says about one field. Populated from a Describe call, not by
    hand.
    """
    api_name: str
    kind: str = "text"            # text | date | number | picklist | boolean
    length: int | None = None     # Salesforce `length` for text fields
    values: tuple[str, ...] = ()  # picklist value set, empty if unrestricted


# ---------------------------------------------------------------------------
# Populated from the target org by load_specs_from_describe(). Empty means
# "unverified": coercion degrades to pass-through rather than inventing a
# constraint that might not exist, and write-back refuses to run.
# ---------------------------------------------------------------------------
FIELD_SPECS: dict[str, FieldSpec] = {}


def spec_for(logical_name: str) -> FieldSpec | None:
    return FIELD_SPECS.get(logical_name)


# Salesforce type -> how we coerce it.
_KIND_BY_SF_TYPE = {
    "date": "date", "datetime": "date",
    "double": "number", "currency": "number", "percent": "number",
    "int": "number", "long": "number",
    "picklist": "picklist", "multipicklist": "picklist",
    "boolean": "boolean",
    # Salesforce validates the format of these server-side and rejects the whole
    # record when it does not match — so they need coercing, not just clipping.
    "email": "email", "phone": "phone", "url": "url",
    "string": "text", "textarea": "text",
}


# Deliberately permissive: this exists to catch extraction noise ("email: N/A",
# a name that landed in the email field, a trailing comma), not to adjudicate
# RFC 5322. Salesforce's own check is roughly this strict.
_EMAIL_RE = re.compile(r"^[^@\s,;]+@[^@\s,;]+\.[A-Za-z]{2,}$")


def email_or_none(value: object) -> str | None:
    """
    Return the value only if Salesforce would accept it as an Email.

    An Email field rejects a malformed address with INVALID_EMAIL_ADDRESS and
    takes the entire record with it, so one bad extraction would lose every
    other field on the candidate.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    # Resumes often list two addresses in one line.
    first = re.split(r"[,;\s]+", text)[0].strip().strip(".<>()")
    if _EMAIL_RE.match(first):
        return first

    logger.info("email_or_none: %r is not a usable address — storing null", text[:40])
    return None


@dataclass
class DescribeLoadReport:
    """What a Describe resolved, and what it could not."""
    sobject: str = ""
    resolved: dict[str, str] = field(default_factory=dict)   # logical -> API name
    unresolved: list[str] = field(default_factory=list)      # no such field in the org
    not_updateable: list[str] = field(default_factory=list)  # exists but read-only

    @property
    def field_count(self) -> int:
        return len(self.resolved)


def _name_candidates(logical_name: str, namespace: str) -> list[str]:
    """
    The API names a logical field might have in a given org, most-likely first.

    The same logical field is `SCSCHAMPS__LinkedIn_URL__c` in a namespaced
    managed-package org and `LinkedIn_URL__c` where it was added locally. Rather
    than guessing, every plausible form is tried against the real Describe and
    the one that exists wins.
    """
    ns = f"{namespace}__" if namespace else ""
    return [
        f"{ns}{logical_name}__c",   # namespaced custom field
        f"{logical_name}__c",       # org-local custom field
        logical_name,               # standard field (FirstName, Email, ...)
        f"{ns}{logical_name}",
    ]


def load_specs_from_describe(
    describe: dict,
    logical_names: list[str],
    namespace: str = "SCSCHAMPS",
) -> DescribeLoadReport:
    """
    Populate FIELD_SPECS from an sObject Describe response.

    Resolves each logical name to a real API name, records the field's type,
    length and picklist value set, and reports anything that does not exist or
    is not updateable. Replaces FIELD_SPECS wholesale so a reload cannot leave
    stale entries behind.
    """
    by_name = {
        f.get("name", "").casefold(): f
        for f in describe.get("fields", [])
        if f.get("name")
    }
    report = DescribeLoadReport(sobject=describe.get("name", ""))
    specs: dict[str, FieldSpec] = {}

    for logical in logical_names:
        found = next(
            (
                by_name[c.casefold()]
                for c in _name_candidates(logical, namespace)
                if c.casefold() in by_name
            ),
            None,
        )
        if found is None:
            report.unresolved.append(logical)
            continue

        api_name = found["name"]
        if not found.get("updateable", False):
            # Formula, rollup or system field — writing it fails the record.
            report.not_updateable.append(f"{logical} -> {api_name}")
            continue

        sf_type = (found.get("type") or "").casefold()
        values: tuple[str, ...] = ()
        if _KIND_BY_SF_TYPE.get(sf_type) == "picklist" and found.get("restrictedPicklist"):
            # Only a *restricted* picklist rejects unknown values. An
            # unrestricted one accepts anything, so an empty value set here
            # correctly means "pass through".
            values = tuple(
                v["value"] for v in found.get("picklistValues", [])
                if v.get("active", True) and v.get("value")
            )

        specs[logical] = FieldSpec(
            api_name=api_name,
            kind=_KIND_BY_SF_TYPE.get(sf_type, "text"),
            length=found.get("length") or None,
            values=values,
        )
        report.resolved[logical] = api_name

    FIELD_SPECS.clear()
    FIELD_SPECS.update(specs)

    logger.info(
        "Describe loaded for %s: %d fields resolved, %d unresolved, %d read-only",
        report.sobject or "(unknown)",
        len(report.resolved), len(report.unresolved), len(report.not_updateable),
    )
    if report.unresolved:
        logger.warning("Describe: no such field in org — %s", ", ".join(report.unresolved))
    return report


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

def iso_date(value: object) -> str | None:
    """
    Normalise a date to `yyyy-MM-dd`, the only form a Salesforce Date field
    accepts over the REST API. Returns None for anything unrecognised —
    including plausible-looking junk — because a wrong date silently recorded is
    worse than an empty one.

    Numeric day/month order is read as day-first (12/05/1990 is 12 May), which
    is correct for the Indian resumes this parser targets and wrong for US ones.
    Only pairs where both parts are <= 12 are ambiguous; the order is asserted
    here rather than guessed per-value.
    """
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    parsed = _try_iso(raw) or _try_numeric(raw) or _try_written_month(raw)
    if parsed is None:
        logger.info("iso_date: could not parse %r — storing null", raw[:40])
        return None

    y, m, d = parsed
    if not (_MIN_YEAR <= y <= _MAX_YEAR):
        logger.info("iso_date: year %d out of range for %r — storing null", y, raw[:40])
        return None
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        # e.g. 31 February — a real extraction error, not a date.
        logger.info("iso_date: %r is not a real calendar date — storing null", raw[:40])
        return None


def _try_iso(raw: str) -> tuple[int, int, int] | None:
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", raw)
    return (int(m[1]), int(m[2]), int(m[3])) if m else None


def _try_numeric(raw: str) -> tuple[int, int, int] | None:
    """dd/mm/yyyy, dd-mm-yyyy, dd.mm.yyyy — and the 2-digit-year variants."""
    m = re.match(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2}|\d{4})$", raw)
    if not m:
        return None
    d, mo, y = int(m[1]), int(m[2]), int(m[3])
    if y < 100:
        # A 2-digit year on a resume is a birth date or an old qualification,
        # never a future one.
        y += 1900 if y > (date.today().year % 100) else 2000
    if mo > 12:
        # Unambiguously month-first (e.g. 05/22/1990) — swap rather than reject.
        d, mo = mo, d
    return (y, mo, d)


def _try_written_month(raw: str) -> tuple[int, int, int] | None:
    """'12 May 1990', '12th May 1990', 'May 12, 1990'."""
    cleaned = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", raw, flags=re.IGNORECASE)
    cleaned = cleaned.replace(",", " ").strip()

    m = re.match(r"^(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})$", cleaned)
    if m and m[2].lower() in _MONTHS:
        return (int(m[3]), _MONTHS[m[2].lower()], int(m[1]))

    m = re.match(r"^([A-Za-z]+)\s+(\d{1,2})\s+(\d{4})$", cleaned)
    if m and m[1].lower() in _MONTHS:
        return (int(m[3]), _MONTHS[m[1].lower()], int(m[2]))

    # 'May 1990' — no day stated. Salesforce Date needs one, and picking the 1st
    # would be a fabrication, so this is rejected.
    return None


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

def to_number(value: object) -> float | None:
    """
    Pull a number out of a compensation or duration string for a Salesforce
    Currency/Number field. Understands the Indian units resumes use.

    '12 LPA' -> 1200000.0 · '₹15,00,000' -> 1500000.0 · '2.5 Cr' -> 25000000.0
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    raw = str(value).strip()
    if not raw:
        return None

    cleaned = re.sub(r"[₹$,\s]", "", raw.upper())
    m = re.search(r"(\d+(?:\.\d+)?)", cleaned)
    if not m:
        return None

    amount = float(m.group(1))
    if "CR" in cleaned:
        amount *= 10_000_000
    elif any(unit in cleaned for unit in ("LPA", "LAKH", "LAC", "L")):
        amount *= 100_000
    elif "K" in cleaned:
        amount *= 1_000
    return amount


# ---------------------------------------------------------------------------
# Text and picklists
# ---------------------------------------------------------------------------

def truncate(value: object, logical_name: str) -> str | None:
    """
    Clip text to the field's declared length. A value one character over its
    limit fails the whole DML with STRING_TOO_LONG, so this is not cosmetic.

    With no spec for the field, the value passes through — the length is
    unverified, not unlimited.
    """
    if value is None:
        return None
    text = str(value)
    if not text:
        return None

    spec = spec_for(logical_name)
    if spec is None or spec.length is None:
        return text
    if len(text) <= spec.length:
        return text

    logger.info(
        "truncate: %s is %d chars, field allows %d — clipping",
        logical_name, len(text), spec.length,
    )
    return text[: spec.length]


def pick(value: object, logical_name: str) -> str | None:
    """
    Snap a value onto the field's picklist value set. A restricted picklist
    rejects anything outside its set with INVALID_OR_NULL_FOR_RESTRICTED_PICKLIST
    and takes the whole record with it, so an unmatched value becomes None.

    Matching is case- and space-insensitive. With no spec, the value passes
    through unchanged.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    spec = spec_for(logical_name)
    if spec is None or not spec.values:
        return text

    normalised = text.casefold().replace(" ", "")
    for allowed in spec.values:
        if allowed.casefold().replace(" ", "") == normalised:
            return allowed

    logger.info(
        "pick: %r is not a value of %s (%s) — storing null",
        text[:40], logical_name, ", ".join(spec.values[:6]),
    )
    return None


def money_or_text(value: object, logical_name: str) -> float | str | None:
    """
    Compensation fields differ between orgs: some are Currency, some are plain
    text holding "12 LPA". Emitting a number into a Text field is harmless;
    emitting "12 LPA" into a Currency field fails the record.

    So: parse to a number when the spec says the field is numeric, and pass the
    original string through otherwise.
    """
    if value is None:
        return None
    spec = spec_for(logical_name)
    if spec is not None and spec.kind == "number":
        return to_number(value)
    text = str(value).strip()
    return text or None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

@dataclass
class CoerceReport:
    """
    What coercion could and could not verify. Lets a caller tell an empty field
    apart from an unchecked one.
    """
    describe_loaded: bool = False
    unverified_fields: list[str] = field(default_factory=list)

    @property
    def safe_to_write(self) -> bool:
        """
        True only when the payload was checked against a real org schema.
        Writing without a Describe means field names, lengths and picklist sets
        are all assumptions.
        """
        return self.describe_loaded


def coerce_report(logical_names: list[str]) -> CoerceReport:
    loaded = bool(FIELD_SPECS)
    return CoerceReport(
        describe_loaded=loaded,
        unverified_fields=[] if loaded else sorted(set(logical_names)),
    )
