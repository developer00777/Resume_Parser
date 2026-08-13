"""
Lifecycle for the org's field mapping.

The mapping is derived from a live Describe of the Candidate sObject, which
makes it correct for whatever org the service points at — but it also makes it
*state*, and state that lives only in process memory has three failure modes
this module exists to close:

  1. A restart or redeploy empties it, so write-back starts refusing everything
     until someone remembers to call /describe/reload by hand.
  2. Salesforce being briefly unreachable at boot would poison the process for
     its whole lifetime.
  3. An admin adding a field or a picklist value would never be seen.

So: load at startup, retry lazily on first use if that failed, and refresh once
the mapping is older than a TTL. Concurrent callers share one load rather than
stampeding the org.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from app.config import settings
from app.schemas.response import SalesforceResumeData
from app.services import salesforce_coerce as coerce

logger = logging.getLogger(__name__)

# One in-flight load at a time. Without this, a burst of requests arriving to a
# cold process would each fire their own Describe.
_lock = asyncio.Lock()


@dataclass
class SchemaState:
    """What the process currently knows about the org's schema."""
    loaded_at: datetime | None = None
    sobject: str = ""
    field_count: int = 0
    resolved: dict[str, str] | None = None
    unresolved: list[str] | None = None
    not_updateable: list[str] | None = None
    last_error: str | None = None
    last_attempt_at: datetime | None = None

    @property
    def loaded(self) -> bool:
        return self.loaded_at is not None and bool(coerce.FIELD_SPECS)

    @property
    def age_seconds(self) -> float | None:
        if self.loaded_at is None:
            return None
        return (datetime.now(timezone.utc) - self.loaded_at).total_seconds()

    @property
    def stale(self) -> bool:
        ttl = settings.sf_describe_ttl_minutes
        if ttl <= 0 or self.loaded_at is None:
            return False
        return (self.age_seconds or 0) > ttl * 60

    def summary(self) -> dict:
        return {
            "loaded": self.loaded,
            "sobject": self.sobject,
            "fields_resolved": self.field_count,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "age_seconds": round(self.age_seconds) if self.age_seconds is not None else None,
            "stale": self.stale,
            "ttl_minutes": settings.sf_describe_ttl_minutes,
            "unresolved": self.unresolved or [],
            "not_updateable": self.not_updateable or [],
            "last_error": self.last_error,
            "last_attempt_at": (
                self.last_attempt_at.isoformat() if self.last_attempt_at else None
            ),
        }


_state = SchemaState()


def state() -> SchemaState:
    return _state


def configured() -> bool:
    """Whether Salesforce credentials exist at all."""
    return bool(settings.sf_client_id and settings.sf_client_secret)


def logical_field_names() -> list[str]:
    """Every field the payload can carry — what we ask the org to resolve."""
    return list(SalesforceResumeData.model_fields.keys())


async def ensure_loaded(force: bool = False) -> SchemaState:
    """
    Guarantee the mapping is loaded and reasonably fresh.

    Returns the current state either way — callers decide what to do when it did
    not load, since refusing to write is the correct response in some paths and
    merely worth reporting in others. Never raises: a Describe failure must not
    turn into a 500 on an unrelated request.
    """
    if not force and _state.loaded and not _state.stale:
        return _state

    if not configured():
        _state.last_error = "Salesforce credentials are not configured."
        return _state

    async with _lock:
        # Re-check inside the lock: another caller may have just loaded it.
        if not force and _state.loaded and not _state.stale:
            return _state
        await _load()

    return _state


async def _load() -> None:
    """Fetch the Describe and rebuild FIELD_SPECS. Records failure, never raises."""
    # Imported here rather than at module scope to keep the import graph
    # acyclic — salesforce.py has no reason to know about this module.
    from app.services.salesforce import fetch_candidate_describe

    _state.last_attempt_at = datetime.now(timezone.utc)
    try:
        describe = await fetch_candidate_describe()
        report = coerce.load_specs_from_describe(describe, logical_field_names())
    except Exception as exc:
        # Keep whatever mapping we already had: a stale mapping that worked is
        # strictly better than none, and the TTL will try again shortly.
        _state.last_error = f"{type(exc).__name__}: {exc}"
        logger.warning("Describe load failed: %s", _state.last_error)
        return

    _state.loaded_at = datetime.now(timezone.utc)
    _state.sobject = report.sobject
    _state.field_count = report.field_count
    _state.resolved = report.resolved
    _state.unresolved = report.unresolved
    _state.not_updateable = report.not_updateable
    _state.last_error = None

    logger.info(
        "Field mapping ready: %d field(s) resolved on %s (%d unresolved, %d read-only)",
        report.field_count, report.sobject or "(unknown)",
        len(report.unresolved), len(report.not_updateable),
    )


async def load_at_startup() -> None:
    """
    Warm the mapping as the process boots.

    Deliberately non-fatal. A parser that starts without Salesforce reachable is
    still fully useful for the upload endpoints, and ensure_loaded() will retry
    on the first write-back request.
    """
    if not configured():
        logger.info(
            "Salesforce credentials not set — field mapping not loaded. "
            "Write-back will be unavailable; upload endpoints are unaffected."
        )
        return

    await ensure_loaded(force=True)
    if not _state.loaded:
        logger.warning(
            "Could not load the field mapping at startup (%s). Write-back will "
            "retry on first use.",
            _state.last_error,
        )


def reset_for_tests() -> None:
    """Clear both the mapping and its state. Test helper only."""
    global _state
    _state = SchemaState()
    coerce.FIELD_SPECS.clear()
