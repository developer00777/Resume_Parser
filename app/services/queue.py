"""
Redis-backed batch queue — Single Responsibility: own every Redis key this app writes.

Why this exists
---------------
The synchronous bulk endpoints have to answer inside Salesforce's 120-second
callout ceiling. A 15-file batch routinely needs longer than that, so the whole
request used to die on a 110-second wall clock and every already-parsed resume
in it was thrown away (HTTP 504).

Here the HTTP request only *accepts* work: file bytes go into Redis, one arq job
per file goes onto the queue, and the caller gets a batch_id in well under a
second. Workers drain the queue at their own pace and write each result back as
it lands, so one slow or broken resume can no longer sink the batch.

Key layout (all binary-safe — file bytes are stored raw):

    rp:batch:{id}            HASH    mode, total, remaining, running, timestamps,
                                     callback_url, callback_token
    rp:batch:{id}:files      HASH    index -> {"filename": ..., "content_type": ...}
    rp:batch:{id}:blob:{i}   STRING  raw uploaded bytes for file i
    rp:batch:{id}:results    HASH    index -> serialised BatchResultItem
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from app.config import settings

logger = logging.getLogger(__name__)

PARSE_TASK = "parse_file"

# Where a queued file's bytes come from.
SOURCE_INLINE = "inline"          # uploaded with the request, stashed in Redis
SOURCE_ATTACHMENT = "attachment"  # ContentVersion / Attachment id, fetched by the worker
SOURCE_CANDIDATE = "candidate"    # Candidate record id, resume looked up from the record
SOURCE_URL = "url"                # direct URL, Salesforce-hosted or external
REFERENCE_SOURCES = (SOURCE_ATTACHMENT, SOURCE_CANDIDATE, SOURCE_URL)

_KEY_PREFIX = "rp:batch"

# Module-level pool, created once at app startup and reused by every request.
_pool: ArqRedis | None = None


# ── Keys ──────────────────────────────────────────────────────────────────────

def _k_batch(batch_id: str) -> str:
    return f"{_KEY_PREFIX}:{batch_id}"


def _k_files(batch_id: str) -> str:
    return f"{_KEY_PREFIX}:{batch_id}:files"


def _k_blob(batch_id: str, index: int) -> str:
    return f"{_KEY_PREFIX}:{batch_id}:blob:{index}"


def _k_results(batch_id: str) -> str:
    return f"{_KEY_PREFIX}:{batch_id}:results"


# ── Connection ────────────────────────────────────────────────────────────────

def redis_settings() -> RedisSettings:
    return RedisSettings.from_dsn(settings.redis_url)


async def init_pool() -> ArqRedis | None:
    """
    Open the shared pool. Never raises — the synchronous endpoints must keep
    serving even when Redis is down; only the queue endpoints degrade.
    """
    global _pool
    if _pool is not None:
        return _pool
    try:
        _pool = await create_pool(redis_settings())
        await _pool.ping()
        logger.info("Redis queue connected (queue=%s)", settings.queue_name)
    except Exception as exc:
        _pool = None
        logger.warning("Redis unavailable — async job endpoints disabled: %s", exc)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        try:
            await _pool.aclose()
        except Exception as exc:  # shutdown must not fail on a half-open socket
            logger.debug("Ignoring error while closing the Redis pool: %s", exc)
        _pool = None


def get_pool() -> ArqRedis | None:
    return _pool


async def health() -> dict:
    """Queue health for /health. Reports depth so a backlog is visible."""
    pool = _pool
    if pool is None:
        return {"connected": False, "error": "Redis pool not initialised."}
    try:
        depth = await pool.zcard(settings.queue_name)
        return {"connected": True, "queued_jobs": int(depth)}
    except Exception as exc:
        return {"connected": False, "error": str(exc)}


# ── Small helpers ─────────────────────────────────────────────────────────────

def _s(value: Any) -> str:
    """Decode a Redis reply that may come back as bytes."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return "" if value is None else str(value)


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(_s(value))
    except (TypeError, ValueError):
        return default


def _i(value: Any, default: int = 0) -> int:
    try:
        return int(float(_s(value)))
    except (TypeError, ValueError):
        return default


def _field(info: dict, name: str) -> Any:
    """Read a hash field whether the client decoded keys or not."""
    if name in info:
        return info[name]
    return info.get(name.encode())


def callback_url_allowed(url: str) -> bool:
    """
    Guard the completion callback against being pointed at arbitrary hosts.

    Empty allowlist means "any host" (the submit endpoint is already API-key
    protected); set JOB_CALLBACK_ALLOWED_HOSTS to lock it to your Salesforce org.
    """
    allowed = [h.strip().lower() for h in settings.job_callback_allowed_hosts.split(",") if h.strip()]
    if not allowed:
        return True
    host = (urlparse(url).hostname or "").lower()
    return any(host == a or host.endswith("." + a) for a in allowed)


# ── Batch lifecycle ───────────────────────────────────────────────────────────

async def _create(
    pool: ArqRedis,
    mode: str,
    entries: list[dict],
    callback_url: str | None,
    callback_token: str | None,
) -> tuple[str, int]:
    """
    Persist a batch and enqueue one job per file.

    An entry is either inline (bytes uploaded with the request, stashed in a blob
    key) or a reference (a Salesforce record the worker downloads for itself).
    Returns (batch_id, total). One job per *file* — not per batch — is what keeps
    a slow resume from blocking the rest and lets results stream back as they land.
    """
    batch_id = uuid.uuid4().hex
    total = len(entries)
    now = time.time()
    ttl = settings.job_ttl

    pipe = pool.pipeline()
    pipe.hset(
        _k_batch(batch_id),
        mapping={
            "mode": mode,
            "total": total,
            "remaining": total,
            "running": 0,
            "created_at": repr(now),
            "updated_at": repr(now),
            "callback_url": callback_url or "",
            "callback_token": callback_token or "",
        },
    )
    pipe.expire(_k_batch(batch_id), ttl)

    meta: dict[str, str] = {}
    for index, entry in enumerate(entries):
        content = entry.pop("content", None)
        meta[str(index)] = json.dumps(entry)
        if content is not None:
            pipe.set(_k_blob(batch_id, index), content, ex=ttl)
    if meta:
        pipe.hset(_k_files(batch_id), mapping=meta)
        pipe.expire(_k_files(batch_id), ttl)

    await pipe.execute()

    for index in range(total):
        await pool.enqueue_job(
            PARSE_TASK,
            batch_id,
            index,
            _queue_name=settings.queue_name,
        )

    logger.info("Batch %s queued: %d file(s), mode=%s", batch_id, total, mode)
    return batch_id, total


async def create_batch(
    pool: ArqRedis,
    mode: str,
    files: Iterable[tuple[str, str, bytes]],
    callback_url: str | None = None,
    callback_token: str | None = None,
) -> tuple[str, int]:
    """
    Queue files whose bytes arrived with the request.

    Each item is (filename, content_type, content) with an optional fourth
    element carrying the caller's external_id.
    """
    entries = [
        {"filename": item[0], "content_type": item[1], "source": SOURCE_INLINE,
         "external_id": item[3] if len(item) > 3 else None, "content": item[2]}
        for item in files
    ]
    return await _create(pool, mode, entries, callback_url, callback_token)


async def create_reference_batch(
    pool: ArqRedis,
    mode: str,
    references: Iterable[tuple[str, str, str | None]],
    callback_url: str | None = None,
    callback_token: str | None = None,
) -> tuple[str, int]:
    """
    Queue files the worker will download from Salesforce itself.

    References are (kind, ref, filename) with an optional fourth element carrying
    the caller's external_id. Apex sends record ids rather than file bytes, so
    the callout payload is a few KB no matter how large the resumes are — which
    sidesteps the 6 MB/12 MB Apex heap ceiling entirely.
    """
    entries = [
        {"filename": item[2] or f"{item[1]}.pdf", "content_type": "",
         "source": item[0], "ref": item[1],
         "external_id": item[3] if len(item) > 3 else None}
        for item in references
    ]
    return await _create(pool, mode, entries, callback_url, callback_token)


async def load_file(pool: ArqRedis, batch_id: str, index: int) -> dict | None:
    """
    Describe one queued file for the worker.

    Returns a dict with filename/content_type/source, plus `content` for inline
    entries or `ref` for ones the worker must fetch. None once the batch has
    expired out of Redis.
    """
    raw_meta = await pool.hget(_k_files(batch_id), str(index))
    if raw_meta is None:
        return None
    entry = json.loads(_s(raw_meta))
    entry.setdefault("source", SOURCE_INLINE)
    if entry["source"] == SOURCE_INLINE:
        content = await pool.get(_k_blob(batch_id, index))
        if content is None:
            return None
        entry["content"] = content
    return entry


async def batch_mode(pool: ArqRedis, batch_id: str) -> str:
    info = await pool.hgetall(_k_batch(batch_id))
    return _s(_field(info or {}, "mode")) or "generic"


async def mark_running(pool: ArqRedis, batch_id: str, delta: int) -> None:
    pipe = pool.pipeline()
    pipe.hincrby(_k_batch(batch_id), "running", delta)
    pipe.hset(_k_batch(batch_id), "updated_at", repr(time.time()))
    await pipe.execute()


async def record_result(
    pool: ArqRedis,
    batch_id: str,
    index: int,
    item: dict,
    was_running: bool = True,
) -> int:
    """
    Store one file's outcome and drop its bytes.

    Returns the number of files still outstanding. HINCRBY is atomic, so exactly
    one worker ever observes 0 — that worker owns the completion callback.
    """
    pipe = pool.pipeline()
    pipe.hset(_k_results(batch_id), str(index), json.dumps(item))
    pipe.expire(_k_results(batch_id), settings.job_ttl)
    pipe.delete(_k_blob(batch_id, index))
    if was_running:
        pipe.hincrby(_k_batch(batch_id), "running", -1)
    pipe.hset(_k_batch(batch_id), "updated_at", repr(time.time()))
    pipe.hincrby(_k_batch(batch_id), "remaining", -1)
    replies = await pipe.execute()
    remaining = int(replies[-1])

    if remaining <= 0:
        await pool.hset(_k_batch(batch_id), "finished_at", repr(time.time()))
    return remaining


async def claim_callback(pool: ArqRedis, batch_id: str) -> tuple[str, str] | None:
    """
    Atomically claim the right to fire this batch's completion callback.

    HSETNX means a retry or a second worker can never double-POST the results.
    """
    info = await pool.hgetall(_k_batch(batch_id))
    if not info:
        return None
    url = _s(_field(info, "callback_url"))
    if not url:
        return None
    claimed = await pool.hsetnx(_k_batch(batch_id), "callback_state", "sent")
    if not claimed:
        return None
    return url, _s(_field(info, "callback_token"))


async def get_batch(pool: ArqRedis, batch_id: str) -> dict | None:
    """
    Assemble the caller-facing view of a batch.

    Self-healing: a batch with nothing running and no progress for
    JOB_STALE_SECONDS has its unfinished files marked failed, so a poller always
    reaches a terminal state even if a worker was killed mid-job.
    """
    raw_info = await pool.hgetall(_k_batch(batch_id))
    if not raw_info:
        return None

    info = {_s(k): v for k, v in raw_info.items()}
    mode = _s(info.get("mode")) or "generic"
    total = _i(info.get("total"))
    remaining = _i(info.get("remaining"))
    running = _i(info.get("running"))
    created_at = _f(info.get("created_at"))
    updated_at = _f(info.get("updated_at"), created_at)
    finished_at = _f(info.get("finished_at")) or None

    raw_results = await pool.hgetall(_k_results(batch_id))
    results: list[dict] = []
    for field, payload in (raw_results or {}).items():
        item = json.loads(_s(payload))
        item.setdefault("index", _i(field))
        results.append(item)

    done_indexes = {item["index"] for item in results}

    # Reap a batch that stalled (worker OOM-killed, container replaced, ...).
    stale = (
        remaining > 0
        and running <= 0
        and (time.time() - updated_at) > settings.job_stale_seconds
    )
    if stale:
        file_meta = await pool.hgetall(_k_files(batch_id))
        reaped = 0
        for field, payload in (file_meta or {}).items():
            index = _i(field)
            if index in done_indexes:
                continue
            meta = json.loads(_s(payload))
            item = {
                "index": index,
                "filename": meta.get("filename", "unknown"),
                "external_id": meta.get("external_id"),
                "success": False,
                "data": None,
                "error": (
                    f"No worker reported this file within {settings.job_stale_seconds}s. "
                    "Resubmit it."
                ),
                "processing_time_ms": 0.0,
            }
            results.append(item)
            reaped += 1
            await record_result(pool, batch_id, index, item, was_running=False)
        remaining = 0
        finished_at = finished_at or time.time()
        logger.warning("Batch %s reaped as stale; %d file(s) marked failed", batch_id, reaped)

    results.sort(key=lambda r: r["index"])
    parsed = sum(1 for r in results if r.get("success"))
    completed = len(results)

    if remaining <= 0:
        status = "completed"
    elif running > 0 or completed > 0:
        status = "processing"
    else:
        status = "queued"

    end = finished_at if finished_at else time.time()
    return {
        "batch_id": batch_id,
        "status": status,
        "mode": mode,
        "total": total,
        "completed": completed,
        "parsed": parsed,
        "failed": completed - parsed,
        "remaining": max(remaining, 0),
        "running": max(running, 0),
        "created_at": created_at,
        "updated_at": updated_at,
        "finished_at": finished_at,
        "total_processing_time_ms": round((end - created_at) * 1000, 2),
        "results": results,
    }


async def delete_batch(pool: ArqRedis, batch_id: str) -> bool:
    """Drop a batch and every blob it still holds."""
    info = await pool.hgetall(_k_batch(batch_id))
    if not info:
        return False
    total = _i(_field({_s(k): v for k, v in info.items()}, "total"))
    keys = [_k_batch(batch_id), _k_files(batch_id), _k_results(batch_id)]
    keys += [_k_blob(batch_id, i) for i in range(total)]
    await pool.delete(*keys)
    return True
