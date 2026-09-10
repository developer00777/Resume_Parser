"""
arq worker — Single Responsibility: parse one queued resume and report the result.

Run it alongside the API (a second container / Railway service):

    arq app.worker.WorkerSettings

Each job is a single file, so the unit of retry, timeout and failure is one
resume rather than a whole batch. Nothing here answers HTTP, and nothing here
can time a caller out: the worker owns its own budget and always writes a result
row, success or failure, so a poller reaches a terminal state.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import ClassVar

import httpx
from arq import Retry
from fastapi import HTTPException

from app.config import settings
from app.schemas.response import map_to_client, map_to_generic, map_to_salesforce
from app.services import queue
from app.services.document import validate_file_bytes
from app.services.llm import parse_resume

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# HTTP statuses worth another attempt — upstream hiccups, not bad input.
_TRANSIENT_STATUS = {429, 500, 502, 503, 504}

# Leave room inside the arq job timeout so our own handler runs and records the
# failure, rather than arq cancelling us and leaving the batch short a result.
_WORK_TIMEOUT_MARGIN = 15


def _map(mode: str, parsed: dict, text: str) -> dict:
    if mode == "salesforce":
        return map_to_salesforce(parsed, raw_text=text).model_dump()
    if mode == "client":
        return map_to_client(parsed).model_dump()
    return map_to_generic(parsed, resume_text=text).model_dump()


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, HTTPException):
        return exc.status_code in _TRANSIENT_STATUS
    return isinstance(
        exc,
        (asyncio.TimeoutError, httpx.TimeoutException, httpx.ConnectError, ConnectionError),
    )


def _error_text(exc: BaseException) -> str:
    if isinstance(exc, HTTPException):
        return str(exc.detail)
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


def _content_type_for(filename: str) -> str:
    return (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        if filename.lower().endswith(".docx")
        else "application/pdf"
    )


async def _resolve(entry: dict) -> tuple[bytes, str]:
    """
    Produce the file's bytes, downloading from Salesforce when the batch was
    submitted as record ids rather than uploads.

    Pulling server-side is what lets Apex send a few KB of ids instead of
    base64-encoded PDFs, which no 6 MB heap could carry in bulk anyway.
    """
    source = entry.get("source", queue.SOURCE_INLINE)

    if source == queue.SOURCE_INLINE:
        content = entry["content"]
        content_type = entry.get("content_type") or _content_type_for(entry.get("filename", ""))
        return content, content_type

    from app.services.salesforce import (
        fetch_resume_by_attachment_id,
        fetch_resume_by_url,
        fetch_resume_from_candidate,
    )

    fetchers = {
        queue.SOURCE_ATTACHMENT: fetch_resume_by_attachment_id,
        queue.SOURCE_CANDIDATE: fetch_resume_from_candidate,
        queue.SOURCE_URL: fetch_resume_by_url,
    }
    fetch = fetchers.get(source)
    if fetch is None:
        raise HTTPException(status_code=400, detail=f"Unknown file source '{source}'.")

    content, filename = await fetch(entry["ref"])
    # Salesforce knows the real filename; prefer it over the placeholder.
    entry["filename"] = filename or entry.get("filename")
    return content, _content_type_for(entry.get("filename") or "")


async def parse_file(ctx: dict, batch_id: str, index: int) -> None:
    """Parse one file from a queued batch and write its result row."""
    redis = ctx["redis"]
    attempt = ctx.get("job_try", 1)
    start = time.time()

    entry = await queue.load_file(redis, batch_id, index)
    if entry is None:
        # Batch expired out of Redis, or was deleted while queued. Nothing to
        # report to — dropping the job is the correct outcome.
        logger.warning("Batch %s file %d is gone from Redis — dropping job", batch_id, index)
        return

    filename = entry.get("filename") or "unknown"
    mode = await queue.batch_mode(redis, batch_id)

    await queue.mark_running(redis, batch_id, 1)
    try:
        budget = max(settings.job_timeout - _WORK_TIMEOUT_MARGIN, 30)
        content, content_type = await asyncio.wait_for(_resolve(entry), timeout=budget)
        filename = entry.get("filename") or filename
        text = await asyncio.wait_for(
            validate_file_bytes(filename, content, content_type),
            timeout=budget,
        )
        parsed = await asyncio.wait_for(parse_resume(text), timeout=budget)
        item = {
            "index": index,
            "filename": filename,
            "external_id": entry.get("external_id"),
            "success": True,
            "data": _map(mode, parsed, text),
            "error": None,
            "processing_time_ms": round((time.time() - start) * 1000, 2),
        }
    except Exception as exc:
        if attempt < settings.job_max_tries and _is_transient(exc):
            await queue.mark_running(redis, batch_id, -1)
            logger.warning(
                "Batch %s file %d ('%s') attempt %d failed transiently: %s — retrying",
                batch_id, index, filename, attempt, _error_text(exc),
            )
            raise Retry(defer=min(2 ** attempt, 30))

        logger.error(
            "Batch %s file %d ('%s') failed after %d attempt(s): %s",
            batch_id, index, filename, attempt, _error_text(exc),
        )
        item = {
            "index": index,
            "filename": filename,
            "external_id": entry.get("external_id"),
            "success": False,
            "data": None,
            "error": _error_text(exc),
            "processing_time_ms": round((time.time() - start) * 1000, 2),
        }

    remaining = await queue.record_result(redis, batch_id, index, item)
    logger.info(
        "Batch %s file %d ('%s') %s in %.1fs — %d left",
        batch_id, index, filename,
        "parsed" if item["success"] else "failed",
        item["processing_time_ms"] / 1000, max(remaining, 0),
    )

    if remaining <= 0:
        await _fire_callback(redis, batch_id)


async def _fire_callback(redis, batch_id: str) -> None:
    """
    POST the finished batch to the caller's callback_url, if one was supplied.

    Claimed atomically so a retried job can never deliver the batch twice.
    Failures are logged, never raised — the results stay readable by polling.
    """
    claim = await queue.claim_callback(redis, batch_id)
    if claim is None:
        return
    url, token = claim

    if not queue.callback_url_allowed(url):
        logger.error("Batch %s: callback host not in JOB_CALLBACK_ALLOWED_HOSTS — skipped", batch_id)
        return

    payload = await queue.get_batch(redis, batch_id)
    if payload is None:
        return

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        async with httpx.AsyncClient(timeout=settings.job_callback_timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
        logger.info("Batch %s callback -> %s returned %d", batch_id, url, resp.status_code)
    except Exception as exc:
        logger.error("Batch %s callback to %s failed: %s", batch_id, url, exc)


async def startup(ctx: dict) -> None:
    logger.info(
        "Worker up — queue=%s concurrency=%d model=%s",
        settings.queue_name, settings.worker_concurrency, settings.openrouter_model,
    )


async def shutdown(ctx: dict) -> None:
    from app.services.llm_client import extraction_client, ocr_client
    await extraction_client.aclose()
    await ocr_client.aclose()
    logger.info("Worker down")


class WorkerSettings:
    functions: ClassVar[list] = [parse_file]
    redis_settings = queue.redis_settings()
    queue_name = settings.queue_name
    on_startup = startup
    on_shutdown = shutdown
    # Resumes parsed concurrently in this process. Add worker replicas to go wider.
    max_jobs = settings.worker_concurrency
    job_timeout = settings.job_timeout
    # parse_file drives its own retry logic via Retry(); this is the hard ceiling.
    max_tries = settings.job_max_tries
    keep_result = 0  # results live in our own batch hash, not arq's
    health_check_interval = 60
