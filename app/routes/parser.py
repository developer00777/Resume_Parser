import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings
from app.schemas.response import (
    BulkClientParseItem,
    BulkClientParseResponse,
    BulkJobStatus,
    BulkParseItem,
    BulkParseResponse,
    BulkSalesforceParseItem,
    BulkSalesforceParseResponse,
    ModelInfo,
    ModelsResponse,
    map_to_client,
    map_to_generic,
    map_to_salesforce,
)
from app.services.document import extract_text
from app.services.llm import parse_resume

logger = logging.getLogger(__name__)

# Historical name kept so nothing that imports it breaks; the mapper itself now
# lives in schemas/response.py so the queue worker can share it.
_to_resume_data = map_to_generic

router = APIRouter(prefix="/api/v1", tags=["parser"])


def _async_hint(mode: str) -> str:
    suffix = "" if mode == "generic" else f"/{mode}"
    return f"POST /api/v1/parse{suffix}/jobs"


def _budget_error(mode: str) -> str:
    return (
        f"Timed out after {settings.bulk_timeout:.0f}s — this batch needs longer than a "
        f"synchronous request can wait. Submit it to the queue instead ({_async_hint(mode)}) "
        f"and poll GET /api/v1/parse/jobs/{{batch_id}}."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _parse_one(file: UploadFile, semaphore: asyncio.Semaphore) -> BulkParseItem:
    """Parse a single resume file under a concurrency semaphore (never raises)."""
    start = time.time()
    filename = file.filename or "unknown"
    try:
        async with semaphore:
            text = await extract_text(file)
            parsed = await parse_resume(text)
        elapsed_ms = round((time.time() - start) * 1000, 2)
        return BulkParseItem(
            filename=filename,
            success=True,
            data=map_to_generic(parsed, resume_text=text),
            processing_time_ms=elapsed_ms,
        )
    except Exception as exc:
        elapsed_ms = round((time.time() - start) * 1000, 2)
        logger.error(f"Failed to parse '{filename}': {exc}")
        return BulkParseItem(
            filename=filename,
            success=False,
            error=str(exc),
            processing_time_ms=elapsed_ms,
        )


async def _parse_one_sf(file: UploadFile, semaphore: asyncio.Semaphore) -> BulkSalesforceParseItem:
    """Parse a single resume and return a Salesforce-mapped item (never raises)."""
    start = time.time()
    filename = file.filename or "unknown"
    try:
        async with semaphore:
            text = await extract_text(file)
            parsed = await parse_resume(text)
        sf_data = map_to_salesforce(parsed, raw_text=text)
        elapsed_ms = round((time.time() - start) * 1000, 2)
        return BulkSalesforceParseItem(
            filename=filename,
            success=True,
            data=sf_data,
            processing_time_ms=elapsed_ms,
        )
    except Exception as exc:
        elapsed_ms = round((time.time() - start) * 1000, 2)
        logger.error(f"Bulk SF: failed to parse '{filename}': {exc}")
        return BulkSalesforceParseItem(
            filename=filename,
            success=False,
            error=str(exc),
            processing_time_ms=elapsed_ms,
        )


async def _parse_one_client(file: UploadFile, semaphore: asyncio.Semaphore) -> BulkClientParseItem:
    """Parse a single resume and return client-1 mapped fields (never raises)."""
    start = time.time()
    filename = file.filename or "unknown"
    try:
        async with semaphore:
            text = await extract_text(file)
            parsed = await parse_resume(text)
        client_data = map_to_client(parsed)
        elapsed_ms = round((time.time() - start) * 1000, 2)
        return BulkClientParseItem(
            filename=filename,
            success=True,
            data=client_data,
            processing_time_ms=elapsed_ms,
        )
    except Exception as exc:
        elapsed_ms = round((time.time() - start) * 1000, 2)
        logger.error(f"Client parse: failed '{filename}': {exc}")
        return BulkClientParseItem(
            filename=filename,
            success=False,
            error=str(exc),
            processing_time_ms=elapsed_ms,
        )


def _validate_files(files: list[UploadFile]) -> None:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(files) > settings.bulk_max_files:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum allowed per request is {settings.bulk_max_files}.",
        )


async def _gather_within_budget(
    coroutines: list[Awaitable],
    files: list[UploadFile],
    on_unfinished: Callable[[str, str], object],
    mode: str,
) -> list:
    """
    Run every parse task and return whatever finished inside the wall-clock budget.

    A synchronous caller (Salesforce caps a callout at 120s) used to get a blanket
    504 the moment the batch ran long, throwing away every resume that had already
    parsed. Now the finished ones come back intact and only the unfinished files
    are marked failed, each carrying a pointer to the async queue endpoint.
    """
    tasks = [asyncio.ensure_future(coro) for coro in coroutines]
    done, pending = await asyncio.wait(tasks, timeout=settings.bulk_timeout)

    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
        logger.warning(
            "Sync %s batch hit the %.0fs budget — returning %d/%d parsed; "
            "the caller should move this workload to the queue.",
            mode, settings.bulk_timeout, len(done), len(tasks),
        )

    results = []
    for task, file in zip(tasks, files):
        filename = file.filename or "unknown"
        if task in done and not task.cancelled():
            exc = task.exception()
            if exc is None:
                results.append(task.result())
                continue
            results.append(on_unfinished(filename, f"{type(exc).__name__}: {exc}"))
        else:
            results.append(on_unfinished(filename, _budget_error(mode)))
    return results


@router.get("/models", response_model=ModelsResponse)
async def get_models():
    """Return the currently configured OpenRouter model."""
    return ModelsResponse(
        models=[ModelInfo(name=settings.openrouter_model)]
    )


# ── Single unified parse endpoint (1–15 files) ───────────────────────────────

@router.post("/parse", response_model=BulkParseResponse)
async def parse(
    files: list[UploadFile] = File(..., description="1 to 15 resume files (PDF or DOCX)"),
):
    """
    Parse 1 to 15 resume files in a single request.

    Send one file for a single resume or up to 15 files for bulk parsing.
    All files are processed concurrently (max 5 at a time).
    Each result is under `results[]` with filename, success flag, and parsed data.

    If the batch outruns the synchronous budget the request still returns 200 with
    the resumes that finished; the rest are reported as failed items. For reliable
    bulk parsing use POST /api/v1/parse/jobs instead.
    """
    _validate_files(files)
    wall_start = time.time()
    semaphore = asyncio.Semaphore(settings.bulk_concurrency)

    results: list[BulkParseItem] = await _gather_within_budget(
        [_parse_one(f, semaphore) for f in files],
        files,
        lambda filename, error: BulkParseItem(filename=filename, success=False, error=error),
        "generic",
    )

    total_ms = round((time.time() - wall_start) * 1000, 2)
    parsed_count = sum(1 for r in results if r.success)

    return BulkParseResponse(
        success=True,
        total=len(results),
        parsed=parsed_count,
        failed=len(results) - parsed_count,
        results=results,
        total_processing_time_ms=total_ms,
    )


@router.post("/parse/salesforce", response_model=BulkSalesforceParseResponse)
async def parse_salesforce(
    files: list[UploadFile] = File(..., description="Up to 15 resume files (PDF or DOCX)"),
):
    """
    Parse up to 15 resume files and return SCSCHAMPS-mapped Salesforce JSON for each.

    Partial results are returned when the batch outruns the synchronous budget, so
    this endpoint no longer answers 504. For bulk loads, submit to
    POST /api/v1/parse/salesforce/jobs and poll — the queue has no wall clock.
    """
    _validate_files(files)
    wall_start = time.time()
    semaphore = asyncio.Semaphore(settings.bulk_concurrency)

    results: list[BulkSalesforceParseItem] = await _gather_within_budget(
        [_parse_one_sf(f, semaphore) for f in files],
        files,
        lambda filename, error: BulkSalesforceParseItem(
            filename=filename, success=False, error=error
        ),
        "salesforce",
    )

    total_ms = round((time.time() - wall_start) * 1000, 2)
    parsed_count = sum(1 for r in results if r.success)

    return BulkSalesforceParseResponse(
        success=True,
        total=len(results),
        parsed=parsed_count,
        failed=len(results) - parsed_count,
        results=results,
        total_processing_time_ms=total_ms,
    )


@router.post("/parse/client", response_model=BulkClientParseResponse)
async def parse_client(
    files: list[UploadFile] = File(..., description="1 to 15 resume files (PDF or DOCX)"),
):
    """
    Parse 1 to 15 resume files and return client-1 mapped JSON for each.

    Output fields match the client's Salesforce org schema:
    Name, FullName__c, Nationality__c, Date_of_Birth__c, Years_of_Experience__c,
    Current_Location__c, CurrentDesignation__c, Email, PhoneNumber__c,
    SCSCHAMPS__PhoneNumber__c, CurrentCompany__c, Consultant__c, Type_1__c,
    Spoken_Language__c (array).
    """
    _validate_files(files)
    wall_start = time.time()
    semaphore = asyncio.Semaphore(settings.bulk_concurrency)

    results: list[BulkClientParseItem] = await _gather_within_budget(
        [_parse_one_client(f, semaphore) for f in files],
        files,
        lambda filename, error: BulkClientParseItem(
            filename=filename, success=False, error=error
        ),
        "client",
    )

    total_ms = round((time.time() - wall_start) * 1000, 2)
    parsed_count = sum(1 for r in results if r.success)

    return BulkClientParseResponse(
        success=True,
        total=len(results),
        parsed=parsed_count,
        failed=len(results) - parsed_count,
        results=results,
        total_processing_time_ms=total_ms,
    )


# ── Legacy in-process job endpoints ──────────────────────────────────────────
# Superseded by the Redis-backed queue (POST /api/v1/parse/jobs). These keep
# state in this process only: results are lost on restart and invisible to other
# replicas. Kept for backward compatibility — prefer the queue for new work.

_jobs: dict[str, BulkJobStatus] = {}


async def _run_bulk_job(job_id: str, file_bytes: list[tuple[str, bytes, str]]) -> None:
    """Background task: parse all files and store results in _jobs."""
    job = _jobs[job_id]
    semaphore = asyncio.Semaphore(settings.bulk_concurrency)
    wall_start = time.time()

    async def _parse_bytes(filename: str, content: bytes, content_type: str) -> BulkParseItem:
        start = time.time()
        try:
            async with semaphore:
                from app.services.document import validate_file_bytes
                text = await validate_file_bytes(filename, content, content_type)
                parsed = await parse_resume(text)
            elapsed_ms = round((time.time() - start) * 1000, 2)
            return BulkParseItem(
                filename=filename, success=True,
                data=map_to_generic(parsed, resume_text=text), processing_time_ms=elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = round((time.time() - start) * 1000, 2)
            logger.error(f"Job {job_id}: failed '{filename}': {exc}")
            return BulkParseItem(
                filename=filename, success=False,
                error=str(exc), processing_time_ms=elapsed_ms,
            )

    try:
        results = await asyncio.gather(*[
            _parse_bytes(fn, content, ct) for fn, content, ct in file_bytes
        ])
        total_ms = round((time.time() - wall_start) * 1000, 2)
        parsed_count = sum(1 for r in results if r.success)
        job.status = "completed"
        job.result = BulkParseResponse(
            success=True,
            total=len(results),
            parsed=parsed_count,
            failed=len(results) - parsed_count,
            results=list(results),
            total_processing_time_ms=total_ms,
        )
    except Exception as exc:
        logger.error(f"Job {job_id} failed: {exc}")
        job.status = "failed"
        job.error = str(exc)


@router.post("/parse/job", response_model=BulkJobStatus, status_code=202, deprecated=True)
async def submit_bulk_job(
    files: list[UploadFile] = File(..., description="1 to 15 resume files (PDF or DOCX)"),
):
    """
    Deprecated — use POST /api/v1/parse/jobs (Redis-backed) instead.

    Submits a parse job held in this process's memory and returns a job_id (202).
    The job dies with the process and is invisible to other replicas.
    """
    _validate_files(files)

    # Read all file bytes eagerly before the background task starts
    # (UploadFile streams are not safe to read after the request scope ends)
    file_bytes: list[tuple[str, bytes, str]] = []
    for f in files:
        content = await f.read()
        file_bytes.append((f.filename or "unknown", content, f.content_type or ""))

    job_id = str(uuid.uuid4())
    job = BulkJobStatus(job_id=job_id, status="processing", total=len(files))
    _jobs[job_id] = job

    asyncio.create_task(_run_bulk_job(job_id, file_bytes))

    logger.info(f"Bulk job {job_id} submitted for {len(files)} file(s).")
    return job


@router.get("/parse/job/{job_id}", response_model=BulkJobStatus, deprecated=True)
async def get_bulk_job(job_id: str):
    """
    Deprecated — use GET /api/v1/parse/jobs/{batch_id} instead.

    - status=processing  — still running, poll again
    - status=completed   — result is populated
    - status=failed      — error is populated
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job
