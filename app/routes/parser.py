import asyncio
import time
import uuid
import logging
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.config import settings
from app.services.document import extract_text
from app.services.llm import parse_resume
from app.schemas.response import (
    ParseResponse,
    ResumeData,
    BulkParseResponse,
    BulkParseItem,
    BulkJobStatus,
    BulkSalesforceParseResponse,
    BulkSalesforceParseItem,
    SalesforceResumeData,
    ModelsResponse,
    ModelInfo,
    map_to_salesforce,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["parser"])

_BULK_MAX_FILES = 20
# Wall-clock budget for synchronous bulk requests (seconds).
_BULK_TIMEOUT = 110.0
# Max parallel LLM calls — prevents OpenRouter rate-limit errors on large batches.
_BULK_CONCURRENCY = 5

# In-memory job store: job_id -> BulkJobStatus
# Suitable for single-instance deployments (Railway, Render, etc.).
_jobs: dict[str, BulkJobStatus] = {}


@router.get("/models", response_model=ModelsResponse)
async def get_models():
    """Return the currently configured OpenRouter model."""
    return ModelsResponse(
        models=[ModelInfo(name=settings.openrouter_model)]
    )


@router.post("/parse", response_model=ParseResponse)
async def parse(file: UploadFile = File(..., description="Resume file (PDF or DOCX)")):
    """Parse an uploaded resume and extract structured fields."""
    start = time.time()

    text = await extract_text(file)
    logger.info(f"Extracted {len(text)} chars from {file.filename}")

    parsed = await parse_resume(text)
    elapsed_ms = round((time.time() - start) * 1000, 2)

    return ParseResponse(
        success=True,
        data=ResumeData(**parsed),
        processing_time_ms=elapsed_ms,
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
            data=ResumeData(**parsed),
            processing_time_ms=elapsed_ms,
        )
    except Exception as exc:
        elapsed_ms = round((time.time() - start) * 1000, 2)
        logger.error(f"Bulk: failed to parse '{filename}': {exc}")
        return BulkParseItem(
            filename=filename,
            success=False,
            error=str(exc),
            processing_time_ms=elapsed_ms,
        )


async def _parse_one_sf(file: UploadFile, semaphore: asyncio.Semaphore) -> BulkSalesforceParseItem:
    """Parse a single resume and return a Salesforce-mapped BulkSalesforceParseItem (never raises)."""
    start = time.time()
    filename = file.filename or "unknown"
    try:
        async with semaphore:
            text = await extract_text(file)
            parsed = await parse_resume(text)
        sf_data = map_to_salesforce(parsed)
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


def _validate_bulk_files(files: list[UploadFile]) -> None:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(files) > _BULK_MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum allowed per request is {_BULK_MAX_FILES}.",
        )


# ── Synchronous bulk endpoints ────────────────────────────────────────────────

@router.post("/parse-bulk", response_model=BulkParseResponse)
async def parse_bulk(
    files: List[UploadFile] = File(..., description="Up to 20 resume files (PDF or DOCX)"),
):
    """
    Parse up to 20 resume files in a single request.

    All resumes are processed concurrently (max 5 at a time). The total
    request completes within ~110 seconds (Salesforce async callout limit).
    Returns generic JSON for each file.
    """
    _validate_bulk_files(files)
    wall_start = time.time()
    semaphore = asyncio.Semaphore(_BULK_CONCURRENCY)

    try:
        results: list[BulkParseItem] = await asyncio.wait_for(
            asyncio.gather(*[_parse_one(f, semaphore) for f in files]),
            timeout=_BULK_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Bulk parsing exceeded the {_BULK_TIMEOUT}s time limit. "
                   "Try fewer files or smaller documents.",
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


@router.post("/parse-bulk/salesforce", response_model=BulkSalesforceParseResponse)
async def parse_bulk_salesforce(
    files: List[UploadFile] = File(..., description="Up to 20 resume files (PDF or DOCX)"),
):
    """
    Parse up to 20 resume files and return SCSCHAMPS-mapped Salesforce JSON for each.

    Same concurrency and timeout rules as /parse-bulk.
    Use this endpoint when feeding results directly into Salesforce SCSCHAMPS fields.
    """
    _validate_bulk_files(files)
    wall_start = time.time()
    semaphore = asyncio.Semaphore(_BULK_CONCURRENCY)

    try:
        results: list[BulkSalesforceParseItem] = await asyncio.wait_for(
            asyncio.gather(*[_parse_one_sf(f, semaphore) for f in files]),
            timeout=_BULK_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Bulk parsing exceeded the {_BULK_TIMEOUT}s time limit. "
                   "Try fewer files or smaller documents.",
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


# ── Async job endpoints ───────────────────────────────────────────────────────

async def _run_bulk_job(job_id: str, file_bytes: list[tuple[str, bytes, str]]) -> None:
    """Background task: parse all files and store results in _jobs."""
    job = _jobs[job_id]
    semaphore = asyncio.Semaphore(_BULK_CONCURRENCY)
    wall_start = time.time()

    async def _parse_bytes(filename: str, content: bytes, content_type: str) -> BulkParseItem:
        start = time.time()
        try:
            async with semaphore:
                from app.services.document import validate_file_bytes
                text = validate_file_bytes(filename, content, content_type)
                parsed = await parse_resume(text)
            elapsed_ms = round((time.time() - start) * 1000, 2)
            return BulkParseItem(
                filename=filename, success=True,
                data=ResumeData(**parsed), processing_time_ms=elapsed_ms,
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


@router.post("/parse-bulk/job", response_model=BulkJobStatus, status_code=202)
async def submit_bulk_job(
    files: List[UploadFile] = File(..., description="Up to 20 resume files (PDF or DOCX)"),
):
    """
    Submit a bulk parse job and get back a job_id immediately (HTTP 202).

    Poll GET /api/v1/parse-bulk/job/{job_id} to check status.
    Use this when you can't wait for a synchronous response (e.g. Salesforce
    async callouts with a 120-second limit but many large files).
    """
    _validate_bulk_files(files)

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


@router.get("/parse-bulk/job/{job_id}", response_model=BulkJobStatus)
async def get_bulk_job(job_id: str):
    """
    Poll the status of a bulk parse job.

    - status=processing  — still running, poll again
    - status=completed   — result is populated
    - status=failed      — error is populated
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job
