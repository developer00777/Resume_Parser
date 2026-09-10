"""
Async batch endpoints — accept resumes onto the Redis queue and report progress.

These exist because a synchronous callout cannot outlive Salesforce's 120-second
limit, and bulk parsing regularly does. Submitting returns HTTP 202 with a
batch_id in well under a second; results are collected by polling
GET /api/v1/parse/jobs/{batch_id} or by supplying a callback_url.

    POST   /api/v1/parse/jobs             — generic JSON output
    POST   /api/v1/parse/salesforce/jobs  — SCSCHAMPS-mapped output
    POST   /api/v1/parse/client/jobs      — client-1 mapped output
    GET    /api/v1/parse/jobs/{batch_id}  — status + results so far
    DELETE /api/v1/parse/jobs/{batch_id}  — drop a batch early

Each submit path also has a `/base64` twin that takes JSON instead of
multipart/form-data, because hand-building a byte-aligned multipart body is the
most error-prone part of an Apex callout.
"""
from __future__ import annotations

import base64
import binascii
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import settings
from app.schemas.response import (
    Base64BatchRequest,
    BatchStatusResponse,
    BatchSubmitResponse,
    RecordBatchRequest,
)
from app.services import queue
from app.services.document import ALLOWED_CONTENT_TYPES

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/parse", tags=["jobs"])

_EXTENSION_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _resolve_content_type(filename: str, content_type: str | None) -> str:
    """
    Trust the declared MIME type, falling back to the extension.

    Salesforce multipart callouts frequently send application/octet-stream, which
    would otherwise be rejected as an unsupported type.
    """
    if content_type in ALLOWED_CONTENT_TYPES:
        return content_type
    lowered = (filename or "").lower()
    for ext, mime in _EXTENSION_TYPES.items():
        if lowered.endswith(ext):
            return mime
    return content_type or ""


def _require_pool():
    pool = queue.get_pool()
    if pool is None:
        raise HTTPException(
            status_code=503,
            detail="Job queue is unavailable — Redis is not reachable. "
                   "Use the synchronous /api/v1/parse endpoints for small batches.",
        )
    return pool


async def _read_files(files: list[UploadFile]) -> list[tuple[str, str, bytes, str | None]]:
    """Validate and buffer every multipart upload before anything is enqueued."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(files) > settings.queue_max_files:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum allowed per batch is {settings.queue_max_files}.",
        )

    buffered: list[tuple[str, str, bytes, str | None]] = []
    for upload in files:
        filename = upload.filename or "unknown"
        buffered.append(_check_one(filename, upload.content_type, await upload.read()))
    return buffered


def _check_one(
    filename: str,
    declared_type: str | None,
    content: bytes,
    external_id: str | None = None,
) -> tuple[str, str, bytes, str | None]:
    """Validate a single file, whatever transport it arrived on."""
    content_type = _resolve_content_type(filename, declared_type)
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type for '{filename}': "
                   f"{declared_type or 'unknown'}. Only PDF and DOCX are accepted.",
        )
    if not content:
        raise HTTPException(status_code=400, detail=f"File '{filename}' is empty.")
    if len(content) > settings.max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"File '{filename}' exceeds the maximum size of "
                   f"{settings.max_file_size} bytes.",
        )
    return filename, content_type, content, external_id


def _decode_base64_files(request: Base64BatchRequest) -> list[tuple[str, str, bytes, str | None]]:
    """Decode and validate a JSON batch before anything is enqueued."""
    if not request.files:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(request.files) > settings.queue_max_files:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum allowed per batch is {settings.queue_max_files}.",
        )

    buffered: list[tuple[str, str, bytes, str | None]] = []
    for item in request.files:
        filename = item.filename or "unknown"
        try:
            content = base64.b64decode(item.content_base64, validate=False)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"File '{filename}' is not valid base64: {exc}",
            ) from exc
        buffered.append(_check_one(filename, item.content_type, content, item.external_id))
    return buffered


async def _enqueue(
    mode: str,
    buffered: list[tuple[str, str, bytes, str | None]],
    callback_url: str | None,
    callback_token: str | None,
) -> BatchSubmitResponse:
    """Shared tail of every submit path: validate the callback, create the batch."""
    pool = _require_pool()

    if callback_url and not queue.callback_url_allowed(callback_url):
        raise HTTPException(
            status_code=400,
            detail="callback_url host is not permitted by JOB_CALLBACK_ALLOWED_HOSTS.",
        )

    batch_id, total = await queue.create_batch(
        pool, mode, buffered, callback_url=callback_url, callback_token=callback_token
    )

    return BatchSubmitResponse(
        batch_id=batch_id,
        status="queued",
        mode=mode,
        total=total,
        poll_url=f"/api/v1/parse/jobs/{batch_id}",
        callback_url=callback_url or None,
    )


async def _submit(
    mode: str,
    files: list[UploadFile],
    callback_url: str | None,
    callback_token: str | None,
) -> BatchSubmitResponse:
    """multipart/form-data submit path."""
    return await _enqueue(mode, await _read_files(files), callback_url, callback_token)


async def _submit_base64(mode: str, request: Base64BatchRequest) -> BatchSubmitResponse:
    """JSON submit path — same batch, no multipart body to hand-build."""
    return await _enqueue(
        mode,
        _decode_base64_files(request),
        request.callback_url,
        request.callback_token,
    )


async def _submit_records(mode: str, request: RecordBatchRequest) -> BatchSubmitResponse:
    """Record-id submit path — the worker downloads the files from Salesforce."""
    pool = _require_pool()

    if not request.records:
        raise HTTPException(status_code=400, detail="No records provided.")
    if len(request.records) > settings.queue_max_files:
        raise HTTPException(
            status_code=400,
            detail=f"Too many records. Maximum allowed per batch is {settings.queue_max_files}.",
        )
    if request.callback_url and not queue.callback_url_allowed(request.callback_url):
        raise HTTPException(
            status_code=400,
            detail="callback_url host is not permitted by JOB_CALLBACK_ALLOWED_HOSTS.",
        )

    references: list[tuple[str, str, str | None, str | None]] = []
    for record in request.records:
        kind = (record.kind or queue.SOURCE_ATTACHMENT).lower()
        if kind not in queue.REFERENCE_SOURCES:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown kind '{record.kind}' for '{record.ref}'. "
                       f"Expected one of: {', '.join(queue.REFERENCE_SOURCES)}.",
            )
        if not record.ref:
            raise HTTPException(status_code=400, detail="A record is missing its 'ref'.")
        references.append((kind, record.ref, record.filename, record.external_id))

    batch_id, total = await queue.create_reference_batch(
        pool, mode, references,
        callback_url=request.callback_url,
        callback_token=request.callback_token,
    )

    return BatchSubmitResponse(
        batch_id=batch_id,
        status="queued",
        mode=mode,
        total=total,
        poll_url=f"/api/v1/parse/jobs/{batch_id}",
        callback_url=request.callback_url or None,
    )


# ── Submit ────────────────────────────────────────────────────────────────────

@router.post("/jobs", response_model=BatchSubmitResponse, status_code=202)
async def submit_generic(
    files: list[UploadFile] = File(..., description="Resume files (PDF or DOCX)"),
    callback_url: str | None = Form(None, description="Optional URL POSTed the finished batch"),
    callback_token: str | None = Form(None, description="Sent as 'Authorization: Bearer' on the callback"),
):
    """Queue resumes for generic JSON output and return a batch_id immediately."""
    return await _submit("generic", files, callback_url, callback_token)


@router.post("/salesforce/jobs", response_model=BatchSubmitResponse, status_code=202)
async def submit_salesforce(
    files: list[UploadFile] = File(..., description="Resume files (PDF or DOCX)"),
    callback_url: str | None = Form(None, description="Optional URL POSTed the finished batch"),
    callback_token: str | None = Form(None, description="Sent as 'Authorization: Bearer' on the callback"),
):
    """
    Queue resumes for SCSCHAMPS-mapped output and return a batch_id immediately.

    This is the async replacement for POST /api/v1/parse/salesforce. Because the
    response comes back in milliseconds it cannot hit the Salesforce callout
    limit, no matter how many resumes the batch holds.
    """
    return await _submit("salesforce", files, callback_url, callback_token)


@router.post("/client/jobs", response_model=BatchSubmitResponse, status_code=202)
async def submit_client(
    files: list[UploadFile] = File(..., description="Resume files (PDF or DOCX)"),
    callback_url: str | None = Form(None, description="Optional URL POSTed the finished batch"),
    callback_token: str | None = Form(None, description="Sent as 'Authorization: Bearer' on the callback"),
):
    """Queue resumes for client-1 mapped output and return a batch_id immediately."""
    return await _submit("client", files, callback_url, callback_token)


# ── Submit as JSON (base64) — the Apex-friendly twins ────────────────────────

@router.post("/jobs/base64", response_model=BatchSubmitResponse, status_code=202)
async def submit_generic_base64(request: Base64BatchRequest):
    """Queue resumes for generic JSON output from a JSON body."""
    return await _submit_base64("generic", request)


@router.post("/salesforce/jobs/base64", response_model=BatchSubmitResponse, status_code=202)
async def submit_salesforce_base64(request: Base64BatchRequest):
    """
    Queue resumes for SCSCHAMPS-mapped output from a JSON body.

    Identical to the multipart endpoint in every respect except transport. Apex
    can build this with JSON.serialize() and EncodingUtil.base64Encode(), instead
    of hand-padding a multipart body to keep base64 boundaries byte-aligned.
    """
    return await _submit_base64("salesforce", request)


@router.post("/client/jobs/base64", response_model=BatchSubmitResponse, status_code=202)
async def submit_client_base64(request: Base64BatchRequest):
    """Queue resumes for client-1 mapped output from a JSON body."""
    return await _submit_base64("client", request)


# ── Submit by record id — no file bytes leave Salesforce's heap ──────────────

@router.post("/jobs/records", response_model=BatchSubmitResponse, status_code=202)
async def submit_generic_records(request: RecordBatchRequest):
    """Queue Salesforce records for generic JSON output."""
    return await _submit_records("generic", request)


@router.post("/salesforce/jobs/records", response_model=BatchSubmitResponse, status_code=202)
async def submit_salesforce_records(request: RecordBatchRequest):
    """
    Queue Salesforce records for SCSCHAMPS-mapped output.

    The recommended path for bulk. Apex posts a list of ContentVersion /
    Attachment / Candidate ids — a few KB whatever the resumes weigh — and the
    worker downloads each file with the Connected App credentials. Nothing large
    ever passes through the Apex heap, and the callout costs milliseconds
    against the 120-second per-transaction budget.

    Requires SF_CLIENT_ID / SF_CLIENT_SECRET to be configured on the parser.
    """
    return await _submit_records("salesforce", request)


@router.post("/client/jobs/records", response_model=BatchSubmitResponse, status_code=202)
async def submit_client_records(request: RecordBatchRequest):
    """Queue Salesforce records for client-1 mapped output."""
    return await _submit_records("client", request)


# ── Poll / delete ─────────────────────────────────────────────────────────────

@router.get("/jobs/{batch_id}", response_model=BatchStatusResponse)
async def get_batch(batch_id: str):
    """
    Report a batch's progress and every result finished so far.

    - status=queued     — accepted, no worker has started it yet
    - status=processing — some files are done; `results` already holds them
    - status=completed  — `remaining` is 0; every file has a result row

    Results stream in as they land, so a caller can consume finished resumes
    without waiting for the slowest file in the batch.
    """
    pool = _require_pool()
    batch = await queue.get_batch(pool, batch_id)
    if batch is None:
        raise HTTPException(
            status_code=404,
            detail=f"Batch '{batch_id}' not found. It may have expired "
                   f"(results are kept for {settings.job_ttl}s).",
        )
    return BatchStatusResponse(**batch)


@router.delete("/jobs/{batch_id}", status_code=204)
async def delete_batch(batch_id: str):
    """Delete a batch and its stored file bytes before the TTL expires them."""
    pool = _require_pool()
    if not await queue.delete_batch(pool, batch_id):
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found.")
