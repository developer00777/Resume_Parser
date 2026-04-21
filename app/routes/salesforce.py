"""
Salesforce-specific routes.

Single-record endpoints:
  POST /api/v1/salesforce/parse-candidate   — one Candidate record ID
  POST /api/v1/salesforce/parse-attachment  — one ContentVersion/Attachment ID
  POST /api/v1/salesforce/parse-url         — one resume URL

Bulk endpoints (1–15 records, parallel):
  POST /api/v1/salesforce/parse-candidates  — list of Candidate record IDs
  POST /api/v1/salesforce/parse-attachments — list of ContentVersion/Attachment IDs
  POST /api/v1/salesforce/parse-urls        — list of resume URLs
"""
import asyncio
import time
import logging
from tempfile import SpooledTemporaryFile
from typing import List

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from starlette.datastructures import Headers, UploadFile

from app.services.document import extract_text
from app.services.llm import parse_resume
from app.services.salesforce import (
    fetch_resume_from_candidate,
    fetch_resume_by_attachment_id,
    fetch_resume_by_url,
)
from app.schemas.response import (
    SalesforceParseResponse,
    BulkSalesforceParseResponse,
    BulkSalesforceParseItem,
    map_to_salesforce,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/salesforce", tags=["salesforce"])

_BULK_MAX = 15
_BULK_CONCURRENCY = 5
_BULK_TIMEOUT = 110.0


class BulkCandidateRequest(BaseModel):
    record_ids: List[str]


class BulkAttachmentRequest(BaseModel):
    attachment_ids: List[str]
    filename: str = "resume.pdf"


class BulkUrlRequest(BaseModel):
    resume_urls: List[str]
    filename: str = "resume.pdf"


def _validate_bulk(items: list) -> None:
    if not items:
        raise HTTPException(status_code=400, detail="No IDs provided.")
    if len(items) > _BULK_MAX:
        raise HTTPException(status_code=400, detail=f"Maximum {_BULK_MAX} records per request.")


# ---------------------------------------------------------------------------
# Helper: wrap raw bytes as a FastAPI-compatible UploadFile
# ---------------------------------------------------------------------------

def _bytes_to_upload(content: bytes, filename: str) -> UploadFile:
    """Wrap raw bytes in an UploadFile so document.extract_text() can handle it."""
    mime = (
        "application/pdf"
        if filename.lower().endswith(".pdf")
        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    tmp = SpooledTemporaryFile()
    tmp.write(content)
    tmp.seek(0)
    headers = Headers(headers={"content-type": mime})
    return UploadFile(filename=filename, headers=headers, file=tmp)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/parse-candidate", response_model=SalesforceParseResponse)
async def parse_candidate(
    record_id: str = Query(..., description="SCSCHAMPS__Candidate__c record ID in Salesforce"),
):
    """
    Fetch resume from a Salesforce Candidate record and return SCSCHAMPS-mapped JSON.

    The endpoint reads SCSCHAMPS__Resume_Attachment_Id__c (or
    SCSCHAMPS__Resume_URL__c as fallback) from the record, downloads the
    file, and parses it.
    """
    start = time.time()

    content, filename = await fetch_resume_from_candidate(record_id)
    upload = _bytes_to_upload(content, filename)
    text = await extract_text(upload)

    logger.info("Candidate %s — extracted %d chars from %s", record_id, len(text), filename)

    parsed = await parse_resume(text)
    sf_data = map_to_salesforce(parsed, raw_text=text)
    elapsed_ms = round((time.time() - start) * 1000, 2)

    return SalesforceParseResponse(success=True, data=sf_data, processing_time_ms=elapsed_ms)


@router.post("/parse-attachment", response_model=SalesforceParseResponse)
async def parse_attachment(
    attachment_id: str = Query(..., description="Salesforce ContentVersion or Attachment ID"),
    filename: str = Query("resume.pdf", description="Hint for file extension (pdf or docx)"),
):
    """
    Download a resume by Salesforce ContentVersion / Attachment ID and parse it.
    Useful when you already have the attachment ID without a full candidate record.
    """
    start = time.time()

    content, resolved_filename = await fetch_resume_by_attachment_id(attachment_id)
    if resolved_filename == attachment_id + ".pdf" and not filename.endswith(".pdf"):
        resolved_filename = filename  # honour caller's hint

    upload = _bytes_to_upload(content, resolved_filename)
    text = await extract_text(upload)

    logger.info("Attachment %s — extracted %d chars", attachment_id, len(text))

    parsed = await parse_resume(text)
    sf_data = map_to_salesforce(parsed, raw_text=text)
    elapsed_ms = round((time.time() - start) * 1000, 2)

    return SalesforceParseResponse(success=True, data=sf_data, processing_time_ms=elapsed_ms)


@router.post("/parse-url", response_model=SalesforceParseResponse)
async def parse_url(
    resume_url: str = Query(..., description="Absolute or Salesforce-relative resume URL"),
    filename: str = Query("resume.pdf", description="Hint for file extension"),
):
    """
    Download a resume from a URL (e.g. SCSCHAMPS__Resume_URL__c) and parse it.
    Supports authenticated Salesforce URLs (token is applied automatically).
    """
    start = time.time()

    content, resolved_filename = await fetch_resume_by_url(resume_url)
    if resolved_filename.endswith(".pdf") is False and filename:
        resolved_filename = filename

    upload = _bytes_to_upload(content, resolved_filename)
    text = await extract_text(upload)

    logger.info("URL resume — extracted %d chars from %s", len(text), resume_url)

    parsed = await parse_resume(text)
    sf_data = map_to_salesforce(parsed, raw_text=text)
    elapsed_ms = round((time.time() - start) * 1000, 2)

    return SalesforceParseResponse(success=True, data=sf_data, processing_time_ms=elapsed_ms)


# ---------------------------------------------------------------------------
# Bulk endpoints
# ---------------------------------------------------------------------------

async def _parse_candidate(record_id: str, semaphore: asyncio.Semaphore) -> BulkSalesforceParseItem:
    start = time.time()
    try:
        async with semaphore:
            content, filename = await fetch_resume_from_candidate(record_id)
            upload = _bytes_to_upload(content, filename)
            text = await extract_text(upload)
            parsed = await parse_resume(text)
        sf_data = map_to_salesforce(parsed, raw_text=text)
        return BulkSalesforceParseItem(
            filename=record_id, success=True, data=sf_data,
            processing_time_ms=round((time.time() - start) * 1000, 2),
        )
    except Exception as exc:
        logger.error(f"Bulk candidate '{record_id}' failed: {exc}")
        return BulkSalesforceParseItem(
            filename=record_id, success=False, error=str(exc),
            processing_time_ms=round((time.time() - start) * 1000, 2),
        )


async def _parse_attachment_bulk(attachment_id: str, filename: str, semaphore: asyncio.Semaphore) -> BulkSalesforceParseItem:
    start = time.time()
    try:
        async with semaphore:
            content, resolved_filename = await fetch_resume_by_attachment_id(attachment_id)
            if resolved_filename == attachment_id + ".pdf" and not filename.endswith(".pdf"):
                resolved_filename = filename
            upload = _bytes_to_upload(content, resolved_filename)
            text = await extract_text(upload)
            parsed = await parse_resume(text)
        sf_data = map_to_salesforce(parsed, raw_text=text)
        return BulkSalesforceParseItem(
            filename=attachment_id, success=True, data=sf_data,
            processing_time_ms=round((time.time() - start) * 1000, 2),
        )
    except Exception as exc:
        logger.error(f"Bulk attachment '{attachment_id}' failed: {exc}")
        return BulkSalesforceParseItem(
            filename=attachment_id, success=False, error=str(exc),
            processing_time_ms=round((time.time() - start) * 1000, 2),
        )


async def _parse_url_bulk(resume_url: str, filename: str, semaphore: asyncio.Semaphore) -> BulkSalesforceParseItem:
    start = time.time()
    try:
        async with semaphore:
            content, resolved_filename = await fetch_resume_by_url(resume_url)
            if not resolved_filename.endswith(".pdf") and filename:
                resolved_filename = filename
            upload = _bytes_to_upload(content, resolved_filename)
            text = await extract_text(upload)
            parsed = await parse_resume(text)
        sf_data = map_to_salesforce(parsed, raw_text=text)
        return BulkSalesforceParseItem(
            filename=resume_url, success=True, data=sf_data,
            processing_time_ms=round((time.time() - start) * 1000, 2),
        )
    except Exception as exc:
        logger.error(f"Bulk URL '{resume_url}' failed: {exc}")
        return BulkSalesforceParseItem(
            filename=resume_url, success=False, error=str(exc),
            processing_time_ms=round((time.time() - start) * 1000, 2),
        )


@router.post("/parse-candidates", response_model=BulkSalesforceParseResponse)
async def parse_candidates(body: BulkCandidateRequest):
    """Parse 1–15 Salesforce Candidate records in parallel. Pass record IDs in the request body."""
    _validate_bulk(body.record_ids)
    wall_start = time.time()
    semaphore = asyncio.Semaphore(_BULK_CONCURRENCY)
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*[_parse_candidate(rid, semaphore) for rid in body.record_ids]),
            timeout=_BULK_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Parsing exceeded {_BULK_TIMEOUT}s. Try fewer records.")
    parsed_count = sum(1 for r in results if r.success)
    return BulkSalesforceParseResponse(
        success=True, total=len(results), parsed=parsed_count,
        failed=len(results) - parsed_count, results=list(results),
        total_processing_time_ms=round((time.time() - wall_start) * 1000, 2),
    )


@router.post("/parse-attachments", response_model=BulkSalesforceParseResponse)
async def parse_attachments(body: BulkAttachmentRequest):
    """Parse 1–15 Salesforce ContentVersion/Attachment IDs in parallel."""
    _validate_bulk(body.attachment_ids)
    wall_start = time.time()
    semaphore = asyncio.Semaphore(_BULK_CONCURRENCY)
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*[_parse_attachment_bulk(aid, body.filename, semaphore) for aid in body.attachment_ids]),
            timeout=_BULK_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Parsing exceeded {_BULK_TIMEOUT}s. Try fewer records.")
    parsed_count = sum(1 for r in results if r.success)
    return BulkSalesforceParseResponse(
        success=True, total=len(results), parsed=parsed_count,
        failed=len(results) - parsed_count, results=list(results),
        total_processing_time_ms=round((time.time() - wall_start) * 1000, 2),
    )


@router.post("/parse-urls", response_model=BulkSalesforceParseResponse)
async def parse_urls(body: BulkUrlRequest):
    """Parse 1–15 resume URLs in parallel."""
    _validate_bulk(body.resume_urls)
    wall_start = time.time()
    semaphore = asyncio.Semaphore(_BULK_CONCURRENCY)
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*[_parse_url_bulk(url, body.filename, semaphore) for url in body.resume_urls]),
            timeout=_BULK_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Parsing exceeded {_BULK_TIMEOUT}s. Try fewer records.")
    parsed_count = sum(1 for r in results if r.success)
    return BulkSalesforceParseResponse(
        success=True, total=len(results), parsed=parsed_count,
        failed=len(results) - parsed_count, results=list(results),
        total_processing_time_ms=round((time.time() - wall_start) * 1000, 2),
    )
