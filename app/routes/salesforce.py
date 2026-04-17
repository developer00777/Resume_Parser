"""
Salesforce-specific routes.

POST /api/v1/salesforce/parse-candidate
    Accepts a Salesforce Candidate record ID, fetches the resume file from
    Salesforce (via OAuth2), parses it, and returns a SCSCHAMPS-mapped JSON
    response ready to write back to Salesforce.

POST /api/v1/salesforce/parse-attachment
    Accepts a raw Salesforce ContentVersion / Attachment ID and an optional
    filename, downloads the file, and returns the SCSCHAMPS-mapped result.

POST /api/v1/salesforce/parse-url
    Accepts a resume URL (absolute or Salesforce-relative) and returns the
    SCSCHAMPS-mapped result.
"""
import io
import time
import logging
from tempfile import SpooledTemporaryFile

from fastapi import APIRouter, Query
from starlette.datastructures import Headers, UploadFile

from app.services.document import extract_text
from app.services.llm import parse_resume
from app.services.salesforce import (
    fetch_resume_from_candidate,
    fetch_resume_by_attachment_id,
    fetch_resume_by_url,
)
from app.schemas.response import SalesforceParseResponse, map_to_salesforce

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/salesforce", tags=["salesforce"])


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
