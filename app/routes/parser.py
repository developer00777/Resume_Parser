import asyncio
import time
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
    ModelsResponse,
    ModelInfo,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["parser"])

_BULK_MAX_FILES = 20
# Total wall-clock budget for a bulk request (seconds).
# All resumes are parsed concurrently so this covers the slowest single resume.
_BULK_TIMEOUT = 110.0


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


async def _parse_one(file: UploadFile) -> BulkParseItem:
    """Parse a single resume file and return a BulkParseItem (never raises)."""
    start = time.time()
    filename = file.filename or "unknown"
    try:
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


@router.post("/parse-bulk", response_model=BulkParseResponse)
async def parse_bulk(
    files: List[UploadFile] = File(..., description="Up to 20 resume files (PDF or DOCX)"),
):
    """
    Parse up to 20 resume files in a single request.

    All resumes are processed concurrently. The total request completes within
    ~110 seconds (Salesforce async callout limit is 120 s).
    """
    if len(files) > _BULK_MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum allowed per request is {_BULK_MAX_FILES}.",
        )

    wall_start = time.time()

    try:
        results: list[BulkParseItem] = await asyncio.wait_for(
            asyncio.gather(*[_parse_one(f) for f in files]),
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
    failed_count = len(results) - parsed_count

    return BulkParseResponse(
        success=True,
        total=len(results),
        parsed=parsed_count,
        failed=failed_count,
        results=results,
        total_processing_time_ms=total_ms,
    )
