"""
Salesforce-specific routes.

POST /api/v1/salesforce/parse-candidate   — one Candidate record ID
POST /api/v1/salesforce/parse-attachment  — one ContentVersion/Attachment ID
POST /api/v1/salesforce/parse-url         — one resume URL
"""
import time
import logging
from tempfile import SpooledTemporaryFile

from fastapi import APIRouter, HTTPException, Query
from starlette.datastructures import Headers, UploadFile

from app.services.document import extract_text
from app.services.llm import parse_resume
from app.services.salesforce import (
    fetch_resume_from_candidate,
    fetch_resume_by_attachment_id,
    fetch_resume_by_url,
    update_candidate,
)
from app.services import salesforce_coerce as coerce
from app.services import salesforce_schema as schema
from app.schemas.response import (
    SalesforceParseResponse,
    SalesforceResumeData,
    SalesforceUpdateResponse,
    extraction_from_parsed,
    map_to_salesforce,
)

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

    return SalesforceParseResponse(
        success=True,
        data=sf_data,
        processing_time_ms=elapsed_ms,
        extraction=extraction_from_parsed(parsed),
    )


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

    return SalesforceParseResponse(
        success=True,
        data=sf_data,
        processing_time_ms=elapsed_ms,
        extraction=extraction_from_parsed(parsed),
    )


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
    # Only fall back to the caller's hint when the server told us nothing
    # useful. The previous condition overwrote a correctly-resolved ".docx"
    # name with the default "resume.pdf", which sent Word files down the PDF
    # path — PyPDF finds no text, the OCR fallback fires, and a DOCX is handed
    # to a vision model.
    if not resolved_filename.lower().endswith((".pdf", ".docx")) and filename:
        resolved_filename = filename

    upload = _bytes_to_upload(content, resolved_filename)
    text = await extract_text(upload)

    logger.info("URL resume — extracted %d chars from %s", len(text), resume_url)

    parsed = await parse_resume(text)
    sf_data = map_to_salesforce(parsed, raw_text=text)
    elapsed_ms = round((time.time() - start) * 1000, 2)

    return SalesforceParseResponse(
        success=True,
        data=sf_data,
        processing_time_ms=elapsed_ms,
        extraction=extraction_from_parsed(parsed),
    )


# ---------------------------------------------------------------------------
# Write-back
#
# The parser owns the write rather than Apex. It already authenticates to the
# org and already computes the mapping, so a PATCH from here collapses the whole
# Apex-side problem: no wrapper classes, no case-insensitive identifier clash,
# no date coercion in a language with no forgiving parser, no multipart, and no
# 6 MB heap ceiling. Apex becomes a Queueable that posts a record ID.
# ---------------------------------------------------------------------------

# Fields Salesforce owns. A resume can never be the source of truth for these,
# so they are never written even when the payload carries a value.
_ORG_OWNED_FIELDS = frozenset({
    "Candidate_Status", "Status", "Background_Check", "Source",
    "Talent_Id", "Job_Id", "job", "Lead", "Recruiter",
    "converted_from_lead", "Ampliz_Contact", "Ampliz_Talent_Name",
    "Resume_URL", "Resume_Attachment_Id",
})

# Which parsed section each payload field derives from. When a chunk fails, its
# fields are withheld instead of written as blanks over real data.
_FIELD_SECTION = {
    "contact": (
        "FirstName", "LastName", "Name", "Email", "AlternateEmail", "Phone",
        "PhoneNumber", "MobilePhone", "AlternatePhoneNumber", "LinkedIn_URL",
        "Web_address", "Current_Location", "City", "State",
    ),
    "personal": (
        "DateOfBirth", "Birthdate", "Gender", "Blood_Group", "Father_s_Name",
        "MotherName", "Nationnality", "PAN_Number", "Passport_Number",
        "AadharNumber", "LanguagesKnown",
    ),
    "skills": (
        "Primary_Skills", "Technical_Skills", "General_Skills", "SkillList",
        "Skill_List", "AutoPopulate_Skillset", "Key_Skillsets_del",
    ),
    "experience": (
        "Title", "Designation", "Department", "Company", "CurrentCompany",
        "CurrentDesignation", "CurrentDuration", "Years_of_Experience",
        "No_of_companies_worked_in", "Current_Employment", "Industry",
        "Current_CTC", "Expected_CTC", "Notice_Period", "Preferred_Location",
        "ResumeRich", "Resume",
    ),
    "education": (
        "Education", "Highest_Degree", "education_start_year",
        "Education_End_Year", "Education_year", "educationDetail",
        "Qualification_1", "Qualification_1_Type", "Qualification_2",
        "Qualification_2_Type", "Institute_1", "Institute_2",
    ),
    "certifications": ("Certification",),
    "awards": ("Awards",),
}

_SECTION_BY_FIELD = {
    field_name: section
    for section, field_names in _FIELD_SECTION.items()
    for field_name in field_names
}


def _writable_payload(
    sf_data: SalesforceResumeData, failed_sections: set[str]
) -> tuple[dict[str, object], dict[str, str]]:
    """
    Decide what may actually be written.

    Returns (api_name -> value, logical_name -> reason_skipped). A field is
    withheld when it is None, when Salesforce owns it, when the chunk that
    produces it failed, or when the org has no updateable field of that name.
    """
    written: dict[str, object] = {}
    skipped: dict[str, str] = {}

    for logical, value in sf_data.model_dump().items():
        if logical == "score_breakdown":
            # Nested detail, not an sObject field. Candidate_Score and
            # Resume_Score carry the numbers Salesforce stores.
            continue
        if value is None:
            skipped[logical] = "no value extracted"
            continue
        if logical in _ORG_OWNED_FIELDS:
            skipped[logical] = "Salesforce owns this field"
            continue

        section = _SECTION_BY_FIELD.get(logical)
        if section and section in failed_sections:
            skipped[logical] = f"'{section}' extraction failed — refusing to overwrite"
            continue

        spec = coerce.spec_for(logical)
        if spec is None:
            skipped[logical] = "no updateable field of this name in the org"
            continue

        written[spec.api_name] = value

    return written, skipped


@router.get("/describe/status")
async def describe_status():
    """
    Report the field mapping's state without touching Salesforce.

    Use this to confirm a deploy came up with a usable mapping, and to see which
    fields the org does not have.
    """
    return schema.state().summary()


@router.post("/describe/reload")
async def reload_describe():
    """
    Rebuild the field mapping from a live Describe of the Candidate sObject.

    Runs automatically at startup and whenever the mapping goes stale, so this
    is only needed to pick up a schema change immediately. Safe to call
    repeatedly.
    """
    state = await schema.ensure_loaded(force=True)
    if not state.loaded:
        raise HTTPException(
            status_code=502,
            detail=f"Could not load the field mapping from Salesforce: {state.last_error}",
        )
    return {"success": True, **state.summary()}


@router.post("/parse-and-update", response_model=SalesforceUpdateResponse)
async def parse_and_update(
    record_id: str = Query(..., description="SCSCHAMPS__Candidate__c record ID"),
    dry_run: bool = Query(False, description="Compute and validate the payload without writing it"),
):
    """
    Parse the resume on a Candidate record and write the results back onto it.

    This is the endpoint Apex should call: it sends a record ID and gets back
    what changed. No file bytes cross the wire in either direction.

    Fields are withheld rather than blanked when the resume did not mention
    them, when Salesforce owns them, or when the LLM call that would have
    extracted them failed.
    """
    start = time.time()

    # Loaded at startup and refreshed on a TTL; this retries if that failed, so
    # a Salesforce blip at boot does not disable write-back for the process's
    # whole lifetime.
    state = await schema.ensure_loaded()
    if not state.loaded:
        raise HTTPException(
            status_code=409,
            detail=(
                "Field mapping is not loaded, so there is nothing safe to write "
                f"({state.last_error or 'unknown reason'}). Writing against "
                "assumed field names could fail the record or write to the "
                "wrong field. Check Salesforce credentials, then retry or call "
                "POST /api/v1/salesforce/describe/reload."
            ),
        )

    content, filename = await fetch_resume_from_candidate(record_id)
    upload = _bytes_to_upload(content, filename)
    text = await extract_text(upload)

    parsed = await parse_resume(text)
    sf_data = map_to_salesforce(parsed, raw_text=text)
    extraction = extraction_from_parsed(parsed)

    written, skipped = _writable_payload(sf_data, set(extraction.failed_sections))

    if extraction.failed_sections:
        logger.warning(
            "Candidate %s: extraction failed for %s — those fields withheld, not blanked",
            record_id, ", ".join(extraction.failed_sections),
        )

    if not dry_run:
        await update_candidate(record_id, written)

    return SalesforceUpdateResponse(
        success=True,
        record_id=record_id,
        dry_run=dry_run,
        fields_written=written,
        fields_skipped=skipped,
        extraction=extraction,
        processing_time_ms=round((time.time() - start) * 1000, 2),
    )
