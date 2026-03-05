"""
Salesforce integration service.

Handles:
 - OAuth2 Username-Password / Connected-App (client_credentials) token flow
 - Fetching a resume file (PDF/DOCX) from Salesforce given:
     • a ContentDocument / ContentVersion ID  (Files)
     • a SCSCHAMPS__Resume_Attachment_Id__c value
     • a direct Resume URL stored on the candidate record
"""
from __future__ import annotations

import logging
from io import BytesIO

import httpx
from fastapi import HTTPException

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OAuth token cache  (in-memory; restarts or multi-process = re-auth)
# ---------------------------------------------------------------------------
_sf_token: str | None = None
_sf_instance_url: str | None = None


async def get_salesforce_token() -> tuple[str, str]:
    """
    Return (access_token, instance_url).
    Uses the cached token if available; otherwise fetches a new one via
    the OAuth2 Username-Password flow (also works as Connected-App
    client_credentials when username/password are omitted in SF setup).
    """
    global _sf_token, _sf_instance_url
    if _sf_token and _sf_instance_url:
        return _sf_token, _sf_instance_url

    if not settings.sf_client_id or not settings.sf_client_secret:
        raise HTTPException(
            status_code=503,
            detail="Salesforce credentials are not configured. "
                   "Set SF_CLIENT_ID, SF_CLIENT_SECRET (and optionally "
                   "SF_USERNAME, SF_PASSWORD, SF_SECURITY_TOKEN) in .env.",
        )

    token_url = f"{settings.sf_login_url}/services/oauth2/token"

    # Prefer client_credentials (Connected App) if no username supplied
    if settings.sf_username and settings.sf_password:
        payload = {
            "grant_type": "password",
            "client_id": settings.sf_client_id,
            "client_secret": settings.sf_client_secret,
            "username": settings.sf_username,
            "password": settings.sf_password + (settings.sf_security_token or ""),
        }
    else:
        payload = {
            "grant_type": "client_credentials",
            "client_id": settings.sf_client_id,
            "client_secret": settings.sf_client_secret,
        }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(token_url, data=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error("Salesforce OAuth failed: %s — %s", exc.response.status_code, exc.response.text)
            raise HTTPException(
                status_code=502,
                detail=f"Salesforce authentication failed: {exc.response.text}",
            )
        except httpx.RequestError as exc:
            logger.error("Salesforce OAuth network error: %s", exc)
            raise HTTPException(status_code=503, detail="Cannot reach Salesforce login endpoint.")

    body = resp.json()
    _sf_token = body["access_token"]
    _sf_instance_url = body["instance_url"]
    logger.info("Salesforce OAuth token acquired. Instance: %s", _sf_instance_url)
    return _sf_token, _sf_instance_url


def invalidate_token() -> None:
    """Clear cached token so the next request re-authenticates."""
    global _sf_token, _sf_instance_url
    _sf_token = None
    _sf_instance_url = None


# ---------------------------------------------------------------------------
# Resume file fetching
# ---------------------------------------------------------------------------

async def fetch_resume_by_attachment_id(attachment_id: str) -> tuple[bytes, str]:
    """
    Download a resume file from Salesforce using a ContentVersion ID or
    Attachment ID (SCSCHAMPS__Resume_Attachment_Id__c).

    Returns (file_bytes, filename).
    Tries ContentVersion first; falls back to Attachment API.
    """
    token, instance_url = await get_salesforce_token()
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        # 1. Try ContentVersion (Lightning Files)
        cv_url = (
            f"{instance_url}/services/data/v{settings.sf_api_version}"
            f"/sobjects/ContentVersion/{attachment_id}/VersionData"
        )
        resp = await client.get(cv_url, headers=headers)
        if resp.status_code == 200:
            filename = _guess_filename(resp.headers, attachment_id, "pdf")
            return resp.content, filename

        if resp.status_code == 401:
            invalidate_token()
            raise HTTPException(status_code=401, detail="Salesforce token expired. Retry the request.")

        # 2. Fall back to classic Attachment
        att_url = (
            f"{instance_url}/services/data/v{settings.sf_api_version}"
            f"/sobjects/Attachment/{attachment_id}/Body"
        )
        resp = await client.get(att_url, headers=headers)
        if resp.status_code == 200:
            filename = _guess_filename(resp.headers, attachment_id, "pdf")
            return resp.content, filename

        logger.error("Could not fetch attachment %s — status %s", attachment_id, resp.status_code)
        raise HTTPException(
            status_code=404,
            detail=f"Resume attachment '{attachment_id}' not found in Salesforce.",
        )


async def fetch_resume_by_url(resume_url: str) -> tuple[bytes, str]:
    """
    Download a resume from an arbitrary URL stored in SCSCHAMPS__Resume_URL__c.
    If the URL is a relative Salesforce path it is resolved against the
    instance URL (requires a valid SF token).
    """
    token, instance_url = await get_salesforce_token()

    if resume_url.startswith("/"):
        resume_url = instance_url + resume_url

    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        try:
            resp = await client.get(resume_url, headers=headers)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to download resume from Salesforce URL: {exc.response.status_code}",
            )

    filename = _guess_filename(resp.headers, "resume", "pdf")
    return resp.content, filename


async def fetch_resume_from_candidate(record_id: str) -> tuple[bytes, str]:
    """
    Given a SCSCHAMPS Candidate record ID, look up the resume attachment:
    1. Read SCSCHAMPS__Resume_Attachment_Id__c from the record
    2. If absent, read SCSCHAMPS__Resume_URL__c
    3. Download and return the file bytes.
    """
    token, instance_url = await get_salesforce_token()
    headers = {"Authorization": f"Bearer {token}"}

    fields = "SCSCHAMPS__Resume_Attachment_Id__c,SCSCHAMPS__Resume_URL__c,SCSCHAMPS__Resume__c"
    query_url = (
        f"{instance_url}/services/data/v{settings.sf_api_version}"
        f"/sobjects/SCSCHAMPS__Candidate__c/{record_id}?fields={fields}"
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(query_url, headers=headers)

    if resp.status_code == 401:
        invalidate_token()
        raise HTTPException(status_code=401, detail="Salesforce token expired. Retry the request.")
    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail=f"Candidate record '{record_id}' not found.")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Salesforce returned {resp.status_code}.")

    record = resp.json()
    attachment_id: str | None = record.get("SCSCHAMPS__Resume_Attachment_Id__c")
    resume_url: str | None = record.get("SCSCHAMPS__Resume_URL__c")

    if attachment_id:
        return await fetch_resume_by_attachment_id(attachment_id)
    if resume_url:
        return await fetch_resume_by_url(resume_url)

    raise HTTPException(
        status_code=422,
        detail="Candidate record has no resume attachment or URL set.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _guess_filename(headers: httpx.Headers, fallback_id: str, default_ext: str) -> str:
    cd = headers.get("content-disposition", "")
    if "filename=" in cd:
        part = cd.split("filename=")[-1].strip().strip('"').strip("'")
        if part:
            return part
    ct = headers.get("content-type", "")
    if "pdf" in ct:
        return f"{fallback_id}.pdf"
    if "word" in ct or "docx" in ct:
        return f"{fallback_id}.docx"
    return f"{fallback_id}.{default_ext}"
