import io
import logging

from pypdf import PdfReader
from docx import Document
from fastapi import UploadFile, HTTPException

from app.config import settings

logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}


def validate_file(file: UploadFile) -> str:
    """Validate uploaded file type and size. Returns file type string."""
    content_type = file.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {content_type}. Only PDF and DOCX are accepted.",
        )
    return ALLOWED_CONTENT_TYPES[content_type]


async def extract_text(file: UploadFile) -> str:
    """Extract text from uploaded PDF or DOCX file."""
    file_type = validate_file(file)
    content = await file.read()

    if len(content) > settings.max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds maximum allowed size of {settings.max_file_size} bytes.",
        )

    if file_type == "pdf":
        return _extract_pdf(content)
    elif file_type == "docx":
        return _extract_docx(content)

    raise HTTPException(status_code=400, detail="Unsupported file type.")


def validate_file_bytes(filename: str, content: bytes, content_type: str) -> str:
    """Extract text from raw bytes (used by background job tasks)."""
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {content_type}. Only PDF and DOCX are accepted.",
        )
    if len(content) > settings.max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"File '{filename}' exceeds maximum allowed size of {settings.max_file_size} bytes.",
        )
    file_type = ALLOWED_CONTENT_TYPES[content_type]
    if file_type == "pdf":
        return _extract_pdf(content)
    return _extract_docx(content)


def _extract_pdf(content: bytes) -> str:
    """Extract text from PDF bytes."""
    try:
        reader = PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        text = "\n".join(pages).strip()
        if not text:
            raise HTTPException(
                status_code=422,
                detail="Could not extract text from PDF. The file may be image-based or empty.",
            )
        return text
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise HTTPException(status_code=422, detail=f"Failed to parse PDF: {e}")


def _extract_docx(content: bytes) -> str:
    """Extract text from DOCX bytes."""
    try:
        doc = Document(io.BytesIO(content))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs).strip()
        if not text:
            raise HTTPException(
                status_code=422,
                detail="Could not extract text from DOCX. The file may be empty.",
            )
        return text
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"DOCX extraction failed: {e}")
        raise HTTPException(status_code=422, detail=f"Failed to parse DOCX: {e}")
