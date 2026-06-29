"""
Document extraction service — Single Responsibility: convert PDF/DOCX bytes to plain text.

OCR is delegated to OCRService (injected), not called directly.
No imports from llm.py — dependency inversion enforced at the module boundary.
"""
import io
import logging

from pypdf import PdfReader
from docx import Document
from fastapi import UploadFile, HTTPException

from app.config import settings
from app.services.ocr import OCRService, ocr_service

logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}

_OCR_THRESHOLD = 100


def validate_file(file: UploadFile) -> str:
    content_type = file.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {content_type}. Only PDF and DOCX are accepted.",
        )
    return ALLOWED_CONTENT_TYPES[content_type]


async def extract_text(file: UploadFile, ocr: OCRService = ocr_service) -> str:
    """Extract text from an uploaded PDF or DOCX file."""
    file_type = validate_file(file)
    content = await file.read()

    if len(content) > settings.max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds maximum allowed size of {settings.max_file_size} bytes.",
        )

    if file_type == "pdf":
        return await _extract_pdf(content, ocr)
    return _extract_docx(content)


async def validate_file_bytes(
    filename: str,
    content: bytes,
    content_type: str,
    ocr: OCRService = ocr_service,
) -> str:
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
        return await _extract_pdf(content, ocr)
    return _extract_docx(content)


async def _extract_pdf(content: bytes, ocr: OCRService) -> str:
    """
    Extract text from PDF bytes.

    1. PyPDF text extraction (fast, free, works for text-based PDFs).
    2. If extracted text < _OCR_THRESHOLD chars, delegate to OCRService
       (uses openai/gpt-4o-mini — the multimodal/vision-capable model).
    """
    try:
        reader = PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                pages.append(text.strip())

        text = "\n\n".join(pages)

        if len(text.strip()) < _OCR_THRESHOLD:
            logger.warning(
                f"PyPDF extracted only {len(text.strip())} chars — triggering OCR fallback"
            )
            return await ocr.extract_text_from_pdf(content)

        return text
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise HTTPException(status_code=422, detail=f"Failed to parse PDF: {e}")


def _extract_docx(content: bytes) -> str:
    """
    Extract text from DOCX bytes.

    Extracts body paragraphs, tables, and text boxes (sidebar-style layouts).
    """
    try:
        doc = Document(io.BytesIO(content))
        parts: list[str] = []

        for para in doc.paragraphs:
            stripped = para.text.strip()
            if stripped:
                parts.append(stripped)

        for table in doc.tables:
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    parts.append(" | ".join(row_cells))

        try:
            from docx.oxml.ns import qn
            body = doc.element.body
            for txbx in body.iter(qn('w:txbxContent')):
                for p in txbx.iter(qn('w:p')):
                    texts = [r.text for r in p.iter(qn('w:t')) if r.text]
                    combined = "".join(texts).strip()
                    if combined and combined not in parts:
                        parts.append(combined)
        except Exception:
            pass

        text = "\n".join(parts).strip()
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
