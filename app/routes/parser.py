import time
import logging

from fastapi import APIRouter, UploadFile, File

from app.services.document import extract_text
from app.services.llm import parse_resume, list_models
from app.schemas.response import (
    ParseResponse,
    ResumeData,
    ModelsResponse,
    ModelInfo,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["parser"])


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


@router.get("/models", response_model=ModelsResponse)
async def get_models():
    """List available Ollama models."""
    models = await list_models()
    return ModelsResponse(
        models=[
            ModelInfo(
                name=m.get("name", ""),
                size=m.get("size"),
                modified_at=m.get("modified_at"),
            )
            for m in models
        ]
    )
