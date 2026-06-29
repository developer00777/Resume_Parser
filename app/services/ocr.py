"""
OCR Service — Single Responsibility: extract text from image-based PDFs via vision LLM.

Depends on LLMClient (injected), not on llm.py internals.
Uses the OCR-designated model (openai/gpt-4o-mini) which is multimodal/vision-capable.
DeepSeek V4 Flash is text-only and cannot be used here.
"""
import base64
import logging

from fastapi import HTTPException

from app.services.llm_client import LLMClient, ocr_client

logger = logging.getLogger(__name__)


class OCRService:
    """
    Converts image-based PDF bytes to plain text using a vision-capable LLM.

    Depends on LLMClient via constructor injection — easily testable with a mock client.
    """

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    async def extract_text_from_pdf(self, content: bytes) -> str:
        logger.info("OCR fallback triggered — PDF is image-based, sending to vision model")

        b64 = base64.b64encode(content).decode()
        data_url = f"data:application/pdf;base64,{b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "This is a scanned resume PDF. Extract ALL text exactly as it appears. "
                            "Preserve names, dates, companies, job titles, skills, education, and all other details. "
                            "Output plain text only — no commentary, no markdown, no JSON."
                        ),
                    },
                    {
                        "type": "file",
                        "file": {"url": data_url},
                    },
                ],
            }
        ]

        try:
            text = await self._client.complete_multimodal(messages, max_tokens=3000)
        except Exception as e:
            logger.error(f"OCR vision call failed: {e}")
            raise HTTPException(
                status_code=422,
                detail="Could not extract text from PDF. The file appears image-based and OCR also failed.",
            )

        logger.info(f"OCR extracted {len(text)} chars from image-based PDF")
        return text


# Module-level singleton — uses the dedicated multimodal ocr_client
ocr_service = OCRService(client=ocr_client)
