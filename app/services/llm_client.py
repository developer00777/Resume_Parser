"""
LLM HTTP client — Single Responsibility: make chat-completion calls to OpenRouter.

Injected into ResumeExtractor and OCRService so neither owns transport concerns.
Dependency Inversion: callers depend on LLMClient (concrete but focused), not on
the scattered _get_client() / _call_openrouter() internals that lived in llm.py.
"""
import asyncio
import logging

import httpx
from fastapi import HTTPException

from app.config import settings

logger = logging.getLogger(__name__)

_CHUNK_TIMEOUT = 45.0


class LLMClient:
    """
    Reusable async HTTP client for OpenRouter chat completions.

    One instance per model slug — extraction and OCR each get their own client
    so they can carry different model IDs without sharing state.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self._client: httpx.AsyncClient | None = None
        self._bound_loop: asyncio.AbstractEventLoop | None = None

    def _get_or_create(self) -> httpx.AsyncClient:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if (
            self._client is None
            or self._client.is_closed
            or (current_loop is not None and self._bound_loop is not current_loop)
        ):
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(_CHUNK_TIMEOUT, connect=10.0),
                base_url=settings.openrouter_base_url,
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://resumeparser-production-45b1.up.railway.app",
                    "X-Title": "Resume Parser API",
                },
            )
            self._bound_loop = current_loop

        return self._client

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 200,
        response_format: dict | None = None,
    ) -> str:
        """Single chat-completion call. Returns raw content string."""
        client = self._get_or_create()
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        if response_format:
            payload["response_format"] = response_format

        try:
            response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.ConnectError:
            logger.error("Cannot connect to OpenRouter")
            raise HTTPException(status_code=503, detail="OpenRouter service is unavailable.")
        except httpx.TimeoutException:
            logger.error("OpenRouter request timed out")
            raise HTTPException(status_code=504, detail="LLM processing timed out.")
        except httpx.HTTPStatusError as e:
            logger.error(
                f"OpenRouter error {e.response.status_code}: {e.response.text[:300]}"
            )
            raise HTTPException(
                status_code=502,
                detail=f"LLM service returned an error: {e.response.status_code}",
            )

        return response.json()["choices"][0]["message"]["content"]

    async def complete_multimodal(self, messages: list[dict], max_tokens: int = 3000) -> str:
        """Multimodal chat-completion (vision/file content). Used by OCR only."""
        client = self._get_or_create()
        try:
            response = await client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                },
            )
            response.raise_for_status()
        except httpx.ConnectError:
            logger.error("Cannot connect to OpenRouter (OCR)")
            raise HTTPException(status_code=503, detail="OpenRouter service is unavailable.")
        except httpx.TimeoutException:
            logger.error("OpenRouter OCR request timed out")
            raise HTTPException(status_code=504, detail="OCR processing timed out.")
        except httpx.HTTPStatusError as e:
            logger.error(f"OpenRouter OCR error {e.response.status_code}: {e.response.text[:300]}")
            raise HTTPException(
                status_code=502,
                detail=f"LLM service returned an error: {e.response.status_code}",
            )

        return response.json()["choices"][0]["message"]["content"]

    async def ping(self) -> bool:
        """Health-check: verify OpenRouter is reachable and the API key is valid."""
        client = self._get_or_create()
        try:
            resp = await client.get("/models")
            return resp.status_code == 200
        except Exception:
            return False

    async def aclose(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# ── Module-level singletons (one per model role) ──────────────────────────────
# Open/Closed: to swap models, change config — no code edits needed here.

extraction_client = LLMClient(model=settings.openrouter_model)
ocr_client = LLMClient(model=settings.openrouter_ocr_model)
