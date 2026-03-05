import logging

import httpx
from fastapi import FastAPI

from app.config import settings
from app.middleware.auth import APIKeyMiddleware
from app.routes.parser import router as parser_router
from app.routes.salesforce import router as salesforce_router
from app.schemas.response import HealthResponse

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Resume Parser API",
    description="Parse PDF/DOCX resumes into structured JSON using Ollama LFM2:2.6B",
    version="1.0.0",
)

# Add authentication middleware
app.add_middleware(APIKeyMiddleware)

# Register routes
app.include_router(parser_router)
app.include_router(salesforce_router)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Health check endpoint. Uses a lightweight Ollama ping to avoid interfering with inference."""
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_host}/")
            ollama_ok = resp.status_code == 200
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if ollama_ok else "degraded",
        ollama_connected=ollama_ok,
        model=settings.ollama_model,
    )
