import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.middleware.auth import APIKeyMiddleware
from app.routes.jobs import router as jobs_router
from app.routes.parser import router as parser_router
from app.routes.salesforce import router as salesforce_router
from app.schemas.response import HealthResponse, QueueHealth
from app.services import queue

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup: validate config, open the Redis pool. Shutdown: close everything."""
    if not settings.openrouter_api_key:
        logger.warning("OPENROUTER_API_KEY is not set — LLM calls will fail with 401.")
    else:
        logger.info(f"Extraction model : {settings.openrouter_model}")
        logger.info(f"OCR model        : {settings.openrouter_ocr_model}")

    # Non-fatal: the synchronous endpoints keep working without Redis, only the
    # queue endpoints degrade to 503.
    await queue.init_pool()

    yield

    from app.services.llm_client import extraction_client, ocr_client
    await extraction_client.aclose()
    await ocr_client.aclose()
    await queue.close_pool()


app = FastAPI(
    title="Resume Parser API",
    description="Parse PDF/DOCX resumes into structured JSON using OpenRouter LLM",
    version="2.1.0",
    lifespan=lifespan,
)

# Add authentication middleware
app.add_middleware(APIKeyMiddleware)

# Register routes
app.include_router(parser_router)
app.include_router(jobs_router)
app.include_router(salesforce_router)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Health check. Pings OpenRouter and reports queue connectivity and depth."""
    from app.services.llm import check_openrouter
    openrouter_ok = await check_openrouter()
    queue_health = QueueHealth(**await queue.health())

    return HealthResponse(
        status="healthy" if openrouter_ok else "degraded",
        openrouter_connected=openrouter_ok,
        model=settings.openrouter_model,
        queue=queue_health,
    )
