import logging

from contextlib import asynccontextmanager
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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup: validate OpenRouter key is configured. Shutdown: close HTTP client."""
    if not settings.openrouter_api_key:
        logger.warning("OPENROUTER_API_KEY is not set — LLM calls will fail with 401.")
    else:
        logger.info(f"OpenRouter configured — model: {settings.openrouter_model}")
    yield
    # Close the reusable HTTP client on shutdown
    from app.services.llm import _openrouter_client
    if _openrouter_client and not _openrouter_client.is_closed:
        await _openrouter_client.aclose()


app = FastAPI(
    title="Resume Parser API",
    description="Parse PDF/DOCX resumes into structured JSON using OpenRouter LLM",
    version="2.0.0",
    lifespan=lifespan,
)

# Add authentication middleware
app.add_middleware(APIKeyMiddleware)

# Register routes
app.include_router(parser_router)
app.include_router(salesforce_router)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Health check endpoint. Pings OpenRouter to verify connectivity."""
    from app.services.llm import check_openrouter
    openrouter_ok = await check_openrouter()

    return HealthResponse(
        status="healthy" if openrouter_ok else "degraded",
        openrouter_connected=openrouter_ok,
        model=settings.openrouter_model,
    )
