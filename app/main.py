import asyncio
import logging
import os

import httpx
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


async def _wait_for_ollama(timeout: int = 300):
    """Wait for Ollama service to become reachable, then pull the model if needed."""
    if os.environ.get("TESTING") == "1":
        return
    logger.info(f"Waiting for Ollama at {settings.ollama_host} (timeout={timeout}s)...")
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        # Poll until Ollama is reachable
        for i in range(timeout):
            try:
                resp = await client.get(f"{settings.ollama_host}/")
                if resp.status_code == 200:
                    logger.info("Ollama is reachable.")
                    break
            except Exception:
                if i == 0:
                    logger.info("Ollama not yet reachable, retrying...")
            await asyncio.sleep(1)
        else:
            logger.warning(f"Ollama not reachable after {timeout}s — starting anyway (degraded mode).")
            return

        # Check if model is already pulled
        try:
            resp = await client.get(f"{settings.ollama_host}/api/tags")
            models = [m["name"] for m in resp.json().get("models", [])]
            model_name = settings.ollama_model
            # Match with or without :latest tag
            if any(model_name in m for m in models):
                logger.info(f"Model '{model_name}' already available.")
                return
        except Exception:
            pass

        # Pull the model (this can take a while on first deploy)
        logger.info(f"Pulling model '{settings.ollama_model}' — this may take several minutes on first deploy...")
        try:
            pull_resp = await client.post(
                f"{settings.ollama_host}/api/pull",
                json={"name": settings.ollama_model, "stream": False},
                timeout=httpx.Timeout(600.0),  # 10 min for large models
            )
            if pull_resp.status_code == 200:
                logger.info(f"Model '{settings.ollama_model}' pulled successfully.")
            else:
                logger.warning(f"Model pull returned status {pull_resp.status_code}: {pull_resp.text[:200]}")
        except Exception as e:
            logger.warning(f"Failed to pull model: {e} — parse requests will fail until model is available.")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup: wait for Ollama and pull model. Shutdown: nothing special."""
    await _wait_for_ollama()
    yield


app = FastAPI(
    title="Resume Parser API",
    description="Parse PDF/DOCX resumes into structured JSON using Ollama LFM2:2.6B",
    version="1.0.0",
    lifespan=lifespan,
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
