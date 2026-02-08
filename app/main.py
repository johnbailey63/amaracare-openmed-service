"""
AmaraCare OpenMed Service
~~~~~~~~~~~~~~~~~~~~~~~~~

FastAPI microservice wrapping OpenMed's biomedical NER and PHI de-identification.
Provides entity extraction, de-identification, and cross-validation endpoints
for the AmaraCare cancer navigation platform.

Usage:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Environment:
    OPENMED_API_KEY       - API key for authenticating requests from Amara
    OPENMED_DEVICE        - "cpu" or "cuda" (default: cpu)
    OPENMED_MODELS        - Comma-separated model names to preload
    OPENMED_CACHE_DIR     - Directory for caching downloaded models
    PORT                  - Server port (default: 8000, Railway sets this)
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routers import ner, deidentify, validate, health
from app.services.model_manager import ModelManager

# ===========================================
# Logging
# ===========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ===========================================
# Lifespan (startup/shutdown)
# ===========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    logger.info("Starting AmaraCare OpenMed Service...")

    # Pre-load models into memory
    manager = ModelManager()
    manager.preload_models()

    logger.info("Service ready.")
    yield

    logger.info("Shutting down AmaraCare OpenMed Service.")


# ===========================================
# App
# ===========================================

app = FastAPI(
    title="AmaraCare OpenMed Service",
    description=(
        "Biomedical NER and PHI de-identification microservice for AmaraCare. "
        "Powered by OpenMed's open-source clinical NLP models."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ===========================================
# CORS
# ===========================================

ALLOWED_ORIGINS = [
    "https://www.amaracare.ai",
    "https://amaracare.ai",
    "http://localhost:3000",      # Next.js dev server
    "http://localhost:3001",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ===========================================
# API Key Authentication
# ===========================================

API_KEY = os.environ.get("OPENMED_API_KEY", "")


@app.middleware("http")
async def authenticate_request(request: Request, call_next):
    """Simple API key authentication via X-API-Key header."""

    # Skip auth for health check and docs
    if request.url.path in ("/health", "/docs", "/openapi.json", "/redoc"):
        return await call_next(request)

    # Skip auth if no key is configured (dev mode)
    if not API_KEY:
        return await call_next(request)

    provided_key = request.headers.get("X-API-Key", "")
    if provided_key != API_KEY:
        return JSONResponse(
            status_code=401,
            content={"error": "Unauthorized", "detail": "Invalid or missing X-API-Key header"},
        )

    return await call_next(request)


# ===========================================
# Global Error Handler
# ===========================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc)},
    )


# ===========================================
# Routers
# ===========================================

app.include_router(ner.router, tags=["NER"])
app.include_router(deidentify.router, tags=["De-identification"])
app.include_router(validate.router, tags=["Validation"])
app.include_router(health.router, tags=["Health"])


# ===========================================
# Root
# ===========================================

@app.get("/")
async def root():
    return {
        "service": "AmaraCare OpenMed Service",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }
