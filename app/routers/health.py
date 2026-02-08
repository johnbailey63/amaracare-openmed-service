"""Health check endpoint — service status and loaded models."""

import logging

from fastapi import APIRouter

from app.models.schemas import HealthResponse, ModelInfo
from app.services.model_manager import ModelManager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns service status, loaded models, uptime, and memory usage.
    Used by Railway health checks and Amara's client for availability detection.
    """
    manager = ModelManager()
    models = manager.get_loaded_models()

    loaded_count = sum(1 for m in models if m["status"] == "loaded")
    failed_count = sum(1 for m in models if m["status"] == "failed")

    if loaded_count == 0:
        status = "error"
    elif failed_count > 0:
        status = "degraded"
    else:
        status = "ok"

    return HealthResponse(
        status=status,
        loaded_models=[
            ModelInfo(
                name=m["name"],
                status=m["status"],
                load_time_ms=m.get("load_time_ms"),
            )
            for m in models
        ],
        models_available=loaded_count,
        uptime_seconds=round(manager.uptime_seconds, 1),
        memory_mb=round(manager.memory_mb, 1),
        version="0.1.0",
    )
