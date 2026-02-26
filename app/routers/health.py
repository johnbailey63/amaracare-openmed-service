"""Health check endpoint — service status, model lifecycle, and memory budget."""

import logging

from fastapi import APIRouter

from app.models.schemas import HealthResponse, ModelInfo
from app.services.model_manager import ModelManager, PII_MODEL_NAME

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns service status, model lifecycle states, memory budget, and uptime.
    Used by Railway health checks and Amara's client for availability detection.

    Status logic:
    - "ok" if PII model is loaded (the only critical-path model)
    - "degraded" if any model failed to load
    - "error" if PII model is not loaded
    NER models being "available" (on disk, not in RAM) is normal operating state.
    """
    manager = ModelManager()
    models = manager.get_loaded_models()
    memory = manager.memory_stats

    # Check PII model status — it's the only critical model
    pii_status = next(
        (m["status"] for m in models if m["name"] == PII_MODEL_NAME),
        "unknown",
    )
    failed_count = sum(1 for m in models if m["status"] == "failed")

    if pii_status != "loaded":
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
                pinned=m.get("pinned", False),
                eviction_count=m.get("eviction_count", 0),
            )
            for m in models
        ],
        models_available=memory["models_in_ram"] + memory["models_on_disk"],
        models_loaded=memory["models_in_ram"],
        models_on_disk=memory["models_on_disk"],
        memory_mb=memory["rss_mb"],
        memory_max_mb=memory["max_mb"],
        memory_utilization_pct=memory["utilization_pct"],
        total_evictions=memory["total_evictions"],
        uptime_seconds=round(manager.uptime_seconds, 1),
        version="0.1.0",
    )
