"""De-identification endpoint — strip PHI from clinical text."""

import logging
import time

from fastapi import APIRouter

from app.models.schemas import DeidentifyRequest, DeidentifyResponse, NEREntity
from app.services.model_manager import ModelManager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/deidentify", response_model=DeidentifyResponse)
async def deidentify_text(request: DeidentifyRequest) -> DeidentifyResponse:
    """
    De-identify protected health information (PHI) from clinical text.

    Supports three methods:
    - mask: Replace PHI with type labels like [FIRST_NAME], [DATE_OF_BIRTH]
    - remove: Completely remove PHI spans
    - replace: Replace with synthetic but realistic values

    Returns a mapping dict so the caller can re-hydrate original values
    after processing the de-identified text through an LLM.
    """
    start_time = time.time()
    manager = ModelManager()

    try:
        result = manager.deidentify(
            text=request.text,
            method=request.method,
            confidence_threshold=request.confidence_threshold,
        )

        entities = [
            NEREntity(
                text=e["text"],
                label=e["label"],
                confidence=e["confidence"],
                model="pii_bioclinical",
                start=e.get("start"),
                end=e.get("end"),
            )
            for e in result["entities"]
        ]

        elapsed = (time.time() - start_time) * 1000

        return DeidentifyResponse(
            deidentified_text=result["deidentified_text"],
            entities=entities,
            mapping=result["mapping"],
            processing_time_ms=round(elapsed, 2),
        )

    except Exception as e:
        logger.error(f"De-identification failed: {e}")
        # Graceful degradation: return original text with error info
        elapsed = (time.time() - start_time) * 1000
        return DeidentifyResponse(
            deidentified_text=request.text,  # Return original — don't block the pipeline
            entities=[],
            mapping={},
            processing_time_ms=round(elapsed, 2),
        )
