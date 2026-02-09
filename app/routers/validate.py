"""Validation endpoint — cross-validate LLM extraction against NER models."""

import logging
import time

from fastapi import APIRouter

from app.models.schemas import (
    ValidateRequest,
    ValidateResponse,
    ValidatedEntity,
)
from app.services.ner_service import cross_validate

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/validate", response_model=ValidateResponse)
def validate_extraction(request: ValidateRequest) -> ValidateResponse:
    """
    Cross-validate entities extracted by an LLM against biomedical NER models.

    Takes the original clinical text and the LLM's extracted entities,
    runs NER models on the same text, and compares results.

    Returns three categories:
    - confirmed: Found by BOTH LLM and NER (high confidence)
    - missed_by_llm: Found by NER but NOT LLM (should be added)
    - llm_only: Found by LLM but NOT NER (may be hallucinated)

    Plus confidence_adjustments: suggested score changes per entity.
    """
    start_time = time.time()

    try:
        result = cross_validate(
            text=request.text,
            extracted_entities={
                "drugs": request.extracted_entities.drugs,
                "diseases": request.extracted_entities.diseases,
                "genes": request.extracted_entities.genes,
                "anatomy": request.extracted_entities.anatomy,
            },
            confidence_threshold=request.confidence_threshold,
        )

        elapsed = (time.time() - start_time) * 1000

        return ValidateResponse(
            confirmed=[ValidatedEntity(**e) for e in result["confirmed"]],
            missed_by_llm=[ValidatedEntity(**e) for e in result["missed_by_llm"]],
            llm_only=[ValidatedEntity(**e) for e in result["llm_only"]],
            confidence_adjustments=result["confidence_adjustments"],
            processing_time_ms=round(elapsed, 2),
        )

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        elapsed = (time.time() - start_time) * 1000

        # Return empty validation — don't block the pipeline
        return ValidateResponse(
            confirmed=[],
            missed_by_llm=[],
            llm_only=[
                ValidatedEntity(
                    text=entity,
                    label=category.upper(),
                    status="llm_only",
                )
                for category, entities in {
                    "drugs": request.extracted_entities.drugs,
                    "diseases": request.extracted_entities.diseases,
                    "genes": request.extracted_entities.genes,
                    "anatomy": request.extracted_entities.anatomy,
                }.items()
                for entity in entities
            ],
            confidence_adjustments={},
            processing_time_ms=round(elapsed, 2),
        )
