"""NER endpoint — extract biomedical entities from clinical text."""

import logging
import time

from fastapi import APIRouter

from app.models.schemas import NERRequest, NERResponse, NEREntity
from app.services.model_manager import ModelManager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ner", response_model=NERResponse)
def extract_entities(request: NERRequest) -> NERResponse:
    """
    Run biomedical NER models on clinical text.

    If no models specified, runs all preloaded NER models (disease, pharma, gene, anatomy).
    Entities are deduplicated across models — if two models find the same span,
    the higher-confidence result is kept.
    """
    start_time = time.time()
    manager = ModelManager()

    # Determine which models to run
    if request.models:
        model_names = request.models
    else:
        model_names = manager.get_available_ner_models()

    if not model_names:
        return NERResponse(
            entities=[],
            models_used=[],
            processing_time_ms=0.0,
            text_length=len(request.text),
        )

    all_entities: list[NEREntity] = []
    models_used: list[str] = []

    for model_name in model_names:
        try:
            raw_entities = manager.analyze(
                request.text,
                model_name,
                confidence_threshold=request.confidence_threshold,
            )
            models_used.append(model_name)

            for entity in raw_entities:
                all_entities.append(NEREntity(
                    text=entity["text"],
                    label=entity["label"],
                    confidence=entity["confidence"],
                    model=model_name,
                    start=entity.get("start"),
                    end=entity.get("end"),
                ))

        except Exception as e:
            logger.error(f"Model '{model_name}' failed: {e}")
            # Continue with other models — don't fail the whole request
            continue

    # Deduplicate: if multiple models found the same text span, keep highest confidence
    deduped = _deduplicate_entities(all_entities)

    elapsed = (time.time() - start_time) * 1000

    return NERResponse(
        entities=deduped,
        models_used=models_used,
        processing_time_ms=round(elapsed, 2),
        text_length=len(request.text),
    )


def _deduplicate_entities(entities: list[NEREntity]) -> list[NEREntity]:
    """
    Deduplicate entities by text span.
    If the same text is found by multiple models, keep the highest-confidence one
    but note which models found it.
    """
    seen: dict[str, NEREntity] = {}

    for entity in entities:
        key = f"{entity.text.lower().strip()}|{entity.label}"
        if key not in seen or entity.confidence > seen[key].confidence:
            seen[key] = entity

    return list(seen.values())
