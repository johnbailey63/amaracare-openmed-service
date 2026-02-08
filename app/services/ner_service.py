"""NER service functions wrapping the ModelManager for business logic."""

import logging
from difflib import SequenceMatcher

from app.services.model_manager import ModelManager, CATEGORY_TO_MODEL

logger = logging.getLogger(__name__)


def fuzzy_match(s1: str, s2: str, threshold: float = 0.8) -> bool:
    """Check if two strings are similar enough to be considered the same entity."""
    s1_lower = s1.lower().strip()
    s2_lower = s2.lower().strip()

    # Exact match
    if s1_lower == s2_lower:
        return True

    # Substring match
    if s1_lower in s2_lower or s2_lower in s1_lower:
        return True

    # Fuzzy ratio
    ratio = SequenceMatcher(None, s1_lower, s2_lower).ratio()
    return ratio >= threshold


def cross_validate(
    text: str,
    extracted_entities: dict[str, list[str]],
    confidence_threshold: float = 0.5,
) -> dict:
    """
    Cross-validate LLM-extracted entities against NER model output.

    Args:
        text: Original clinical text
        extracted_entities: Dict of category -> list of entity strings from LLM
        confidence_threshold: Minimum NER confidence

    Returns:
        Dict with confirmed, missed_by_llm, llm_only, confidence_adjustments
    """
    manager = ModelManager()
    confirmed = []
    missed_by_llm = []
    llm_only = []
    confidence_adjustments = {}

    for category, llm_entities in extracted_entities.items():
        model_name = CATEGORY_TO_MODEL.get(category)
        if not model_name:
            logger.warning(f"No NER model for category '{category}', skipping")
            # All LLM entities in this category are unvalidated
            for entity_text in llm_entities:
                llm_only.append({
                    "text": entity_text,
                    "label": category.upper(),
                    "status": "llm_only",
                    "ner_confidence": None,
                    "model": None,
                })
            continue

        try:
            ner_entities = manager.analyze(text, model_name, confidence_threshold)
        except Exception as e:
            logger.error(f"NER analysis failed for '{model_name}': {e}")
            # Can't validate — mark all as llm_only
            for entity_text in llm_entities:
                llm_only.append({
                    "text": entity_text,
                    "label": category.upper(),
                    "status": "llm_only",
                    "ner_confidence": None,
                    "model": None,
                })
            continue

        # Track which NER entities were matched
        ner_matched = set()

        # Check each LLM entity against NER results
        for llm_text in llm_entities:
            found_match = False
            for i, ner_entity in enumerate(ner_entities):
                if fuzzy_match(llm_text, ner_entity["text"]):
                    confirmed.append({
                        "text": llm_text,
                        "label": ner_entity["label"],
                        "status": "confirmed",
                        "ner_confidence": ner_entity["confidence"],
                        "model": model_name,
                    })
                    # Boost confidence for confirmed entities
                    confidence_adjustments[llm_text] = min(
                        1.0,
                        ner_entity["confidence"] * 1.15  # 15% boost
                    )
                    ner_matched.add(i)
                    found_match = True
                    break

            if not found_match:
                llm_only.append({
                    "text": llm_text,
                    "label": category.upper(),
                    "status": "llm_only",
                    "ner_confidence": None,
                    "model": None,
                })
                # Reduce confidence for unconfirmed LLM entities
                confidence_adjustments[llm_text] = 0.6  # penalize

        # NER entities not found by LLM
        for i, ner_entity in enumerate(ner_entities):
            if i not in ner_matched:
                missed_by_llm.append({
                    "text": ner_entity["text"],
                    "label": ner_entity["label"],
                    "status": "missed_by_llm",
                    "ner_confidence": ner_entity["confidence"],
                    "model": model_name,
                })

    return {
        "confirmed": confirmed,
        "missed_by_llm": missed_by_llm,
        "llm_only": llm_only,
        "confidence_adjustments": confidence_adjustments,
    }
