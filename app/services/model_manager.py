"""Singleton model manager for loading and caching OpenMed models."""

import logging
import os
import re
import time
import threading
import psutil
from typing import Optional

logger = logging.getLogger(__name__)


# ===========================================
# Default Models to Preload
# ===========================================

# Model names are OpenMed library aliases that resolve to HuggingFace repos:
#   disease_detection_superclinical -> OpenMed/OpenMed-NER-DiseaseDetect-SuperClinical-434M
#   pharma_detection_superclinical  -> OpenMed/OpenMed-NER-PharmaDetect-SuperClinical-434M
#   genome_detection_bioclinical    -> OpenMed/OpenMed-NER-GenomeDetect-BioClinical-108M
#   anatomy_detection_electramed    -> OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-109M
#
# PII model uses a direct HuggingFace repo (not an OpenMed alias) since the
# openmed library doesn't have a PII alias in its registry.

DEFAULT_MODELS = {
    # NER models for clinical entity extraction
    "disease_detection_superclinical": "Disease entities (434M)",
    "pharma_detection_superclinical": "Drug/medication entities (434M)",
    "genome_detection_bioclinical": "Gene/protein entities (108M)",
    "anatomy_detection_electramed": "Anatomy/body part entities (109M)",
    # PHI de-identification — loaded directly from HuggingFace
    "pii_bioclinical": "PHI/PII detection (149M)",
}

# The HuggingFace repo for the PII model (loaded directly, not via openmed alias)
PII_MODEL_HF_REPO = "OpenMed/OpenMed-PII-BioClinicalModern-Base-149M-v1"
PII_MODEL_NAME = "pii_bioclinical"

# NER-only models (exclude PII from default NER runs)
NER_MODELS = {k: v for k, v in DEFAULT_MODELS.items() if k != PII_MODEL_NAME}

# Mapping from entity category to model name (for validation endpoint)
CATEGORY_TO_MODEL = {
    "drugs": "pharma_detection_superclinical",
    "diseases": "disease_detection_superclinical",
    "genes": "genome_detection_bioclinical",
    "anatomy": "anatomy_detection_electramed",
}


class ModelManager:
    """
    Thread-safe singleton that loads and caches OpenMed models.
    Models are loaded into memory on startup and reused across requests.
    """

    _instance: Optional["ModelManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._models: dict[str, object] = {}
        self._model_status: dict[str, str] = {}  # "loaded" | "loading" | "failed"
        self._model_load_times: dict[str, float] = {}
        self._start_time = time.time()
        self._inference_lock = threading.Lock()
        self._pii_pipeline = None  # Separate pipeline for PII model

        # Configure OpenMed cache directory
        cache_dir = os.environ.get("OPENMED_CACHE_DIR", None)
        if cache_dir:
            os.environ["HF_HOME"] = cache_dir
            os.environ["TRANSFORMERS_CACHE"] = cache_dir

        # Configure HuggingFace auth for gated models
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
            logger.info("HF_TOKEN configured — gated model access enabled")

    def preload_models(self):
        """Load all default models into memory. Called on app startup."""
        models_env = os.environ.get("OPENMED_MODELS", "")
        if models_env:
            model_names = [m.strip() for m in models_env.split(",") if m.strip()]
        else:
            model_names = list(DEFAULT_MODELS.keys())

        logger.info(f"Preloading {len(model_names)} models...")

        for model_name in model_names:
            self._load_model(model_name)

        loaded = sum(1 for s in self._model_status.values() if s == "loaded")
        failed = sum(1 for s in self._model_status.values() if s == "failed")
        logger.info(f"Model preload complete: {loaded} loaded, {failed} failed")

    def _load_model(self, model_name: str):
        """Load a single model into memory."""
        if model_name in self._models:
            return

        self._model_status[model_name] = "loading"
        start = time.time()

        try:
            if model_name == PII_MODEL_NAME:
                # PII model loaded directly from HuggingFace via transformers
                self._load_pii_model(model_name)
            else:
                # NER models loaded via openmed library
                self._load_ner_model(model_name)

        except Exception as e:
            self._model_status[model_name] = "failed"
            self._model_load_times[model_name] = (time.time() - start) * 1000
            logger.error(f"Failed to load model '{model_name}': {e}")

    def _load_ner_model(self, model_name: str):
        """Load an NER model via the openmed library."""
        start = time.time()

        try:
            from openmed import ModelLoader

            loader = ModelLoader()
            pipeline = loader.create_pipeline(
                model_name=model_name,
                task="token-classification",
                aggregation_strategy="simple",
            )
            self._models[model_name] = pipeline
            self._model_status[model_name] = "loaded"
            elapsed = (time.time() - start) * 1000
            self._model_load_times[model_name] = elapsed
            logger.info(f"Loaded NER model '{model_name}' in {elapsed:.0f}ms")

        except ImportError:
            # ModelLoader not available — use analyze_text API instead
            logger.warning(f"ModelLoader not available, will use analyze_text() for '{model_name}'")
            self._models[model_name] = "api_mode"
            self._model_status[model_name] = "loaded"
            elapsed = (time.time() - start) * 1000
            self._model_load_times[model_name] = elapsed

    def _load_pii_model(self, model_name: str):
        """Load the PII model directly from HuggingFace transformers."""
        start = time.time()

        try:
            from transformers import pipeline as hf_pipeline

            device = 0 if os.environ.get("OPENMED_DEVICE", "cpu") == "cuda" else -1
            pii_pipe = hf_pipeline(
                "token-classification",
                model=PII_MODEL_HF_REPO,
                device=device,
                aggregation_strategy="simple",
            )
            self._pii_pipeline = pii_pipe
            self._models[model_name] = pii_pipe
            self._model_status[model_name] = "loaded"
            elapsed = (time.time() - start) * 1000
            self._model_load_times[model_name] = elapsed
            logger.info(f"Loaded PII model '{PII_MODEL_HF_REPO}' in {elapsed:.0f}ms")

        except ImportError:
            raise RuntimeError("transformers package required for PII model. Install with: pip install transformers")

    def analyze(self, text: str, model_name: str, confidence_threshold: float = 0.5) -> list[dict]:
        """
        Run NER analysis on text using the preloaded pipeline.
        Returns list of entity dicts: { text, label, confidence, start, end }

        Uses the cached HuggingFace pipeline directly (loaded at startup)
        instead of calling openmed.analyze_text() which would re-download
        models from HuggingFace on every request.
        """
        # Lazy-load if not preloaded
        if model_name not in self._models:
            self._load_model(model_name)

        if self._model_status.get(model_name) != "loaded":
            raise RuntimeError(f"Model '{model_name}' is not available (status: {self._model_status.get(model_name)})")

        pipeline = self._models[model_name]

        # Fallback: if pipeline was stored as "api_mode" (ModelLoader unavailable),
        # use the openmed analyze_text API as a last resort
        if pipeline == "api_mode":
            return self._analyze_via_api(text, model_name, confidence_threshold)

        try:
            with self._inference_lock:
                raw_predictions = pipeline(text)

            # Post-process raw HuggingFace predictions into clean entity dicts.
            # The pipeline uses aggregation_strategy="simple" so predictions have
            # entity_group, score, word, start, end keys.
            entities = []
            for pred in raw_predictions:
                score = pred.get("score", 0.0)
                if score < confidence_threshold:
                    continue

                # Extract entity text: prefer span from original text, fall back to token word
                start = pred.get("start")
                end = pred.get("end")
                if isinstance(start, int) and isinstance(end, int):
                    entity_text = text[start:end].strip()
                else:
                    entity_text = pred.get("word", "")

                # Clean up tokenizer artifacts (subword markers)
                if not entity_text:
                    entity_text = pred.get("word", "").replace("##", "").replace("▁", " ").strip()

                # Clean label: remove B-/I- prefix if present
                raw_label = pred.get("entity_group") or pred.get("entity") or "UNKNOWN"
                label = raw_label.replace("B-", "").replace("I-", "")

                if entity_text:
                    entities.append({
                        "text": entity_text,
                        "label": label,
                        "confidence": score,
                        "start": start,
                        "end": end,
                    })

            # Merge adjacent entities with the same label.
            # Some models (especially genome_detection) produce per-character
            # predictions for gene names (E, G, F, R instead of EGFR).
            # We merge entities that are adjacent or overlapping with the same label.
            merged = self._merge_adjacent_entities(entities, text)

            return merged

        except Exception as e:
            logger.error(f"Inference failed for model '{model_name}': {e}")
            raise

    def _analyze_via_api(self, text: str, model_name: str, confidence_threshold: float) -> list[dict]:
        """Fallback: use openmed.analyze_text() when ModelLoader was unavailable at startup."""
        try:
            from openmed import analyze_text

            with self._inference_lock:
                result = analyze_text(
                    text,
                    model_name=model_name,
                    confidence_threshold=confidence_threshold,
                )

            entities = []
            for entity in result.entities:
                entities.append({
                    "text": entity.text,
                    "label": entity.label,
                    "confidence": entity.confidence,
                    "start": getattr(entity, "start", None),
                    "end": getattr(entity, "end", None),
                })

            return entities

        except Exception as e:
            logger.error(f"Fallback inference via analyze_text() failed for model '{model_name}': {e}")
            raise

    @staticmethod
    def _merge_adjacent_entities(entities: list[dict], original_text: str) -> list[dict]:
        """
        Merge adjacent entities with the same label into single entities.

        Some NER models (especially genome_detection_bioclinical) tokenize gene
        names at the character level, producing E, G, F, R instead of EGFR.
        This merges adjacent same-label entities by checking if they touch or
        are separated by at most 1 character in the original text.
        """
        if not entities:
            return entities

        # Sort by start position
        sorted_ents = sorted(
            [e for e in entities if isinstance(e.get("start"), int)],
            key=lambda e: e["start"],
        )
        # Keep entities without position info as-is
        no_position = [e for e in entities if not isinstance(e.get("start"), int)]

        if not sorted_ents:
            return no_position

        merged = [sorted_ents[0].copy()]

        for ent in sorted_ents[1:]:
            prev = merged[-1]

            # Merge if same label and adjacent (gap ≤ 1 character)
            if (
                ent["label"] == prev["label"]
                and isinstance(prev.get("end"), int)
                and isinstance(ent.get("start"), int)
                and ent["start"] <= prev["end"] + 1
            ):
                # Extend the previous entity
                prev["end"] = max(prev["end"], ent["end"])
                prev["text"] = original_text[prev["start"]:prev["end"]].strip()
                # Average the confidence scores
                prev["confidence"] = (prev["confidence"] + ent["confidence"]) / 2
            else:
                merged.append(ent.copy())

        return merged + no_position

    def deidentify(self, text: str, method: str = "mask", confidence_threshold: float = 0.3) -> dict:
        """
        De-identify PHI in text using the PII NER model.

        Approach: Run the PII token-classification model to detect PHI entities,
        then mask/remove/replace them in the original text.

        Returns: { deidentified_text, entities, mapping }
        """
        # Ensure PII model is loaded
        if PII_MODEL_NAME not in self._models:
            self._load_model(PII_MODEL_NAME)

        if self._model_status.get(PII_MODEL_NAME) != "loaded":
            raise RuntimeError(f"PII model not available (status: {self._model_status.get(PII_MODEL_NAME)})")

        if self._pii_pipeline is None:
            raise RuntimeError("PII pipeline not initialized")

        try:
            with self._inference_lock:
                raw_entities = self._pii_pipeline(text)

            # Filter by confidence and build entity list
            entities = []
            for ent in raw_entities:
                score = ent.get("score", 0)
                if score < confidence_threshold:
                    continue

                entities.append({
                    "text": ent.get("word", ""),
                    "label": ent.get("entity_group", ent.get("entity", "PHI")),
                    "confidence": score,
                    "start": ent.get("start"),
                    "end": ent.get("end"),
                })

            # Sort entities by position (end to start) for safe replacement
            sorted_entities = sorted(entities, key=lambda e: e.get("start", 0), reverse=True)

            # Build mapping and de-identified text
            mapping = {}
            placeholder_counts: dict[str, int] = {}
            deidentified = text

            for ent in sorted_entities:
                label = ent["label"]
                start = ent.get("start")
                end = ent.get("end")
                original = ent["text"]

                if start is None or end is None:
                    continue

                # Generate placeholder key
                count = placeholder_counts.get(label, 0)
                placeholder_key = f"[{label}_{count}]"
                placeholder_counts[label] = count + 1
                mapping[placeholder_key] = original

                # Apply de-identification method
                if method == "mask":
                    replacement = placeholder_key
                elif method == "remove":
                    replacement = ""
                elif method == "replace":
                    replacement = self._generate_synthetic(label)
                else:
                    replacement = placeholder_key

                deidentified = deidentified[:start] + replacement + deidentified[end:]

            return {
                "deidentified_text": deidentified,
                "entities": entities,
                "mapping": mapping,
            }

        except Exception as e:
            logger.error(f"De-identification failed: {e}")
            raise

    @staticmethod
    def _generate_synthetic(label: str) -> str:
        """Generate a synthetic replacement value for a PHI label."""
        synthetics = {
            "FIRST_NAME": "Jane",
            "LAST_NAME": "Doe",
            "NAME": "Jane Doe",
            "DATE": "2000-01-01",
            "DATE_OF_BIRTH": "1960-01-01",
            "AGE": "60",
            "LOCATION": "Springfield",
            "HOSPITAL": "General Hospital",
            "PHONE": "555-0100",
            "EMAIL": "patient@example.com",
            "ID": "000-00-0000",
            "SSN": "000-00-0000",
            "MRN": "MRN-000000",
        }
        return synthetics.get(label.upper(), f"[{label}]")

    def get_loaded_models(self) -> list[dict]:
        """Return info about all models and their status."""
        return [
            {
                "name": name,
                "status": self._model_status.get(name, "unknown"),
                "load_time_ms": self._model_load_times.get(name),
            }
            for name in {**DEFAULT_MODELS, **{k: "" for k in self._models}}
        ]

    def get_available_ner_models(self) -> list[str]:
        """Return names of loaded NER models (excluding PII)."""
        return [
            name for name in self._models
            if self._model_status.get(name) == "loaded"
            and name != PII_MODEL_NAME
        ]

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    @property
    def memory_mb(self) -> float:
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
