"""Singleton model manager for loading and caching OpenMed models."""

import logging
import os
import time
import threading
import psutil
from typing import Optional

logger = logging.getLogger(__name__)


# ===========================================
# Default Models to Preload
# ===========================================

DEFAULT_MODELS = {
    # NER models for clinical entity extraction
    "disease_detection_superclinical": "Disease entities (434M)",
    "pharma_detection_superclinical": "Drug/medication entities (434M)",
    "gene_detection_genecorpus": "Gene/protein entities (109M)",
    "anatomy_detection_electramed": "Anatomy/body part entities (109M)",
    # PHI de-identification
    "pii_detection_superclinical": "PHI/PII detection (434M)",
}

# NER-only models (exclude PII from default NER runs)
NER_MODELS = {k: v for k, v in DEFAULT_MODELS.items() if k != "pii_detection_superclinical"}

# Mapping from entity category to model name (for validation endpoint)
CATEGORY_TO_MODEL = {
    "drugs": "pharma_detection_superclinical",
    "diseases": "disease_detection_superclinical",
    "genes": "gene_detection_genecorpus",
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

        # Configure OpenMed cache directory
        cache_dir = os.environ.get("OPENMED_CACHE_DIR", None)
        if cache_dir:
            os.environ["HF_HOME"] = cache_dir
            os.environ["TRANSFORMERS_CACHE"] = cache_dir

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
            # Import openmed here to avoid import errors if not installed
            from openmed import ModelLoader

            loader = ModelLoader()
            device = os.environ.get("OPENMED_DEVICE", "cpu")
            pipeline = loader.create_pipeline(model_name=model_name, task="ner", device=device)
            self._models[model_name] = pipeline
            self._model_status[model_name] = "loaded"
            elapsed = (time.time() - start) * 1000
            self._model_load_times[model_name] = elapsed
            logger.info(f"Loaded model '{model_name}' in {elapsed:.0f}ms")

        except ImportError:
            # OpenMed not installed — use analyze_text API instead
            logger.warning(f"ModelLoader not available, will use analyze_text() for '{model_name}'")
            self._models[model_name] = "api_mode"
            self._model_status[model_name] = "loaded"
            elapsed = (time.time() - start) * 1000
            self._model_load_times[model_name] = elapsed

        except Exception as e:
            self._model_status[model_name] = "failed"
            self._model_load_times[model_name] = (time.time() - start) * 1000
            logger.error(f"Failed to load model '{model_name}': {e}")

    def analyze(self, text: str, model_name: str, confidence_threshold: float = 0.5) -> list[dict]:
        """
        Run NER analysis on text using a specific model.
        Returns list of entity dicts: { text, label, confidence, start, end }
        """
        # Lazy-load if not preloaded
        if model_name not in self._models:
            self._load_model(model_name)

        if self._model_status.get(model_name) != "loaded":
            raise RuntimeError(f"Model '{model_name}' is not available (status: {self._model_status.get(model_name)})")

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
            logger.error(f"Inference failed for model '{model_name}': {e}")
            raise

    def deidentify(self, text: str, method: str = "mask", confidence_threshold: float = 0.3) -> dict:
        """
        De-identify PHI in text.
        Returns: { deidentified_text, entities, mapping }
        """
        try:
            from openmed import extract_pii, deidentify

            with self._inference_lock:
                # First extract PII entities
                pii_result = extract_pii(
                    text,
                    model_name="pii_detection_superclinical",
                    confidence_threshold=confidence_threshold,
                )

                # Then de-identify
                deid_result = deidentify(
                    text,
                    method=method,
                    model_name="pii_detection_superclinical",
                    confidence_threshold=confidence_threshold,
                )

            # Build a mapping from placeholder to original for re-hydration
            mapping = {}
            entities = []
            placeholder_counts: dict[str, int] = {}

            for entity in pii_result.entities:
                label = entity.label
                count = placeholder_counts.get(label, 0)
                placeholder_key = f"[{label}]_{count}"
                placeholder_counts[label] = count + 1
                mapping[placeholder_key] = entity.text

                entities.append({
                    "text": entity.text,
                    "label": entity.label,
                    "confidence": entity.confidence,
                    "start": getattr(entity, "start", None),
                    "end": getattr(entity, "end", None),
                })

            return {
                "deidentified_text": deid_result.deidentified_text,
                "entities": entities,
                "mapping": mapping,
            }

        except Exception as e:
            logger.error(f"De-identification failed: {e}")
            raise

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
            and name != "pii_detection_superclinical"
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
