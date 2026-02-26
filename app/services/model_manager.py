"""Singleton model manager with lazy loading and LRU eviction for OpenMed models."""

import gc
import logging
import os
import time
import threading
import psutil
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

logger = logging.getLogger(__name__)


# ===========================================
# Default Models
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
    # Cancer-specific NER — trained on BioNLP 2013 Cancer Genetics corpus (27 entity types).
    # Loaded directly from HuggingFace (not an openmed alias). Complements the general-
    # purpose models above with higher-precision gene/cancer detection (F1=0.923, 93% prec).
    "oncology_pubmed": "Oncology entities — cancer, genes, chemicals (335M)",
    # PHI de-identification — loaded directly from HuggingFace
    "pii_bioclinical": "PHI/PII detection (149M)",
}

# The HuggingFace repo for the PII model (loaded directly, not via openmed alias)
PII_MODEL_HF_REPO = "OpenMed/OpenMed-PII-BioClinicalModern-Base-149M-v1"
PII_MODEL_NAME = "pii_bioclinical"

# OncologyDetect HuggingFace repo for the production model (oncology_pubmed).
# Loaded directly from HuggingFace via transformers, not via openmed alias.
ONCOLOGY_PUBMED_HF_REPO = "OpenMed/OpenMed-NER-OncologyDetect-PubMed-335M"
ONCOLOGY_PUBMED_MODEL_NAME = "oncology_pubmed"

# Other OncologyDetect variants — kept for on-demand comparison testing only.
# NOT in DEFAULT_MODELS. Loaded on-demand via `models` parameter on /ner.
ONCOLOGY_MODELS = {
    "oncology_superclinical": "OpenMed/OpenMed-NER-OncologyDetect-SuperClinical-434M",
    "oncology_multimed": "OpenMed/OpenMed-NER-OncologyDetect-MultiMed-568M",
    "oncology_pubmed": ONCOLOGY_PUBMED_HF_REPO,
}

# NER-only models (exclude PII from default NER runs)
NER_MODELS = {k: v for k, v in DEFAULT_MODELS.items() if k != PII_MODEL_NAME}

# Mapping from entity category to model name (for validation endpoint)
CATEGORY_TO_MODEL = {
    "drugs": "pharma_detection_superclinical",
    "diseases": "disease_detection_superclinical",
    "genes": "genome_detection_bioclinical",
    "anatomy": "anatomy_detection_electramed",
}


# ===========================================
# Lazy Loading & Eviction Config
# ===========================================

# Estimated RSS memory per model in MB (for pre-eviction sizing decisions).
# Actual RSS is used for threshold checks; these are for "will it fit?" predictions.
MODEL_MEMORY_ESTIMATES_MB: dict[str, float] = {
    "pii_bioclinical": 600,
    "disease_detection_superclinical": 1700,
    "pharma_detection_superclinical": 1700,
    "oncology_pubmed": 1300,
    "genome_detection_bioclinical": 400,
    "anatomy_detection_electramed": 400,
}

# Models that must never be evicted from memory
PINNED_MODELS: set[str] = {PII_MODEL_NAME}

# Default memory ceiling in MB — configurable via OPENMED_MAX_MEMORY_MB
DEFAULT_MAX_MEMORY_MB = 2048


class ModelManager:
    """
    Thread-safe singleton that manages OpenMed model lifecycle.

    Only critical models (PHI de-identification) are loaded at startup.
    NER models are lazy-loaded on first request and evicted LRU-style
    when memory pressure exceeds the configured ceiling.
    """

    _instance: Optional["ModelManager"] = None
    _singleton_lock = threading.Lock()

    def __new__(cls) -> "ModelManager":
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # Model storage — OrderedDict for LRU tracking (most recently used at end)
        self._models: OrderedDict[str, object] = OrderedDict()
        self._model_status: dict[str, str] = {}  # "loaded" | "available" | "loading" | "evicted" | "failed"
        self._model_load_times: dict[str, float] = {}
        self._model_last_used: dict[str, float] = {}
        self._eviction_count: dict[str, int] = {}
        self._start_time = time.time()
        self._pii_pipeline = None  # Separate reference for PII model

        # Thread safety — per-model locks + global lock for eviction decisions
        self._global_lock = threading.Lock()
        self._model_locks: dict[str, threading.Lock] = {}

        # Memory configuration
        self._max_memory_mb: float = float(
            os.environ.get("OPENMED_MAX_MEMORY_MB", str(DEFAULT_MAX_MEMORY_MB))
        )

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

    # ===========================================
    # Startup
    # ===========================================

    def _parse_preload_env(self) -> list[str]:
        """Parse which models to eagerly preload at startup."""
        # New env var takes priority
        env_val = os.environ.get("OPENMED_PRELOAD_MODELS", "")
        if env_val:
            return [m.strip() for m in env_val.split(",") if m.strip()]

        # Backward compat: old OPENMED_MODELS env var
        old_env = os.environ.get("OPENMED_MODELS", "")
        if old_env:
            logger.warning(
                "OPENMED_MODELS is deprecated — use OPENMED_PRELOAD_MODELS instead. "
                "With lazy loading, only critical models need preloading (default: pii_bioclinical)."
            )
            return [m.strip() for m in old_env.split(",") if m.strip()]

        # Default: only PII model (critical path for every chat turn)
        return [PII_MODEL_NAME]

    def preload_models(self):
        """Load only critical models at startup. Others are lazy-loaded on demand."""
        preload_list = self._parse_preload_env()

        # Register ALL default models as "available" (on disk, not in RAM)
        for model_name in DEFAULT_MODELS:
            if model_name not in self._model_status:
                self._model_status[model_name] = "available"

        # Preload only the configured subset
        for model_name in preload_list:
            if model_name in DEFAULT_MODELS or model_name in ONCOLOGY_MODELS:
                logger.info(f"Preloading critical model: {model_name}")
                self._load_model(model_name)
                self._touch_model(model_name)

        loaded = sum(1 for s in self._model_status.values() if s == "loaded")
        available = sum(1 for s in self._model_status.values() if s == "available")
        logger.info(
            f"Model init complete: {loaded} loaded (in RAM), "
            f"{available} available (on disk, lazy-load), "
            f"memory: {self.memory_mb:.0f} MB / {self._max_memory_mb:.0f} MB limit"
        )

    # ===========================================
    # LRU Tracking
    # ===========================================

    def _touch_model(self, model_name: str):
        """Mark model as recently used (move to end of LRU OrderedDict)."""
        self._model_last_used[model_name] = time.time()
        if model_name in self._models:
            self._models.move_to_end(model_name)

    def _get_model_lock(self, model_name: str) -> threading.Lock:
        """Get or create a per-model lock for thread-safe loading."""
        with self._global_lock:
            if model_name not in self._model_locks:
                self._model_locks[model_name] = threading.Lock()
            return self._model_locks[model_name]

    # ===========================================
    # Lazy Loading
    # ===========================================

    def _ensure_model_loaded(self, model_name: str):
        """Ensure a model is in memory, lazy-loading and evicting as needed. Thread-safe."""
        # Fast path: already loaded (no lock needed for read)
        if model_name in self._models and self._model_status.get(model_name) == "loaded":
            self._touch_model(model_name)
            return

        # Slow path: need to load — acquire per-model lock
        model_lock = self._get_model_lock(model_name)
        with model_lock:
            # Double-check after acquiring lock (another thread may have loaded it)
            if model_name in self._models and self._model_status.get(model_name) == "loaded":
                self._touch_model(model_name)
                return

            # Eviction needs the global lock (modifies shared state)
            with self._global_lock:
                estimated_mb = MODEL_MEMORY_ESTIMATES_MB.get(model_name, 500)
                self._evict_if_needed(estimated_mb)

            # Load (outside global lock to avoid blocking other threads during IO)
            logger.info(
                f"Lazy-loading model '{model_name}' "
                f"(memory: {self.memory_mb:.0f} MB / {self._max_memory_mb:.0f} MB)"
            )
            self._load_model(model_name)
            self._touch_model(model_name)

    def ensure_models_loaded(self, model_names: list[str]):
        """
        Load multiple models, parallelizing cold loads.

        When /ner is called with all 5 NER models and some are evicted, loading
        them sequentially would take 30-45s. Parallel loading reduces this to
        ~12-15s (limited by the single largest model).
        """
        needs_loading = [
            name for name in model_names
            if name not in self._models or self._model_status.get(name) != "loaded"
        ]

        if not needs_loading:
            # All models already loaded — just touch LRU
            for name in model_names:
                self._touch_model(name)
            return

        if len(needs_loading) == 1:
            self._ensure_model_loaded(needs_loading[0])
            return

        # Pre-evict for total estimated memory needed
        with self._global_lock:
            total_needed = sum(
                MODEL_MEMORY_ESTIMATES_MB.get(n, 500) for n in needs_loading
            )
            self._evict_if_needed(total_needed)

        # Parallel load via thread pool
        logger.info(f"Parallel-loading {len(needs_loading)} models: {needs_loading}")
        with ThreadPoolExecutor(max_workers=min(len(needs_loading), 3)) as executor:
            futures = {
                executor.submit(self._ensure_model_loaded, name): name
                for name in needs_loading
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Parallel load failed for '{name}': {e}")

    # ===========================================
    # LRU Eviction
    # ===========================================

    def _evict_if_needed(self, needed_mb: float):
        """
        Evict least-recently-used models until there's room for needed_mb.
        Must be called with self._global_lock held.
        """
        current_mb = self.memory_mb

        if current_mb + needed_mb <= self._max_memory_mb:
            return  # Enough room

        logger.info(
            f"Memory pressure: {current_mb:.0f} MB + ~{needed_mb:.0f} MB needed "
            f"> {self._max_memory_mb:.0f} MB limit. Starting eviction..."
        )

        # Walk LRU order (front = least recently used), skip pinned
        eviction_candidates = [
            name for name in self._models
            if name not in PINNED_MODELS and self._model_status.get(name) == "loaded"
        ]

        for model_name in eviction_candidates:
            if self.memory_mb + needed_mb <= self._max_memory_mb:
                break
            self._evict_model(model_name)

        # Force garbage collection after evictions
        gc.collect()
        logger.info(f"Post-eviction memory: {self.memory_mb:.0f} MB")

    def _evict_model(self, model_name: str):
        """Remove a model from memory, freeing its RAM."""
        if model_name in PINNED_MODELS:
            logger.warning(f"Refusing to evict pinned model '{model_name}'")
            return

        if model_name not in self._models:
            return

        last_used = self._model_last_used.get(model_name, 0)
        ago = time.time() - last_used if last_used else 0
        logger.info(f"Evicting model '{model_name}' (last used {ago:.0f}s ago)")

        # Delete the pipeline object
        pipeline = self._models.pop(model_name, None)
        del pipeline

        self._model_status[model_name] = "evicted"
        self._eviction_count[model_name] = self._eviction_count.get(model_name, 0) + 1

    # ===========================================
    # Model Loading (internal)
    # ===========================================

    def _load_model(self, model_name: str):
        """Load a single model into memory."""
        if model_name in self._models and self._model_status.get(model_name) == "loaded":
            return

        self._model_status[model_name] = "loading"
        start = time.time()

        try:
            if model_name == PII_MODEL_NAME:
                self._load_pii_model(model_name)
            elif model_name in ONCOLOGY_MODELS:
                self._load_oncology_model(model_name)
            else:
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

    def _load_oncology_model(self, model_name: str):
        """Load an OncologyDetect model directly from HuggingFace transformers."""
        start = time.time()

        try:
            from transformers import pipeline as hf_pipeline

            hf_repo = ONCOLOGY_MODELS[model_name]
            device = 0 if os.environ.get("OPENMED_DEVICE", "cpu") == "cuda" else -1
            pipe = hf_pipeline(
                "token-classification",
                model=hf_repo,
                device=device,
                aggregation_strategy="simple",
            )
            self._models[model_name] = pipe
            self._model_status[model_name] = "loaded"
            elapsed = (time.time() - start) * 1000
            self._model_load_times[model_name] = elapsed
            logger.info(f"Loaded OncologyDetect model '{hf_repo}' in {elapsed:.0f}ms")

        except ImportError:
            raise RuntimeError("transformers package required for OncologyDetect models. Install with: pip install transformers")

    # ===========================================
    # Inference
    # ===========================================

    def analyze(self, text: str, model_name: str, confidence_threshold: float = 0.5) -> list[dict]:
        """
        Run NER analysis on text using a model pipeline.
        Returns list of entity dicts: { text, label, confidence, start, end }

        Lazy-loads the model if not already in memory, evicting LRU models
        if needed to stay within the memory ceiling.
        """
        self._ensure_model_loaded(model_name)

        if self._model_status.get(model_name) != "loaded":
            raise RuntimeError(f"Model '{model_name}' is not available (status: {self._model_status.get(model_name)})")

        pipeline = self._models[model_name]

        # Fallback: if pipeline was stored as "api_mode" (ModelLoader unavailable),
        # use the openmed analyze_text API as a last resort
        if pipeline == "api_mode":
            return self._analyze_via_api(text, model_name, confidence_threshold)

        try:
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
        # Ensure PII model is loaded (pinned, so this is a fast no-op check)
        self._ensure_model_loaded(PII_MODEL_NAME)

        if self._model_status.get(PII_MODEL_NAME) != "loaded":
            raise RuntimeError(f"PII model not available (status: {self._model_status.get(PII_MODEL_NAME)})")

        if self._pii_pipeline is None:
            raise RuntimeError("PII pipeline not initialized")

        try:
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

    # ===========================================
    # Status & Health
    # ===========================================

    def get_loaded_models(self) -> list[dict]:
        """Return info about all models and their lifecycle status."""
        all_models = set(DEFAULT_MODELS.keys()) | set(self._models.keys()) | set(self._model_status.keys())
        return [
            {
                "name": name,
                "status": self._model_status.get(name, "unknown"),
                "load_time_ms": self._model_load_times.get(name),
                "last_used": self._model_last_used.get(name),
                "eviction_count": self._eviction_count.get(name, 0),
                "pinned": name in PINNED_MODELS,
                "estimated_mb": MODEL_MEMORY_ESTIMATES_MB.get(name),
            }
            for name in sorted(all_models)
        ]

    def get_available_ner_models(self) -> list[str]:
        """Return names of NER models that can be used (loaded, available on disk, or evicted).
        oncology_pubmed IS included (production model in DEFAULT_MODELS).
        oncology_superclinical/oncology_multimed are excluded (on-demand comparison only)."""
        # Experimental models that should NOT run by default
        experimental_oncology = {k for k in ONCOLOGY_MODELS if k != ONCOLOGY_PUBMED_MODEL_NAME}
        return [
            name for name in DEFAULT_MODELS
            if name != PII_MODEL_NAME
            and name not in experimental_oncology
            and self._model_status.get(name) in ("loaded", "available", "evicted")
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

    @property
    def memory_stats(self) -> dict:
        """Memory budget summary for the health endpoint."""
        rss = self.memory_mb
        return {
            "rss_mb": round(rss, 1),
            "max_mb": self._max_memory_mb,
            "utilization_pct": round((rss / self._max_memory_mb) * 100, 1) if self._max_memory_mb > 0 else 0,
            "models_in_ram": sum(1 for s in self._model_status.values() if s == "loaded"),
            "models_on_disk": sum(1 for s in self._model_status.values() if s in ("available", "evicted")),
            "total_evictions": sum(self._eviction_count.values()),
        }
