"""
Pre-download OpenMed models during Docker build.

This ensures models are cached in the image so the service starts instantly
without downloading models on first request.

NER models are downloaded via the openmed library's analyze_text() API.
The PII model is downloaded directly from HuggingFace via transformers,
since it's not registered as an openmed NER alias.

For gated models, set the HF_TOKEN environment variable with a HuggingFace
token that has access to the gated repositories.
"""

import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set cache directory
cache_dir = os.environ.get("OPENMED_CACHE_DIR", "/app/.cache")
os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir

# HuggingFace auth for gated models
hf_token = os.environ.get("HF_TOKEN", "")
if hf_token:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    logger.info("HF_TOKEN set — gated model downloads enabled")
else:
    logger.warning("HF_TOKEN not set — gated models may fail to download")

# NER models loaded via openmed library
NER_MODELS_TO_DOWNLOAD = [
    "disease_detection_superclinical",
    "pharma_detection_superclinical",
    "genome_detection_bioclinical",
    "anatomy_detection_electramed",
]

# PII model loaded directly from HuggingFace (not an openmed alias)
PII_MODEL_HF_REPO = "OpenMed/OpenMed-PII-BioClinicalModern-Base-149M-v1"

# OncologyDetect production model — cancer-specific NER (loaded directly from HuggingFace)
ONCOLOGY_PUBMED_HF_REPO = "OpenMed/OpenMed-NER-OncologyDetect-PubMed-335M"


def download_ner_models():
    """Download NER models via openmed's analyze_text API."""
    try:
        from openmed import analyze_text

        test_text = "Patient diagnosed with cancer."

        for model_name in NER_MODELS_TO_DOWNLOAD:
            logger.info(f"Downloading NER model: {model_name}")
            try:
                analyze_text(test_text, model_name=model_name)
                logger.info(f"  ✓ {model_name} downloaded successfully")
            except Exception as e:
                logger.warning(f"  ✗ Failed to download {model_name}: {e}")

    except ImportError:
        logger.error("openmed package not installed. Skipping NER model download.")


def download_pii_model():
    """Download PII model directly from HuggingFace via transformers."""
    try:
        from transformers import pipeline as hf_pipeline

        logger.info(f"Downloading PII model: {PII_MODEL_HF_REPO}")
        try:
            hf_pipeline(
                "token-classification",
                model=PII_MODEL_HF_REPO,
                device=-1,  # CPU
                aggregation_strategy="simple",
            )
            logger.info(f"  ✓ PII model downloaded successfully")
        except Exception as e:
            logger.warning(f"  ✗ Failed to download PII model: {e}")

    except ImportError:
        logger.error("transformers package not installed. Skipping PII model download.")


def download_oncology_model():
    """Download OncologyDetect-PubMed production model directly from HuggingFace."""
    try:
        from transformers import pipeline as hf_pipeline

        logger.info(f"Downloading OncologyDetect model: {ONCOLOGY_PUBMED_HF_REPO}")
        try:
            hf_pipeline(
                "token-classification",
                model=ONCOLOGY_PUBMED_HF_REPO,
                device=-1,  # CPU
                aggregation_strategy="simple",
            )
            logger.info(f"  ✓ OncologyDetect-PubMed downloaded successfully")
        except Exception as e:
            logger.warning(f"  ✗ Failed to download OncologyDetect-PubMed: {e}")

    except ImportError:
        logger.error("transformers package not installed. Skipping OncologyDetect model download.")


def download_models():
    download_ner_models()
    download_pii_model()
    download_oncology_model()
    logger.info("Model download complete.")


if __name__ == "__main__":
    download_models()
