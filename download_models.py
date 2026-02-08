"""
Pre-download OpenMed models during Docker build.

This ensures models are cached in the image so the service starts instantly
without downloading models on first request.
"""

import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set cache directory
cache_dir = os.environ.get("OPENMED_CACHE_DIR", "/app/.cache")
os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir

MODELS_TO_DOWNLOAD = [
    "disease_detection_superclinical",
    "pharma_detection_superclinical",
    "gene_detection_genecorpus",
    "anatomy_detection_electramed",
    "pii_detection_superclinical",
]


def download_models():
    try:
        from openmed import analyze_text

        # Run a dummy inference on each model to trigger download + caching
        test_text = "Patient diagnosed with cancer."

        for model_name in MODELS_TO_DOWNLOAD:
            logger.info(f"Downloading model: {model_name}")
            try:
                analyze_text(test_text, model_name=model_name)
                logger.info(f"  ✓ {model_name} downloaded successfully")
            except Exception as e:
                logger.warning(f"  ✗ Failed to download {model_name}: {e}")

        logger.info("Model download complete.")

    except ImportError:
        logger.error("openmed package not installed. Skipping model download.")


if __name__ == "__main__":
    download_models()
