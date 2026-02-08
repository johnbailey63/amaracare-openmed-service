# AmaraCare OpenMed Service

FastAPI microservice wrapping [OpenMed](https://openmed.life)'s biomedical NER and PHI de-identification models for the [AmaraCare](https://www.amaracare.ai) cancer navigation platform.

## What It Does

- **`POST /ner`** — Extract biomedical entities (diseases, drugs, genes, anatomy) from clinical text
- **`POST /deidentify`** — Strip protected health information (PHI) from patient records
- **`POST /validate`** — Cross-validate LLM-extracted entities against NER ground truth
- **`GET /health`** — Service status and loaded models

## Models Preloaded

| Model | Size | Entities |
|---|---|---|
| `disease_detection_superclinical` | 434M | Diseases, conditions, diagnoses |
| `pharma_detection_superclinical` | 434M | Drugs, medications, treatments |
| `gene_detection_genecorpus` | 109M | Genes, proteins |
| `anatomy_detection_electramed` | 109M | Body parts, organs, anatomy |
| `pii_detection_superclinical` | 434M | Names, dates, SSNs, phones, emails, addresses |

## Quick Start

```bash
# Clone
git clone https://github.com/johnbailey63/amaracare-openmed-service.git
cd amaracare-openmed-service

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn app.main:app --reload --port 8000
```

## API Examples

### Extract entities
```bash
curl -X POST http://localhost:8000/ner \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient started on imatinib for chronic myeloid leukemia."}'
```

### De-identify PHI
```bash
curl -X POST http://localhost:8000/deidentify \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient John Smith, DOB 01/15/1960, SSN 123-45-6789", "method": "mask"}'
```

### Cross-validate LLM extraction
```bash
curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Patient started on imatinib for chronic myeloid leukemia.",
    "extracted_entities": {
      "drugs": ["imatinib"],
      "diseases": ["chronic myeloid leukemia"],
      "genes": [],
      "anatomy": []
    }
  }'
```

## Deployment (Railway)

1. Push to GitHub
2. Connect repo in [Railway](https://railway.app)
3. Set environment variables (see `.env.example`)
4. Railway auto-deploys on push to `main`

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Server port (Railway sets automatically) |
| `OPENMED_API_KEY` | — | API key for request authentication |
| `OPENMED_DEVICE` | `cpu` | `cpu` or `cuda` |
| `OPENMED_MODELS` | all 5 | Comma-separated model names to preload |
| `OPENMED_CACHE_DIR` | `/app/.cache` | Model cache directory |

## Architecture

```
Amara (Next.js / Vercel)
    |
    | HTTP POST with X-API-Key header
    v
OpenMed Service (FastAPI / Railway)
    |
    | Local inference (no external API calls)
    v
OpenMed Models (HuggingFace transformers, cached locally)
```

## License

Apache-2.0 — Same as OpenMed.
