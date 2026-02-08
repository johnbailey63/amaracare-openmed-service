FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models at build time so cold starts don't download them
# HF_TOKEN is needed for gated models (gene, PII). Set as build arg in Railway.
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}
COPY download_models.py .
RUN python download_models.py

# Copy application code
COPY app/ app/

# Set environment defaults
ENV PORT=8000
ENV OPENMED_DEVICE=cpu
ENV OPENMED_CACHE_DIR=/app/.cache
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE ${PORT}

# Run with uvicorn — use sh -c for proper $PORT expansion on Railway
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
