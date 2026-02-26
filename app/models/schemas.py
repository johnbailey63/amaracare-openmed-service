"""Pydantic request/response models for the OpenMed API."""

from pydantic import BaseModel, Field
from typing import Optional


# ===========================================
# Shared Entity Model
# ===========================================

class NEREntity(BaseModel):
    """A single named entity extracted from text."""
    text: str = Field(..., description="The entity text span")
    label: str = Field(..., description="Entity type (e.g., DISEASE, DRUG, GENE)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    model: str = Field(..., description="Model that detected this entity")
    start: Optional[int] = Field(None, description="Start character offset in source text")
    end: Optional[int] = Field(None, description="End character offset in source text")


# ===========================================
# NER Endpoint
# ===========================================

class NERRequest(BaseModel):
    """Request body for POST /ner."""
    text: str = Field(..., min_length=1, max_length=50000, description="Clinical text to analyze")
    models: Optional[list[str]] = Field(
        None,
        description="Specific model names to run. If None, runs all preloaded NER models."
    )
    confidence_threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score to include an entity"
    )


class NERResponse(BaseModel):
    """Response body for POST /ner."""
    entities: list[NEREntity] = Field(default_factory=list)
    models_used: list[str] = Field(default_factory=list)
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    text_length: int = Field(..., description="Length of input text in characters")


# ===========================================
# De-identification Endpoint
# ===========================================

class DeidentifyRequest(BaseModel):
    """Request body for POST /deidentify."""
    text: str = Field(..., min_length=1, max_length=50000, description="Text containing PHI to de-identify")
    method: str = Field(
        "mask",
        pattern="^(mask|remove|replace)$",
        description="De-identification method: mask ([NAME]), remove, or replace (synthetic)"
    )
    confidence_threshold: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to treat as PHI (lower = more aggressive stripping)"
    )


class DeidentifyResponse(BaseModel):
    """Response body for POST /deidentify."""
    deidentified_text: str = Field(..., description="Text with PHI removed/masked/replaced")
    entities: list[NEREntity] = Field(default_factory=list, description="PHI entities detected")
    mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of placeholders to original values for re-hydration"
    )
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


# ===========================================
# Validation Endpoint
# ===========================================

class ExtractedEntities(BaseModel):
    """Entities extracted by an LLM that need cross-validation."""
    drugs: list[str] = Field(default_factory=list)
    diseases: list[str] = Field(default_factory=list)
    genes: list[str] = Field(default_factory=list)
    anatomy: list[str] = Field(default_factory=list)


class ValidateRequest(BaseModel):
    """Request body for POST /validate."""
    text: str = Field(..., min_length=1, max_length=50000, description="Original clinical text")
    extracted_entities: ExtractedEntities = Field(
        ...,
        description="Entities extracted by an LLM to cross-validate against NER"
    )
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)


class ValidatedEntity(BaseModel):
    """An entity with its validation status."""
    text: str
    label: str
    status: str = Field(..., description="confirmed | missed_by_llm | llm_only")
    ner_confidence: Optional[float] = Field(None, description="NER model confidence if found")
    model: Optional[str] = Field(None, description="NER model that found it")


class ValidateResponse(BaseModel):
    """Response body for POST /validate."""
    confirmed: list[ValidatedEntity] = Field(
        default_factory=list,
        description="Entities found by BOTH LLM and NER"
    )
    missed_by_llm: list[ValidatedEntity] = Field(
        default_factory=list,
        description="Entities found by NER but NOT by LLM"
    )
    llm_only: list[ValidatedEntity] = Field(
        default_factory=list,
        description="Entities found by LLM but NOT by NER (may be hallucinated)"
    )
    confidence_adjustments: dict[str, float] = Field(
        default_factory=dict,
        description="Suggested confidence adjustments for LLM entities"
    )
    processing_time_ms: float


# ===========================================
# Health Endpoint
# ===========================================

class ModelInfo(BaseModel):
    """Info about a model and its lifecycle state."""
    name: str
    status: str = Field(..., description="loaded | available | loading | evicted | failed")
    load_time_ms: Optional[float] = None
    pinned: bool = Field(False, description="Whether this model is pinned (never evicted)")
    eviction_count: int = Field(0, description="Number of times this model was evicted from memory")


class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status: str = Field(..., description="ok | degraded | error")
    loaded_models: list[ModelInfo] = Field(default_factory=list)
    models_available: int = 0
    models_loaded: int = Field(0, description="Models currently in RAM")
    models_on_disk: int = Field(0, description="Models on disk (lazy-loadable)")
    memory_mb: float = 0.0
    memory_max_mb: float = Field(0.0, description="Configured memory ceiling in MB")
    memory_utilization_pct: float = Field(0.0, description="Memory utilization percentage")
    total_evictions: int = Field(0, description="Total model evictions since startup")
    uptime_seconds: float = 0.0
    version: str = "0.1.0"
