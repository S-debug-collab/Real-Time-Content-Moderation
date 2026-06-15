"""Pydantic request/response schemas for the moderation API."""

from typing import Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to classify")
    threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Override confidence threshold for flagging"
    )


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64)
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    label: str
    confidence: float
    flagged: bool
    probabilities: dict[str, float]
    latency_ms: float
    backend: str


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    total_latency_ms: float
    backend: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    backend: str
    threshold: float


class ThresholdUpdateRequest(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0)


class MetricsResponse(BaseModel):
    accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    macro_precision: Optional[float] = None
    macro_recall: Optional[float] = None
    per_class: Optional[dict] = None
