"""
Real-Time Content Moderation API

System Design:
    User/Client → POST /predict → FastAPI → Preprocess → Model Inference (ONNX/PyTorch)
                                                                    ↓
                                                              Log prediction
                                                                    ↓
                                                              JSON response (<100ms)
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.dependencies import model_manager
from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
    ThresholdUpdateRequest,
)
from src.config import (
    API_HOST,
    API_PORT,
    DEFAULT_CONFIDENCE_THRESHOLD,
    LOGS_DIR,
    METRICS_PATH,
    MODELS_DIR,
    USE_ONNX_BY_DEFAULT,
)
from src.inference import PredictionResult

# Logging setup
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "predictions.log"),
    ],
)
logger = logging.getLogger("moderation_api")


def _result_to_response(result: PredictionResult, backend: str) -> PredictResponse:
    return PredictResponse(
        label=result.label,
        confidence=result.confidence,
        flagged=result.flagged,
        probabilities=result.probabilities,
        latency_ms=result.latency_ms,
        backend=backend,
    )


def _log_prediction(text: str, result: PredictionResult, client_ip: str) -> None:
    """Structured prediction logging for monitoring and audit."""
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "client_ip": client_ip,
        "text_preview": text[:100],
        "label": result.label,
        "confidence": result.confidence,
        "flagged": result.flagged,
        "latency_ms": result.latency_ms,
    }
    logger.info(json.dumps(log_entry))

    # Append to JSONL for analytics
    with open(LOGS_DIR / "predictions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup — avoids per-request loading overhead."""
    try:
        model_manager.load(use_onnx=USE_ONNX_BY_DEFAULT, threshold=DEFAULT_CONFIDENCE_THRESHOLD)
        logger.info(f"Model loaded (backend={model_manager.backend})")
    except FileNotFoundError as exc:
        logger.warning(f"Model not loaded at startup: {exc}. API will return 503 until trained.")
    yield


app = FastAPI(
    title="Real-Time Content Moderation API",
    description="Detect toxic, hate speech, and offensive content in real time using DistilBERT + ONNX",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend demo
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/")
async def root():
    index = frontend_dir / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "Content Moderation API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "degraded",
        model_loaded=model_manager.is_loaded,
        backend=model_manager.backend,
        threshold=model_manager.threshold,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, req: Request):
    """
    Classify a single text for toxic content.

    Returns label (neutral/toxic/hate_speech/offensive), confidence score,
    per-class probabilities, and whether content is flagged above threshold.
    """
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")

    model = model_manager.get_model()
    if request.threshold is not None:
        model.threshold = request.threshold

    result = model.predict(request.text)
    _log_prediction(request.text, result, req.client.host if req.client else "unknown")

    if request.threshold is not None:
        model.threshold = model_manager.threshold

    return _result_to_response(result, model_manager.backend)


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """Batch inference for multiple texts in a single request."""
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    model = model_manager.get_model()
    if request.threshold is not None:
        model.threshold = request.threshold

    start = time.perf_counter()
    results = model.predict_batch(request.texts)
    total_latency = round((time.perf_counter() - start) * 1000, 2)

    if request.threshold is not None:
        model.threshold = model_manager.threshold

    return BatchPredictResponse(
        predictions=[_result_to_response(r, model_manager.backend) for r in results],
        total_latency_ms=total_latency,
        backend=model_manager.backend,
    )


@app.put("/threshold")
async def update_threshold(request: ThresholdUpdateRequest):
    """
    Tune moderation strictness at runtime.

    Higher threshold (e.g. 0.8) = fewer false positives, more lenient.
    Lower threshold (e.g. 0.3) = more aggressive filtering, higher recall.
    """
    model_manager.set_threshold(request.threshold)
    return {"threshold": request.threshold, "message": "Threshold updated"}


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Return saved model evaluation metrics."""
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Metrics not available. Train the model first.")

    with open(METRICS_PATH) as f:
        data = json.load(f)

    return MetricsResponse(
        accuracy=data.get("accuracy"),
        macro_f1=data.get("macro_f1"),
        macro_precision=data.get("macro_precision"),
        macro_recall=data.get("macro_recall"),
        per_class=data.get("per_class"),
    )


def main():
    import uvicorn

    uvicorn.run("api.app:app", host=API_HOST, port=API_PORT, reload=False)


if __name__ == "__main__":
    main()
