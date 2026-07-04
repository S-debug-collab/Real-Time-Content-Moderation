"""
app.py
------
FastAPI backend for the Real-Time Content Moderation System.

Exposes:
    POST /predict   -> classify a single comment
    GET  /health     -> simple health check
    GET  /            -> serves the frontend (frontend/index.html)

Run with:
    uvicorn app:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from predict import ModerationPredictor

app = FastAPI(
    title="Real-Time Content Moderation API",
    description="Classifies user comments as Neutral, Toxic, Offensive, or Hate Speech.",
    version="1.0.0",
)

# Allow the static frontend (served from any origin/port during local dev)
# to call the API without CORS errors.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once at startup, not per-request.
predictor: ModerationPredictor | None = None


@app.on_event("startup")
def load_model():
    global predictor
    predictor = ModerationPredictor()


class CommentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, examples=["You are stupid"])


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float


@app.post("/predict", response_model=PredictionResponse)
def predict(request: CommentRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is still loading, try again shortly.")
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    result = predictor.predict(request.text)
    return PredictionResponse(**result)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


# Serve the frontend static files (CSS/JS) and the index page.
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")
