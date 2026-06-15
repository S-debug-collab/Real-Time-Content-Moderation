"""Production inference engine supporting PyTorch and ONNX backends."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import (
    BASELINE_MODEL_PATH,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DISTILBERT_MODEL_NAME,
    ID2LABEL,
    LABEL2ID,
    MAX_SEQ_LENGTH,
    MODELS_DIR,
    ONNX_MODEL_FILENAME,
    PYTORCH_MODEL_DIR,
)
from src.preprocessing import clean_text


@dataclass
class PredictionResult:
    text: str
    label: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    flagged: bool = False


class BaseModerationModel(ABC):
    @abstractmethod
    def predict(self, text: str) -> PredictionResult:
        ...

    @abstractmethod
    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        ...


class PyTorchModerationModel(BaseModerationModel):
    """DistilBERT inference via HuggingFace PyTorch."""

    def __init__(
        self,
        model_path: Path | str = PYTORCH_MODEL_DIR,
        threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        device: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)

    def _infer_logits(self, texts: list[str]) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits.cpu().numpy()

        latency_ms = (time.perf_counter() - start) * 1000
        return logits, latency_ms

    def _logits_to_result(self, text: str, logits: np.ndarray, latency_ms: float) -> PredictionResult:
        probs = self._softmax(logits)
        pred_id = int(np.argmax(probs))
        label = ID2LABEL[pred_id]
        confidence = float(probs[pred_id])
        prob_dict = {ID2LABEL[i]: float(probs[i]) for i in range(len(probs))}

        return PredictionResult(
            text=text,
            label=label,
            confidence=round(confidence, 4),
            probabilities={k: round(v, 4) for k, v in prob_dict.items()},
            latency_ms=round(latency_ms, 2),
            flagged=label != "neutral" and confidence >= self.threshold,
        )

    def predict(self, text: str) -> PredictionResult:
        cleaned = clean_text(text)
        logits, latency_ms = self._infer_logits([cleaned])
        return self._logits_to_result(cleaned, logits[0], latency_ms)

    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        cleaned = [clean_text(t) for t in texts]
        logits, total_latency = self._infer_logits(cleaned)
        per_item_latency = total_latency / max(len(texts), 1)
        return [
            self._logits_to_result(cleaned[i], logits[i], per_item_latency)
            for i in range(len(texts))
        ]


class ONNXModerationModel(BaseModerationModel):
    """DistilBERT inference via ONNX Runtime for optimized low-latency serving."""

    def __init__(
        self,
        onnx_path: Path | str = MODELS_DIR / ONNX_MODEL_FILENAME,
        model_path: Path | str = PYTORCH_MODEL_DIR,
        threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        import onnxruntime as ort

        self.onnx_path = Path(onnx_path)
        self.threshold = threshold

        if not self.onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {self.onnx_path}. "
                "Run: python -m src.onnx_export"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [inp.name for inp in self.session.get_inputs()]

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)

    def _infer_logits(self, texts: list[str]) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="np",
        )

        ort_inputs = {
            "input_ids": encoding["input_ids"].astype(np.int64),
            "attention_mask": encoding["attention_mask"].astype(np.int64),
        }

        logits = self.session.run(None, ort_inputs)[0]
        latency_ms = (time.perf_counter() - start) * 1000
        return logits, latency_ms

    def _logits_to_result(self, text: str, logits: np.ndarray, latency_ms: float) -> PredictionResult:
        probs = self._softmax(logits)
        pred_id = int(np.argmax(probs))
        label = ID2LABEL[pred_id]
        confidence = float(probs[pred_id])

        return PredictionResult(
            text=text,
            label=label,
            confidence=round(confidence, 4),
            probabilities={ID2LABEL[i]: round(float(probs[i]), 4) for i in range(len(probs))},
            latency_ms=round(latency_ms, 2),
            flagged=label != "neutral" and confidence >= self.threshold,
        )

    def predict(self, text: str) -> PredictionResult:
        cleaned = clean_text(text)
        logits, latency_ms = self._infer_logits([cleaned])
        return self._logits_to_result(cleaned, logits[0], latency_ms)

    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        cleaned = [clean_text(t) for t in texts]
        logits, total_latency = self._infer_logits(cleaned)
        per_item_latency = total_latency / max(len(texts), 1)
        return [
            self._logits_to_result(cleaned[i], logits[i], per_item_latency)
            for i in range(len(texts))
        ]


class BaselineModerationModel(BaseModerationModel):
    """TF-IDF + Logistic Regression — fast fallback when DistilBERT is not trained."""

    def __init__(
        self,
        model_path: Path | str = BASELINE_MODEL_PATH,
        threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        import joblib

        self.model_path = Path(model_path)
        self.threshold = threshold
        if not self.model_path.exists():
            raise FileNotFoundError(f"Baseline model not found at {self.model_path}")
        self.pipeline = joblib.load(self.model_path)

    def _to_result(self, text: str, pred_id: int, probs: np.ndarray, latency_ms: float) -> PredictionResult:
        label = ID2LABEL[pred_id]
        confidence = float(probs[pred_id])
        return PredictionResult(
            text=text,
            label=label,
            confidence=round(confidence, 4),
            probabilities={ID2LABEL[i]: round(float(probs[i]), 4) for i in range(len(probs))},
            latency_ms=round(latency_ms, 2),
            flagged=label != "neutral" and confidence >= self.threshold,
        )

    def predict(self, text: str) -> PredictionResult:
        cleaned = clean_text(text)
        start = time.perf_counter()
        pred_id = int(self.pipeline.predict([cleaned])[0])
        probs = self.pipeline.predict_proba([cleaned])[0]
        latency_ms = (time.perf_counter() - start) * 1000
        return self._to_result(cleaned, pred_id, probs, latency_ms)

    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        cleaned = [clean_text(t) for t in texts]
        start = time.perf_counter()
        pred_ids = self.pipeline.predict(cleaned)
        probas = self.pipeline.predict_proba(cleaned)
        total_latency = (time.perf_counter() - start) * 1000
        per_item = total_latency / max(len(texts), 1)
        return [
            self._to_result(cleaned[i], int(pred_ids[i]), probas[i], per_item)
            for i in range(len(texts))
        ]


def load_model(
    use_onnx: bool = True,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    model_path: Optional[Path] = None,
) -> BaseModerationModel:
    """Factory function to load the appropriate inference backend."""
    model_path = model_path or PYTORCH_MODEL_DIR

    if use_onnx and (MODELS_DIR / ONNX_MODEL_FILENAME).exists() and model_path.exists():
        return ONNXModerationModel(threshold=threshold, model_path=model_path)

    if model_path.exists():
        return PyTorchModerationModel(model_path=model_path, threshold=threshold)

    if BASELINE_MODEL_PATH.exists():
        return BaselineModerationModel(threshold=threshold)

    raise FileNotFoundError(
        "No trained model found. Run: python scripts/run_pipeline.py --source demo --sample-size 2000"
    )


def log_misclassification(
    text: str,
    predicted: str,
    actual: str,
    confidence: float,
    log_path: Path = MODELS_DIR / "misclassified.jsonl",
) -> None:
    """Append misclassified examples for model improvement iteration."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "text": text,
        "predicted": predicted,
        "actual": actual,
        "confidence": confidence,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
