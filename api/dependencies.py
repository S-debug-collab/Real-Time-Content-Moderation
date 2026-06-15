"""FastAPI dependency injection for model singleton."""

from __future__ import annotations

from typing import Optional

from src.inference import (
    BaseModerationModel,
    BaselineModerationModel,
    ONNXModerationModel,
    PyTorchModerationModel,
    load_model,
)


class ModelManager:
    """Singleton model manager — loads once at startup, serves all requests."""

    def __init__(self):
        self._model: Optional[BaseModerationModel] = None
        self._use_onnx: bool = True
        self._threshold: float = 0.5

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def backend(self) -> str:
        if self._model is None:
            return "none"
        if isinstance(self._model, ONNXModerationModel):
            return "onnx"
        if isinstance(self._model, BaselineModerationModel):
            return "baseline"
        if isinstance(self._model, PyTorchModerationModel):
            return "pytorch"
        return "unknown"

    @property
    def threshold(self) -> float:
        return self._threshold

    def load(self, use_onnx: bool = True, threshold: float = 0.5) -> None:
        self._use_onnx = use_onnx
        self._threshold = threshold
        self._model = load_model(use_onnx=use_onnx, threshold=threshold)

    def get_model(self) -> BaseModerationModel:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model

    def set_threshold(self, threshold: float) -> None:
        self._threshold = threshold
        if self._model is not None:
            self._model.threshold = threshold


model_manager = ModelManager()
