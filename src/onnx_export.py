"""Export fine-tuned DistilBERT to ONNX format for optimized inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import MAX_SEQ_LENGTH, MODELS_DIR, ONNX_MODEL_FILENAME, PYTORCH_MODEL_DIR


def export_to_onnx(
    model_path: Path | str = PYTORCH_MODEL_DIR,
    output_path: Path | str = MODELS_DIR / ONNX_MODEL_FILENAME,
    opset_version: int = 14,
) -> Path:
    """
    Convert PyTorch DistilBERT to ONNX.

    Why ONNX improves performance:
    - Graph-level optimizations (operator fusion, constant folding)
    - ONNX Runtime uses highly optimized CPU kernels (MKL/OpenMP)
    - Eliminates Python/PyTorch overhead during inference
    - Enables cross-platform deployment (same model on cloud, edge, mobile)

    Tradeoffs:
    - Export is static — dynamic shapes require explicit configuration
    - Debugging is harder than native PyTorch
    - Training still requires PyTorch; ONNX is inference-only
    - Minor numerical differences possible (<0.1% probability delta)
    """
    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(
            f"PyTorch model not found at {model_path}. Train first: python -m src.train"
        )

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    dummy_text = "This is a sample comment for ONNX export."
    encoding = tokenizer(
        dummy_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    print(f"Exporting to ONNX (opset {opset_version})...")
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # Validate ONNX model
    import onnx

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model validated and saved to {output_path}")

    # Quick parity check
    _verify_parity(model, output_path, tokenizer, dummy_text)
    return output_path


def _verify_parity(model, onnx_path: Path, tokenizer, text: str, tolerance: float = 1e-3) -> None:
    """Verify PyTorch and ONNX outputs match within tolerance."""
    import numpy as np
    import onnxruntime as ort

    encoding = tokenizer(text, return_tensors="pt", padding="max_length", max_length=MAX_SEQ_LENGTH, truncation=True)

    with torch.no_grad():
        pt_logits = model(**encoding).logits.numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        "input_ids": encoding["input_ids"].numpy().astype(np.int64),
        "attention_mask": encoding["attention_mask"].numpy().astype(np.int64),
    }
    onnx_logits = session.run(None, ort_inputs)[0]

    max_diff = np.max(np.abs(pt_logits - onnx_logits))
    if max_diff > tolerance:
        print(f"WARNING: PyTorch vs ONNX max diff = {max_diff:.6f} (tolerance={tolerance})")
    else:
        print(f"Parity check passed (max diff = {max_diff:.6f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DistilBERT to ONNX")
    parser.add_argument("--model-path", default=str(PYTORCH_MODEL_DIR))
    parser.add_argument("--output-path", default=str(MODELS_DIR / ONNX_MODEL_FILENAME))
    parser.add_argument("--opset", type=int, default=14)
    args = parser.parse_args()

    export_to_onnx(args.model_path, args.output_path, args.opset)
