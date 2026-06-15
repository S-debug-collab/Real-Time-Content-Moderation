"""Benchmark PyTorch vs ONNX inference latency."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from src.config import (
    BENCHMARK_TIMED_RUNS,
    BENCHMARK_WARMUP_RUNS,
    MODELS_DIR,
    PYTORCH_MODEL_DIR,
)
from src.inference import ONNXModerationModel, PyTorchModerationModel


SAMPLE_TEXTS = [
    "Thanks for the helpful tutorial!",
    "You are an idiot and nobody likes you.",
    "People like you don't belong here.",
    "This movie was pretty good overall.",
    "Shut up moron, learn to read.",
    "Looking forward to the team meeting tomorrow.",
    "I hope something bad happens to your family.",
    "Great presentation, well done everyone.",
]


def benchmark_model(model, texts: list[str], warmup: int, runs: int) -> dict:
    """Measure single-text and batch inference latency."""
    # Warmup
    for _ in range(warmup):
        for text in texts:
            model.predict(text)
        model.predict_batch(texts)

    # Single-text latency
    single_latencies = []
    for _ in range(runs):
        for text in texts:
            start = time.perf_counter()
            model.predict(text)
            single_latencies.append((time.perf_counter() - start) * 1000)

    # Batch latency
    batch_latencies = []
    for _ in range(runs // 4):
        start = time.perf_counter()
        model.predict_batch(texts)
        batch_latencies.append((time.perf_counter() - start) * 1000)

    return {
        "single_mean_ms": round(statistics.mean(single_latencies), 2),
        "single_median_ms": round(statistics.median(single_latencies), 2),
        "single_p95_ms": round(sorted(single_latencies)[int(len(single_latencies) * 0.95)], 2),
        "single_p99_ms": round(sorted(single_latencies)[int(len(single_latencies) * 0.99)], 2),
        "batch_mean_ms": round(statistics.mean(batch_latencies), 2),
        "batch_per_item_ms": round(statistics.mean(batch_latencies) / len(texts), 2),
        "warmup_runs": warmup,
        "timed_runs": runs,
    }


def run_benchmark(warmup: int = BENCHMARK_WARMUP_RUNS, runs: int = BENCHMARK_TIMED_RUNS) -> dict:
    """Compare PyTorch vs ONNX latency."""
    results = {"texts_count": len(SAMPLE_TEXTS), "benchmark_texts": SAMPLE_TEXTS}

    if PYTORCH_MODEL_DIR.exists():
        print("Benchmarking PyTorch model...")
        pt_model = PyTorchModerationModel()
        results["pytorch"] = benchmark_model(pt_model, SAMPLE_TEXTS, warmup, runs)
        print(f"  PyTorch single mean: {results['pytorch']['single_mean_ms']}ms")
    else:
        print(f"PyTorch model not found at {PYTORCH_MODEL_DIR}, skipping.")

    try:
        print("Benchmarking ONNX model...")
        onnx_model = ONNXModerationModel()
        results["onnx"] = benchmark_model(onnx_model, SAMPLE_TEXTS, warmup, runs)
        print(f"  ONNX single mean: {results['onnx']['single_mean_ms']}ms")
    except FileNotFoundError as exc:
        print(f"ONNX benchmark skipped: {exc}")

    if "pytorch" in results and "onnx" in results:
        speedup = results["pytorch"]["single_mean_ms"] / results["onnx"]["single_mean_ms"]
        results["speedup_factor"] = round(speedup, 2)
        print(f"\nONNX speedup: {speedup:.2f}x")

        target_ms = 100
        onnx_p95 = results["onnx"]["single_p95_ms"]
        results["meets_100ms_target"] = onnx_p95 < target_ms
        print(f"P95 latency: {onnx_p95}ms — {'PASS' if onnx_p95 < target_ms else 'FAIL'} (<{target_ms}ms target)")

    output_path = MODELS_DIR / "benchmark_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark inference latency")
    parser.add_argument("--warmup", type=int, default=BENCHMARK_WARMUP_RUNS)
    parser.add_argument("--runs", type=int, default=BENCHMARK_TIMED_RUNS)
    args = parser.parse_args()
    run_benchmark(warmup=args.warmup, runs=args.runs)
