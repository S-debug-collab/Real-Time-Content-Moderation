#!/usr/bin/env python3
"""End-to-end pipeline: prepare data → train baseline → train DistilBERT → export ONNX → benchmark."""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Run full moderation ML pipeline")
    parser.add_argument("--source", default="demo", choices=["demo", "huggingface", "jigsaw", "twitter"])
    parser.add_argument("--sample-size", type=int, default=2000, help="Use 20000+ for real training")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-distilbert", action="store_true")
    args = parser.parse_args()

    from src.dataset import prepare_data

    print("\n[1/5] Preparing dataset...")
    prepare_data(source=args.source, sample_size=args.sample_size)

    if not args.skip_baseline:
        print("\n[2/5] Training baseline (TF-IDF + Logistic Regression)...")
        from src.train_baseline import train_baseline
        train_baseline(prepare=False)
    else:
        print("\n[2/5] Skipping baseline training.")

    if not args.skip_distilbert:
        print("\n[3/5] Fine-tuning DistilBERT...")
        from src.train import train_distilbert
        train_distilbert(prepare=False, epochs=args.epochs, sample_size=args.sample_size)
    else:
        print("\n[3/5] Skipping DistilBERT training.")

    print("\n[4/5] Exporting to ONNX...")
    try:
        from src.onnx_export import export_to_onnx
        export_to_onnx()
    except Exception as exc:
        print(f"ONNX export failed: {exc}")

    print("\n[5/5] Running latency benchmark...")
    try:
        from src.benchmark import run_benchmark
        run_benchmark(warmup=5, runs=30)
    except Exception as exc:
        print(f"Benchmark failed: {exc}")

    print("\nPipeline complete! Start API: python -m api.app")


if __name__ == "__main__":
    main()
