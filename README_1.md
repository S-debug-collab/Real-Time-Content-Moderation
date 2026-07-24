# Real-Time Content Moderation — DistilBERT + ONNX + FastAPI

A binary toxicity classifier for social media posts and comments, fine-tuned from DistilBERT,
optimized with ONNX Runtime for low-latency inference, and served through a FastAPI endpoint.

## Overview

Given a piece of text, the model outputs a toxicity probability and a flag/don't-flag decision
based on a tuned confidence threshold. Built for the real-time moderation use case: scanning
incoming posts/comments and surfacing likely-toxic content for review, rather than requiring
every post to be read manually.

## Dataset

Two sources, combined and balanced to roughly equal size:

| Source | Rows | Description |
|---|---|---|
| Jigsaw Toxic Comment Classification | 31,959 (downsampled from 159,571) | Wikipedia talk-page comments, 6 toxicity categories collapsed into a single binary label |
| Twitter Hate Speech Detection | 29,527 | Real tweets, natively binary-labeled |
| **Combined total** | **61,486** | ~91.5% clean / 8.5% toxic |

Jigsaw is stratified-downsampled to match Twitter's size (preserving its own label ratio)
before merging, so the model isn't dominated by one platform's writing style.

## Model & Training

- Base model: `distilbert-base-uncased`
- Task: binary sequence classification (toxic / not toxic)
- Loss: class-weighted cross-entropy (compensates for the ~9:1 clean/toxic imbalance)
- Split: stratified 80/10/10 train/val/test
- Tokenization: max length 128, truncated/padded
- 3 epochs, learning rate 2e-5, batch size 32 (train) / 64 (eval)

## Evaluation

No fixed precision/recall target is assumed. The notebook computes the full precision-recall
curve on the held-out test set and reports the trade-off across multiple thresholds:

| Threshold | Precision | Recall |
|---|---|---|
| 0.301 | 71.1% | 80.0% |
| 0.502 | 74.5% | 78.1% |
| 0.710 | 77.2% | 77.4% |
| 0.801 | 78.3% | 76.0% |
| 0.900 | 80.5% | 74.5% |
| 0.951 | 83.2% | 72.4% |

**Best-F1 balance point (used as the default operating threshold):** 0.734 → 77.8% precision,
77.4% recall, 77.6% F1.

Pick a different row from the table above if your use case needs a different precision/recall
trade-off (e.g., higher threshold for higher precision at the cost of recall).

## Inference Optimization

The trained PyTorch model is exported to ONNX and quantized to INT8 for faster CPU inference:

| Runtime | Mean | p50 | p95 |
|---|---|---|---|
| PyTorch (CPU, eager) | 154.93ms | 131.41ms | 290.83ms |
| ONNX Runtime (fp32) | 131.63ms | 118.89ms | 182.35ms |
| ONNX Runtime (INT8, quantized) | 84.15ms | 82.77ms | 92.94ms |

**1.8x speedup** (INT8 ONNX vs. PyTorch eager, mean latency). All benchmarks run on CPU — the
realistic serving target for this model, since GPU inference on single short sequences is
dominated by kernel-launch overhead rather than compute.

## Serving

A FastAPI application wraps the quantized ONNX model:

- `POST /predict` — body `{"text": "..."}` → returns toxicity probability and flag decision
- `GET /health` — health check
- CORS enabled for browser-based clients

Example response:
```json
{
  "text": "You are a worthless idiot and I hope something bad happens to you",
  "toxic_probability": 0.9991,
  "flagged": true,
  "threshold_used": 0.734
}
```

## Requirements

```
transformers==4.46.3
accelerate==1.1.1
datasets==3.1.0
evaluate==0.4.3
scikit-learn
onnx==1.17.0
onnxruntime==1.19.2
onnxscript
fastapi
uvicorn
pyngrok        # only needed for the optional public-demo cell
```

Note: `torchvision` must be uninstalled before installing the above — a preinstalled Colab
version conflicts with the `datasets` library's import checks. `optimum` is intentionally not
used — the ONNX export and quantization steps use `torch.onnx.export` and
`onnxruntime.quantization` directly, since `optimum` introduced unrelated dependency conflicts
with `diffusers`/`transformers` version resolution during development.

## How to Run

1. Open `toxicity_moderation_pipeline_v2.ipynb` in Google Colab (Runtime → Change runtime
   type → T4 GPU).
2. Run Section 1 (environment setup) — includes two required session restarts, noted inline.
3. Run Sections 2-7 in order — upload `train.csv` (Jigsaw dataset) when prompted; the Twitter
   dataset downloads automatically. Training takes roughly 8-10 minutes on a T4.
4. Run Sections 8-13 — evaluation, ONNX export, quantization, latency benchmarking, and a
   live prediction test.
5. Run Section 14 for the FastAPI server:
   - 14 + 14a — defines and tests the API in-process (`TestClient`), no external server needed
   - 14b + 14c (optional) — exposes a public URL via `ngrok` and a minimal HTML demo page;
     requires a free ngrok account and auth token

## Repository Contents

- `toxicity_moderation_pipeline_v2.ipynb` — full pipeline: data loading, training, evaluation,
  ONNX export, quantization, benchmarking, and FastAPI serving
- `README.md` — this file
