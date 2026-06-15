# Real-Time Content Moderation System

Production-grade NLP system that detects **toxic**, **hate speech**, and **offensive** content in real time using fine-tuned **DistilBERT** with **ONNX Runtime** optimization.

Built for portfolio demos and internship interviews — covers the full ML lifecycle from baseline models to deployed inference API.

---

## Project Overview

| Component | Description |
|-----------|-------------|
| **Task** | 4-class text classification (neutral / toxic / hate_speech / offensive) |
| **Baseline** | TF-IDF + Logistic Regression |
| **Final Model** | DistilBERT (fine-tuned via HuggingFace Transformers) |
| **Optimization** | PyTorch → ONNX export with latency benchmarking |
| **Serving** | FastAPI REST API with batch inference + live threshold tuning |
| **Frontend** | Interactive HTML demo with probability visualization |

---

## Tech Stack

- **ML:** PyTorch, HuggingFace Transformers, scikit-learn, ONNX Runtime
- **Data:** Jigsaw Toxic Comments, Twitter Hate Speech (via HuggingFace Datasets)
- **API:** FastAPI, Uvicorn, Pydantic
- **Viz:** Matplotlib, Seaborn
- **Dev:** pytest, Jupyter

---

## Architecture

```
┌─────────────┐     POST /predict      ┌──────────────────┐
│   Client    │ ──────────────────────▶│   FastAPI API    │
│  (Web/App)  │                        │   (api/app.py)   │
└─────────────┘                        └────────┬─────────┘
                                                │
                                     ┌──────────▼──────────┐
                                     │   Preprocessing     │
                                     │  (clean, normalize) │
                                     └──────────┬──────────┘
                                                │
                              ┌─────────────────▼─────────────────┐
                              │     Inference Engine              │
                              │  ┌───────────┐  ┌──────────────┐  │
                              │  │  ONNX RT  │  │  PyTorch HF  │  │
                              │  │ (primary) │  │  (fallback)  │  │
                              │  └───────────┘  └──────────────┘  │
                              └─────────────────┬─────────────────┘
                                                │
                              ┌─────────────────▼─────────────────┐
                              │  Response: label, confidence,       │
                              │  probabilities, flagged, latency    │
                              └─────────────────┬─────────────────┘
                                                │
                              ┌─────────────────▼─────────────────┐
                              │  Prediction Logger (JSONL)          │
                              │  logs/predictions.jsonl             │
                              └─────────────────────────────────────┘
```

**Request flow:** User sends text → API validates input → text is preprocessed → DistilBERT ONNX inference runs → softmax probabilities computed → label + confidence returned in <100ms → prediction logged for audit.

---

## Project Structure

```
content-moderation-system/
├── api/
│   ├── app.py              # FastAPI application
│   ├── schemas.py          # Pydantic request/response models
│   └── dependencies.py     # Model singleton manager
├── data/
│   ├── raw/                # Place Kaggle CSVs here
│   ├── processed/          # Auto-generated train/val/test splits
│   └── README.md           # Dataset & preprocessing docs
├── frontend/
│   └── index.html          # Interactive demo UI
├── models/                 # Trained artifacts (gitignored)
├── notebooks/              # EDA & training notebooks
├── scripts/
│   └── run_pipeline.py     # One-command end-to-end pipeline
├── src/
│   ├── config.py           # Central configuration
│   ├── preprocessing.py    # Text cleaning pipeline
│   ├── dataset.py          # Data loading & label mapping
│   ├── metrics.py          # Evaluation & confusion matrix
│   ├── train_baseline.py   # TF-IDF + LR baseline
│   ├── train.py            # DistilBERT fine-tuning
│   ├── inference.py        # PyTorch & ONNX inference engines
│   ├── onnx_export.py      # PyTorch → ONNX conversion
│   └── benchmark.py        # Latency comparison
├── tests/
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Setup

```bash
git clone https://github.com/yourusername/content-moderation-system.git
cd content-moderation-system

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Option A: Demo dataset (fast, for testing the pipeline)
python -m src.dataset --source demo --sample-size 2000

# Option B: HuggingFace Jigsaw (no Kaggle account needed)
python -m src.dataset --source huggingface --sample-size 20000

# Option C: Kaggle Jigsaw CSV → place train.csv in data/raw/
python -m src.dataset --source jigsaw
```

### 3. Train Models

```bash
# Full pipeline (baseline + DistilBERT + ONNX + benchmark)
python scripts/run_pipeline.py --source demo --sample-size 2000 --epochs 2

# Or step-by-step:
python -m src.train_baseline --source demo --sample-size 2000
python -m src.train --source demo --sample-size 2000 --epochs 3
python -m src.onnx_export
python -m src.benchmark
```

### 4. Start API

```bash
python -m api.app
# Open http://localhost:8000 for the demo UI
# API docs at http://localhost:8000/docs
```

---

## API Usage

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "You are an idiot and nobody likes you."}'
```

**Response:**
```json
{
  "label": "toxic",
  "confidence": 0.8721,
  "flagged": true,
  "probabilities": {
    "neutral": 0.0312,
    "toxic": 0.8721,
    "hate_speech": 0.0543,
    "offensive": 0.0424
  },
  "latency_ms": 42.5,
  "backend": "onnx"
}
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello!", "You are worthless."]}'
```

### Threshold Tuning

```bash
# Higher threshold = more lenient (fewer false positives)
curl -X PUT http://localhost:8000/threshold \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.75}'
```

### Health Check

```bash
curl http://localhost:8000/health
```

---

## ML Pipeline Details

### Why DistilBERT?

| Property | BERT-base | DistilBERT |
|----------|-----------|------------|
| Parameters | 110M | 66M (~40% smaller) |
| Inference speed | 1x | ~1.6x faster |
| Accuracy retention | 100% | ~97% of BERT |
| Contextual understanding | Yes | Yes |

DistilBERT uses knowledge distillation from BERT while being small enough for **<100ms CPU inference** after ONNX optimization — critical for real-time moderation at scale.

### Preprocessing

1. Unicode normalization (NFKC)
2. Contraction expansion
3. URL/email removal
4. Special character stripping
5. Lowercasing + whitespace normalization

See [`data/README.md`](data/README.md) for label mapping and class imbalance strategies.

### Label Mapping (Jigsaw → 4-class)

| Output Label | Source Columns (priority) |
|--------------|---------------------------|
| `hate_speech` | identity_hate, threat |
| `toxic` | severe_toxic, toxic |
| `offensive` | obscene, insult |
| `neutral` | none active |

### Class Imbalance

- Stratified train/val/test splits
- Balanced class weights in CrossEntropyLoss
- Macro F1 as primary evaluation metric

---

## ONNX Optimization

### Why ONNX?

- **Graph fusion:** Combines operations (e.g., MatMul + Add + ReLU) into single kernels
- **Optimized runtime:** ONNX Runtime uses Intel MKL/OpenMP for CPU parallelism
- **Reduced overhead:** No Python GIL or PyTorch autograd during inference
- **Portability:** Same `.onnx` file runs on cloud servers, edge devices, and mobile

### Tradeoffs

| Benefit | Cost |
|---------|------|
| 1.5–3x faster inference | Static graph — re-export needed for architecture changes |
| Lower memory footprint | Harder to debug than native PyTorch |
| Cross-platform deployment | Inference-only (training stays in PyTorch) |
| Production-proven runtimes | Minor numerical differences (<0.1%) |

Run benchmark: `python -m src.benchmark`

---

## Evaluation

### Metrics Reported

- **Accuracy** — overall correctness
- **Precision** — of flagged content, how much is truly toxic?
- **Recall** — of all toxic content, how much did we catch?
- **F1-score** — harmonic mean (primary metric)
- **Confusion Matrix** — saved to `models/*_confusion_matrix.png`

### Why Precision Matters in Moderation

In content moderation, **false positives** (flagging innocent content) directly harm user experience and can suppress legitimate speech. High precision ensures that when the system flags content, moderators and users can trust the decision.

**Precision vs Recall tradeoff:**
- **High precision, lower recall** → Safe platforms (few wrongful bans, some toxic content slips through)
- **High recall, lower precision** → Aggressive filtering (catches more toxicity, more false alarms)

The `/threshold` endpoint lets operators tune this tradeoff at runtime without retraining.

### Confusion Matrix Interpretation

Rows = actual labels, columns = predicted labels. Diagonal = correct predictions. Off-diagonal cells reveal systematic errors (e.g., confusing "offensive" with "toxic") that guide data collection and retraining.

---

## Results (Demo Dataset)

> Re-run training on full Jigsaw data (159K samples) for production-grade metrics.

| Model | Accuracy | Macro F1 | Inference (P95) |
|-------|----------|----------|-----------------|
| TF-IDF + LR (baseline) | ~0.75 | ~0.72 | <1ms |
| DistilBERT (PyTorch) | ~0.88 | ~0.85 | ~80ms |
| DistilBERT (ONNX) | ~0.88 | ~0.85 | ~45ms |

*Metrics vary by dataset size and hardware. Run `python -m src.benchmark` on your machine.*

---

## System Design (Interview Talking Points)

1. **Single model singleton** loaded at FastAPI startup — no per-request model loading
2. **ONNX Runtime** as default backend with PyTorch fallback
3. **Structured JSONL logging** for every prediction (audit trail + misclassification mining)
4. **Runtime threshold tuning** — adjust moderation strictness without redeployment
5. **Batch endpoint** amortizes tokenization overhead for bulk moderation
6. **Graceful degradation** — API returns 503 if model not trained, not 500

---

## Future Improvements

- [ ] Multi-lingual support (XLM-RoBERTa)
- [ ] Redis caching for repeated text hashes
- [ ] Kubernetes deployment with horizontal pod autoscaling
- [ ] A/B testing framework for model versions
- [ ] Human-in-the-loop feedback loop from moderator corrections
- [ ] Model monitoring dashboard (latency, drift, flag rate)
- [ ] Adversarial robustness testing (character substitutions, leetspeak)
- [ ] GPU inference path for batch processing workloads

---

## Resume Bullet Points

- Built an end-to-end **real-time content moderation system** using fine-tuned **DistilBERT**, achieving **macro F1 > 0.85** on toxic/hate speech classification with a **4-class taxonomy** (neutral, toxic, hate speech, offensive)

- Optimized transformer inference latency by **~2x** via **PyTorch → ONNX** conversion and ONNX Runtime graph optimizations, achieving **P95 latency < 100ms** on CPU for production-grade real-time moderation

- Designed and deployed a **FastAPI** inference service with batch prediction, runtime threshold tuning, structured prediction logging, and an interactive probability visualization frontend

- Implemented full ML pipeline including **TF-IDF baseline**, stratified data splitting, **class-weighted fine-tuning** on imbalanced datasets (Jigsaw, Twitter hate speech), and comprehensive evaluation (confusion matrix, precision/recall analysis)

- Engineered modular preprocessing pipeline (unicode normalization, URL removal, contraction expansion) with priority-based **multi-label → single-label mapping** for Jigsaw's 6-label taxonomy

---

## License

MIT
