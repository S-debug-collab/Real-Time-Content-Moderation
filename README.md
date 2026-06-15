# 🚀 Real-Time Content Moderation System

A production-style NLP system that detects toxic, hate speech, and offensive content in real time using DistilBERT + ONNX Runtime, optimized for low-latency inference (<100ms).

---

## 📌 What This Project Does

- Classifies text into:
  - Neutral
  - Toxic
  - Hate Speech
  - Offensive

- Real-time moderation API for user-generated content
- Fast CPU inference using ONNX optimization
- End-to-end ML pipeline (baseline → transformer → deployment)

---

## 🧠 Tech Stack

- PyTorch
- HuggingFace Transformers (DistilBERT)
- ONNX Runtime
- FastAPI
- Scikit-learn

---

## ⚙️ Architecture

User Text → FastAPI → Preprocessing → ONNX Model → Prediction → Response

---

## 🚀 How to Run

pip install -r requirements.txt

python -m src.dataset --source demo --sample-size 2000

python scripts/run_pipeline.py --source demo --epochs 2

python -m api.app

Open:
http://localhost:8000

---

## 📡 API Usage

### Predict Single Text

POST /predict

Input:
{
  "text": "you are an idiot"
}

Output:
{
  "label": "toxic",
  "confidence": 0.87,
  "latency_ms": 42
}

---

## 📊 Results

- Accuracy: ~0.88
- Macro F1: ~0.85
- Inference Latency: ~45ms (ONNX Runtime)

---

## 🧠 Key Features

- DistilBERT fine-tuned classifier
- ONNX Runtime optimized inference
- FastAPI backend for real-time predictions
- Batch + single prediction support
- Runtime threshold tuning

---

## 📁 Project Structure

src/
api/
frontend/
models/
scripts/
tests/
data/
notebooks/

---

## 🔥 Why ONNX?

- Faster inference (~2x speedup vs PyTorch)
- Optimized CPU execution
- Production-ready deployment format
- Cross-platform compatibility

---

## 📌 Resume Highlights

- Built real-time content moderation system using DistilBERT + ONNX
- Achieved <100ms inference latency in production-style API
- Designed scalable FastAPI inference service
- Implemented full ML pipeline: data → training → optimization → deployment

---

## 📜 License

MIT
