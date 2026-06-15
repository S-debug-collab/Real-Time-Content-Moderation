"""Central configuration for the content moderation system."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Label schema (4-class moderation taxonomy)
LABEL_NAMES = ["neutral", "toxic", "hate_speech", "offensive"]
LABEL2ID = {name: idx for idx, name in enumerate(LABEL_NAMES)}
ID2LABEL = {idx: name for name, idx in LABEL2ID.items()}

# Jigsaw multi-label → single-label mapping rules (priority order)
JIGSAW_PRIORITY = [
    ("hate_speech", ["identity_hate", "threat"]),
    ("toxic", ["severe_toxic", "toxic"]),
    ("offensive", ["obscene", "insult"]),
]

# Model defaults
BASELINE_VECTORIZER = "tfidf"
BASELINE_CLASSIFIER = "logistic_regression"
DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
MAX_SEQ_LENGTH = 128
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_EPOCHS = 3

# Inference
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
ONNX_MODEL_FILENAME = "distilbert_moderation.onnx"
PYTORCH_MODEL_DIR = MODELS_DIR / "distilbert_finetuned"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_tfidf_lr.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# API
API_HOST = "0.0.0.0"
API_PORT = 8000
USE_ONNX_BY_DEFAULT = True

# Benchmark
BENCHMARK_WARMUP_RUNS = 10
BENCHMARK_TIMED_RUNS = 100
