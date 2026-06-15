# Dataset Instructions

## Supported Datasets

### 1. Jigsaw Toxic Comment Classification (Recommended)
- **Source:** [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
- **Place file at:** `data/raw/train.csv`
- **Load via:** `python -m src.dataset --source jigsaw`

**Original labels (multi-label):** toxic, severe_toxic, obscene, threat, insult, identity_hate

**Our mapping (priority order):**
| Our Label     | Jigsaw Columns              |
|---------------|-----------------------------|
| hate_speech   | identity_hate, threat       |
| toxic         | severe_toxic, toxic         |
| offensive     | obscene, insult             |
| neutral       | none of the above           |

### 2. HuggingFace (No Kaggle account needed)
```bash
python -m src.dataset --source huggingface --sample-size 20000
```

### 3. Twitter Hate Speech
- Place CSV at `data/raw/twitter_hate.csv` with columns: `text`/`tweet`, `label`/`class`
- Load via: `python -m src.dataset --source twitter`

### 4. Demo (Development only)
```bash
python -m src.dataset --source demo --sample-size 2000
```

## Class Imbalance Handling

Real-world moderation datasets are heavily skewed toward neutral content (~90%+).

Our strategies:
1. **Stratified splits** — preserve label ratios in train/val/test
2. **Class weights** — `compute_class_weight('balanced')` in DistilBERT loss
3. **Balanced sampling** — optional per-class subsampling during data prep
4. **Macro F1** — primary metric (treats all classes equally)

## Preprocessing Pipeline

1. Unicode normalization (NFKC)
2. Contraction expansion (`won't` → `will not`)
3. URL/email removal
4. Special character stripping (preserve apostrophes)
5. Lowercasing + whitespace normalization
6. Empty text filtering

Processed splits are saved to `data/processed/`.
