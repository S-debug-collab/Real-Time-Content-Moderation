# Real-Time Content Moderation System

A classic-NLP (no deep learning) pipeline that classifies user comments into
**Neutral**, **Toxic**, **Offensive**, or **Hate Speech**, served through a
FastAPI backend with a small web frontend. Built as an interview-ready,
medium-difficulty ML project using Pandas, NumPy, NLTK, and scikit-learn.

---

## Project Structure

```
content-moderation/
├── dataset/                # Raw data (Jigsaw CSV or bundled synthetic sample)
├── model/                  # Saved model.pkl, vectorizer.pkl, label_encoder.pkl, confusion_matrix.png
├── api/                    # Reserved for splitting routes out of app.py as the API grows
├── frontend/               # index.html, style.css, script.js (calls POST /predict)
├── preprocessing.py        # Text cleaning, tokenization, stopwords, stemming, lemmatization
├── feature_extraction.py   # Bag-of-Words vs TF-IDF, side-by-side comparison
├── train.py                # Full training pipeline (load -> clean -> vectorize -> train -> evaluate -> save)
├── predict.py               # Loads saved artifacts, runs inference
├── app.py                  # FastAPI app: POST /predict, GET /health, serves the frontend
├── requirements.txt
└── README.md
```

## Tech Stack

Python · Pandas · NumPy · NLTK · scikit-learn · FastAPI

No PyTorch, TensorFlow, Hugging Face, BERT, or LLMs — this project is
intentionally built on classic NLP + linear ML to demonstrate the
fundamentals clearly.

---

## Dataset

This project targets the **Jigsaw Toxic Comment Classification Challenge**
dataset (Kaggle). Download `train.csv` from Kaggle and place it at:

```
dataset/train.csv
```

Kaggle link: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

**If you don't have Kaggle access**, `train.py` automatically falls back to
generating a small synthetic sample dataset (`dataset/sample_data.csv`) with
the *same schema*, so the entire pipeline (cleaning, feature extraction,
training, evaluation, API) is runnable out of the box. Swap in the real
Kaggle CSV for meaningful accuracy numbers — the synthetic sample exists
purely so the project runs end-to-end without external downloads.

### From 6 Jigsaw tags to 4 moderation classes

Jigsaw's raw labels are six independent binary tags per comment (`toxic`,
`severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`), not a single
4-class label. `train.py` derives one class per comment with a priority rule
(most severe tag wins):

| Priority | Condition                                      | Assigned label |
|----------|--------------------------------------------------|----------------|
| 1        | `identity_hate == 1`                              | Hate Speech    |
| 2        | `toxic == 1` or `severe_toxic == 1`               | Toxic          |
| 3        | `obscene == 1`, `insult == 1`, or `threat == 1`   | Offensive      |
| 4        | none of the above                                 | Neutral        |

This is a documented, explainable heuristic — a natural talking point for an
interview ("how did you turn a multi-label dataset into single-label
classes?").

---

## Results (trained on the real Jigsaw dataset, 159,571 comments)

The model in `model/` was trained on the actual Kaggle `train.csv`, not the
synthetic sample. TF-IDF beat Bag-of-Words and was selected automatically:

| Method | Accuracy | Weighted F1 |
|--------|----------|-------------|
| BoW    | 0.930    | 0.932       |
| TF-IDF | **0.941**| **0.941**   |

Final Logistic Regression (TF-IDF) on the held-out test set:

| Class       | Precision | Recall | F1   | Support |
|-------------|-----------|--------|------|---------|
| Neutral     | 0.97      | 0.98   | 0.97 | 28,644  |
| Toxic       | 0.71      | 0.69   | 0.70 | 2,798   |
| Hate Speech | 0.47      | 0.48   | 0.48 | 281     |
| Offensive   | 0.06      | 0.05   | 0.05 | 165     |
| **Overall** | **0.94**  | **0.94** | **0.94** | 31,888 |

*(Overall accuracy: 0.941.)*

### A real bug we hit and fixed: `class_weight="balanced"` on tiny classes

An earlier version of this project used plain `class_weight="balanced"`,
which weights each class inversely to its frequency
(`n_samples / (n_classes * n_class)`). For the "Offensive" class — only 828
of 159,571 comments — that works out to roughly **48x** the weight of a
"Neutral" example.

In practice this blew up: the comment **"Loved the content"** was
classified as **Offensive with 88% confidence**. Digging into the model's
coefficients showed why — the word "loved" had a coefficient of **+4.68**
for the Offensive class, far higher than for any other class. Checking the
training data explained it: of the 131 training comments containing
"loved," exactly **one** happened to be labeled Offensive (coincidentally,
for unrelated reasons). With 48x weighting on a class that small, that
single coincidental example was enough to convince a linear model that
"loved" was strong evidence of an offensive comment.

**Fix:** `feature_extraction.py` now computes a *square-root-dampened*
version of balanced class weights
(`compute_dampened_class_weights()`) instead of using `"balanced"`
directly. This still up-weights rare classes so the model doesn't just
ignore them, but far less aggressively (≈7x instead of ≈48x for
Offensive), which stops single coincidental training examples from
dominating the model. We also added `min_df=3` to both vectorizers to drop
ultra-rare tokens that are more likely to produce this kind of spurious
correlation in the first place.

The fix improved every headline number — overall accuracy went from 0.878
to 0.941, and weighted F1 from 0.901 to 0.941 — while also fixing the
"Loved the content" bug and similar false positives on clearly benign text.

**Why "Offensive" still performs poorly (F1 0.05) even after the fix:**

The label-derivation priority rule (see above) assigns "Offensive" only
when `obscene`/`insult`/`threat` fires **and** `toxic`/`severe_toxic` does
*not*. In the real data, obscene/insulting comments are overwhelmingly
*also* tagged `toxic`, so they get claimed by the "Toxic" class first. That
leaves "Offensive" as a tiny (828 comments, 0.5% of the dataset) residual
category that's inherently hard to learn — no amount of reweighting fixes a
class that's both rare AND poorly separated from Toxic in the feature
space. This is a genuine, common real-world issue: **deriving single-label
classes from overlapping multi-label tags is lossy**, and the priority
order you choose directly determines which class "absorbs" the ambiguous
cases. Good follow-up answers in an interview:
- Reorder priority (e.g. let "Offensive" claim obscene/insult even when
  toxic also fires) and see if that better matches the intended business
  definition of each class.
- Treat this as a multi-label problem instead (predict each of the 6 tags
  independently) rather than forcing one label per comment.
- Collect more labeled "Offensive-but-not-Toxic" examples, or merge
  Offensive and Toxic into one class if the distinction isn't reliably
  recoverable from the available tags.

This combination — catching a real overfitting bug via coefficient
inspection, fixing it with a principled reweighting scheme, and then
correctly diagnosing which remaining weakness is a *data/label design*
issue rather than a model issue — is a much stronger interview story than
a project that only reports a single clean accuracy number.

---

## Setup

```bash
cd content-moderation
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

NLTK resources (stopwords, punkt tokenizer, wordnet) are downloaded
automatically the first time `preprocessing.py` runs.

> **Note:** the shipped `model/` artifacts were trained in a sandboxed
> environment without internet access, so NLTK's real stopword list and
> WordNetLemmatizer could not be downloaded — an offline approximation was
> used instead (sklearn's built-in English stopword list + a simple
> suffix-based lemmatizer fallback). The results above are still genuine,
> reproducible numbers from the real 159,571-row Jigsaw dataset, but
> re-running `python train.py` in a normal internet-connected environment
> (with real NLTK resources) may shift metrics slightly and is recommended
> before treating this as a final benchmark.

## Train the model

```bash
python train.py
```

This runs the full pipeline end to end:

1. **Load dataset** — real Jigsaw CSV if present, else synthetic sample
2. **Data exploration** — shape, missing values, class distribution
3. **Handle missing values** — drop empty/NaN comments
4. **Clean text** — lowercase, strip URLs, strip HTML tags, strip
   punctuation, strip numbers
5. **Tokenize → remove stopwords → lemmatize** (see `preprocessing.py`)
6. **Derive 4-class labels** from the 6 Jigsaw tag columns
7. **Compare Bag-of-Words vs TF-IDF** — trains a quick Logistic Regression
   with each and reports accuracy/F1 so the better method is chosen
   automatically
8. **Train final Logistic Regression** (`class_weight="balanced"` to handle
   class imbalance, which is severe in the real Jigsaw data — most comments
   are Neutral)
9. **Evaluate** — accuracy, precision, recall, F1 (weighted), full
   classification report, and a confusion matrix heatmap saved to
   `model/confusion_matrix.png`
10. **Persist artifacts** — `model.pkl`, `vectorizer.pkl`,
    `label_encoder.pkl`, `feature_method.pkl` saved to `model/`

## Run the API

```bash
uvicorn app:app --reload
```

Then open **http://127.0.0.1:8000/** for the web UI, or call the API
directly:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "You are stupid"}'
```

```json
{
  "prediction": "Offensive",
  "confidence": 0.91
}
```

Interactive API docs (Swagger UI) are auto-generated by FastAPI at
`http://127.0.0.1:8000/docs`.

---

## Why these design choices? (interview talking points)

- **Bag-of-Words vs TF-IDF, compared not assumed.** Rather than picking
  TF-IDF by default, `feature_extraction.py` trains a quick Logistic
  Regression with both on an identical split and compares weighted F1,
  so the choice is data-driven and easy to defend in an interview.
- **Lemmatization over stemming for the final model.** Both are
  implemented (`preprocessing.py`), but the production pipeline uses
  lemmatization since it produces real dictionary words, which keeps
  TF-IDF/BoW vocabularies more interpretable. Stemming is kept available
  for side-by-side comparison.
- **`class_weight="balanced"`.** Toxic comment datasets are heavily
  skewed toward "Neutral" comments; balancing class weights prevents the
  model from just predicting the majority class.
- **Confidence = max class probability.** Logistic Regression's
  `predict_proba` gives a probability per class; we return the top class's
  probability as the "confidence" score, which is intuitive for an API
  consumer and cheap to compute (no extra calibration step needed for a
  linear model with well-separated classes).
- **Model loaded once at API startup**, not per request — important for a
  system billed as "real-time."
- **Why not deep learning?** Logistic Regression on TF-IDF/BoW features
  is fast to train, easy to explain end-to-end (a strength in interviews),
  runs comfortably on a laptop CPU, and is a strong, standard baseline for
  text classification before reaching for anything heavier.

## Possible extensions

- Swap Logistic Regression for `LinearSVC` or `MultinomialNB` and compare.
- Add n-gram features (bigrams) to `feature_extraction.py`.
- Add a `/feedback` endpoint to collect corrected labels for retraining.
- Add rate limiting / auth to the FastAPI service for production use.
