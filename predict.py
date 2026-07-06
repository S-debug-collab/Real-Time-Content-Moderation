import os
import joblib

from preprocessing import clean_text
from labels import LABEL_COLUMNS

MODEL_DIR = "model"
THRESHOLD = 0.5


class ModerationPredictor:

    def __init__(self, model_dir=MODEL_DIR):

        model_path = os.path.join(model_dir, "model.pkl")
        vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError("model.pkl not found.")

        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError("vectorizer.pkl not found.")

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, text: str) -> dict:

        cleaned = clean_text(text)

        features = self.vectorizer.transform([cleaned])

        probabilities = self.model.predict_proba(features)[0]

        probs = {
            label: float(prob)
            for label, prob in zip(LABEL_COLUMNS, probabilities)
        }

        predicted_labels = [
            label
            for label, prob in probs.items()
            if prob >= THRESHOLD
        ]

        return {
            "neutral": len(predicted_labels) == 0,
            "predicted_labels": predicted_labels,
            "probabilities": probs,
        }
