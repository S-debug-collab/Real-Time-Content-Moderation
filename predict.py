"""
predict.py
----------
Loads the trained model + vectorizer + label encoder from model/ and
exposes a single `predict_comment()` function used by both the FastAPI
backend (app.py) and any ad-hoc CLI testing.
"""

import os

import joblib

from preprocessing import clean_text

MODEL_DIR = "model"


class ModerationPredictor:
    """Wraps the trained artifacts so they're loaded ONCE (at API startup)
    instead of on every request, which matters for a "real-time" service.
    """

    def __init__(self, model_dir=MODEL_DIR):
        model_path = os.path.join(model_dir, "model.pkl")
        vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
        encoder_path = os.path.join(model_dir, "label_encoder.pkl")

        for path in (model_path, vectorizer_path, encoder_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing artifact: {path}. Run `python train.py` first "
                    "to train and save the model."
                )

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.label_encoder = joblib.load(encoder_path)

    def predict(self, text: str) -> dict:
        """Cleans the input text, vectorizes it, and returns the predicted
        class label along with the model's confidence (max class
        probability from Logistic Regression's predict_proba).
        """
        cleaned = clean_text(text)
        features = self.vectorizer.transform([cleaned])

        probabilities = self.model.predict_proba(features)[0]
        predicted_idx = probabilities.argmax()
        predicted_label = self.label_encoder.inverse_transform([predicted_idx])[0]
        confidence = float(probabilities[predicted_idx])

        return {
            "prediction": predicted_label,
            "confidence": round(confidence, 4),
        }


if __name__ == "__main__":
    predictor = ModerationPredictor()
    test_comments = [
        "You are stupid",
        "Thanks so much for your help today!",
        "I hate people from that country, they should leave.",
        "Get out of here, you worthless idiot.",
    ]
    for comment in test_comments:
        result = predictor.predict(comment)
        print(f"{comment!r} -> {result}")
