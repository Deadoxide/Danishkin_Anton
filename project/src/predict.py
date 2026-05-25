from functools import lru_cache

import joblib

from src.config import MODEL_PATH
from src.preprocessing import clean_text


@lru_cache(maxsize=1)
def load_model():
    """
    Load trained sklearn Pipeline from artifacts.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Run `python -m src.train` first."
        )

    return joblib.load(MODEL_PATH)


def predict_sentiment(text: str) -> dict:
    """
    Predict sentiment and confidence for a single review text.
    """
    cleaned_text = clean_text(text)

    if not cleaned_text:
        raise ValueError("Text is empty after preprocessing.")

    model = load_model()

    sentiment = model.predict([cleaned_text])[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([cleaned_text])[0]
        confidence = float(probabilities.max())

    return {
        "sentiment": sentiment,
        "confidence": confidence,
    }