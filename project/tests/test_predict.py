import pytest

from src.config import LABELS, MODEL_PATH
from src.predict import predict_sentiment


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Model artifact is missing. Run `python -m src.train` first.",
)
def test_predict_sentiment_returns_valid_response():
    result = predict_sentiment("Товар хороший, качество понравилось")

    assert "sentiment" in result
    assert "confidence" in result
    assert result["sentiment"] in LABELS
    assert result["confidence"] is None or 0 <= result["confidence"] <= 1