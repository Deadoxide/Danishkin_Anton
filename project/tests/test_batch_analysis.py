import pandas as pd
import pytest

from src import batch_analysis


def test_analyze_texts_handles_empty_text_without_prediction(monkeypatch):
    calls = []

    def fake_predict_sentiment(text: str) -> dict:
        calls.append(text)
        return {"sentiment": "positive", "confidence": 0.9}

    monkeypatch.setattr(batch_analysis, "predict_sentiment", fake_predict_sentiment)

    result = batch_analysis.analyze_texts(["good review", "   ", None])

    assert calls == ["good review"]
    assert result == [
        {
            "text": "good review",
            "sentiment": "positive",
            "confidence": 0.9,
            "error": None,
        },
        {
            "text": "   ",
            "sentiment": None,
            "confidence": None,
            "error": "empty_text",
        },
        {
            "text": None,
            "sentiment": None,
            "confidence": None,
            "error": "empty_text",
        },
    ]


def test_analyze_dataframe_returns_new_result_dataframe(monkeypatch):
    def fake_predict_sentiment(text: str) -> dict:
        return {"sentiment": "neutral", "confidence": 0.6}

    monkeypatch.setattr(batch_analysis, "predict_sentiment", fake_predict_sentiment)

    source_df = pd.DataFrame({"review": ["ok"], "other": [1]})
    result = batch_analysis.analyze_dataframe(source_df, text_column="review")

    assert list(result.columns) == ["text", "sentiment", "confidence", "error"]
    assert result.iloc[0].to_dict() == {
        "text": "ok",
        "sentiment": "neutral",
        "confidence": 0.6,
        "error": None,
    }
    assert list(source_df.columns) == ["review", "other"]


def test_analyze_dataframe_raises_for_missing_text_column():
    df = pd.DataFrame({"review": ["ok"]})

    with pytest.raises(ValueError, match="Column 'text' was not found"):
        batch_analysis.analyze_dataframe(df)


def test_get_sentiment_distribution_ignores_error_rows():
    df = pd.DataFrame(
        [
            {"sentiment": "positive", "error": None},
            {"sentiment": "positive", "error": None},
            {"sentiment": "neutral", "error": None},
            {"sentiment": None, "error": "empty_text"},
        ]
    )

    result = batch_analysis.get_sentiment_distribution(df)

    assert result.to_dict("records") == [
        {"sentiment": "positive", "count": 2},
        {"sentiment": "neutral", "count": 1},
        {"sentiment": "negative", "count": 0},
    ]
