import pandas as pd

from src.predict import predict_sentiment


SENTIMENT_ORDER = ["positive", "neutral", "negative"]


def analyze_texts(texts: list[str]) -> list[dict]:
    """
    Analyze a list of texts and return sentiment results for each text.
    """
    results = []

    for text in texts:
        if pd.isna(text) or not str(text).strip():
            results.append(
                {
                    "text": text,
                    "sentiment": None,
                    "confidence": None,
                    "error": "empty_text",
                }
            )
            continue

        try:
            prediction = predict_sentiment(str(text))
            results.append(
                {
                    "text": text,
                    "sentiment": prediction["sentiment"],
                    "confidence": prediction["confidence"],
                    "error": None,
                }
            )
        except Exception as error:
            results.append(
                {
                    "text": text,
                    "sentiment": None,
                    "confidence": None,
                    "error": str(error),
                }
            )

    return results


def analyze_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
) -> pd.DataFrame:
    """
    Analyze texts from a DataFrame column and return a new DataFrame with results.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' was not found in DataFrame.")

    results = analyze_texts(df[text_column].tolist())

    return pd.DataFrame(
        results,
        columns=["text", "sentiment", "confidence", "error"],
    )


def get_sentiment_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count sentiments for rows without errors.
    """
    valid_rows = df[df["error"].isna()]
    counts = valid_rows["sentiment"].value_counts()

    distribution = pd.DataFrame(
        {
            "sentiment": SENTIMENT_ORDER,
            "count": [int(counts.get(sentiment, 0)) for sentiment in SENTIMENT_ORDER],
        }
    )

    return distribution
