import json

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    ARTIFACTS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.preprocessing import clean_text


def load_training_data() -> pd.DataFrame:
    """
    Load prepared dataset if it exists.
    Otherwise load raw dataset and apply preprocessing.
    """
    if PROCESSED_DATA_PATH.exists():
        df = pd.read_csv(PROCESSED_DATA_PATH)
        return df[["text", "sentiment"]].dropna()

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found. Expected file: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH, sep="\t")
    df = df[["review", "sentiment"]].copy()
    df = df.rename(columns={"review": "text"})
    df["sentiment"] = df["sentiment"].replace({"neautral": "neutral"})
    df["text"] = df["text"].apply(clean_text)
    df = df.dropna(subset=["text", "sentiment"])
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text"])
    df = df.reset_index(drop=True)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    return df


def build_model() -> Pipeline:
    """
    Final sklearn Pipeline: TF-IDF + Logistic Regression.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])


def train() -> dict:
    df = load_training_data()

    X = df["text"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "model": "TfidfVectorizer + LogisticRegression",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    metadata_path = ARTIFACTS_DIR / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=4)

    return metrics


if __name__ == "__main__":
    result = train()

    summary = {
        "model": result["model"],
        "accuracy": round(result["accuracy"], 4),
        "precision_macro": round(result["precision_macro"], 4),
        "recall_macro": round(result["recall_macro"], 4),
        "f1_macro": round(result["f1_macro"], 4),
        "train_size": result["train_size"],
        "test_size": result["test_size"],
        "model_path": str(MODEL_PATH),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=4))