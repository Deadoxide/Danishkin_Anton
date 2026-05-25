import logging
from time import perf_counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import LOG_LEVEL, MAX_TEXT_LENGTH, MODEL_PATH
from src.predict import predict_sentiment, load_model


logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("sentiment-api")

app = FastAPI(
    title="Sentiment Review Service",
    description="ML-service for automatic sentiment classification of user reviews.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="User review text",
        examples=["Доставка задержалась, но товар хороший"],
    )


class PredictResponse(BaseModel):
    sentiment: str
    confidence: float | None


@app.on_event("startup")
def startup_event():
    """
    Load model during application startup.
    """
    load_model()
    logger.info("Model loaded from %s", MODEL_PATH)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.exists(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start_time = perf_counter()

    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=422,
            detail="Text must not be empty.",
        )

    try:
        result = predict_sentiment(text)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.") from error

    elapsed_ms = round((perf_counter() - start_time) * 1000, 2)

    logger.info(
        "prediction completed | text_length=%s | sentiment=%s | confidence=%s | elapsed_ms=%s",
        len(text),
        result["sentiment"],
        result["confidence"],
        elapsed_ms,
    )

    return result