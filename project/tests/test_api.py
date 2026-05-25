import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.config import LABELS, MODEL_PATH


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Model artifact is missing. Run `python -m src.train` first.",
)
def test_health_endpoint():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Model artifact is missing. Run `python -m src.train` first.",
)
def test_predict_endpoint():
    client = TestClient(app)

    response = client.post(
        "/predict",
        json={"text": "Платье отличное, качество хорошее"},
    )

    data = response.json()

    assert response.status_code == 200
    assert data["sentiment"] in LABELS
    assert data["confidence"] is None or 0 <= data["confidence"] <= 1


def test_predict_empty_text():
    client = TestClient(app)

    response = client.post(
        "/predict",
        json={"text": "   "},
    )

    assert response.status_code == 422