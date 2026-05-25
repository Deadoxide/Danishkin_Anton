from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(dotenv_path=PROJECT_ROOT / "configs" / ".env")

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "women-clothing-accessories.3-class.balanced.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "reviews_prepared.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = Path(os.getenv("MODEL_PATH", ARTIFACTS_DIR / "sentiment_model.joblib"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "5000"))

RANDOM_STATE = 42
TEST_SIZE = 0.2

LABELS = ["negative", "neutral", "positive"]