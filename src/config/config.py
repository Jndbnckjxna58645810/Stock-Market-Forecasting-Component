import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

BASE_DIR = Path(__file__).resolve().parent.parent.parent

RUN_DIR = BASE_DIR / "runs"

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
MODELS_CONFIG_DIR = MODELS_DIR / "config"
TRAINED_MODELS_DIR = MODELS_DIR / "trained_models"