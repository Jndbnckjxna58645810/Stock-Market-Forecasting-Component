import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import numpy as np
import random
import os

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

CONFIG_DIR = BASE_DIR / "config"

MODELS_DIR = BASE_DIR / "models"
MODELS_CONFIG_DIR = CONFIG_DIR / "models"
TRAIN_CONFIG_DIR = CONFIG_DIR / "train"
PREDICTION_CONFIG_DIR = CONFIG_DIR / "predictions"
EVALUATE_CONFIG_DIR = CONFIG_DIR / "evaluate"
