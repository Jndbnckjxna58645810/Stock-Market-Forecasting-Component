import joblib

from src.settings.config import *

from src.utils.logging_utils import get_logger

logger = get_logger("models.load_models.load_model_tabular")

def load_model_tabular(name):
    path = MODELS_DIR / name

    logger.info(f"Loading model from {path}")
    return {"model": joblib.load(path / "model.pkl")}