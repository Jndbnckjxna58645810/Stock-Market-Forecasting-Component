import joblib
from tensorflow.keras.models import load_model as keras_load_model # type: ignore

from src.settings.config import *

from src.utils.logging_utils import get_logger

logger = get_logger("models.load_models.load_model_sequence")

def load_model_sequence(name):
    path = MODELS_DIR / name

    logger.info(f"Loading model from {path}")
    return {
        "model": keras_load_model(path / "model.keras"),
        "x_scaler": joblib.load(path / "x_scaler.pkl"),
        "y_scaler": joblib.load(path / "y_scaler.pkl")
    }