import joblib

from src.settings.config import *

def load_model_tabular(name):
    return joblib.load(MODELS_DIR / f"{name}" / "model.pkl")