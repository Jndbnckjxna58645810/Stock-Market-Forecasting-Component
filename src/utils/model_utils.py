import joblib

from src.config.config import *

def save_model(model, model_config, X_columns):
    joblib.dump({"model": model, "config": model_config, "features": X_columns.tolist()},
                MODELS_DIR / f"example.pkl")

def load_model(name): return joblib.load(MODELS_DIR / f"{name}.pkl")