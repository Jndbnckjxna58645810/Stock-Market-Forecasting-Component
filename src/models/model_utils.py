import joblib

from config.config import *

def save_model(model, name):
    path = MODELS_DIR / f"{name}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def load_model(name): return joblib.load(MODELS_DIR / f"{name}.pkl")