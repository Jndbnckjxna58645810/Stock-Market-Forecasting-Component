import joblib

from src.data_loader import MODELS_DIR

def save_model(model, name):
    path = MODELS_DIR / f"{name}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def load_model(name): return joblib.load(MODELS_DIR / f"{name}.pkl")