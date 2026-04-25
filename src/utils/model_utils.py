import joblib
import json

from src.config.config import *
from src.data.loader import resolve_path

def load_config(path, directory):
    with open(resolve_path(path, directory), "r") as f: return json.load(f)

def load_run_config(path): return load_config(path, RUN_DIR)

def load_model_config(path): return load_config(path, MODELS_CONFIG_DIR)

def save_model(model, name):
    path = TRAINED_MODELS_DIR / f"{name}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def load_model(name): return joblib.load(TRAINED_MODELS_DIR / f"{name}.pkl")