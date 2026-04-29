import joblib
import json

from src.settings.config import *

from src.config.run_config import RunConfig
from src.config.model_config import ModelConfig

from src.utils.config_utils import ensure_run_config, ensure_model_config, load_model_config
    
def save_json(data, path):
    with open(path, "w") as f: json.dump(data, f, indent=4, default=str)

def save_model(model, metadata, run: RunConfig, model_config: ModelConfig):
    run = ensure_run_config(run)
    model_config = ensure_model_config(model_config)

    m, t, i = metadata["model"]["name"], metadata["ticker"], metadata["interval"]
    timestamp = metadata["created_at"]

    directory = MODELS_DIR / f"{m}_{t}_{i}_{timestamp}"
    directory.mkdir(parents=True, exist_ok=True)

    save_json(metadata, directory / "metadata.json")
    joblib.dump(model, directory / "model.pkl")

    return directory

def load_model(name): return joblib.load(MODELS_DIR / f"{name}.pkl")