import joblib

from src.settings.config import *

from src.config.train_config import TrainConfig
from src.config.model_config import ModelConfig

from src.utils.config_utils import ensure
from src.utils.io_utils import save_json

from src.config.train_config import TrainConfig
from src.config.model_config import ModelConfig

def save_model_tabular(model_bundle, metadata, run: TrainConfig, model_config=None):
    run = ensure(run, TrainConfig)
    if model_config == None:
        model_config = ModelConfig.from_name(run.model_config_path)
    model_config = ensure(model_config, ModelConfig)

    m, t, i = metadata["model"]["name"], metadata["ticker"], metadata["interval"]
    timestamp = metadata["created_at"]

    directory = MODELS_DIR / f"{m}_{t}_{i}_{timestamp}"
    directory.mkdir(parents=True, exist_ok=True)

    save_json(metadata, directory / "metadata.json")
    joblib.dump(model_bundle["model"], directory / "model.pkl")

    x_scaler, y_scaler = model_bundle.get("x_scaler"), model_bundle.get("y_scaler")
    
    if x_scaler: joblib.dump(x_scaler, directory / "x_scaler.pkl")
    if y_scaler: joblib.dump(y_scaler, directory / "y_scaler.pkl")

    return directory