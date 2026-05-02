import joblib

from src.settings.config import *

from src.config.train_config import TrainConfig
from src.config.model_config import ModelConfig

from src.utils.config_utils import ensure
from src.utils.io_utils import save_json
from src.utils.logging_utils import get_logger

from src.config.train_config import TrainConfig
from src.config.model_config import ModelConfig

logger = get_logger("models.save_models.save_model_sequence")

def save_model_sequence(model_bundle, metadata, run: TrainConfig, model_config=None):
    run = ensure(run, TrainConfig)
    if model_config == None:
        model_config = ModelConfig.from_name(run.model_config_path)
    model_config = ensure(model_config, ModelConfig)

    m, t, i = metadata["model"]["name"], metadata["ticker"], metadata["interval"]
    timestamp = metadata["created_at"]

    base_name = f"{m}_{t}_{i}_{timestamp}"
    directory = MODELS_DIR / base_name
    directory.mkdir(parents=True, exist_ok=True)

    model_bundle["model"].save(directory / "model.keras")

    joblib.dump(model_bundle["x_scaler"], directory / "x_scaler.pkl")
    joblib.dump(model_bundle["y_scaler"], directory / "y_scaler.pkl")

    save_json(metadata, directory / "metadata.json")

    logger.info(f"Model saved to directory {base_name} | Full path: {directory}")

    return base_name