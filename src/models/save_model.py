from src.utils.config_utils import ensure

from src.models.predict_models.predict_tabular import predict_tabular
from src.models.predict_models.predict_sequence import predict_sequence

from src.config.train_config import TrainConfig
from src.config.model_config import ModelConfig

from src.models.save_models.save_model_tabular import save_model_tabular
from src.models.save_models.save_model_sequence import save_model_sequence

def save_model(model, metadata, run: TrainConfig, model_config=None):
    run = ensure(run, TrainConfig)
    if model_config == None:
        model_config = ModelConfig.from_name(run.model_config_path)
    model_config = ensure(model_config, ModelConfig)

    model_name = model_config.model["name"]
    if model_name == "xgb" or model_name == "rf": return save_model_tabular(model, metadata, run, model_config)
    elif model_name == "lstm": return save_model_sequence(model, metadata, run, model_config)
    else: raise ValueError("Unknown model")