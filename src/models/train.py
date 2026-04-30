from src.utils.config_utils import ensure

from src.models.train_models.train_lstm import train_lstm
from src.models.train_models.train_rf import train_rf
from src.models.train_models.train_xgb import train_xgb

from src.config.run_config import RunConfig
from src.config.model_config import ModelConfig

def train(run: RunConfig, model_config: ModelConfig):
    run = ensure(run, RunConfig)
    if run.model_config_path == None: raise ValueError("Training requires model_config")
    model_config = ensure(model_config, ModelConfig)

    model_name = model_config.model["name"]
    if model_name == "xgb": return train_xgb(run, model_config)
    elif model_name == "rf": return train_rf(run, model_config)
    elif model_name == "lstm": return train_lstm(run, model_config)
    else: raise ValueError("Unknown model")
