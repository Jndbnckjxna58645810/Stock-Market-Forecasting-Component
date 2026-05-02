import pandas as pd

from src.settings.config import *
from src.utils.io_utils import save_csv, load_csv
from src.utils.config_utils import ensure, make_signature

from src.config.train_config import TrainConfig
from src.config.model_config import ModelConfig

def get_processed_filename(run: TrainConfig, model_config: ModelConfig):
    run = ensure(run, TrainConfig)
    if model_config == None:
        model_config = ModelConfig.from_name(run.model_config_path)
    t, s, e, i = run.ticker, run.start_date, run.end_date, run.interval

    sig = make_signature({
        "features": model_config.features,
        "macro": model_config.macro_features
    })
    return f"{t}_{s}_{e}_{i}_{sig}.csv"

def save_processed_data(df, run: TrainConfig, model_config=None):
    run = ensure(run, TrainConfig)
    if model_config is None:
        model_config = ModelConfig.from_name(run.model_config_path)
    
    if not run.data_config.get("processed", {}).get("save", False): return

    return save_csv(df, run.data_config["processed"].get("path") or (
        PROCESSED_DATA_DIR / get_processed_filename(run, model_config)))

def load_processed_data(run: TrainConfig, model_config=None):
    run = ensure(run, TrainConfig)
    if model_config == None:
        model_config = ModelConfig.from_name(run.model_config_path)
    
    path = run.data_config["processed"]["path"]
    base_name = get_processed_filename(run, model_config) if path == None else path
    
    return load_csv(base_name, PROCESSED_DATA_DIR)