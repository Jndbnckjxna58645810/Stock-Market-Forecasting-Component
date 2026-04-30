import pandas as pd

from src.settings.config import *
from src.utils.vesrioning_utils import make_signature
from src.utils.path_utils import resolve_path

from src.config.train_config import TrainConfig
from src.config.model_config import ModelConfig

def save_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return path

def save_processed_csv(df, run):
    if not isinstance(run, TrainConfig): run = TrainConfig.from_name(run)

    t = run.ticker
    s, e = run.start_date, run.end_date
    i = run.interval
    features=ModelConfig.from_name(run.model_config_path).features

    if not run.data_config["processed"]["save"]: return

    base_name = f"{t}_{s}_{e}_{i}_{make_signature(features)}.csv"
    path = run.data_config["processed"]["path"]

    return save_csv(df, PROCESSED_DATA_DIR / base_name if path == None else path)

def load_csv(path): return pd.read_csv(resolve_path(path), index_col=0, parse_dates=True)

def load_processed_csv(run : TrainConfig):
    t = run.ticker
    s, e = run.start_date, run.end_date
    i = run.interval
    features=ModelConfig.from_name(run.model_config_path).features

    base_name = f"{t}_{s}_{e}_{i}_{make_signature(features)}.csv"
    path = run.data_config["processed"]["path"]
    
    return pd.read_csv(resolve_path(base_name if path == None else path, PROCESSED_DATA_DIR), index_col=0, parse_dates=True)