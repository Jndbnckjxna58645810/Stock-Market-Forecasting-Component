import pandas as pd

from src.config.config import *
from src.utils.vesrioning_utils import make_signature
from src.utils.path_utils import resolve_path
from src.utils.config_utils import load_model_config, load_run_config

def save_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return path

def save_processed_csv(df, run):
    run_config = load_run_config(run)

    t = run_config["ticker"]
    s, e = run_config["start_date"], run_config["end_date"]
    i = run_config["interval"]
    features=load_model_config(run_config["model_config"])["features"]

    if not run_config["data"]["processed"]["save"]: return

    base_name = f"{t}_{s}_{e}_{i}_{make_signature(features)}.csv"
    path = run_config["data"]["processed"]["path"]

    return save_csv(df, PROCESSED_DATA_DIR / base_name if path == None else path)

def load_csv(path): return pd.read_csv(resolve_path(path), index_col=0, parse_dates=True)