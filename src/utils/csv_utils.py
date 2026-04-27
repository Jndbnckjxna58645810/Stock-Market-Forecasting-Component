import pandas as pd

from src.config.config import *
from src.utils.vesrioning_utils import get_next_version
from src.utils.path_utils import resolve_path

def save_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)

def save_processed_csv(df, ticker, start_date, end_date, interval="1d", path=None):
    base_name = f"{ticker}_{start_date}_{end_date}_{interval}"
    save_csv(df, PROCESSED_DATA_DIR / f"{base_name}_v{get_next_version(base_name, PROCESSED_DATA_DIR)}.csv" if path == None else path)
    return path

def load_csv(path): return pd.read_csv(resolve_path(path), index_col=0, parse_dates=True)