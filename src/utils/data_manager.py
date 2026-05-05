import pandas as pd

from src.settings.config import *

from src.utils.config_utils import make_signature
from src.utils.io_utils import load_csv, save_csv

def get_raw_technical_filename(ticker: str, interval: str) -> str:
    return f"tech_{ticker}_{interval}.csv"

def get_raw_macro_filename(macro_features: list) -> str:
    return f"macro_{make_signature(macro_features)}.csv"

def get_processed_filename(ticker, start_date, end_date, interval, features, macro_features):
    sig = make_signature({"features": features, "macro": macro_features})
    return f"processed_{ticker}_{start_date}_{end_date}_{interval}_{sig}.csv"

def load_data(data_type: str, data_config: dict, **kwargs) -> pd.DataFrame:
    if not data_config:  return pd.DataFrame()

    type_config = data_config.get(data_type, {})
    if type_config.get("force_download", False):
        return pd.DataFrame()
        
    if data_type == "technical":
        path = RAW_DATA_DIR / get_raw_technical_filename(kwargs['ticker'], kwargs['interval'])
    elif data_type == "macro":
        path = RAW_DATA_DIR / get_raw_macro_filename(kwargs['macro_features'])
    elif data_type == "processed":
        path = PROCESSED_DATA_DIR / get_processed_filename(
            kwargs['ticker'], kwargs['start_date'], kwargs['end_date'], 
            kwargs['interval'], kwargs['features'], kwargs['macro_features']
        )
    else:
        return pd.DataFrame()

    return load_csv(path) if path.exists() else pd.DataFrame()


def save_data(df: pd.DataFrame, data_type: str, data_config: dict, **kwargs) -> None:
    if df.empty or not data_config: return
        
    type_config = data_config.get(data_type, {})
    if not type_config.get("save", False): return
        
    if data_type == "technical":
        path = RAW_DATA_DIR / get_raw_technical_filename(kwargs['ticker'], kwargs['interval'])
    elif data_type == "macro":
        path = RAW_DATA_DIR / get_raw_macro_filename(kwargs['macro_features'])
    elif data_type == "processed":
        path = PROCESSED_DATA_DIR / get_processed_filename(
            kwargs['ticker'], kwargs['start_date'], kwargs['end_date'], 
            kwargs['interval'], kwargs['features'], kwargs['macro_features'])
    else: return

    save_csv(df, path)