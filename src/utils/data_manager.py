import pandas as pd

from src.settings.config import *

from src.utils.config_utils import make_signature
from src.utils.io_utils import load_csv, save_csv
from src.utils.logging_utils import get_logger

logger = get_logger("utils.data_manager")

def get_raw_technical_filename(ticker: str, interval: str) -> str:
    return f"tech_{ticker}_{interval}.csv"

def get_raw_macro_filename(macro_features: list) -> str:
    return f"macro_{make_signature(macro_features)}.csv"

def get_processed_filename(ticker, start_date, end_date, interval, features, macro_features):
    sig = make_signature({"features": features, "macro": macro_features})
    return f"processed_{ticker}_{start_date}_{end_date}_{interval}_{sig}.csv"

def load_data(data_type: str, data_config: dict, **kwargs) -> pd.DataFrame:
    if not data_config: return pd.DataFrame()

    if data_type == "technical":
        if data_config.technical.get("force_download", False): return pd.DataFrame()
        path = RAW_DATA_DIR / get_raw_technical_filename(kwargs['ticker'], kwargs['interval'])

    elif data_type == "macro":
        if data_config.macro.get("force_download", False): return pd.DataFrame()
        path = RAW_DATA_DIR / get_raw_macro_filename(kwargs['macro_features'])

    elif data_type == "processed":
        if data_config.processed.get("force_download", False): return pd.DataFrame()
        path = PROCESSED_DATA_DIR / get_processed_filename(
            kwargs['ticker'], kwargs['start_date'], kwargs['end_date'], 
            kwargs['interval'], kwargs['features'], kwargs['macro_features'])
        
    else: return pd.DataFrame()

    return load_csv(path) if path.exists() else pd.DataFrame()

def save_data(df: pd.DataFrame, data_type: str, data_config: dict, **kwargs) -> None:
    if df is None or df.empty or data_config == None:
        logger.warning(f"Attempted to save empty {data_type} DataFrame. Skipping.")
        return
    
    save_allowed = False
    if data_type == "technical":
        save_allowed = data_config.technical.get("save", False)
    elif data_type == "macro":
        save_allowed = data_config.macro.get("save", False)
    elif data_type == "processed":
        save_allowed = data_config.processed.get("save", False)

    if not save_allowed: return
    
    path = None
    if data_type == "technical":
        path = RAW_DATA_DIR / get_raw_technical_filename(kwargs['ticker'], kwargs['interval'])
    elif data_type == "macro":
        path = RAW_DATA_DIR / get_raw_macro_filename(kwargs['macro_features'])
    elif data_type == "processed":
        path = PROCESSED_DATA_DIR / get_processed_filename(
            kwargs['ticker'], kwargs['start_date'], kwargs['end_date'], 
            kwargs['interval'], kwargs['features'], kwargs['macro_features'])

    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_csv(df, path)
        logger.info(f"Successfully saved {data_type} data to {path}")