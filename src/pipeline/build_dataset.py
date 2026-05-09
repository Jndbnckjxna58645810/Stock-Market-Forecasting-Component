import pandas as pd

from src.data.technical import load_technical_by_parameters
from src.data.macro import load_macro_by_parameters

from src.utils.logging_utils import get_logger
from src.utils.data_manager import save_data, load_data

from src.pipeline.apply_features import apply_features_by_parameters
from src.pipeline.preprocessing import handle_missing, get_max_lookback_by_parameters, get_max_horizon_by_parameters, merge_and_align_datasets, convert_bars_to_days

logger = get_logger("pipeline.build_dataset")

def build_dataset_by_parameters(ticker, start_date, end_date, interval,
                                features, macro_features, targets,
                                hyperparameters, data_config=None):
    lookback_days = convert_bars_to_days((hyperparameters.get("seq_len", 0) * 2
                                          + get_max_lookback_by_parameters(features)), interval)
    lookforward_days = convert_bars_to_days(get_max_horizon_by_parameters(targets), interval)
    
    effective_start = pd.to_datetime(start_date) - pd.DateOffset(days=lookback_days)
    effective_end = pd.to_datetime(end_date) + pd.DateOffset(days=lookforward_days)

    technical_raw = load_data("technical", data_config, ticker=ticker, interval=interval)
    if technical_raw.empty:
        technical_raw = load_technical_by_parameters(ticker, effective_start, effective_end, interval)
        save_data(technical_raw, "technical", data_config, ticker=ticker, interval=interval)
    technical = technical_raw.loc[effective_start:effective_end]

    macro_raw = load_data("macro", data_config, macro_features=macro_features)
    if macro_raw.empty:
        macro_raw = load_macro_by_parameters(macro_features, effective_start, effective_end)
        save_data(macro_raw, "macro", data_config, macro_features=macro_features)
    macro = macro_raw.loc[effective_start:effective_end]

    df = merge_and_align_datasets(technical, macro)
    df = apply_features_by_parameters(df, features)
    df = handle_missing(df, method="ffill")
    df = df.dropna()
    
    last_date = pd.to_datetime(df.index[-1]).date()
    requested_start = pd.to_datetime(start_date).date()

    if df.empty or (last_date < requested_start):
        logger.error(f"Insufficient data for {ticker} after offsets.")
        raise ValueError(f"Insufficient data for {ticker} after offsets.")
    
    logger.info(f"Processed dataset built for {ticker}" +
                f" | Period: {start_date} to {end_date}" +
                f" | Offset period: {effective_start} to {effective_end}" +
                f" | Interval: {interval}")
    return df