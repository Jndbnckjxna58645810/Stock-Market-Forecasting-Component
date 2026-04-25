from src.data.sources.technical import load_technical
from src.data.sources.macro import load_macro
from src.data.loader import save_processed_csv
from src.features.pipeline import apply_features
from src.preprocessing.preprocessing import merge_df, handle_missing, normalize_df
from src.utils.model_utils import load_run_config, load_model_config

import datetime as dt
import pandas as pd

def get_config_parameters(run):
    run_config = load_run_config(run)
    return run_config["ticker"], run_config["start_date"], run_config["end_date"], run_config["interval"], load_model_config(run_config["model_config"])

def get_max_lookback(features):
    max_lookback = 0

    for f in features:
        params = f.get("params", {})
        for v in params.values():
            if isinstance(v, int):
                max_lookback = max(max_lookback, v)

    return max_lookback

def build_dataset(ticker, start_date, end_date, interval, model_config, save_technical=False, save_macro=False, save_processed=False, force_download=False):
    df = load_technical(ticker, start_date, end_date, offset=get_max_lookback(model_config["features"]),
                        interval=interval, save=save_technical, force_download=force_download)
    macro = load_macro(model_config["macro_features"], start_date, end_date, save=save_macro)

    df = normalize_df(df, ticker=ticker)
    df = merge_df(df, macro)
    df = apply_features(df, model_config["features"])
    df = handle_missing(df, method="ffill")
    df = handle_missing(df, "drop")

    if save_processed: save_processed_csv(df, ticker, start_date, end_date, interval=interval)
    return df

def build_dataset_from_run(run, save_technical=False, save_macro=False, save_processed=False, force_download=False):
    return build_dataset(get_config_parameters(run), save_technical=save_technical, save_macro=save_macro, save_processed=save_processed, force_download=force_download)

def build_input(run, date):
    ticker, start_date, end_date, interval, model_config = get_config_parameters(run)
    return build_dataset(ticker, date - pd.DateOffset(days=get_max_lookback(model_config["features"])),
                         date, interval, model_config).iloc[-1:]
