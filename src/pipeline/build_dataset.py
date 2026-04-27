import datetime as dt
import pandas as pd

from src.data.technical import load_technical
from src.data.macro import load_macro

from src.utils.csv_utils import save_processed_csv
from src.utils.config_utils import get_config_parameters

from src.pipeline.apply_features import apply_features
from src.pipeline.preprocessing import merge_df, handle_missing, normalize_df, get_max_lookback

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
    ticker, start_date, end_date, interval, model_config = get_config_parameters(run)
    return build_dataset(ticker, start_date, end_date, interval, model_config, save_technical=save_technical, save_macro=save_macro, save_processed=save_processed, force_download=force_download)

def build_input(run, date):
    ticker, start_date, end_date, interval, model_config = get_config_parameters(run)
    return build_dataset(ticker, date - pd.DateOffset(days=get_max_lookback(model_config["features"])),
                         date, interval, model_config).iloc[-1:]
