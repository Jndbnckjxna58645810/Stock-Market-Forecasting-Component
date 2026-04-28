import pandas as pd

from src.data.technical import load_technical
from src.data.macro import load_macro

from src.utils.csv_utils import save_processed_csv
from src.utils.config_utils import load_run_config, load_model_config

from src.pipeline.apply_features import apply_features
from src.pipeline.preprocessing import merge_df, handle_missing, get_max_lookback

def build_dataset(run, input_date=None):
    technical = load_technical(run, input_date=input_date)
    macro = load_macro(run, input_date=input_date)

    df = merge_df(technical, macro)
    df = apply_features(df, run)
    df = handle_missing(df, method="ffill")
    df = handle_missing(df, "drop")

    if input_date == None: save_processed_csv(df, run)

    return df if input_date == None else df.iloc[-1:]
