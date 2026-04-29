import pandas as pd

from src.data.technical import load_technical
from src.data.macro import load_macro

from src.utils.csv_utils import save_processed_csv, load_processed_csv

from src.pipeline.apply_features import apply_features
from src.pipeline.preprocessing import merge_df, handle_missing

from src.config.run_config import RunConfig

def load_dataset(run : RunConfig):
    if not run.data_config["processed"]["force_download"]:
        try: return load_processed_csv(run)
        except FileNotFoundError: return pd.DataFrame()

def build_dataset(run : RunConfig, input_date=None):
    loaded = load_dataset(run)
    if input_date == None and not loaded.empty: return loaded

    technical = load_technical(run, input_date=input_date)
    macro = load_macro(run, input_date=input_date)

    df = merge_df(technical, macro)
    df = apply_features(df, run)
    df = handle_missing(df, method="ffill")
    df = handle_missing(df, "drop")

    if input_date == None: save_processed_csv(df, run)

    return df if input_date == None else df.iloc[-1:]
