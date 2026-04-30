import pandas as pd
import datetime as dt

from src.data.technical import load_technical_dataset, load_technical_input
from src.data.macro import load_macro_dataset, load_macro_input

from src.utils.csv_utils import save_processed_csv, load_processed_csv
from src.utils.config_utils import ensure

from src.pipeline.apply_features import apply_features_to_dataset, apply_features_to_input
from src.pipeline.preprocessing import merge_df, handle_missing

from src.config.run_config import RunConfig
from src.config.predict_config import PredictConfig

def load_dataset(run: RunConfig):
    run = ensure(run, RunConfig)
    if not run.data_config["processed"]["force_download"]:
        try: return load_processed_csv(run)
        except FileNotFoundError: return pd.DataFrame()

def build_dataset(run: RunConfig):
    run = ensure(run, RunConfig)

    loaded = load_dataset(run)
    if not run.data_config["processed"]["force_download"] or not loaded.empty: return loaded

    technical = load_technical_dataset(run)
    macro = load_macro_dataset(run)

    df = merge_df(technical, macro)
    df = apply_features_to_dataset(df, run)
    df = handle_missing(df, method="ffill")
    df = handle_missing(df, "drop")

    save_processed_csv(df, run)
    return df

def build_input(predict_config: PredictConfig):
    predict_config = ensure(predict_config, PredictConfig)

    technical = load_technical_input(predict_config)
    macro = load_macro_input(predict_config)

    df = merge_df(technical, macro)
    df = apply_features_to_input(df, predict_config)
    df = handle_missing(df, method="ffill")

    return df.loc[predict_config.input_date:].iloc[0:1]
