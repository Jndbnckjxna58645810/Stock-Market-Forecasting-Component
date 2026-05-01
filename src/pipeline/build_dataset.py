import pandas as pd

from src.data.technical import load_technical_dataset, load_technical_input, load_technical_evaluation_dataset
from src.data.macro import load_macro_dataset, load_macro_input, load_macro_evaluation_dataset

from src.utils.csv_utils import save_processed_csv, load_processed_csv
from src.utils.config_utils import ensure

from src.pipeline.apply_features import apply_features_to_dataset, apply_features_to_input, apply_features_to_evaluation_dataset
from src.pipeline.preprocessing import merge_df, handle_missing

from src.config.train_config import TrainConfig
from src.config.predict_config import PredictConfig
from src.config.evaluate_config import EvaluateConfig
from src.config.model_metadata import ModelMetadata
from src.config.model_config import ModelConfig

def load_dataset(run: TrainConfig, model_config=None):
    run = ensure(run, TrainConfig)
    if model_config == None:
        if run.model_config_path == None: raise ValueError("Training requires model_config")
        model_config = ModelConfig.from_name(run.model_config_path)

    if not run.data_config["processed"]["force_download"]:
        try: return load_processed_csv(run, model_config)
        except FileNotFoundError: return pd.DataFrame()
    return pd.DataFrame()

def build_dataset(run: TrainConfig, model_config=None):
    run = ensure(run, TrainConfig)
    if model_config == None:
        if run.model_config_path == None: raise ValueError("Training requires model_config")
        model_config = ModelConfig.from_name(run.model_config_path)

    loaded = load_dataset(run, model_config)
    if not loaded.empty: return loaded

    technical = load_technical_dataset(run)
    macro = load_macro_dataset(run, model_config)

    df = merge_df(technical, macro)
    df = apply_features_to_dataset(df, run, model_config)
    df = handle_missing(df, method="ffill")
    df = handle_missing(df, "drop")
    df = df.loc[run.start_date:run.end_date]

    save_processed_csv(df, run, model_config)
    return df

def build_input(predict_config: PredictConfig):
    predict_config = ensure(predict_config, PredictConfig)

    technical = load_technical_input(predict_config)
    macro = load_macro_input(predict_config)

    df = merge_df(technical, macro)
    df = apply_features_to_input(df, predict_config)
    df = handle_missing(df, method="ffill")
    df = df.loc[predict_config.start_date:predict_config.end_date]

    if df.empty: raise ValueError("No available data before input_date")
    return df

def build_evaluation_dataset(evaluate_config: EvaluateConfig, model_metadata: ModelMetadata):
    evaluate_config = ensure(evaluate_config, EvaluateConfig)
    model_metadata = ensure(model_metadata, ModelMetadata)

    technical = load_technical_evaluation_dataset(evaluate_config)
    macro = load_macro_evaluation_dataset(evaluate_config, model_metadata)

    df = merge_df(technical, macro)
    df = apply_features_to_evaluation_dataset(df, model_metadata)
    df = handle_missing(df, method="ffill")
    df = handle_missing(df, "drop")

    return df.loc[evaluate_config.start_date:evaluate_config.end_date]
