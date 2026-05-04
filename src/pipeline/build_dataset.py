import pandas as pd

from src.data.technical import load_technical_dataset, load_technical_input, load_technical_evaluation_dataset
from src.data.macro import load_macro_dataset, load_macro_input, load_macro_evaluation_dataset

from src.utils.data_manager import save_processed_data, load_processed_data
from src.utils.config_utils import ensure
from src.utils.logging_utils import get_logger

from src.pipeline.apply_features import apply_features_to_dataset, apply_features_to_input, apply_features_to_evaluation_dataset
from src.pipeline.preprocessing import merge_df, handle_missing

from src.config.train_config import TrainConfig
from src.config.predict_config import PredictConfig
from src.config.evaluate_config import EvaluateConfig
from src.config.model_metadata import ModelMetadata
from src.config.model_config import ModelConfig

logger = get_logger("pipeline.build_dataset")

def load_dataset(run: TrainConfig, model_config=None):
    run = ensure(run, TrainConfig)
    if model_config == None:
        if run.model_config_path == None: raise ValueError("Training requires model_config")
        model_config = ModelConfig.from_name(run.model_config_path)

    if not run.data_config["processed"]["force_download"]:
        try: return load_processed_data(run, model_config)
        except FileNotFoundError: return pd.DataFrame()
    return pd.DataFrame()

def build_dataset(run: TrainConfig, model_config=None):
    run = ensure(run, TrainConfig)
    if model_config == None:
        if run.model_config_path == None: raise ValueError("Training requires model_config")
        model_config = ModelConfig.from_name(run.model_config_path)

    loaded = load_dataset(run, model_config)
    if not loaded.empty:
        logger.info(f"Saved processed dataset loaded for {run.ticker}" +
                    f" | Period: {run.start_date} to {run.end_date}" +
                    f" | Interval: {run.interval}" +
                    f" | Features from configuration file: {run.model_config_path}")

        return loaded

    technical = load_technical_dataset(run)
    macro = load_macro_dataset(run, model_config)

    df = merge_df(technical, macro)
    df = apply_features_to_dataset(df, run, model_config)
    df = handle_missing(df, method="ffill")
    df = handle_missing(df, "drop")
    df = df.loc[run.start_date:run.end_date]

    logger.info(f"Processed dataset built for {run.ticker}" +
                f" | Period: {run.start_date} to {run.end_date}" +
                f" | Interval: {run.interval}" + (
                    f" | Features from configuration file: {run.model_config_path}"
                    if run.model_config_path else ""))

    save_processed_data(df, run, model_config)
    return df

def build_input(predict_config: PredictConfig):
    predict_config = ensure(predict_config, PredictConfig)
    model_metadata = ModelMetadata.from_name(predict_config.model_path)

    technical = load_technical_input(predict_config)
    macro = load_macro_input(predict_config)

    df = merge_df(technical, macro)
    df = apply_features_to_input(df, predict_config)
    df = handle_missing(df, method="ffill")

    if model_metadata.hyperparameters.get("seq_len"):
            offset_start = pd.to_datetime(predict_config.start_date) - pd.DateOffset(
                days=model_metadata.hyperparameters.get("seq_len")*2)
            df = df.loc[offset_start:predict_config.end_date]
    else:
        df = df.loc[predict_config.start_date:predict_config.end_date]

    message = (f"Processed input built for {model_metadata.ticker}" +
               f" | Period: {model_metadata.start_date} to {model_metadata.end_date}" +
               f" | Interval: {model_metadata.interval}" +
               f" | Features from model metadata: {predict_config.model_path}")
    logger.info(f"Processed input built for {message}")

    if df.empty:
        logger.error(f"No available data for {message}")

        raise ValueError(f"No available data for {message}")
    return df

def build_evaluation_dataset(evaluate_config: EvaluateConfig, model_metadata: ModelMetadata):
    evaluate_config = ensure(evaluate_config, EvaluateConfig)
    model_metadata = ensure(model_metadata, ModelMetadata)

    technical = load_technical_evaluation_dataset(evaluate_config, model_metadata)
    macro = load_macro_evaluation_dataset(evaluate_config, model_metadata)

    df = merge_df(technical, macro)
    df = apply_features_to_evaluation_dataset(df, model_metadata)
    df = handle_missing(df, method="ffill")
    df = handle_missing(df, "drop")
    
    if model_metadata.hyperparameters.get("seq_len"):
            offset_start = pd.to_datetime(evaluate_config.start_date) - pd.DateOffset(
                days=model_metadata.hyperparameters.get("seq_len")*2)
            df = df.loc[offset_start:evaluate_config.end_date]
    else:
        df = df.loc[evaluate_config.start_date:evaluate_config.end_date]

    logger.info(f"Processed evaluation built for {model_metadata.ticker}" +
                f" | Period: {model_metadata.start_date} to {model_metadata.end_date}" +
                f" | Interval: {model_metadata.interval}" +
                f" | Features from model metadata (evaluation)")

    return df
