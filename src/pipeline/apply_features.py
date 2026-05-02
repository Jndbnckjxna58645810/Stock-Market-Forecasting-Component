from src.features.features import *

from src.utils.config_utils import ensure
from src.utils.logging_utils import get_logger

from src.config.train_config import TrainConfig
from src.config.model_config import ModelConfig
from src.config.predict_config import PredictConfig
from src.config.model_metadata import ModelMetadata

logger = get_logger("pipeline.apply_features")

FEATURE_FUNCTIONS = {
    "sma": sma, "ema": ema, "momentum": momentum,
    "volatility": volatility, "volatility_ratio" : volatility_ratio,
    "rsi": rsi, "macd": macd, "macd_hist" : macd_hist,
    "lag": lag, "return_lag": return_lag, "log_return": log_return,
    "bband_upper": bband_upper, "bband_lower": bband_lower,
    "range" : range_feature, "hl_position" : hl_position,
    "volume_change" : volume_change, "volume_sma" : volume_sma, "volume_ratio" : volume_ratio,
    "body" : body, "dist_sma" : dist_sma, "zscore_close" : zscore_close,
    "rolling_max" : rolling_max, "rolling_min" : rolling_min,
    "breakout_up" : breakout_up, "breakout_down" : breakout_down,
    "day_of_week" : day_of_week_feature, "month" : month_feature
}

def apply_features_by_parameters(df, features):
    for feature in features:
        name = feature["name"]
        params = feature.get("params", {})
        custom_name = feature.get("col_name")

        func = FEATURE_FUNCTIONS[name]
        result = func(df, **params)

        if isinstance(result, tuple):
            for i, col in enumerate(result):
                col_name = (
                    custom_name[i]
                    if isinstance(custom_name, list)
                    else f"{name}_{i}"
                )
                df[col_name] = col

                logger.info(f"Feature applied: {col_name} from name '{name}' with parameters: {params}")
        else:
            if custom_name: col_name = custom_name
            else:
                suffix = "_".join(str(v) for v in params.values()) if params else ""
                col_name = f"{name}_{suffix}" if suffix else name
            df[col_name] = result

            logger.info(f"Feature applied: {col_name} from name '{name}' with parameters: {params}")
            
    return df

def apply_features_to_dataset(df, run: TrainConfig, model_config=None):
    run = ensure(run, TrainConfig)
    if model_config == None:
        model_config = ModelConfig.from_name(run.model_config_path)

    logger.info(f"Applying features to training dataset for {run.ticker}" +
                    f" | Period: {run.start_date} to {run.end_date}" +
                    f" | Interval: {run.interval}" + (
                        f" | Features from configuration file: {run.model_config_path}"
                        if run.model_config_path else ""))

    return apply_features_by_parameters(df, model_config.features)

def apply_features_to_input(df, predict_config: PredictConfig):
    predict_config = ensure(predict_config, PredictConfig)

    model_metadata = ModelMetadata.from_name(predict_config.model_path)
    logger.info(f"Applying features to input for {model_metadata.ticker}" +
                f" | Period: {model_metadata.start_date} to {model_metadata.end_date}" +
                f" | Interval: {model_metadata.interval}" +
                f" | Features from model metadata: {predict_config.model_path}")

    return apply_features_by_parameters(df, ModelMetadata.from_name(predict_config.model_path).features)

def apply_features_to_evaluation_dataset(df, model_metadata: ModelMetadata):
    model_metadata = ensure(model_metadata, ModelMetadata)

    logger.info(f"Applying features to evaluation dataset for {model_metadata.ticker}" +
                f" | Period: {model_metadata.start_date} to {model_metadata.end_date}" +
                f" | Interval: {model_metadata.interval}" +
                f" | Features from model metadata (evaluation)")

    return apply_features_by_parameters(df, model_metadata.features)
