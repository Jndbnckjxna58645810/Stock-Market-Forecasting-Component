from src.features.features import *

from src.utils.logging_utils import get_logger

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
    for i, feature in enumerate(features):
        name = feature["name"]
        params = feature.get("params", {})

        func = FEATURE_FUNCTIONS[name]
        result = func(df, **params)

        param_str = "_".join(f"{k[0]}{v}" for k, v in params.items())
        suffix = f"_{i}_{param_str}" if param_str else f"_{i}"

        if isinstance(result, tuple):
            for j, col_data in enumerate(result):
                col_name = f"{name}_{j}{suffix}"
                df[col_name] = col_data
                logger.info(f"Feature applied: {col_name}")
        else:
            col_name = f"{name}{suffix}"
            df[col_name] = result
            logger.info(f"Feature applied: {col_name}")
            
    return df