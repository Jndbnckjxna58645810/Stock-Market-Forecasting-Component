from src.features.features import *

from src.utils.config_utils import load_run_config, load_model_config

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

def apply_features(df, run):
    features = load_model_config(load_run_config(run)["model_config"])["features"]
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
        else:
            if custom_name: col_name = custom_name
            else:
                suffix = "_".join(str(v) for v in params.values()) if params else ""
                col_name = f"{name}_{suffix}" if suffix else name
            df[col_name] = result
            
    return df

def get_feature_names(features_config):
    names = []
    for f in features_config:
        name = f["name"]
        params = f.get("params", {})

        if "window" in params:
            names.append(f"{name}_{params['window']}")
        elif "n" in params:
            names.append(f"{name}_{params['n']}")
        else:
            names.append(name)

    return names