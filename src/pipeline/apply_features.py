from src.features.features import *

FEATURE_FUNCTIONS = {
    "sma": sma, "ema": ema, "momentum": momentum, "volatility": volatility, "rsi": rsi, "lag": lag,
    "return_lag": return_lag, "macd": macd, "bband_upper": bband_upper, "bband_lower": bband_lower,
    "range" : range_feature, "volume_change" : volume_change, "hl_position" : hl_position,
    "volatility_ratio" : volatility_ratio, "volume_sma" : volume_sma, "volume_ratio" : volume_ratio,
    "body" : body, "dist_sma" : dist_sma, "macd_hist" : macd_hist, "rolling_max" : rolling_max,
    "rolling_min" : rolling_min, "breakout_up" : breakout_up, "breakout_down" : breakout_down
}

def apply_features(df, features):
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