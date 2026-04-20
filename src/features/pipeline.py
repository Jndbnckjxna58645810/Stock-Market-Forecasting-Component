from src.features.features import *

FEATURE_FUNCTIONS = {
    "sma": sma, "ema": ema, "momentum": momentum, "volatility": volatility, "rsi": rsi, "lag": lag,
    "return_lag": return_lag, "macd": macd, "bband_upper": bband_upper, "bband_lower": bband_lower
}

def apply_features(df, features):
    for name, params in features:
        func = FEATURE_FUNCTIONS[name]

        result = func(df, **params)

        if isinstance(result, tuple):
            for i, col in enumerate(result):
                df[f"{name}_{i}"] = col
        else:
            suffix = "_".join(str(v) for v in params.values()) if params else ""
            col_name = f"{name}_{suffix}" if suffix else name

            df[col_name] = result

    return df