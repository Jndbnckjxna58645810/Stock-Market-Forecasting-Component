import numpy as np
import pandas as pd

def target_return(df, horizon=1, log=False, smoothing=1):
    if log: ret = np.log(df["close"] / df["close"].shift(1))
    else: ret = df["close"].pct_change()

    target = ret.shift(-horizon)

    return target if smoothing <= 1 else target.rolling(smoothing).mean()

def target_direction(df, horizon=1, threshold=0.0):
    return (df["close"].pct_change().shift(-horizon) > threshold).astype(int)

def target_price(df, horizon=1): return df["close"].shift(-horizon)

def target_multi_return(df, horizons=[1,2,3]):
    ret = df["close"].pct_change()

    data = {}
    for h in horizons: data[f"target_return_h{h}"] = ret.shift(-h)

    return pd.DataFrame(data)