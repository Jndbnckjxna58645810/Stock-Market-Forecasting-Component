import pandas as pd
import numpy as np

from src.preprocessing import *

def add_sma(df, window=20):
    df[f"sma_{window}"] = df["close"].rolling(window).mean()
    return df

def add_ema(df, window=20):
    df[f"ema_{window}"] = df["close"].ewm(span=window).mean()
    return df

def add_sma(df, window=20):
    df[f"sma_{window}"] = df["close"].rolling(window).mean()
    return df

def add_ema(df, window=20):
    df[f"ema_{window}"] = df["close"].ewm(span=window).mean()
    return df

def add_momentum(df, window=5):
    df[f"momentum_{window}"] = df["close"].pct_change(window)
    return df

def add_volatility(df, window=10):
    df[f"volatility_{window}"] = df["close"].rolling(window).std()
    return df

def add_rsi(df, window=14):
    delta = df["close"].diff()

    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()

    df["rsi"] = 100 - (100 / (1 + gain / loss))

    return df

def add_macd(df):
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()

    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()

    return df

def add_bbands(df, window=20):
    sma = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()

    df["bb_upper"] = sma + 2 * std
    df["bb_lower"] = sma - 2 * std

    return df

def apply_features(df):
    df = add_sma(df, 20)
    df = add_ema(df, 20)
    df = add_momentum(df, 5)
    df = add_volatility(df, 10)
    df = add_rsi(df, 14)
    df = add_macd(df)
    df = add_bbands(df, 20)

    return clean_df(df)