import pandas as pd
import numpy as np

from src.preprocessing.preprocessing import *
from src.data.loader import *
from src.config import *

def ema(df, window=20): return df["close"].ewm(span=window).mean()

def sma(df, window=20): return df["close"].rolling(window).mean()

def momentum(df, window=5): return df["close"].pct_change(window)

def volatility(df, window=10): return df["close"].rolling(window).std()

def rsi(df, window=14):
    delta = df["close"].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean().replace(0, np.nan)

    return 100 - (100 / (1 + avg_gain / avg_loss))

def macd(df):
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()

    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9).mean()

    return macd_line, signal

def bband_upper(df, window=20): return sma(df, window) + 2 * volatility(df, window)

def bband_lower(df, window=20): return sma(df, window) - 2 * volatility(df, window)

def lag(df, n=1): return df["close"].shift(n)

def return_lag(df): return df["close"].pct_change().shift(1)