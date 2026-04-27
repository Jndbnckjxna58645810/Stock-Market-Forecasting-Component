import numpy as np

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

def return_lag(df, n=1): return df["close"].pct_change().shift(n)

def range_feature(df): return (df["high"] - df["low"]) / df["close"]

def volume_change(df): return df["volume"].pct_change()

def hl_position(df): return (df["close"] - df["low"]) / (df["high"] - df["low"])

def body(df): return (df["close"] - df["open"]) / df["open"]

def volatility_ratio(df): return volatility(df) / df["close"]

def volume_sma(df, window=10): return df["volume"].rolling(window).mean()

def volume_ratio(df): return df["volume"] / volume_sma(df)

def dist_sma(df, window=20):
    sma_window = sma(df, window)
    return (df["close"] - sma_window) / sma_window

def macd_hist(df):
    macd_line, signal = macd(df)
    return macd_line - signal

def rolling_max(df, window=10): return df["high"].rolling(window).max()

def rolling_min(df, window=10): return df["low"].rolling(window).max()

def breakout_up(df): return (df["close"] > rolling_max(df)).astype(int)

def breakout_down(df): return (df["close"] > rolling_min(df)).astype(int)