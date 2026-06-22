"""
Data pipeline: OHLCV fetch + technical indicator features.

NOTE: live exchange calls need outbound network access to the exchange API, which this
sandbox doesn't have. fetch_live_ohlcv() will work fine on your own machine. For testing
inside this environment, use generate_synthetic_ohlcv() instead — see main.py.
"""

import numpy as np
import pandas as pd
import ta

import config


def fetch_live_ohlcv(symbol: str = None, timeframe: str = None, limit: int = None) -> pd.DataFrame:
    """Pulls OHLCV from a real exchange via ccxt. Run this on a machine with network access."""
    import ccxt

    symbol = symbol or config.SYMBOL
    timeframe = timeframe or config.TIMEFRAME
    limit = limit or config.LOOKBACK_CANDLES

    exchange_class = getattr(ccxt, config.EXCHANGE)
    exchange = exchange_class({"enableRateLimit": True})

    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


def generate_synthetic_ohlcv(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Random-walk synthetic price series, ONLY for demoing the pipeline end-to-end inside
    this sandbox. Do not draw any trading conclusions from results on synthetic data —
    it has no real market structure.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0001, scale=0.012, size=n)
    close = 50000 * np.exp(np.cumsum(returns))

    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    volume = rng.lognormal(mean=10, sigma=0.5, size=n)

    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="4h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds RSI, MACD, Bollinger Bands, ATR. Returns a new dataframe (input not mutated)."""
    out = df.copy()

    out["rsi"] = ta.momentum.RSIIndicator(out["close"], window=config.RSI_PERIOD).rsi()

    macd = ta.trend.MACD(
        out["close"],
        window_fast=config.MACD_FAST,
        window_slow=config.MACD_SLOW,
        window_sign=config.MACD_SIGNAL,
    )
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_hist"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(out["close"], window=config.BOLLINGER_PERIOD)
    out["bb_high"] = bb.bollinger_hband()
    out["bb_low"] = bb.bollinger_lband()
    out["bb_pct"] = bb.bollinger_pband()  # where price sits within the bands, 0-1

    out["atr"] = ta.volatility.AverageTrueRange(
        out["high"], out["low"], out["close"], window=config.ATR_PERIOD
    ).average_true_range()

    out["returns_1"] = out["close"].pct_change(1)
    out["returns_5"] = out["close"].pct_change(5)
    out["volume_zscore"] = (out["volume"] - out["volume"].rolling(50).mean()) / out["volume"].rolling(50).std()

    # Target: did price go up over the next PREDICTION_HORIZON candles (binary classification)
    out["target"] = (
        out["close"].shift(-config.PREDICTION_HORIZON) > out["close"]
    ).astype(int)

    out.dropna(inplace=True)
    return out


FEATURE_COLUMNS = [
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_pct", "atr", "returns_1", "returns_5", "volume_zscore",
]
