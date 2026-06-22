"""
Sab adjustable settings yahan hain. Code mein kahin aur hardcode nahi kiya.
"""

# --- Market ---
EXCHANGE = "binance"          # any ccxt-supported exchange id
SYMBOL = "BTC/USDT"           # change for XAU/Oil etc (use the right exchange for that asset class)
TIMEFRAME = "4h"              # swing style. Use "1m"/"5m" only if you actually have low-latency infra.
LOOKBACK_CANDLES = 2000

# --- Features ---
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BOLLINGER_PERIOD = 20
ATR_PERIOD = 14

# --- Model / Validation ---
N_WALKFORWARD_SPLITS = 5
PREDICTION_HORIZON = 1         # predict direction N candles ahead
PROBABILITY_THRESHOLD = 0.60   # only trade if model confidence > this (NOT 0.92 — see README)

# --- Risk Engine ---
MAX_RISK_PER_TRADE = 0.01      # 1% of equity per trade, per your own earlier note
KELLY_FRACTION_CAP = 0.5       # use half-Kelly, full Kelly is too aggressive for live trading
MAX_DRAWDOWN_HALT = 0.15       # stop trading if equity drawdown exceeds 15%, review before resuming
MIN_REWARD_RISK_RATIO = 2.0

# --- Backtest costs (do NOT zero these out, real numbers depend on your exchange) ---
TAKER_FEE = 0.0004
SLIPPAGE_BPS = 5                # basis points
FUNDING_RATE_8H = 0.0001        # only relevant for perpetual futures
