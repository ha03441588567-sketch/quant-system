"""
Simple vectorized backtest. Deliberately NOT using leverage by default, and deliberately
including every cost line item — a backtest that ignores fees/slippage is just fiction.
"""

import numpy as np
import pandas as pd

import config
import risk_engine


def run_backtest(df: pd.DataFrame, probabilities: np.ndarray, initial_equity: float = 10_000.0) -> dict:
    """
    df must align index-for-index with `probabilities` (same rows used for prediction).
    Strategy: go long when prob > threshold, flat otherwise. (Add short logic yourself if
    your venue/asset supports it cleanly — shorting crypto perps has funding-rate asymmetries
    worth handling separately rather than copy-pasting the long logic.)
    """
    assert len(df) == len(probabilities), "df and probabilities must be the same length"

    equity = initial_equity
    equity_curve = []
    trade_log = []
    guard = risk_engine.DrawdownGuard()

    position = 0.0
    entry_price = None

    closes = df["close"].values
    atrs = df["atr"].values

    for i in range(len(df) - 1):
        price = closes[i]
        next_price = closes[i + 1]
        prob = probabilities[i]

        # Use ATR as the stop distance -> reward:risk ~ MIN_REWARD_RISK_RATIO by construction
        stop_distance = atrs[i] if not np.isnan(atrs[i]) else price * 0.01
        stop_price = price - stop_distance
        reward_risk = config.MIN_REWARD_RISK_RATIO  # simplification: target = RR * stop_distance

        trading_allowed = guard.update(equity)

        if position == 0.0 and trading_allowed:
            size = risk_engine.position_size(equity, prob, reward_risk, price, stop_price)
            if size > 0:
                cost = price * size * (config.TAKER_FEE + config.SLIPPAGE_BPS / 10_000)
                equity -= cost
                position = size
                entry_price = price
                trade_log.append({"i": i, "action": "buy", "price": price, "size": size, "cost": cost})

        elif position > 0.0:
            pnl = position * (next_price - price)
            funding_cost = position * price * config.FUNDING_RATE_8H / 2  # rough per-candle accrual
            equity += pnl - funding_cost

            # exit if probability flips bearish, or stop is hit
            hit_stop = next_price <= stop_price
            flip = prob < 0.5
            if hit_stop or flip:
                exit_cost = next_price * position * (config.TAKER_FEE + config.SLIPPAGE_BPS / 10_000)
                equity -= exit_cost
                trade_log.append({"i": i, "action": "sell", "price": next_price,
                                   "size": position, "cost": exit_cost})
                position = 0.0
                entry_price = None

        equity_curve.append(equity)

    return {
        "equity_curve": pd.Series(equity_curve, index=df.index[: len(equity_curve)]),
        "trade_log": pd.DataFrame(trade_log),
        "final_equity": equity,
        "halted_on_drawdown": guard.halted,
    }


def performance_metrics(equity_curve: pd.Series, periods_per_year: int = 365 * 6) -> dict:
    """periods_per_year default assumes 4h candles (~6/day). Adjust if you change TIMEFRAME."""
    returns = equity_curve.pct_change().dropna()
    if len(returns) == 0 or returns.std() == 0:
        return {"sharpe": float("nan"), "sortino": float("nan"), "max_drawdown": float("nan")}

    sharpe = (returns.mean() / returns.std()) * np.sqrt(periods_per_year)

    downside = returns[returns < 0]
    sortino = (
        (returns.mean() / downside.std()) * np.sqrt(periods_per_year)
        if len(downside) > 0 and downside.std() != 0 else float("nan")
    )

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "max_drawdown_pct": round(float(max_drawdown) * 100, 2),
        "total_return_pct": round((equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100, 2),
    }
