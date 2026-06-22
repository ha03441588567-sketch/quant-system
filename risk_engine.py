"""
Risk engine: turns a model probability into a position size, and enforces survival rules.
This is the part that matters more than the model — most blown trading accounts die from
position sizing mistakes, not from a bad signal.
"""

import config


def kelly_fraction(win_prob: float, reward_risk_ratio: float) -> float:
    """
    Kelly criterion: f* = p - (1-p)/b   where b = reward/risk ratio.
    Returns the fraction of capital to risk. We cap it hard — full Kelly is too volatile
    for live trading with imperfect probability estimates (your model's 0.65 is not a true 0.65).
    """
    p = win_prob
    b = reward_risk_ratio
    if b <= 0:
        return 0.0

    f = p - (1 - p) / b
    f = max(0.0, f)  # never go negative (would mean "don't take this trade")
    return min(f * config.KELLY_FRACTION_CAP, config.MAX_RISK_PER_TRADE)


def should_take_trade(win_prob: float, reward_risk_ratio: float) -> bool:
    if win_prob < config.PROBABILITY_THRESHOLD:
        return False
    if reward_risk_ratio < config.MIN_REWARD_RISK_RATIO:
        return False
    return True


class DrawdownGuard:
    """Halts new trades once equity drawdown from peak exceeds the configured threshold.
    This single rule prevents the classic 'model degrades silently, account bleeds out' failure."""

    def __init__(self):
        self.peak_equity = None
        self.halted = False

    def update(self, current_equity: float) -> bool:
        if self.peak_equity is None or current_equity > self.peak_equity:
            self.peak_equity = current_equity

        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        if drawdown >= config.MAX_DRAWDOWN_HALT:
            self.halted = True

        return not self.halted

    def reset(self):
        """Call manually after you've reviewed why drawdown hit the limit — don't auto-reset."""
        self.halted = False
        self.peak_equity = None


def position_size(equity: float, win_prob: float, reward_risk_ratio: float,
                   entry_price: float, stop_price: float) -> float:
    """Returns position size in base-asset units, given account equity and a stop-loss level."""
    if not should_take_trade(win_prob, reward_risk_ratio):
        return 0.0

    risk_fraction = kelly_fraction(win_prob, reward_risk_ratio)
    risk_capital = equity * risk_fraction

    price_risk_per_unit = abs(entry_price - stop_price)
    if price_risk_per_unit == 0:
        return 0.0

    return risk_capital / price_risk_per_unit
