"""
Single-run executor for GitHub Actions scheduled triggers — no 24/7 server, no card, no account
beyond a free GitHub account. Each invocation: loads the saved model, checks the exchange directly
for current position/balance (exchange is the source of truth, no local state needed for that),
decides, exits. Runs again on the next schedule tick.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import config
import data_pipeline
import model_ensemble
import risk_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("github-executor")

DRY_RUN = os.environ.get("DRY_RUN", "true").lower() != "false"
MODEL_PATH = "trained_model.joblib"
STATE_PATH = Path("state.json")
MAX_DAILY_LOSS_PCT = 0.05


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"day": None, "start_equity": None, "halted_today": False}


def save_state(state: dict):
    STATE_PATH.write_text(json.dumps(state, indent=2))


def build_exchange():
    import ccxt
    api_key = os.environ.get("EXCHANGE_API_KEY")
    api_secret = os.environ.get("EXCHANGE_API_SECRET")
    if not DRY_RUN and (not api_key or not api_secret):
        log.error("DRY_RUN is False but no API credentials found (set them as GitHub Secrets). Exiting.")
        sys.exit(1)
    exchange_class = getattr(ccxt, config.EXCHANGE)
    return exchange_class({"apiKey": api_key, "secret": api_secret, "enableRateLimit": True})


def get_equity(exchange) -> float:
    if DRY_RUN:
        return 10_000.0  # fixed placeholder, dry-run has no real balance to read
    return float(exchange.fetch_balance()["total"].get("USDT", 0.0))


def get_open_position(exchange) -> float:
    if DRY_RUN:
        return 0.0  # dry-run is stateless by design — always evaluates as if flat
    base = config.SYMBOL.split("/")[0]
    return float(exchange.fetch_balance()["free"].get(base, 0.0))


def main():
    if not os.path.exists(MODEL_PATH):
        log.error(f"{MODEL_PATH} not found in repo. Run train_and_save.py locally once and "
                  f"commit the resulting file.")
        sys.exit(1)

    model = model_ensemble.EnsembleModel.load(MODEL_PATH)
    exchange = build_exchange()

    state = load_state()
    today = datetime.now(timezone.utc).date().isoformat()
    equity = get_equity(exchange)

    if state["day"] != today:
        state = {"day": today, "start_equity": equity, "halted_today": False}

    loss_pct = (state["start_equity"] - equity) / state["start_equity"] if state["start_equity"] else 0.0
    if loss_pct >= MAX_DAILY_LOSS_PCT:
        state["halted_today"] = True
    save_state(state)

    if state["halted_today"]:
        log.warning(f"Daily loss limit hit ({loss_pct:.2%}). Skipping this run.")
        return

    raw = data_pipeline.fetch_live_ohlcv(limit=300)
    df = data_pipeline.add_technical_features(raw)
    latest = df.iloc[[-1]][data_pipeline.FEATURE_COLUMNS]
    prob = float(model.predict_proba(latest)[0])
    price = float(df["close"].iloc[-1])
    atr = float(df["atr"].iloc[-1])
    position = get_open_position(exchange)

    log.info(f"price={price:.2f} prob={prob:.3f} equity={equity:.2f} position={position}")

    if position == 0 and risk_engine.should_take_trade(prob, config.MIN_REWARD_RISK_RATIO):
        stop_price = price - atr
        size = risk_engine.position_size(equity, prob, config.MIN_REWARD_RISK_RATIO, price, stop_price)
        if size > 0:
            if DRY_RUN:
                log.info(f"[DRY RUN] Would BUY {size:.6f} {config.SYMBOL} at ~{price:.2f}")
            else:
                order = exchange.create_market_buy_order(config.SYMBOL, size)
                log.info(f"[LIVE] Buy order placed: {order}")
    elif position > 0 and prob < 0.5:
        if DRY_RUN:
            log.info(f"[DRY RUN] Would SELL position at ~{price:.2f}")
        else:
            order = exchange.create_market_sell_order(config.SYMBOL, position)
            log.info(f"[LIVE] Sell order placed: {order}")
    else:
        log.info("No action this run.")


if __name__ == "__main__":
    main()
