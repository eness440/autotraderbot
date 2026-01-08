"""
stop_order_watchdog.py
---------------------------------
This module provides a standalone asynchronous watchdog for monitoring
open positions on OKX and ensuring that each position has an active
stop‑loss order on the exchange.  If a position is found without a
reduce‑only stop market order, the watchdog will create one using a
conservative fallback price based on the current market price and a
configurable buffer percentage.  This helps prevent liquidations
caused by missing stop orders and maintains protective risk controls.

Usage:

    python3 stop_order_watchdog.py

The script will continuously poll the exchange every ``CHECK_INTERVAL``
seconds.  A shorter interval provides faster recovery but increases API
usage.  You can adjust the frequency and stop buffer via the
environment variables ``WATCHDOG_INTERVAL`` and ``WATCHDOG_BUFFER_FACTOR``.
It is recommended to run this watchdog alongside your trading bot
(e.g. using ``screen`` or as a systemd service) to safeguard against
network or logic failures that might leave a position without a
stop‑loss order.
"""

import asyncio
import os
from typing import Optional, List

import ccxt

from logger import get_logger
from settings import OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE, OKX_USE_TESTNET
from safe_order_wrapper import (
    call,
    safe_update_stop_loss,
)

# Configuration
CHECK_INTERVAL = int(os.getenv("WATCHDOG_INTERVAL", "60"))
STOP_BUFFER_FACTOR = float(os.getenv("WATCHDOG_BUFFER_FACTOR", "1.5"))

# Logger
log = get_logger("stop_order_watchdog")


async def _create_exchange() -> ccxt.okx:
    """Instantiate and return a ccxt OKX exchange object."""
    opts = {
        "apiKey": OKX_API_KEY,
        "secret": OKX_API_SECRET,
        "password": OKX_API_PASSPHRASE,
        "enableRateLimit": True,
        "testnet": OKX_USE_TESTNET,
    }
    return ccxt.okx(opts)


def _convert_instid_to_symbol(inst_id: str) -> Optional[str]:
    """Convert an OKX instrument ID (e.g. BTC-USDT-SWAP) to bot symbol (BTC/USDT)."""
    try:
        parts = inst_id.split("-")
        if len(parts) >= 3:
            base, quote = parts[0], parts[1]
            return f"{base}/{quote}"
    except Exception:
        pass
    return None


async def _fetch_positions(exchange: ccxt.okx) -> List[dict]:
    """Fetch all positions with rate‑limit handling."""
    try:
        positions = await call(exchange.fetch_positions, label="WD-POS")
        if positions is None:
            return []
        return positions
    except Exception as e:
        log.error(f"[Watchdog] fetch_positions failed: {e}")
        return []


async def _fetch_open_orders(exchange: ccxt.okx, symbol: str) -> List[dict]:
    """Fetch open orders for a given symbol with rate‑limit handling."""
    try:
        orders = await call(exchange.fetch_open_orders, symbol, label=f"WD-ORD-{symbol}")
        return orders or []
    except Exception as e:
        log.error(f"[Watchdog] fetch_open_orders({symbol}) failed: {e}")
        return []


async def _fetch_ticker_last(exchange: ccxt.okx, symbol: str) -> Optional[float]:
    """Fetch the last traded price for a symbol."""
    try:
        ticker = await call(exchange.fetch_ticker, symbol, label=f"WD-TICK-{symbol}")
        if ticker:
            return float(ticker.get("last"))
    except Exception as e:
        log.error(f"[Watchdog] fetch_ticker({symbol}) failed: {e}")
    return None


async def _watch_positions():
    """Main watchdog loop.  Ensures each open position has a stop‑loss order."""
    exchange = await _create_exchange()
    log.info("Stop‑order watchdog started.")
    while True:
        positions = await _fetch_positions(exchange)
        for pos in positions:
            # Determine side and quantity
            qty = 0.0
            side = None
            symbol = None
            try:
                amt = float(pos.get("positionAmt", 0.0))
                if abs(amt) < 1e-8:
                    continue
                qty = abs(amt)
                side = "long" if amt > 0 else "short"
                inst_id = pos.get("symbol") or pos.get("info", {}).get("instId")
                if not inst_id:
                    continue
                symbol = _convert_instid_to_symbol(inst_id)
                if not symbol:
                    continue
            except Exception:
                continue

            # Fetch open orders and check if reduce‑only stop exists
            open_orders = await _fetch_open_orders(exchange, symbol)
            has_stop = False
            for order in open_orders:
                try:
                    info = order.get("info", {})
                    reduce_only = info.get("reduceOnly")
                    sl_px = info.get("stopLossPrice") or info.get("slTriggerPx")
                    if reduce_only and sl_px:
                        has_stop = True
                        break
                except Exception:
                    continue
            if has_stop:
                continue

            # Place fallback stop if none exists
            last_price = await _fetch_ticker_last(exchange, symbol)
            if last_price is None:
                continue
            try:
                from main_bot_async import STOP_LOSS_PERCENT as BASE_SL_PCT  # type: ignore
                sl_pct = BASE_SL_PCT * STOP_BUFFER_FACTOR
            except Exception:
                sl_pct = 0.005 * STOP_BUFFER_FACTOR
            if side == "long":
                new_sl = last_price * (1.0 - sl_pct)
            else:
                new_sl = last_price * (1.0 + sl_pct)
            try:
                # dry-run ayarını config.json içindeki trade_parameters.dry_run üzerinden oku.
                dry_flag = False
                try:
                    from pathlib import Path
                    import json
                    cfg_path = Path(__file__).resolve().parent / "config.json"
                    if cfg_path.exists():
                        cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
                        params = cfg_data.get("trade_parameters") or {}
                        if params.get("dry_run") is not None:
                            dry_flag = bool(params.get("dry_run"))
                except Exception:
                    dry_flag = False
                await safe_update_stop_loss(exchange, symbol, new_sl, dry_run=dry_flag)
                log.warning(
                    f"[Watchdog] Added stop‑loss for {symbol} (side={side}) at {new_sl:.4f}"
                )
            except Exception as e:
                log.error(f"[Watchdog] safe_update_stop_loss failed for {symbol}: {e}")
                continue
        await asyncio.sleep(CHECK_INTERVAL)


def run_watchdog() -> None:
    """Run the asynchronous watchdog loop."""
    try:
        asyncio.run(_watch_positions())
    except KeyboardInterrupt:
        log.info("Stop‑order watchdog stopped by user.")


if __name__ == "__main__":
    run_watchdog()