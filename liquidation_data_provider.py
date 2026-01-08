"""
liquidation_data_provider.py
---------------------------

Liquidation events occur when leveraged positions are forcibly closed and
often precede heightened volatility.  Monitoring large liquidation
clusters can provide early warning signals for rapid price movements.

This module offers a simple interface to retrieve recent liquidation data
from derivatives exchanges or analytics services.  The primary
function, ``fetch_liquidation_heatmap``, returns a dictionary keyed by
price levels with values representing the cumulative notional size of
liquidations at each level.

Example usage::

    from liquidation_data_provider import fetch_liquidation_heatmap
    heatmap = fetch_liquidation_heatmap("BTC", interval="1h")
    print(heatmap)

Custom implementations can use exchanges' liquidation feeds or
third-party services like Coinalyze.  API keys can be supplied via
environment variables ``LIQUIDATION_API_KEY``.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Any

try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore

# Import retry and caching.  Fallback definitions ensure that missing
# dependencies do not break the module.
try:
    from retry_utils import retry  # type: ignore
except Exception:
    def retry(*args, **kwargs):  # type: ignore
        def decorator(func):  # type: ignore
            return func
        return decorator
try:
    from cache_manager import file_cache  # type: ignore
except Exception:
    def file_cache(cache_name: str, ttl: int = 3600):  # type: ignore
        def decorator(func):  # type: ignore
            return func
        return decorator

logger = logging.getLogger(__name__)


@file_cache("liquidation_heatmap_cache.json", ttl=600)
@retry()
def fetch_liquidation_heatmap(symbol: str = "BTC", interval: str = "1h") -> Dict[str, Any]:
    """Retrieve a price-level liquidation heatmap.

    :param symbol: the underlying asset symbol (e.g. ``BTC``)
    :param interval: the time interval over which to aggregate liquidations (e.g. ``1h``)
    :return: mapping from price levels (as strings) to notional size of liquidations
      at that level.  Returns an empty dict if no data is available or if
      the API key is missing.

    The function is wrapped in ``file_cache`` and ``retry`` decorators to
    minimise API calls and handle transient failures gracefully.
    """
    # Read configuration from environment: base URL and API key (optional)
    api_url = os.getenv("LIQUIDATION_API_URL", "").strip()
    api_key = os.getenv("LIQUIDATION_API_KEY", "").strip()
    # If requests is not available or no API URL is provided, attempt a
    # fallback using the user's Binance credentials.  The Binance
    # futures endpoint ``/fapi/v1/forceOrders`` returns recent forced
    # liquidation orders and requires signing the request.  When
    # ``BINANCE_API_KEY`` and ``BINANCE_SECRET_KEY`` environment
    # variables are set, this fallback aggregates the notional value of
    # those orders.  If no credentials are available or an error
    # occurs, an empty heatmap is returned.
    if (not api_url or not api_url.strip()) or requests is None:
        # Skip fallback entirely if the requests library is unavailable
        if requests is None:
            logger.debug("Liquidation data API and fallback unavailable; returning empty heatmap")
            return {}
        api_key = os.getenv("BINANCE_API_KEY", "").strip()
        # Retrieve the Binance secret key.  Some deployments use
        # ``BINANCE_API_SECRET`` instead of ``BINANCE_SECRET_KEY``.  Fall back
        # to ``BINANCE_API_SECRET`` if the latter is not set.
        secret_key = os.getenv("BINANCE_SECRET_KEY", "").strip()
        if not secret_key:
            secret_key = os.getenv("BINANCE_API_SECRET", "").strip()
        if not api_key or not secret_key:
            logger.debug(
                "No liquidation API configured and Binance credentials missing; returning empty heatmap"
            )
            return {}
        try:
            import hmac
            import hashlib
            import time
            import urllib.parse

            # Prepare request parameters.
            # Binance Futures expects symbols like BTCUSDT (no separator).
            # Upstream callers sometimes pass base assets only (e.g. "BTC").
            # Normalise to a USDT-quoted symbol for forceOrders.
            sym = (symbol or "").replace("/", "").strip().upper()
            if sym and not sym.endswith("USDT"):
                sym = f"{sym}USDT"
            params = {
                "symbol": sym,
                "limit": 50,
                "timestamp": int(time.time() * 1000),
            }
            query = urllib.parse.urlencode(params)
            signature = hmac.new(secret_key.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
            base_url = os.getenv("BINANCE_FAPI_BASE_URL", "https://fapi.binance.com").rstrip("/")
            url = f"{base_url}/fapi/v1/forceOrders?{query}&signature={signature}"
            headers = {"X-MBX-APIKEY": api_key}
            resp = requests.get(url, headers=headers, timeout=10)  # type: ignore
            resp.raise_for_status()
            data = resp.json() or []
            total_notional = 0.0
            # Each forced order may include origQty and avgPrice (or shorthand keys)
            for item in data:
                try:
                    qty = float(item.get("origQty") or item.get("o") or 0.0)
                    price = float(item.get("avgPrice") or item.get("p") or 0.0)
                    total_notional += abs(qty) * price
                except Exception:
                    continue
            # Return simplified heatmap: downstream logic sums the values
            return {"TOTAL": total_notional}
        except Exception as exc:
            logger.warning("Failed to fetch Binance force orders for %s via fallback: %s", symbol, exc)
            return {}
    # Prepare query parameters.  Symbols for many APIs use uppercase without
    # separators (e.g. BTCUSDT).  Consumers can override the API to handle
    # other formats if necessary.
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
    }
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = api_key
    try:
        resp = requests.get(api_url, params=params, headers=headers, timeout=10)  # type: ignore
        resp.raise_for_status()
        data = resp.json()
        heatmap: Dict[str, Any] = {}
        # If the response is a list, assume each item contains a price and notional
        if isinstance(data, list):
            for item in data:
                try:
                    price_val = item.get("price") or item.get("p") or item.get("PX")
                    notional_val = (
                        item.get("notional")
                        or item.get("qty")
                        or item.get("volume")
                        or item.get("size")
                    )
                    if price_val is None or notional_val is None:
                        continue
                    price_str = str(price_val)
                    notional_num = float(notional_val)
                    heatmap[price_str] = heatmap.get(price_str, 0.0) + notional_num
                except Exception:
                    continue
            return heatmap
        # If the response is a dict, it may already map price levels to notional sizes
        if isinstance(data, dict):
            for k, v in data.items():
                try:
                    heatmap[str(k)] = float(v)
                except Exception:
                    continue
            return heatmap
        logger.debug("Unexpected liquidation API response format: %s", type(data))
        return {}
    except Exception as exc:
        logger.warning("Failed to fetch liquidation heatmap for %s: %s", symbol, exc)
        return {}


# ---------------------------------------------------------------------------
# WebSocket streaming for liquidation data

async def start_liquidation_ws(
    exchange: str = "binance",
    symbols: list[str] | None = None,
    update_callback=None,
    reconnect_delay: int = 5,
) -> None:
    """
    Connect to a derivatives exchange's liquidation order stream via
    WebSocket and forward real‑time liquidation events to a callback.

    This helper supports **Binance** and **Bybit** public streams without
    requiring any API keys.  For Binance, you can subscribe to either a
    single symbol (``<symbol>@forceOrder``) or the global liquidation
    stream (``!forceOrder@arr``).  For Bybit, the ``allLiquidation.<symbol>``
    topic is used.  The function runs indefinitely until cancelled and
    automatically reconnects on errors using exponential backoff.

    Parameters
    ----------
    exchange : str, optional
        The exchange name: ``"binance"`` or ``"bybit"``.  Defaults to
        ``"binance"``.
    symbols : list of str, optional
        List of symbols to subscribe to (e.g. ["BTCUSDT", "ETHUSDT"]).
        If omitted or empty for Binance, the global stream is used.  For
        Bybit a non‑empty list is required.
    update_callback : callable, optional
        A callback function or coroutine that accepts a single event
        dictionary.  Each event contains ``exchange``, ``symbol``,
        ``side`` (``Buy``/``Sell``), ``quantity``, ``price``, ``notional``
        and ``timestamp``.  If the callback is a coroutine it will be
        awaited; otherwise it will be called synchronously.
    reconnect_delay : int, optional
        Initial delay in seconds before attempting to reconnect after an
        error.  The delay doubles after each failure up to 60 seconds.

    Notes
    -----
    The ``websockets`` library must be installed.  If it is missing,
    the function logs an error and returns immediately.  Real‑time
    liquidation feeds can generate high event rates; consider
    aggregation, throttling or filtering in the callback to avoid
    overwhelming your application.
    """
    try:
        import asyncio
        import json
        import websockets  # type: ignore
    except Exception:
        logger.error("websockets library is not available; cannot start liquidation stream")
        return
    ex = (exchange or "").lower()
    if ex not in {"binance", "bybit"}:
        logger.error("Unsupported exchange for liquidation WS: %s", exchange)
        return
    # Normalise symbols list
    symbol_list = symbols or []
    async def _handle_event(event: Dict[str, Any]) -> None:
        if update_callback is None:
            return
        try:
            if asyncio.iscoroutinefunction(update_callback):
                await update_callback(event)
            else:
                update_callback(event)
        except Exception as exc:
            logger.debug("Liquidation callback error: %s", exc)
    async def _connect_and_listen_binance():
        # Determine the appropriate WebSocket URL for Binance
        if symbol_list:
            streams = "/".join([f"{s.lower()}@forceOrder" for s in symbol_list])
            url = f"wss://fstream.binance.com/stream?streams={streams}"
        else:
            # All market liquidation snapshot
            url = "wss://fstream.binance.com/ws/!forceOrder@arr"
        # Disable ping keepalive tasks to prevent lingering asyncio tasks on shutdown
        async with websockets.connect(url, ping_interval=None) as ws:
            async for msg in ws:
                try:
                    data = json.loads(msg)
                except Exception:
                    continue
                # Multi‑stream messages have 'data' field
                payload = data.get("data") if isinstance(data, dict) else data
                if not isinstance(payload, dict):
                    continue
                ev = payload.get("o")
                if not isinstance(ev, dict):
                    continue
                try:
                    symbol = ev.get("s")
                    side = ev.get("S")
                    qty = float(ev.get("q") or ev.get("l") or 0.0)
                    price = float(ev.get("ap") or ev.get("p") or 0.0)
                    notional = qty * price
                    ts = int(ev.get("T") or payload.get("E") or 0)
                    event = {
                        "exchange": "binance",
                        "symbol": symbol,
                        "side": side,
                        "quantity": qty,
                        "price": price,
                        "notional": notional,
                        "timestamp": ts,
                    }
                    await _handle_event(event)
                except Exception:
                    continue
    async def _connect_and_listen_bybit():
        if not symbol_list:
            logger.error("Bybit liquidation stream requires a non‑empty symbol list")
            return
        url = "wss://stream.bybit.com/v5/public/linear"
        # Disable ping keepalive tasks to prevent lingering asyncio tasks on shutdown
        async with websockets.connect(url, ping_interval=None) as ws:
            # Subscribe to each topic
            topics = [f"allLiquidation.{s.upper()}" for s in symbol_list]
            sub_msg = {"op": "subscribe", "args": topics}
            await ws.send(json.dumps(sub_msg))
            async for msg in ws:
                try:
                    data = json.loads(msg)
                except Exception:
                    continue
                if not isinstance(data, dict):
                    continue
                if data.get("type") != "snapshot":
                    continue
                ts_ms = int(data.get("ts") or 0)
                for item in data.get("data", []) or []:
                    try:
                        symbol = item.get("s")
                        side = item.get("S")
                        qty = float(item.get("v") or 0.0)
                        price = float(item.get("p") or 0.0)
                        notional = qty * price
                        ts_event = int(item.get("T") or ts_ms)
                        event = {
                            "exchange": "bybit",
                            "symbol": symbol,
                            "side": side,
                            "quantity": qty,
                            "price": price,
                            "notional": notional,
                            "timestamp": ts_event,
                        }
                        await _handle_event(event)
                    except Exception:
                        continue
    # Main reconnect loop
    delay = reconnect_delay
    while True:
        try:
            if ex == "binance":
                await _connect_and_listen_binance()
            else:
                await _connect_and_listen_bybit()
            return  # exit if websocket closes normally
        except Exception as exc:
            logger.warning("Liquidation WS error (%s): %s; reconnecting in %ds", ex, exc, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)