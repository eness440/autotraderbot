# -*- coding: utf-8 -*-
"""
metrics_manager.py
===================

This module centralizes runtime metrics collection for the trading bot.

The purpose of collecting metrics is twofold:

1. **Visibility into system health**: Latency, error rates and retry
   counts help identify connectivity problems or API degradation.  When
   non‑rate‑limit errors spike, the circuit breaker can halt new
   positions automatically.  According to best practices for REST
   integrations, clients should track call attempts and retries and
   implement idempotency and backoff mechanisms to ensure reliability
   under load【698560911221261†L34-L47】.  Recording latency and error
   counts per call surfaces when the exchange or network is unstable.

2. **Performance analytics**: Realized and unrealized PnL,
   portfolio exposure and expectancy (average return per trade) offer
   real‑time feedback on strategy effectiveness.  Expectancy, defined
   as the mean win/loss per trade, is preferred over simple win rate as
   it accounts for the magnitude of profits and losses.  By monitoring
   expectancy and exposure the bot can raise alarms when risk or
   drawdowns exceed configured thresholds.

Metrics are persisted to ``metrics/metrics.json`` under the project
directory.  The structure is a simple dictionary; callers should
update values via the functions below rather than writing directly.

Most update functions are synchronous except ``update_from_exchange``
which is async because it fetches positions from the exchange.

"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

# Atomic write helper – use a fallback if import fails (e.g. during tests)
try:
    from .atomic_io import safe_write_json  # type: ignore
except Exception:
    def safe_write_json(path: Path, data: dict) -> None:  # type: ignore
        try:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass

from datetime import datetime

try:
    # get_daily_realized_pnl is used to compute realized PnL across the day
    from .trade_logger import get_daily_realized_pnl  # type: ignore
except Exception:
    # If import fails (e.g. during unit tests), define a no‑op fallback
    def get_daily_realized_pnl() -> float:  # type: ignore
        return 0.0


# Base directory (project root)
ROOT_DIR = Path(__file__).resolve().parent

# Metrics directory.  All metrics JSON files are stored here.
METRICS_DIR = ROOT_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Primary metrics file path
METRICS_FILE = METRICS_DIR / "metrics.json"

# In‑memory metrics state.  Keys are explained in comments below.
_metrics: Dict[str, float | int | dict | list] = {
    # API call count (successful and failed attempts)
    "api_calls": 0,
    # Number of calls that raised exceptions (non‑rate‑limit errors)
    "api_errors": 0,
    # Number of times rate‑limit retry logic triggered (i.e. attempts > 1)
    "api_retries": 0,
    # Cumulative latency of all API calls in seconds
    "total_latency": 0.0,
    # Last computed realized PnL for the day (USDT)
    "realized_pnl": 0.0,
    # Current unrealized PnL across open positions (USDT)
    "unrealized_pnl": 0.0,
    # Total notional exposure of open positions (absolute sum of size*price)
    "exposure_usd": 0.0,
    # Expectancy (average return per trade).  Expressed as a fraction (e.g. 0.02 = 2%).
    "expectancy": 0.0,
    # Number of closed trades considered when computing expectancy
    "trade_count": 0,
    # Timestamp of last metrics update (ISO string)
    "last_updated": None,
}


def _write_metrics() -> None:
    """
    Persist the in‑memory metrics dictionary to disk.  Uses an atomic
    write helper to avoid race conditions when multiple threads or
    processes update metrics concurrently.  Any exceptions are
    suppressed so metrics recording never propagates errors.
    """
    try:
        data = dict(_metrics)
        # Convert any non‑serialisable values (e.g. datetime) to strings
        if data.get("last_updated") is not None and not isinstance(data["last_updated"], str):
            data["last_updated"] = str(data["last_updated"])
        safe_write_json(METRICS_FILE, data)
    except Exception:
        pass


def record_api_call(latency: float, retries: int) -> None:
    """
    Record an API call with its latency and retry count.

    Args:
        latency: Wall clock time in seconds consumed by the call.
        retries: Number of retry attempts due to rate limiting.  A value of
            0 indicates no retries (i.e. success on first try).
    """
    try:
        _metrics["api_calls"] = int(_metrics.get("api_calls", 0)) + 1
        _metrics["api_retries"] = int(_metrics.get("api_retries", 0)) + int(retries)
        _metrics["total_latency"] = float(_metrics.get("total_latency", 0.0)) + float(latency)
        _metrics["last_updated"] = datetime.utcnow().isoformat()
        _write_metrics()
    except Exception:
        # Fail silently; metrics are non‑critical
        pass


def record_api_error() -> None:
    """
    Increment the count of API errors.  Should be called when a non‑rate‑limit
    exception is raised during an exchange call.
    """
    try:
        _metrics["api_errors"] = int(_metrics.get("api_errors", 0)) + 1
        _metrics["last_updated"] = datetime.utcnow().isoformat()
        _write_metrics()
    except Exception:
        pass


async def update_from_exchange(exchange) -> None:
    """
    Update metrics derived from exchange state: unrealized PnL and exposure.

    This function fetches positions asynchronously from the provided
    ``exchange`` (assumed to be a ccxt exchange instance).  For each
    open position, we compute the unrealized PnL and notional exposure
    using the mark price.  Realized PnL is pulled from the trade log via
    ``get_daily_realized_pnl``.  Finally, expectancy is updated by
    examining the trade_log.json file.

    This function is designed to be called periodically (e.g. at the
    beginning of each trading loop).
    """
    exposure = 0.0
    upnl = 0.0
    try:
        # Fetch positions (ccxt position object list).  This call may
        # throw; call within try block.
        positions = await exchange.fetch_positions()
        for p in positions or []:
            try:
                raw_sym = p.get("symbol") or ""
                # symbol may contain colon; take portion before colon
                sym = raw_sym.split(":")[0] if raw_sym else raw_sym
                # Determine size; OKX returns 'contracts' or 'size'
                size_val = p.get("contracts") or p.get("size") or p.get("positionAmt")
                mark_price = p.get("markPrice") or p.get("last") or p.get("info", {}).get("markPrice")
                entry = p.get("entryPrice") or 0.0
                side = None
                try:
                    size_f = float(size_val)
                    if abs(size_f) < 1e-12:
                        continue
                    # Determine side from sign of size
                    side = "long" if size_f > 0 else "short"
                    size_abs = abs(size_f)
                except Exception:
                    continue
                try:
                    mp = float(mark_price)
                except Exception:
                    mp = None
                try:
                    ep = float(entry)
                except Exception:
                    ep = None
                if mp is None or ep is None:
                    # If price info is missing, skip PnL calculation but include exposure if mark available
                    if mp is not None:
                        exposure += size_abs * mp
                    continue
                # Notional exposure
                exposure += size_abs * mp
                # Unrealized PnL for this position
                if side == "long":
                    upnl += (mp - ep) * size_abs
                else:
                    upnl += (ep - mp) * size_abs
            except Exception:
                continue
    except Exception:
        # If positions fetch fails, leave exposure/unrealized unchanged
        pass

    # Update metrics
    try:
        _metrics["unrealized_pnl"] = round(float(upnl), 8)
        _metrics["exposure_usd"] = round(float(exposure), 8)
    except Exception:
        pass

    # Realized PnL from trade_log
    try:
        realized = float(get_daily_realized_pnl())
        _metrics["realized_pnl"] = round(realized, 8)
    except Exception:
        pass

    # Expectancy: mean of pnl_frac from trade_log.json
    try:
        from pathlib import Path as _Path
        import json as _json
        tlog = _Path(__file__).resolve().parent / "trade_log.json"
        if tlog.exists():
            txt = tlog.read_text(encoding="utf-8").strip()
            if txt:
                rows = _json.loads(txt)
                # Support both list and {'rows': [...]}
                if isinstance(rows, dict):
                    rows = rows.get("rows", [])
                total_frac = 0.0
                count = 0
                for row in rows:
                    pnl_pct = row.get("pnl_pct")
                    if pnl_pct is None:
                        continue
                    try:
                        pnl_frac = float(pnl_pct) / 100.0
                        total_frac += pnl_frac
                        count += 1
                    except Exception:
                        continue
                if count > 0:
                    _metrics["expectancy"] = total_frac / count
                    _metrics["trade_count"] = count
    except Exception:
        pass

    _metrics["last_updated"] = datetime.utcnow().isoformat()
    _write_metrics()


def get_error_rate() -> float:
    """Return the proportion of API calls that resulted in errors."""
    calls = max(1, int(_metrics.get("api_calls", 1)))
    errs = int(_metrics.get("api_errors", 0))
    return errs / calls


def get_average_latency() -> float:
    """Return the average latency per API call in seconds."""
    calls = max(1, int(_metrics.get("api_calls", 1)))
    total = float(_metrics.get("total_latency", 0.0))
    return total / calls


def check_alerts() -> None:
    """
    Inspect current metrics and log warnings when thresholds are breached.

    - Error rate > 10% triggers a warning.
    - Expectancy negative for more than 10 trades triggers a warning.
    - Exposure exceeding 3× current balance triggers a warning.

    This function does not raise exceptions; it only logs messages.
    """
    try:
        from logger import get_logger  # type: ignore
        log = get_logger("metrics_manager")
    except Exception:
        # If logger cannot be imported, fall back to print
        import builtins as _builtins  # type: ignore
        def log_warning(msg: str) -> None: _builtins.print(msg)
        class _LogObj:
            def warning(self, msg: str, *args: object, **kwargs: object) -> None:
                log_warning(str(msg))
        log = _LogObj()

    try:
        error_rate = get_error_rate()
        if error_rate > 0.10 and _metrics.get("api_calls", 0) > 20:
            log.warning(f"[ALARM] API error rate high: {error_rate:.2%} (>{0.10:.2%})")
    except Exception:
        pass

    try:
        exp = float(_metrics.get("expectancy", 0.0))
        count = int(_metrics.get("trade_count", 0))
        if count >= 10 and exp < 0.0:
            log.warning(f"[ALARM] Negative expectancy detected: {exp:.4f} over {count} trades")
    except Exception:
        pass

    try:
        # Exposure to balance ratio; requires CURRENT_BALANCE global from main_bot_async.
        from .main_bot_async import CURRENT_BALANCE  # type: ignore
        balance = float(CURRENT_BALANCE)
        exposure = float(_metrics.get("exposure_usd", 0.0))
        if balance > 0 and exposure > 3.0 * balance:
            log.warning(
                f"[ALARM] Exposure high: ${exposure:.2f} ≈ {exposure/balance:.1f}× account equity"
            )
    except Exception:
        pass
