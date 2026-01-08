"""
backtester.py
-------------

Simple backtesting utility for the AutoTraderBot.  It implements a
moving average cross strategy on historical OHLC data stored in
``metrics/ohlc_history.json``.  You can specify lists of ``fast`` and
``slow`` window lengths to sweep over and evaluate multiple parameter
combinations.  Results are written to ``metrics/backtest_results.json``.

This module is not intended to be a comprehensive backtesting
framework, but rather a lightweight tool to approximate strategy
performance and compute common metrics such as Sharpe ratio, max
drawdown and win rate.  For more advanced analysis consider using
dedicated backtesting libraries.

Usage::

    python backtester.py --fast 5 10 --slow 20 40

If no arguments are provided, defaults are used (fast=[5,10], slow=[20,30]).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Any


def load_ohlc(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load OHLC records from the given JSON file and group them by symbol.
    The expected format is ``{"rows": [{"symbol": "BTCUSDT", "ts": ..., ...}, ...]}``.

    Args:
        path: Path to the ohlc_history.json file.
    Returns:
        Dictionary mapping symbol to list of rows sorted by timestamp.
    """
    if not path.exists():
        raise FileNotFoundError(f"OHLC history file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    if isinstance(data, dict) and "rows" in data:
        rows = data["rows"]
    elif isinstance(data, list):
        rows = data
    symbols: Dict[str, List[Dict[str, Any]]] = {}
    for rec in rows:
        sym = rec.get("symbol") or rec.get("symbol_pair")
        if not sym:
            continue
        # unify symbol name (e.g. 'BTC/USDT' -> 'BTCUSDT')
        sym = sym.replace("/", "")
        symbols.setdefault(sym, []).append(rec)
    # sort each list by timestamp
    for sym, lst in symbols.items():
        lst.sort(key=lambda r: r.get("ts"))
    return symbols


def compute_strategy(records: List[Dict[str, Any]], fast: int, slow: int) -> Tuple[List[float], int, int]:
    """
    Run a simple moving average cross strategy on the given records.  We
    assume a long‑only strategy: enter long when fast MA crosses above
    slow MA; exit when it crosses below.  Uses closing prices.  Returns
    a list of fractional returns per trade along with counts of total
    trades and winning trades.

    Args:
        records: Sorted list of OHLC rows for a single symbol.
        fast: Window length for the fast moving average.
        slow: Window length for the slow moving average (must be > fast).
    Returns:
        (list of returns, total trades, winning trades)
    """
    if slow <= fast:
        raise ValueError("slow window must be greater than fast window")
    closes: List[float] = [float(r["close"]) for r in records if r.get("close") is not None]
    if len(closes) < slow + 2:
        return ([], 0, 0)
    position_open = False
    entry_price: float = 0.0
    returns: List[float] = []
    wins = 0
    for idx in range(slow, len(closes)):
        # compute moving averages over the previous N closes
        fast_ma = sum(closes[idx - fast:idx]) / fast
        slow_ma = sum(closes[idx - slow:idx]) / slow
        price = closes[idx]
        if not position_open and fast_ma > slow_ma:
            # open long position
            position_open = True
            entry_price = price
        elif position_open and fast_ma < slow_ma:
            # close position
            ret = (price - entry_price) / entry_price
            returns.append(ret)
            if ret > 0:
                wins += 1
            position_open = False
    # Close any open position at end of dataset
    if position_open:
        price = closes[-1]
        ret = (price - entry_price) / entry_price
        returns.append(ret)
        if ret > 0:
            wins += 1
    return (returns, len(returns), wins)


def performance_metrics(returns: List[float]) -> Dict[str, float]:
    """
    Compute performance metrics for a list of trade returns.

    Metrics include:
      - sharpe_ratio: Mean return divided by standard deviation multiplied by sqrt of count.
      - max_drawdown: Maximum peak‑to‑trough decline in cumulative returns.
      - win_rate: Fraction of trades with positive return.
    """
    n = len(returns)
    if n == 0:
        return {"sharpe_ratio": 0.0, "max_drawdown": 0.0, "win_rate": 0.0}
    mean_ret = statistics.mean(returns)
    std_ret = statistics.pstdev(returns) if n > 1 else 0.0
    sharpe = (mean_ret / std_ret * math.sqrt(n)) if std_ret > 0 else 0.0
    # Equity curve
    cum = [0.0]
    total = 0.0
    for r in returns:
        total += r
        cum.append(total)
    # Max drawdown
    peak = float('-inf')
    max_dd = 0.0
    for v in cum:
        if v > peak:
            peak = v
        drawdown = peak - v
        if drawdown > max_dd:
            max_dd = drawdown
    win_rate = sum(1 for r in returns if r > 0) / float(n)
    return {"sharpe_ratio": sharpe, "max_drawdown": max_dd, "win_rate": win_rate}


def run_backtest(ohlc_path: Path, fast_windows: List[int], slow_windows: List[int]) -> Dict[str, Any]:
    """
    Perform a parameter sweep backtest across provided window lengths.  For
    each symbol in the dataset and each valid (fast, slow) pair (fast < slow),
    compute the strategy returns and performance metrics.  Results are
    aggregated into a nested dictionary keyed by symbol and parameter
    combination.
    """
    symbols = load_ohlc(ohlc_path)
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for sym, records in symbols.items():
        sym_results: Dict[str, Dict[str, float]] = {}
        for fast in fast_windows:
            for slow in slow_windows:
                if slow <= fast:
                    continue
                key = f"fast{fast}_slow{slow}"
                returns, trades, wins = compute_strategy(records, fast, slow)
                perf = performance_metrics(returns)
                perf["trades"] = trades
                perf["wins"] = wins
                sym_results[key] = perf
        results[sym] = sym_results
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest moving average strategy.")
    parser.add_argument("--ohlc", type=str, default="metrics/ohlc_history.json", help="Path to OHLC history JSON")
    parser.add_argument("--fast", type=int, nargs="*", default=[5, 10], help="Fast MA window lengths (space separated)")
    parser.add_argument("--slow", type=int, nargs="*", default=[20, 30], help="Slow MA window lengths (space separated)")
    parser.add_argument("--output", type=str, default="metrics/backtest_results.json", help="Output JSON for results")
    args = parser.parse_args()
    ohlc_path = Path(args.ohlc)
    fast_windows = args.fast
    slow_windows = args.slow
    try:
        results = run_backtest(ohlc_path, fast_windows, slow_windows)
    except Exception as e:
        print(f"[backtester] Hata: {e}")
        return
    out_path = Path(args.output)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[backtester] Sonuçlar {out_path} dosyasına yazıldı.")
    except Exception as e:
        print(f"[backtester] Sonuçlar yazılamadı: {e}")


if __name__ == "__main__":
    main()