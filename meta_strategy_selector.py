"""
meta_strategy_selector.py
------------------------

This module implements a simple meta-strategy selector for choosing
between different trading strategies (e.g. trend-following vs mean-reversion)
based on current market conditions.  The goal is to improve overall
performance by dynamically selecting the most appropriate strategy.

The selector computes indicators such as momentum and volatility from
recent price data and decides which strategy to activate.  It is
intended to be integrated into the main bot loop or decision layer.

Example usage::

    from meta_strategy_selector import select_strategy
    strategy = select_strategy(price_series, lookback=60)
    if strategy == "trend":
        # run trend-following signals
    elif strategy == "mean_reversion":
        # run mean-reversion signals
"""

from __future__ import annotations

from typing import Iterable, Optional, Any
import numpy as np


def select_strategy(
    prices: Iterable[float],
    lookback: int = 60,
    vol_threshold: float = 0.02,
    transformer_predictor: Optional[Any] = None,
    pred_threshold: float = 0.001,
    breakout_threshold: float = 0.005,
) -> str:
    """Choose a trading strategy based on momentum, volatility and an optional transformer prediction.

    This meta‑selector examines recent price series to decide whether a trend‑following
    or mean‑reversion strategy is more appropriate.  It supports optional integration
    of a transformer predictor (via the ``transformer_predictor`` argument).  When a
    predictor is provided, the function computes the predicted next price and
    compares the implied return against ``pred_threshold``.  A predicted return
    above the threshold yields ``"trend"``, while a return below the negative
    threshold yields ``"mean_reversion"``.  If the prediction is inconclusive or
    unavailable, the decision falls back to a combination of momentum and
    volatility heuristics.

    Args:
        prices: sequence of recent prices (most recent last)
        lookback: number of periods to use for momentum/volatility calculations
        vol_threshold: standard deviation threshold used when momentum is
            insufficient to determine regime
        transformer_predictor: optional object with a ``predict`` method and a
            ``context_length`` attribute.  Should return a float prediction
            when given an iterable of floats.
        pred_threshold: minimum absolute predicted return (fraction) required
            to override momentum/volatility logic.

    Returns:
        A string indicating which strategy to employ.  Possible values
        include ``"trend"``, ``"mean_reversion"`` and ``"breakout"``.
    """
    prices_arr = np.asarray(list(prices), dtype=float)
    if len(prices_arr) < lookback + 1:
        return "mean_reversion"
    # Optional: use transformer prediction
    if transformer_predictor is not None:
        try:
            # Determine how much context the predictor needs
            context_len = getattr(transformer_predictor, "context_length", lookback)
            if context_len > 0:
                seq = prices_arr[-context_len:]
                pred_price = transformer_predictor.predict(seq)
                last_price = seq[-1]
                if last_price != 0:
                    predicted_return = (pred_price - last_price) / abs(last_price)
                    if predicted_return > pred_threshold:
                        return "trend"
                    if predicted_return < -pred_threshold:
                        return "mean_reversion"
        except Exception:
            # If prediction fails, fall back to momentum/volatility
            pass
    # Compute momentum and volatility
    window = prices_arr[-lookback:]
    returns = np.diff(window) / (window[:-1] + 1e-8)
    momentum = returns.sum()
    vol = returns.std()
    # Breakout detection: if the latest price breaks above the recent high
    # or below the recent low by more than ``breakout_threshold`` then
    # classify it as a breakout regime.
    last_price = window[-1]
    high = window.max()
    low = window.min()
    if last_price > high * (1.0 + breakout_threshold):
        return "breakout"
    if last_price < low * (1.0 - breakout_threshold):
        return "breakout"
    # Decision based on relative momentum and volatility
    if abs(momentum) > vol:
        return "trend"
    if vol < vol_threshold:
        return "mean_reversion"
    return "mean_reversion"