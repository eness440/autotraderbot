"""
bilstm_inference.py
=====================

Utility functions to load and infer the BiLSTM model trained by
``bilstm_train.py``.  This module caches the model and historical OHLC
data and automatically reloads them when the underlying files change on
disk.  It exposes a single function, ``predict_prob``, which returns
the probability that the next directional move for a given symbol is
upwards.  If the model or data are unavailable, the function returns
``0.5`` (neutral probability).

The BiLSTM architecture defined here mirrors the architecture used in
``bilstm_train.py`` and ``bilstm_predict.py``: a bidirectional GRU
followed by a small feedforward head.  The model is loaded lazily on
first use and reloaded whenever ``models/bilstm_best.pt`` is updated.

This module is standalone and has no side effects; it does not log
predictions or write any files.  Other parts of the system should call
``predict_prob(symbol)`` to obtain a probability in ``[0.0, 1.0]``.
"""

from __future__ import annotations

import json
import pathlib
from typing import Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore

# Locate project root (ml/ is nested one level under repo root)
ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
METRICS = ROOT / "metrics"

# Cache globals for model and data
_bilstm_model: Optional[nn.Module] = None
_bilstm_in_dim: int = 1
_bilstm_window: int = 60
_bilstm_model_mtime: float | None = None
_bilstm_data_mtime: float | None = None
_bilstm_df: Optional[pd.DataFrame] = None


class BiLSTM(nn.Module):
    """Bidirectional GRU model identical to the training architecture."""

    def __init__(self, in_dim: int = 1, hidden: int = 64, layers: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o, _ = self.gru(x)
        last = o[:, -1, :]
        return self.head(last)


def _load_bilstm_resources() -> None:
    """
    Reload the BiLSTM model and OHLC data if their modification times have
    changed since the last call.  Uses globals to cache state.
    """
    global _bilstm_model, _bilstm_in_dim, _bilstm_window
    global _bilstm_model_mtime, _bilstm_data_mtime, _bilstm_df

    # Model reloading
    model_path = MODELS / "bilstm_best.pt"
    try:
        cur_model_mtime = model_path.stat().st_mtime
    except Exception:
        cur_model_mtime = None
    if cur_model_mtime is not None and cur_model_mtime != _bilstm_model_mtime:
        if model_path.exists():
            try:
                ckpt = torch.load(model_path, map_location="cpu")
                in_dim = ckpt.get("in_dim", 1)
                window = ckpt.get("window", 60)
                state = ckpt.get("model") or ckpt
                model = BiLSTM(in_dim=int(in_dim))
                model.load_state_dict(state)
                model.eval()
                _bilstm_model = model
                _bilstm_in_dim = int(in_dim)
                _bilstm_window = int(window)
                _bilstm_model_mtime = cur_model_mtime
            except Exception:
                # On failure, invalidate the cache so callers get neutral
                _bilstm_model = None
                _bilstm_model_mtime = cur_model_mtime
        else:
            # Model file missing: invalidate
            _bilstm_model = None
            _bilstm_model_mtime = cur_model_mtime

    # Data reloading
    data_path = METRICS / "ohlc_history.json"
    try:
        cur_data_mtime = data_path.stat().st_mtime
    except Exception:
        cur_data_mtime = None
    if cur_data_mtime is not None and cur_data_mtime != _bilstm_data_mtime:
        if data_path.exists():
            try:
                raw_txt = data_path.read_text(encoding="utf-8")
                data = json.loads(raw_txt)
                rows = data.get("rows", data) if isinstance(data, dict) else data
                df = pd.DataFrame(rows)
                if not df.empty:
                    df["symbol"] = df["symbol"].astype(str)
                    # Sort by timestamp ascending
                    try:
                        df_sorted = df.sort_values("ts")
                    except Exception:
                        df_sorted = df
                    _bilstm_df = df_sorted
                else:
                    _bilstm_df = None
                _bilstm_data_mtime = cur_data_mtime
            except Exception:
                _bilstm_df = None
                _bilstm_data_mtime = cur_data_mtime
        else:
            _bilstm_df = None
            _bilstm_data_mtime = cur_data_mtime


def predict_prob(symbol: str, window: int | None = None) -> float:
    """
    Compute the probability that the given symbol's price will increase.

    Parameters
    ----------
    symbol: str
        The trading pair identifier (e.g. "BTC/USDT").  Case-insensitive.
    window: int, optional
        Number of past bars to use for the normalised return sequence.  If
        omitted, the window used during training will be employed.

    Returns
    -------
    float
        Probability in the range [0.0, 1.0].  Returns 0.5 if data or model is
        missing or an error occurs during inference.
    """
    # Load or refresh resources on demand
    _load_bilstm_resources()
    model = _bilstm_model
    df = _bilstm_df
    if model is None or df is None:
        return 0.5
    # Normalise symbol to uppercase for comparison
    sym_upper = str(symbol).upper()
    try:
        df_sym = df[df["symbol"].str.upper() == sym_upper]
    except Exception:
        return 0.5
    # Determine sequence length
    seq_len = _bilstm_window if window is None else int(window)
    if df_sym.empty or len(df_sym) <= seq_len:
        return 0.5
    try:
        closes = df_sym["close"].astype(float).values
        closes = closes[-seq_len:]
        base = closes[0] if closes[0] != 0 else float(np.mean(closes) or 1.0)
        seq = closes / base - 1.0
        x = torch.tensor(seq, dtype=torch.float32).view(1, -1, 1)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            prob = float(probs[0, 1].item())
        # Clamp to [0, 1]
        if prob < 0.0:
            prob = 0.0
        if prob > 1.0:
            prob = 1.0
        return prob
    except Exception:
        return 0.5