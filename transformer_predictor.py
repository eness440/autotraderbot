"""
transformer_predictor.py
-----------------------

This module provides predictor classes for time-series forecasting.  It
includes a simple moving-average baseline and a transformer-based predictor
which loads a pre-trained transformer model from ``ml/transformer_model.py``.

The goal is to abstract away the details of model loading and prediction so
that the trading bot can easily switch between a naive predictor and a
sophisticated model without changing its own logic.  When a trained
transformer model is not available, the predictor falls back to the
``SimpleTransformerPredictor``, which performs a moving-average forecast.

Example usage::

    from transformer_predictor import TransformerPricePredictor
    predictor = TransformerPricePredictor(model_path="models/price_transformer.pt", context_length=30)
    next_value = predictor.predict(recent_prices)

Note that input series should be normalised in the same way as during
training (see ``ml/transformer_model.py``) for best performance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

try:
    from ml.transformer_model import load_transformer_model
except Exception as exc:  # pragma: no cover
    load_transformer_model = None  # type: ignore

logger = logging.getLogger(__name__)


class SimpleTransformerPredictor:
    """Fallback predictor that uses a simple moving average to forecast the next value."""

    def __init__(self, window: int = 5) -> None:
        self.window = window

    def predict(self, series: Iterable[float]) -> float:
        seq = list(series)
        if len(seq) < self.window:
            return seq[-1] if seq else 0.0
        return sum(seq[-self.window :]) / self.window


class TransformerPricePredictor:
    """
    Wrapper for a transformer-based price predictor.

    Attempts to load a model from ``model_path``; if unsuccessful, uses
    ``SimpleTransformerPredictor`` as a fallback.  The context length
    determines how many past samples the model requires.
    """

    def __init__(self, model_path: str | Path, context_length: int = 30, fallback_to_ma: bool = False) -> None:
        self.model_path = Path(model_path)
        self.context_length = context_length
        self._model = None
        self._available = False
        self._disabled_reason = ""
        self._fallback_enabled = bool(fallback_to_ma)
        self._fallback = SimpleTransformerPredictor(window=context_length) if self._fallback_enabled else None
        if load_transformer_model is None:
            self._available = False
            self._disabled_reason = "torch_unavailable"
            logger.warning("Transformer predictor disabled: PyTorch not available")
        else:
            try:
                if self.model_path.exists():
                    self._model = load_transformer_model(self.model_path, context_length)
                    logger.info("Loaded transformer model from %s", self.model_path)
                    self._available = True
                else:
                    self._available = False
                    self._disabled_reason = "model_missing"
                    # Explicit, one-time-ish warning to avoid "silent" fallback.
                    logger.warning("Transformer predictor disabled: model file not found at %s", self.model_path)
            except Exception as exc:  # pragma: no cover
                logger.error("Failed to load transformer model: %s", exc)
                self._model = None
                self._available = False
                self._disabled_reason = "load_failed"

    @property
    def is_available(self) -> bool:
        """True when a real transformer model is loaded and usable."""
        return bool(self._available and self._model is not None)

    def predict(self, series: Iterable[float]) -> float:
        seq = list(series)
        if self._model is None:
            if self._fallback_enabled and self._fallback is not None:
                return self._fallback.predict(seq)
            raise RuntimeError(
                f"Transformer model unavailable ({self._disabled_reason or 'unknown'})."
            )
        try:
            if len(seq) < self.context_length:
                # extend with last value repeated
                seq = (seq + [seq[-1]] * self.context_length)[: self.context_length]
            return self._model.predict(seq)
        except Exception as exc:
            logger.warning("Transformer prediction failed: %s; falling back", exc)
            if self._fallback_enabled and self._fallback is not None:
                return self._fallback.predict(seq)
            raise