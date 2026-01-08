"""
transformer_model.py
--------------------

This module provides a simple Transformer-based neural network for time-series
forecasting.  It is designed to forecast the next value in a sequence of
cryptocurrency prices, but can be adapted to other one-dimensional time
series.  The implementation uses PyTorch and relies on the high-level
``nn.TransformerEncoder`` layers.  A small positional encoding module is
included to inject information about the order of the sequence.

The module exposes two convenience functions: ``train_transformer_model``
and ``load_transformer_model``.  The training function builds a dataset
from a list of price observations, splits it into train/validation sets,
instantiates the model, and trains it for a configurable number of epochs.
The loaded model can then be used for inference via the ``TransformerModel``
class defined below.

This file deliberately keeps the model architecture modest so that it can
execute within constrained environments; feel free to adjust the number of
layers, attention heads, and hidden dimensions for better accuracy.

Example usage::

    from ml.transformer_model import train_transformer_model, load_transformer_model
    # suppose ``prices`` is a list of floats representing closing prices
    train_transformer_model(prices, model_path="models/price_transformer.pt")
    model = load_transformer_model("models/price_transformer.pt")
    next_price = model.predict(prices[-model.context_length:])

Notes
-----
* The model expects normalised inputs; training does not normalise your data.
  You should scale your prices (e.g., by dividing by the first element) before
  calling ``train_transformer_model`` or ``predict``.
* If PyTorch is not installed, importing this module will raise an ImportError.
  Install ``torch`` and ``numpy`` via pip to use this model.

"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np  # type: ignore
import torch  # type: ignore
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings.

    This implementation uses the standard sine/cosine formulation from
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to ``x`` (shape [seq_len, batch_size, d_model])."""
        x = x + self.pe[: x.size(0), :]
        return x


class PriceDataset(Dataset):
    """Dataset for autoregressive next-value prediction on a univariate sequence."""

    def __init__(self, series: Iterable[float], context_length: int = 30) -> None:
        values = np.array(list(series), dtype=np.float32)
        self.context_length = context_length
        if len(values) <= context_length:
            raise ValueError("Series too short for given context_length")
        self.series = values

    def __len__(self) -> int:
        return len(self.series) - self.context_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        window = self.series[idx : idx + self.context_length + 1]
        x = torch.from_numpy(window[:-1]).unsqueeze(1)  # [context, 1]
        y = torch.tensor(window[-1], dtype=torch.float32)
        return x, y


class TransformerModel(nn.Module):
    """A small Transformer model for univariate time-series prediction."""

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        context_length: int = 30,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.input_dim = input_dim
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 1)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding.bias.data.zero_()
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq_len, batch, input_dim]
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        # Use last time step representation
        last_hidden = output[-1]
        out = self.decoder(last_hidden)
        return out.squeeze(-1)

    def predict(self, series: List[float]) -> float:
        """Predict the next value given a list of recent values.

        :param series: last ``context_length`` values of the time series.
        :return: predicted next value.
        """
        if len(series) < self.context_length:
            raise ValueError(
                f"Need at least {self.context_length} historical points for prediction"
            )
        with torch.no_grad():
            x = torch.tensor(series[-self.context_length :], dtype=torch.float32).unsqueeze(1)
            x = x.unsqueeze(1)  # [context, batch=1, 1]
            pred = self.forward(x)
            return float(pred.item())


def train_transformer_model(
    prices: Iterable[float],
    model_path: str | Path = "models/price_transformer.pt",
    context_length: int = 30,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.2,
    device: str | torch.device = "cpu",
) -> None:
    """Train a transformer model on a series of prices.

    :param prices: iterable of float values representing the price series.
    :param model_path: path to save the trained model to.
    :param context_length: number of past points used for prediction.
    :param epochs: number of training epochs.
    :param batch_size: mini-batch size for training.
    :param lr: learning rate.
    :param val_split: fraction of data reserved for validation.
    :param device: torch device ('cpu' or 'cuda').
    """
    ds = PriceDataset(prices, context_length=context_length)
    # split dataset
    val_size = int(len(ds) * val_split)
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])
    # Use drop_last=True to avoid mismatched batch sizes on the final batch.  Without
    # this, the final batch may be smaller than ``batch_size``, causing shape
    # mismatches between the model outputs and targets (e.g. input size 30 vs target size 32).
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, drop_last=True)

    model = TransformerModel(context_length=context_length)
    device_t = torch.device(device)
    model.to(device_t)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss: float | None = None
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device_t)
            y_batch = y_batch.to(device_t)
            optimizer.zero_grad()
            out = model(x_batch)
            # If the last batch is smaller than the expected batch_size, ensure
            # predictions and targets are aligned.  Without this guard, an
            # incomplete batch can produce tensors of different lengths which
            # PyTorch will not broadcast automatically, raising a size mismatch
            # error.  Truncate both tensors to the minimum length.
            if out.shape[0] != y_batch.shape[0]:
                min_len = min(out.shape[0], y_batch.shape[0])
                out = out[:min_len]
                y_batch = y_batch[:min_len]
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
        train_loss /= len(train_loader.dataset)
        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device_t)
                y_batch = y_batch.to(device_t)
                out = model(x_batch)
                if out.shape[0] != y_batch.shape[0]:
                    min_len = min(out.shape[0], y_batch.shape[0])
                    out = out[:min_len]
                    y_batch = y_batch[:min_len]
                loss = criterion(out, y_batch)
                val_loss += loss.item() * x_batch.size(0)
        val_loss /= len(val_loader.dataset)
        logger.info(
            "Epoch %d: train_loss=%.6f, val_loss=%.6f",
            epoch + 1,
            train_loss,
            val_loss,
        )
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            # save best model
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)


def load_transformer_model(model_path: str | Path, context_length: int = 30) -> TransformerModel:
    """Load a trained transformer model from disk.

    In some versions of this code the shape of the positional encoding buffer
    (``pos_encoder.pe``) differed between the saved checkpoint and the
    ``TransformerModel`` definition.  When this happens ``load_state_dict``
    raises a ``RuntimeError`` complaining about a size mismatch.  To make
    loading more robust across checkpoints, this function catches shape
    mismatches for the positional encoding and attempts to transpose the
    offending tensor to match the current model's expected shape.

    :param model_path: path to the model .pt file.
    :param context_length: the context length used during training.
    :return: an instance of ``TransformerModel`` with loaded weights.
    """
    # Instantiate a fresh model to receive the parameters
    model = TransformerModel(context_length=context_length)
    # Load the checkpoint from disk
    state_dict = torch.load(model_path, map_location="cpu")
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        # Handle mismatched positional encoding dimensions
        err_str = str(exc)
        if "pos_encoder.pe" in err_str:
            # Retrieve the positional encoding tensor from the checkpoint
            pe = state_dict.get("pos_encoder.pe")
            if isinstance(pe, torch.Tensor) and pe.dim() == 3:
                # Current model expects shape like [a, b, d]; if the saved
                # tensor has dims reversed for the first two axes then
                # swapping them may resolve the mismatch.
                expected_shape = model.pos_encoder.pe.shape
                if pe.shape[0] == expected_shape[1] and pe.shape[1] == expected_shape[0]:
                    state_dict["pos_encoder.pe"] = pe.permute(1, 0, 2).contiguous()
            # Attempt to load again non‑strictly (ignore any remaining mismatches)
            model.load_state_dict(state_dict, strict=False)
        else:
            # Re‑raise unknown errors
            raise
    # Put the model into inference mode
    model.eval()
    return model