"""
ml/bilstm_predict.py
====================

This inference script loads a binary classification BiLSTM model trained
to predict the short‑term direction of a given cryptocurrency pair.  It
takes the most recent closing prices over a specified window, normalises
the returns relative to the first value in the window, and feeds the
sequence into a bidirectional LSTM.  The model outputs a probability
between 0 and 1 indicating whether the price is likely to increase
(``1``) or decrease (`0``) beyond a small change threshold.  If the
probability of an upward move is at least 0.5, the script emits an
``enter`` signal with that confidence; otherwise it returns ``skip``.

Models are saved in the ``models/`` directory under the name
``bilstm_best.pt`` by the companion training script ``bilstm_train.py``.
"""
import json, pathlib, argparse, numpy as np, pandas as pd, torch, torch.nn as nn
from datetime import datetime, timezone

MODELS = pathlib.Path("models")
METRICS = pathlib.Path("metrics")

class BiLSTM(nn.Module):
    """
    GRU-based model used for BiLSTM predictions.  The training script
    (bilstm_train.py) actually uses a bidirectional GRU rather than an LSTM,
    so we replicate that architecture here to ensure compatibility.  This
    model outputs two logits (num_classes=2) corresponding to the negative
    and positive classes.  Hidden size and layer count are kept modest to
    align with the training configuration (hidden=64, layers=1).
    """

    def __init__(self, in_dim: int = 1, hidden: int = 64, layers: int = 1, num_classes: int = 2):
        super().__init__()
        # Use a bidirectional GRU (matches training).  No dropout for a single layer.
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

def last_sequence(symbol="BTC/USDT", window=60):
    f = METRICS/"ohlc_history.json"
    rows = json.loads(f.read_text(encoding="utf-8"))
    df = pd.DataFrame(rows)
    df = df[df["symbol"].astype(str).str.upper()==symbol.upper()].sort_values("ts")
    closes = df["close"].astype(float).values
    if len(closes) < window+1:
        raise SystemExit("Veri yetersiz.")
    seq = closes[-window:]
    seq = seq / seq[0] - 1.0
    return torch.tensor(seq, dtype=torch.float32).view(1, -1, 1), df.iloc[-1]["ts"], float(closes[-1])

def append_prediction(record: dict):
    p = METRICS / "ai_predictions.json"
    data = []
    if p.exists():
        try: data = json.loads(p.read_text(encoding="utf-8"))
        except: data = []
    data.append(record)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--window", type=int, default=60)
    args = ap.parse_args()

    # Load the trained model checkpoint.  In case of errors (e.g. missing
    # file, architecture mismatch), handle gracefully.
    ckpt_path = MODELS / "bilstm_best.pt"
    if not ckpt_path.exists():
        print(f"[BiLSTM-PREDICT] Model bulunamadı: {ckpt_path}")
        return
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        in_dim = ckpt.get("in_dim", 1)
        model = BiLSTM(in_dim=in_dim)
        state = ckpt.get("model") or ckpt
        model.load_state_dict(state)
        model.eval()
    except Exception as e:
        print(f"[BiLSTM-PREDICT] Model yükleme hatası: {e}")
        return

    x, ts, price = last_sequence(args.symbol, args.window)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0,1].item()  # long olasılığı
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "symbol": args.symbol,
        "model": "bilstm",
        "action": "enter" if prob>=0.5 else "skip",
        "confidence": round(float(prob), 4),
        "price": price,
        "note": "BiLSTM long-prob"
    }
    path = append_prediction(rec)
    print("WROTE ->", path, "|", rec)

if __name__ == "__main__":
    main()
