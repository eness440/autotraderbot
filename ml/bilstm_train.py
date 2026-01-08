"""
ml/bilstm_train.py
===================

Train a binary BiLSTM model to forecast the directional return of a
cryptocurrency pair.  The model learns from a supervised dataset built
by ``ml/build_dataset.py`` where each sample consists of a window of
technical features (e.g. returns, moving averages, RSI, ATR) and a
binary label ``y``.  The label is assigned based on the log return
between the current price and a future price separated by a horizon
(``horizon``) and an optional prediction gap (``pred_gap``).  If the
log return exceeds a positive threshold, the label is ``1`` (price
increase); if it falls below the negative threshold, the label is ``0``
(price decrease).  Neutral samples are discarded during dataset
creation.

This script supports loading data either from a ``.npz`` file (for
fast loading and strict time-based splits) or from a ``.parquet`` file.
Training uses a bidirectional GRU model by default, but you can modify
``GRU_Model`` or substitute your own PyTorch RNN.

Command line arguments:
    --data:      Name of the parquet/npz file inside ``data/``.  The
                 default ``supervised_w180_h12_g0.parquet`` corresponds
                 to a 180-step window with a 12-step horizon and 0 gap.
    --epochs:    Number of training epochs (default 20).
    --batch:     Batch size for DataLoader (default 256).
    --lr:        Learning rate (default 1e-3).

After training, the best model weights are saved to ``models/bilstm_best.pt``
and metrics (e.g. accuracy, AUC) can be logged separately via the
dashboard or training harness.
"""
import argparse
import pathlib
import json
import random
import os
import time
from collections.abc import Sequence
import ast
from datetime import datetime, timezone  # for manifest timestamps

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# Notification helper for overfitting and training events.  The
# ``send_notification`` function will dispatch messages via
# Telegram/Discord/e‑mail if configured.  If no channels are set up,
# the call will be a no‑op.  We import lazily to avoid forcing a
# dependency on requests during unit tests.
try:
    from ..notification import send_notification  # type: ignore
except Exception:
    # Fallback: define a stub that returns False
    def send_notification(message: str, subject: str | None = None) -> bool:  # type: ignore
        return False

ROOT   = pathlib.Path(__file__).resolve().parents[1]
DATA   = (ROOT / "data").resolve()
MODELS = (ROOT / "models").resolve()
LOGS   = (ROOT / "logs").resolve()
MODELS.mkdir(exist_ok=True, parents=True)
LOGS.mkdir(exist_ok=True, parents=True)

# -----------------------------------------------------------------------------
# Model update manifest paths
#
# When training is run out‑of‑process, we need a way to signal the running bot
# that new weights are available.  We accomplish this by writing a small JSON
# manifest into the models directory.  ``controller_async.py`` watches for
# changes to this file and will hot‑reload the new model without requiring a
# process restart.  We also append entries to a JSONL log for history.  The
# manifest contains a ``model`` name and the relative ``path`` to the weight
# file.  See ``publish_model_update`` below.
UPDATE_MANIFEST = MODELS / "model_update.json"
UPDATE_LOG = MODELS / "model_updates.jsonl"

def _utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO‑8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path: pathlib.Path, text: str, encoding: str = "utf-8") -> None:
    """
    Write text to ``path`` atomically using the existing ``_atomic_replace`` helper.

    Data is first written to a temporary file in the same directory; once the
    write is complete and flushed, the temp file is atomically moved over the
    destination.  This prevents other processes from reading a partially
    written file.
    """
    tmp = _tmp_path_for(path)
    with open(tmp, "w", encoding=encoding) as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    _atomic_replace(tmp, path)


def publish_model_update(payload: dict) -> None:
    """
    Publish a model update manifest for hot‑reload support.

    The provided ``payload`` should minimally include ``model`` and ``path`` keys.
    A timestamp will be added automatically.  The manifest file is overwritten
    atomically, and a JSONL log is appended to record all published updates.
    """
    entry = dict(payload)
    entry["published_at"] = _utc_now_iso()
    # Write manifest atomically
    try:
        _atomic_write_text(UPDATE_MANIFEST, json.dumps(entry, ensure_ascii=False, indent=2))
    except Exception as e:
        # If we can't write the manifest, log silently; training will still succeed
        print(f"[WARN] manifest could not be written: {e}")
    # Append to log (best effort)
    try:
        with open(UPDATE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ---------------- ATOMIC SAVE HELPERS ---------------- #
def _atomic_replace(src: pathlib.Path, dst: pathlib.Path, retries: int = 40, delay_s: float = 0.25) -> None:
    """
    Windows'ta hedef dosya anlık kilitliyse replace bazen fail olur.
    Bu yüzden kısa retry ile deneriz. Başarırsa atomic switch olur.
    """
    last_err: Exception | None = None
    for _ in range(retries):
        try:
            os.replace(str(src), str(dst))
            return
        except Exception as e:
            last_err = e
            time.sleep(delay_s)
    if last_err is not None:
        raise last_err


def _tmp_path_for(dst: pathlib.Path) -> pathlib.Path:
    # Aynı klasörde temp üret (aynı filesystem → replace atomic)
    stamp = f"{int(time.time())}_{os.getpid()}"
    return dst.with_name(dst.name + f".tmp_{stamp}")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _read_feature_meta():
    meta_path = DATA / "feature_meta.json"
    window = 180
    in_dim = 14

    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            feat_names = meta.get("feature_names", [])
            if isinstance(feat_names, list) and len(feat_names) > 0:
                in_dim = len(feat_names)
            window = int(meta.get("window", window))
        except Exception:
            pass

    return window, in_dim


def super_force_loader(series: pd.Series, window: int, in_dim: int) -> np.ndarray:
    """
    Parquet içindeki X kolonundan (list/ndarray/string JSON) 3D numpy array üretir.
    Uymayan her satırı sıfır matris yapar.
    """
    n_rows = len(series)
    print(f"  [PREPROC] Yükleniyor: {n_rows} satır (Hedef: {window} x {in_dim})...")

    raw_list = series.tolist()
    valid_arrays = []
    expected_size = window * in_dim

    for item in raw_list:
        arr = None

        # 1) Doğrudan list/ndarray
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            try:
                arr = np.asarray(item, dtype=np.float32).flatten()
            except Exception:
                arr = None

        # 2) String ise: JSON veya python list
        if arr is None and isinstance(item, (str, bytes, bytearray)):
            txt = item.decode() if isinstance(item, (bytes, bytearray)) else item
            parsed = None
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(txt)
                    break
                except Exception:
                    continue
            if parsed is not None:
                try:
                    arr = np.asarray(parsed, dtype=np.float32).flatten()
                except Exception:
                    arr = None

        # 3) Hâlâ yoksa → sıfır
        if arr is None:
            arr = np.zeros(expected_size, dtype=np.float32)

        if arr.size != expected_size:
            arr = np.zeros(expected_size, dtype=np.float32)

        valid_arrays.append(arr.reshape(window, in_dim))

    try:
        matrix = np.stack(valid_arrays).astype(np.float32)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        return matrix
    except Exception as e:
        print(f"  [KRİTİK] Stacking hatası: {e}")
        return np.zeros((n_rows, window, in_dim), dtype=np.float32)


class BinaryDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class GRU_Model(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 1):
        super().__init__()
        # Daha hafif model: hidden 128→64, layers 2→1
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0  # tek katman, dropout kapalı
        )
        self.norm = nn.LayerNorm(hidden * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        o, _ = self.gru(x)
        o_last = o[:, -1, :]
        o_last = self.norm(o_last)
        out = self.head(o_last)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="supervised_w180_h12_g0.parquet")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)  # 128 → 256
    ap.add_argument("--lr", type=float, default=0.001)
    args = ap.parse_args()

    set_seed(42)

    # CPU thread ayarı: PyTorch'un CPU'da kullanacağı thread sayısını sınırlıyoruz
    try:
        import multiprocessing
        max_threads = max(1, multiprocessing.cpu_count() - 1)
        torch.set_num_threads(max_threads)
        print(f"[INFO] torch.set_num_threads({max_threads})")
    except Exception as e:
        print(f"[WARN] CPU thread ayarlanamadı: {e}")

    window, in_dim = _read_feature_meta()

    # === Önce NPZ arayacağız ===
    data_name = args.data
    if data_name.endswith(".parquet"):
        base = data_name[:-8]  # .parquet
    else:
        base = pathlib.Path(data_name).stem

    npz_path = DATA / f"{base}.npz"
    parquet_path = DATA / f"{base}.parquet"

    # ---------------- NPZ BRANCH (ZAMAN BAZLI SPLIT) ---------------- #
    if npz_path.exists():
        print(f"[INFO] NPZ bulundu, buradan yüklenecek: {npz_path.name}")
        npz = np.load(npz_path)
        X = npz["X"].astype(np.float32)  # (N, window, feat)
        y = npz["y"].astype(np.float32).ravel()
        print(f"[INFO] NPZ X shape: {X.shape}, y shape: {y.shape}")

        # ZAMAN BAZLI SPLIT: ilk %80 train, son %20 valid
        n = len(y)
        if n < 10:
            print("[ERR] NPZ kayıt sayısı çok az.")
            return
        split_idx = int(n * 0.8)
        X_tr, X_va = X[:split_idx], X[split_idx:]
        y_tr, y_va = y[:split_idx], y[split_idx:]

    # ---------------- PARQUET BRANCH (ZAMAN + COIN BAZLI SPLIT) ---------------- #
    else:
        if not parquet_path.exists():
            files = list(DATA.glob("*.parquet"))
            if not files:
                print("[ERR] data/ klasöründe ne npz ne parquet var.")
                return
            parquet_path = max(files, key=os.path.getctime)
            print(f"[WARN] {data_name} yok, en yeni parquet kullanılıyor: {parquet_path.name}")

        print(f"[INFO] Dataset (parquet): {parquet_path.name}")
        df = pd.read_parquet(parquet_path)

        if "X" not in df.columns or "y" not in df.columns:
            print("[ERR] Dataset içinde 'X' veya 'y' kolonu yok. build_dataset.py ile yeniden üret.")
            return

        # Zaman bazlı + coin bazlı split:
        # Her sembol için ts'e göre sırala, ilk %80 train, son %20 valid.
        if "symbol" in df.columns and "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
            df = df.dropna(subset=["ts"])
            train_parts = []
            valid_parts = []
            for sym, g in df.groupby("symbol", sort=False):
                g = g.sort_values("ts").reset_index(drop=True)
                n_sym = len(g)
                if n_sym < 10:
                    continue
                split_idx = int(n_sym * 0.8)
                if split_idx <= 0 or split_idx >= n_sym:
                    continue
                train_parts.append(g.iloc[:split_idx])
                valid_parts.append(g.iloc[split_idx:])
            if not train_parts or not valid_parts:
                print("[WARN] Coin bazlı split yapılamadı, global zaman split deneniyor.")
                df = df.sort_values("ts").reset_index(drop=True)
                n = len(df)
                split_idx = int(n * 0.8)
                tr_df = df.iloc[:split_idx].reset_index(drop=True)
                va_df = df.iloc[split_idx:].reset_index(drop=True)
            else:
                tr_df = pd.concat(train_parts, ignore_index=True)
                va_df = pd.concat(valid_parts, ignore_index=True)
        else:
            # symbol veya ts yoksa: sadece global index bazlı split
            n = len(df)
            split_idx = int(n * 0.8)
            tr_df = df.iloc[:split_idx].reset_index(drop=True)
            va_df = df.iloc[split_idx:].reset_index(drop=True)

        X_tr = super_force_loader(tr_df["X"], window, in_dim)
        y_tr = tr_df["y"].values.astype(np.float32)

        X_va = super_force_loader(va_df["X"], window, in_dim)
        y_va = va_df["y"].values.astype(np.float32)

    # ---------------- DEBUG & SHAPE KONTROL ---------------- #
    print("\n" + "!" * 40)
    print(f"[DEBUG] X_train Shape: {X_tr.shape}")
    print(f"[DEBUG] X_train Mean : {np.mean(X_tr):.6f}")
    print("!" * 40 + "\n")

    if X_tr.ndim == 2:
        N, total_features = X_tr.shape
        new_window = window
        new_in_dim = in_dim
        if total_features % window == 0:
            new_in_dim = total_features // window
            new_window = window
            print(f"[INFO] X_train 2D; split to (window={new_window}, features={new_in_dim}) using meta window")
        else:
            new_window = total_features
            new_in_dim = 1
            print(f"[INFO] X_train 2D; meta window mismatch. reshape → (N={N}, {new_window}, 1)")
        X_tr = X_tr.reshape(N, new_window, new_in_dim)
        X_va = X_va.reshape(X_va.shape[0], X_va.shape[1], new_in_dim)
        window = new_window
        in_dim = new_in_dim

    if X_tr.ndim != 3:
        print("🛑 [HATA] X_train 3D değil. (N, window, feat) bekleniyor.")
        return

    if X_tr.shape[2] != in_dim:
        print(f"[WARN] in_dim meta={in_dim}, gerçek={X_tr.shape[2]} → in_dim güncelleniyor.")
        in_dim = X_tr.shape[2]

    # ---------------- SCALING ---------------- #
    print("[INFO] Scaling...")
    N_tr, W, F = X_tr.shape
    N_va = X_va.shape[0]
    scaler = StandardScaler()

    X_tr_2d = X_tr.reshape(-1, F)
    X_va_2d = X_va.reshape(-1, F)

    X_tr_scaled = scaler.fit_transform(X_tr_2d).reshape(N_tr, W, F)
    X_va_scaled = scaler.transform(X_va_2d).reshape(N_va, W, F)

    X_tr = X_tr_scaled
    X_va = X_va_scaled

    tr_ds = BinaryDataset(X_tr, y_tr)
    va_ds = BinaryDataset(X_va, y_va)

    workers = 0 if os.name == "nt" else 2
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=workers)
    va_dl = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = GRU_Model(in_dim=in_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    best_auc = 0.0
    best_path = MODELS / "bilstm_best.pt"
    # Track the training accuracy corresponding to the best validation
    # accuracy.  This will be used after training to detect possible
    # overfitting (i.e. when training accuracy greatly exceeds
    # validation accuracy).  We initialise to NaN so that if it never
    # gets updated, the overfitting check is skipped.
    best_train_acc = float("nan")

    print("\n🚀 EĞİTİM BAŞLIYOR...\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for xb, yb in tr_dl:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())

        # Validasyon
        model.eval()
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for xb, yb in va_dl:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_targets.append(yb.numpy())

        y_prob = np.concatenate(all_probs, axis=0)
        y_true = np.concatenate(all_targets, axis=0)

        y_pred = (y_prob > 0.5).astype(int)
        acc = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = float("nan")

        print(f"Ep {epoch:02d} | Loss: {np.mean(losses):.4f} | Val Acc: {acc:.3f} | Val AUC: {auc:.3f}")

        if acc > best_acc:
            best_acc = acc
            best_auc = auc

            payload = {"model": model.state_dict(), "in_dim": in_dim, "window": window}
            tmp_path = _tmp_path_for(best_path)
            try:
                torch.save(payload, tmp_path)
                _atomic_replace(tmp_path, best_path)
            except Exception as e:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass
                raise e

            # Compute training accuracy when a new best validation accuracy
            # is observed.  We evaluate the current model on the training
            # set using the same threshold (0.5) and store the result.
            try:
                model.eval()
                all_train_probs = []
                all_train_targets = []
                with torch.no_grad():
                    for xb, yb in tr_dl:
                        xb = xb.to(device)
                        logits = model(xb)
                        probs = torch.sigmoid(logits).cpu().numpy()
                        all_train_probs.append(probs)
                        all_train_targets.append(yb.numpy())
                y_train_prob = np.concatenate(all_train_probs, axis=0)
                y_train_true = np.concatenate(all_train_targets, axis=0)
                y_train_pred = (y_train_prob > 0.5).astype(int)
                train_acc = accuracy_score(y_train_true, y_train_pred)
            except Exception:
                train_acc = float("nan")
            best_train_acc = train_acc

    # Publish update manifest so a running bot can reload the new model.
    try:
        # When publishing a new model update, include a ``canary`` flag.
        # Downstream components (e.g. main_bot_async) can inspect this
        # flag and reduce risk exposure while the new model is being
        # validated in live trading.  The canary period and capital
        # allocation are controlled outside of this script.
        publish_model_update({"model": "bilstm", "path": best_path.name, "canary": True})
    except Exception as e:
        print(f"[WARN] publish_model_update failed: {e}")

    # Metric kaydı
    METRICS_DIR = ROOT / "metrics"
    METRICS_DIR.mkdir(exist_ok=True, parents=True)
    metrics_path = METRICS_DIR / "bilstm_metrics.json"
    try:
        from datetime import datetime, timezone
        metrics_data = {
            "last_accuracy": float(best_acc),
            "last_auc": float(best_auc),
            "time": datetime.now(timezone.utc).isoformat(),
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] BiLSTM metrics kaydedilemedi: {e}")

    print(f"\n✅ EĞİTİM BİTTİ. Best Acc: {best_acc:.3f} | Best AUC: {best_auc:.3f}")
    print(f"Model: {best_path}")

    # -------------------------------------------------------------------------
    # Overfitting alarm & notification
    #
    # After training completes, compare the training accuracy observed when
    # the best validation accuracy was achieved (best_train_acc) with the
    # best validation accuracy (best_acc).  If the gap is larger than
    # 10 percentage points, raise an overfitting warning.  We avoid
    # sending notifications for trivial differences or when either
    # accuracy is NaN.
    try:
        # Guard against NaN (can't compare reliably)
        import math as _math
        if not (_math.isnan(best_train_acc) or _math.isnan(best_acc)):
            acc_gap = best_train_acc - best_acc
            # Overfitting threshold: > 0.10 (10 percentage points)
            if acc_gap > 0.10:
                msg = (
                    f"⚠️ BiLSTM overfitting detected: training accuracy {best_train_acc:.3f} "
                    f"vs validation accuracy {best_acc:.3f}. The gap of {acc_gap:.3f} suggests "
                    f"the model may not generalise well. Consider early stopping, regularisation or "
                    f"walk‑forward validation."
                )
                try:
                    send_notification(msg, subject="BiLSTM Overfitting Alert")
                except Exception:
                    pass
    except Exception:
        pass


if __name__ == "__main__":
    main()
