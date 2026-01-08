"""
ml/rl_predict.py
================
RL (PPO) modeli ile canlı tahmin (inference) yapar.
Eğitilmiş 'models/ppo_multi.zip' modelini yükler, belirtilen sembol için
son verileri çeker, normalize eder ve modele sorar.

Çıktı:
    metrics/ai_predictions.json dosyasına eklenir.
"""
import json
import pathlib
import argparse
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timezone

# Stable-baselines3 yoksa graceful exit
try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None

# Dosya yolları
ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
METRICS = ROOT / "metrics"
OHLC_FILE = METRICS / "ohlc_history.json"
PRED_FILE = METRICS / "ai_predictions.json"

def get_latest_data(symbol: str, window: int):
    """ohlc_history.json'dan belirtilen sembolün son verilerini çeker ve işler."""
    if not OHLC_FILE.exists():
        return None, None, None

    try:
        data = json.loads(OHLC_FILE.read_text(encoding="utf-8"))
        rows = data.get("rows", data) if isinstance(data, dict) else data
    except Exception:
        return None, None, None

    df = pd.DataFrame(rows)
    if df.empty:
        return None, None, None

    # Sembol temizliği (BTC/USDT -> BTCUSDT)
    # Veri setindeki semboller genelde slashsız veya 'BTC/USDT' formatında olabilir.
    # Burada esnek bir eşleşme yapıyoruz.
    target_clean = symbol.replace("/", "").replace(":", "").upper()
    
    # DataFrame içindeki sembolleri de temizleyerek karşılaştır
    df["symbol_clean"] = df["symbol"].astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    
    # İlgili sembolü filtrele ve tarihe göre sırala
    df_sym = df[df["symbol_clean"] == target_clean].sort_values("ts")
    
    if len(df_sym) < window:
        return None, None, None

    # Son 'window' kadar kapanış fiyatını al
    closes = df_sym["close"].astype(float).values[-window:]
    last_ts = df_sym.iloc[-1]["ts"]
    last_price = float(closes[-1])

    # Normalizasyon (rl_env.py _obs mantığıyla BİREBİR AYNI olmalı)
    # Formül: x = window_slice / base - 1.0
    base = closes[0] if closes[0] != 0 else 1.0
    normalized = (closes / base) - 1.0
    
    # Model girdisi: (window, 1) şeklinde float32 numpy array
    # PPO (MlpPolicy) genelde (N_envs, obs_dim) bekler ama predict fonksiyonu 
    # tekil observation'ı (obs_dim,) veya (1, obs_dim) olarak da kabul edebilir.
    # rl_env'deki observation space (window, 1) olduğu için:
    obs = normalized.reshape(window, 1).astype(np.float32)
    
    return obs, last_ts, last_price

def append_prediction(record: dict):
    data = []
    if PRED_FILE.exists():
        try:
            data = json.loads(PRED_FILE.read_text(encoding="utf-8"))
        except:
            data = []
    
    # Dosya çok şişmesin, son 2000 tahmin yeterli
    if len(data) > 2000:
        data = data[-2000:]
        
    data.append(record)
    PRED_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return PRED_FILE

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--window", type=int, default=60)
    args = ap.parse_args()

    if PPO is None:
        print("[RL-PREDICT] stable_baselines3 kurulu değil.")
        return

    # ARTIK TEKİL MODEL DEĞİL, MULTI MODEL YÜKLÜYORUZ
    model_path = MODELS / "ppo_multi.zip"
    if not model_path.exists():
        print(f"[RL-PREDICT] Model bulunamadı: {model_path}")
        return

    # Veriyi hazırla
    obs, ts, price = get_latest_data(args.symbol, args.window)
    if obs is None:
        print(f"[RL-PREDICT] {args.symbol} için yeterli veri ({args.window} bar) yok.")
        return

    # Modeli yükle ve tahmin yap
    try:
        # device="cpu" ile yüklemek güvenlidir (tahmin için GPU şart değil)
        model = PPO.load(model_path, device="cpu")
        
        # predict, (action, state) döner.
        # action: 0=Flat, 1=Long, 2=Short (rl_env.py tanımı)
        action, _states = model.predict(obs, deterministic=True)
        
        # Action sayısal değerini al (numpy array gelebilir)
        if isinstance(action, np.ndarray):
            action = action.item()
        
        # Güven skoru PPO'da direkt gelmez, action üzerinden yapay bir skor atayalım.
        # Bu skor hibrit sistemde ağırlıklandırılacak.
        if action == 1:
            decision = "long"
            confidence = 0.65  # RL bir yön seçtiyse, nötrden hallicedir.
        elif action == 2:
            decision = "short"
            confidence = 0.65
        else:
            decision = "neutral"
            confidence = 0.5

        print(f"RL Tahmini ({args.symbol}): Action={action} ({decision}) | Conf={confidence:.2f}")

        rec = {
            "model": "ppo_rl",
            "symbol": args.symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_ts": ts,
            "price": price,
            "action": decision, # long/short/neutral
            "confidence": confidence,
            "raw_action": int(action)
        }
        path = append_prediction(rec)
        print(f"WROTE -> {path}")

    except Exception as e:
        print(f"[RL-PREDICT] Tahmin hatası: {e}")

if __name__ == "__main__":
    main()