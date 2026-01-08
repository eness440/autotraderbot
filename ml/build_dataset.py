# ml/build_dataset.py (FINAL: Smart Fallback + Strict Shape + NPZ Save)
import json, argparse, pathlib, datetime, sys, subprocess
import pandas as pd
import numpy as np

# === PATHS ===
ROOT    = pathlib.Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"
OUTDIR  = ROOT / "data"
LOGDIR  = ROOT / "logs"
OUTDIR.mkdir(parents=True, exist_ok=True)
LOGDIR.mkdir(parents=True, exist_ok=True)

LOGFILE = LOGDIR / "dataset_build.log"
FEAT_META_FILE = OUTDIR / "feature_meta.json"

# === TARGET COIN LIST ===
TARGET_COINS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT", "BNB/USDT",
    "LTC/USDT", "LINK/USDT", "OP/USDT", "ARB/USDT", "AVAX/USDT", "DOT/USDT", "TRX/USDT",
    "MATIC/USDT", "TON/USDT", "INJ/USDT", "SUI/USDT", "SEI/USDT", "APT/USDT", "APE/USDT",
    "NEAR/USDT", "FIL/USDT", "RUNE/USDT", "UNI/USDT", "SAND/USDT", "AXS/USDT", "LDO/USDT",
    "GMT/USDT", "STX/USDT", "AAVE/USDT", "MKR/USDT", "SNX/USDT", "DYDX/USDT", "IMX/USDT",
    "RPL/USDT", "PYTH/USDT", "JUP/USDT", "WIF/USDT", "PEPE/USDT", "FLOKI/USDT", "SHIB/USDT",
    "BONK/USDT", "TIA/USDT", "ORDI/USDT", "SATS/USDT", "JTO/USDT", "FTM/USDT", "GALA/USDT",
    "ROSE/USDT", "EGLD/USDT", "KAS/USDT", "JASMY/USDT", "VET/USDT", "AR/USDT", "ZETA/USDT",
    "POL/USDT", "YGG/USDT", "SSV/USDT", "PENDLE/USDT", "BIGTIME/USDT", "HOOK/USDT", "BLUR/USDT",
    "MAGIC/USDT", "HIGH/USDT", "WLD/USDT", "LUNC/USDT", "LUNA/USDT", "CELO/USDT", "CHZ/USDT",
    "XLM/USDT", "HBAR/USDT", "EOS/USDT", "THETA/USDT", "ETC/USDT", "KAVA/USDT", "MINA/USDT",
    "GMX/USDT", "CRV/USDT", "1INCH/USDT", "COMP/USDT", "CAKE/USDT", "SFP/USDT", "ANKR/USDT",
    "CELR/USDT", "ACH/USDT", "BEL/USDT", "BAND/USDT", "ALGO/USDT", "OGN/USDT", "ENJ/USDT",
    "ZIL/USDT", "HOT/USDT", "ICX/USDT", "XMR/USDT", "RSR/USDT", "SKL/USDT", "CHR/USDT",
    "DENT/USDT", "LIT/USDT"
]

REQUIRED_COLS = {"ts", "symbol", "open", "high", "low", "close", "volume"}

# 14 feature kolonu (window=180 x 14 -> model in_dim=14)
DEFAULT_FEAT_COLS = [
    "close", "ret1", "ret5", "mom10", "ma10", "ma20",
    "rsi14", "atr14", "boll_pct_b", "vwap20_delta", "volume", "vol_z20",
    "boll_width", "atr14_pct"
]


def log(msg: str):
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with LOGFILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_ohlc_multi(limit: int = 3000) -> pd.DataFrame:
    f = METRICS / "ohlc_history.json"
    if not f.exists():
        log(f"UYARI: {f} bulunamadı. Boş veri seti ile devam ediliyor.")
        return pd.DataFrame(columns=list(REQUIRED_COLS))

    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        rows = data.get("rows", data) if isinstance(data, dict) else data
    except Exception as e:
        log(f"HATA: JSON okuma hatası: {e}")
        return pd.DataFrame(columns=list(REQUIRED_COLS))

    if not rows:
        log("UYARI: JSON içerisinde veri satırı yok.")
        return pd.DataFrame(columns=list(REQUIRED_COLS))

    df = pd.DataFrame(rows)
    df.columns = [c.lower() for c in df.columns]

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        log(f"HATA: Eksik zorunlu kolon(lar): {missing}")
        return pd.DataFrame(columns=list(REQUIRED_COLS))

    # sembol normalize
    df["symbol"] = df["symbol"].astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    allowed = {s.upper().replace("/", "").replace(":", "") for s in TARGET_COINS}
    df = df[df["symbol"].isin(allowed)]

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = (
        df.dropna(subset=["ts", "close"])
          .sort_values(["symbol", "ts"])
          .reset_index(drop=True)
    )

    # Her sembol için son 'limit' kadar veriyi al
    df = df.groupby("symbol", group_keys=False).tail(limit)

    log(f"Yüklenen: {df['symbol'].nunique()} sembol, {len(df)} satır.")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = []

    for sym, g in df.groupby("symbol", group_keys=False):
        g = g.sort_values("ts").reset_index(drop=True)
        
        # Yeterli veri yoksa bu coini atla
        if len(g) < 50:
            continue

        # 5 dakikalık resample
        g = g.set_index("ts").resample("5min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "symbol": "last"
        })
        g = g.dropna().reset_index()

        c = g["close"]

        # Temel Featurelar
        g["ret1"] = c.pct_change(1)
        g["ret5"] = c.pct_change(5)
        g["mom10"] = c.pct_change(10)
        g["ma10"] = c.rolling(10).mean()
        g["ma20"] = c.rolling(20).mean()

        # RSI
        delta = c.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        rs = up.rolling(14).mean() / (down.rolling(14).mean() + 1e-9)
        g["rsi14"] = 100 - (100 / (1 + rs))

        # ATR
        tr = pd.concat([
            (g["high"] - g["low"]),
            (g["high"] - c.shift(1)).abs(),
            (g["low"] - c.shift(1)).abs()
        ], axis=1).max(axis=1)
        g["atr14"] = tr.rolling(14).mean()
        g["atr14_pct"] = g["atr14"] / (c + 1e-9)

        # Bollinger
        std = c.rolling(20).std()
        g["boll_width"] = (4 * std) / (g["ma20"] + 1e-9)
        g["boll_pct_b"] = (c - (g["ma20"] - 2 * std)) / (4 * std + 1e-9)

        # Volume Z
        vol_ma = g["volume"].rolling(20).mean()
        vol_std = g["volume"].rolling(20).std()
        g["vol_z20"] = (g["volume"] - vol_ma) / (vol_std + 1e-9)

        # VWAP proxy
        g["vwap20_delta"] = (c - c.rolling(20).mean()) / (c + 1e-9)

        # NaN içeren ilk satırları temizle
        g = g.dropna(subset=DEFAULT_FEAT_COLS)
        if not g.empty:
            out.append(g)

    if not out:
        log("UYARI: Feature oluşturulduktan sonra veri kalmadı (yetersiz bar sayısı).")
        return pd.DataFrame()

    feat_df = pd.concat(out).reset_index(drop=True)
    log(f"Feature DF: {feat_df['symbol'].nunique()} sembol, {len(feat_df)} satır işlendi.")
    return feat_df


def _generate_supervised_rows(df, window, horizon, gap, feat_cols, change_threshold):
    """
    Verilen parametrelerle X, y dizilerini oluşturur.
    """
    out_rows = []
    expected_shape = (window, len(feat_cols))

    for sym, g in df.groupby("symbol", group_keys=False):
        g = g.sort_values("ts").reset_index(drop=True)

        missing = [c for c in feat_cols if c not in g.columns]
        if missing:
            continue

        if len(g) < window + horizon + gap + 1:
            continue

        try:
            Xmat = (
                g[feat_cols]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
                .values
                .astype(np.float32)
            )
            closes = g["close"].values.astype(np.float32)
        except Exception:
            continue

        for i in range(window, len(g) - horizon - gap):
            seq = Xmat[i - window: i]

            if seq.shape != expected_shape:
                continue

            curr = closes[i + gap]
            future = closes[i + horizon + gap]
            change = np.log(future / (curr + 1e-9))

            # Threshold kontrolü
            if change > change_threshold:
                y = 1
            elif change < -change_threshold:
                y = 0
            else:
                continue 

            out_rows.append({
                "symbol": sym,
                "ts": g.loc[i, "ts"],
                "X": seq.tolist(),
                "y": int(y)
            })
    
    return out_rows


def make_supervised(
    df: pd.DataFrame,
    window: int = 60,
    horizon: int = 5,
    gap: int = 0,
    feat_cols=None,
    change_threshold: float = 0.003,
    no_undersample: bool = False
):
    feat_cols = feat_cols or DEFAULT_FEAT_COLS

    # 1. Deneme: Orijinal Threshold ile
    log(f"Supervised veri hazırlanıyor. Threshold: {change_threshold}")
    out_rows = _generate_supervised_rows(df, window, horizon, gap, feat_cols, change_threshold)

    # 2. Deneme: Eğer veri yoksa Threshold düşür (Fallback - Cold Start Mode)
    if not out_rows and change_threshold > 0.0001:
        log("UYARI: Yeterli veri çıkmadı. Threshold = 0.0 (Cold Start Mode) ile tekrar deneniyor...")
        out_rows = _generate_supervised_rows(df, window, horizon, gap, feat_cols, 0.0)

    if not out_rows:
        log("HATA: Tüm denemelere rağmen supervised veri seti boş.")
        return pd.DataFrame()

    sup = pd.DataFrame(out_rows)
    log(f"Supervised ham satır sayısı: {len(sup)} | class distribution: {sup['y'].value_counts().to_dict()}")

    # Meta datasını kaydet
    meta = {
        "feature_names": feat_cols,
        "window": window,
        "horizon": horizon,
        "gap": gap,
        "change_threshold": change_threshold,
        "no_undersample": no_undersample,
        "final_rows": len(sup)
    }
    FEAT_META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Undersampling (Dengeleme) Mantığı
    # Veri seti küçükse dengeleme yapma
    if len(sup) < 1000:
        log("UYARI: Veri seti küçük (<1000), undersampling devre dışı bırakıldı.")
    elif not no_undersample:
        c = sup["y"].value_counts()
        if len(c) > 1:
            m = c.min()
            sup = pd.concat([
                sup[sup["y"] == 0].sample(m, random_state=42),
                sup[sup["y"] == 1].sample(m, random_state=42)
            ]).sample(frac=1, random_state=42).reset_index(drop=True)
            log(f"Undersample sonrası satır sayısı: {len(sup)} | dağılım: {sup['y'].value_counts().to_dict()}")

    return sup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="Dataset oluştuktan sonra eğitimi tetikler.")
    ap.add_argument("--window", type=int, default=180, help="Geçmişe bakış penceresi (bar sayısı)")
    ap.add_argument("--horizon", type=int, default=12, help="Kaç bar sonrasını tahmin edecek")
    ap.add_argument("--limit", type=int, default=3000, help="OHLC geçmişinden çekilecek maksimum bar")
    ap.add_argument("--threshold", type=float, default=0.002, help="Yükseliş/Düşüş kabul edilecek % değişim")
    ap.add_argument("--pred_gap", type=int, default=0)
    ap.add_argument("--no_undersample", action="store_true", help="Sınıf dengelemesini kapat")
    # dummy args for compatibility
    ap.add_argument("--scale", action="store_true")
    ap.add_argument("--scale_method", type=str, default="minmax")

    args = ap.parse_args()
    log("==== Dataset Builder (Smart Fallback Enabled) ====")

    # 1. Veri Yükle
    df = load_ohlc_multi(limit=args.limit)
    if df.empty:
        log("Veri yüklenemedi, çıkılıyor.")
        return

    # 2. Özellik Ekle
    df = add_features(df)
    if df.empty:
        log("Özellik ekleme başarısız, çıkılıyor.")
        return

    # 3. Etiketle ve Şekillendir
    sup = make_supervised(
        df,
        window=args.window,
        horizon=args.horizon,
        gap=args.pred_gap,
        feat_cols=DEFAULT_FEAT_COLS,
        change_threshold=args.threshold,
        no_undersample=args.no_undersample
    )

    base_name = f"supervised_w{args.window}_h{args.horizon}_g{args.pred_gap}"
    parquet_path = OUTDIR / f"{base_name}.parquet"
    npz_path = OUTDIR / f"{base_name}.npz"

    if sup.empty:
        log("KRİTİK: Veri seti oluşturulamadı (Boş). Eğitim yapılamaz.")
        return

    # Parquet Kaydet
    try:
        sup.to_parquet(parquet_path, index=False)
        log(f"Kaydedildi: {parquet_path} | Satır: {len(sup)}")
    except Exception as e:
        log(f"Parquet kayıt hatası: {e}")

    # NPZ Kaydet (Eğitim scripti bunu sever)
    try:
        X_list = sup["X"].to_list()
        X = np.stack(X_list).astype(np.float32)  # (N, window, feat)
        y = sup["y"].values.astype(np.float32)
        
        np.savez_compressed(npz_path, X=X, y=y)
        log(f"NPZ kaydedildi: {npz_path} | X shape: {X.shape}, y shape: {y.shape}")
    except Exception as e:
        log(f"NPZ kaydedilemedi: {e}")

    # 4. Eğitim Tetikle (Opsiyonel)
    if args.train:
        log("Eğitim scriptleri tetikleniyor...")
        
        # BiLSTM
        script_bilstm = ROOT / "ml" / "bilstm_train.py"
        if not script_bilstm.exists():
            script_bilstm = ROOT / "bilstm_train.py"
        
        if script_bilstm.exists():
            try:
                subprocess.run([sys.executable, str(script_bilstm)], check=False)
            except Exception as e:
                log(f"BiLSTM eğitim hatası: {e}")

        # RL
        script_rl = ROOT / "ml" / "rl_train.py"
        if not script_rl.exists():
            script_rl = ROOT / "rl_train.py"
        
        if script_rl.exists():
            try:
                subprocess.run([sys.executable, str(script_rl)], check=False)
            except Exception as e:
                log(f"RL eğitim hatası: {e}")


if __name__ == "__main__":
    main()