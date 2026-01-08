# risk_optimizer.py
# -*- coding: utf-8 -*-
"""
Risk Optimizer – Volatiliteye göre kaldıraç ve cüzdan yüzdesi kalibrasyonu
Girişler:
- metrics/ohlc_history.json (varsa) veya default
Çıkış:
- risk_config.json dosyasını (yoksa oluşturur) günceller
- Örn. yüksek volatilitede kaldıraçları bir kademe düşür, min_confidence'i +0.02 artır

Kural (örnek):
- 24h ATR% medyanı:
    < 1.5%  -> normal
    1.5–3%  -> kaldıraç -1 kademe, min_confidence +0.01
    > 3%    -> kaldıraç -2 kademe, min_confidence +0.02, max_corr_positions -1
"""

import os, json, statistics

ROOT = os.path.abspath(os.path.dirname(__file__))
METRICS_PATH = os.path.join(ROOT, "metrics", "ohlc_history.json")
RISK_CONF_PATH = os.path.join(ROOT, "risk_config.json")

DEFAULT_TIERS = [
    # (min_conf, max_conf, leverage, wallet_pct)
    (0.60, 0.65, 10, 0.20),
    (0.65, 0.70, 15, 0.25),
    (0.70, 0.75, 20, 0.30),
    (0.75, 0.80, 30, 0.35),
    (0.80, 0.85, 40, 0.40),
    (0.85, 0.90, 50, 0.50),
    (0.90, 0.95, 65, 0.55),
    (0.95, 1.01, 75, 0.60),
]

def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def estimate_atr_percent(metrics):
    # Basit bir tahmin: metrics içinden ATR% veya bar range% medyanı
    # Beklenen biçim: {"BTC/USDT": [{"atr_pct": 1.2, ...}, ...], ...}
    vals = []
    if isinstance(metrics, dict):
        for sym, rows in metrics.items():
            for r in rows[-200:]:
                v = None
                if isinstance(r, dict):
                    v = r.get("atr_pct") or r.get("true_range_pct") or r.get("range_pct")
                if v is not None:
                    vals.append(float(v))
    if len(vals) < 20:
        return 1.5  # veri yoksa makul varsayım
    return statistics.median(vals)

def _average_slippage_from_trade_log(n: int = 100) -> float:
    """Compute the average realised slippage from the last ``n`` closed trades.

    This helper reads the ``trade_log.json`` file and extracts the
    ``slippage_pct`` fields from the most recent ``n`` records.  The
    returned value is the arithmetic mean.  If no valid data is
    available, 0.0 is returned.
    """
    try:
        import json
        from pathlib import Path
        log_path = Path(__file__).resolve().parent / "trade_log.json"
        if not log_path.exists():
            return 0.0
        txt = log_path.read_text(encoding="utf-8").strip()
        if not txt:
            return 0.0
        data = json.loads(txt)
        if isinstance(data, dict):
            rows = data.get("rows", [])
        elif isinstance(data, list):
            rows = data
        else:
            return 0.0
        # Use only the last n
        recent = rows[-n:]
        values = []
        for rec in recent:
            slip = rec.get("slippage_pct")
            if slip is None:
                continue
            try:
                s = float(slip)
                # Consider only non‑zero slippage
                if s > 0:
                    values.append(s)
            except Exception:
                continue
        if not values:
            return 0.0
        import statistics
        return statistics.mean(values)
    except Exception:
        return 0.0


def adjust_tiers(tiers, atr_med):
    """Adjust leverage tiers based on ATR and average slippage.

    This function first applies volatility‑based adjustments as before
    (``atr_med``) and then further penalises leverage if the average
    realised slippage over recent trades is elevated.  Larger slippage
    suggests that order execution quality is degrading; reducing
    leverage and increasing confidence thresholds makes the bot more
    conservative in such conditions.
    """
    # ATR based shift
    lev_shift = 0
    conf_add = 0.0
    max_corr_positions = 3

    if atr_med < 1.5:
        lev_shift = 0
        conf_add = 0.00
        max_corr_positions = 3
    elif atr_med < 3.0:
        lev_shift = -1
        conf_add = 0.01
        max_corr_positions = 3
    else:
        lev_shift = -2
        conf_add = 0.02
        max_corr_positions = 2

    # Slippage penalty: further reduce leverage if slippage is high
    try:
        avg_slip = _average_slippage_from_trade_log(100)
        # For slippage over 2%, reduce one more tier; over 5%, reduce two tiers
        if avg_slip > 0.05:
            lev_shift -= 2
            conf_add += 0.02
        elif avg_slip > 0.02:
            lev_shift -= 1
            conf_add += 0.01
    except Exception:
        pass

    new_tiers = []
    for lo, hi, lev, pct in tiers:
        # Adjust leverage: each shift corresponds to 5x change
        new_lev = max(5, lev + lev_shift * 5)
        new_lo = round(lo + conf_add, 3)
        new_hi = round(hi + conf_add, 3)
        if new_hi > 1.01:
            new_hi = 1.01
        new_tiers.append((new_lo, new_hi, new_lev, pct))
    return new_tiers, max_corr_positions, conf_add

def save_risk(risk):
    with open(RISK_CONF_PATH, "w", encoding="utf-8") as f:
        json.dump(risk, f, indent=2, ensure_ascii=False)

def main():
    metrics = load_metrics()
    atr_med = estimate_atr_percent(metrics) if metrics else 1.5
    new_tiers, max_corr_positions, conf_add = adjust_tiers(DEFAULT_TIERS, atr_med)

    out = {
        "atr_median_pct": atr_med,
        "confidence_offset": conf_add,
        "max_highly_correlated_positions": max_corr_positions,
        "tiers": [
            {"min_conf": a, "max_conf": b, "leverage": l, "wallet_pct": p}
            for (a, b, l, p) in new_tiers
        ]
    }
    save_risk(out)
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
