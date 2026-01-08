# audit_smoke.py
# -*- coding: utf-8 -*-
"""
Audit Smoke – Hızlı duman testleri:
- Ortam değişkenleri (anahtarlar maskelenir)
- Paketler: ccxt, pandas, numpy, talib (opsiyonel)
- config.json şeması (temel alanlar)
- ccxt OKX init (API anahtarı olmadan sadece client init)
- DRY-RUN güvenli çağrılar (varsa)
- Dashboard ve metrics yolları
"""

import os, json, sys, importlib
from datetime import datetime

ROOT = os.path.abspath(os.path.dirname(__file__))
OK = True
MSG = []

def add(msg):
    print(msg)
    MSG.append(msg)

def check_env():
    keys = ["OPENAI_API_KEY", "DEEPSEEK_API_KEY", "OKX_API_KEY", "OKX_API_SECRET", "OKX_API_PASSPHRASE"]
    for k in keys:
        v = os.environ.get(k, "")
        add(f"[ENV] {k}: {'OK' if v else 'MISSING'}")
        if not v and "OKX" in k:
            # OKX anahtarı yoksa sadece init testinde public endpointlere düşülür
            pass

def check_packages():
    pkgs = ["ccxt", "pandas", "numpy"]
    missing = []
    for p in pkgs:
        try:
            importlib.import_module(p)
            add(f"[PKG] {p}: OK")
        except Exception as e:
            add(f"[PKG] {p}: MISSING ({e})")
            missing.append(p)
    try:
        import talib  # optional
        add("[PKG] talib: OK")
    except Exception:
        add("[PKG] talib: missing (optional)")
    return missing

def check_config_schema():
    path_json = os.path.join(ROOT, "config.json")
    if not os.path.exists(path_json):
        add("[CONFIG] config.json bulunamadı (SKIP)")
        return
    try:
        with open(path_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # Basit alanlar
        for k in ["theme", "auto_refresh_sec"]:
            if k in cfg:
                add(f"[CONFIG] {k}: {cfg[k]}")
        # AI fiyat/usage
        pr = cfg.get("token_price_side", {})
        if pr:
            add("[CONFIG] token_price_side: OK")
        else:
            add("[CONFIG] token_price_side: MISSING (opsiyonel)")
    except Exception as e:
        add(f"[CONFIG] HATA: {e}")

def check_ccxt_init():
    try:
        import ccxt
        ex = ccxt.okx({
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
        _markets = ex.load_markets()
        add("[CCXT] OKX init & load_markets: OK")
    except Exception as e:
        add(f"[CCXT] OKX init problem: {e}")

def check_paths():
    for p in ["logs", "metrics", "runtime", "state"]:
        fp = os.path.join(ROOT, p)
        if not os.path.isdir(fp):
            try:
                os.makedirs(fp, exist_ok=True)
                add(f"[PATH] {p}/ oluşturuldu")
            except Exception as e:
                add(f"[PATH] {p}/ HATA: {e}")
        else:
            add(f"[PATH] {p}/ OK")

def main():
    add(f"[SMOKE] Başlangıç: {datetime.utcnow().isoformat()}Z")
    check_env()
    missing = check_packages()
    check_config_schema()
    check_ccxt_init()
    check_paths()
    add("[SMOKE] Tamamlandı.")

if __name__ == "__main__":
    main()
