# -*- coding: utf-8 -*-
"""
regime_detector_async.py
- Haftalık olarak 4h veriden piyasa rejimi (bull/bear/sideways) tespiti.
- Sonucu .cache/regime.json içine yazar. Controller bu sinyali bonus olarak kullanabilir.
"""
import asyncio
import json
import os
from statistics import mean
from typing import Literal
from datetime import datetime, timedelta

from logger import get_logger

log = get_logger("regime")

CACHE_DIR = ".cache"
CACHE_FILE = os.path.join(CACHE_DIR, "regime.json")

def _classify_regime(returns_4h, vol_4h) -> Literal["bull","bear","sideways"]:
    # çok basit bir örnek eşikleme: ortalama getiri ve volatilite
    mu = mean(returns_4h) if returns_4h else 0.0
    v = mean(vol_4h) if vol_4h else 0.0
    # eşikler kaba örnek: (istersen konfigürasyona al)
    if mu > 0 and v < 0.015:
        return "bull"
    if mu < 0 and v > 0.02:
        return "bear"
    return "sideways"

async def detect_weekly_regime(exchange, symbol="BTC/USDT"):
    """
    4h 1200+ bar çek, rejimi hesapla, cache'e yaz.
    """
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe="4h", limit=1500)
        closes = [c[4] for c in ohlcv]
        # basit 4h getiriler
        rets = []
        vols = []
        for i in range(1, len(closes)):
            r = (closes[i] - closes[i-1]) / closes[i-1]
            rets.append(r)
        # 12 barlık (2 gün) basit volatilite proxysi
        w = 12
        for i in range(w, len(closes)):
            window = closes[i-w:i]
            mu = mean(window)
            vols.append(abs(closes[i] - mu) / mu)

        regime = _classify_regime(rets[-150:], vols[-150:])
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "symbol": symbol,
                "regime": regime,
                "updated": datetime.utcnow().isoformat()+"Z"
            }, f, ensure_ascii=False, indent=2)
        log.info(f"REGIME={regime} | kaydedildi → {CACHE_FILE}")
    except Exception as e:
        log.warning(f"regime detect hatası: {e}")

async def start_regime_scheduler(exchange, symbol="BTC/USDT"):
    """
    Haftada 1 çalıştır. İlk başta da bir kez çalıştır.
    """
    await detect_weekly_regime(exchange, symbol=symbol)
    while True:
        await asyncio.sleep(7 * 24 * 3600)  # haftalık
        await detect_weekly_regime(exchange, symbol=symbol)
