# -*- coding: utf-8 -*-
"""
anomaly_detector.py

Fiyat anomalilerini tespit etmeye yönelik basit yardımcı fonksiyon.
Özellikle ani fiyat sıçramaları veya ATR'nin birkaç katı kadar hızlı
hareketleri, slipaj ve likidasyon riskini artırır. Bu modül, son
fiyat ile güncel fiyat arasındaki farkı ATR'ye oranlayarak bir eşik
üzerinde olup olmadığını denetler.

Fonksiyon:
    is_anomalous_price(last_price, current_price, atr, threshold)

Kullanım:
    # _analyze_one fonksiyonu içinde:
    if is_anomalous_price(LAST_PRICES.get(sym), price, atr):
        ta_pack['anomaly'] = True
"""
from __future__ import annotations

def is_anomalous_price(
    last_price: float | None,
    current_price: float | None,
    atr_value: float | None,
    threshold: float = 3.0,
) -> bool:
    """
    Son fiyat ve güncel fiyatın farkı ATR'nin belirli bir katından büyükse
    anomali olarak kabul edilir.

    Args:
        last_price: Önceki periyottaki kapanış fiyatı veya None.
        current_price: Güncel fiyat.
        atr_value: ATR değeri (aynı timeframe için).
        threshold: ATR'ye göre anomali eşiği (örn. 3 → 3*ATR).

    Returns:
        True ise anomali var, False ise yok veya veriler eksik.
    """
    try:
        if last_price is None or current_price is None or atr_value is None:
            return False
        lp = float(last_price)
        cp = float(current_price)
        atr = float(atr_value)
        if atr <= 0:
            return False
        diff = abs(cp - lp)
        return diff > threshold * atr
    except Exception:
        return False