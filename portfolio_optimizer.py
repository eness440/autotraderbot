# -*- coding: utf-8 -*-
"""
portfolio_optimizer.py

Bu modül, her trade için önerilen kaldıraç (lev) ve cüzdan kullanım
oranını (wallet_allocation_percent) daha rafine bir şekilde ayarlamak
üzere basit bir Kelly/mean-variance esinli optimizasyon sağlar.

Fonksiyonlar:
    kelly_fraction(p, r): İdeal sermaye oranını hesaplar.
    optimize_leverage_and_allocation(master_conf, current_lever, current_alloc):
        Verilen master güven skoru, mevcut kaldıraç ve cüzdan oranını
        kullanarak yeni değerler döndürür.

Bu modül, risk_manager veya controller katmanında çağrılabilir ve
botun toplam riskini daha matematiksel bir yaklaşımla ayarlayabilir.
"""
from __future__ import annotations

def kelly_fraction(p: float, r: float = 1.0) -> float:
    """
    Basit Kelly formülü.

    Args:
        p: Kazanma olasılığı (0–1 arası).
        r: Risk/ödül oranı (ortalama getirinin ortalama risk oranına oranı).

    Returns:
        0–1 aralığında ideal sermaye oranı. Negatif sonuçlar 0 olarak döner.
    """
    try:
        p = float(p)
        r = float(r)
        if r <= 0:
            return 0.0
        f = (p * (r + 1.0) - 1.0) / r
        if f < 0.0:
            f = 0.0
        if f > 1.0:
            f = 1.0
        return f
    except Exception:
        return 0.0

def optimize_leverage_and_allocation(
    master_conf: float,
    current_leverage: int,
    current_alloc: float,
    risk_reward: float = 1.0,
    max_leverage: int = 30,
    min_leverage: int = 1,
    max_alloc: float = 0.40,
    min_alloc: float = 0.05,
) -> tuple[int, float]:
    """
    Master confidence ve mevcut risk parametrelerine göre yeni kaldıraç ve
    cüzdan kullanım yüzdesi hesapla.

    - Kelly formülünden elde edilen 'f' değeri, mevcut kaldıraç ve cüzdan
      oranları ile harmanlanır.
    - r değeri, pozisyonun ortalama risk/ödül oranı olarak varsayılır.

    Args:
        master_conf: 0–1 aralığında güven skoru.
        current_leverage: Mevcut kaldıraç (örneğin 10).
        current_alloc: Mevcut cüzdan kullanım yüzdesi (0–1 arası).
        risk_reward: Varsayılan risk/ödül oranı (TP mesafesi / SL mesafesi).
        max_leverage: Kaldıraç için üst sınır.
        min_leverage: Kaldıraç için alt sınır.
        max_alloc: Cüzdan oranı için üst sınır.
        min_alloc: Cüzdan oranı için alt sınır.

    Returns:
        (new_leverage, new_alloc) ikilisi.
    """
    try:
        p = max(0.0, min(1.0, float(master_conf)))
        f = kelly_fraction(p, risk_reward)
        # Kaldıraç: mevcut kaldıraç ile Kelly fraksiyonu arasında lineer
        new_lev = int(round(min_leverage + f * (max_leverage - min_leverage)))
        new_alloc = min_alloc + f * (max_alloc - min_alloc)
        # Uygulanan mevcut parametreleri sınırlamak için bir ortalama
        final_lev = int(round((new_lev + current_leverage) / 2.0))
        final_alloc = (new_alloc + current_alloc) / 2.0
        # Clamp final values
        if final_lev < min_leverage:
            final_lev = min_leverage
        if final_lev > max_leverage:
            final_lev = max_leverage
        if final_alloc < min_alloc:
            final_alloc = min_alloc
        if final_alloc > max_alloc:
            final_alloc = max_alloc
        return final_lev, final_alloc
    except Exception:
        # On error, return current values
        return current_leverage, current_alloc