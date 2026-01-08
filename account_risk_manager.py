"""
account_risk_manager.py

Bu modül, hesap bazlı risk ve marjin yönetimi sağlar. Amaç, yeni bir
pozisyon açmadan önce mevcut açık risk miktarını ve kullanılabilir
marjini hesaba katarak pozisyon büyüklüğünü güvenli bir şekilde
ayarlamaktır. OKX üzerinde "51008" (yetersiz marjin) ve "51004"
(maksimum pozisyon limiti) gibi hatalar alma olasılığını azaltmak
için bu modülden hesaplanan dinamik pozisyon boyutu kullanılabilir.

Fonksiyonlar asenkron olarak tasarlanmıştır; CCXT ile çalışan
asenkron yöntemler `safe_order_wrapper.call` ile sarılarak
kullanılmalıdır. Örneğin `adjust_position_size` fonksiyonu,
`safe_submit_entry_plan` içinde çağrıldığında, bot yeni bir
i̇şlem açmadan önce hesabın toplam riskini ve kullanılabilir marjini
değerlendirir ve `current_qty` değerini buna göre küçültür. Eğer
kalan risk kapasitesi yoksa 0 döner ve işlem açılmamalıdır.

Not: Bu modül hesap verilerini yaklaşık olarak hesaplar. OKX'ten
gelen `fetch_positions` ve `fetch_balance` yanıtları kimi zaman eksik
olabilir. Bu nedenle, risk oranı düşük tutularak hareket edilir ve
kalan marjin eksiye düşmeyecek şekilde pozisyon küçültülür.
"""

from __future__ import annotations

import ccxt
from typing import Tuple

from safe_order_wrapper import call  # rate-limit aware call helper

# Hesap bazlı maksimum açık risk yüzdesi. Örneğin 0.05 → toplam bakiyenin %5'i
MAX_PORTFOLIO_RISK_PERCENTAGE = 0.05

async def _fetch_total_balance_usdt(exchange: ccxt.Exchange) -> Tuple[float, float]:
    """
    Hesabın toplam ve kullanılabilir USDT bakiyesini döndürür.

    Args:
        exchange: ccxt OKX nesnesi

    Returns:
        (total_balance, available_balance)
    """
    try:
        bal = await call(exchange.fetch_balance, label="ACC-BAL")
        if bal is None:
            return 0.0, 0.0
        total = 0.0
        free = 0.0
        # CCXT standardında USDT cüzdan bilgisi 'USDT' veya 'USDT/USDT' altında olabilir.
        try:
            usdt = bal.get("USDT") or bal.get("USDT/USDT") or {}
            total = float(usdt.get("total", 0.0))
            free = float(usdt.get("free", 0.0))
        except Exception:
            pass
        # Eğer bu alanlar yoksa, genel toplam ve free alanlarını kullan
        if total == 0.0:
            try:
                total = float(bal.get("total", 0.0))
                free = float(bal.get("free", 0.0))
            except Exception:
                pass
        return max(total, 0.0), max(free, 0.0)
    except Exception:
        return 0.0, 0.0

async def _fetch_open_positions_notional(exchange: ccxt.Exchange) -> float:
    """
    Açık pozisyonların toplam notional değerini (USD cinsinden) döndürür.
    fetch_positions çağrısından dönen listede her bir pozisyon için
    notional alanını kullanır. Eğer notional yoksa, contracts*markPrice
    tahmini yapılır.

    Args:
        exchange: ccxt OKX nesnesi

    Returns:
        float: Toplam açık notional (mutlak değerlerin toplamı)
    """
    try:
        positions = await call(exchange.fetch_positions, label="ACC-POS")
        total_notional = 0.0
        if positions:
            for p in positions:
                try:
                    notional = p.get("notional")
                    if notional is None:
                        # contracts ve markPrice varsa hesapla
                        contracts = p.get("contracts") or p.get("positionAmt")
                        price = p.get("markPrice") or p.get("info", {}).get("markPx")
                        if contracts is not None and price is not None:
                            notional = float(contracts) * float(price)
                    if notional is not None:
                        total_notional += abs(float(notional))
                except Exception:
                    continue
        return total_notional
    except Exception:
        return 0.0

async def adjust_position_size(
    exchange: ccxt.Exchange,
    symbol: str,
    current_qty: float,
    leverage: int,
    entry_price: float,
    max_risk_pct: float = MAX_PORTFOLIO_RISK_PERCENTAGE,
) -> float:
    """
    Hesap bazlı risk limitini aşmamak için pozisyon büyüklüğünü ayarlar.

    Yeni pozisyonun notional değerini, mevcut açık risk ve izin verilen
    risk limiti ile karşılaştırır. Eğer toplam risk limiti aşılacaksa,
    `current_qty` değerini orantısal olarak küçültür. Kalan risk kapasitesi
    yoksa 0 döner (işlem açmamalı).

    Args:
        exchange: ccxt OKX nesnesi
        symbol: İşlem yapılacak sembol (örn. "BTC/USDT")
        current_qty: Planlanan coin miktarı
        leverage: Planlanan kaldıraç (şu anda kullanılmıyor, ileride risk
                   hesaplarında kullanılabilir)
        entry_price: Planlanan giriş fiyatı
        max_risk_pct: Hesap bakiyesinin maksimum açık risk yüzdesi

    Returns:
        float: Ayarlanmış miktar. 0 ise pozisyon açılmamalıdır.
    """
    try:
        qty = float(current_qty)
    except Exception:
        return current_qty
    if entry_price is None or entry_price <= 0 or qty <= 0:
        return current_qty
    try:
        # Bakiye ve açık risk
        total_bal, _ = await _fetch_total_balance_usdt(exchange)
        if total_bal <= 0:
            return current_qty
        open_risk = await _fetch_open_positions_notional(exchange)
        # Yeni pozisyonun notional'ı
        new_notional = abs(qty * float(entry_price))
        # İzin verilen toplam açık risk
        allowed_open = total_bal * float(max_risk_pct)
        # Eğer mevcut + yeni risk limitin altındaysa, miktarı değiştirme
        if open_risk + new_notional <= allowed_open:
            return current_qty
        remaining = allowed_open - open_risk
        if remaining <= 0:
            return 0.0
        factor = remaining / new_notional
        if factor <= 0:
            return 0.0
        adj_qty = qty * factor
        if adj_qty < 1e-8:
            return 0.0
        return adj_qty
    except Exception:
        return current_qty