"""
slippage_model.py
-----------------

Bu modül, işlemler sırasında oluşabilecek slipajı (fiyat kaymasını)
tahmin etmek için basit bir arayüz sağlar.  Gerçek bir slipaj modeli,
emir defteri derinliği, işlem hacmi, spread ve market volatilitelerine
göre bir tahmin üretebilir.  Bu stub versiyonu, import hatalarını
önlemek ve kodun diğer parçalarını bozmadan çalıştırmak amacıyla
sabit 0.0 slipaj döndürür.

Fonksiyonlar:
    estimate_slippage(exchange, symbol, amount, side, depth)
        İşlem parametrelerine göre slipaj tahmini döner.
"""
from __future__ import annotations
from typing import Optional

import os
try:
    import requests  # type: ignore
except Exception:
    # Fallback if the requests library is unavailable.  In that case, the
    # slipaj fonksiyonu her zaman 0.0 dönecektir.
    requests = None  # type: ignore


async def estimate_slippage(
    exchange: object,
    symbol: str,
    amount: float | None = None,
    side: str | None = None,
    depth: int = 20,
) -> float:
    """
    Verilen işlem parametrelerine göre slipaj (fiyat kayması) tahmini
    döner.  Bu stub, slipaj tahmini özelliği devre dışı olduğunda
    kullanılmak üzere tasarlanmıştır ve her zaman 0.0 döndürür.

    Args:
        exchange: Borsa API nesnesi (kullanılmıyor)
        symbol (str): İşlem sembolü
        amount (float|None): İşlem miktarı
        side (str|None): 'buy' veya 'sell'
        depth (int): Orderbook derinliği

    Returns:
        float: Tahmini slipaj (USDT cinsinden).  Bu stub her zaman 0.0
        döndürür.
    """
    # If no order size is provided or zero, assume no slipaj.  Also
    # handle missing side.
    if amount is None or amount <= 0:
        return 0.0
    # Use Binance futures order book as the default source.  The base
    # URL can be overridden via the environment variable
    # ``SLIPPAGE_BOOK_BASE_URL``.  Remove separators from symbol (e.g.
    # ``BTC/USDT`` → ``BTCUSDT``).  Note: if ``requests`` is not
    # available, fall back to a zero slipaj value.
    if requests is None:
        return 0.0
    symbol_normalized = symbol.replace("/", "").upper()
    base_url = os.getenv("SLIPPAGE_BOOK_BASE_URL", "https://fapi.binance.com")
    try:
        resp = requests.get(
            f"{base_url}/fapi/v1/depth",
            params={"symbol": symbol_normalized, "limit": depth},
            timeout=5,
        )
        resp.raise_for_status()
        ob = resp.json() or {}
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        # Convert to list of (price, qty) floats
        bids = [(float(p), float(q)) for p, q in bids]
        asks = [(float(p), float(q)) for p, q in asks]
        if not bids or not asks:
            return 0.0
        # Determine side; if not provided, assume symmetrical slip of zero
        if not side:
            return 0.0
        side_lower = side.lower()
        # Compute the average fill price based on amount and side
        avg_price = None
        if side_lower.startswith("buy"):
            qty_remaining = amount
            cost = 0.0
            total_qty = 0.0
            for price, qty in asks:
                if qty_remaining <= 0:
                    break
                trade_qty = qty_remaining if qty >= qty_remaining else qty
                cost += trade_qty * price
                total_qty += trade_qty
                qty_remaining -= trade_qty
            if total_qty > 0:
                avg_price = cost / total_qty
                best = asks[0][0]
                # Relative slipaj: (avg price - best ask) / best ask
                slippage = (avg_price - best) / best
                return slippage
            return 0.0
        elif side_lower.startswith("sell"):
            qty_remaining = amount
            revenue = 0.0
            total_qty = 0.0
            for price, qty in bids:
                if qty_remaining <= 0:
                    break
                trade_qty = qty_remaining if qty >= qty_remaining else qty
                revenue += trade_qty * price
                total_qty += trade_qty
                qty_remaining -= trade_qty
            if total_qty > 0:
                avg_price = revenue / total_qty
                best = bids[0][0]
                # Relative slipaj: (best bid - avg price) / best bid
                slippage = (best - avg_price) / best
                return slippage
            return 0.0
        else:
            return 0.0
    except Exception:
        # On failure (network issues etc.), return zero slipaj
        return 0.0