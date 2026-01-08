"""
orderbook_analyzer.py
----------------------

Bu modül, sipariş defteri tabanlı göstergeler ve ayarlamalar için basit
fonksiyonlar sağlar.  Eğer gerçek bir ``orderbook_analyzer`` modülü
kullanılamıyorsa, bu stub modül botun diğer bölümlerinde import hatalarını
önlemek ve varsayılan davranışlar sunmak için eklenmiştir.

Fonksiyonlar:
    adjust_confidence_with_imbalance(base_conf, imbalance, direction, scale)
        Master confidence değerini, emir defterindeki alış/satış dengesine
        göre ayarlar.  Eğer ``imbalance`` None ise, ``base_conf`` aynen
        döner.  Aksi halde long pozisyonlarda pozitif dengesizlik
        confidence'i artırır, short pozisyonlarda negatif dengesizlik etkili
        olur.  Sonuç 0..1 aralığına kırpılır.

    calculate_imbalance(exchange, symbol, depth)
        Verilen sembol ve derinlik için orderbook dengesini hesaplar.
        Gerçek implementasyon API üzerinden orderbook verisi çekip
        alış/satış hacimlerini kıyaslamalıdır.  Bu stub versiyonu her zaman
        ``None`` döner; kullanıcı kendi analizini entegre etmek isterse bu
        fonksiyonu genişletebilir.
"""
from __future__ import annotations
from typing import Optional


def adjust_confidence_with_imbalance(
    base_conf: float,
    imbalance: Optional[float],
    direction: str,
    scale: float = 0.20,
) -> float:
    """
    Sipariş defteri dengesine göre master confidence değerini ayarlar.

    Args:
        base_conf (float): Orijinal confidence değeri (0 ile 1 arasında).
        imbalance (float|None): Orderbook dengesizliği.  Pozitif
            değerler alışların, negatif değerler satışların baskın olduğunu
            gösterir.  ``None`` ise hiçbir ayarlama yapılmaz.
        direction (str): 'long' veya 'short'.  Long için pozitif dengesizlik
            confidence'i artırır; short için negatif dengesizlik dikkate alınır.
        scale (float): Denge etkisinin çarpanı.  Varsayılan 0.20.

    Returns:
        float: Ayarlanmış confidence, 0..1 aralığında kırpılmış.
    """
    try:
        # Baz değer 0..1 arasında olsun
        conf = float(base_conf)
    except Exception:
        return 0.0
    # Denge verisi yoksa değişiklik yapma
    if imbalance is None:
        return max(0.0, min(1.0, conf))
    try:
        imbal = float(imbalance)
    except Exception:
        return max(0.0, min(1.0, conf))
    # Yön faktörü
    adjustment = 0.0
    if direction.lower().startswith("long"):
        adjustment = imbal * scale
    elif direction.lower().startswith("short"):
        adjustment = -imbal * scale
    # Yeni confidence'i hesapla ve 0..1'e kliple
    new_conf = conf + adjustment
    return max(0.0, min(1.0, new_conf))


async def calculate_imbalance(
    exchange: object,
    symbol: str,
    depth: int = 20,
) -> Optional[float]:
    """
    Sipariş defteri alış/satış dengesini hesaplar.

    Bu fonksiyon, verilen sembol için Binance vadeli işlemler (Futures) API'si
    üzerinden sipariş defteri derinlik verisini çeker ve ilk ``depth``
    seviyedeki alış ve satış hacimlerini karşılaştırarak bir dengesizlik
    ölçüsü hesaplar. Dengesizlik, (toplam alış hacmi - toplam satış hacmi)
    / (toplam alış hacmi + toplam satış hacmi) formülü ile [-1, 1]
    aralığına normalize edilir. Sonuç, long işlemler için pozitif değer
    alış baskısını, short işlemler için negatif değer satış baskısını ifade
    eder.  Herhangi bir hata durumunda veya veri alınamazsa ``None`` döner.

    Args:
        exchange: Borsa API nesnesi (kullanılmıyor, geriye dönük uyum için)
        symbol (str): İşlem sembolü (örn. "BTC/USDT" veya "BTCUSDT")
        depth (int): Orderbook'tan çekilecek seviye sayısı (en fazla 100)

    Returns:
        Optional[float]: Alış/satış dengesizliği veya hata halinde ``None``.
    """
    # Talep edilen derinliği sınırla (Binance API max 100)
    depth = max(1, min(int(depth), 100))

    # Sembolü API formatına dönüştür (örn. "BTC/USDT" -> "BTCUSDT")
    normalized_symbol = symbol.replace("/", "").upper()

    # API adresini ortam değişkeninden veya varsayılan değerden oku
    import os
    base_url = os.getenv("BINANCE_FAPI_BASE_URL", "https://fapi.binance.com")
    endpoint = f"{base_url}/fapi/v1/depth?symbol={normalized_symbol}&limit={depth}"

    # İsteği göndermek için requests modülünü kullan; mevcut değilse None döndür
    try:
        import requests  # type: ignore
    except Exception:
        return None

    try:
        response = requests.get(endpoint, timeout=5)
        if response.status_code != 200:
            return None
        data = response.json()
        bids = data.get("bids") or []
        asks = data.get("asks") or []
        # Toplam alış ve satış hacimlerini hesapla (sadece miktarlar)
        total_bid_vol = 0.0
        total_ask_vol = 0.0
        # Bids [price, quantity] listelerinden quantity al
        for entry in bids[:depth]:
            try:
                qty = float(entry[1])
                total_bid_vol += qty
            except Exception:
                continue
        for entry in asks[:depth]:
            try:
                qty = float(entry[1])
                total_ask_vol += qty
            except Exception:
                continue
        # Toplam hacim sıfırsa dengesizlik hesaplanamaz
        denom = total_bid_vol + total_ask_vol
        if denom == 0:
            return None
        imbalance = (total_bid_vol - total_ask_vol) / denom
        # [-1, 1] aralığına kırp (her ne kadar formül gereği burada kalması
        # bekleniyor olsa da, güvenlik için sınırlıyoruz)
        if imbalance > 1:
            imbalance = 1.0
        elif imbalance < -1:
            imbalance = -1.0
        return imbalance
    except Exception:
        # herhangi bir hata durumunda None döndür
        return None