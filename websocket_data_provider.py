"""
websocket_data_provider.py
-------------------------

Bu modül, kripto borsalarından veya veri sağlayıcılarından gerçek zamanlı
fiyat ve hacim verisini WebSocket protokolü kullanarak almak için bir
iskelet sunar. Projenin mevcut hali HTTP bazlı "polling" yöntemini
kullandığından, veri gecikmesi ve gereksiz CPU tüketimi meydana
gelmektedir. WebSocket'ler, olay tabanlı (event‑driven) bir mimari
sunarak gecikmeyi azaltabilir ve kaynak kullanımını iyileştirebilir.

**Önemli:** Çoğu borsa WebSocket API'si kimlik doğrulama gerektirir ve
belirli rate limit kurallarına tabidir. Aşağıda verilen örnek, Binance
Smart Chain borsası gibi halka açık bir ticker akışını dinlemek için
yazılmıştır. Gerçek projede kullanılmadan önce her borsanın belgelerine
bakılmalıdır.

Kullanım örneği (asenkron):

    import asyncio
    from websocket_data_provider import stream_prices

    async def main():
        async for symbol, price in stream_prices(["btcusdt", "ethusdt"]):
            print(symbol, price)

    asyncio.run(main())

"""

import asyncio
import json
import logging
from typing import List, AsyncGenerator, Tuple

logger = logging.getLogger(__name__)

try:
    import websockets  # type: ignore
except ImportError:
    websockets = None  # pragma: no cover


async def stream_prices(symbols: List[str]) -> AsyncGenerator[Tuple[str, float], None]:
    """
    Belirtilen sembollerin fiyatlarını bir WebSocket üzerinden dinler ve
    her güncellemede (sembol, fiyat) çifti üretir. Eğer websockets
    kütüphanesi kurulu değilse, bu fonksiyon uyarı verir ve sonsuza dek
    boş bir generator üretir.

    Args:
        symbols: Örneğin ["btcusdt", "ethusdt"] gibi borsa sembolleri.

    Yields:
        (symbol, price) tupleri.
    """
    if websockets is None:
        logger.warning(
            "websockets kütüphanesi kurulu değil. WebSocket akışı kullanılamıyor."
        )
        while True:
            await asyncio.sleep(10.0)
        return
    # Binance örneği: wss://stream.binance.com:9443/ws/<stream>
    streams = "/".join([f"{symbol}@ticker" for symbol in symbols])
    url = f"wss://stream.binance.com:9443/stream?streams={streams}"
    # Disable the internal ping/keepalive interval on the underlying WebSocket
    # connection.  The default ping behaviour of the websockets library creates
    # background tasks that are not always cancelled when the connection is
    # closed, leading to "Task was destroyed but it is pending" warnings on
    # shutdown.  Passing ``ping_interval=None`` disables the periodic ping and
    # prevents the library from spawning keepalive tasks.
    async with websockets.connect(url, ping_interval=None) as ws:  # type: ignore
        async for message in ws:
            try:
                data = json.loads(message)
                # Binance çoklu stream paket yapısı: {"stream": ..., "data": {...}}
                payload = data.get("data", {})
                sym = payload.get("s").lower()
                price = float(payload.get("c"))
                yield (sym, price)
            except Exception:
                continue


if __name__ == "__main__":
    # Örnek çalışma
    async def _demo():
        async for sym, price in stream_prices(["btcusdt"]):
            print(sym, price)
    try:
        asyncio.run(_demo())
    except KeyboardInterrupt:
        pass