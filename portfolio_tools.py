import ccxt
import pandas as pd
import numpy as np

def calculate_correlation(exchange, symbols, timeframe='4h', limit=100):
    """
    Faz 3.1: Belirtilen coin'lerin kapanış fiyatları arasındaki korelasyon matrisini hesaplar.
    """
    price_df = pd.DataFrame()
    
    # 1. Tüm Sembollerin Kapanış Fiyatlarını Topla
    for symbol in symbols:
        try:
            # 4 saatlik (Ana Trend) kapanış fiyatlarını çekiyoruz
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # DataFrame'e sadece kapanış fiyatını ekle
            price_df[symbol] = df['close']
            
        except Exception:
            # Hata durumunda (örneğin veri yoksa) o sembolü atla
            continue

    # 2. Korelasyon Matrisini Hesapla
    price_df.dropna(inplace=True)
    correlation_matrix = price_df.corr(method='pearson')
    
    return correlation_matrix