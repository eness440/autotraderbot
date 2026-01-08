import ccxt
import requests
import json
import time
from pathlib import Path

"""
This module collects various sentiment data sources used by the trading bot.
In addition to OKX funding and open interest data and the Fear & Greed index,
this file now includes a social sentiment component. The social sentiment score
is computed from a local metrics file (metrics/social_sentiment.json) which
aggregates signals such as Twitter mentions, Reddit threads and news sentiment.
Values in that file should be in the range [-1, 1] where -1 is very bearish
and +1 is very bullish. These values are mapped to [0, 1] internally. If the
file is missing or malformed the social sentiment contribution defaults to
neutral (0.5).
"""

# Path to the social sentiment metrics file. If you add your own data
# pipeline for social media sentiment, write the results into this file
# using keys 'tweet_sentiment', 'reddit_sentiment' and 'news_sentiment'.
SOCIAL_SENTIMENT_FILE = Path("metrics/social_sentiment.json")

def get_social_sentiment() -> dict:
    """
    Reads social sentiment values from the metrics/social_sentiment.json file.
    The file is expected to contain keys like 'tweet_sentiment',
    'reddit_sentiment' and 'news_sentiment' with values in the range [-1, 1].
    Returns an empty dict if the file does not exist or cannot be parsed.
    """
    try:
        if SOCIAL_SENTIMENT_FILE.exists():
            txt = SOCIAL_SENTIMENT_FILE.read_text(encoding="utf-8").strip()
            if not txt:
                return {}
            data = json.loads(txt)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}

def _compute_social_score(data: dict) -> float:
    """
    Converts raw social sentiment values into a 0–1 score.
    Accepts a dictionary with sentiment values keyed by 'tweet_sentiment',
    'reddit_sentiment' and 'news_sentiment'. Each value should be in [-1, 1].
    Values outside that range are clamped. The returned score is the
    arithmetic mean of the mapped values (bearish -1 maps to 0.0, bullish +1
    maps to 1.0). If no valid values are present, returns 0.5 (neutral).
    """
    if not isinstance(data, dict):
        return 0.5
    scores = []
    for key in ("tweet_sentiment", "reddit_sentiment", "news_sentiment"):
        v = data.get(key)
        try:
            f = float(v)
            if f > 1.0:
                f = 1.0
            elif f < -1.0:
                f = -1.0
            scores.append((f + 1.0) / 2.0)
        except Exception:
            continue
    if scores:
        avg = sum(scores) / len(scores)
        # Clamp to [0,1]
        if avg < 0.0:
            avg = 0.0
        if avg > 1.0:
            avg = 1.0
        return avg
    return 0.5

# --- 1. OKX API Verileri ---

def get_okx_funding_and_position_data(exchange, symbol='BTC/USDT'):
    """
    OKX'ten Fonlama Oranını ve Açık Pozisyonların büyüklüğünü çeker.
    """
    sentiment_data = {}
    
    # DÜZELTME: OKX Swap piyasası için doğru sembol formatını oluştur.
    # ccxt'nin birleşik formatı 'BASE/QUOTE:QUOTE' şeklindedir (örn: 'BTC/USDT:USDT')
    swap_symbol = symbol
    if ':' not in symbol and '/USDT' in symbol:
        # Gelen sembol 'BTC/USDT' ise, onu 'BTC/USDT:USDT' formatına çevir.
        swap_symbol = symbol.replace('/USDT', '/USDT:USDT')
    
    try:
        # a) Fonlama Oranı (Funding Rate) Çekme - Düzeltilmiş sembol ile
        funding_rate_data = exchange.fetch_funding_rate(swap_symbol)
        sentiment_data['funding_rate'] = funding_rate_data.get('fundingRate')
        
    except Exception as e:
        sentiment_data['funding_rate'] = None
        # Hata mesajını daha anlaşılır hale getirelim
        if 'swap markets' in str(e):
             print(f"OKX Fonlama Oranı ({swap_symbol}) çekilemedi: Bu API anahtarı Swap piyasası için yetkili olmayabilir.")
        else:
            print(f"OKX Fonlama Oranı ({swap_symbol}) çekilemedi: {e}")


    # b) Açık Pozisyon Büyürlüğü (Open Interest) Çekme - Düzeltilmiş sembol ile
    try:
        open_interest_data = exchange.fetch_open_interest(swap_symbol)
        
        if open_interest_data:
            sentiment_data['open_interest'] = open_interest_data.get('openInterestAmount')
        else:
            sentiment_data['open_interest'] = None
            
    except Exception as e:
        sentiment_data['open_interest'] = None
        if 'contract markets' in str(e):
            print(f"OKX Açık Pozisyon ({swap_symbol}) çekilemedi: Bu API anahtarı Kontrat piyasası için yetkili olmayabilir.")
        else:
            print(f"OKX Açık Pozisyon ({swap_symbol}) çekilemedi: {e}")
        
    return sentiment_data

# --- 2. Harici Sentiment Verileri ---
def get_fear_greed_index():
    """
    Alternative.me API'sini kullanarak Korku ve Açgözlülük Endeksi'ni çeker.
    """
    URL = "https://api.alternative.me/fng/?limit=1"
    
    try:
        response = requests.get(URL, timeout=10)
        response.raise_for_status() # HTTP hatalarını yakala
        data = response.json()
        
        if data and 'data' in data and data['data']:
            return {
                'fng_value': int(data['data'][0]['value']),
                'fng_class': data['data'][0]['value_classification']
            }
            
    except requests.exceptions.RequestException as e:
        print(f"Fear & Greed Index API Hatası: {e}")
        return {'fng_value': 50, 'fng_class': 'Neutral (Default)'} # Hata durumunda nötr değer döndür

# --- 3. Ana Fonksiyon (Tüm Verileri Toplama) ---
def get_combined_sentiment_data(exchange, symbol):
    """
    Gather sentiment data from multiple sources and merge into a single dict.

    This function now aggregates:
      * OKX funding rate and open interest via get_okx_funding_and_position_data.
      * Fear & Greed Index via get_fear_greed_index.
      * Social sentiment via get_social_sentiment (tweet, reddit and news scores).

    Social sentiment values should be in [-1, 1] range in metrics/social_sentiment.json.
    They are mapped to a combined 'social_score' in [0, 1]. If the file is
    missing, the social score defaults to 0.5 and individual keys are None.
    A placeholder 'oi_change' key is also included for future open interest delta.
    """
    print("-> Piyasa Duyarlılığı (Sentiment) verileri çekiliyor...")

    okx_sentiment = get_okx_funding_and_position_data(exchange, symbol)
    fng_sentiment = get_fear_greed_index() or {}
    social_raw = get_social_sentiment()
    social_score = _compute_social_score(social_raw)

    # Merge all sources into a single dict.  If the Fear & Greed API returns
    # 'fng_value' we also set a legacy 'fear_greed' key for backward
    # compatibility with sentiment scoring logic.
    combined_sentiment = {
        'timestamp': int(time.time()),
        'symbol': symbol,
        **(okx_sentiment or {}),
        **fng_sentiment,
        'tweet_sentiment': social_raw.get('tweet_sentiment') if isinstance(social_raw, dict) else None,
        'reddit_sentiment': social_raw.get('reddit_sentiment') if isinstance(social_raw, dict) else None,
        'news_sentiment': social_raw.get('news_sentiment') if isinstance(social_raw, dict) else None,
        'social_score': social_score,
        'oi_change': None,
    }
    # If a fear & greed value exists under fng_value, copy it to fear_greed
    try:
        if 'fng_value' in combined_sentiment and 'fear_greed' not in combined_sentiment:
            fv = combined_sentiment['fng_value']
            # Ensure numeric
            fv_float = float(fv)
            combined_sentiment['fear_greed'] = fv_float
    except Exception:
        pass
    return combined_sentiment