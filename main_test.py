import ccxt
import time
import json
import os
from datetime import datetime
from controller import decide_for_symbol
from analyzer import get_multi_timeframe_analysis
from portfolio_tools import calculate_correlation

CONFIG_FILE = 'config.json'
TARGET_SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT']
MIN_CONFIDENCE_FOR_TRADE = 0.65
MAX_LEVERAGE = 75
MIN_LEVERAGE = 10

def get_dynamic_leverage(gscore):
    if gscore < 65:
        return 0
    elif gscore >= 100:
        return MAX_LEVERAGE
    else:
        ratio = (gscore - 65) / (100 - 65)
        leverage = MIN_LEVERAGE + ratio * (MAX_LEVERAGE - MIN_LEVERAGE)
        return round(leverage)

def initialize_exchange():
    try:
        with open(CONFIG_FILE, 'r') as f:
            cfg = json.load(f)
        config = cfg.get('okx_api', {})
        use_testnet = cfg.get('use_testnet', True)
    except Exception as e:
        print(f"HATA: config.json yüklenemedi: {e}")
        return None

    exchange = ccxt.okx({
        'apiKey': config.get('api_key', ''),
        'secret': config.get('secret', ''),
        'password': config.get('password', ''),
        'enableRateLimit': True,
    })
    try:
        exchange.set_sandbox_mode(use_testnet)
        exchange.load_markets()
        print(f"OKX bağlantısı başarılı ({'Demo' if use_testnet else 'Canlı'}).")
        return exchange
    except Exception as e:
        print(f"HATA: OKX bağlantısı başarısız: {e}")
        return None

def simulate_decision_cycle(exchange):
    print("\n=== TEST MODU: TİCARET SİMÜLASYONU ===\n")
    corr_matrix = calculate_correlation(exchange, TARGET_SYMBOLS)
    print("Korelasyon Matrisi hesaplandı:", corr_matrix.shape)

    for symbol in TARGET_SYMBOLS:
        try:
            ta_data = get_multi_timeframe_analysis(exchange, symbol)
            df = ta_data.get("15m")
            if df is None or df.empty:
                continue

            last = df.iloc[-1]
            ta_pack = {
                "price": float(last["close_15m"]),
                "ema": {"fast": float(last["EMA_20_15m"]), "slow": float(last["EMA_50_15m"])},
                "rsi": float(last["RSI_15m"]),
                "adx": float(last["ADX_15m"]),
                "atr": float(last["ATR_15m"]),
                "trend": {
                    "h1_macd_hist": float(last.get("MACD_Hist_1h", 0)),
                    "h4_macd_hist": float(last.get("MACD_Hist_4h", 0)),
                },
                "base_decision": "long" if last["EMA_20_15m"] > last["EMA_50_15m"] else "short"
            }

            senti = {
                "funding": -0.001,
                "oi_change": 0.02,
                "fear_greed": 40
            }

            decision = decide_for_symbol(symbol, "15m", ta_pack, senti, gscore_seed=0)
            master_conf = round(decision.get("master_confidence", 0) * 100, 2)
            action = decision.get("action", "skip")
            reason = decision.get("reason", "")

            if master_conf < 65:
                print(f"{symbol}: ❌ Skip (Güven {master_conf}%) | {reason}")
                continue

            lev = get_dynamic_leverage(master_conf)
            print(f"{symbol}: ✅ {action.upper()} | Güven: {master_conf}% | Kaldıraç: {lev}x | Sebep: {reason}")

        except Exception as e:
            print(f"HATA: {symbol} analiz/karar alınamadı: {e}")
            continue

    print("\n=== TEST TAMAMLANDI ===")

if __name__ == '__main__':
    okx = initialize_exchange()
    if okx:
        simulate_decision_cycle(okx)
    else:
        print("Bot başlatılamadı.")
