# -*- coding: utf-8 -*-
"""
AutoTraderBot – OHLC Bulk Generator (Fixed: TESTNET/LIVE Auto-Detect + 2000 Candles + Detailed Log)
-------------------------------------------------------------
Bu script birden fazla coin ve zaman diliminden eşzamanlı olarak veri toplar.
FIX: OKX API Rate Limit (429) hatasını önlemek için istekler arasına gecikme ekler.
FIX: 'defaultType': 'swap' ayarı ile Vadeli İşlem çiftlerini yükler.
FIX: 'APIKey does not match' hatası için Testnet/Live ayarını settings.py'dan alır.
UPDATE: Her timeframe için limit 2000 muma çıkarıldı ve loglar detaylandırıldı.
"""

import json
import pathlib
import time
import ccxt
from datetime import datetime
from analyzer import fetch_and_analyze_data
from settings import OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE, OKX_USE_TESTNET

# ──────────────────────────────────────────────
# AYARLAR
# ──────────────────────────────────────────────
TIMEFRAMES = ['5m', '15m', '1h', '4h']
LIMIT = 2000  # [GÜNCELLENDİ] İstenildiği gibi 2000 mum geçmiş veri

# ──────────────────────────────────────────────
# DOSYA YOLLARI
# ──────────────────────────────────────────────
METRICS_DIR = pathlib.Path("metrics")
OHLC_FILE = METRICS_DIR / "ohlc_history.json"
BACKUP_DIR = METRICS_DIR / "backups"
SYMBOLS_FILE = pathlib.Path("symbols_okx.json")

BACKUP_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ──────────────────────────────────────────────
def load_symbols():
    """symbols_okx.json varsa oradan, yoksa default listeden yükler."""
    if SYMBOLS_FILE.exists():
        try:
            content = json.loads(SYMBOLS_FILE.read_text(encoding="utf-8"))
            if isinstance(content, dict) and "symbols" in content:
                return content["symbols"]
            elif isinstance(content, list):
                return content
        except Exception as e:
            print(f"[WARN] symbols_okx.json okunamadı: {e}")
    
    # Fallback liste
    return [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", 
        "ADA/USDT", "AVAX/USDT", "BNB/USDT", "TRX/USDT", "LINK/USDT"
    ]

def backup_ohlc_file():
    if OHLC_FILE.exists():
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_file = BACKUP_DIR / f"ohlc_history_{ts}.json"
        try:
            OHLC_FILE.rename(backup_file)
            print(f"[BACKUP] Eski veri yedeklendi -> {backup_file.name}")
        except OSError:
            pass

def init_exchange():
    """
    CCXT'yi settings.py ayarına göre başlatır.
    """
    print("="*40)
    print(f"[INIT] Bağlantı Modu: {'TESTNET (DEMO)' if OKX_USE_TESTNET else 'LIVE (GERÇEK)'}")
    print(f"[INIT] Hedef Mum Sayısı (Her TF): {LIMIT}")
    print("="*40)

    exchange = ccxt.okx({
        'apiKey': OKX_API_KEY,
        'secret': OKX_API_SECRET,
        'password': OKX_API_PASSPHRASE,
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'} # Vadeli işlem verisi çek
    })
    
    # Testnet ise sandbox modunu aç (Kritik Nokta)
    if OKX_USE_TESTNET:
        exchange.set_sandbox_mode(True)
    
    try:
        exchange.load_markets()
    except Exception as e:
        print(f"[ERR] Marketler yüklenemedi: {e}")
        
    return exchange

# ──────────────────────────────────────────────
# ANA İŞLEM
# ──────────────────────────────────────────────
def main():
    print("==== OHLC Bulk Data Generator ====")
    
    # 1. Hazırlık
    backup_ohlc_file()
    exchange = init_exchange()
    target_symbols = load_symbols()
    
    print(f"[INFO] Hedef Sembol Sayısı: {len(target_symbols)}")
    
    all_data = []
    
    # 2. Veri Toplama Döngüsü
    for i, symbol in enumerate(target_symbols):
        print(f"[{i+1}/{len(target_symbols)}] İşleniyor: {symbol} ... ", end="", flush=True)
        
        symbol_data_count = 0
        tf_stats = [] # Log için istatistik tutucu
        
        for tf in TIMEFRAMES:
            try:
                # Rate Limit Koruması
                time.sleep(0.2) 
                
                # Veriyi çek (LIMIT 2000 olarak gidiyor)
                df = fetch_and_analyze_data(exchange, symbol, timeframe=tf, limit=LIMIT)
                
                if df is None or df.empty:
                    tf_stats.append(f"{tf}:0")
                    continue
                
                count = len(df)
                tf_stats.append(f"{tf}:{count}") # Örn: 5m:2000

                # Listeye ekle
                for ts, row in df.iterrows():
                    record = {
                        "symbol": symbol.replace("/", ""),
                        "ts": ts.isoformat(),
                        "open": float(row['open']),
                        "high": float(row['high']),
                        "low": float(row['low']),
                        "close": float(row['close']),
                        "volume": float(row['volume'])
                    }
                    all_data.append(record)
                    symbol_data_count += 1
                    
            except Exception as e:
                tf_stats.append(f"{tf}:ERR")
                pass
        
        # LOGLAMA (İstediğin detay burada)
        stats_str = ", ".join(tf_stats)
        if symbol_data_count > 0:
            print(f"✅ Toplam: {symbol_data_count} | Detay: [{stats_str}]")
        else:
            print("❌ (Veri yok)")
            
        # Her 10 coinde bir ekstra bekleme
        if (i + 1) % 10 == 0:
            time.sleep(1)

    # 3. Kaydetme
    if all_data:
        print(f"\n[SAVE] Toplam {len(all_data)} satır veri kaydediliyor...")
        try:
            with open(OHLC_FILE, "w", encoding="utf-8") as f:
                json.dump({"rows": all_data}, f, indent=0) 
            print(f"✅ Başarılı: {OHLC_FILE}")
            print("👉 Şimdi 'python build_dataset.py --train' komutunu çalıştırabilirsin.")
        except Exception as e:
            print(f"❌ Kayıt hatası: {e}")
    else:
        print("⚠️ Hiç veri toplanamadı.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[EXIT] Çıkış yapıldı.")