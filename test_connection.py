# test_connection.py
# Bu dosya sadece bağlantıyı test eder.

import ccxt
import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

API_KEY = os.getenv("OKX_API_KEY")
SECRET = os.getenv("OKX_API_SECRET")
PASSPHRASE = os.getenv("OKX_API_PASSPHRASE")
USE_TESTNET = os.getenv("OKX_USE_TESTNET", "False").lower() in ("true", "1", "yes")

print("="*40)
print(f"AYAR KONTROLÜ:")
print(f"• API Key Mevcut mu?: {'EVET' if API_KEY else 'HAYIR'}")
print(f"• Mod: {'TESTNET (DEMO PARASI)' if USE_TESTNET else 'LIVE (GERÇEK PARA)'}")
print("="*40)

if not API_KEY:
    print("❌ HATA: .env dosyasında OKX_API_KEY bulunamadı!")
    exit()

try:
    # Borsa nesnesini oluştur
    exchange = ccxt.okx({
        'apiKey': API_KEY,
        'secret': SECRET,
        'password': PASSPHRASE,
        'options': {'defaultType': 'swap'} # Vadeli işlem modu
    })
    
    # Sandbox modunu ayara göre aç/kapat
    exchange.set_sandbox_mode(USE_TESTNET)

    print("\n1. BAĞLANTI DENENİYOR...")
    # Marketleri yükle (İlk temas)
    exchange.load_markets()
    print("✅ Bağlantı Başarılı! Marketler yüklendi.")

    print("\n2. BAKİYE KONTROLÜ (Yetki Testi)...")
    balance = exchange.fetch_balance()
    usdt = balance['total'].get('USDT', 0)
    print(f"✅ Giriş Başarılı! Cüzdandaki USDT: {usdt}")

    print("\n3. VERİ ÇEKME TESTİ (BTC-USDT-SWAP)...")
    ticker = exchange.fetch_ticker('BTC-USDT-SWAP')
    print(f"✅ Veri Geliyor! BTC Fiyatı: {ticker['last']}")

    print("\n🎉 SONUÇ: Ayarların DOĞRU. Bot çalışabilir.")

except ccxt.AuthenticationError as e:
    print("\n⛔ KİMLİK DOĞRULAMA HATASI!")
    print("Muhtemel Sebepler:")
    print("1. .env dosyasındaki API Key, Secret veya Passphrase yanlış kopyalanmış.")
    print("2. Demo anahtarı ile Gerçek sunucuya bağlanmaya çalışıyorsun (veya tam tersi).")
    print(f"Borsa Hatası: {e}")

except ccxt.NetworkError as e:
    print("\n⛔ AĞ/İNTERNET HATASI!")
    print("İnternet bağlantını veya VPN durumunu kontrol et.")
    print(f"Hata: {e}")

except Exception as e:
    print("\n⛔ BİLİNMEYEN HATA!")
    print(f"Hata Detayı: {e}")