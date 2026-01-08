# gen_okx_symbols.py
# OKX USDT-margined swap (perpetual) sembollerini çekip symbols_okx.json'a yazar.

import json
from pathlib import Path

import ccxt

# Projendeki ayarları kullan
from settings import OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE, OKX_USE_TESTNET


def main():
    if not OKX_API_KEY or not OKX_API_SECRET or not OKX_API_PASSPHRASE:
        raise RuntimeError("OKX API bilgileri eksik. settings.py / .env kontrol et.")

    ex = ccxt.okx({
        "apiKey": OKX_API_KEY,
        "secret": OKX_API_SECRET,
        "password": OKX_API_PASSPHRASE,
        "enableRateLimit": True,
    })
    ex.set_sandbox_mode(bool(OKX_USE_TESTNET))

    markets = ex.load_markets()
    symbols = []

    for sym, m in markets.items():
        # Sadece USDT-quoted, swap (perpetual/futures) ve linear (USDT-margined) olanlar
        try:
            quote = m.get("quote")
            is_swap = bool(m.get("swap")) or m.get("type") == "swap"
            is_linear = bool(m.get("linear"))
            if quote == "USDT" and is_swap and is_linear:
                symbols.append(sym)
        except Exception:
            continue

    # Tekilleştir ve sırala
    symbols = sorted(set(symbols))

    # İstersen ilk 100'le sınırla
    MAX_SYMBOLS = 100
    symbols = symbols[:MAX_SYMBOLS]

    out = {"symbols": symbols}

    out_path = Path("symbols_okx.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{len(symbols)} sembol symbols_okx.json dosyasına yazıldı.")
    for s in symbols:
        print(" -", s)


if __name__ == "__main__":
    main()
