"""
performance_analyzer.py

Bu script, gün sonunda trade_log.json dosyasını inceleyerek her sembol için
kazanma oranı, ortalama PnL ve R-multiple gibi metrikleri hesaplar. Kötü
performans gösteren semboller için bir "cooldown" süresi belirlenir ve
"data/cooldowns.json" dosyasına yazılır. Bu dosya, bot tarafından yeni
işlem açarken okunarak ilgili semboller geçici olarak atlanabilir.

Kullanım:

    python performance_analyzer.py

Varsayılan parametreler:
  - min_trades_for_analysis: Bir sembolün analiz edilebilmesi için minimum işlem sayısı (10).
  - win_rate_threshold: Kazanma oranı bu değerin altındaysa sembol cooldown'a alınır (0.4 → %40).
  - cooldown_days: Kötü performans halinde sembolün skip edileceği gün sayısı (3).

Bu script manuel olarak veya AutoUpdater tarafından günlük tetiklenebilir.
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

# Dosya yolları
TRADE_LOG_PATH = Path(__file__).resolve().parent / "trade_log.json"
COOLDOWN_PATH = Path(__file__).resolve().parent / "data" / "cooldowns.json"

# Varsayılan parametreler
MIN_TRADES = 10
WIN_RATE_THRESHOLD = 0.40  # %40 altı kazanma oranı → cooldown
COOLDOWN_DAYS = 3

def _load_trade_log() -> list[Dict[str, Any]]:
    """trade_log.json dosyasını list olarak yükler."""
    try:
        if TRADE_LOG_PATH.exists():
            txt = TRADE_LOG_PATH.read_text(encoding="utf-8").strip()
            if txt:
                data = json.loads(txt)
                if isinstance(data, list):
                    return data
    except Exception:
        pass
    return []

def _load_cooldowns() -> Dict[str, str]:
    """Mevcut cooldown dosyasını yükler."""
    try:
        if COOLDOWN_PATH.exists():
            txt = COOLDOWN_PATH.read_text(encoding="utf-8").strip()
            if txt:
                data = json.loads(txt)
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}

def _save_cooldowns(cooldowns: Dict[str, str]):
    """Cooldown verisini dosyaya yazar."""
    try:
        COOLDOWN_PATH.parent.mkdir(parents=True, exist_ok=True)
        COOLDOWN_PATH.write_text(json.dumps(cooldowns, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"Cooldown kaydedilemedi: {e}")

def analyze_and_update_cooldowns():
    """Performans analizini yapar ve cooldown dosyasını günceller."""
    trades = _load_trade_log()
    # Sembolleri grupla
    grouped: Dict[str, list[Dict[str, Any]]] = {}
    for tr in trades:
        sym = str(tr.get("symbol"))
        if not sym:
            continue
        grouped.setdefault(sym, []).append(tr)
    # Mevcut cooldown'ları oku
    cooldowns = _load_cooldowns()
    now = datetime.now(timezone.utc)
    # Eski cooldown'ları süresine göre filtrele (süresi dolmuşsa çıkar)
    to_delete = []
    for sym, until_str in cooldowns.items():
        try:
            until_dt = datetime.fromisoformat(until_str)
            if now >= until_dt:
                to_delete.append(sym)
        except Exception:
            to_delete.append(sym)
    for sym in to_delete:
        cooldowns.pop(sym, None)
    # Performans analizi ve yeni cooldown ekleme
    for sym, lst in grouped.items():
        n = len(lst)
        if n < MIN_TRADES:
            continue
        wins = 0
        for tr in lst:
            try:
                pnl = float(tr.get("pnl_abs") or tr.get("pnl_usd") or 0.0)
                if pnl > 0:
                    wins += 1
            except Exception:
                continue
        win_rate = wins / n if n > 0 else 0.0
        # Kötü performans → cooldown
        if win_rate < WIN_RATE_THRESHOLD:
            until = now + timedelta(days=COOLDOWN_DAYS)
            cooldowns[sym] = until.isoformat()
    # Kaydet
    _save_cooldowns(cooldowns)
    print(f"Cooldown listesi güncellendi: {len(cooldowns)} sembol işaretli.")

if __name__ == "__main__":
    analyze_and_update_cooldowns()