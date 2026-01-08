"""
session_filter.py

Basit bir seans (session) filtresi. Bu modül, günün saatine bağlı olarak
farklı risk katsayıları ve işlem izinleri tanımlar. Kullanıcı, `data/session_config.json`
dosyasında seansların başlangıç/bitiş saatlerini ve risk çarpanlarını
belirleyebilir. Bu sayede Asya, Londra ve New York gibi farklı seanslarda
bot davranışı (örneğin kaldıraç veya işlem sıklığı) otomatik olarak
ayarlanabilir.

Konfigürasyon formatı (JSON listesi):
[
  {
    "name": "asia",
    "start": "00:00",
    "end": "07:59",
    "risk_multiplier": 0.6,
    "enabled": true
  },
  ...
]

Saatler HH:MM formatında ve 24 saat diliminde girilmelidir. Modül,
yerel sistem saatini esas alır. Eğer farklı bir zaman dilimi kullanmak
istiyorsanız, seansları UTC'ye göre tanımlamanız önerilir.
"""
from __future__ import annotations

import json
from datetime import datetime, time
from pathlib import Path
from typing import Optional, Dict, Any, List

DATA_DIR = Path(__file__).resolve().parent / "data"
_SESSION_FILE = DATA_DIR / "session_config.json"
_sessions_cache: Optional[List[Dict[str, Any]]] = None

def _load_sessions() -> List[Dict[str, Any]]:
    """Konfigürasyon dosyasından seans tanımlarını yükler."""
    global _sessions_cache
    if _sessions_cache is None:
        sessions: List[Dict[str, Any]] = []
        try:
            if _SESSION_FILE.exists():
                txt = _SESSION_FILE.read_text(encoding="utf-8").strip()
                if txt:
                    data = json.loads(txt)
                    if isinstance(data, list):
                        for item in data:
                            if not isinstance(item, dict):
                                continue
                            start_str = str(item.get("start", "00:00")).strip()
                            end_str = str(item.get("end", "23:59")).strip()
                            try:
                                start_parts = start_str.split(":")
                                end_parts = end_str.split(":")
                                start_time = time(hour=int(start_parts[0]), minute=int(start_parts[1]))
                                end_time = time(hour=int(end_parts[0]), minute=int(end_parts[1]))
                            except Exception:
                                continue
                            sessions.append({
                                "name": str(item.get("name", "session")),
                                "start": start_time,
                                "end": end_time,
                                "risk_multiplier": float(item.get("risk_multiplier", 1.0)),
                                "enabled": bool(item.get("enabled", True))
                            })
        except Exception:
            sessions = []
        _sessions_cache = sessions
    return _sessions_cache or []

def _is_time_in_range(t: time, start: time, end: time) -> bool:
    """Verilen zamanın [start, end] aralığında olup olmadığını kontrol eder."""
    if start <= end:
        return start <= t <= end
    # Aralık gece yarısını aşıyorsa
    return t >= start or t <= end

def get_current_session(now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    """
    Şu anki saate göre aktif seansı döndürür. Eğer hiçbir seans eşleşmiyorsa
    None döner.

    Args:
        now: Opsiyonel olarak özel bir datetime objesi. None ise sistem
             saatini kullanır.

    Returns:
        Aktif seans dict veya None.
    """
    sessions = _load_sessions()
    now_dt = now or datetime.now()
    current_time = now_dt.time()
    for sess in sessions:
        if _is_time_in_range(current_time, sess["start"], sess["end"]):
            return sess
    return None

def get_risk_multiplier(now: Optional[datetime] = None) -> float:
    """Aktif seansın risk çarpanını döndürür. Yoksa 1.0."""
    sess = get_current_session(now)
    if sess and sess.get("enabled", True):
        try:
            return float(sess.get("risk_multiplier", 1.0))
        except Exception:
            return 1.0
    return 1.0

def is_trading_enabled(now: Optional[datetime] = None) -> bool:
    """Aktif seans işlem yapmaya izin veriyor mu?"""
    sess = get_current_session(now)
    if sess is None:
        return True
    return bool(sess.get("enabled", True))