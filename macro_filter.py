# -*- coding: utf-8 -*-
"""
macro_filter.py

Basit bir ekonomik takvim filtresi. Bu modül, data/macro_events.json
dosyasında tanımlanan önemli makro-ekonomik olayları okur ve işlem
zamanı bu olayların hemen öncesine veya sırasında denk geliyorsa
güven skoruna bir risk indirimi uygular. Böylece yüksek volatilite
yapabilecek veri açıklamaları sırasında bot daha temkinli olur.

Örnek macro_events.json formatı:
[
  {
    "name": "FOMC Meeting",
    "start": "2025-11-30T14:00:00Z",
    "end": "2025-11-30T15:00:00Z",
    "multiplier": 0.5,
    "pre_minutes": 60
  },
  ...
]

Alanlar:
  - name: Olayın adı (bilgilendirme amaçlıdır).
  - start: ISO8601 formatında olayın başlangıç zamanı (UTC).
  - end: ISO8601 formatında olayın bitiş zamanı (UTC). Opsiyoneldir; verilmezse
    start ile aynı kabul edilir.
  - multiplier: Olay sırasında uygulanacak risk çarpanı. Örneğin 0.5,
    master confidence'i yarıya indirir.
  - pre_minutes: Olaydan bu kadar dakika önce de aynı çarpan uygulanır.
    Opsiyoneldir; verilmezse varsayılan 30 dakika kullanılır.

Bu modül, olay listesinde sırayla tarama yapar ve eğer şimdiki zaman
(UTC) bir veya birden fazla olayın öncesine veya aktif süresine denk
geliyorsa en küçük çarpanı döndürür. Aksi halde 1.0 döndürülür.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

_events_cache: Optional[List[Dict[str, Any]]] = None

def _load_events() -> List[Dict[str, Any]]:
    """macro_events.json dosyasını okuyup olayları döndür."""
    global _events_cache
    if _events_cache is not None:
        return _events_cache  # type: ignore
    try:
        path = Path(__file__).resolve().parent / "data" / "macro_events.json"
        if path.exists():
            txt = path.read_text(encoding="utf-8").strip()
            if txt:
                data = json.loads(txt)
                if isinstance(data, list):
                    _events_cache = data  # type: ignore
                    return _events_cache  # type: ignore
    except Exception:
        pass
    _events_cache = []
    return _events_cache  # type: ignore

def get_macro_risk_multiplier(now: Optional[datetime] = None) -> float:
    """
    Şu anki zamanın makro olaylara yakınlığına göre bir risk çarpanı döndür.

    - Eğer bir olayın başlamasına ``pre_minutes`` dakikadan az kaldıysa
      veya olay hâlâ sürüyorsa, olayın ``multiplier`` değeri uygulanır.
    - Aynı anda birden fazla olay varsa, en düşük multiplier tercih edilir.
    - Olay tanımlı değilse veya dosya okunamazsa 1.0 döndürülür.

    Args:
        now: Opsiyonel olarak kontrol edilecek zaman (UTC). None ise ``datetime.utcnow()``
             kullanılır.

    Returns:
        0.0–1.0 aralığında bir risk çarpanı. 1.0 = değişiklik yok.
    """
    current = now or datetime.utcnow().replace(tzinfo=timezone.utc)
    events = _load_events()
    if not events:
        return 1.0
    multiplier = 1.0
    active_event_names: List[str] = []
    for ev in events:
        try:
            start_str = ev.get("start")
            if not start_str:
                continue
            # Start & end time
            start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            end_str = ev.get("end") or start_str
            end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            # Pre-event window
            pre_minutes = int(ev.get("pre_minutes", 30))
            pre_window_start = start_dt - timedelta(minutes=pre_minutes)
            # Check if within window
            if pre_window_start <= current <= end_dt:
                m = float(ev.get("multiplier", 0.5))
                # Record active event name if provided
                name = ev.get("name")
                if isinstance(name, str) and name:
                    active_event_names.append(name)
                # Choose the smallest multiplier if multiple events
                if m < multiplier:
                    multiplier = m
        except Exception:
            continue
    # Bound multiplier to [0.2, 1.0]
    if multiplier < 0.2:
        multiplier = 0.2
    if multiplier > 1.0:
        multiplier = 1.0
    # Log when the macro multiplier is less than 1.0 to aid debugging.  We
    # intentionally do not import the global logger here to avoid
    # unnecessary dependencies; instead print a concise message.  Callers
    # can redirect stdout if they wish to suppress these notices.  List
    # up to three active events for brevity.
    if multiplier < 1.0 and active_event_names:
        names_preview = ", ".join(active_event_names[:3])
        print(f"[macro_filter] Active macro events: {names_preview} → multiplier={multiplier:.2f}")
    return multiplier