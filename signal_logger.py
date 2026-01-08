# -*- coding: utf-8 -*-
"""
signal_logger.py
-----------------

Bu modül, her potansiyel işlem (enter sinyali) için özellikleri ve
sonradan etiketlenecek sonucu bir CSV dosyasına yazar. Amaç, master
confidence skorunun ve diğer sinyal bileşenlerinin gerçek başarı
oranlarıyla istatistiksel olarak kalibre edilmesini sağlayacak bir
veri seti toplamaktır. Kayıt formatı oldukça basittir ve zaman
damgası, sembol, zaman dilimi (tf), ham master skoru (kalibrasyondan
önce), AI/teknik/sentiment/RL skorları, temel yön (base_decision),
beklenen giriş fiyatı, piyasa rejimi ve son olarak etiket (label)
içerir. Etiket 1=kazanç, 0=kayıp olarak tanımlanır ve başlangıçta
``None`` olarak yazılır; trade kapandıktan sonra ``update_label``
fonksiyonu ile güncellenmelidir.

Dosya ``data/signal_dataset.csv`` konumunda tutulur. İlk satırda
başlıklar yer alır. Yeni kayıtlar append modunda eklenir. Bir trade
ile sinyal kaydı arasında bağlantı kurmak için benzersiz bir
``signal_id`` kullanılır; bu id, zaman damgası ve sembolün birleşiminden
oluşturulur ve trade kapandığında bu id üzerinden etiket güncellenir.

Örnek satır:

```
signal_id,timestamp,symbol,tf,master_conf_raw,ai_score,tech_score,sent_score,rl_score,base_decision,price_entry_planned,regime,label
2025-11-30T04:03:10_BTC/USDT,2025-11-30T04:03:10,BTC/USDT,5m,0.6821,0.71,0.95,0.78,0.50,short,38350.2,BULL,
```

Bu modül, concurrency açısından basit tutulmuştur; dosya yazarken
kilitleme kullanılmaz. Yoğun bir üretim ortamında dosya tabanlı
kilitleme veya veritabanı entegrasyonu eklemek gerekebilir.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional


DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Sinyal veri seti dosya yolu
SIGNAL_DATASET_FILE = DATA_DIR / "signal_dataset.csv"


def _ensure_header() -> None:
    """Dosya yoksa başlık satırını yazar."""
    if not SIGNAL_DATASET_FILE.exists():
        with SIGNAL_DATASET_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "signal_id", "timestamp", "symbol", "tf", "master_conf_raw",
                "ai_score", "tech_score", "sent_score", "rl_score", "base_decision",
                "price_entry_planned", "regime", "label",
            ])


def log_signal(
    symbol: str,
    tf: str,
    master_conf_raw: float,
    ai_score: float,
    tech_score: float,
    sent_score: float,
    rl_score: float,
    base_decision: str,
    price_entry_planned: Optional[float],
    regime: Optional[str],
    timestamp: Optional[datetime] = None,
) -> str:
    """
    Yeni bir sinyal kaydı ekler ve benzersiz ``signal_id`` döndürür.

    Args:
        symbol: İşlem yapılan sembol (örn. "BTC/USDT").
        tf: Zaman dilimi (örn. "5m", "15m").
        master_conf_raw: Kalibrasyondan önceki master güven skoru (0..1).
        ai_score: AI modelinden gelen skor (0..1).
        tech_score: Teknik analiz skoru (0..1).
        sent_score: Sentiment skoru (0..1).
        rl_score: Takviye öğrenme skorunun 0..1 aralığındaki değeri.
        base_decision: "long" veya "short" gibi temel karar.
        price_entry_planned: İşlem için planlanan giriş fiyatı.
        regime: Piyasa rejimi ("BULL", "BEAR", "SIDEWAYS" veya None).
        timestamp: Opsiyonel, sinyal zamanı. None ise UTC now kullanılır.

    Returns:
        signal_id: Kaydın benzersiz kimliği (``{timestamp_iso}_{symbol}``).
    """
    _ensure_header()
    ts = timestamp or datetime.utcnow()
    ts_iso = ts.replace(microsecond=0).isoformat()
    signal_id = f"{ts_iso}_{symbol}"
    row = [
        signal_id,
        ts_iso,
        symbol,
        tf,
        round(float(master_conf_raw), 6) if master_conf_raw is not None else None,
        round(float(ai_score), 6) if ai_score is not None else None,
        round(float(tech_score), 6) if tech_score is not None else None,
        round(float(sent_score), 6) if sent_score is not None else None,
        round(float(rl_score), 6) if rl_score is not None else None,
        base_decision,
        round(float(price_entry_planned), 6) if price_entry_planned is not None else None,
        regime,
        None,
    ]
    with SIGNAL_DATASET_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    return signal_id


def update_label(signal_id: str, label: int) -> None:
    """
    Varolan bir sinyal kaydının etiketini (label) günceller.

    Bu fonksiyon, trade kapandığında kazanç/kayıp durumuna göre çağrılmalı
    ve ilgili sinyalin ``label`` alanına 1 (kazanç) veya 0 (kayıp) yazmalıdır.
    ``signal_id`` eşleşen ilk kayıt güncellenir; birden fazla eşleşme
    beklenmez.

    Args:
        signal_id: log_signal tarafından döndürülen benzersiz kimlik.
        label: 1 (kazanç) veya 0 (kayıp) değeri.
    """
    if label not in (0, 1):
        return
    if not SIGNAL_DATASET_FILE.exists():
        return
    rows = []
    updated = False
    with SIGNAL_DATASET_FILE.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for r in reader:
            rows.append(r)
    # Header varsayım
    headers = rows[0] if rows else []
    out_rows = [headers]
    for r in rows[1:]:
        if r and r[0] == signal_id and not updated:
            # label sütunu son indeks
            r[-1] = str(int(label))
            updated = True
        out_rows.append(r)
    if updated:
        with SIGNAL_DATASET_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(out_rows)
    # Güncellenmediyse sessizce devam edilir

    # Otomatik kalibrasyon: etiket güncellendikten sonra lojistik modelleri yeniden eğit
    # Bu çağrı küçük veri setlerinde kısa sürede tamamlanır. Veri seti çok küçükse
    # veya yeterli satır yoksa calibrate fonksiyonu sessizce çıkacaktır.
    try:
        # calibrate_signal_scores modülünden calibrate fonksiyonunu içe aktar
        from .calibrate_signal_scores import calibrate as _auto_calibrate
        # Kalibrasyon dosya yollarını hazırla
        base_dir = Path(__file__).resolve().parent
        dataset_path = base_dir / "data" / "signal_dataset.csv"
        calibration_path = base_dir / "calibration.json"
        weights_path = base_dir / "logistic_weights.json"
        # Veri setini yüklemeden önce varlık ve satır sayısı kontrolü
        if dataset_path.exists():
            # Okumadan satır sayısını belirlemek için sadece satırları say
            # İlk satır header olduğundan >=21 satır veriye karşılık gelir
            line_count = sum(1 for _ in dataset_path.open("r", encoding="utf-8"))
            if line_count >= 21:
                # Yeterli veri varsa kalibrasyonu çalıştır
                _auto_calibrate(dataset_path, calibration_path, weights_path)
    except Exception:
        # Kalibrasyon çağrısı başarısızsa hatayı yut; manuel çalıştırılabilir
        pass
