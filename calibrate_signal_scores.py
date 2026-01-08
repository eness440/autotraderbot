# -*- coding: utf-8 -*-
"""
calibrate_signal_scores.py
-------------------------

Bu script, ``data/signal_dataset.csv`` dosyasından sinyal ve sonuç
verilerini okuyarak iki adet lojistik modeli eğitir:

1. **Skor Kalibrasyonu**: ``master_conf_raw`` değerini gerçek başarı
   olasılığına haritalamak için tek değişkenli bir lojistik regresyon modeli.
   Sonuç ``calibration.json`` dosyasına yazılır ve parametreler ``a``
   (eğim) ve ``b`` (kesişim) olarak kaydedilir. Bu parametreler,
   ``controller_async`` içinde master confidence skorunu kalibre etmek için
   kullanılacaktır.

2. **Ağırlık Öğrenimi**: AI, teknik ve sentiment skorlarının önemini
   belirlemek için çok değişkenli bir lojistik regresyon modeli.
   Sonuç ``logistic_weights.json`` dosyasına yazılır ve model
   parametreleri ``w0`` (bias), ``w_ai``, ``w_tech`` ve ``w_sent``
   katsayıları olarak kaydedilir. Çok uç değerleri önlemek için
   katsayılar belirli aralıklarda sıkıştırılır.

Kullanım:

```
python calibrate_signal_scores.py
```

Bu script kendi kendine çalıştırılabilir ve projenin kök dizininde
bulunan ``data/signal_dataset.csv`` dosyasının varlığını varsayar.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def calibrate(dataset_path: Path, calibration_path: Path, weights_path: Path) -> None:
    """Veri kümesinden kalibrasyon ve ağırlık modellerini öğrenir."""
    if not dataset_path.exists():
        print(f"[calibrate] Veri seti bulunamadı: {dataset_path}")
        return
    df = pd.read_csv(dataset_path)
    # Etiket mevcut olmayan satırları dışla
    if "label" not in df.columns:
        print("[calibrate] Veri seti 'label' kolonunu içermiyor.")
        return
    df = df.dropna(subset=["label", "master_conf_raw", "ai_score", "tech_score", "sent_score"])
    # Veri seti çok küçükse güvenli kalibrasyon ve ağırlıklar yazıp çık
    if df.empty or len(df) < 50:
        print("[calibrate] Veri seti çok küçük; güvenli varsayılan kalibrasyon ve ağırlıklar yazılıyor.")
        calibration = {"type": "logistic", "a": 1.0, "b": 0.0}
        calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
        weights = {"w0": 0.0, "w_ai": 1.0, "w_tech": 1.0, "w_sent": 1.0}
        weights_path.write_text(json.dumps(weights, indent=2), encoding="utf-8")
        return
    # label'i integer yap
    try:
        y = df["label"].astype(int).values
    except Exception:
        y = df["label"].values
    # 1) Kalibrasyon: master_conf_raw → label
    X_cal = df[["master_conf_raw"]].values
    try:
        # Kullan class_weight="balanced" ve regularisation C=0.5
        model_cal = LogisticRegression(class_weight="balanced", C=0.5)
        model_cal.fit(X_cal, y)
        a = float(model_cal.coef_[0][0])
        b = float(model_cal.intercept_[0])
        calibration = {"type": "logistic", "a": a, "b": b}
        calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
        print(f"[calibrate] calibration.json güncellendi: a={a:.4f}, b={b:.4f}")
    except Exception as e:
        print(f"[calibrate] Kalibrasyon eğitimi başarısız: {e}")
    # 2) Ağırlık modeli: ai_score, tech_score, sent_score → label
    X_feat = df[["ai_score", "tech_score", "sent_score"]].values
    try:
        # Kullan class_weight="balanced" ve regularisation C=0.5
        model_feat = LogisticRegression(class_weight="balanced", C=0.5)
        model_feat.fit(X_feat, y)
        w0 = float(model_feat.intercept_[0])
        w_ai = float(model_feat.coef_[0][0])
        w_tech = float(model_feat.coef_[0][1])
        w_sent = float(model_feat.coef_[0][2])
        # Aşırı değerleri sınırla
        def _clamp(val: float, lo: float, hi: float) -> float:
            return lo if val < lo else hi if val > hi else val
        # Bias ±2, feature katsayıları ±1
        w0_clamped = _clamp(w0, -2.0, 2.0)
        w_ai_clamped = _clamp(w_ai, -1.0, 1.0)
        w_tech_clamped = _clamp(w_tech, -1.0, 1.0)
        w_sent_clamped = _clamp(w_sent, -1.0, 1.0)
        weights = {
            "w0": w0_clamped,
            "w_ai": w_ai_clamped,
            "w_tech": w_tech_clamped,
            "w_sent": w_sent_clamped,
        }
        weights_path.write_text(json.dumps(weights, indent=2), encoding="utf-8")
        print(
            f"[calibrate] logistic_weights.json güncellendi: w0={w0_clamped:.4f}, "
            f"w_ai={w_ai_clamped:.4f}, w_tech={w_tech_clamped:.4f}, w_sent={w_sent_clamped:.4f}"
        )
    except Exception as e:
        print(f"[calibrate] Ağırlık modeli eğitimi başarısız: {e}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    dataset_path = base_dir / "data" / "signal_dataset.csv"
    calibration_path = base_dir / "calibration.json"
    weights_path = base_dir / "logistic_weights.json"
    calibrate(dataset_path, calibration_path, weights_path)
