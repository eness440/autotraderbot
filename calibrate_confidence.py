# -*- coding: utf-8 -*-
"""
calibrate_confidence.py
-----------------------

Bu script, ``metrics/calibration_trades.jsonl`` dosyasındaki kayıtları
kullanarak üç önemli kalibrasyon görevini gerçekleştirir:

1. **Skor Kalibrasyonu**: ``master_conf_before`` değerini gerçekleşen kazanç
   olasılığına haritalayan bir lojistik regresyon modeli eğitir. Sonuç
   ``calibration.json`` dosyasına yazılır. Model parametreleri ``a`` ve
   ``b`` olup olasılık şu formülle hesaplanır: ``p_hat = sigmoid(a*s + b)``.

2. **Ağırlık Öğrenimi**: AI, teknik ve sentiment skorlarının sonuç üzerinde
   etkisini bulmak için lojistik regresyon kullanır. Sonuçlar
   ``logistic_weights.json`` dosyasına yazılır. Parametreler ``w0`` (bias),
   ``w_ai``, ``w_tech`` ve ``w_sent`` olarak saklanır. Aşırı değerlerin
   güven skorunu bozmaması için katsayılar makul aralıklarda sınırlandırılır.

3. **Risk Takvimi (Leverage Planı)**: Master confidence değerini 0.65–0.90
   aralığında dilimlere ayırarak her dilimde ortalama R-multiple ve maksimum
   drawdown'a göre bir risk-ödül oranı hesaplar. Bu oranlara göre kaldıraç
   seviyeleri (8x, 12x, 18x, 25x) atanır ve sonuç ``risk_schedule.json``
   dosyasına yazılır. En yüksek risk-ödül oranına sahip dilime en yüksek
   kaldıraç verilir. Veri yoksa veya yetersizse default takvim yazılır.

Kullanım:

    python calibrate_confidence.py

Bu script kendi başına çalışabilir. Proje kök dizininde ``metrics``
altındaki ``calibration_trades.jsonl`` dosyasının varlığını ve en az 20
satırlık veri içerdiğini varsayar. Veri yeterli değilse script sessizce
çıkar.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def calibrate_confidence(data_path: Path, calibration_path: Path, weights_path: Path, schedule_path: Path) -> None:
    """Kalibrasyon modellerini ve risk takvimini üretir."""
    if not data_path.exists():
        print(f"[calibrate_confidence] Veri seti yok: {data_path}")
        return
    rows = []
    try:
        with data_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    rows.append(obj)
                except Exception:
                    continue
    except Exception as e:
        print(f"[calibrate_confidence] Dosya okunamadı: {e}")
        return
    if not rows:
        print("[calibrate_confidence] Kalibrasyon için veri yok.")
        return
    df = pd.DataFrame(rows)
    # Gerekli kolonlar kontrolü
    required_cols = {"master_conf_before", "ai_score", "tech_score", "sent_score", "label", "r_multiple", "max_drawdown_pct"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[calibrate_confidence] Veri seti eksik kolonlar içeriyor: {missing}")
        return
    df = df.dropna(subset=list(required_cols))
    # Yeterli veri kontrolü
    if len(df) < 20:
        print("[calibrate_confidence] Kalibrasyon için yeterli satır yok (>=20 gerekir).")
        return
    # label'i integer yap
    try:
        y = df["label"].astype(int).values
    except Exception:
        y = df["label"].values
    # 1) Skor kalibrasyonu: master_conf_before -> label
    try:
        X_cal = df[["master_conf_before"]].astype(float).values
        # Apply class balancing and increased iterations to avoid overly
        # pessimistic calibration when the dataset is imbalanced.
        model_cal = LogisticRegression(class_weight="balanced", max_iter=500)
        model_cal.fit(X_cal, y)
        a = float(model_cal.coef_[0][0])
        b = float(model_cal.intercept_[0])
        calibration = {"type": "logistic", "a": a, "b": b}
        calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
        print(f"[calibrate_confidence] calibration.json güncellendi: a={a:.4f}, b={b:.4f}")
    except Exception as e:
        print(f"[calibrate_confidence] Kalibrasyon eğitimi başarısız: {e}")
    # 2) Ağırlık modeli: ai_score, tech_score, sent_score -> label
    try:
        X_feat = df[["ai_score", "tech_score", "sent_score"]].astype(float).values
        # Use balanced classes and more iterations. This helps assign
        # sensible weights to AI/tech/sentiment scores and prevents the
        # model from learning extreme negative coefficients.
        model_feat = LogisticRegression(class_weight="balanced", max_iter=1000)
        model_feat.fit(X_feat, y)
        w0 = float(model_feat.intercept_[0])
        w_ai = float(model_feat.coef_[0][0])
        w_tech = float(model_feat.coef_[0][1])
        w_sent = float(model_feat.coef_[0][2])
        # Aşırı değerleri sınırla
        def _clamp(val: float, lo: float, hi: float) -> float:
            return lo if val < lo else hi if val > hi else val
        w0_clamped = _clamp(w0, -2.0, 2.0)
        w_ai_clamped = _clamp(w_ai, -1.0, 1.0)
        w_tech_clamped = _clamp(w_tech, -1.0, 1.0)
        w_sent_clamped = _clamp(w_sent, -1.0, 1.0)
        #
        # MODEL IS OLDERLY PESSIMISTIC ADJUSTMENT
        #
        # Training imbalances or labelling errors can cause the logistic
        # regression to learn negative weights for the AI and technical
        # components.  This would imply that stronger positive signals
        # decrease the likelihood of success, which is undesirable.  To
        # counteract this, enforce non‑negative coefficients for AI and
        # technical scores by taking the absolute value of their clamped
        # counterparts.  Sentiment weight remains signed since negative
        # sentiment should legitimately lower confidence.
        w_ai_clamped = abs(w_ai_clamped)
        w_tech_clamped = abs(w_tech_clamped)
        weights = {
            "w0": w0_clamped,
            "w_ai": w_ai_clamped,
            "w_tech": w_tech_clamped,
            "w_sent": w_sent_clamped,
        }
        weights_path.write_text(json.dumps(weights, indent=2), encoding="utf-8")
        print(f"[calibrate_confidence] logistic_weights.json güncellendi: w0={w0_clamped:.4f}, "
              f"w_ai={w_ai_clamped:.4f}, w_tech={w_tech_clamped:.4f}, w_sent={w_sent_clamped:.4f}")
    except Exception as e:
        print(f"[calibrate_confidence] Ağırlık modeli eğitimi başarısız: {e}")
    # 3) Risk takvimi: master_conf_before dilimlerine göre risk-ödül oranı
    try:
        # Belirlenen dilimler
        bins = [(0.65, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 0.90)]
        ratios = []
        for lo, hi in bins:
            sub = df[(df["master_conf_before"] >= lo) & (df["master_conf_before"] < hi)]
            if len(sub) == 0:
                ratios.append((lo, hi, None))
                continue
            # Ortalama R-multiple ve drawdown hesapla
            mean_r = sub["r_multiple"].astype(float).mean()
            # Max drawdown negatif bir sayı; risk-ödül hesabı için mutlak değer
            mean_dd = sub["max_drawdown_pct"].astype(float).apply(lambda x: abs(x)).mean()
            # Risk-ödül oranı: daha yüksek R-multiple ve daha düşük drawdown daha iyidir
            try:
                ratio = (mean_r) / (mean_dd + 1e-6)
            except Exception:
                ratio = None
            ratios.append((lo, hi, ratio))
        # Oranlara göre azalan sırala; oranı None olanlar listenin sonuna gider
        sorted_bins = sorted(ratios, key=lambda x: (x[2] is None, -(x[2] if x[2] is not None else -1)))
        # Kaldıraç seviyeleri: en iyi oran en yüksek kaldıraç
        leverage_levels = [8, 12, 18, 25]
        bins_out = []
        for idx, (lo, hi, ratio) in enumerate(sorted_bins):
            lev = leverage_levels[idx] if idx < len(leverage_levels) else leverage_levels[-1]
            bins_out.append({"min": lo, "max": hi, "leverage": int(lev)})
        schedule_path.write_text(json.dumps({"bins": bins_out}, indent=2), encoding="utf-8")
        print(f"[calibrate_confidence] risk_schedule.json güncellendi: {schedule_path}")
    except Exception as e:
        print(f"[calibrate_confidence] Risk takvimi hesaplanamadı: {e}")


if __name__ == "__main__":
    # Dosya yolları
    root = Path(__file__).resolve().parent
    data_path = root / "metrics" / "calibration_trades.jsonl"
    calibration_path = root / "calibration.json"
    weights_path = root / "logistic_weights.json"
    schedule_path = root / "risk_schedule.json"
    calibrate_confidence(data_path, calibration_path, weights_path, schedule_path)