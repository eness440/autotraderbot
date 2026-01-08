# -*- coding: utf-8 -*-
"""
risk_calibrator.py
------------------

Bu script, ``data/risk_dataset.csv`` veri setinden iki önemli modeli
öğrenir:

1. **Skor Kalibrasyonu**: ``master_confidence`` değerini gerçek kazanma
   olasılığına haritalayan basit bir lojistik regresyon. Sonuç ``calibration.json``
   dosyasına yazılır. Model parametreleri ``a`` (eğim) ve ``b`` (kesişim)
   olup olasılık şu şekilde hesaplanır: `p_hat = sigmoid(a * s + b)`.

2. **Ağırlık Öğrenimi**: AI, teknik ve sentiment skorlarının önemini
   istatistiksel olarak belirlemek için lojistik regresyon kullanılır. Bu
   model "y_win" etiketine karşı eğitilir ve sonuçlar ``logistic_weights.json``
   dosyasına yazılır. Model parametreleri ``w0`` (bias), ``w_ai``, ``w_tech``
   ve ``w_sent`` katsayılarıdır.

Kullanım:
    python risk_calibrator.py

Not: Veri seti çok küçükse veya hiç yoksa script sessizce çıkar.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss


def train_models(dataset_path: Path, calibration_path: Path, weights_path: Path) -> None:
    if not dataset_path.exists():
        print(f"[risk_calibrator] Dataset bulunamadı: {dataset_path}")
        return
    df = pd.read_csv(dataset_path)
    # Yeterli sütun olup olmadığını kontrol et
    required_cols = {"ai_score", "tech_score", "sent_score", "master_confidence", "y_win"}
    if not required_cols.issubset(set(df.columns)):
        print("[risk_calibrator] Dataset gerekli kolonları içermiyor.")
        return
    # Boş satırları çıkar
    df = df.dropna(subset=["ai_score", "tech_score", "sent_score", "master_confidence", "y_win"])
    # Veri seti çok küçükse güvenli varsayılan kalibrasyon ve ağırlıkları yazıp çık
    if len(df) < 50:
        print("[risk_calibrator] Veri seti çok küçük; güvenli varsayılan kalibrasyon ve ağırlıklar yazılıyor.")
        # Güvenli kalibrasyon: linear pass-through (a=1, b=0)
        calibration = {"type": "logistic", "a": 1.0, "b": 0.0}
        calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
        # Güvenli ağırlıklar: bias=0, tüm feature ağırlıkları 1 (eşit)
        weights = {"w0": 0.0, "w_ai": 1.0, "w_tech": 1.0, "w_sent": 1.0}
        weights_path.write_text(json.dumps(weights, indent=2), encoding="utf-8")
        return
    # y_win'i integer yap
    try:
        y = df["y_win"].astype(int).values
    except Exception:
        y = df["y_win"].values
    # 1) Skor → y_win kalibrasyonu.  To reduce overfitting and obtain
    # a better generalisation, we split the dataset into training and
    # validation sets.  The logistic regression is fit on the training
    # portion and evaluated on the validation portion.  A simple
    # accuracy metric is printed to aid diagnosis.  Finally, the model
    # is refit on the full data and saved to calibration.json.
    X_cal = df[["master_confidence"]].values
    X_train, X_val, y_train, y_val = train_test_split(X_cal, y, test_size=0.2, random_state=42, stratify=y)
    try:
        # Balance classes and add regularisation (C=0.5) to mitigate overfitting.
        # Increasing the iteration limit stabilises convergence on small/noisy data.
        model_cal = LogisticRegression(class_weight="balanced", max_iter=500, C=0.5)
        model_cal.fit(X_train, y_train)
        # Evaluate on validation set
        y_pred = model_cal.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        # Fit on full data for final parameters
        model_cal.fit(X_cal, y)
        a = float(model_cal.coef_[0][0])
        b = float(model_cal.intercept_[0])
        calibration = {"type": "logistic", "a": a, "b": b}
        calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
        print(
            f"[risk_calibrator] calibration.json yazıldı: a={a:.4f}, b={b:.4f} (val_acc={acc:.3f})"
        )
    except Exception as e:
        print(f"[risk_calibrator] Kalibrasyon eğitimi başarısız: {e}")
    # 2) AI/tech/sent ağırlıkları
    X_feat = df[["ai_score", "tech_score", "sent_score"]].values
    try:
        # Split for validation
        Xf_train, Xf_val, yf_train, yf_val = train_test_split(X_feat, y, test_size=0.2, random_state=42, stratify=y)
        # Use balanced class weights and regularisation (C=0.5) with a larger
        # iteration budget.  This counters the tendency for the model to
        # produce extreme negative weights for useful features when losing
        # trades dominate the dataset.
        model_feat = LogisticRegression(class_weight="balanced", max_iter=1000, C=0.5)
        model_feat.fit(Xf_train, yf_train)
        # Evaluate accuracy on validation split
        y_pred_f = model_feat.predict(Xf_val)
        acc_f = accuracy_score(yf_val, y_pred_f)
        # Fit on full dataset to obtain final coefficients
        model_feat.fit(X_feat, y)
        w0 = float(model_feat.intercept_[0])
        w_ai = float(model_feat.coef_[0][0])
        w_tech = float(model_feat.coef_[0][1])
        w_sent = float(model_feat.coef_[0][2])
        # Clamp learned weights to prevent extreme values which can lead
        # to overconfident scores.  Bias is limited to ±2 and features to ±1.
        def _clamp(val: float, lo: float, hi: float) -> float:
            return lo if val < lo else hi if val > hi else val

        w0_clamped = _clamp(w0, -2.0, 2.0)
        w_ai_clamped = _clamp(w_ai, -1.0, 1.0)
        w_tech_clamped = _clamp(w_tech, -1.0, 1.0)
        w_sent_clamped = _clamp(w_sent, -1.0, 1.0)
        #
        # MODEL IS OLDERLY PESSIMISTIC ADJUSTMENT
        #
        # In practice we expect positive AI and technical coefficients: a stronger AI or
        # technical “buy” signal should *increase* the confidence of a profitable
        # trade rather than reduce it.  When the training data is imbalanced or
        # incorrectly labelled the learned weights may be negative, leading to
        # counterintuitive behaviour (e.g. strong AI signals producing lower
        # confidence).  To prevent this, force the AI and technical weights to
        # be non‑negative by taking their absolute values.  Sentiment weights are
        # left unchanged because negative sentiment can legitimately reduce
        # confidence.  This adjustment mitigates overly pessimistic models that
        # would otherwise skip most trades.
        w_ai_clamped = abs(w_ai_clamped)
        w_tech_clamped = abs(w_tech_clamped)
        weights = {
            "w0": w0_clamped,
            "w_ai": w_ai_clamped,
            "w_tech": w_tech_clamped,
            "w_sent": w_sent_clamped,
        }
        weights_path.write_text(json.dumps(weights, indent=2), encoding="utf-8")
        print(
            f"[risk_calibrator] logistic_weights.json yazıldı: w0={w0_clamped:.4f}, w_ai={w_ai_clamped:.4f}, "
            f"w_tech={w_tech_clamped:.4f}, w_sent={w_sent_clamped:.4f} (val_acc={acc_f:.3f})"
        )
    except Exception as e:
        print(f"[risk_calibrator] Ağırlık modeli eğitimi başarısız: {e}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    dataset_path = base_dir / "data" / "risk_dataset.csv"
    calibration_path = base_dir / "calibration.json"
    weights_path = base_dir / "logistic_weights.json"
    train_models(dataset_path, calibration_path, weights_path)