# -*- coding: utf-8 -*-
"""
risk_schedule_builder.py
-----------------------

Bu script, ``data/risk_dataset.csv`` dosyasından master confidence
değerlerine göre aralıklar oluşturarak her aralık için optimum kaldıraç
değerleri tahmin eder. Hesaplama yöntemi basitçe her aralıkta ortalama
``pnl_pct`` ile ``max_drawdown_pct`` arasındaki oranı değerlendirir ve
risk-ödül oranı yüksek olan aralıklara daha yüksek kaldıraç atar. Sonuç
``risk_schedule.json`` dosyasına yazılır ve runtime'da kaldıraç hesaplamak
için kullanılır.

Kullanım:
    python risk_schedule_builder.py

Eğer yeterli veri yoksa veya dataset bulunamazsa, default kaldıraç tablosu
yazılır.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd


def build_risk_schedule(dataset_path: Path, schedule_path: Path) -> None:
    """Risk dataset'ten kaldıraç takvimini üretir ve JSON'a yazar."""
    default_bins = [
        (0.65, 0.70, 8),
        (0.70, 0.75, 12),
        (0.75, 0.80, 18),
        (0.80, 0.90, 25),
    ]
    if not dataset_path.exists():
        # default yaz
        bins_out = [
            {"min": lo, "max": hi, "leverage": lev} for (lo, hi, lev) in default_bins
        ]
        schedule_path.write_text(json.dumps({"bins": bins_out}, indent=2), encoding="utf-8")
        print(f"[risk_schedule_builder] Dataset yok. Default risk_schedule.json yazıldı.")
        return
    df = pd.read_csv(dataset_path)
    if "master_confidence" not in df.columns:
        bins_out = [
            {"min": lo, "max": hi, "leverage": lev} for (lo, hi, lev) in default_bins
        ]
        schedule_path.write_text(json.dumps({"bins": bins_out}, indent=2), encoding="utf-8")
        print(f"[risk_schedule_builder] Kolon bulunamadı. Default risk_schedule.json yazıldı.")
        return
    # bins tanımla
    bins = [(0.65, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 0.90)]
    ratios = []
    for (lo, hi) in bins:
        sub = df[(df["master_confidence"] >= lo) & (df["master_confidence"] < hi)]
        if len(sub) == 0:
            ratios.append((lo, hi, None))
            continue
        mean_pnl = sub["pnl_pct"].dropna().mean() if "pnl_pct" in sub.columns else 0.0
        mean_dd = sub["max_drawdown_pct"].dropna().mean() if "max_drawdown_pct" in sub.columns else 0.0
        # Risk ödül oranı: yüksek kar ve düşük drawdown daha iyidir
        try:
            ratio = (mean_pnl) / (abs(mean_dd) + 1e-6)
        except Exception:
            ratio = None
        ratios.append((lo, hi, ratio))
    # Oranlara göre sıralama; None olanlar en alta
    sorted_bins = sorted(ratios, key=lambda x: (x[2] is None, -(x[2] if x[2] is not None else -1)))
    leverage_levels = [8, 12, 18, 25]
    bins_out = []
    for idx, (lo, hi, ratio) in enumerate(sorted_bins):
        lev = leverage_levels[idx] if idx < len(leverage_levels) else leverage_levels[-1]
        bins_out.append({"min": lo, "max": hi, "leverage": lev})
    schedule_path.write_text(json.dumps({"bins": bins_out}, indent=2), encoding="utf-8")
    print(f"[risk_schedule_builder] risk_schedule.json yazıldı: {schedule_path}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    dataset_path = base_dir / "data" / "risk_dataset.csv"
    schedule_path = base_dir / "risk_schedule.json"
    build_risk_schedule(dataset_path, schedule_path)