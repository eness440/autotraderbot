# -*- coding: utf-8 -*-
"""
risk_dataset_builder.py
----------------------

Bu script, ``trade_log.json`` dosyasındaki tamamlanmış işlem kayıtlarını
okuyarak bilimsel risk modelleme için kullanılacak bir veri seti oluşturur.
Her trade için giriş özellikleri (ai_score, tech_score, sent_score, master_confidence,
atr_pct, fgi_norm, adx, rsi, side vb.) ve etiketler (y_win, y_big_loss, pnl_pct)
hesaplanır. Oluşturulan veri seti ``data/risk_dataset.csv`` yoluna yazılır.

Kullanım:
    python risk_dataset_builder.py

Bu dosya kendi başına çalıştırılabilir; proje kök dizininde veya script'in
bulunduğu dizinde ``trade_log.json`` mevcut olmalıdır. Eğer yeterli veri yoksa
script uyarı vererek sessizce çıkar.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import csv
from typing import List, Dict, Any

import pandas as pd


def build_dataset(log_path: Path, out_path: Path) -> None:
    """``trade_log.json`` dosyasını okuyup risk_dataset.csv üretir."""
    if not log_path.exists():
        print(f"[risk_dataset_builder] trade_log bulunamadı: {log_path}")
        return
    # kayıtları oku
    try:
        text = log_path.read_text(encoding="utf-8").strip()
        if not text:
            print("[risk_dataset_builder] trade_log boş.")
            return
        data = json.loads(text)
        if isinstance(data, dict):
            rows = data.get("rows", [])
        elif isinstance(data, list):
            rows = data
        else:
            rows = []
    except Exception as e:
        print(f"[risk_dataset_builder] trade_log okunamadı: {e}")
        return
    dataset_rows: List[Dict[str, Any]] = []
    for row in rows:
        # sadece kapanmış işlemleri kullan
        if not row or row.get("timestamp_close") is None:
            continue
        try:
            ai_score = float(row.get("ai_score")) if row.get("ai_score") is not None else None
            tech_score = float(row.get("tech_score")) if row.get("tech_score") is not None else None
            sent_score = float(row.get("sent_score")) if row.get("sent_score") is not None else None
            master_conf = float(row.get("master_confidence")) if row.get("master_confidence") is not None else None
            entry_price = float(row.get("entry_price")) if row.get("entry_price") is not None else None
            atr = float(row.get("atr")) if row.get("atr") is not None else None
            fgi = float(row.get("fgi")) if row.get("fgi") is not None else None
            adx = float(row.get("adx")) if row.get("adx") is not None else None
            rsi = float(row.get("rsi")) if row.get("rsi") is not None else None
            side = row.get("side")
            pnl_pct = float(row.get("pnl_pct")) if row.get("pnl_pct") is not None else None
            max_drawdown_pct = row.get("max_drawdown_pct")
            if max_drawdown_pct is not None:
                try:
                    max_drawdown_pct = float(max_drawdown_pct)
                except Exception:
                    max_drawdown_pct = None
            # feature engineering
            atr_pct = (atr / entry_price) if (atr is not None and entry_price) else None
            fgi_norm = (fgi / 100.0) if fgi is not None else None
            side_code = None
            if side == "long":
                side_code = 1
            elif side == "short":
                side_code = 0
            y_win = None
            if pnl_pct is not None:
                y_win = 1 if pnl_pct > 0 else 0
            y_big_loss = None
            if max_drawdown_pct is not None:
                y_big_loss = 1 if max_drawdown_pct < -5.0 else 0
            dataset_rows.append({
                "ai_score": ai_score,
                "tech_score": tech_score,
                "sent_score": sent_score,
                "master_confidence": master_conf,
                "atr_pct": atr_pct,
                "fgi_norm": fgi_norm,
                "adx": adx,
                "rsi": rsi,
                "side": side_code,
                "pnl_pct": pnl_pct,
                "max_drawdown_pct": max_drawdown_pct,
                "y_win": y_win,
                "y_big_loss": y_big_loss,
            })
        except Exception:
            continue
    if not dataset_rows:
        print("[risk_dataset_builder] İşlenecek yeterli trade verisi yok.")
        return
    df = pd.DataFrame(dataset_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[risk_dataset_builder] Dataset yazıldı: {out_path} ({len(df)} satır)")


if __name__ == "__main__":
    # trade_log.json dosyası script ile aynı dizinde varsayılır
    current_dir = Path(__file__).resolve().parent
    log_path = current_dir / "trade_log.json"
    out_path = current_dir / "data" / "risk_dataset.csv"
    build_dataset(log_path, out_path)