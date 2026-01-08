# -*- coding: utf-8 -*-
"""
hyperparameter_search.py

Bu komut satırı aracı, risk_dataset.csv dosyası üzerinde lojistik regresyon
modeli için basit bir hiperparametre taraması yapar. Amaç, farklı
cezalandırma parametreleri (C) ve solver kombinasyonları için modeli
değerlendirip en iyi parametreleri bulmaktır.

Çalıştırmak için:

    (venv) python hyperparameter_search.py

Tarama sonuçları ``data/hyperparameters.json`` dosyasına yazılır ve
console üzerine raporlanır. Veri dosyası bulunamazsa, script hiçbir
işlem yapmadan çıkar.

Not: Bu script, çevrimdışı kullanım içindir. Ana bot içerisinde
otomatik tetiklenmez.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score


def perform_grid_search(csv_path: Path, output_path: Path, verbose: bool = True) -> None:
    """Hiperparametre taraması yap ve sonuçları yaz.

    Args:
        csv_path: Veri kümesinin yolu (risk_dataset.csv).
        output_path: Sonuçların yazılacağı json dosyası.
        verbose: True ise ara sonuçları ekrana yaz.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[hyperparameter_search] Veri seti okunamadı: {e}")
        return
    # Hedef değişken
    y = df.get("y_win")
    if y is None:
        print("[hyperparameter_search] Veri kümesinde 'y_win' sütunu bulunamadı.")
        return
    # Özellikler
    features = [col for col in df.columns if col not in ("y_win", "symbol", "timestamp")]
    X = df[features].values
    # Parametre grid'i
    param_grid = {
        'C': [0.1, 1.0, 10.0, 50.0],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [200]
    }
    # 5-katlı çapraz doğrulama
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    model = LogisticRegression()
    search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)
    try:
        search.fit(X, y)
    except Exception as e:
        print(f"[hyperparameter_search] Tarama başarısız: {e}")
        return
    best_params = search.best_params_
    best_score = search.best_score_
    if verbose:
        print(f"[hyperparameter_search] En iyi parametreler: {best_params}")
        print(f"[hyperparameter_search] Ortalama doğruluk: {best_score:.4f}")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as f:
            json.dump({'best_params': best_params, 'best_score': best_score}, f, ensure_ascii=False, indent=2)
        if verbose:
            print(f"[hyperparameter_search] Sonuçlar {output_path} dosyasına yazıldı.")
    except Exception as e:
        print(f"[hyperparameter_search] Sonuçlar yazılamadı: {e}")


if __name__ == "__main__":
    csv_path = Path("data/risk_dataset.csv")
    output_path = Path("data/hyperparameters.json")
    perform_grid_search(csv_path, output_path)