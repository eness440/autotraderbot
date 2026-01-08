"""
parameter_optimizer.py
---------------------

Bu modül, botun çeşitli hiperparametrelerini optimize etmek için bir
araç sağlar. Logistic regresyon katsayıları, stop‑loss çarpanları,
indikator periyotları ve trade cooldown süreleri gibi parametreler,
geçmiş verilere dayalı olarak otomatik olarak ayarlanabilir. Böylece
manuel ayar süresi azalır ve overfitting riski daha iyi kontrol edilir.

Şu an için bu dosya, Optuna kütüphanesi ile basit bir örnek sunar.
Gerçek optimizasyon senaryoları için uygun maliyet fonksiyonları,
cross‑validation ve data pipeline tanımlamak gerekmektedir.

Çalıştırmak için:

    python parameter_optimizer.py

Optuna kurulu değilse, bu script uyarı vererek çıkacaktır.
"""

import logging
import os
import random

try:
    import optuna  # type: ignore
except ImportError:
    optuna = None  # pragma: no cover

import json
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _objective(trial: "optuna.Trial") -> float:
    """Composite objective function for parameter optimisation.

    In addition to tuning the logistic regression regularisation
    strength ``C``, this objective samples a handful of other strategy
    parameters such as the stop‑loss multiplier, indicator periods and
    trade cooldown durations.  The primary evaluation metric remains
    the negative log‑loss of a logistic regression on the risk dataset
    (lower is better).  Random values for the auxiliary parameters are
    returned via the trial to aid subsequent configuration, though they
    do not directly impact the loss computation.  If the risk dataset
    or scikit‑learn is unavailable, the function falls back to a
    random score.
    """
    # Suggest hyperparameters.  Even if unused in this simplified
    # objective, they are recorded in ``trial.params`` for later use.
    # Use the updated suggest_float API; `log=True` replaces suggest_loguniform.
    c_val = trial.suggest_float("C", 1e-3, 1e2, log=True)
    # Stop‑loss multiplier (e.g. 0.005–0.05 for 0.5–5% distance)
    trial.suggest_float("stop_loss_mult", 0.005, 0.05)
    # Short and long moving average periods (10–100 and 20–200)
    trial.suggest_int("ma_short", 10, 100)
    trial.suggest_int("ma_long", 20, 200)
    # Cooldown period between trades in minutes (1–60)
    trial.suggest_int("cooldown_min", 1, 60)
    try:
        import pandas as pd  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.model_selection import cross_val_score  # type: ignore
        # Path to risk dataset
        data_path = os.path.join("data", "risk_dataset.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError
        df = pd.read_csv(data_path)
        # Identify feature columns (at least two of ai_score, tech_score, sent_score)
        feature_cols = [col for col in ["ai_score", "tech_score", "sent_score"] if col in df.columns]
        if len(feature_cols) < 2:
            raise ValueError("Required feature columns missing in risk dataset")
        # Identify target column: accept 'y', 'y_win' or 'label'
        target_col = None
        for col in ["y", "y_win", "label"]:
            if col in df.columns:
                target_col = col
                break
        if target_col is None:
            raise ValueError("Required target column missing in risk dataset")
        X = df[feature_cols]
        y = df[target_col]
        # Initialise model with class balancing and adequate iterations
        model = LogisticRegression(C=c_val, max_iter=1000, class_weight="balanced")
        scores = cross_val_score(model, X, y, cv=3, scoring="neg_log_loss")
        # cross_val_score returns negative log loss; multiply by -1 to minimise
        return -float(scores.mean())
    except Exception:
        # Fallback to random score if dataset or dependencies unavailable
        return random.random()


def run_optimization(n_trials: int = 20) -> Dict[str, Any]:
    """
    Optuna ile hiperparametre araması gerçekleştirir. Optuna kurulu
    değilse veya ``n_trials`` 0 ise boş bir sözlük döndürür.

    Args:
        n_trials: Denenecek deneme sayısı.

    Returns:
        En iyi denemenin parametreleri ve skorunu içeren bir sözlük.
    """
    if optuna is None:
        logger.warning("Optuna kurulu değil. Parametre optimizasyonu yapılamıyor.")
        return {}
    if n_trials <= 0:
        return {}
    study = optuna.create_study(direction="minimize")
    study.optimize(_objective, n_trials=n_trials)
    best_params = study.best_params
    best_value = study.best_value
    logger.info(f"Optuna en iyi değer: {best_value}, parametreler: {best_params}")
    return {"best_params": best_params, "best_value": best_value}


if __name__ == "__main__":
    res = run_optimization()
    if res:
        print(json.dumps(res, indent=2))