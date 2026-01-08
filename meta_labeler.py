"""
meta_labeler.py
----------------

Bu stub modül, meta‑labelling veya ikinci görüş modeli için basit bir
fonksiyon sunar.  Gerçek implementasyon, AI/teknik/sentiment skorlarını
kullanarak bir trade'in kazançlı olup olmayacağına dair olasılık tahmini
üretebilir.  Mevcut projenizde bu dosya eksik olduğu için import
hatalarını engellemek amacıyla basit bir versiyon eklenmiştir.

Fonksiyonlar:
    compute_meta_probability(master_conf, ai_score, tech_score, sent_score, default)
        Verilen skorları kullanarak 0..1 aralığında bir meta olasılık döner.

Kullanım:
    from .meta_labeler import compute_meta_probability
"""
from __future__ import annotations
from typing import Optional


def compute_meta_probability(
    master_conf: float,
    ai_score: Optional[float],
    tech_score: Optional[float],
    sent_score: Optional[float],
    default: float = 1.0,
) -> float:
    """
    AI, teknik analiz ve sentiment skorlarına göre bir meta olasılık
    hesaplar.  Varsayılan implementasyon olarak, proje kökünde bulunan
    ``logistic_weights.json`` dosyasındaki lojistik regresyon katsayıları
    kullanılarak sigmoid fonksiyonu üzerinden bir olasılık hesaplanır.
    Bu dosya bulunamaz veya okunamazsa, mevcut skorların basit ortalaması
    alınır.  Eğer hiçbir skor mevcut değilse ``default`` değeri döner.

    Lojistik fonksiyon:

        p = 1 / (1 + exp(-(w0 + w_ai*ai_score + w_tech*tech_score + w_sent*sent_score)))

    Burada ``w0`` sabit terim (intercept), ``w_ai`` AI skorunun katsayısı,
    ``w_tech`` teknik skor katsayısı ve ``w_sent`` sentiment skor katsayısıdır.

    Args:
        master_conf (float): Birleşik güven skoru (0..1).  Bu parametre
            burada kullanılmaz ancak imza için tutulmuştur.
        ai_score (float|None): AI bileşeninin skoru (0..1)
        tech_score (float|None): Teknik bileşenin skoru (0..1)
        sent_score (float|None): Sentiment bileşenin skoru (0..1)
        default (float): Mevcut skor yoksa döndürülecek varsayılan değer.

    Returns:
        float: 0..1 aralığında meta etiket olasılığı.
    """
    # Kullanılabilir skorları topla
    scores: dict[str, float] = {}
    try:
        if ai_score is not None:
            scores["ai"] = float(ai_score)
    except Exception:
        pass
    try:
        if tech_score is not None:
            scores["tech"] = float(tech_score)
    except Exception:
        pass
    try:
        if sent_score is not None:
            scores["sent"] = float(sent_score)
    except Exception:
        pass

    # Eğer hiç skor yoksa varsayılan değeri döndür
    if not scores:
        try:
            return float(default)
        except Exception:
            return 1.0

    # JSON'dan lojistik ağırlıkları yükle (dosya yoksa exception yakala)
    import os
    import json
    weights = None
    try:
        # Dosya yolu, modülün bulunduğu dizine göre hesaplanır
        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(base_dir, "logistic_weights.json")
        if os.path.exists(weights_path):
            with open(weights_path, "r", encoding="utf-8") as f:
                weights = json.load(f)
    except Exception:
        weights = None

    if weights:
        # Lojistik regresyon katsayıları kullanarak olasılık hesapla
        w0 = float(weights.get("w0", 0.0))
        # Katsayılar dosyada varsa ilgili skora uygula, yoksa 0 kabul et
        w_ai = float(weights.get("w_ai", 0.0))
        w_tech = float(weights.get("w_tech", 0.0))
        w_sent = float(weights.get("w_sent", 0.0))
        # Eksik skorlar için sıfır katsayıları yok sayılır
        z = w0
        z += w_ai * scores.get("ai", 0.0)
        z += w_tech * scores.get("tech", 0.0)
        z += w_sent * scores.get("sent", 0.0)
        # Sigmoid fonksiyonu
        try:
            import math
            p = 1.0 / (1.0 + math.exp(-z))
        except Exception:
            p = 0.5
        # Aralık dışına taşmaması için kliple
        if p < 0.0:
            p = 0.0
        elif p > 1.0:
            p = 1.0
        return p
    else:
        # Ağırlıklar yoksa basit ortalama
        avg = sum(scores.values()) / len(scores)
        if avg < 0.0:
            avg = 0.0
        elif avg > 1.0:
            avg = 1.0
        return avg