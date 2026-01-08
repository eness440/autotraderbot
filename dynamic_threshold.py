"""
dynamic_threshold.py

Bu modül, her sembol için dinamik bir güven eşiği (master confidence threshold)
hesaplar. Varsayılan durumda global bir eşik yerine, semboller kategori
bilgisine göre farklı eşikler kullanılır. Kategoriler ve eşikler data
dizini altındaki JSON dosyalarından yüklenir. Eğer bir sembolün
kategorisi tanımlı değilse, en muhafazakâr eşik kullanılır.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

# Kategori ve eşik dosyalarının yolları
DATA_DIR = Path(__file__).resolve().parent / "data"
_CATEGORIES_FILE = DATA_DIR / "symbol_categories.json"
_THRESHOLDS_FILE = DATA_DIR / "confidence_thresholds.json"

# Modül içi cache'ler
_symbol_to_category: Dict[str, str] | None = None
_category_thresholds: Dict[str, float] | None = None

def _load_symbol_categories() -> Dict[str, str]:
    """JSON dosyasından sembol-kategori haritasını yükler."""
    global _symbol_to_category
    if _symbol_to_category is None:
        try:
            if _CATEGORIES_FILE.exists():
                txt = _CATEGORIES_FILE.read_text(encoding="utf-8").strip()
                if txt:
                    data = json.loads(txt)
                    if isinstance(data, dict):
                        _symbol_to_category = {
                            str(k).strip().upper(): str(v).strip().lower()
                            for k, v in data.items()
                        }
                        return _symbol_to_category
        except Exception:
            pass
        _symbol_to_category = {}
    return _symbol_to_category

def _load_thresholds() -> Dict[str, float]:
    """Kategori bazlı eşik değerlerini yükler."""
    global _category_thresholds
    if _category_thresholds is None:
        try:
            if _THRESHOLDS_FILE.exists():
                txt = _THRESHOLDS_FILE.read_text(encoding="utf-8").strip()
                if txt:
                    data = json.loads(txt)
                    if isinstance(data, dict):
                        # threshold değerlerini 0–1 aralığında tut
                        # When loading per‑category thresholds, apply a slight
                        # relaxation factor to encourage more opportunities.
                        # Historical analysis showed that high static thresholds
                        # combined with conservative logistic models resulted in
                        # almost no trades.  Multiplying by 0.9 reduces each
                        # threshold by 10% (e.g. 0.70 → 0.63).  Values are
                        # subsequently clamped into [0,1].
                        _category_thresholds = {}
                        for k, v in data.items():
                            if isinstance(v, (int, float)):
                                try:
                                    val = float(v) * 0.9
                                except Exception:
                                    val = float(v)
                                val = max(0.0, min(1.0, val))
                                _category_thresholds[str(k).strip().lower()] = val
                        return _category_thresholds
        except Exception:
            pass
        _category_thresholds = {}
    return _category_thresholds

def get_category(symbol: str) -> Optional[str]:
    """Verilen sembol için kategori döndürür. Yoksa None."""
    if not symbol:
        return None
    cats = _load_symbol_categories()
    return cats.get(symbol.upper())

def get_threshold(symbol: str, default: float = 0.70) -> float:
    """
    Sembol için önerilen master confidence eşiğini döndürür. Kategori yoksa
    default değeri kullanılır. Eşik 0.0–1.0 aralığında sınırlandırılır.

    Args:
        symbol: "BTC/USDT" gibi sembol ismi.
        default: Kategori bilinmediğinde kullanılacak eşik.

    Returns:
        float: Eşik değeri (0–1).
    """
    try:
        cat = get_category(symbol)
        if cat:
            thresholds = _load_thresholds()
            val = thresholds.get(cat)
            if isinstance(val, (int, float)):
                return max(0.0, min(1.0, float(val)))
    except Exception:
        pass
    # Eğer kategori veya eşik tanımlı değilse, default değeri sınırla
    try:
        d = float(default)
    except Exception:
        d = 0.65
    return max(0.0, min(1.0, d))