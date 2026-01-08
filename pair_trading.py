# -*- coding: utf-8 -*-
"""
pair_trading.py

Bu modül, yüksek korelasyona sahip kripto para çiftleri için basit
"pair trading" sinyalleri üretmek üzere tasarlanmıştır. Amaç,
fiyat oranlarının veya spread'lerin kısa vadeli sapmalarından
yararlanarak uzun/kısa pozisyonlar açmayı önermektir.

CHANGELOG:
- v1.1: Added statsmodels import guard for environments without it
- v1.2: Added fallback cointegration function when statsmodels unavailable

Fonksiyonlar:

    load_pairs(path: str) -> list[tuple[str, str]]
        data/pairs.json dosyasından işlem yapılacak coin çiftlerini okur.

    compute_zscore(series: list[float]) -> float
        Verilen bir zaman serisindeki son değerin z-skorunu hesaplar.

    find_spread_opportunities(price_history: dict[str, list[float]], threshold: float = 2.0) -> list[dict[str, Any]]
        Verilen fiyat geçmişi için, yüklenen her sembol çifti bazında z-skorunu hesaplayarak
        sınırı aşan sapmalara göre trade önerileri döndürür.

Kullanım:

    >>> from pair_trading import find_spread_opportunities
    >>> price_hist = {"BTC/USDT": [10000, 10100, ...], "ETH/USDT": [500, 505, ...]}
    >>> signals = find_spread_opportunities(price_hist, threshold=2.0)
    >>> for sig in signals:
    ...     print(sig)

Sinyal çıktısı:
    {
      "pair": ("BTC/USDT", "ETH/USDT"),
      "long": "BTC/USDT",
      "short": "ETH/USDT",
      "zscore": 2.5,
      "timestamp": "2025-11-30T12:34:56Z"
    }

Bu modül ana bot içinde otomatik olarak kullanılmamaktadır; entegrasyon
durumunda, main_bot_async veya controller katmanında uygun bir yerde
``find_spread_opportunities`` çağrılarak üretilen sinyaller trade
sürecine dahil edilebilir.
"""

from __future__ import annotations

import json
import logging
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Setup logger
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FIXED: statsmodels import guard
# ---------------------------------------------------------------------------
# Try to import statsmodels for cointegration testing
# If not available, use a fallback that always returns p=1.0 (no cointegration)
try:
    from statsmodels.tsa.stattools import coint as _statsmodels_coint  # type: ignore
    HAS_STATSMODELS = True
    logger.info("statsmodels available - cointegration testing enabled")
except ImportError:
    HAS_STATSMODELS = False
    logger.warning(
        "statsmodels not installed - cointegration testing disabled. "
        "Install with: pip install statsmodels"
    )
    
    def _statsmodels_coint(x, y, **kwargs) -> Tuple[float, float, Any]:
        """
        Fallback cointegration function when statsmodels is not available.
        Always returns p_value=1.0 (no cointegration detected).
        """
        return (0.0, 1.0, None)


def coint(x, y, **kwargs) -> Tuple[float, float, Any]:
    """
    Wrapper for cointegration test that handles missing statsmodels gracefully.
    
    Args:
        x: First time series
        y: Second time series
        **kwargs: Additional arguments passed to statsmodels.coint
        
    Returns:
        Tuple of (score, p_value, critical_values)
        If statsmodels is not available, returns (0.0, 1.0, None)
    """
    try:
        return _statsmodels_coint(x, y, **kwargs)
    except Exception as e:
        logger.warning(f"Cointegration test failed: {e}")
        return (0.0, 1.0, None)


def load_pairs(path: str | None = None) -> List[Tuple[str, str]]:
    """data/pairs.json dosyasından işlem yapılacak coin çiftlerini okuyun.

    Dosya yoksa veya bozuksa, BTC/ETH ve BTC/SOL gibi örnek çiftler
    varsayılan olarak döndürülür.

    Args:
        path: Opsiyonel dosya yolu. None ise ``project_folder/data/pairs.json``
              varsayılır.

    Returns:
        Bir liste halinde sembol çiftleri. Örneğin [("BTC/USDT", "ETH/USDT"), ...]
    """
    if path is None:
        path = str(Path(__file__).resolve().parent / "data" / "pairs.json")
    try:
        p = Path(path)
        if p.exists():
            txt = p.read_text(encoding="utf-8").strip()
            if txt:
                data = json.loads(txt)
                pairs: List[Tuple[str, str]] = []
                if isinstance(data, list):
                    for item in data:
                        if (isinstance(item, list) or isinstance(item, tuple)) and len(item) == 2:
                            s1, s2 = str(item[0]).strip(), str(item[1]).strip()
                            if s1 and s2:
                                pairs.append((s1, s2))
                return pairs
    except Exception:
        pass
    # Varsayılan örnek çiftler
    return [("BTC/USDT", "ETH/USDT"), ("BTC/USDT", "SOL/USDT"), ("ETH/USDT", "SOL/USDT")]


def compute_zscore(series: List[float]) -> float:
    """Basit z-skoru hesaplama.

    Son değerin serinin ortalamasından kaç standart sapma uzaklıkta
    olduğunu döndürür. Serideki eleman sayısı 2'den azsa 0 döner.

    Args:
        series: Floatlardan oluşan bir zaman serisi.

    Returns:
        Son değerin z-skoru.
    """
    try:
        values = np.array(series, dtype=float)
        if values.size < 2:
            return 0.0
        mean = np.mean(values)
        std = np.std(values)
        if std == 0.0:
            return 0.0
        z = (values[-1] - mean) / std
        return float(z)
    except Exception:
        return 0.0


def is_cointegration_available() -> bool:
    """Check if cointegration testing is available."""
    return HAS_STATSMODELS


def find_spread_opportunities(
    price_history: Dict[str, List[float]],
    threshold: float = 2.0,
    lookback: int = 50,
) -> List[Dict[str, Any]]:
    """Seçili sembol çiftleri için spread sapma veya momentum sinyalleri üret.

    Bu fonksiyon, her sembol çifti için fiyat oranı serisinin z-skorunu hesaplar
    ve cointegrasyon testi uygular.  Cointegrasyon p-değeri belirli bir eşiğin
    altındaysa (örn. 0.05), geleneksel z-skor yaklaşımı kullanılır.  Eğer
    cointegrasyon yoksa, oran serisinin momentum'u (ilk ve son değer
    arasındaki yüzdesel değişim) hesaplanır.  Momentum belirli bir eşiği
    aşarsa long/short sinyali oluşturulur.  Böylece z‑skor dışında
    cointegrasyon testi ve momentum stratejisi desteklenir.

    Args:
        price_history: Her sembol için son ``lookback`` periyottaki fiyat
            verilerini içeren bir sözlük.
        threshold: Z-skoru eşiği. Örneğin 2.0 ile ±2 std. sapma kontrolü.
        lookback: Hesaplamada kullanılacak bar sayısı.

    Ortam Değişkenleri:
        PAIR_COINTEGRATION_P: Cointegrasyon testi için p-değeri eşiği (varsayılan 0.05).
        PAIR_MOMENTUM_THRESHOLD: Momentum yüzdesi eşiği (varsayılan 0.05).
        PAIR_USE_COINTEGRATION: "0" veya "false" verilirse cointegrasyon testi atlanır.

    Returns:
        Spread veya momentum sinyallerinin listesi.  Her sinyal dict yapısında
        ``pair``, ``long``, ``short``, ``zscore`` veya ``momentum`` ve ``timestamp`` anahtarları içerir.
    """
    import os

    # Ortamdan eşik değerlerini oku
    try:
        p_thresh = float(os.getenv("PAIR_COINTEGRATION_P", "0.05"))
    except Exception:
        p_thresh = 0.05
    try:
        mom_thresh = float(os.getenv("PAIR_MOMENTUM_THRESHOLD", "0.05"))
    except Exception:
        mom_thresh = 0.05
    
    use_coint_env = os.getenv("PAIR_USE_COINTEGRATION", "1").strip().lower()
    use_coint = use_coint_env not in ("0", "false", "no") and HAS_STATSMODELS

    pairs = load_pairs()
    signals: List[Dict[str, Any]] = []
    
    for s1, s2 in pairs:
        try:
            series1 = price_history.get(s1)
            series2 = price_history.get(s2)
            if not series1 or not series2:
                continue
            # Son lookback periyodu al.
            # NOTE: statsmodels.coint() requires endog/exog to be the same length.
            # Live OHLCV pulls can differ by a few candles (API delay / missing bars).
            n = min(len(series1), len(series2), lookback)
            arr1 = series1[-n:]
            arr2 = series2[-n:]
            # Her iki seride veri olup olmadığını kontrol et
            if n < 2:
                continue
            # Oran serisi: s1 / s2
            ratio_series = [a / b for a, b in zip(arr1, arr2) if b != 0]
            if len(ratio_series) < 2:
                continue
            
            # Cointegrasyon testi (opsiyonel)
            if use_coint:
                p_value = 1.0
                try:
                    # coint returns (score, pvalue, crit)
                    _, p_value, _ = coint(arr1, arr2)
                except Exception:
                    p_value = 1.0
                    
                if p_value < p_thresh:
                    # Cointegrasyon var → z‑skor bazlı strateji
                    z = compute_zscore(ratio_series)
                    if z >= threshold:
                        signals.append({
                            "pair": (s1, s2),
                            "long": s2,
                            "short": s1,
                            "zscore": float(z),
                            "cointegrated": True,
                            "p_value": float(p_value),
                            "timestamp": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
                        })
                        continue
                    elif z <= -threshold:
                        signals.append({
                            "pair": (s1, s2),
                            "long": s1,
                            "short": s2,
                            "zscore": float(z),
                            "cointegrated": True,
                            "p_value": float(p_value),
                            "timestamp": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
                        })
                        continue
                    # Eğer z eşiği aşmıyorsa momentum yöntemiyle değerlendirelim
                    
            # Momentum hesapla: ilk ve son oran arasındaki yüzdesel değişim
            try:
                momentum = (ratio_series[-1] - ratio_series[0]) / ratio_series[0]
            except Exception:
                momentum = 0.0
                
            if momentum >= mom_thresh:
                # oran artmış: s1 pahalılaştı → s1 short, s2 long
                signals.append({
                    "pair": (s1, s2),
                    "long": s2,
                    "short": s1,
                    "momentum": float(momentum),
                    "cointegrated": False,
                    "timestamp": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
                })
            elif momentum <= -mom_thresh:
                # oran azalmış: s1 ucuzladı → s1 long, s2 short
                signals.append({
                    "pair": (s1, s2),
                    "long": s1,
                    "short": s2,
                    "momentum": float(momentum),
                    "cointegrated": False,
                    "timestamp": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
                })
        except Exception as e:
            logger.debug(f"Pair {s1}/{s2} analizi sırasında hata: {e}")
            continue
            
    return signals
